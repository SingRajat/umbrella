"""Deterministic regex-based text cleaning and preprocessing functions."""
import re
from collections import Counter
from typing import List

from src.ingestion.loaders import RawDocument, RawPage


def remove_control_characters(text: str) -> str:
    """Strip non-printable ASCII and control characters except standard whitespace (\\n, \\t, \\r)."""
    return re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)


def dehyphenate(text: str) -> str:
    """
    Rejoin words split across line breaks (e.g. 'docu-\\nment' -> 'document').
    Preserves deliberate markdown or bullet lists.
    """
    # Matches a lowercase/uppercase word fragment, a hyphen, newline, and trailing word fragment
    return re.sub(r"(\b[A-Za-z]+)-\s*\n\s*([A-Za-z]+\b)", r"\1\2", text)


def normalize_whitespace(text: str) -> str:
    """
    Collapse multiple spaces/tabs into single spaces and normalize excessive newlines.
    Preserves paragraph breaks (max 2 consecutive newlines).
    """
    # Replace non-breaking spaces and tabs with standard space
    text = text.replace("\u00a0", " ").replace("\t", " ")

    # Strip trailing and leading whitespace per line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    # Collapse horizontal whitespace
    text = re.sub(r"[ ]{2,}", " ", text)

    # Collapse 3 or more consecutive newlines to 2 (paragraph break)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def strip_repeated_headers_footers(pages: List[RawPage], min_occurrences: int = 3) -> List[RawPage]:
    """
    Detect and strip lines that repeat identically across multiple pages
    (e.g., running headers, confidentiality footers, page numbering artifacts).
    """
    if len(pages) < min_occurrences:
        return pages

    # Count occurrences of all non-empty lines across pages
    line_counts = Counter()
    for page in pages:
        unique_page_lines = set(line.strip() for line in page.text.split("\n") if line.strip())
        for line in unique_page_lines:
            # Only count short/header-like lines (less than 120 chars) to avoid stripping recurring content paragraphs
            if len(line) < 120:
                line_counts[line] += 1

    # Find repeated header/footer lines that appear in >= min_occurrences pages
    repeated_lines = {line for line, count in line_counts.items() if count >= min_occurrences}

    if not repeated_lines:
        return pages

    cleaned_pages = []
    for page in pages:
        page_lines = page.text.split("\n")
        filtered_lines = [l for l in page_lines if l.strip() not in repeated_lines]
        cleaned_pages.append(RawPage(page_num=page.page_num, text="\n".join(filtered_lines)))

    return cleaned_pages


def clean_text(text: str) -> str:
    """Apply sequential regex cleaning transformations to a single string."""
    text = remove_control_characters(text)
    text = dehyphenate(text)
    text = normalize_whitespace(text)
    return text


def clean_document(raw_doc: RawDocument) -> RawDocument:
    """
    Clean all pages in a RawDocument:
    1. Strips repeating running headers/footers across pages.
    2. Applies control char removal, dehyphenation, and whitespace normalization per page.
    """
    # 1. Header/footer stripping
    processed_pages = strip_repeated_headers_footers(raw_doc.pages)

    # 2. Individual page cleaning
    cleaned_pages = []
    for page in processed_pages:
        cleaned_text = clean_text(page.text)
        cleaned_pages.append(RawPage(page_num=page.page_num, text=cleaned_text))

    return RawDocument(
        doc_id=raw_doc.doc_id,
        filename=raw_doc.filename,
        source_type=raw_doc.source_type,
        uploaded_at=raw_doc.uploaded_at,
        pages=cleaned_pages,
    )
