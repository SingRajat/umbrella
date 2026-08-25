import re
import unicodedata
from collections import Counter
from src.ingestion.loaders import RawPage


def remove_control_characters(text: str) -> str:
    """Strip non-printable and ASCII control characters except standard whitespace."""
    return "".join(ch for ch in text if ch in "\n\r\t" or (unicodedata.category(ch)[0] != "C" and ord(ch) >= 32))


def normalize_whitespace(text: str) -> str:
    """Normalize line endings and collapse excessive spaces/blank lines while preserving paragraphs."""
    # Standardize line breaks
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse multiple inline spaces to single space
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse 3 or more newlines to double newline
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip leading and trailing whitespace per line
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def dehyphenate(text: str) -> str:
    """Rejoin words split across line breaks (e.g. 'instruc-\ntion' -> 'instruction')."""
    def _rejoin(match: re.Match) -> str:
        part1 = match.group(1)
        part2 = match.group(2)
        # Only rejoin if both parts look like word fragments
        if part1.isalpha() and part2.isalpha():
            return f"{part1}{part2}"
        return match.group(0)

    return re.sub(r"(\b[a-zA-Z]+)-\n\s*([a-zA-Z]+\b)", _rejoin, text)


def strip_repetitive_headers_footers(pages: list[RawPage]) -> list[RawPage]:
    """Detect lines repeating across 3 or more pages (e.g., running headers/footers) and strip them."""
    if len(pages) < 3:
        return pages

    line_counts = Counter()
    for page in pages:
        page_lines = set(line.strip() for line in page.text.split("\n") if len(line.strip()) > 3)
        line_counts.update(page_lines)

    # Candidate headers/footers: appear in at least 3 pages or >60% of pages
    threshold = max(3, int(len(pages) * 0.6))
    repetitive_lines = {line for line, count in line_counts.items() if count >= threshold}

    cleaned_pages: list[RawPage] = []
    for page in pages:
        filtered_lines = [
            line for line in page.text.split("\n")
            if line.strip() not in repetitive_lines
        ]
        cleaned_pages.append(RawPage(page_num=page.page_num, text="\n".join(filtered_lines)))

    return cleaned_pages


def clean_text(text: str) -> str:
    """Pure function applying the full deterministic cleaning sequence on raw text."""
    text = remove_control_characters(text)
    text = dehyphenate(text)
    text = normalize_whitespace(text)
    return text


def clean_pages(pages: list[RawPage]) -> list[RawPage]:
    """Cleans a sequence of RawPages including multi-page header/footer detection."""
    pages = strip_repetitive_headers_footers(pages)
    return [
        RawPage(
            page_num=p.page_num,
            text=clean_text(p.text),
        )
        for p in pages
    ]
