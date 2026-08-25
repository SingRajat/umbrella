import hashlib
import io
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
import pdfplumber
import docx

from src.common.errors import IngestionError
from src.common.logging import logger


@dataclass
class RawPage:
    """Represents a single page or segment of a raw document."""
    page_num: int | None
    text: str


@dataclass
class RawDocument:
    """Represents an extracted document before cleaning and chunking."""
    doc_id: str
    filename: str
    source_type: str
    uploaded_at: str
    pages: list[RawPage]
    content_hash: str


def compute_sha256(content: bytes) -> str:
    """Computes SHA-256 hash for raw file bytes."""
    return hashlib.sha256(content).hexdigest()


class PDFLoader:
    """Extracts pages and text from PDF documents using pdfplumber."""

    @staticmethod
    def load(file_bytes: bytes, filename: str) -> list[RawPage]:
        pages: list[RawPage] = []
        try:
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                if not pdf.pages:
                    raise IngestionError(f"PDF '{filename}' contains no pages.")

                for idx, page in enumerate(pdf.pages, start=1):
                    extracted = page.extract_text() or ""
                    pages.append(RawPage(page_num=idx, text=extracted))
        except Exception as e:
            logger.error(f"PDF extraction error for {filename}: {e}")
            raise IngestionError(f"Failed to parse PDF '{filename}': {e}") from e

        return pages


class DOCXLoader:
    """Extracts text from DOCX documents using python-docx."""

    @staticmethod
    def load(file_bytes: bytes, filename: str) -> list[RawPage]:
        try:
            doc = docx.Document(io.BytesIO(file_bytes))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            full_text = "\n\n".join(paragraphs)
            return [RawPage(page_num=None, text=full_text)]
        except Exception as e:
            logger.error(f"DOCX extraction error for {filename}: {e}")
            raise IngestionError(f"Failed to parse DOCX '{filename}': {e}") from e


class TXTLoader:
    """Extracts text from plain text files."""

    @staticmethod
    def load(file_bytes: bytes, filename: str) -> list[RawPage]:
        try:
            text = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = file_bytes.decode("latin-1")
            except Exception as e:
                raise IngestionError(f"Failed to decode text file '{filename}': {e}") from e
        return [RawPage(page_num=None, text=text)]


class MDLoader:
    """Extracts text from markdown files."""

    @staticmethod
    def load(file_bytes: bytes, filename: str) -> list[RawPage]:
        try:
            text = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = file_bytes.decode("latin-1")
            except Exception as e:
                raise IngestionError(f"Failed to decode markdown file '{filename}': {e}") from e
        return [RawPage(page_num=None, text=text)]


def load_document(file_bytes: bytes, filename: str, extension: str) -> RawDocument:
    """Factory function to load and parse documents into a RawDocument object."""
    ext = extension.lower().lstrip(".")
    if ext == "pdf":
        pages = PDFLoader.load(file_bytes, filename)
    elif ext == "docx":
        pages = DOCXLoader.load(file_bytes, filename)
    elif ext == "txt":
        pages = TXTLoader.load(file_bytes, filename)
    elif ext == "md":
        pages = MDLoader.load(file_bytes, filename)
    else:
        raise IngestionError(f"Unsupported file extension: .{ext}")

    # Validate that non-empty text was extracted across all pages
    total_text = "".join(p.text for p in pages).strip()
    if not total_text:
        raise IngestionError(f"Document '{filename}' extracted to empty text. Cannot index empty document.")

    doc_id = str(uuid.uuid4())
    uploaded_at = datetime.now(timezone.utc).isoformat()
    content_hash = compute_sha256(file_bytes)

    return RawDocument(
        doc_id=doc_id,
        filename=filename,
        source_type=ext,
        uploaded_at=uploaded_at,
        pages=pages,
        content_hash=content_hash,
    )
