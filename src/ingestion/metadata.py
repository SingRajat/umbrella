"""Chunk metadata schema and extraction utilities."""
import re
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.ingestion.loaders import RawDocument, RawPage


@dataclass
class ChunkRecord:
    """Standardized record representing an indexed text chunk with citation metadata."""
    chunk_id: str
    doc_id: str
    doc_name: str
    source_type: str
    page_number: Optional[int]
    section_heading: Optional[str]
    chunk_index: int
    char_start: int
    char_end: int
    text: str
    ingested_at: str
    content_hash: Optional[str] = None

    def to_metadata_dict(self) -> Dict[str, Any]:
        """Convert chunk metadata to flat dictionary for ChromaDB vector storage."""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "doc_name": self.doc_name,
            "source_type": self.source_type,
            "page_number": self.page_number if self.page_number is not None else -1,
            "section_heading": self.section_heading or "",
            "chunk_index": self.chunk_index,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "ingested_at": self.ingested_at,
            "content_hash": self.content_hash or "",
        }


def extract_section_heading(text: str) -> Optional[str]:
    """
    Detect the most prominent heading in the chunk text.
    Looks for Markdown headers (e.g. # Title, ## Section) or colon-terminated headers.
    """
    # 1. Markdown headers
    md_match = re.search(r"^(?:#{1,6})\s+(.+)$", text, re.MULTILINE)
    if md_match:
        return md_match.group(1).strip()

    # 2. Capitalized section line ending with colon (e.g. 'Overview:', 'SECTION 1:')
    colon_match = re.search(r"^[A-Z0-9\s_-]{3,50}:", text, re.MULTILINE)
    if colon_match:
        return colon_match.group(0).rstrip(":").strip()

    return None
