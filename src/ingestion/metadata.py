import re
from typing import Any
from src.ingestion.chunker import Chunk
from src.ingestion.loaders import RawDocument
from src.storage.chroma import ChunkRecord


def extract_section_heading(text: str) -> str | None:
    """Attempts to extract markdown heading or section title if present at chunk beginning."""
    lines = text.strip().split("\n")
    if lines:
        first_line = lines[0].strip()
        # Markdown heading match (e.g. ## Overview)
        match = re.match(r"^#{1,6}\s+(.+)$", first_line)
        if match:
            return match.group(1).strip()
        # All-caps short section title match (e.g. INTRODUCTION)
        if first_line.isupper() and 3 <= len(first_line) <= 50:
            return first_line
    return None


def construct_chunk_records(
    raw_doc: RawDocument,
    chunks: list[Chunk],
    domain_tag: str | None = "general",
) -> list[ChunkRecord]:
    """Attaches complete PRD §3.4 metadata schema and creates ChunkRecord objects."""
    records: list[ChunkRecord] = []

    for chunk in chunks:
        section_heading = extract_section_heading(chunk.text)

        metadata: dict[str, Any] = {
            "chunk_id": chunk.chunk_id,
            "doc_id": raw_doc.doc_id,
            "doc_name": raw_doc.filename,
            "source_type": raw_doc.source_type,
            "page_number": chunk.page_number if chunk.page_number is not None else "",
            "section_heading": section_heading if section_heading is not None else "",
            "chunk_index": chunk.chunk_index,
            "char_start": chunk.char_start,
            "char_end": chunk.char_end,
            "domain_tag": domain_tag or "general",
            "ingested_at": raw_doc.uploaded_at,
            "content_hash": raw_doc.content_hash,
        }

        records.append(
            ChunkRecord(
                chunk_id=chunk.chunk_id,
                doc_id=raw_doc.doc_id,
                text=chunk.text,
                metadata=metadata,
            )
        )

    return records
