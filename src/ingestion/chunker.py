"""Chunking logic using LangChain RecursiveCharacterTextSplitter."""
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingestion.loaders import RawDocument
from src.ingestion.metadata import ChunkRecord, extract_section_heading
from src.common.logging import get_logger

logger = get_logger("umbrella.ingestion.chunker")


def get_text_splitter(chunk_size: int = 800, chunk_overlap: int = 100) -> RecursiveCharacterTextSplitter:
    """Create a configured RecursiveCharacterTextSplitter instance."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        strip_whitespace=True,
    )


def chunk_document(
    doc: RawDocument,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    content_hash: Optional[str] = None,
) -> List[ChunkRecord]:
    """
    Split a cleaned RawDocument into ChunkRecords with citation metadata.

    - Multi-page documents (PDF): Chunks are created per page to preserve exact page references.
    - Single-segment documents (DOCX/TXT/MD): Chunks span document with page_number=None.
    """
    splitter = get_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    ingested_at = datetime.now(timezone.utc).isoformat()
    chunks: List[ChunkRecord] = []
    chunk_index = 0
    current_doc_offset = 0

    for page in doc.pages:
        page_text = page.text.strip()
        if not page_text:
            continue

        # Split text into chunks
        split_texts = splitter.split_text(page_text)
        page_offset = 0

        for text_piece in split_texts:
            clean_piece = text_piece.strip()
            if not clean_piece:
                continue

            # Calculate character offsets within the page / document
            char_pos = page_text.find(clean_piece, page_offset)
            if char_pos == -1:
                char_pos = page_offset
            char_start = current_doc_offset + char_pos
            char_end = char_start + len(clean_piece)
            page_offset = char_pos + len(clean_piece)

            # Detect section heading
            heading = extract_section_heading(clean_piece)

            chunk_record = ChunkRecord(
                chunk_id=str(uuid.uuid4()),
                doc_id=doc.doc_id,
                doc_name=doc.filename,
                source_type=doc.source_type,
                page_number=page.page_num,
                section_heading=heading,
                chunk_index=chunk_index,
                char_start=char_start,
                char_end=char_end,
                text=clean_piece,
                ingested_at=ingested_at,
                content_hash=content_hash,
            )
            chunks.append(chunk_record)
            chunk_index += 1

        current_doc_offset += len(page_text)

    logger.info(
        f"Chunked document '{doc.filename}' (doc_id: {doc.doc_id}) into {len(chunks)} chunks "
        f"(chunk_size: {chunk_size}, overlap: {chunk_overlap})"
    )

    return chunks
