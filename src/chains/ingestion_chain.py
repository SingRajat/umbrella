"""LCEL document ingestion pipeline chain."""
from pathlib import Path
from typing import Any, Dict, List, Optional
from langchain_core.runnables import RunnableLambda

from src.config.settings import get_settings
from src.common.errors import IngestionError
from src.common.logging import get_logger
from src.ingestion.loaders import RawDocument, load_docx, load_pdf, load_txt
from src.ingestion.cleaner import clean_document
from src.ingestion.chunker import chunk_document
from src.ingestion.metadata import ChunkRecord
from src.storage.chroma import ChromaVectorStore, get_vector_store

logger = get_logger("umbrella.chains.ingestion")


def _load_step(input_data: Dict[str, Any]) -> RawDocument:
    """Step 1: Load file based on extension into standardized RawDocument."""
    file_path = input_data["file_path"]
    doc_id = input_data.get("doc_id")
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return load_pdf(file_path, doc_id=doc_id)
    elif suffix == ".docx":
        return load_docx(file_path, doc_id=doc_id)
    elif suffix in (".txt", ".md"):
        return load_txt(file_path, doc_id=doc_id)
    else:
        raise IngestionError(
            f"Unsupported document format '{suffix}'. Supported: .pdf, .docx, .txt, .md",
            error_code="UNSUPPORTED_FORMAT",
            status_code=400,
        )


def _clean_step(raw_doc: RawDocument) -> RawDocument:
    """Step 2: Clean and normalize text across all document pages."""
    return clean_document(raw_doc)


def _make_chunk_step(settings, content_hash: Optional[str] = None):
    """Step 3: Chunk cleaned document into ChunkRecords with citation metadata."""
    def _chunk(cleaned_doc: RawDocument) -> List[ChunkRecord]:
        return chunk_document(
            cleaned_doc,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            content_hash=content_hash,
        )
    return _chunk


def _make_store_step(vector_store: ChromaVectorStore):
    """Step 4: Generate embeddings and index ChunkRecords into ChromaDB."""
    def _store(chunks: List[ChunkRecord]) -> Dict[str, Any]:
        if not chunks:
            raise IngestionError("Document produced 0 chunks after processing.", error_code="EMPTY_CHUNKS", status_code=422)

        vector_store.upsert(chunks)
        first_chunk = chunks[0]
        return {
            "doc_id": first_chunk.doc_id,
            "filename": first_chunk.doc_name,
            "source_type": first_chunk.source_type,
            "chunk_count": len(chunks),
            "ingested_at": first_chunk.ingested_at,
            "status": "ingested",
        }
    return _store


def create_ingestion_chain(vector_store: Optional[ChromaVectorStore] = None, content_hash: Optional[str] = None):
    """
    Construct the LCEL Document Ingestion Pipeline:
    load -> clean -> chunk -> store -> summary
    """
    settings = get_settings()
    store = vector_store or get_vector_store()

    chain = (
        RunnableLambda(_load_step)
        | RunnableLambda(_clean_step)
        | RunnableLambda(_make_chunk_step(settings, content_hash=content_hash))
        | RunnableLambda(_make_store_step(store))
    )
    return chain


def run_ingestion_pipeline(
    file_path: str,
    doc_id: Optional[str] = None,
    content_hash: Optional[str] = None,
    vector_store: Optional[ChromaVectorStore] = None,
) -> Dict[str, Any]:
    """Execute the end-to-end LCEL ingestion pipeline synchronously."""
    chain = create_ingestion_chain(vector_store=vector_store, content_hash=content_hash)
    payload = {
        "file_path": file_path,
        "doc_id": doc_id,
        "content_hash": content_hash,
    }
    logger.info(f"Starting LCEL document ingestion pipeline for '{file_path}'")
    result = chain.invoke(payload)
    logger.info(f"Completed LCEL ingestion: {result['filename']} ({result['chunk_count']} chunks indexed)")
    return result
