from typing import Any
from langchain_core.runnables import RunnableLambda

from src.common.errors import DuplicateDocumentError, IngestionError
from src.common.logging import StageTimer, logger
from src.ingestion.chunker import default_chunker
from src.ingestion.cleaner import clean_pages
from src.ingestion.loaders import compute_sha256, load_document
from src.ingestion.metadata import construct_chunk_records
from src.storage.chroma import storage_client


def _check_idempotency_step(input_data: dict[str, Any]) -> dict[str, Any]:
    file_bytes = input_data["file_bytes"]
    content_hash = compute_sha256(file_bytes)
    existing_doc_id = storage_client.check_content_hash(content_hash)
    if existing_doc_id:
        logger.info(f"Duplicate document upload detected with hash {content_hash}. Doc ID: {existing_doc_id}")
        raise DuplicateDocumentError(
            message=f"Document with identical content already exists (doc_id: {existing_doc_id}).",
            existing_doc_id=existing_doc_id,
        )
    return input_data


def _load_step(input_data: dict[str, Any]) -> dict[str, Any]:
    with StageTimer("ingestion_load", extra={"filename": input_data["filename"]}):
        raw_doc = load_document(
            file_bytes=input_data["file_bytes"],
            filename=input_data["filename"],
            extension=input_data["extension"],
        )
    input_data["raw_doc"] = raw_doc
    return input_data


def _clean_step(input_data: dict[str, Any]) -> dict[str, Any]:
    with StageTimer("ingestion_clean", extra={"filename": input_data["filename"]}):
        cleaned_pages = clean_pages(input_data["raw_doc"].pages)
    input_data["cleaned_pages"] = cleaned_pages
    return input_data


def _chunk_step(input_data: dict[str, Any]) -> dict[str, Any]:
    with StageTimer("ingestion_chunk", extra={"page_count": len(input_data["cleaned_pages"])}):
        chunks = default_chunker.chunk(input_data["cleaned_pages"])
    if not chunks:
        raise IngestionError("Chunking resulted in zero chunks.")
    input_data["chunks"] = chunks
    return input_data


def _metadata_step(input_data: dict[str, Any]) -> dict[str, Any]:
    with StageTimer("ingestion_metadata"):
        records = construct_chunk_records(
            raw_doc=input_data["raw_doc"],
            chunks=input_data["chunks"],
            domain_tag=input_data.get("domain_tag", "general"),
        )
    input_data["records"] = records
    return input_data


def _storage_step(input_data: dict[str, Any]) -> dict[str, Any]:
    with StageTimer("ingestion_storage", extra={"chunk_count": len(input_data["records"])}):
        storage_client.upsert(input_data["records"])

    raw_doc = input_data["raw_doc"]
    return {
        "doc_id": raw_doc.doc_id,
        "filename": raw_doc.filename,
        "status": "ingested",
        "chunk_count": len(input_data["records"]),
        "source_type": raw_doc.source_type,
        "ingested_at": raw_doc.uploaded_at,
    }


# Declarative LCEL Ingestion Chain
ingestion_chain = (
    RunnableLambda(_check_idempotency_step)
    | RunnableLambda(_load_step)
    | RunnableLambda(_clean_step)
    | RunnableLambda(_chunk_step)
    | RunnableLambda(_metadata_step)
    | RunnableLambda(_storage_step)
)


def run_ingestion(
    file_bytes: bytes,
    filename: str,
    extension: str,
    domain_tag: str | None = "general",
) -> dict[str, Any]:
    """Runs the declarative LCEL ingestion pipeline."""
    return ingestion_chain.invoke(
        {
            "file_bytes": file_bytes,
            "filename": filename,
            "extension": extension,
            "domain_tag": domain_tag,
        }
    )
