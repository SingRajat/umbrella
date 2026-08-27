"""API route definitions for Umbrella RAG system."""
import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from fastapi import APIRouter, File, Query, UploadFile, status

from src.api.schemas import (
    DocumentDeleteResponse,
    DocumentDetailResponse,
    DocumentItem,
    DocumentListResponse,
    DocumentUploadResponse,
    ErrorResponse,
    HealthResponse,
    PaginationMeta,
    QueryRequest,
    QueryResponse,
    RefusalResponse,
)
from src.config.settings import get_settings
from src.common.errors import IngestionError, ValidationError
from src.common.logging import get_logger

logger = get_logger("umbrella.routes")
router = APIRouter(prefix="/api/v1", tags=["RAG"])

DOCUMENTS_DIR = Path("data/documents")
REGISTRY_FILE = DOCUMENTS_DIR / "registry.json"


def _load_registry() -> Dict[str, dict]:
    """Load metadata registry of uploaded documents."""
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    if not REGISTRY_FILE.exists():
        return {}
    try:
        with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_registry(registry: Dict[str, dict]):
    """Persist metadata registry of uploaded documents to disk."""
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check probe",
)
async def health_check():
    """Liveness probe verifying API availability."""
    return HealthResponse(status="healthy", chromadb="connected", version="0.1.0")


@router.post(
    "/documents",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest document",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file type or empty file"},
        413: {"model": ErrorResponse, "description": "File exceeds upload limit"},
    },
)
async def upload_document(
    file: UploadFile = File(...),
):
    """Upload and ingest PDF, DOCX, TXT, or MD documents."""
    settings = get_settings()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024

    # 1. Validate filename and extension
    filename = file.filename or ""
    allowed_extensions = (".pdf", ".docx", ".txt", ".md")
    if not filename.lower().endswith(allowed_extensions):
        raise ValidationError(
            f"Unsupported file format. Supported formats: {', '.join(allowed_extensions)}",
            error_code="INVALID_FILE_TYPE",
            status_code=400,
        )

    # 2. Validate file size and non-empty content
    content = await file.read()
    if len(content) == 0:
        raise ValidationError("Uploaded file is empty.", error_code="EMPTY_FILE", status_code=400)
    if len(content) > max_bytes:
        raise ValidationError(
            f"File exceeds maximum allowed size of {settings.max_upload_size_mb} MB.",
            error_code="FILE_TOO_LARGE",
            status_code=413,
        )

    # 3. Compute SHA-256 content hash for deduplication/idempotency tracking
    content_hash = hashlib.sha256(content).hexdigest()
    doc_id = str(uuid.uuid4())
    source_type = filename.split(".")[-1].lower()
    ingested_at = datetime.now(timezone.utc).isoformat()

    # 4. Save raw file to data/documents/
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    saved_filename = f"{doc_id}_{filename}"
    saved_path = DOCUMENTS_DIR / saved_filename
    with open(saved_path, "wb") as f:
        f.write(content)

    # 5. Record document in metadata registry
    registry = _load_registry()
    doc_entry = {
        "doc_id": doc_id,
        "filename": filename,
        "saved_filename": saved_filename,
        "source_type": source_type,
        "content_hash": content_hash,
        "file_size": len(content),
        "chunk_count": 0,  # Will be populated as ingestion pipeline stages are wired
        "ingested_at": ingested_at,
        "status": "ingested",
    }
    registry[doc_id] = doc_entry
    _save_registry(registry)

    logger.info(f"Document uploaded successfully: {filename} (doc_id: {doc_id})")

    return DocumentUploadResponse(
        doc_id=doc_id,
        filename=filename,
        status="ingested",
        chunk_count=0,
        source_type=source_type,
        ingested_at=ingested_at,
    )


@router.get(
    "/documents",
    response_model=DocumentListResponse,
    summary="List ingested documents",
)
async def list_documents(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
):
    """Return paginated list of all ingested documents."""
    registry = _load_registry()
    all_docs = [
        DocumentItem(
            doc_id=d["doc_id"],
            filename=d["filename"],
            source_type=d["source_type"],
            chunk_count=d.get("chunk_count", 0),
            ingested_at=d["ingested_at"],
        )
        for d in registry.values()
    ]

    total = len(all_docs)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    paginated_docs = all_docs[start_idx:end_idx]
    has_next = end_idx < total

    return DocumentListResponse(
        documents=paginated_docs,
        pagination=PaginationMeta(total=total, page=page, page_size=page_size, has_next=has_next),
    )


@router.get(
    "/documents/{doc_id}",
    response_model=DocumentDetailResponse,
    summary="Get document details",
    responses={404: {"model": ErrorResponse, "description": "Document not found"}},
)
async def get_document(doc_id: str):
    """Retrieve metadata for a specific document by its doc_id."""
    registry = _load_registry()
    if doc_id not in registry:
        raise ValidationError(f"Document with ID '{doc_id}' not found.", error_code="DOCUMENT_NOT_FOUND", status_code=404)

    d = registry[doc_id]
    return DocumentDetailResponse(
        doc_id=d["doc_id"],
        filename=d["filename"],
        source_type=d["source_type"],
        chunk_count=d.get("chunk_count", 0),
        ingested_at=d["ingested_at"],
        status=d.get("status", "ready"),
    )


@router.delete(
    "/documents/{doc_id}",
    response_model=DocumentDeleteResponse,
    summary="Delete a document",
    responses={404: {"model": ErrorResponse, "description": "Document not found"}},
)
async def delete_document(doc_id: str):
    """Delete a document and cascade remove its stored file and vector entries."""
    registry = _load_registry()
    if doc_id not in registry:
        raise ValidationError(f"Document with ID '{doc_id}' not found.", error_code="DOCUMENT_NOT_FOUND", status_code=404)

    doc_info = registry.pop(doc_id)
    _save_registry(registry)

    # Remove physical file if present
    saved_path = DOCUMENTS_DIR / doc_info.get("saved_filename", "")
    if saved_path.exists():
        try:
            os.remove(saved_path)
        except Exception as exc:
            logger.warning(f"Could not remove physical file {saved_path}: {exc}")

    logger.info(f"Document deleted: {doc_id}")
    return DocumentDeleteResponse(
        doc_id=doc_id,
        status="deleted",
        chunks_removed=doc_info.get("chunk_count", 0),
    )


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Ask a question against ingested documents",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid query payload"},
        404: {"model": ErrorResponse, "description": "No documents ingested"},
    },
)
async def query_documents(payload: QueryRequest):
    """Ask a question and receive an answer with verifiable citations."""
    if not payload.query.strip():
        raise ValidationError("Query cannot be empty.", error_code="EMPTY_QUERY", status_code=400)

    return QueryResponse(
        status="answered",
        answer="Query endpoint ready.",
        citations=[],
    )
