import json
import os
from typing import Any
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse

from src.api.schemas import (
    CitationItem,
    DocumentDeleteResponse,
    DocumentDetailResponse,
    DocumentInfo,
    DocumentListResponse,
    DocumentUploadResponse,
    EvalResultsResponse,
    EvalRunRequest,
    EvalRunResponse,
    HealthResponse,
    PaginationMeta,
    QueryAnswerResponse,
    QueryRefusalResponse,
    QueryRequest,
)
from src.chains.ingestion_chain import run_ingestion
from src.chains.query_chain import run_query, run_query_stream
from src.common.errors import DocumentNotFoundError, FileTooLargeError, UnsupportedFileTypeError
from src.common.logging import logger
from src.eval.metrics_store import get_eval_result, run_eval_pipeline
from src.storage.chroma import storage_client

router = APIRouter(prefix="/api/v1")

MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB limit
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


# ------------------------------------------------------------------------------
# Document Management Endpoints
# ------------------------------------------------------------------------------
@router.post(
    "/documents",
    response_model=DocumentUploadResponse,
    summary="Upload and ingest document",
    status_code=200,
)
async def upload_document(file: UploadFile = File(...)):
    """Upload a PDF, DOCX, TXT, or MD document, parse, clean, chunk, and index."""
    filename = file.filename or "uploaded_document"
    ext = os.path.splitext(filename)[1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"Unsupported file format '{ext}'. Allowed formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    content = await file.read()
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise FileTooLargeError(
            f"File size ({len(content)} bytes) exceeds the maximum allowed 25MB."
        )

    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Execute ingestion chain
    result = run_ingestion(file_bytes=content, filename=filename, extension=ext)
    return DocumentUploadResponse(**result)


@router.get(
    "/documents",
    response_model=DocumentListResponse,
    summary="List all indexed documents",
)
async def list_documents(
    page: int = Query(default=1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(default=20, ge=1, le=100, description="Items per page"),
):
    """Returns paginated list of ingested documents."""
    all_docs = storage_client.list_documents()
    total = len(all_docs)

    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    paged_docs = all_docs[start_idx:end_idx]

    documents = [
        DocumentInfo(
            doc_id=doc["doc_id"],
            filename=doc["filename"],
            source_type=doc["source_type"],
            chunk_count=doc["chunk_count"],
            ingested_at=doc["ingested_at"],
        )
        for doc in paged_docs
    ]

    return DocumentListResponse(
        documents=documents,
        pagination=PaginationMeta(
            total=total,
            page=page,
            page_size=page_size,
            has_next=end_idx < total,
        ),
    )


@router.get(
    "/documents/{doc_id}",
    response_model=DocumentDetailResponse,
    summary="Get document details by ID",
)
async def get_document(doc_id: str):
    """Fetch metadata and ingestion status for a document."""
    doc = storage_client.get_document(doc_id)
    if not doc:
        raise DocumentNotFoundError(f"Document with ID '{doc_id}' not found.")
    return DocumentDetailResponse(**doc)


@router.delete(
    "/documents/{doc_id}",
    response_model=DocumentDeleteResponse,
    summary="Delete document and its indexed chunks",
)
async def delete_document(doc_id: str):
    """Cascade delete document chunks from vector store."""
    doc = storage_client.get_document(doc_id)
    if not doc:
        raise DocumentNotFoundError(f"Document with ID '{doc_id}' not found.")

    removed_chunks = storage_client.delete_by_doc_id(doc_id)
    return DocumentDeleteResponse(doc_id=doc_id, status="deleted", chunks_removed=removed_chunks)


# ------------------------------------------------------------------------------
# Query & Generation Endpoints
# ------------------------------------------------------------------------------
@router.post(
    "/query",
    response_model=QueryAnswerResponse | QueryRefusalResponse,
    summary="Ask a question grounded in ingested documents",
)
async def query_rag(request: QueryRequest):
    """Execute RAG pipeline: retrieve relevant context, validate, and generate grounded answer with citations."""
    if request.stream:
        return StreamingResponse(
            run_query_stream(query=request.query, doc_id=request.doc_id),
            media_type="text/event-stream",
        )

    response = run_query(query=request.query, doc_id=request.doc_id)
    return response


# ------------------------------------------------------------------------------
# Health Check Endpoint
# ------------------------------------------------------------------------------
@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
)
async def health_check():
    """Verify backend and ChromaDB connectivity."""
    is_healthy = storage_client.is_healthy()
    return HealthResponse(
        status="healthy" if is_healthy else "degraded",
        chromadb="connected" if is_healthy else "error",
        version="1.0.0",
    )


# ------------------------------------------------------------------------------
# Evaluation Endpoints
# ------------------------------------------------------------------------------
@router.post(
    "/eval/run",
    response_model=EvalRunResponse,
    summary="Trigger offline evaluation run",
)
async def trigger_eval(request: EvalRunRequest):
    """Run RAGAS evaluation on the configured dataset."""
    result = run_eval_pipeline(config_overrides=request.config_overrides)
    return EvalRunResponse(**result)


@router.get(
    "/eval/results/{run_id}",
    response_model=EvalResultsResponse,
    summary="Get results for an evaluation run",
)
async def get_eval_results(run_id: str):
    """Fetch stored historical experiment evaluation results."""
    result = get_eval_result(run_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Eval run '{run_id}' not found.")
    return EvalResultsResponse(**result)
