from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, Field


# ------------------------------------------------------------------------------
# Error Schema
# ------------------------------------------------------------------------------
class ErrorResponse(BaseModel):
    """Standardized error format across all API endpoints."""
    error_code: str = Field(..., description="Machine-readable error identifier")
    message: str = Field(..., description="Human-readable error explanation")
    correlation_id: str = Field(..., description="Request trace correlation ID")


# ------------------------------------------------------------------------------
# Document Schemas
# ------------------------------------------------------------------------------
class DocumentUploadResponse(BaseModel):
    doc_id: str = Field(..., description="Unique UUID for the uploaded document")
    filename: str = Field(..., description="Original filename of the document")
    status: str = Field(default="ingested", description="Ingestion status")
    chunk_count: int = Field(..., description="Number of chunks stored in vector database")
    source_type: str = Field(..., description="Format: pdf, docx, txt, md")
    ingested_at: str = Field(..., description="ISO 8601 ingestion timestamp")


class DocumentInfo(BaseModel):
    doc_id: str = Field(..., description="Document UUID")
    filename: str = Field(..., description="Document filename")
    source_type: str = Field(..., description="Format: pdf, docx, txt, md")
    chunk_count: int = Field(..., description="Number of indexed chunks")
    ingested_at: str = Field(..., description="ISO 8601 ingestion timestamp")


class PaginationMeta(BaseModel):
    total: int = Field(..., description="Total number of documents")
    page: int = Field(..., description="Current page number (1-indexed)")
    page_size: int = Field(..., description="Items per page")
    has_next: bool = Field(..., description="Whether next page exists")


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo] = Field(default_factory=list)
    pagination: PaginationMeta


class DocumentDetailResponse(BaseModel):
    doc_id: str
    filename: str
    source_type: str
    chunk_count: int
    ingested_at: str
    status: str = "ready"


class DocumentDeleteResponse(BaseModel):
    doc_id: str
    status: str = "deleted"
    chunks_removed: int


# ------------------------------------------------------------------------------
# Query & Citation Schemas
# ------------------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="User question")
    doc_id: str | None = Field(default=None, description="Optional document ID to scope retrieval")
    stream: bool = Field(default=False, description="Whether to stream response tokens")


class CitationItem(BaseModel):
    chunk_id: str = Field(..., description="Chunk identifier")
    doc_id: str = Field(..., description="Source document identifier")
    doc_name: str = Field(..., description="Source document filename")
    page_number: int | list[int] | None = Field(default=None, description="Page number if applicable")
    section_heading: str | None = Field(default=None, description="Heading context if detected")
    text_excerpt: str = Field(..., description="Relevant passage from chunk")


class QueryAnswerResponse(BaseModel):
    status: Literal["answered"] = "answered"
    answer: str = Field(..., description="Grounded factual answer")
    citations: list[CitationItem] = Field(default_factory=list, description="Claim citations")


class QueryRefusalResponse(BaseModel):
    status: Literal["refused"] = "refused"
    reason: str = Field(..., description="Reason for refusal: insufficient_context, low_relevance, etc.")
    retrieved_chunk_ids: list[str] = Field(default_factory=list, description="Retrieved candidate chunk IDs")
    query: str = Field(..., description="Original user query")


# ------------------------------------------------------------------------------
# Health & Evaluation Schemas
# ------------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str = "healthy"
    chromadb: str = "connected"
    version: str = "1.0.0"


class EvalRunRequest(BaseModel):
    config_overrides: dict[str, Any] | None = None


class EvalRunResponse(BaseModel):
    run_id: str
    status: str = "completed"
    config_hash: str


class EvalResultsResponse(BaseModel):
    run_id: str
    config_hash: str
    metrics: dict[str, Any]
    created_at: str
