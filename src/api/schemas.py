"""Pydantic schemas for API request and response validation."""
from typing import List, Optional, Union
from pydantic import BaseModel, Field


# --- Document Schemas ---

class DocumentUploadResponse(BaseModel):
    """Response returned after a successful document upload and ingestion."""
    doc_id: str
    filename: str
    status: str = "ingested"
    chunk_count: int
    source_type: str
    ingested_at: str


class DocumentItem(BaseModel):
    """Summary of an ingested document."""
    doc_id: str
    filename: str
    source_type: str
    chunk_count: int
    ingested_at: str


class PaginationMeta(BaseModel):
    """Pagination metadata for list queries."""
    total: int
    page: int
    page_size: int
    has_next: bool


class DocumentListResponse(BaseModel):
    """Paginated list of ingested documents."""
    documents: List[DocumentItem]
    pagination: PaginationMeta


class DocumentDetailResponse(BaseModel):
    """Detailed metadata for a single document."""
    doc_id: str
    filename: str
    source_type: str
    chunk_count: int
    ingested_at: str
    status: str = "ready"


class DocumentDeleteResponse(BaseModel):
    """Confirmation returned when a document is deleted."""
    doc_id: str
    status: str = "deleted"
    chunks_removed: int


# --- Query & Citation Schemas ---

class CitationItem(BaseModel):
    """Citation metadata identifying the source chunk."""
    chunk_id: str
    doc_id: str
    doc_name: str
    page_number: Optional[Union[int, List[int]]] = None
    section_heading: Optional[str] = None
    text_excerpt: str


class QueryRequest(BaseModel):
    """User query request payload."""
    query: str = Field(..., min_length=1, description="Question text to query against documents")
    doc_id: Optional[str] = Field(None, description="Optional document ID to restrict query scope")
    stream: bool = Field(False, description="Whether to stream response tokens")


class QueryResponse(BaseModel):
    """Successful RAG answer response with citations."""
    status: str = "answered"
    answer: str
    citations: List[CitationItem] = Field(default_factory=list)


class RefusalResponse(BaseModel):
    """Refusal response returned when context is insufficient."""
    status: str = "refused"
    reason: str
    retrieved_chunk_ids: List[str] = Field(default_factory=list)
    query: str


# --- System & Error Schemas ---

class HealthResponse(BaseModel):
    """Liveness probe response."""
    status: str = "healthy"
    chromadb: str = "connected"
    version: str = "0.1.0"


class ErrorResponse(BaseModel):
    """Standardized error response payload."""
    error_code: str
    message: str
