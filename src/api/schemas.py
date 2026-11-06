"""Pydantic schemas for API request and response validation."""
from typing import List, Dict, Optional, Union
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

class ConversationMessage(BaseModel):
    """A single user or assistant conversation turn."""
    role: str = Field(..., pattern="^(user|assistant)$", description="Speaker role: 'user' or 'assistant'")
    content: str = Field(..., min_length=1, max_length=10000, description="Message text content")


class QueryRequest(BaseModel):
    """User query request payload."""
    query: str = Field(..., min_length=1, description="Question text to query against documents")
    doc_id: str = Field(..., min_length=1, description="Document ID to scope the query")
    stream: bool = Field(False, description="Whether to stream response tokens")
    conversation_history: Optional[List[ConversationMessage]] = Field(
        None, description="Recent user/assistant conversation turns"
    )
    conversation_summary: Optional[str]= Field(
        None, max_length=4000, description="Rolling summary of older conversation context"
    )

class QueryResponse(BaseModel):
    """Successful RAG answer response with citations."""
    status: str = "answered"
    answer: str
    citations: List[CitationItem] = Field(default_factory=list)
    token_usage: Optional[Dict[str, int]] = None
    active_history: List[ConversationMessage]= Field(default_factory=list)
    conversation_summary: Optional[str]= None

class RefusalResponse(BaseModel):
    """Refusal response returned when context is insufficient."""
    status: str = "refused"
    reason: str
    retrieved_chunk_ids: List[str] = Field(default_factory=list)
    token_usage: Optional[Dict[str, int]] = None
    query: str
    active_history: List[ConversationMessage]= Field(default_factory=list)
    conversation_summary: Optional[str] = None

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
