"""PostgreSQL + pgvector storage client placeholder.

This module provides the future extension point for PostgreSQL / Supabase storage
when migrating from ChromaDB as outlined in the PRD and TechStack.md.
"""
from typing import Any
from src.storage.chroma import ChunkRecord, RetrievedChunk, VectorStoreClient


class PostgresVectorStore(VectorStoreClient):
    """Postgres / pgvector storage client implementation (Future V2/Production candidate)."""

    def __init__(self, connection_string: str | None = None):
        self.connection_string = connection_string

    def upsert(self, chunks: list[ChunkRecord]) -> None:
        raise NotImplementedError("PostgreSQL storage is deferred for V1. Use ChromaVectorStore.")

    def query(
        self, query_text: str, k: int = 3, filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        raise NotImplementedError("PostgreSQL storage is deferred for V1. Use ChromaVectorStore.")

    def delete_by_doc_id(self, doc_id: str) -> int:
        raise NotImplementedError("PostgreSQL storage is deferred for V1. Use ChromaVectorStore.")

    def list_documents(self) -> list[dict[str, Any]]:
        raise NotImplementedError("PostgreSQL storage is deferred for V1. Use ChromaVectorStore.")

    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        raise NotImplementedError("PostgreSQL storage is deferred for V1. Use ChromaVectorStore.")

    def check_content_hash(self, content_hash: str) -> str | None:
        raise NotImplementedError("PostgreSQL storage is deferred for V1. Use ChromaVectorStore.")
