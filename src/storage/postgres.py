"""PostgreSQL storage client extension placeholder for future V2 hybrid storage."""
from typing import List, Optional
from src.ingestion.metadata import ChunkRecord
from src.storage.chroma import RetrievedChunk, VectorStoreClient


class PostgresStore(VectorStoreClient):
    """
    Placeholder for PostgreSQL/pgvector implementation.
    Deferred to future versions following baseline evaluation.
    """

    def upsert(self, chunks: List[ChunkRecord]) -> None:
        raise NotImplementedError("PostgreSQL storage is deferred to future architecture versions.")

    def query(self, query_text: str, k: int = 3, doc_id: Optional[str] = None) -> List[RetrievedChunk]:
        raise NotImplementedError("PostgreSQL storage is deferred to future architecture versions.")

    def delete_by_doc_id(self, doc_id: str) -> int:
        raise NotImplementedError("PostgreSQL storage is deferred to future architecture versions.")

    def check_content_hash(self, content_hash: str) -> Optional[str]:
        raise NotImplementedError("PostgreSQL storage is deferred to future architecture versions.")

    def count(self) -> int:
        raise NotImplementedError("PostgreSQL storage is deferred to future architecture versions.")
