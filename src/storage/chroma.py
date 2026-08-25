import os
import time
from dataclasses import dataclass
from typing import Any, Protocol
import chromadb
from chromadb.config import Settings as ChromaSettings

from src.common.errors import StorageConnectionError
from src.common.logging import logger
from src.config.settings import settings


@dataclass
class ChunkRecord:
    """Represents a chunk prepared for storage in vector database."""
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict[str, Any]


@dataclass
class RetrievedChunk:
    """Represents a retrieved chunk with similarity score."""
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict[str, Any]
    similarity_score: float


class VectorStoreClient(Protocol):
    """Protocol interface for swappable vector store implementations."""

    def upsert(self, chunks: list[ChunkRecord]) -> None:
        """Insert or update chunks and their embeddings."""
        ...

    def query(
        self, query_text: str, k: int = 3, filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        """Query vector database with text query and optional metadata filters."""
        ...

    def delete_by_doc_id(self, doc_id: str) -> int:
        """Delete all chunks belonging to a document ID. Returns removed chunk count."""
        ...

    def list_documents(self) -> list[dict[str, Any]]:
        """List distinct documents with metadata."""
        ...

    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        """Get document details by doc_id."""
        ...

    def check_content_hash(self, content_hash: str) -> str | None:
        """Check if document with given SHA-256 hash already exists. Returns doc_id if found."""
        ...


def with_retry(max_attempts: int = 3, base_delay: float = 0.5):
    """Decorator to retry transient database operations with exponential backoff."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            last_err = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_err = e
                    delay = base_delay * (2 ** (attempt - 1))
                    logger.warning(
                        f"Database operation {func.__name__} failed attempt {attempt}/{max_attempts}: {e}. Retrying in {delay}s"
                    )
                    time.sleep(delay)
            logger.error(f"Database operation {func.__name__} failed after {max_attempts} attempts: {last_err}")
            raise StorageConnectionError(f"Storage operation failed: {last_err}") from last_err

        return wrapper

    return decorator


class ChromaVectorStore:
    """ChromaDB vector store client with built-in embeddings and bounded retry."""

    def __init__(
        self,
        persist_dir: str = settings.chroma_persist_dir,
        collection_name: str = settings.chroma_collection_name,
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        os.makedirs(self.persist_dir, exist_ok=True)
        self._init_client()

    def _init_client(self):
        try:
            self.client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            # Default embedding function: all-MiniLM-L6-v2 via onnxruntime
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {e}")
            raise StorageConnectionError(f"Cannot initialize ChromaDB: {e}") from e

    @with_retry()
    def upsert(self, chunks: list[ChunkRecord]) -> None:
        if not chunks:
            return

        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = []
        for c in chunks:
            # ChromaDB metadata values must be primitive types (str, int, float, bool)
            meta = {}
            for k, v in c.metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    meta[k] = v
                elif isinstance(v, list):
                    meta[k] = ",".join(map(str, v))
                elif v is None:
                    meta[k] = ""
                else:
                    meta[k] = str(v)
            metadatas.append(meta)

        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info(f"Upserted {len(chunks)} chunks into ChromaDB collection {self.collection_name}")

    @with_retry()
    def query(
        self, query_text: str, k: int = 3, filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        # If collection is empty, return empty list immediately
        if self.collection.count() == 0:
            return []

        where_clause = None
        if filters:
            # Convert filter dict to ChromaDB where clause if needed
            valid_filters = {k: v for k, v in filters.items() if v is not None}
            if len(valid_filters) == 1:
                where_clause = valid_filters
            elif len(valid_filters) > 1:
                where_clause = {"$and": [{k: v} for k, v in valid_filters.items()]}

        results = self.collection.query(
            query_texts=[query_text],
            n_results=min(k, self.collection.count()),
            where=where_clause,
            include=["documents", "metadatas", "distances"],
        )

        retrieved: list[RetrievedChunk] = []
        if not results or not results["ids"] or not results["ids"][0]:
            return retrieved

        ids = results["ids"][0]
        docs = results["documents"][0] if results["documents"] else [""] * len(ids)
        metas = results["metadatas"][0] if results["metadatas"] else [{}] * len(ids)
        distances = results["distances"][0] if results.get("distances") and results["distances"] else [0.0] * len(ids)

        for chunk_id, doc_text, meta, dist in zip(ids, docs, metas, distances):
            # ChromaDB cosine distance: distance = 1 - cosine_similarity
            similarity = round(max(0.0, 1.0 - float(dist)), 4)
            doc_id = str(meta.get("doc_id", ""))
            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    text=doc_text,
                    metadata=meta,
                    similarity_score=similarity,
                )
            )

        return retrieved

    @with_retry()
    def delete_by_doc_id(self, doc_id: str) -> int:
        existing = self.collection.get(where={"doc_id": doc_id})
        ids_to_delete = existing.get("ids", [])
        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)
            logger.info(f"Deleted {len(ids_to_delete)} chunks for doc_id {doc_id}")
            return len(ids_to_delete)
        return 0

    @with_retry()
    def list_documents(self) -> list[dict[str, Any]]:
        count = self.collection.count()
        if count == 0:
            return []

        all_records = self.collection.get(include=["metadatas"])
        docs_map: dict[str, dict[str, Any]] = {}

        for meta in all_records.get("metadatas", []):
            doc_id = meta.get("doc_id")
            if not doc_id:
                continue

            if doc_id not in docs_map:
                docs_map[doc_id] = {
                    "doc_id": doc_id,
                    "filename": meta.get("doc_name", "unknown"),
                    "source_type": meta.get("source_type", "txt"),
                    "chunk_count": 0,
                    "ingested_at": meta.get("ingested_at", ""),
                }
            docs_map[doc_id]["chunk_count"] += 1

        return list(docs_map.values())

    @with_retry()
    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        records = self.collection.get(where={"doc_id": doc_id}, include=["metadatas"])
        metas = records.get("metadatas", [])
        if not metas:
            return None

        first_meta = metas[0]
        return {
            "doc_id": doc_id,
            "filename": first_meta.get("doc_name", "unknown"),
            "source_type": first_meta.get("source_type", "txt"),
            "chunk_count": len(metas),
            "ingested_at": first_meta.get("ingested_at", ""),
            "status": "ready",
        }

    @with_retry()
    def check_content_hash(self, content_hash: str) -> str | None:
        if self.collection.count() == 0:
            return None
        records = self.collection.get(where={"content_hash": content_hash}, limit=1, include=["metadatas"])
        metas = records.get("metadatas", [])
        if metas and len(metas) > 0:
            return metas[0].get("doc_id")
        return None

    @with_retry()
    def is_healthy(self) -> bool:
        """Check heartbeat/count of collection to ensure healthy storage."""
        self.collection.count()
        return True


# Global default instance
storage_client = ChromaVectorStore()
