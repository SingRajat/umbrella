"""ChromaDB vector store integration using LangChain Chroma wrapper with bounded retry."""
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.config.settings import get_settings
from src.common.errors import RetrievalError
from src.common.logging import get_logger
from src.ingestion.metadata import ChunkRecord

logger = get_logger("umbrella.storage.chroma")


class ChromaDefaultEmbeddings(Embeddings):
    """LangChain Embeddings adapter wrapping ChromaDB's built-in default embedding function."""

    def __init__(self):
        self._fn = DefaultEmbeddingFunction()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed list of document texts using Chroma's built-in model."""
        return self._fn(list(texts))

    def embed_query(self, text: str) -> List[float]:
        """Embed single query string using Chroma's built-in model."""
        return self._fn([text])[0]


@dataclass
class RetrievedChunk:
    """Represents a retrieved document chunk with relevance scoring."""
    chunk_id: str
    doc_id: str
    doc_name: str
    source_type: str
    page_number: Optional[int]
    section_heading: Optional[str]
    chunk_index: int
    char_start: int
    char_end: int
    text: str
    similarity_score: float


class VectorStoreClient(Protocol):
    """Protocol interface defining vector store operations."""

    def upsert(self, chunks: List[ChunkRecord]) -> None: ...
    def query(self, query_text: str, k: int = 3, doc_id: Optional[str] = None) -> List[RetrievedChunk]: ...
    def delete_by_doc_id(self, doc_id: str) -> int: ...
    def check_content_hash(self, content_hash: str) -> Optional[str]: ...
    def count(self) -> int: ...
    def get_langchain_vectorstore(self) -> Chroma: ...


class ChromaVectorStore:
    """ChromaDB vector store implementation using LangChain Chroma for LCEL orchestration."""

    def __init__(self, persist_dir: Optional[str] = None, collection_name: str = "umbrella_docs"):
        settings = get_settings()
        self.persist_dir = Path(persist_dir or settings.chroma_persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self._embeddings = ChromaDefaultEmbeddings()

        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # LangChain Chroma instance for LCEL pipeline composition
        self._langchain_store = Chroma(
            client=self._client,
            collection_name=self.collection_name,
            embedding_function=self._embeddings,
        )
        logger.info(f"Initialized LangChain Chroma vector store at '{self.persist_dir}'")

    def get_langchain_vectorstore(self) -> Chroma:
        """Return the underlying LangChain Chroma instance for LCEL retriever chains."""
        return self._langchain_store

    def _execute_with_retry(self, operation, *args, **kwargs):
        """Execute a database operation with bounded retry and exponential backoff."""
        max_attempts = 3
        backoff = 0.5
        last_err = None

        for attempt in range(1, max_attempts + 1):
            try:
                return operation(*args, **kwargs)
            except Exception as exc:
                last_err = exc
                logger.warning(f"ChromaDB operation failed (attempt {attempt}/{max_attempts}): {exc}")
                if attempt < max_attempts:
                    time.sleep(backoff)
                    backoff *= 2

        logger.error(f"ChromaDB operation aborted after {max_attempts} attempts: {last_err}")
        raise RetrievalError(f"Vector database operation failed: {last_err}") from last_err

    def upsert(self, chunks: List[ChunkRecord]) -> None:
        """Add or update chunk records and generate embeddings in ChromaDB."""
        if not chunks:
            return

        ids = [c.chunk_id for c in chunks]
        documents = [
            Document(page_content=c.text, metadata=c.to_metadata_dict())
            for c in chunks
        ]

        def _do_upsert():
            self._langchain_store.add_documents(documents=documents, ids=ids)

        self._execute_with_retry(_do_upsert)
        logger.info(f"Upserted {len(chunks)} chunks via LangChain Chroma into '{self.collection_name}'")

    def query(self, query_text: str, k: int = 3, doc_id: Optional[str] = None) -> List[RetrievedChunk]:
        """Query ChromaDB for top-k similar chunks with cosine relevance scores."""
        if not query_text.strip():
            return []

        where_filter = {"doc_id": doc_id} if doc_id else None

        def _do_query():
            return self._langchain_store.similarity_search_with_score(
                query=query_text,
                k=k,
                filter=where_filter,
            )

        results = self._execute_with_retry(_do_query)

        retrieved: List[RetrievedChunk] = []
        for doc, score in results:
            meta = doc.metadata
            similarity = max(0.0, 1.0 - float(score)) if isinstance(score, (int, float)) else 0.0
            page_num = meta.get("page_number")
            actual_page = page_num if page_num != -1 else None

            retrieved.append(
                RetrievedChunk(
                    chunk_id=meta.get("chunk_id", ""),
                    doc_id=meta.get("doc_id", ""),
                    doc_name=meta.get("doc_name", ""),
                    source_type=meta.get("source_type", "txt"),
                    page_number=actual_page,
                    section_heading=meta.get("section_heading") or None,
                    chunk_index=meta.get("chunk_index", 0),
                    char_start=meta.get("char_start", 0),
                    char_end=meta.get("char_end", 0),
                    text=doc.page_content,
                    similarity_score=round(similarity, 4),
                )
            )

        return retrieved

    def delete_by_doc_id(self, doc_id: str) -> int:
        """Delete all vectors associated with a specific doc_id."""
        def _do_delete():
            existing = self._collection.get(where={"doc_id": doc_id})
            ids_to_del = existing.get("ids", [])
            if ids_to_del:
                self._collection.delete(ids=ids_to_del)
            return len(ids_to_del)

        removed_count = self._execute_with_retry(_do_delete)
        logger.info(f"Deleted {removed_count} chunks for doc_id '{doc_id}' from ChromaDB")
        return removed_count

    def check_content_hash(self, content_hash: str) -> Optional[str]:
        """Check if a document with the same content_hash already exists in the vector store."""
        if not content_hash:
            return None

        def _do_check():
            res = self._collection.get(where={"content_hash": content_hash}, limit=1)
            if res and res.get("metadatas") and len(res["metadatas"]) > 0:
                return res["metadatas"][0].get("doc_id")
            return None

        return self._execute_with_retry(_do_check)

    def count(self) -> int:
        """Return total number of chunks stored in the collection."""
        return self._execute_with_retry(lambda: self._collection.count())


# Global singleton instance
_vector_store_instance: Optional[ChromaVectorStore] = None


def get_vector_store() -> ChromaVectorStore:
    """Return singleton instance of ChromaVectorStore."""
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = ChromaVectorStore()
    return _vector_store_instance
