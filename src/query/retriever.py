"""Retrieval logic querying the vector store with LangChain retriever support."""
from typing import List, Optional
from langchain_core.retrievers import BaseRetriever

from src.config.settings import get_settings
from src.common.logging import get_logger
from src.storage.chroma import ChromaVectorStore, RetrievedChunk, get_vector_store

logger = get_logger("umbrella.query.retriever")


def retrieve_chunks(
    query_text: str,
    doc_id: Optional[str] = None,
    k: Optional[int] = None,
    vector_store: Optional[ChromaVectorStore] = None,
) -> List[RetrievedChunk]:
    """
    Retrieve top-k relevant document chunks for a query using cosine similarity.

    Args:
        query_text: Search query string.
        doc_id: Optional document ID to filter query scope.
        k: Number of chunks to retrieve (defaults to settings.top_k = 3).
        vector_store: Optional ChromaVectorStore instance.

    Returns:
        List of RetrievedChunk objects ordered by relevance.
    """
    settings = get_settings()
    store = vector_store or get_vector_store()
    top_k = k if k is not None else settings.top_k

    logger.info(f"Executing retrieval for query='{query_text[:50]}...' (top_k={top_k}, doc_id={doc_id})")
    chunks = store.query(query_text=query_text, k=top_k, doc_id=doc_id)
    logger.info(f"Retrieved {len(chunks)} chunks from vector store")
    return chunks


def get_langchain_retriever(
    doc_id: Optional[str] = None,
    k: Optional[int] = None,
    vector_store: Optional[ChromaVectorStore] = None,
) -> BaseRetriever:
    """Return a LangChain BaseRetriever from ChromaVectorStore for LCEL composition."""
    settings = get_settings()
    store = vector_store or get_vector_store()
    top_k = k if k is not None else settings.top_k
    where_filter = {"doc_id": doc_id} if doc_id else None

    return store.get_langchain_vectorstore().as_retriever(
        search_kwargs={"k": top_k, "filter": where_filter}
    )
