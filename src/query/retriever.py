from typing import Any
from src.common.logging import StageTimer, logger
from src.config.settings import settings
from src.storage.chroma import RetrievedChunk, storage_client


def retrieve_context(
    query: str,
    k: int = settings.top_k,
    filters: dict[str, Any] | None = None,
) -> list[RetrievedChunk]:
    """Retrieve top-k chunks from vector store using similarity search."""
    with StageTimer("query_retrieval", extra={"k": k, "query_len": len(query)}):
        chunks = storage_client.query(query_text=query, k=k, filters=filters)
    logger.info(f"Retrieved {len(chunks)} candidate chunks for query: '{query[:50]}...'")
    return chunks
