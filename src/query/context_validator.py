from dataclasses import dataclass
from src.common.logging import logger
from src.config.settings import settings
from src.storage.chroma import RetrievedChunk


@dataclass
class ValidationResult:
    """Result of the context relevance gate."""
    is_valid: bool
    filtered_chunks: list[RetrievedChunk]
    refusal_reason: str | None = None


def validate_retrieved_context(
    chunks: list[RetrievedChunk],
    threshold: float = settings.similarity_threshold,
    enable_threshold_filtering: bool = True,
) -> ValidationResult:
    """Relevance gate running before generation to filter low-confidence context and trigger refusal if needed."""
    if not chunks:
        return ValidationResult(
            is_valid=False,
            filtered_chunks=[],
            refusal_reason="insufficient_context",
        )

    if not enable_threshold_filtering:
        return ValidationResult(is_valid=True, filtered_chunks=chunks)

    passing_chunks: list[RetrievedChunk] = []
    dropped_chunks: list[tuple[str, float]] = []

    for chunk in chunks:
        if chunk.similarity_score >= threshold:
            passing_chunks.append(chunk)
        else:
            dropped_chunks.append((chunk.chunk_id, chunk.similarity_score))

    if dropped_chunks:
        logger.info(f"Context validator dropped {len(dropped_chunks)} chunks below threshold {threshold}: {dropped_chunks}")

    if not passing_chunks:
        logger.warning("Zero chunks passed relevance threshold. Short-circuiting to refusal.")
        return ValidationResult(
            is_valid=False,
            filtered_chunks=[],
            refusal_reason="low_relevance",
        )

    return ValidationResult(
        is_valid=True,
        filtered_chunks=passing_chunks,
    )
