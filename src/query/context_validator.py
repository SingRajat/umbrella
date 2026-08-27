"""Retrieved context validation and pre-generation guardrails."""
from dataclasses import dataclass, field
from typing import List, Optional

from src.config.settings import get_settings
from src.common.logging import get_logger
from src.storage.chroma import RetrievedChunk

logger = get_logger("umbrella.query.validator")


@dataclass
class ContextValidationResult:
    """Represents the outcome of context validation before LLM generation."""
    is_valid: bool
    reason: Optional[str] = None
    valid_chunks: List[RetrievedChunk] = field(default_factory=list)


def validate_retrieved_context(
    chunks: List[RetrievedChunk],
    similarity_threshold: Optional[float] = None,
) -> ContextValidationResult:
    """
    Validate that retrieved chunks contain sufficient relevance for generation.

    Guardrails:
    1. If 0 chunks retrieved -> Refusal: "No relevant documents found in knowledge base."
    2. If top similarity score < threshold (default 0.5) -> Refusal: "Top retrieved evidence is below confidence threshold."

    Args:
        chunks: List of RetrievedChunk objects from vector store.
        similarity_threshold: Minimum cosine similarity score required (default 0.5).

    Returns:
        ContextValidationResult with validity flag, refusal reason, and filtered chunks.
    """
    settings = get_settings()
    threshold = similarity_threshold if similarity_threshold is not None else settings.similarity_threshold

    if not chunks:
        logger.warning("Context validation failed: 0 chunks retrieved.")
        return ContextValidationResult(
            is_valid=False,
            reason="No relevant document context found. Please ensure relevant documents are uploaded.",
            valid_chunks=[],
        )

    # Filter chunks meeting the similarity threshold
    above_threshold = [c for c in chunks if c.similarity_score >= threshold]

    if not above_threshold:
        top_score = max(c.similarity_score for c in chunks)
        logger.warning(
            f"Context validation failed: top similarity score {top_score:.3f} is below threshold {threshold:.3f}"
        )
        return ContextValidationResult(
            is_valid=False,
            reason=f"Retrieved context confidence ({top_score:.2f}) is below the required threshold ({threshold:.2f}). Refusing to answer to avoid hallucination.",
            valid_chunks=[],
        )

    logger.info(f"Context validation passed: {len(above_threshold)}/{len(chunks)} chunks above threshold {threshold:.2f}")
    return ContextValidationResult(
        is_valid=True,
        reason=None,
        valid_chunks=above_threshold,
    )
