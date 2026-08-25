from dataclasses import dataclass
from src.api.schemas import CitationItem, QueryAnswerResponse, QueryRefusalResponse
from src.common.logging import logger
from src.config.settings import settings
from src.query.generator import AnswerWithCitations
from src.storage.chroma import RetrievedChunk


@dataclass
class OutputValidationResult:
    is_valid: bool
    response: QueryAnswerResponse | QueryRefusalResponse


def validate_and_construct_response(
    answer_obj: AnswerWithCitations,
    retrieved_chunks: list[RetrievedChunk],
    query: str,
    strict_refusal: bool = settings.strict_refusal,
) -> OutputValidationResult:
    """Validates citations against retrieved chunks and constructs standard API response."""
    retrieved_map: dict[str, RetrievedChunk] = {c.chunk_id: c for c in retrieved_chunks}
    valid_citations: list[CitationItem] = []
    hallucinated_citations: list[str] = []

    for cid in answer_obj.citations:
        if cid in retrieved_map:
            chunk = retrieved_map[cid]
            valid_citations.append(
                CitationItem(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    doc_name=chunk.metadata.get("doc_name", "unknown"),
                    page_number=chunk.metadata.get("page_number") or None,
                    section_heading=chunk.metadata.get("section_heading") or None,
                    text_excerpt=chunk.text[:250].strip() + ("..." if len(chunk.text) > 250 else ""),
                )
            )
        else:
            hallucinated_citations.append(cid)

    if hallucinated_citations:
        logger.warning(
            f"Caught {len(hallucinated_citations)} hallucinated citations: {hallucinated_citations}"
        )
        if strict_refusal:
            logger.info("Strict refusal triggered due to hallucinated citation.")
            return OutputValidationResult(
                is_valid=False,
                response=QueryRefusalResponse(
                    status="refused",
                    reason="unsupported_claim",
                    retrieved_chunk_ids=[c.chunk_id for c in retrieved_chunks],
                    query=query,
                ),
            )

    # Citation coverage check: if answer claims it cannot answer or has 0 citations
    answer_lower = answer_obj.answer.lower()
    refusal_phrases = ["cannot answer", "not enough information", "insufficient context", "not mentioned in the context"]
    if any(phrase in answer_lower for phrase in refusal_phrases) or not valid_citations:
        if len(answer_obj.answer.strip()) > 30 and not valid_citations and strict_refusal:
            logger.warning("Non-trivial answer produced with 0 valid citations. Refusing per safety guardrails.")
            return OutputValidationResult(
                is_valid=False,
                response=QueryRefusalResponse(
                    status="refused",
                    reason="insufficient_context",
                    retrieved_chunk_ids=[c.chunk_id for c in retrieved_chunks],
                    query=query,
                ),
            )

    return OutputValidationResult(
        is_valid=True,
        response=QueryAnswerResponse(
            status="answered",
            answer=answer_obj.answer,
            citations=valid_citations,
        ),
    )
