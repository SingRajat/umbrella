"""Answer and citation validation, refusal detection, and source mapping."""
import re
from dataclasses import dataclass, field
from typing import List, Optional, Set

from src.api.schemas import CitationItem
from src.common.logging import get_logger
from src.storage.chroma import RetrievedChunk

logger = get_logger("umbrella.query.output_validator")


@dataclass
class ValidatedOutput:
    """Outcome of validating LLM-generated answer and citations."""
    is_refusal: bool
    refusal_reason: Optional[str] = None
    answer: str = ""
    citations: List[CitationItem] = field(default_factory=list)


def extract_cited_indices(answer_text: str) -> Set[int]:
    """
    Extract citation numbers from bracketed notations like [1], [2], [1, 2], [1, 3, 4].
    """
    cited_indices: Set[int] = set()
    # Matches patterns like [1], [1, 2], [1, 2, 3]
    matches = re.findall(r"\[([0-9,\s]+)\]", answer_text)
    for match in matches:
        parts = match.split(",")
        for p in parts:
            p_clean = p.strip()
            if p_clean.isdigit():
                cited_indices.add(int(p_clean))

    return cited_indices


def validate_generated_output(
    answer: str,
    retrieved_chunks: List[RetrievedChunk],
) -> ValidatedOutput:
    """
    Validate LLM response for refusal phrases, citation consistency, and map cited chunks.

    Guardrails:
    1. If response contains 'INSUFFICIENT_CONTEXT', classify as RefusalResponse.
    2. Extract all chunk citation markers [N].
    3. Verify that cited indices match retrieved chunks (1-indexed).
    4. If no citations found and strict citations are required, include all retrieved chunks as reference.
    """
    raw_answer = answer.strip()

    # 1. Check for refusal signal
    if "INSUFFICIENT_CONTEXT" in raw_answer:
        reason = raw_answer.replace("INSUFFICIENT_CONTEXT:", "").strip()
        if not reason:
            reason = "The provided documents do not contain enough evidence to answer this question."
        logger.info(f"Answer classified as Refusal: {reason}")
        return ValidatedOutput(
            is_refusal=True,
            refusal_reason=reason,
            answer="",
            citations=[],
        )

    # 2. Extract cited chunk numbers
    cited_numbers = extract_cited_indices(raw_answer)
    citations: List[CitationItem] = []
    seen_chunk_ids: Set[str] = set()

    for num in sorted(cited_numbers):
        # 1-indexed to 0-indexed
        idx = num - 1
        if 0 <= idx < len(retrieved_chunks):
            chunk = retrieved_chunks[idx]
            if chunk.chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(chunk.chunk_id)
                citations.append(
                    CitationItem(
                        chunk_id=chunk.chunk_id,
                        doc_id=chunk.doc_id,
                        doc_name=chunk.doc_name,
                        page_number=chunk.page_number,
                        section_heading=chunk.section_heading,
                        text_excerpt=chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
                    )
                )

    # Fallback: if LLM answered without bracket notations, attach the top retrieved chunks as sources
    if not citations and retrieved_chunks:
        for chunk in retrieved_chunks[:2]:
            citations.append(
                CitationItem(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    doc_name=chunk.doc_name,
                    page_number=chunk.page_number,
                    section_heading=chunk.section_heading,
                    text_excerpt=chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
                )
            )

    logger.info(f"Validated answer with {len(citations)} citations attached")
    return ValidatedOutput(
        is_refusal=False,
        refusal_reason=None,
        answer=raw_answer,
        citations=citations,
    )
