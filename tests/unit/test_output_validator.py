import pytest
from src.query.generator import AnswerWithCitations
from src.query.output_validator import validate_and_construct_response
from src.storage.chroma import RetrievedChunk


def test_output_validator_valid():
    retrieved = [
        RetrievedChunk(
            chunk_id="c1",
            doc_id="d1",
            text="Python was created by Guido van Rossum.",
            metadata={"doc_name": "python.txt", "page_number": 1},
            similarity_score=0.9,
        )
    ]
    answer_obj = AnswerWithCitations(
        answer="Python was created by Guido van Rossum.",
        citations=["c1"],
    )
    result = validate_and_construct_response(
        answer_obj=answer_obj,
        retrieved_chunks=retrieved,
        query="Who created Python?",
        strict_refusal=True,
    )
    assert result.is_valid is True
    assert result.response.status == "answered"
    assert len(result.response.citations) == 1
    assert result.response.citations[0].chunk_id == "c1"


def test_output_validator_hallucinated_citation_strict():
    retrieved = [
        RetrievedChunk(
            chunk_id="c1",
            doc_id="d1",
            text="Some fact.",
            metadata={},
            similarity_score=0.9,
        )
    ]
    answer_obj = AnswerWithCitations(
        answer="Hallucinated claim.",
        citations=["non_existent_chunk_id"],
    )
    result = validate_and_construct_response(
        answer_obj=answer_obj,
        retrieved_chunks=retrieved,
        query="Some question?",
        strict_refusal=True,
    )
    assert result.is_valid is False
    assert result.response.status == "refused"
    assert result.response.reason == "unsupported_claim"
