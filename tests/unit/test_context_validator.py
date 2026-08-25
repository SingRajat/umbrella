import pytest
from src.query.context_validator import validate_retrieved_context
from src.storage.chroma import RetrievedChunk


def test_context_validator_empty():
    res = validate_retrieved_context([])
    assert res.is_valid is False
    assert res.refusal_reason == "insufficient_context"


def test_context_validator_threshold_pass_and_drop():
    chunks = [
        RetrievedChunk(chunk_id="c1", doc_id="d1", text="high match", metadata={}, similarity_score=0.85),
        RetrievedChunk(chunk_id="c2", doc_id="d1", text="low match", metadata={}, similarity_score=0.30),
    ]
    res = validate_retrieved_context(chunks, threshold=0.5)
    assert res.is_valid is True
    assert len(res.filtered_chunks) == 1
    assert res.filtered_chunks[0].chunk_id == "c1"


def test_context_validator_all_dropped():
    chunks = [
        RetrievedChunk(chunk_id="c1", doc_id="d1", text="low match", metadata={}, similarity_score=0.20),
    ]
    res = validate_retrieved_context(chunks, threshold=0.5)
    assert res.is_valid is False
    assert res.refusal_reason == "low_relevance"
