import pytest
from src.query.prompt import build_grounded_prompt, format_context_chunk
from src.storage.chroma import RetrievedChunk


def test_format_context_chunk():
    chunk = RetrievedChunk(
        chunk_id="chunk-abc",
        doc_id="doc-123",
        text="The speed limit is 45 mph.",
        metadata={"doc_name": "traffic.pdf", "page_number": 3, "section_heading": "Speed Rules"},
        similarity_score=0.92,
    )
    formatted = format_context_chunk(chunk)
    assert '[chunk_id=chunk-abc | doc="traffic.pdf" | p.3 | section="Speed Rules"]' in formatted
    assert "The speed limit is 45 mph." in formatted


def test_build_grounded_prompt():
    chunks = [
        RetrievedChunk(
            chunk_id="c1",
            doc_id="d1",
            text="Context fact.",
            metadata={"doc_name": "fact.txt"},
            similarity_score=0.8,
        )
    ]
    system_prompt, user_prompt = build_grounded_prompt(query="What is the fact?", chunks=chunks)
    assert "You are Umbrella, an evidence-grounded AI assistant." in system_prompt
    assert "CONTEXT PASSAGES:" in user_prompt
    assert "USER QUESTION:" in user_prompt
    assert "What is the fact?" in user_prompt
