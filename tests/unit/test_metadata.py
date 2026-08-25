import pytest
from src.ingestion.chunker import Chunk
from src.ingestion.loaders import RawDocument, RawPage
from src.ingestion.metadata import construct_chunk_records, extract_section_heading


def test_extract_section_heading():
    md_heading = "## Architecture Overview\nSome body text follows."
    assert extract_section_heading(md_heading) == "Architecture Overview"

    caps_heading = "INTRODUCTION\nBody text"
    assert extract_section_heading(caps_heading) == "INTRODUCTION"

    regular_text = "Just a regular sentence without heading."
    assert extract_section_heading(regular_text) is None


def test_construct_chunk_records():
    raw_doc = RawDocument(
        doc_id="doc-123",
        filename="test.pdf",
        source_type="pdf",
        uploaded_at="2026-08-26T00:00:00Z",
        pages=[RawPage(page_num=1, text="Sample text")],
        content_hash="abcde12345",
    )
    chunks = [
        Chunk(
            chunk_id="chunk-1",
            text="## Header\nChunk content",
            page_number=1,
            char_start=0,
            char_end=22,
            chunk_index=0,
        )
    ]
    records = construct_chunk_records(raw_doc, chunks)
    assert len(records) == 1
    rec = records[0]
    assert rec.chunk_id == "chunk-1"
    assert rec.doc_id == "doc-123"
    assert rec.metadata["doc_name"] == "test.pdf"
    assert rec.metadata["section_heading"] == "Header"
    assert rec.metadata["page_number"] == 1
    assert rec.metadata["content_hash"] == "abcde12345"
