import pytest
from src.ingestion.chunker import RecursiveChunker
from src.ingestion.loaders import RawPage


def test_recursive_chunker_basic():
    chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)
    text = (
        "Paragraph 1 is here. It has some text to make it long enough.\n\n"
        "Paragraph 2 is here. It also contains important information for chunking testing.\n\n"
        "Paragraph 3 is also present to verify multi-chunk splitting behavior."
    )
    pages = [RawPage(page_num=1, text=text)]
    chunks = chunker.chunk(pages)

    assert len(chunks) >= 2
    assert all(c.chunk_id for c in chunks)
    assert all(c.page_number == 1 for c in chunks)
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1


def test_recursive_chunker_multi_page():
    chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)
    pages = [
        RawPage(page_num=1, text="Page one content with sufficient length."),
        RawPage(page_num=2, text="Page two content with additional text."),
    ]
    chunks = chunker.chunk(pages)
    assert len(chunks) == 2
    assert chunks[0].page_number == 1
    assert chunks[1].page_number == 2
