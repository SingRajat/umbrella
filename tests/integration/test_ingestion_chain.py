import pytest
from src.chains.ingestion_chain import run_ingestion
from src.common.errors import DuplicateDocumentError, FileTooLargeError, IngestionError
from src.storage.chroma import storage_client


def test_ingestion_txt_success():
    content = (
        b"Umbrella is an open-source research project for studying RAG systems.\n"
        b"It supports incremental evaluation with RAGAS metrics.\n"
        b"ChromaDB is used as the default local vector database."
    )
    result = run_ingestion(
        file_bytes=content,
        filename="umbrella_intro.txt",
        extension="txt",
    )
    assert result["status"] == "ingested"
    assert result["chunk_count"] >= 1
    assert result["filename"] == "umbrella_intro.txt"
    assert result["source_type"] == "txt"

    # Verify retrieval from storage directly
    doc = storage_client.get_document(result["doc_id"])
    assert doc is not None
    assert doc["filename"] == "umbrella_intro.txt"

    # Test idempotency (duplicate upload)
    with pytest.raises(DuplicateDocumentError):
        run_ingestion(
            file_bytes=content,
            filename="umbrella_intro_duplicate.txt",
            extension="txt",
        )

    # Clean up test document
    storage_client.delete_by_doc_id(result["doc_id"])


def test_ingestion_md_success():
    content = (
        b"# Markdown Test\n\n"
        b"This is a markdown document testing header extraction.\n\n"
        b"## Features\n- Fast\n- Reliable"
    )
    result = run_ingestion(
        file_bytes=content,
        filename="test_guide.md",
        extension="md",
    )
    assert result["status"] == "ingested"
    assert result["chunk_count"] >= 1
    storage_client.delete_by_doc_id(result["doc_id"])


def test_ingestion_empty_fails():
    with pytest.raises(IngestionError):
        run_ingestion(
            file_bytes=b"   \n\t  ",
            filename="empty.txt",
            extension="txt",
        )
