"""Document ingestion, loading, cleaning, chunking, and metadata tagging package."""
from src.ingestion.loaders import RawDocument, RawPage, load_pdf, load_docx, load_txt
from src.ingestion.cleaner import clean_text, clean_document, dehyphenate, normalize_whitespace, remove_control_characters
from src.ingestion.metadata import ChunkRecord, extract_section_heading
from src.ingestion.chunker import chunk_document, get_text_splitter

__all__ = [
    "RawDocument",
    "RawPage",
    "load_pdf",
    "load_docx",
    "load_txt",
    "clean_text",
    "clean_document",
    "dehyphenate",
    "normalize_whitespace",
    "remove_control_characters",
    "ChunkRecord",
    "extract_section_heading",
    "chunk_document",
    "get_text_splitter",
]
