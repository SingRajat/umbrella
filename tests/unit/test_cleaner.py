import pytest
from src.ingestion.cleaner import (
    clean_pages,
    clean_text,
    dehyphenate,
    normalize_whitespace,
    remove_control_characters,
    strip_repetitive_headers_footers,
)
from src.ingestion.loaders import RawPage


def test_remove_control_characters():
    dirty = "Hello\x00World\x08\x1f!\nNew line\tTab"
    cleaned = remove_control_characters(dirty)
    assert cleaned == "HelloWorld!\nNew line\tTab"


def test_normalize_whitespace():
    dirty = "Line 1   with   spaces\r\n\r\n\r\n\r\nLine 2"
    cleaned = normalize_whitespace(dirty)
    assert cleaned == "Line 1 with spaces\n\nLine 2"


def test_dehyphenate():
    dirty = "This is an instruc-\ntion on how to clean text."
    cleaned = dehyphenate(dirty)
    assert cleaned == "This is an instruction on how to clean text."


def test_clean_text_full():
    dirty = "   Docu-\nment title\x00   \n\n\n\nSection 1   "
    cleaned = clean_text(dirty)
    assert cleaned == "Document title\n\nSection 1"


def test_strip_repetitive_headers_footers():
    header = "CONFIDENTIAL INTERNAL POLICY REPORT"
    pages = [
        RawPage(page_num=1, text=f"{header}\nContent for page 1\nPage 1 of 4"),
        RawPage(page_num=2, text=f"{header}\nContent for page 2\nPage 2 of 4"),
        RawPage(page_num=3, text=f"{header}\nContent for page 3\nPage 3 of 4"),
        RawPage(page_num=4, text=f"{header}\nContent for page 4\nPage 4 of 4"),
    ]
    cleaned = strip_repetitive_headers_footers(pages)
    assert len(cleaned) == 4
    for p in cleaned:
        assert header not in p.text
        assert "Content for page" in p.text
