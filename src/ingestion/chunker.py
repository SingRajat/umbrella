import uuid
from dataclasses import dataclass
from typing import Protocol
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config.settings import settings
from src.ingestion.loaders import RawPage


@dataclass
class Chunk:
    """Represents a text chunk produced from document pages."""
    chunk_id: str
    text: str
    page_number: int | list[int] | None
    char_start: int
    char_end: int
    chunk_index: int


class Chunker(Protocol):
    """Protocol for pluggable document chunking implementations."""

    def chunk(self, pages: list[RawPage]) -> list[Chunk]:
        """Chunk a list of cleaned pages into Chunk objects."""
        ...


class RecursiveChunker:
    """Chunks document pages using LangChain's RecursiveCharacterTextSplitter."""

    def __init__(
        self,
        chunk_size: int = settings.chunk_size,
        chunk_overlap: int = settings.chunk_overlap,
        separators: list[str] | None = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
        )

    def chunk(self, pages: list[RawPage]) -> list[Chunk]:
        chunks: list[Chunk] = []
        global_char_offset = 0
        chunk_index = 0

        for page in pages:
            page_text = page.text.strip()
            if not page_text:
                continue

            splits = self.splitter.split_text(page_text)
            for split_text in splits:
                split_len = len(split_text)
                chunk_id = str(uuid.uuid4())
                char_start = global_char_offset
                char_end = global_char_offset + split_len

                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        text=split_text,
                        page_number=page.page_num,
                        char_start=char_start,
                        char_end=char_end,
                        chunk_index=chunk_index,
                    )
                )
                chunk_index += 1
                global_char_offset += split_len + 1  # include separator offset

        return chunks


default_chunker = RecursiveChunker()
