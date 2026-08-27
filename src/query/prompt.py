"""Prompt templates and context formatting for grounded RAG generation."""
from typing import List
from langchain_core.prompts import ChatPromptTemplate

from src.storage.chroma import RetrievedChunk

SYSTEM_PROMPT = """You are Umbrella, an expert AI assistant providing verifiable, citation-backed answers grounded strictly in the retrieved context.

You must strictly adhere to the following rules:
1. Answer the question using ONLY the factual evidence provided in the Context below. Do NOT use outside knowledge or extrapolate assumptions.
2. Every substantive statement or claim you make MUST include an inline bracketed citation pointing to the evidence chunk number(s), e.g. [1], [2], or [1, 2].
3. Multiple sources must be cited appropriately when synthesizing information across chunks, e.g. "Term A applies [1], whereas Policy B governs exceptions [2, 3]."
4. If the provided Context does NOT contain sufficient evidence to answer the question accurately, you MUST reply with:
   "INSUFFICIENT_CONTEXT: The provided documents do not contain enough information to answer this question."
5. Never invent or speculate facts. If only part of the question can be answered, answer that part with citations and explicitly state what is missing.
"""

USER_TEMPLATE = """Context:
{context}

Question: {question}

Answer (with strict inline citations):"""


def format_context_blocks(chunks: List[RetrievedChunk]) -> str:
    """
    Format retrieved chunks into numbered context blocks for prompt injection.
    Example:
    [1] (Document: contract.pdf, Page: 2, Section: Scope)
    The contractor shall deliver...
    """
    formatted_blocks = []
    for idx, chunk in enumerate(chunks, start=1):
        doc_info = f"Document: {chunk.doc_name}"
        page_info = f", Page: {chunk.page_number}" if chunk.page_number is not None else ""
        section_info = f", Section: {chunk.section_heading}" if chunk.section_heading else ""
        header = f"[{idx}] ({doc_info}{page_info}{section_info})"
        formatted_blocks.append(f"{header}\n{chunk.text.strip()}")

    return "\n\n".join(formatted_blocks)


def get_rag_prompt_template() -> ChatPromptTemplate:
    """Return LangChain ChatPromptTemplate configured with system instructions."""
    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", USER_TEMPLATE),
    ])
