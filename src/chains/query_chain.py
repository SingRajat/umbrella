"""LCEL query and generation pipeline chain with streaming support."""
import json
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

from src.api.schemas import CitationItem, QueryResponse, RefusalResponse
from src.common.logging import get_logger
from src.query.context_validator import validate_retrieved_context
from src.query.generator import generate_with_retry, get_chat_llm
from src.query.output_validator import validate_generated_output
from src.query.prompt import format_context_blocks, get_rag_prompt_template
from src.query.retriever import retrieve_chunks
from src.storage.chroma import ChromaVectorStore, RetrievedChunk, get_vector_store

logger = get_logger("umbrella.chains.query")


def run_query_pipeline(
    query: str,
    doc_id: Optional[str] = None,
    vector_store: Optional[ChromaVectorStore] = None,
    llm: Optional[BaseChatModel] = None,
) -> Union[QueryResponse, RefusalResponse]:
    """
    Execute the end-to-end RAG query pipeline:
    1. Retrieval -> Fetch top-k chunks from ChromaDB.
    2. Context Validation -> Check confidence threshold (0.5).
    3. Prompt Formatting -> Inject numbered context into ChatPromptTemplate.
    4. Generation -> Invoke ChatGroq with retry.
    5. Output Validation -> Extract inline citations and verify facts.
    """
    store = vector_store or get_vector_store()
    chat_llm = llm or get_chat_llm()

    logger.info(f"Initiating RAG query pipeline for query: '{query[:60]}...'")

    # Step 1: Retrieval
    retrieved_chunks = retrieve_chunks(query_text=query, doc_id=doc_id, vector_store=store)

    # Step 2: Context Validation
    val_result = validate_retrieved_context(retrieved_chunks)
    if not val_result.is_valid:
        logger.warning(f"Query refused during context validation: {val_result.reason}")
        return RefusalResponse(
            status="refused",
            reason=val_result.reason or "Retrieved evidence is insufficient.",
            retrieved_chunk_ids=[c.chunk_id for c in retrieved_chunks],
            query=query,
        )

    usable_chunks = val_result.valid_chunks

    # Step 3: Context Formatting & Prompt Construction
    context_text = format_context_blocks(usable_chunks)
    prompt_template = get_rag_prompt_template()

    # Step 4: LCEL Generation Chain
    generation_chain = prompt_template | chat_llm | StrOutputParser()

    try:
        raw_answer = generate_with_retry(
            generation_chain,
            {"context": context_text, "question": query},
        )
    except Exception as exc:
        logger.error(f"Error during LLM generation: {exc}")
        raise

    # Step 5: Output & Citation Validation
    output_result = validate_generated_output(raw_answer, usable_chunks)

    if output_result.is_refusal:
        return RefusalResponse(
            status="refused",
            reason=output_result.refusal_reason or "Context insufficient.",
            retrieved_chunk_ids=[c.chunk_id for c in usable_chunks],
            query=query,
        )

    return QueryResponse(
        status="answered",
        answer=output_result.answer,
        citations=output_result.citations,
    )


def stream_query_pipeline(
    query: str,
    doc_id: Optional[str] = None,
    vector_store: Optional[ChromaVectorStore] = None,
    llm: Optional[BaseChatModel] = None,
) -> Iterator[str]:
    """
    Stream token chunks using Server-Sent Events (SSE) protocol.
    Yields data lines formatted as SSE:
    data: {"type": "token", "content": "..."}\n\n
    data: {"type": "citations", "citations": [...]}\n\n
    data: {"type": "done"}\n\n
    """
    store = vector_store or get_vector_store()
    chat_llm = llm or get_chat_llm()

    retrieved_chunks = retrieve_chunks(query_text=query, doc_id=doc_id, vector_store=store)
    val_result = validate_retrieved_context(retrieved_chunks)

    if not val_result.is_valid:
        refusal_event = {
            "type": "refusal",
            "status": "refused",
            "reason": val_result.reason or "Context insufficient.",
            "retrieved_chunk_ids": [c.chunk_id for c in retrieved_chunks],
        }
        yield f"data: {json.dumps(refusal_event)}\n\n"
        yield "data: {\"type\": \"done\"}\n\n"
        return

    usable_chunks = val_result.valid_chunks
    context_text = format_context_blocks(usable_chunks)
    prompt_template = get_rag_prompt_template()

    generation_chain = prompt_template | chat_llm | StrOutputParser()

    full_text = []
    for token in generation_chain.stream({"context": context_text, "question": query}):
        full_text.append(token)
        token_event = {"type": "token", "content": token}
        yield f"data: {json.dumps(token_event)}\n\n"

    # Validate full output and emit citations
    accumulated_answer = "".join(full_text)
    output_result = validate_generated_output(accumulated_answer, usable_chunks)

    if output_result.is_refusal:
        refusal_event = {
            "type": "refusal",
            "status": "refused",
            "reason": output_result.refusal_reason or "Context insufficient.",
        }
        yield f"data: {json.dumps(refusal_event)}\n\n"
    else:
        citations_payload = [
            c.model_dump() if hasattr(c, "model_dump") else c.__dict__
            for c in output_result.citations
        ]
        citation_event = {"type": "citations", "citations": citations_payload}
        yield f"data: {json.dumps(citation_event)}\n\n"

    yield "data: {\"type\": \"done\"}\n\n"
