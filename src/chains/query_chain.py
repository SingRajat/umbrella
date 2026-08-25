import json
from typing import Any, AsyncGenerator
from langchain_core.runnables import RunnableLambda

from src.api.schemas import QueryAnswerResponse, QueryRefusalResponse
from src.common.logging import StageTimer, logger
from src.query.context_validator import validate_retrieved_context
from src.query.generator import default_generator
from src.query.output_validator import validate_and_construct_response
from src.query.prompt import build_grounded_prompt
from src.query.retriever import retrieve_context


def _retrieval_step(input_data: dict[str, Any]) -> dict[str, Any]:
    filters = {"doc_id": input_data["doc_id"]} if input_data.get("doc_id") else None
    chunks = retrieve_context(query=input_data["query"], filters=filters)
    input_data["retrieved_chunks"] = chunks
    return input_data


def _context_validation_step(input_data: dict[str, Any]) -> dict[str, Any]:
    with StageTimer("query_context_validation"):
        validation = validate_retrieved_context(input_data["retrieved_chunks"])
    input_data["validation"] = validation
    return input_data


def _generation_or_refusal_step(input_data: dict[str, Any]) -> QueryAnswerResponse | QueryRefusalResponse:
    validation = input_data["validation"]
    query = input_data["query"]

    # Short-circuit if context is insufficient or low relevance
    if not validation.is_valid:
        return QueryRefusalResponse(
            status="refused",
            reason=validation.refusal_reason or "insufficient_context",
            retrieved_chunk_ids=[c.chunk_id for c in input_data["retrieved_chunks"]],
            query=query,
        )

    valid_chunks = validation.filtered_chunks
    system_prompt, user_prompt = build_grounded_prompt(query=query, chunks=valid_chunks)

    # Call LLM Generator
    answer_obj = default_generator.generate(system_prompt=system_prompt, user_prompt=user_prompt)

    # Validate citations & construct response
    with StageTimer("query_output_validation"):
        output_val = validate_and_construct_response(
            answer_obj=answer_obj,
            retrieved_chunks=valid_chunks,
            query=query,
        )

    return output_val.response


# Declarative LCEL Query Chain
query_chain = (
    RunnableLambda(_retrieval_step)
    | RunnableLambda(_context_validation_step)
    | RunnableLambda(_generation_or_refusal_step)
)


def run_query(query: str, doc_id: str | None = None) -> QueryAnswerResponse | QueryRefusalResponse:
    """Executes the LCEL query chain and returns grounded answer or structured refusal."""
    return query_chain.invoke({"query": query, "doc_id": doc_id})


async def run_query_stream(query: str, doc_id: str | None = None) -> AsyncGenerator[str, None]:
    """Streams answer tokens via Server-Sent Events (SSE)."""
    filters = {"doc_id": doc_id} if doc_id else None
    chunks = retrieve_context(query=query, filters=filters)
    validation = validate_retrieved_context(chunks)

    if not validation.is_valid:
        refusal = QueryRefusalResponse(
            status="refused",
            reason=validation.refusal_reason or "insufficient_context",
            retrieved_chunk_ids=[c.chunk_id for c in chunks],
            query=query,
        )
        yield f"data: {json.dumps(refusal.model_dump())}\n\n"
        return

    valid_chunks = validation.filtered_chunks
    system_prompt, user_prompt = build_grounded_prompt(query=query, chunks=valid_chunks)

    for token in default_generator.generate_stream(system_prompt=system_prompt, user_prompt=user_prompt):
        yield f"data: {json.dumps({'token': token})}\n\n"
