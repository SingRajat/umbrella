from src.storage.chroma import RetrievedChunk

SYSTEM_PROMPT = """You are Umbrella, an evidence-grounded AI assistant.
Your job is to answer the user's question strictly and exclusively based on the provided context passages.

NON-NEGOTIABLE RULES:
1. Answer ONLY using information explicitly stated in the provided context passages. Do NOT extrapolate or bring in external knowledge.
2. If the context does NOT contain enough information to answer the question, state: "I cannot answer this question based on the provided context." and provide an empty citations list.
3. Every factual claim must be backed by one or more citations referencing the exact chunk_id(s) where the fact is stated.
4. Output MUST be formatted as a valid JSON object matching this schema:
{
  "answer": "Your detailed grounded answer here...",
  "citations": ["chunk_id_1", "chunk_id_2"]
}
Do not enclose the JSON in markdown fences (such as ```json) if possible, or ensure it is strictly parseable JSON.
"""


def format_context_chunk(chunk: RetrievedChunk) -> str:
    """Renders a single chunk with full source and citation tags."""
    doc_name = chunk.metadata.get("doc_name", "unknown")
    page = chunk.metadata.get("page_number", "")
    section = chunk.metadata.get("section_heading", "")

    tag_parts = [f"chunk_id={chunk.chunk_id}", f'doc="{doc_name}"']
    if page:
        tag_parts.append(f"p.{page}")
    if section:
        tag_parts.append(f'section="{section}"')

    tag_header = " | ".join(tag_parts)
    return f"[{tag_header}]\n{chunk.text.strip()}"


def build_grounded_prompt(query: str, chunks: list[RetrievedChunk]) -> tuple[str, str]:
    """Pure function returning (system_prompt, user_prompt) with all context rendered."""
    formatted_chunks = "\n\n".join(format_context_chunk(c) for c in chunks)
    user_prompt = f"""CONTEXT PASSAGES:
{formatted_chunks}

USER QUESTION:
{query}

JSON RESPONSE:"""
    return SYSTEM_PROMPT, user_prompt
