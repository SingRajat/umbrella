"""Conversation history Management, message trimming, and query rewritting."""

from langchain_core.messages.utils import count_tokens_approximately
import logging
from typing import List, Optional, Tuple

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage, BaseMessage, HumanMessage, SystemMessage, trim_messages,
)
from langchain_core.output_parsers import StrOutputParser
from src.query.prompt import rewrite_prompt

logger = logging.getLogger("umbrella.query.history")

MAX_PAYLOAD_HISTORY_TOKENS = 10_000   # Fast DoS guard in API route
MAX_TRIMMED_HISTORY_TOKENS = 1_500    # Active turns budget for answer model prompt
MAX_SUMMARY_TOKENS = 500              # Maximum budget for rolling summary
SUMMARY_THRESHOLD_TOKENS = 1_200      # Trigger threshold to summarize older turns
DEFAULT_RETAINED_TURNS = 4            # Keep latest 2 user-assistant pairs uncompressed

def get_history_trimmer():
    """Return trim_messages runnable using LangChain's built-in approximate counter."""
    return trim_messages(
        max_tokens=MAX_TRIMMED_HISTORY_TOKENS,
        strategy="last",
        token_counter="approximate",
        include_system=False,
        allow_partial=False,
        start_on="human",
    )

def dicts_to_messages(dicts: List[dict]) -> List[BaseMessage]:
    """Convert API transport dicts to LangChain BaseMessage objects."""
    msgs = []
    for d in dicts:
        role = d.get("role")
        content = d.get("content", "")
        if role == "user":
            msgs.append(HumanMessage(content=content))
        elif role == "assistant":
            msgs.append(AIMessage(content=content))
    return msgs

# Alias for compatibility
dict_to_messages = dicts_to_messages


def messages_to_dicts(messages: List[BaseMessage]) -> List[dict]:
    """Convert LangChain BaseMessage objects to plain API dicts."""
    return [
        {
            "role": "user" if isinstance(m, HumanMessage) else "assistant",
            "content": m.content,
        }
        for m in messages
    ]


SUMMARIZE_PROMPT = (
    "Summarize the following conversation concisely for topic continuity. "
    "Focus strictly on topics discussed. Do NOT treat past answers as verified facts. "
    "Keep under 500 tokens."
)


def summarize_history(llm: BaseChatModel, existing_summary: Optional[str], turns: List[BaseMessage]) -> str:
    """Compress older conversation turns into a concise summary using the LLM."""
    try:
        context = f"Previous summary:\n{existing_summary}\n\n" if existing_summary else ""
        conv_text = "\n".join(
            f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
            for m in turns
        )
        prompt = [
            SystemMessage(content=SUMMARIZE_PROMPT),
            HumanMessage(content=f"{context}Conversation:\n{conv_text}"),
        ]
        res = llm.invoke(prompt)
        return res.content.strip() if hasattr(res, "content") else str(res).strip()
    except Exception as exc:
        logger.warning(f"Summarization failed: {exc}. Retaining existing summary.")
        return existing_summary or ""


def process_conversation_history(
    history_messages: Optional[List[BaseMessage]],
    conversation_summary: Optional[str],
    llm: BaseChatModel,
) -> Tuple[List[BaseMessage], Optional[str]]:
    """Universal pre-step: summarize older turns if token budget is exceeded, then trim active turns."""
    if not history_messages:
        return [], conversation_summary

    active = list(history_messages)
    effective_summary = conversation_summary

    if count_tokens_approximately(active) > SUMMARY_THRESHOLD_TOKENS:
        if len(active) > DEFAULT_RETAINED_TURNS:
            older = active[:-DEFAULT_RETAINED_TURNS]
            active = active[-DEFAULT_RETAINED_TURNS:]
            effective_summary = summarize_history(llm, conversation_summary, older)

    trimmed = get_history_trimmer().invoke(active)
    return trimmed, effective_summary


def rewrite_query(
    query: str,
    history: Optional[List[BaseMessage]],
    summary: Optional[str],
    llm: BaseChatModel,
) -> str:
    """Rewrite a follow-up question into a standalone search query using conversation context.
    
    Bypasses the LLM on turn 1 (when no context exists) to eliminate latency.
    Falls back gracefully to the original query on any LLM failure.
    """
    if not history and not summary:
        return query

    try:
        prompt_template = rewrite_prompt(has_summary=bool(summary))
        prompt_input = {
            "question": query,
            "chat_history": history or [],
        }
        if summary:
            prompt_input["summary"] = summary

        rewrite_chain = prompt_template | llm | StrOutputParser()
        rewritten = rewrite_chain.invoke(prompt_input).strip()

        if not rewritten:
            logger.warning("Query rewrite produced an empty string; falling back to original query.")
            return query

        logger.info(f"Query rewritten: '{query}' -> '{rewritten}'")
        return rewritten
    except Exception as exc:
        logger.warning(f"Query rewrite failed: {exc}. Falling back to original query.")
        return query