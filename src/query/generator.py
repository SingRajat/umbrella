"""LLM generation client using LangChain ChatGroq with bounded retry."""
import os
import time
from typing import Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from src.config.settings import get_settings
from src.common.errors import GenerationError
from src.common.logging import get_logger

logger = get_logger("umbrella.query.generator")


def get_chat_llm(
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    groq_api_key: Optional[str] = None,
) -> BaseChatModel:
    """Instantiate LangChain ChatGroq model configured with settings."""
    settings = get_settings()
    api_key = groq_api_key or settings.groq_api_key or os.getenv("GROQ_API_KEY", "") or "gsk_dummy_placeholder_key"

    return ChatGroq(
        model=model or settings.groq_model,
        temperature=temperature if temperature is not None else settings.temperature,
        groq_api_key=api_key,
        max_retries=2,
    )


def generate_with_retry(chain, prompt_input: dict, max_attempts: int = 3) -> str:
    """Execute LCEL generation chain with bounded retries and exponential backoff."""
    backoff = 1.0
    last_err = None

    for attempt in range(1, max_attempts + 1):
        try:
            return chain.invoke(prompt_input)
        except Exception as exc:
            last_err = exc
            logger.warning(f"Groq generation failed (attempt {attempt}/{max_attempts}): {exc}")
            if attempt < max_attempts:
                time.sleep(backoff)
                backoff *= 2

    logger.error(f"LLM generation failed after {max_attempts} attempts: {last_err}")
    raise GenerationError(f"LLM generation failed: {last_err}", error_code="GROQ_API_ERROR", status_code=502) from last_err
