"""Application configuration management using pydantic-settings."""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the Umbrella RAG system."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM Settings
    groq_api_key: str = ""
    groq_model: str = "qwen/qwen3.8-27b"
    temperature: float = 0.7

    # RAG Settings
    chunk_size: int = 800
    chunk_overlap: int = 100
    top_k: int = 3
    similarity_threshold: float = 0.5

    # Storage Settings
    chroma_persist_dir: str = "data/chroma_db"

    # API & Security
    max_upload_size_mb: int = 25
    rate_limit_rpm: int = 60
    strict_refusal: bool = True
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton instance of Settings."""
    return Settings()
