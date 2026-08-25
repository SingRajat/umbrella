from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pathlib import Path


class Settings(BaseSettings):
    """Application configuration loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Groq LLM Configuration
    groq_api_key: str = Field(default="mock_key", alias="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.3-70b-versatile", alias="GROQ_MODEL")
    temperature: float = Field(default=0.7, alias="TEMPERATURE")

    # Ingestion & Chunking
    chunk_size: int = Field(default=800, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=100, alias="CHUNK_OVERLAP")

    # Retrieval
    top_k: int = Field(default=3, alias="TOP_K")
    similarity_threshold: float = Field(default=0.5, alias="SIMILARITY_THRESHOLD")

    # Storage
    chroma_persist_dir: str = Field(default="data/chroma_db", alias="CHROMA_PERSIST_DIR")
    chroma_collection_name: str = Field(default="umbrella_docs", alias="CHROMA_COLLECTION_NAME")

    # API & Security
    api_host: str = Field(default="127.0.0.1", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    rate_limit_rpm: int = Field(default=60, alias="RATE_LIMIT_RPM")
    strict_refusal: bool = Field(default=True, alias="STRICT_REFUSAL")

    # Observability
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    @property
    def persist_path(self) -> Path:
        path = Path(self.chroma_persist_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path


settings = Settings()
