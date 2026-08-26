"""Unit tests for configuration, errors, and logging modules."""
from src.config.settings import Settings, get_settings
from src.common.errors import (
    UmbrellaError,
    IngestionError,
    RetrievalError,
    GenerationError,
    ValidationError,
)
from src.common.logging import get_logger


def test_settings_defaults():
    settings = get_settings()
    assert settings.chunk_size == 800
    assert settings.chunk_overlap == 100
    assert settings.top_k == 3
    assert settings.temperature == 0.7
    assert settings.max_upload_size_mb == 25
    assert settings.chroma_persist_dir == "data/chroma_db"


def test_custom_errors():
    err = IngestionError("Corrupted file")
    assert err.status_code == 422
    assert err.error_code == "INGESTION_ERROR"
    assert isinstance(err, UmbrellaError)

    val_err = ValidationError("Bad input")
    assert val_err.status_code == 400
    assert val_err.error_code == "VALIDATION_ERROR"


def test_logger():
    logger = get_logger("test_logger")
    assert logger is not None
    assert logger.name == "test_logger"
