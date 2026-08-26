"""Structured logging configuration for Umbrella."""
import json
import logging
import sys
from typing import Any, Dict


class JSONFormatter(logging.Formatter):
    """Formats log records as JSON objects for structured observability."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: Dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Include custom attributes if present
        if hasattr(record, "request_id"):
            log_entry["request_id"] = getattr(record, "request_id")
        if hasattr(record, "correlation_id"):
            log_entry["correlation_id"] = getattr(record, "correlation_id")
        if hasattr(record, "retrieval_latency"):
            log_entry["retrieval_latency"] = getattr(record, "retrieval_latency")
        if hasattr(record, "llm_latency"):
            log_entry["llm_latency"] = getattr(record, "llm_latency")
        if hasattr(record, "token_usage"):
            log_entry["token_usage"] = getattr(record, "token_usage")

        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger with JSON formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
