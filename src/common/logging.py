import json
import logging
import sys
import time
from contextvars import ContextVar
from typing import Any

# Context variable for correlation ID tracking across async calls
correlation_id_ctx: ContextVar[str] = ContextVar("correlation_id", default="system")


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": correlation_id_ctx.get(),
        }
        if hasattr(record, "duration_ms"):
            log_obj["duration_ms"] = getattr(record, "duration_ms")
        if hasattr(record, "extra_data"):
            log_obj["extra"] = getattr(record, "extra_data")
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)


def setup_logger(name: str = "umbrella", level: str = "INFO") -> logging.Logger:
    """Configures structured JSON logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level.upper())
    logger.propagate = False

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)

    return logger


logger = setup_logger()


class StageTimer:
    """Context manager to measure and log execution duration for pipeline stages."""

    def __init__(self, stage_name: str, extra: dict[str, Any] | None = None):
        self.stage_name = stage_name
        self.extra = extra or {}
        self.start_time = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        logger.info(f"Starting stage: {self.stage_name}", extra={"extra_data": self.extra})
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = round((time.perf_counter() - self.start_time) * 1000, 2)
        if exc_type:
            logger.error(
                f"Stage {self.stage_name} failed after {duration_ms}ms",
                extra={"duration_ms": duration_ms, "extra_data": self.extra},
            )
        else:
            logger.info(
                f"Stage {self.stage_name} completed in {duration_ms}ms",
                extra={"duration_ms": duration_ms, "extra_data": self.extra},
            )
