import time
import uuid
from collections import defaultdict
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from src.common.logging import correlation_id_ctx, logger
from src.config.settings import settings


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Assigns and propagates a unique correlation ID for every request."""

    async def dispatch(self, request: Request, call_next):
        # Extract existing correlation ID or generate a new UUID4
        correlation_id = request.headers.get("X-Correlation-ID") or str(uuid.uuid4())
        token = correlation_id_ctx.set(correlation_id)

        start_time = time.perf_counter()
        try:
            response: Response = await call_next(request)
            response.headers["X-Correlation-ID"] = correlation_id
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.info(
                f"{request.method} {request.url.path} responded {response.status_code} in {duration_ms}ms"
            )
            return response
        finally:
            correlation_id_ctx.reset(token)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Token-bucket rate limiter per client IP address."""

    def __init__(self, app, rpm: int = settings.rate_limit_rpm):
        super().__init__(app)
        self.rpm = rpm
        # ip -> list of timestamps
        self.requests: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health check
        if request.url.path.endswith("/health"):
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        window_start = now - 60.0

        # Filter out requests older than 1 minute
        history = [ts for ts in self.requests[client_ip] if ts > window_start]
        self.requests[client_ip] = history

        if len(history) >= self.rpm:
            correlation_id = correlation_id_ctx.get()
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            return JSONResponse(
                status_code=429,
                content={
                    "error_code": "RATE_LIMIT_EXCEEDED",
                    "message": f"Rate limit of {self.rpm} requests per minute exceeded.",
                    "correlation_id": correlation_id,
                },
                headers={"X-Correlation-ID": correlation_id},
            )

        self.requests[client_ip].append(now)
        return await call_next(request)
