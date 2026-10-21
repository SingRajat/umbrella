"""API Middleware for Rate Limiting, Security Headers, and Request Auditing."""
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from fastapi import Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from src.config.settings import get_settings
from src.common.logging import get_logger

logger = get_logger("umbrella.api.middleware")


class RateLimiter:
    """Sliding-window in-memory rate limiter per client IP."""

    def __init__(self, requests_per_minute: Optional[int] = None):
        if requests_per_minute is None:
            requests_per_minute = get_settings().rate_limit_rpm
        self.requests_per_minute = requests_per_minute
        self.window_seconds = 60
        self._clients: Dict[str, List[float]] = defaultdict(list)
        self._last_cleanup = time.time()

    def _cleanup_stale_clients(self, window_start: float):
        """Remove inactive client buckets to prevent memory leaks."""
        stale_ips = [ip for ip, timestamps in self._clients.items() if not timestamps or timestamps[-1] <= window_start]
        for ip in stale_ips:
            del self._clients[ip]

    def is_allowed(self, client_ip: str) -> Tuple[bool, int, int]:
        """
        Check if client is within rate limits.
        Returns (is_allowed, remaining_requests, retry_after_seconds).
        """
        now = time.time()
        window_start = now - self.window_seconds

        # Periodically evict inactive client IPs every 5 minutes
        if now - self._last_cleanup > 300:
            self._cleanup_stale_clients(window_start)
            self._last_cleanup = now

        # Prune timestamps outside the current window for this client
        timestamps = self._clients[client_ip]
        active = [ts for ts in timestamps if ts > window_start]
        self._clients[client_ip] = active

        current_count = len(active)
        remaining = max(0, self.requests_per_minute - current_count)

        if current_count >= self.requests_per_minute:
            oldest = active[0]
            retry_after = int(self.window_seconds - (now - oldest)) + 1
            return False, 0, max(1, retry_after)

        self._clients[client_ip].append(now)
        return True, remaining - 1, 0


_global_rate_limiter: RateLimiter = RateLimiter()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Enforces per-client IP rate limits and attaches rate limit headers."""

    async def dispatch(self, request: Request, call_next):
        # Exclude health check and docs from strict rate limiting
        if request.url.path in ("/api/v1/health", "/docs", "/openapi.json", "/redoc"):
            return await call_next(request)

        client_ip = request.client.host if request.client else "127.0.0.1"
        # Only trust X-Forwarded-For if explicitly configured behind a reverse proxy
        if get_settings().trust_proxy:
            forwarded = request.headers.get("x-forwarded-for")
            if forwarded:
                client_ip = forwarded.split(",")[0].strip()

        limit = _global_rate_limiter.requests_per_minute
        is_allowed, remaining, retry_after = _global_rate_limiter.is_allowed(client_ip)

        if not is_allowed:
            logger.warning(f"Rate limit exceeded for client IP: {client_ip}")
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error_code": "RATE_LIMIT_EXCEEDED",
                    "message": f"Rate limit of {limit} requests/minute exceeded. Try again in {retry_after} seconds.",
                    "status_code": 429,
                },
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0",
                },
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Attaches standard security headers to all HTTP responses."""

    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response
