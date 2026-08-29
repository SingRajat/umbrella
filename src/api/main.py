"""FastAPI application factory and entry point."""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

from src.api.middleware import RateLimitMiddleware, SecurityHeadersMiddleware
from src.api.routes import router
from src.config.settings import get_settings
from src.common.errors import UmbrellaError
from src.common.logging import get_logger

logger = get_logger("umbrella.api")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    settings = get_settings()

    app = FastAPI(
        title="Umbrella RAG API",
        description="Production-oriented RAG system backend",
        version="0.1.0",
    )

    # Attach security headers middleware
    app.add_middleware(SecurityHeadersMiddleware)

    # Attach rate limiting middleware
    app.add_middleware(RateLimitMiddleware)

    # Allow Streamlit frontend and local clients to make HTTP requests
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Centralized exception handler for custom Umbrella domain errors
    @app.exception_handler(UmbrellaError)
    async def umbrella_error_handler(request: Request, exc: UmbrellaError) -> JSONResponse:
        logger.error(f"Domain error occurred: {exc.message} (code: {exc.error_code})")
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error_code": exc.error_code,
                "message": exc.message,
                "status_code": exc.status_code,
            },
        )

    # Global unhandled exception handler
    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error(f"Unhandled server exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error_code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected internal server error occurred.",
                "status_code": 500,
            },
        )

    # Root route providing system info and redirect link
    @app.get("/", summary="Root status", tags=["System"])
    async def root():
        return {
            "system": "Umbrella RAG API",
            "status": "online",
            "version": "0.1.0",
            "docs": "/docs",
            "health": "/api/v1/health",
        }

    # Register API v1 routes
    app.include_router(router)

    return app


# Default app instance for uvicorn (e.g. uvicorn src.api.main:app)
app = create_app()
