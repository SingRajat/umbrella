"""FastAPI application factory and entry point."""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

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
            },
        )

    # Register API v1 routes
    app.include_router(router)

    return app


# Default app instance for uvicorn (e.g. uvicorn src.api.main:app)
app = create_app()
