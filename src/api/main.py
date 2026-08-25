from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.middleware import CorrelationIdMiddleware, RateLimitMiddleware
from src.api.routes import router
from src.common.errors import UmbrellaError
from src.common.logging import correlation_id_ctx, logger
from src.config.settings import settings
from src.storage.chroma import storage_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Initializing Umbrella RAG Backend...")
    # Ensure Chroma storage is warm
    try:
        storage_client.is_healthy()
        logger.info(f"Connected to ChromaDB storage at: {settings.chroma_persist_dir}")
    except Exception as e:
        logger.error(f"Failed to connect to storage during startup: {e}")
    yield
    logger.info("Shutting down Umbrella RAG Backend...")


def create_app() -> FastAPI:
    """FastAPI application factory."""
    app = FastAPI(
        title="Umbrella RAG API",
        description="Production-Grade RAG Backend with LCEL Orchestration and Groq Inference",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Custom Middleware
    app.add_middleware(CorrelationIdMiddleware)
    app.add_middleware(RateLimitMiddleware, rpm=settings.rate_limit_rpm)

    # Exception Handlers
    @app.exception_handler(UmbrellaError)
    async def umbrella_exception_handler(request: Request, exc: UmbrellaError):
        correlation_id = correlation_id_ctx.get()
        logger.error(f"UmbrellaError [{exc.error_code}]: {exc.message}")
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error_code": exc.error_code,
                "message": exc.message,
                "correlation_id": correlation_id,
            },
            headers={"X-Correlation-ID": correlation_id},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        correlation_id = correlation_id_ctx.get()
        logger.error(f"Validation Error: {exc.errors()}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error_code": "INVALID_REQUEST_PAYLOAD",
                "message": str(exc.errors()),
                "correlation_id": correlation_id,
            },
            headers={"X-Correlation-ID": correlation_id},
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        correlation_id = correlation_id_ctx.get()
        logger.exception(f"Unhandled Internal Error: {exc}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error_code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected internal server error occurred.",
                "correlation_id": correlation_id,
            },
            headers={"X-Correlation-ID": correlation_id},
        )

    # Register API routes
    app.include_router(router)

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host=settings.api_host, port=settings.api_port, reload=True)
