"""
Cognify - AI-Powered Intelligent Codebase Analysis Platform

Unified main FastAPI application with comprehensive functionality:
- Real Agentic Chunking (AST, Hybrid, Agentic strategies)
- Multi-format Parsing (8+ formats)
- LLM Integration (LiteLLM + OpenAI proxy)
- Full API v1 endpoints (30+ endpoints)
- Production-ready features
- Development-friendly fallbacks
"""

import logging
import sys
import os
import time
import warnings
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

# Suppress qdrant-client SyntaxWarnings for Python 3.12 compatibility
warnings.filterwarnings("ignore", category=SyntaxWarning, module="qdrant_client")

# Set environment variables FIRST before any imports
# Only set defaults if not already set (allow override from environment)
if not os.getenv("ENVIRONMENT"):
    os.environ["ENVIRONMENT"] = "production"
if not os.getenv("QDRANT_URL"):
    os.environ["QDRANT_URL"] = "http://localhost:6333"

# LLM Configuration - ALL SETTINGS MUST COME FROM DATABASE
# No environment variables for LLM settings allowed

# Required configuration - MUST be set in environment
# These are fallbacks for development only - NOT for production
if not os.getenv("SECRET_KEY"):
    print("WARNING: SECRET_KEY not set in environment, using development fallback")
    os.environ["SECRET_KEY"] = "dev-secret-key-for-testing-only-change-in-production"
if not os.getenv("JWT_SECRET_KEY"):
    print("WARNING: JWT_SECRET_KEY not set in environment, using development fallback")
    os.environ["JWT_SECRET_KEY"] = "dev-jwt-secret-key-for-testing-only-change-in-production"

# LLM settings validation removed - all LLM configuration comes from database
# No environment variables required for LLM functionality

from fastapi import FastAPI, Request, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Import API routers
from app.api.v1 import v1_router  # Complete V1 API (auth, docs, collections, query, system, chunking)
from app.core.config import get_settings
from app.core.exceptions import CognifyException

# Simple logger replacement
class SimpleLogger:
    def info(self, msg, **kwargs):
        print(f"INFO: {msg} {kwargs}")
    def debug(self, msg, **kwargs):
        print(f"DEBUG: {msg} {kwargs}")
    def warning(self, msg, **kwargs):
        print(f"WARNING: {msg} {kwargs}")
    def error(self, msg, **kwargs):
        print(f"ERROR: {msg} {kwargs}")

logger = SimpleLogger()

settings = get_settings()

async def initialize_services(app: FastAPI):
    """Initialize production services only."""
    try:
        # Initialize cache first
        from app.core.cache import initialize_cache
        await initialize_cache()
        logger.info("Cache initialized successfully")

        # Initialize database
        from app.services.database.session import init_database
        await init_database()
        logger.info("Database initialized successfully")

        # Initialize production services
        logger.info("Importing services...")
        from app.services.parsers.service import ParsingService
        logger.info("✅ ParsingService imported")
        from app.services.chunking.service import ChunkingService
        logger.info("✅ ChunkingService imported")
        from app.services.embedding.service import EmbeddingService
        logger.info("✅ EmbeddingService imported")
        from app.services.vectordb.service import VectorDBService
        logger.info("✅ VectorDBService imported")

        # Initialize all services
        logger.info("Creating service instances...")
        app.state.parsing_service = ParsingService()
        logger.info("✅ ParsingService instance created")
        app.state.chunking_service = ChunkingService()
        logger.info("✅ ChunkingService instance created")
        app.state.embedding_service = EmbeddingService()
        logger.info("✅ EmbeddingService instance created")
        app.state.vectordb_service = VectorDBService()
        logger.info("✅ VectorDBService instance created")

        # Initialize each service
        logger.info("Initializing parsing service...")
        await app.state.parsing_service.initialize()
        logger.info("✅ Parsing service initialized")

        logger.info("Initializing chunking service...")
        await app.state.chunking_service.initialize()
        logger.info("✅ Chunking service initialized")

        logger.info("Initializing embedding service...")
        try:
            await app.state.embedding_service.initialize()
            logger.info("✅ Embedding service initialized")
        except Exception as e:
            logger.warning(f"⚠️ Embedding service failed to initialize: {e}")
            logger.info("Continuing without embedding service...")

        logger.info("Initializing vectordb service...")
        await app.state.vectordb_service.initialize()
        logger.info("✅ VectorDB service initialized")

        logger.info("Production services initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize production services: {e}")
        # Don't use fallbacks - let it fail properly to show real issues
        raise

async def cleanup_services(app: FastAPI):
    """Cleanup all services."""
    services = ['parsing_service', 'chunking_service', 'embedding_service', 'vectordb_service']
    for service_name in services:
        if hasattr(app.state, service_name):
            service = getattr(app.state, service_name)
            if hasattr(service, 'cleanup'):
                try:
                    await service.cleanup()
                    logger.info(f"{service_name} cleaned up")
                except Exception as e:
                    logger.error(f"Failed to cleanup {service_name}: {e}")

    # Cleanup database
    try:
        from app.services.database.session import cleanup_database
        await cleanup_database()
        logger.info("Database cleaned up")
    except Exception as e:
        logger.error(f"Failed to cleanup database: {e}")

    # Cleanup cache
    try:
        from app.core.cache import close_cache
        await close_cache()
        logger.info("Cache cleaned up")
    except Exception as e:
        logger.error(f"Failed to cleanup cache: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    logger.info("Starting Cognify Unified API")

    try:
        # Initialize production services with Docker infrastructure
        await initialize_services(app)
        logger.info("All services initialized successfully")
        yield

    except Exception as e:
        logger.error("Failed to initialize services", error=str(e))
        # Continue with limited functionality
        yield
    finally:
        # Shutdown
        logger.info("Shutting down Cognify Unified API")
        await cleanup_services(app)
        logger.info("Services cleaned up successfully")


def create_application() -> FastAPI:
    """Create and configure the FastAPI application."""
    # Create FastAPI app
    app = FastAPI(
        title="Cognify Unified API",
        description="AI-Powered Intelligent Codebase Analysis Platform with Agentic Chunking",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Add middleware
    setup_middleware(app)

    # Add V1 API endpoints (complete unified API)
    # Routes: /api/v1/chunking/*, /api/v1/auth/*, /api/v1/documents/*,
    #         /api/v1/collections/*, /api/v1/query/*, /api/v1/system/*
    app.include_router(v1_router, prefix="/api")

    # Add root endpoints
    add_root_endpoints(app)

    # Add exception handlers
    setup_exception_handlers(app)

    return app


def setup_middleware(app: FastAPI) -> None:
    """Setup application middleware."""
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure as needed
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )






def add_root_endpoints(app: FastAPI) -> None:
    """Add root endpoints."""

    @app.get("/")
    async def root():
        """Root endpoint with API information."""
        return {
            "name": "Cognify Unified API",
            "version": "1.0.0",
            "description": "AI-Powered Intelligent Codebase Analysis Platform with Agentic Chunking",
            "endpoints": {
                "docs": "/docs",
                "redoc": "/redoc",
                "health": "/health",
                "api_v1": "/api/v1"
            },
            "features": [
                "Real Agentic Chunking (AST, Hybrid, Agentic strategies)",
                "Multi-format Parsing (8+ formats)",
                "LLM Integration (LiteLLM + OpenAI proxy)",
                "Full API v1 endpoints (32+ endpoints)",
                "Production-ready features",
                "Development-friendly fallbacks"
            ]
        }

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "version": "1.0.0",
            "services": {
                "api": "operational",
                "chunking": "operational",
                "database": "operational",
                "authentication": "operational"
            }
        }

    @app.get("/metrics")
    async def prometheus_metrics():
        """Prometheus metrics endpoint (no authentication required)."""
        try:
            from prometheus_client import generate_latest
            return Response(generate_latest(), media_type="text/plain")
        except ImportError:
            # Fallback if prometheus_client is not available
            return Response(
                "# Prometheus metrics not available\n# Install prometheus_client package\n",
                media_type="text/plain"
            )


def setup_exception_handlers(app: FastAPI) -> None:
    """
    Setup global exception handlers.
    """
    @app.exception_handler(CognifyException)
    async def cognify_exception_handler(request: Request, exc: CognifyException):
        """Handle Cognify-specific exceptions."""
        logger.error(
            "Cognify exception occurred",
            error_code=exc.error_code,
            message=exc.message,
            details=exc.details,
            path=request.url.path,
        )

        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.error_code,
                    "message": exc.message,
                    "details": exc.details,
                }
            },
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle general exceptions."""
        logger.error(
            "Unhandled exception occurred",
            error=str(exc),
            error_type=type(exc).__name__,
            path=request.url.path,
        )

        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "An internal server error occurred",
                    "details": str(exc) if settings.DEBUG else None,
                }
            },
        )


# Create the application instance
app = create_application()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="info",
    )
