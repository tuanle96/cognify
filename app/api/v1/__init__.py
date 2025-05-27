"""
API v1 package for Cognify RAG system.
"""

from fastapi import APIRouter
from .documents import router as documents_router
from .query import router as query_router
from .collections import router as collections_router
from .auth import router as auth_router
from .system import router as system_router
from .chunking import router as chunking_router

# Create v1 API router
v1_router = APIRouter(prefix="/v1")

# Include all sub-routers
v1_router.include_router(chunking_router, prefix="/chunking", tags=["chunking"])
v1_router.include_router(documents_router, prefix="/documents", tags=["documents"])
v1_router.include_router(query_router, prefix="/query", tags=["query"])
v1_router.include_router(collections_router, prefix="/collections", tags=["collections"])
v1_router.include_router(auth_router, prefix="/auth", tags=["authentication"])
v1_router.include_router(system_router, prefix="/system", tags=["system"])

__all__ = ["v1_router"]