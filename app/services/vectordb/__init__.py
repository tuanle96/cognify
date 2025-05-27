"""
Vector database services for storing and searching embeddings.

This module provides a unified interface for multiple vector database providers
including Qdrant, Milvus, and others. It supports vector similarity search,
metadata filtering, and collection management.
"""

from .base import (
    VectorDBClient, VectorPoint, SearchResult, SearchRequest,
    VectorDBProvider, CollectionInfo, VectorDBError
)
from .factory import VectorDBFactory
from .service import VectorDBService

__all__ = [
    "VectorDBClient",
    "VectorPoint", 
    "SearchResult",
    "SearchRequest",
    "VectorDBProvider",
    "CollectionInfo",
    "VectorDBError",
    "VectorDBFactory",
    "VectorDBService",
]
