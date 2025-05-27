"""
Embedding services for text and code vectorization.

This module provides a unified interface for multiple embedding providers
including OpenAI, Voyage AI, and Cohere. It supports both text and code
embeddings with intelligent caching and batch processing.
"""

from .service import EmbeddingService, embedding_service

__all__ = [
    "EmbeddingService",
    "embedding_service"
]
