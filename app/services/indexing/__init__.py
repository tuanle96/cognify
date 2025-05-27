"""
Document indexing services for building searchable knowledge bases.

This module provides a unified interface for indexing documents through
the complete RAG pipeline: Parse → Chunk → Embed → Store. It supports
batch processing, incremental updates, and progress tracking.
"""

from .base import (
    IndexingRequest, IndexingResponse, IndexedDocument, IndexingProgress,
    IndexingStatus, IndexingError, IndexingConfig
)
from .service import IndexingService
from .manager import IndexManager

__all__ = [
    "IndexingRequest",
    "IndexingResponse", 
    "IndexedDocument",
    "IndexingProgress",
    "IndexingStatus",
    "IndexingError",
    "IndexingConfig",
    "IndexingService",
    "IndexManager",
]
