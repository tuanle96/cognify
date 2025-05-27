"""
Database models for Cognify RAG system.

This package contains all SQLAlchemy models for persistent data storage.
"""

from .users import User, UserProfile, UserSession
from .documents import Document, DocumentMetadata, DocumentChunk
from .shared_content import SharedContent, SharedContentChunk
from .collections import Collection, CollectionMember, CollectionStats
from .queries import Query, QueryResult, QueryFeedback
from .analytics import UserActivity, SystemMetrics, PerformanceLog
from .base import Base

__all__ = [
    # Base
    "Base",

    # User models
    "User",
    "UserProfile",
    "UserSession",

    # Document models
    "Document",
    "DocumentMetadata",
    "DocumentChunk",

    # Shared content models
    "SharedContent",
    "SharedContentChunk",

    # Collection models
    "Collection",
    "CollectionMember",
    "CollectionStats",

    # Query models
    "Query",
    "QueryResult",
    "QueryFeedback",

    # Analytics models
    "UserActivity",
    "SystemMetrics",
    "PerformanceLog",
]
