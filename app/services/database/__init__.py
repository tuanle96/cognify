"""
Database service layer for Cognify RAG system.

This package provides high-level database operations and repository patterns
for interacting with the PostgreSQL database.
"""

from .session import DatabaseSession, get_db_session
from .repositories import (
    BaseRepository,
    UserRepository,
    DocumentRepository,
    CollectionRepository,
    QueryRepository
)

__all__ = [
    # Session management
    "DatabaseSession",
    "get_db_session",

    # Repositories
    "BaseRepository",
    "UserRepository",
    "DocumentRepository",
    "CollectionRepository",
    "QueryRepository",
]
