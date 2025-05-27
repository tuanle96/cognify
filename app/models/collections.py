"""
Collection-related database models.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List

from sqlalchemy import Boolean, DateTime, Enum as SQLEnum, String, Text, Integer, Float, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import BaseModel, UserTrackingMixin


class CollectionStatus(str, Enum):
    """Collection status enumeration."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    CREATING = "creating"
    DELETING = "deleting"
    ERROR = "error"


class CollectionVisibility(str, Enum):
    """Collection visibility enumeration."""
    PRIVATE = "private"
    SHARED = "shared"
    PUBLIC = "public"


class MemberRole(str, Enum):
    """Collection member role enumeration."""
    OWNER = "owner"
    ADMIN = "admin"
    EDITOR = "editor"
    VIEWER = "viewer"


class Collection(BaseModel, UserTrackingMixin):
    """
    Collection model for organizing documents and vectors.
    """

    __tablename__ = "collections"

    # Basic information
    name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True
    )

    display_name: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True
    )

    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True
    )

    # Collection configuration
    status: Mapped[CollectionStatus] = mapped_column(
        SQLEnum(CollectionStatus),
        nullable=False,
        default=CollectionStatus.CREATING,
        index=True
    )

    visibility: Mapped[CollectionVisibility] = mapped_column(
        SQLEnum(CollectionVisibility),
        nullable=False,
        default=CollectionVisibility.PRIVATE,
        index=True
    )

    # Vector configuration
    embedding_dimension: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=384
    )

    distance_metric: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="cosine"
    )

    # Content statistics
    document_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0
    )

    chunk_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0
    )

    vector_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0
    )

    total_size: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0
    )

    # Usage statistics
    query_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0
    )

    last_queried_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )

    # Configuration
    settings: Mapped[Dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
        default=dict
    )

    metadata_schema: Mapped[Dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True
    )

    # Indexing configuration
    indexing_config: Mapped[Dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True
    )

    # Performance metrics
    average_query_time: Mapped[float | None] = mapped_column(
        Float,
        nullable=True
    )

    index_quality_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True
    )

    # Relationships
    members: Mapped[List["CollectionMember"]] = relationship(
        "CollectionMember",
        back_populates="collection",
        cascade="all, delete-orphan"
    )

    stats: Mapped["CollectionStats"] = relationship(
        "CollectionStats",
        back_populates="collection",
        uselist=False,
        cascade="all, delete-orphan"
    )

    @property
    def is_active(self) -> bool:
        """Check if collection is active."""
        return self.status == CollectionStatus.ACTIVE

    @property
    def is_public(self) -> bool:
        """Check if collection is public."""
        return self.visibility == CollectionVisibility.PUBLIC

    def increment_query_count(self) -> None:
        """Increment query count and update last queried time."""
        self.query_count += 1
        self.last_queried_at = datetime.utcnow()

    def update_content_stats(self, documents: int = 0, chunks: int = 0, vectors: int = 0, size: int = 0) -> None:
        """Update content statistics."""
        self.document_count += documents
        self.chunk_count += chunks
        self.vector_count += vectors
        self.total_size += size


class CollectionMember(BaseModel):
    """
    Collection membership model for access control.
    """

    __tablename__ = "collection_members"

    # Foreign keys
    collection_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("collections.id"),
        nullable=False,
        index=True
    )

    user_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("users.id"),
        nullable=False,
        index=True
    )

    # Member information
    role: Mapped[MemberRole] = mapped_column(
        SQLEnum(MemberRole),
        nullable=False,
        default=MemberRole.VIEWER,
        index=True
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        index=True
    )

    # Invitation information
    invited_by: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False),
        nullable=True
    )

    invited_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )

    joined_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )

    # Permissions
    can_read: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True
    )

    can_write: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False
    )

    can_admin: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False
    )

    # Usage tracking
    last_access_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )

    access_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0
    )

    # Relationships
    collection: Mapped["Collection"] = relationship(
        "Collection",
        back_populates="members"
    )

    @property
    def is_owner(self) -> bool:
        """Check if member is owner."""
        return self.role == MemberRole.OWNER

    @property
    def is_admin(self) -> bool:
        """Check if member is admin or owner."""
        return self.role in (MemberRole.OWNER, MemberRole.ADMIN)

    def record_access(self) -> None:
        """Record member access."""
        self.last_access_at = datetime.utcnow()
        self.access_count += 1

    def update_permissions(self) -> None:
        """Update permissions based on role."""
        if self.role == MemberRole.OWNER:
            self.can_read = True
            self.can_write = True
            self.can_admin = True
        elif self.role == MemberRole.ADMIN:
            self.can_read = True
            self.can_write = True
            self.can_admin = True
        elif self.role == MemberRole.EDITOR:
            self.can_read = True
            self.can_write = True
            self.can_admin = False
        elif self.role == MemberRole.VIEWER:
            self.can_read = True
            self.can_write = False
            self.can_admin = False


class CollectionStats(BaseModel):
    """
    Collection statistics and analytics.
    """

    __tablename__ = "collection_stats"

    # Foreign key
    collection_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("collections.id"),
        nullable=False,
        unique=True,
        index=True
    )

    # Document statistics
    total_documents: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0
    )

    documents_by_type: Mapped[Dict[str, int] | None] = mapped_column(
        JSONB,
        nullable=True,
        default=dict
    )

    documents_by_language: Mapped[Dict[str, int] | None] = mapped_column(
        JSONB,
        nullable=True,
        default=dict
    )

    # Chunk statistics
    total_chunks: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0
    )

    average_chunk_size: Mapped[float | None] = mapped_column(
        Float,
        nullable=True
    )

    chunk_quality_distribution: Mapped[Dict[str, int] | None] = mapped_column(
        JSONB,
        nullable=True
    )

    # Vector statistics
    total_vectors: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0
    )

    vector_dimension: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True
    )

    index_size: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True
    )

    # Query statistics
    total_queries: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0
    )

    successful_queries: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0
    )

    average_query_time: Mapped[float | None] = mapped_column(
        Float,
        nullable=True
    )

    query_types_distribution: Mapped[Dict[str, int] | None] = mapped_column(
        JSONB,
        nullable=True,
        default=dict
    )

    # Performance metrics
    indexing_time: Mapped[float | None] = mapped_column(
        Float,
        nullable=True
    )

    last_indexing_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )

    search_performance: Mapped[Dict[str, float] | None] = mapped_column(
        JSONB,
        nullable=True
    )

    # Usage patterns
    daily_query_counts: Mapped[Dict[str, int] | None] = mapped_column(
        JSONB,
        nullable=True,
        default=dict
    )

    popular_queries: Mapped[List[Dict[str, Any]] | None] = mapped_column(
        JSONB,
        nullable=True,
        default=list
    )

    # Storage statistics
    total_storage_size: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0
    )

    storage_by_type: Mapped[Dict[str, int] | None] = mapped_column(
        JSONB,
        nullable=True,
        default=dict
    )

    # Last update tracking
    last_stats_update: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow
    )

    # Relationships
    collection: Mapped["Collection"] = relationship(
        "Collection",
        back_populates="stats"
    )

    def update_document_stats(self, document_type: str, language: str = None) -> None:
        """Update document statistics."""
        self.total_documents += 1

        # Update by type
        if self.documents_by_type is None:
            self.documents_by_type = {}
        self.documents_by_type[document_type] = self.documents_by_type.get(document_type, 0) + 1

        # Update by language
        if language:
            if self.documents_by_language is None:
                self.documents_by_language = {}
            self.documents_by_language[language] = self.documents_by_language.get(language, 0) + 1

        self.last_stats_update = datetime.utcnow()

    def update_query_stats(self, query_type: str, response_time: float, success: bool = True) -> None:
        """Update query statistics."""
        self.total_queries += 1
        if success:
            self.successful_queries += 1

        # Update average query time
        if self.average_query_time is None:
            self.average_query_time = response_time
        else:
            # Running average
            self.average_query_time = (self.average_query_time * (self.total_queries - 1) + response_time) / self.total_queries

        # Update query types distribution
        if self.query_types_distribution is None:
            self.query_types_distribution = {}
        self.query_types_distribution[query_type] = self.query_types_distribution.get(query_type, 0) + 1

        self.last_stats_update = datetime.utcnow()
