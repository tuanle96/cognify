"""
Shared content models for content deduplication.

Allows multiple documents to reference the same content, saving storage space
and processing time while maintaining user isolation.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
from sqlalchemy import String, Text, Integer, Float, DateTime, Index, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.models.base import BaseModel


class SharedContent(BaseModel):
    """
    Shared content storage for deduplication.

    Multiple documents can reference the same shared content,
    saving storage space and processing time.
    """
    __tablename__ = "shared_contents"

    # Content identification
    content_hash: Mapped[str] = mapped_column(
        String(64),  # SHA-256 hash
        nullable=False,
        unique=True,
        index=True
    )

    # Content data
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False
    )

    # Content metadata
    content_size: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        index=True
    )

    content_type: Mapped[str] = mapped_column(
        String(100),
        nullable=True,
        index=True
    )

    language: Mapped[str | None] = mapped_column(
        String(10),
        nullable=True,
        index=True
    )

    encoding: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True
    )

    # Processing status
    is_processed: Mapped[bool] = mapped_column(
        nullable=False,
        default=False,
        index=True
    )

    processing_started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )

    processing_completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )

    # Processing results
    chunk_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0
    )

    embedding_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0
    )

    # Quality metrics
    processing_quality_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True
    )

    chunking_quality_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True
    )

    # Processing configuration used
    processing_config: Mapped[Dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True
    )

    # Usage statistics
    reference_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        index=True
    )

    last_accessed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )

    # Relationships
    chunks: Mapped[List["SharedContentChunk"]] = relationship(
        "SharedContentChunk",
        back_populates="shared_content",
        cascade="all, delete-orphan",
        order_by="SharedContentChunk.chunk_index"
    )

    # Indexes for performance
    __table_args__ = (
        Index('idx_shared_content_hash_processed', 'content_hash', 'is_processed'),
        Index('idx_shared_content_type_language', 'content_type', 'language'),
        Index('idx_shared_content_size_refs', 'content_size', 'reference_count'),
    )


class SharedContentChunk(BaseModel):
    """
    Chunks for shared content.

    Pre-processed chunks that can be reused across multiple documents.
    """
    __tablename__ = "shared_content_chunks"

    # Reference to shared content
    shared_content_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("shared_contents.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Chunk identification
    chunk_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        index=True
    )

    content_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True
    )

    # Chunk content
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False
    )

    # Position information
    start_position: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True
    )

    end_position: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True
    )

    # Chunk metadata
    chunk_type: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
        index=True
    )

    language: Mapped[str | None] = mapped_column(
        String(10),
        nullable=True
    )

    # Quality metrics
    quality_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True
    )

    coherence_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True
    )

    completeness_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True
    )

    # Vector information
    vector_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        unique=True,
        index=True
    )

    embedding_model: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True
    )

    embedding_dimension: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True
    )

    # Chunk metadata
    chunk_metadata: Mapped[Dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True
    )

    # Relationships
    shared_content: Mapped["SharedContent"] = relationship(
        "SharedContent",
        back_populates="chunks"
    )

    # Indexes for performance
    __table_args__ = (
        Index('idx_shared_chunk_content_index', 'shared_content_id', 'chunk_index'),
        Index('idx_shared_chunk_vector', 'vector_id', 'embedding_model'),
        Index('idx_shared_chunk_type_quality', 'chunk_type', 'quality_score'),
    )
