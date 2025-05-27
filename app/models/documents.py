"""
Document-related database models.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List

from sqlalchemy import Boolean, DateTime, Enum as SQLEnum, String, Text, Integer, Float, LargeBinary, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import BaseModel, UserTrackingMixin


class DocumentType(str, Enum):
    """Document type enumeration."""
    CODE = "code"
    MARKDOWN = "markdown"
    TEXT = "text"
    PDF = "pdf"
    JSON = "json"
    YAML = "yaml"
    XML = "xml"
    CSV = "csv"
    HTML = "html"
    UNKNOWN = "unknown"


class DocumentStatus(str, Enum):
    """Document processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    ARCHIVED = "archived"


class ProcessingStage(str, Enum):
    """Document processing stage."""
    UPLOADED = "uploaded"
    PARSING = "parsing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    COMPLETED = "completed"
    FAILED = "failed"


class Document(BaseModel, UserTrackingMixin):
    """
    Document model for storing uploaded documents and their metadata.
    """

    __tablename__ = "documents"

    # Basic document information
    filename: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True
    )

    original_filename: Mapped[str] = mapped_column(
        String(255),
        nullable=False
    )

    title: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
        index=True
    )

    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True
    )

    # Document type and format
    document_type: Mapped[DocumentType] = mapped_column(
        SQLEnum(DocumentType),
        nullable=False,
        index=True
    )

    mime_type: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True
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

    # File information
    file_size: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        index=True
    )

    file_hash: Mapped[str] = mapped_column(
        String(64),  # SHA-256 hash
        nullable=False,
        index=True  # Removed unique=True for deduplication
    )

    file_path: Mapped[str | None] = mapped_column(
        String(1000),
        nullable=True
    )

    # Content (can be stored directly or referenced)
    content: Mapped[str | None] = mapped_column(
        Text,
        nullable=True
    )

    content_preview: Mapped[str | None] = mapped_column(
        String(1000),
        nullable=True
    )

    # Shared content reference for deduplication
    shared_content_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("shared_contents.id", ondelete="SET NULL"),
        nullable=True,
        index=True
    )

    # Processing status
    status: Mapped[DocumentStatus] = mapped_column(
        SQLEnum(DocumentStatus),
        nullable=False,
        default=DocumentStatus.PENDING,
        index=True
    )

    processing_stage: Mapped[ProcessingStage] = mapped_column(
        SQLEnum(ProcessingStage),
        nullable=False,
        default=ProcessingStage.UPLOADED,
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

    processing_errors: Mapped[List[str] | None] = mapped_column(
        ARRAY(Text),
        nullable=True
    )

    # Collection association
    collection_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False),
        nullable=True,
        index=True
    )

    # Batch processing
    batch_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False),
        nullable=True,
        index=True
    )

    # Statistics
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

    query_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0
    )

    # Quality metrics
    parsing_quality_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True
    )

    chunking_quality_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True
    )

    # Processing configuration
    processing_config: Mapped[Dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True
    )

    # Relationships
    metadata_record: Mapped["DocumentMetadata"] = relationship(
        "DocumentMetadata",
        back_populates="document",
        uselist=False,
        cascade="all, delete-orphan"
    )

    shared_content: Mapped["SharedContent"] = relationship(
        "SharedContent",
        foreign_keys=[shared_content_id]
    )

    chunks: Mapped[List["DocumentChunk"]] = relationship(
        "DocumentChunk",
        back_populates="document",
        cascade="all, delete-orphan"
    )

    @property
    def is_processed(self) -> bool:
        """Check if document is fully processed."""
        return self.status == DocumentStatus.COMPLETED

    @property
    def processing_time(self) -> float | None:
        """Calculate processing time in seconds."""
        if self.processing_started_at and self.processing_completed_at:
            delta = self.processing_completed_at - self.processing_started_at
            return delta.total_seconds()
        return None

    def start_processing(self, stage: ProcessingStage = ProcessingStage.PARSING) -> None:
        """Mark document processing as started."""
        self.status = DocumentStatus.PROCESSING
        self.processing_stage = stage
        self.processing_started_at = datetime.utcnow()

    def complete_processing(self) -> None:
        """Mark document processing as completed."""
        self.status = DocumentStatus.COMPLETED
        self.processing_stage = ProcessingStage.COMPLETED
        self.processing_completed_at = datetime.utcnow()

    def fail_processing(self, error: str, stage: ProcessingStage = None) -> None:
        """Mark document processing as failed."""
        self.status = DocumentStatus.FAILED
        if stage:
            self.processing_stage = stage

        if self.processing_errors is None:
            self.processing_errors = []
        self.processing_errors.append(f"{datetime.utcnow().isoformat()}: {error}")

    def get_content(self) -> str:
        """Get document content from shared content or direct storage."""
        if self.shared_content_id and self.shared_content:
            return self.shared_content.content
        return self.content or ""

    def get_chunks(self):
        """Get document chunks from shared content or direct storage."""
        if self.shared_content_id and self.shared_content:
            return self.shared_content.chunks
        return self.chunks


class DocumentMetadata(BaseModel):
    """
    Extended metadata for documents.
    """

    __tablename__ = "document_metadata"

    # Foreign key to document
    document_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("documents.id"),
        nullable=False,
        unique=True,
        index=True
    )

    # Extracted metadata
    author: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True
    )

    title: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True
    )

    subject: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True
    )

    keywords: Mapped[List[str] | None] = mapped_column(
        ARRAY(String),
        nullable=True
    )

    creation_date: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )

    modification_date: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )

    # Content analysis
    word_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True
    )

    character_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True
    )

    line_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True
    )

    # Code-specific metadata
    programming_language: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True
    )

    functions_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True
    )

    classes_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True
    )

    imports_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True
    )

    # Document structure
    sections: Mapped[List[Dict[str, Any]] | None] = mapped_column(
        JSONB,
        nullable=True
    )

    outline: Mapped[Dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True
    )

    # Additional metadata
    extracted_entities: Mapped[List[Dict[str, Any]] | None] = mapped_column(
        JSONB,
        nullable=True
    )

    tags: Mapped[List[str] | None] = mapped_column(
        ARRAY(String),
        nullable=True
    )

    # Relationships
    document: Mapped["Document"] = relationship(
        "Document",
        back_populates="metadata_record"
    )


class DocumentChunk(BaseModel, UserTrackingMixin):
    """
    Document chunks for vector storage and retrieval.
    """

    __tablename__ = "document_chunks"

    # Foreign key to document
    document_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("documents.id"),
        nullable=False,
        index=True
    )

    # Chunk information
    chunk_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        index=True
    )

    content: Mapped[str] = mapped_column(
        Text,
        nullable=False
    )

    content_hash: Mapped[str] = mapped_column(
        String(64),  # SHA-256 hash
        nullable=False,
        index=True
    )

    # Position in document
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

    # Usage statistics
    retrieval_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0
    )

    last_retrieved_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )

    # Chunking context
    parent_chunk_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False),
        nullable=True,
        index=True
    )

    child_chunks: Mapped[List[str] | None] = mapped_column(
        ARRAY(UUID(as_uuid=False)),
        nullable=True
    )

    # Relationships
    document: Mapped["Document"] = relationship(
        "Document",
        back_populates="chunks"
    )

    def record_retrieval(self) -> None:
        """Record that this chunk was retrieved."""
        self.retrieval_count += 1
        self.last_retrieved_at = datetime.utcnow()
