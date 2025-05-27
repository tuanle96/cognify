"""
Query-related database models.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List

from sqlalchemy import Boolean, DateTime, Enum as SQLEnum, String, Text, Integer, Float, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import BaseModel, UserTrackingMixin


class QueryType(str, Enum):
    """Query type enumeration."""
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"
    QUESTION = "question"
    CODE = "code"
    SIMILARITY = "similarity"


class QueryStatus(str, Enum):
    """Query status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FeedbackType(str, Enum):
    """Feedback type enumeration."""
    RELEVANCE = "relevance"
    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    USEFULNESS = "usefulness"
    GENERAL = "general"


class Query(BaseModel, UserTrackingMixin):
    """
    Query model for storing user queries and their metadata.
    """

    __tablename__ = "queries"

    # Query content
    query_text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        index=True
    )

    query_hash: Mapped[str] = mapped_column(
        String(64),  # SHA-256 hash
        nullable=False,
        index=True
    )

    # Query configuration
    query_type: Mapped[QueryType] = mapped_column(
        SQLEnum(QueryType),
        nullable=False,
        default=QueryType.SEMANTIC,
        index=True
    )

    collection_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False),
        nullable=True,
        index=True
    )

    # Processing status
    status: Mapped[QueryStatus] = mapped_column(
        SQLEnum(QueryStatus),
        nullable=False,
        default=QueryStatus.PENDING,
        index=True
    )

    # Query parameters
    max_results: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=10
    )

    min_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0
    )

    rerank_results: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True
    )

    include_metadata: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True
    )

    # Query processing
    processing_started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )

    processing_completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )

    processing_time: Mapped[float | None] = mapped_column(
        Float,
        nullable=True
    )

    # Results
    result_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0
    )

    has_results: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True
    )

    # Query expansion and processing
    expanded_query: Mapped[str | None] = mapped_column(
        Text,
        nullable=True
    )

    query_intent: Mapped[Dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True
    )

    extracted_keywords: Mapped[List[str] | None] = mapped_column(
        ARRAY(String),
        nullable=True
    )

    # Performance metrics
    embedding_time: Mapped[float | None] = mapped_column(
        Float,
        nullable=True
    )

    search_time: Mapped[float | None] = mapped_column(
        Float,
        nullable=True
    )

    rerank_time: Mapped[float | None] = mapped_column(
        Float,
        nullable=True
    )

    # Context and session
    session_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False),
        nullable=True,
        index=True
    )

    ip_address: Mapped[str | None] = mapped_column(
        String(45),  # IPv6 max length
        nullable=True
    )

    user_agent: Mapped[str | None] = mapped_column(
        Text,
        nullable=True
    )

    # Error handling
    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True
    )

    error_details: Mapped[Dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True
    )

    # Relationships
    results: Mapped[List["QueryResult"]] = relationship(
        "QueryResult",
        back_populates="query",
        cascade="all, delete-orphan"
    )

    feedback: Mapped[List["QueryFeedback"]] = relationship(
        "QueryFeedback",
        back_populates="query",
        cascade="all, delete-orphan"
    )

    @property
    def is_successful(self) -> bool:
        """Check if query was successful."""
        return self.status == QueryStatus.COMPLETED and self.has_results

    @property
    def total_processing_time(self) -> float | None:
        """Calculate total processing time."""
        if self.processing_started_at and self.processing_completed_at:
            delta = self.processing_completed_at - self.processing_started_at
            return delta.total_seconds()
        return self.processing_time

    def start_processing(self) -> None:
        """Mark query processing as started."""
        self.status = QueryStatus.PROCESSING
        self.processing_started_at = datetime.utcnow()

    def complete_processing(self, result_count: int = 0) -> None:
        """Mark query processing as completed."""
        self.status = QueryStatus.COMPLETED
        self.processing_completed_at = datetime.utcnow()
        self.result_count = result_count
        self.has_results = result_count > 0

        if self.processing_started_at:
            delta = self.processing_completed_at - self.processing_started_at
            self.processing_time = delta.total_seconds()

    def fail_processing(self, error: str, details: Dict[str, Any] = None) -> None:
        """Mark query processing as failed."""
        self.status = QueryStatus.FAILED
        self.error_message = error
        self.error_details = details
        self.processing_completed_at = datetime.utcnow()


class QueryResult(BaseModel):
    """
    Individual query result model.
    """

    __tablename__ = "query_results"

    # Foreign key to query
    query_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("queries.id"),
        nullable=False,
        index=True
    )

    # Result position
    rank: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        index=True
    )

    # Source information
    document_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        nullable=False,
        index=True
    )

    chunk_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        nullable=False,
        index=True
    )

    # Content
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False
    )

    content_preview: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True
    )

    # Scoring
    score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        index=True
    )

    vector_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True
    )

    rerank_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True
    )

    # Highlighting and snippets
    highlights: Mapped[List[str] | None] = mapped_column(
        ARRAY(Text),
        nullable=True
    )

    snippets: Mapped[List[str] | None] = mapped_column(
        ARRAY(Text),
        nullable=True
    )

    # Metadata
    result_metadata: Mapped[Dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True
    )

    # User interaction
    clicked: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False
    )

    clicked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )

    view_duration: Mapped[float | None] = mapped_column(
        Float,
        nullable=True
    )

    # Relationships
    query: Mapped["Query"] = relationship(
        "Query",
        back_populates="results"
    )

    def record_click(self) -> None:
        """Record that this result was clicked."""
        self.clicked = True
        self.clicked_at = datetime.utcnow()


class QueryFeedback(BaseModel, UserTrackingMixin):
    """
    User feedback on query results.
    """

    __tablename__ = "query_feedback"

    # Foreign key to query
    query_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("queries.id"),
        nullable=False,
        index=True
    )

    # Feedback content
    feedback_type: Mapped[FeedbackType] = mapped_column(
        SQLEnum(FeedbackType),
        nullable=False,
        index=True
    )

    rating: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        index=True
    )

    comments: Mapped[str | None] = mapped_column(
        Text,
        nullable=True
    )

    # Specific result feedback
    helpful_results: Mapped[List[str] | None] = mapped_column(
        ARRAY(UUID(as_uuid=False)),
        nullable=True
    )

    unhelpful_results: Mapped[List[str] | None] = mapped_column(
        ARRAY(UUID(as_uuid=False)),
        nullable=True
    )

    # Improvement suggestions
    suggested_query: Mapped[str | None] = mapped_column(
        Text,
        nullable=True
    )

    missing_information: Mapped[str | None] = mapped_column(
        Text,
        nullable=True
    )

    # Context
    feedback_context: Mapped[Dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True
    )

    # Processing
    processed: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False
    )

    processed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )

    # Relationships
    query: Mapped["Query"] = relationship(
        "Query",
        back_populates="feedback"
    )

    @property
    def is_positive(self) -> bool:
        """Check if feedback is positive (rating >= 4)."""
        return self.rating >= 4

    @property
    def is_negative(self) -> bool:
        """Check if feedback is negative (rating <= 2)."""
        return self.rating <= 2

    def mark_processed(self) -> None:
        """Mark feedback as processed."""
        self.processed = True
        self.processed_at = datetime.utcnow()
