"""
Base model classes and mixins for Cognify database models.
"""

from datetime import datetime
from typing import Any, Dict
from uuid import uuid4

from sqlalchemy import DateTime, String, Text, func
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for all database models."""
    
    # Use JSONB for JSON columns in PostgreSQL
    type_annotation_map = {
        dict: JSONB,
        Dict[str, Any]: JSONB,
    }


class TimestampMixin:
    """Mixin for adding created_at and updated_at timestamps."""
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True
    )
    
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        index=True
    )


class UUIDMixin:
    """Mixin for adding UUID primary key."""
    
    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=lambda: str(uuid4()),
        nullable=False
    )


class SoftDeleteMixin:
    """Mixin for soft delete functionality."""
    
    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True
    )
    
    @property
    def is_deleted(self) -> bool:
        """Check if the record is soft deleted."""
        return self.deleted_at is not None
    
    def soft_delete(self) -> None:
        """Mark the record as deleted."""
        self.deleted_at = datetime.utcnow()
    
    def restore(self) -> None:
        """Restore a soft deleted record."""
        self.deleted_at = None


class MetadataMixin:
    """Mixin for adding metadata JSON field."""
    
    metadata: Mapped[Dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
        default=dict
    )


class UserTrackingMixin:
    """Mixin for tracking user who created/modified records."""
    
    created_by: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False),
        nullable=True,
        index=True
    )
    
    updated_by: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False),
        nullable=True,
        index=True
    )


class BaseModel(Base, UUIDMixin, TimestampMixin, SoftDeleteMixin, MetadataMixin):
    """
    Base model class with common fields and functionality.
    
    Includes:
    - UUID primary key
    - Created/updated timestamps
    - Soft delete functionality
    - Metadata JSON field
    """
    
    __abstract__ = True
    
    def to_dict(self, exclude_fields: set = None) -> Dict[str, Any]:
        """
        Convert model instance to dictionary.
        
        Args:
            exclude_fields: Set of field names to exclude
            
        Returns:
            Dictionary representation of the model
        """
        exclude_fields = exclude_fields or set()
        result = {}
        
        for column in self.__table__.columns:
            if column.name not in exclude_fields:
                value = getattr(self, column.name)
                
                # Handle datetime serialization
                if isinstance(value, datetime):
                    value = value.isoformat()
                
                result[column.name] = value
        
        return result
    
    def update_from_dict(self, data: Dict[str, Any], exclude_fields: set = None) -> None:
        """
        Update model instance from dictionary.
        
        Args:
            data: Dictionary with field values
            exclude_fields: Set of field names to exclude from update
        """
        exclude_fields = exclude_fields or {'id', 'created_at'}
        
        for key, value in data.items():
            if key not in exclude_fields and hasattr(self, key):
                setattr(self, key, value)
    
    def __repr__(self) -> str:
        """String representation of the model."""
        return f"<{self.__class__.__name__}(id={self.id})>"


class AuditLogMixin:
    """Mixin for audit logging functionality."""
    
    action: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True
    )
    
    entity_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True
    )
    
    entity_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        nullable=False,
        index=True
    )
    
    old_values: Mapped[Dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True
    )
    
    new_values: Mapped[Dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True
    )
    
    user_id: Mapped[str | None] = mapped_column(
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


class VersionedMixin:
    """Mixin for versioning support."""
    
    version: Mapped[int] = mapped_column(
        nullable=False,
        default=1
    )
    
    def increment_version(self) -> None:
        """Increment the version number."""
        self.version += 1
