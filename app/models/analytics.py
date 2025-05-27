"""
Analytics and monitoring database models.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List

from sqlalchemy import Boolean, DateTime, Enum as SQLEnum, String, Text, Integer, Float
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .base import BaseModel, TimestampMixin, UUIDMixin


class ActivityType(str, Enum):
    """User activity type enumeration."""
    LOGIN = "login"
    LOGOUT = "logout"
    DOCUMENT_UPLOAD = "document_upload"
    DOCUMENT_DELETE = "document_delete"
    QUERY_SUBMIT = "query_submit"
    COLLECTION_CREATE = "collection_create"
    COLLECTION_DELETE = "collection_delete"
    FEEDBACK_SUBMIT = "feedback_submit"
    SETTINGS_UPDATE = "settings_update"
    PASSWORD_CHANGE = "password_change"
    PROFILE_UPDATE = "profile_update"


class MetricType(str, Enum):
    """System metric type enumeration."""
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    DISK_USAGE = "disk_usage"
    NETWORK_IO = "network_io"
    REQUEST_RATE = "request_rate"
    ERROR_RATE = "error_rate"
    RESPONSE_TIME = "response_time"
    ACTIVE_USERS = "active_users"
    QUERY_THROUGHPUT = "query_throughput"
    INDEXING_RATE = "indexing_rate"


class LogLevel(str, Enum):
    """Performance log level enumeration."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class UserActivity(BaseModel):
    """
    User activity tracking for analytics and auditing.
    """
    
    __tablename__ = "user_activities"
    
    # User information
    user_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False),
        nullable=True,
        index=True
    )
    
    session_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False),
        nullable=True,
        index=True
    )
    
    # Activity details
    activity_type: Mapped[ActivityType] = mapped_column(
        SQLEnum(ActivityType),
        nullable=False,
        index=True
    )
    
    activity_description: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True
    )
    
    # Context information
    ip_address: Mapped[str | None] = mapped_column(
        String(45),  # IPv6 max length
        nullable=True,
        index=True
    )
    
    user_agent: Mapped[str | None] = mapped_column(
        Text,
        nullable=True
    )
    
    # Request details
    endpoint: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True
    )
    
    method: Mapped[str | None] = mapped_column(
        String(10),
        nullable=True
    )
    
    status_code: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        index=True
    )
    
    response_time: Mapped[float | None] = mapped_column(
        Float,
        nullable=True
    )
    
    # Activity data
    activity_data: Mapped[Dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True
    )
    
    # Resource information
    resource_type: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
        index=True
    )
    
    resource_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False),
        nullable=True,
        index=True
    )
    
    # Success/failure
    success: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        index=True
    )
    
    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True
    )
    
    # Geolocation (optional)
    country: Mapped[str | None] = mapped_column(
        String(2),  # ISO country code
        nullable=True,
        index=True
    )
    
    city: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True
    )
    
    # Device information
    device_type: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True,
        index=True
    )
    
    browser: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True
    )
    
    os: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True
    )


class SystemMetrics(UUIDMixin, TimestampMixin):
    """
    System performance metrics for monitoring.
    """
    
    __tablename__ = "system_metrics"
    
    # Metric information
    metric_type: Mapped[MetricType] = mapped_column(
        SQLEnum(MetricType),
        nullable=False,
        index=True
    )
    
    metric_name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True
    )
    
    # Metric values
    value: Mapped[float] = mapped_column(
        Float,
        nullable=False
    )
    
    unit: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True
    )
    
    # Context
    service_name: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
        index=True
    )
    
    instance_id: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        index=True
    )
    
    # Additional data
    labels: Mapped[Dict[str, str] | None] = mapped_column(
        JSONB,
        nullable=True
    )
    
    dimensions: Mapped[Dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True
    )
    
    # Aggregation period
    period_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True
    )
    
    period_end: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True
    )
    
    # Statistical values
    min_value: Mapped[float | None] = mapped_column(
        Float,
        nullable=True
    )
    
    max_value: Mapped[float | None] = mapped_column(
        Float,
        nullable=True
    )
    
    avg_value: Mapped[float | None] = mapped_column(
        Float,
        nullable=True
    )
    
    sample_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True
    )


class PerformanceLog(BaseModel):
    """
    Performance and error logging for system monitoring.
    """
    
    __tablename__ = "performance_logs"
    
    # Log level and source
    level: Mapped[LogLevel] = mapped_column(
        SQLEnum(LogLevel),
        nullable=False,
        index=True
    )
    
    logger_name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True
    )
    
    # Message
    message: Mapped[str] = mapped_column(
        Text,
        nullable=False
    )
    
    # Context
    service_name: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
        index=True
    )
    
    function_name: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True
    )
    
    file_name: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True
    )
    
    line_number: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True
    )
    
    # Request context
    request_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False),
        nullable=True,
        index=True
    )
    
    user_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False),
        nullable=True,
        index=True
    )
    
    session_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False),
        nullable=True,
        index=True
    )
    
    # Performance data
    execution_time: Mapped[float | None] = mapped_column(
        Float,
        nullable=True
    )
    
    memory_usage: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True
    )
    
    cpu_usage: Mapped[float | None] = mapped_column(
        Float,
        nullable=True
    )
    
    # Error information
    exception_type: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        index=True
    )
    
    exception_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True
    )
    
    stack_trace: Mapped[str | None] = mapped_column(
        Text,
        nullable=True
    )
    
    # Additional data
    extra_data: Mapped[Dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True
    )
    
    # Tags for filtering
    tags: Mapped[List[str] | None] = mapped_column(
        JSONB,
        nullable=True
    )
    
    # Environment
    environment: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True,
        index=True
    )
    
    version: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True
    )
    
    @property
    def is_error(self) -> bool:
        """Check if this is an error log."""
        return self.level in (LogLevel.ERROR, LogLevel.CRITICAL)
    
    @property
    def is_performance_issue(self) -> bool:
        """Check if this indicates a performance issue."""
        if self.execution_time and self.execution_time > 5.0:  # 5 seconds threshold
            return True
        if self.memory_usage and self.memory_usage > 1024 * 1024 * 100:  # 100MB threshold
            return True
        return False
