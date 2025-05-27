"""
User-related database models.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List

from sqlalchemy import Boolean, DateTime, Enum as SQLEnum, String, Text, Integer, func, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import BaseModel, TimestampMixin, UUIDMixin


class UserRole(str, Enum):
    """User role enumeration."""
    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"


class UserStatus(str, Enum):
    """User status enumeration."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING_VERIFICATION = "pending_verification"


class User(BaseModel):
    """
    User model for authentication and authorization.
    """

    __tablename__ = "users"

    # Basic user information
    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
        index=True
    )

    username: Mapped[str | None] = mapped_column(
        String(100),
        unique=True,
        nullable=True,
        index=True
    )

    full_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False
    )

    # Authentication
    password_hash: Mapped[str] = mapped_column(
        String(255),
        nullable=False
    )

    # Status and role
    status: Mapped[UserStatus] = mapped_column(
        SQLEnum(UserStatus),
        nullable=False,
        default=UserStatus.PENDING_VERIFICATION,
        index=True
    )

    role: Mapped[UserRole] = mapped_column(
        SQLEnum(UserRole),
        nullable=False,
        default=UserRole.USER,
        index=True
    )

    # Verification
    is_verified: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True
    )

    email_verified_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )

    # Login tracking
    last_login_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True
    )

    login_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0
    )

    # Security
    failed_login_attempts: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0
    )

    locked_until: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )

    # Preferences
    preferences: Mapped[Dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
        default=dict
    )

    # Relationships
    profile: Mapped["UserProfile"] = relationship(
        "UserProfile",
        back_populates="user",
        uselist=False,
        cascade="all, delete-orphan"
    )

    sessions: Mapped[List["UserSession"]] = relationship(
        "UserSession",
        back_populates="user",
        cascade="all, delete-orphan"
    )

    @property
    def is_active(self) -> bool:
        """Check if user is active."""
        return self.status == UserStatus.ACTIVE and not self.is_deleted

    @property
    def is_admin(self) -> bool:
        """Check if user is admin."""
        return self.role == UserRole.ADMIN

    @property
    def is_locked(self) -> bool:
        """Check if user account is locked."""
        if self.locked_until is None:
            return False
        return datetime.utcnow() < self.locked_until

    def verify_email(self) -> None:
        """Mark email as verified."""
        self.is_verified = True
        self.email_verified_at = datetime.utcnow()
        if self.status == UserStatus.PENDING_VERIFICATION:
            self.status = UserStatus.ACTIVE

    def record_login(self) -> None:
        """Record successful login."""
        self.last_login_at = datetime.utcnow()
        self.login_count += 1
        self.failed_login_attempts = 0
        self.locked_until = None

    def record_failed_login(self, max_attempts: int = 5, lockout_minutes: int = 30) -> None:
        """Record failed login attempt."""
        self.failed_login_attempts += 1

        if self.failed_login_attempts >= max_attempts:
            from datetime import timedelta
            self.locked_until = datetime.utcnow() + timedelta(minutes=lockout_minutes)


class UserProfile(BaseModel):
    """
    Extended user profile information.
    """

    __tablename__ = "user_profiles"

    # Foreign key to user
    user_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("users.id"),
        nullable=False,
        unique=True,
        index=True
    )

    # Profile information
    avatar_url: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True
    )

    bio: Mapped[str | None] = mapped_column(
        Text,
        nullable=True
    )

    location: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True
    )

    website: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True
    )

    company: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True
    )

    job_title: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True
    )

    # Preferences
    timezone: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
        default="UTC"
    )

    language: Mapped[str | None] = mapped_column(
        String(10),
        nullable=True,
        default="en"
    )

    theme: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True,
        default="light"
    )

    # Notification preferences
    email_notifications: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True
    )

    push_notifications: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True
    )

    # Usage statistics
    total_documents: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0
    )

    total_queries: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0
    )

    total_collections: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0
    )

    # Relationships
    user: Mapped["User"] = relationship(
        "User",
        back_populates="profile"
    )


class UserSession(BaseModel):
    """
    User session tracking for security and analytics.
    """

    __tablename__ = "user_sessions"

    # Foreign key to user
    user_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("users.id"),
        nullable=False,
        index=True
    )

    # Session information
    session_token: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        unique=True,
        index=True
    )

    refresh_token: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        unique=True,
        index=True
    )

    # Session metadata
    ip_address: Mapped[str | None] = mapped_column(
        String(45),  # IPv6 max length
        nullable=True,
        index=True
    )

    user_agent: Mapped[str | None] = mapped_column(
        Text,
        nullable=True
    )

    device_info: Mapped[Dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True
    )

    # Session status
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        index=True
    )

    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True
    )

    last_activity_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=func.now(),
        index=True
    )

    # Logout information
    logged_out_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )

    logout_reason: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True
    )

    # Relationships
    user: Mapped["User"] = relationship(
        "User",
        back_populates="sessions"
    )

    @property
    def is_expired(self) -> bool:
        """Check if session is expired."""
        return datetime.utcnow() > self.expires_at

    @property
    def is_valid(self) -> bool:
        """Check if session is valid (active and not expired)."""
        return self.is_active and not self.is_expired

    def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity_at = datetime.utcnow()

    def logout(self, reason: str = "user_logout") -> None:
        """Mark session as logged out."""
        self.is_active = False
        self.logged_out_at = datetime.utcnow()
        self.logout_reason = reason


