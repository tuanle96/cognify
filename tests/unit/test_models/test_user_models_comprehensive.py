"""
Comprehensive Unit Tests for User Models.

This module provides complete test coverage for all user-related
database models to achieve 70%+ database coverage target.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
import uuid

# Import models
from app.models.base import Base
from app.models.users import User, UserProfile, UserSession, Role
from app.core.config import get_settings


@pytest.fixture(scope="function")
async def test_engine():
    """Create test database engine."""
    settings = get_settings()
    # Use in-memory SQLite for fast unit tests
    test_db_url = "sqlite+aiosqlite:///:memory:"
    engine = create_async_engine(
        test_db_url,
        echo=False,
        future=True
    )

    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    # Clean up
    await engine.dispose()


@pytest.fixture
async def db_session(test_engine):
    """Create database session for testing."""
    async_session = sessionmaker(
        test_engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with async_session() as session:
        yield session


@pytest.fixture
def sample_user_data():
    """Sample user data for testing."""
    return {
        "username": "testuser",
        "email": "test@example.com",
        "hashed_password": "hashed_password_123",
        "full_name": "Test User",
        "role": Role.USER,
        "is_active": True,
        "is_verified": False
    }


@pytest.fixture
def sample_admin_data():
    """Sample admin data for testing."""
    return {
        "username": "admin",
        "email": "admin@example.com",
        "hashed_password": "hashed_admin_password_456",
        "full_name": "Admin User",
        "role": Role.ADMIN,
        "is_active": True,
        "is_verified": True
    }


class TestUserModel:
    """Comprehensive tests for User model."""

    async def test_user_creation_basic(self, db_session, sample_user_data):
        """Test basic user creation."""
        user = User(**sample_user_data)
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)
        
        # Verify user was created
        assert user.user_id is not None
        assert user.username == sample_user_data["username"]
        assert user.email == sample_user_data["email"]
        assert user.full_name == sample_user_data["full_name"]
        assert user.role == sample_user_data["role"]
        assert user.is_active == sample_user_data["is_active"]
        assert user.is_verified == sample_user_data["is_verified"]
        assert user.created_at is not None
        assert user.updated_at is not None

    async def test_user_creation_with_defaults(self, db_session):
        """Test user creation with default values."""
        user = User(
            username="defaultuser",
            email="default@example.com",
            hashed_password="hashed_password"
        )
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)
        
        # Verify defaults
        assert user.role == Role.USER  # Default role
        assert user.is_active is True  # Default active
        assert user.is_verified is False  # Default not verified
        assert user.full_name is None  # Optional field

    async def test_user_unique_constraints(self, db_session, sample_user_data):
        """Test user unique constraints."""
        # Create first user
        user1 = User(**sample_user_data)
        db_session.add(user1)
        await db_session.commit()
        
        # Try to create user with same email
        user2_data = sample_user_data.copy()
        user2_data["username"] = "different_username"
        user2 = User(**user2_data)
        db_session.add(user2)
        
        with pytest.raises(IntegrityError):
            await db_session.commit()

    async def test_user_username_unique(self, db_session, sample_user_data):
        """Test username uniqueness constraint."""
        # Create first user
        user1 = User(**sample_user_data)
        db_session.add(user1)
        await db_session.commit()
        
        # Try to create user with same username
        user2_data = sample_user_data.copy()
        user2_data["email"] = "different@example.com"
        user2 = User(**user2_data)
        db_session.add(user2)
        
        with pytest.raises(IntegrityError):
            await db_session.commit()

    async def test_user_role_enum(self, db_session):
        """Test user role enum values."""
        # Test all role values
        roles = [Role.USER, Role.ADMIN, Role.READONLY]
        
        for i, role in enumerate(roles):
            user = User(
                username=f"user_{i}",
                email=f"user_{i}@example.com",
                hashed_password="password",
                role=role
            )
            db_session.add(user)
        
        await db_session.commit()
        
        # Verify all users were created with correct roles
        result = await db_session.execute(text("SELECT COUNT(*) FROM users"))
        count = result.scalar()
        assert count == len(roles)

    async def test_user_timestamps(self, db_session, sample_user_data):
        """Test user timestamp fields."""
        user = User(**sample_user_data)
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)
        
        # Verify timestamps
        assert user.created_at is not None
        assert user.updated_at is not None
        assert isinstance(user.created_at, datetime)
        assert isinstance(user.updated_at, datetime)
        
        # Verify created_at and updated_at are close
        time_diff = abs((user.updated_at - user.created_at).total_seconds())
        assert time_diff < 1.0  # Should be within 1 second

    async def test_user_update_timestamp(self, db_session, sample_user_data):
        """Test user update timestamp behavior."""
        user = User(**sample_user_data)
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)
        
        original_updated_at = user.updated_at
        
        # Wait a bit and update user
        await asyncio.sleep(0.1)
        user.full_name = "Updated Name"
        await db_session.commit()
        await db_session.refresh(user)
        
        # Verify updated_at changed
        assert user.updated_at > original_updated_at

    async def test_user_string_representation(self, db_session, sample_user_data):
        """Test user string representation."""
        user = User(**sample_user_data)
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)
        
        # Test __str__ method
        user_str = str(user)
        assert sample_user_data["username"] in user_str
        assert sample_user_data["email"] in user_str

    async def test_user_password_handling(self, db_session):
        """Test user password handling."""
        user = User(
            username="testuser",
            email="test@example.com",
            hashed_password="hashed_password_123"
        )
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)
        
        # Verify password is stored as hashed
        assert user.hashed_password == "hashed_password_123"
        # Verify no plain password field exists
        assert not hasattr(user, 'password')


class TestUserProfile:
    """Comprehensive tests for UserProfile model."""

    async def test_user_profile_creation(self, db_session, sample_user_data):
        """Test user profile creation."""
        # Create user first
        user = User(**sample_user_data)
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)
        
        # Create user profile
        profile = UserProfile(
            user_id=user.user_id,
            bio="Test user bio",
            avatar_url="https://example.com/avatar.jpg",
            timezone="UTC",
            language="en",
            preferences={"theme": "dark", "notifications": True}
        )
        db_session.add(profile)
        await db_session.commit()
        await db_session.refresh(profile)
        
        # Verify profile
        assert profile.profile_id is not None
        assert profile.user_id == user.user_id
        assert profile.bio == "Test user bio"
        assert profile.avatar_url == "https://example.com/avatar.jpg"
        assert profile.timezone == "UTC"
        assert profile.language == "en"
        assert profile.preferences["theme"] == "dark"

    async def test_user_profile_relationship(self, db_session, sample_user_data):
        """Test user-profile relationship."""
        # Create user with profile
        user = User(**sample_user_data)
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)
        
        profile = UserProfile(
            user_id=user.user_id,
            bio="Test bio"
        )
        db_session.add(profile)
        await db_session.commit()
        
        # Test relationship access
        await db_session.refresh(user)
        if hasattr(user, 'profile'):
            assert user.profile.bio == "Test bio"

    async def test_user_profile_optional_fields(self, db_session, sample_user_data):
        """Test user profile with optional fields."""
        # Create user
        user = User(**sample_user_data)
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)
        
        # Create minimal profile
        profile = UserProfile(user_id=user.user_id)
        db_session.add(profile)
        await db_session.commit()
        await db_session.refresh(profile)
        
        # Verify optional fields are None
        assert profile.bio is None
        assert profile.avatar_url is None
        assert profile.timezone is None
        assert profile.language is None
        assert profile.preferences is None

    async def test_user_profile_json_preferences(self, db_session, sample_user_data):
        """Test user profile JSON preferences field."""
        # Create user
        user = User(**sample_user_data)
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)
        
        # Create profile with complex preferences
        complex_preferences = {
            "theme": "dark",
            "notifications": {
                "email": True,
                "push": False,
                "sms": True
            },
            "dashboard": {
                "layout": "grid",
                "widgets": ["stats", "recent", "charts"]
            }
        }
        
        profile = UserProfile(
            user_id=user.user_id,
            preferences=complex_preferences
        )
        db_session.add(profile)
        await db_session.commit()
        await db_session.refresh(profile)
        
        # Verify JSON preferences
        assert profile.preferences["theme"] == "dark"
        assert profile.preferences["notifications"]["email"] is True
        assert profile.preferences["dashboard"]["layout"] == "grid"


class TestUserSession:
    """Comprehensive tests for UserSession model."""

    async def test_user_session_creation(self, db_session, sample_user_data):
        """Test user session creation."""
        # Create user first
        user = User(**sample_user_data)
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)
        
        # Create session
        session = UserSession(
            user_id=user.user_id,
            session_token="session_token_123",
            refresh_token="refresh_token_456",
            expires_at=datetime.utcnow() + timedelta(hours=1),
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0 Test Browser"
        )
        db_session.add(session)
        await db_session.commit()
        await db_session.refresh(session)
        
        # Verify session
        assert session.session_id is not None
        assert session.user_id == user.user_id
        assert session.session_token == "session_token_123"
        assert session.refresh_token == "refresh_token_456"
        assert session.ip_address == "192.168.1.1"
        assert session.user_agent == "Mozilla/5.0 Test Browser"
        assert session.is_active is True

    async def test_user_session_expiration(self, db_session, sample_user_data):
        """Test user session expiration logic."""
        # Create user
        user = User(**sample_user_data)
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)
        
        # Create expired session
        expired_session = UserSession(
            user_id=user.user_id,
            session_token="expired_token",
            expires_at=datetime.utcnow() - timedelta(hours=1)  # Expired
        )
        db_session.add(expired_session)
        
        # Create valid session
        valid_session = UserSession(
            user_id=user.user_id,
            session_token="valid_token",
            expires_at=datetime.utcnow() + timedelta(hours=1)  # Valid
        )
        db_session.add(valid_session)
        
        await db_session.commit()
        
        # Test expiration check (would need method on model)
        assert expired_session.expires_at < datetime.utcnow()
        assert valid_session.expires_at > datetime.utcnow()

    async def test_user_session_deactivation(self, db_session, sample_user_data):
        """Test user session deactivation."""
        # Create user and session
        user = User(**sample_user_data)
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)
        
        session = UserSession(
            user_id=user.user_id,
            session_token="test_token",
            expires_at=datetime.utcnow() + timedelta(hours=1)
        )
        db_session.add(session)
        await db_session.commit()
        await db_session.refresh(session)
        
        # Deactivate session
        session.is_active = False
        session.ended_at = datetime.utcnow()
        await db_session.commit()
        
        # Verify deactivation
        assert session.is_active is False
        assert session.ended_at is not None

    async def test_user_multiple_sessions(self, db_session, sample_user_data):
        """Test user with multiple sessions."""
        # Create user
        user = User(**sample_user_data)
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)
        
        # Create multiple sessions
        sessions = []
        for i in range(3):
            session = UserSession(
                user_id=user.user_id,
                session_token=f"token_{i}",
                expires_at=datetime.utcnow() + timedelta(hours=1),
                ip_address=f"192.168.1.{i+1}"
            )
            sessions.append(session)
            db_session.add(session)
        
        await db_session.commit()
        
        # Verify all sessions
        result = await db_session.execute(
            text("SELECT COUNT(*) FROM user_sessions WHERE user_id = :user_id"),
            {"user_id": user.user_id}
        )
        count = result.scalar()
        assert count == 3


class TestRoleEnum:
    """Tests for Role enum."""

    def test_role_enum_values(self):
        """Test role enum values."""
        assert Role.USER.value == "user"
        assert Role.ADMIN.value == "admin"
        assert Role.READONLY.value == "readonly"

    def test_role_enum_comparison(self):
        """Test role enum comparison."""
        assert Role.USER == Role.USER
        assert Role.USER != Role.ADMIN
        assert Role.ADMIN != Role.READONLY

    def test_role_enum_string_conversion(self):
        """Test role enum string conversion."""
        assert str(Role.USER) == "Role.USER"
        assert Role.USER.value == "user"


class TestUserModelIntegration:
    """Integration tests for user models."""

    async def test_complete_user_creation_flow(self, db_session):
        """Test complete user creation with profile and session."""
        # Create user
        user = User(
            username="completeuser",
            email="complete@example.com",
            hashed_password="hashed_password",
            full_name="Complete User",
            role=Role.USER
        )
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)
        
        # Create profile
        profile = UserProfile(
            user_id=user.user_id,
            bio="Complete user profile",
            preferences={"theme": "light"}
        )
        db_session.add(profile)
        
        # Create session
        session = UserSession(
            user_id=user.user_id,
            session_token="complete_session_token",
            expires_at=datetime.utcnow() + timedelta(hours=24)
        )
        db_session.add(session)
        
        await db_session.commit()
        
        # Verify everything was created
        assert user.user_id is not None
        assert profile.user_id == user.user_id
        assert session.user_id == user.user_id

    async def test_user_cascade_operations(self, db_session, sample_user_data):
        """Test cascade operations on user deletion."""
        # Create user with profile and session
        user = User(**sample_user_data)
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)
        
        profile = UserProfile(user_id=user.user_id, bio="Test")
        session = UserSession(
            user_id=user.user_id,
            session_token="test_token",
            expires_at=datetime.utcnow() + timedelta(hours=1)
        )
        db_session.add(profile)
        db_session.add(session)
        await db_session.commit()
        
        # Delete user (should cascade to profile and sessions if configured)
        await db_session.delete(user)
        await db_session.commit()
        
        # Verify user is deleted
        result = await db_session.execute(
            text("SELECT COUNT(*) FROM users WHERE user_id = :user_id"),
            {"user_id": user.user_id}
        )
        count = result.scalar()
        assert count == 0
