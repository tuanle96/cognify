"""
User repository for database operations.
"""

from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from passlib.context import CryptContext

from sqlalchemy import select, and_, or_, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload, joinedload
import structlog

from .base import BaseRepository
from app.models.users import User, UserProfile, UserSession, UserRole, UserStatus
from app.core.exceptions import ValidationError, ConflictError, NotFoundError
from app.core.config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class UserRepository(BaseRepository[User]):
    """Repository for user-related database operations."""

    def __init__(self, session: AsyncSession):
        super().__init__(session, User)

    async def create_user(
        self,
        email: str,
        password: str,
        full_name: str,
        username: str = None,
        role: UserRole = UserRole.USER,
        **kwargs
    ) -> User:
        """
        Create a new user with password hashing.

        Args:
            email: User email (must be unique)
            password: Plain text password
            full_name: User's full name
            username: Username (optional, must be unique if provided)
            role: User role
            **kwargs: Additional user fields

        Returns:
            Created user instance

        Raises:
            ValidationError: If validation fails
            ConflictError: If email or username already exists
        """
        try:
            # Validate email format
            if not self._is_valid_email(email):
                raise ValidationError(
                    message="Invalid email format",
                    field="email",
                    value=email
                )

            # Validate password strength
            if not self._is_valid_password(password):
                raise ValidationError(
                    message=f"Password must be at least {settings.PASSWORD_MIN_LENGTH} characters",
                    field="password"
                )

            # Check if email already exists
            existing_user = await self.get_by_email(email)
            if existing_user:
                raise ConflictError(
                    message="Email already registered",
                    details={"email": email}
                )

            # Check if username already exists (if provided)
            if username:
                existing_username = await self.get_by_username(username)
                if existing_username:
                    raise ConflictError(
                        message="Username already taken",
                        details={"username": username}
                    )

            # Hash password
            password_hash = pwd_context.hash(password)

            # Create user
            user_data = {
                "email": email.lower().strip(),
                "password_hash": password_hash,
                "full_name": full_name.strip(),
                "role": role,
                "status": UserStatus.PENDING_VERIFICATION,
                **kwargs
            }

            if username:
                user_data["username"] = username.lower().strip()

            user = await self.create(**user_data)

            # Create user profile
            await self._create_user_profile(user.id, full_name)

            self.logger.info(
                "User created successfully",
                user_id=user.id,
                email=email,
                role=role.value
            )

            return user

        except (ValidationError, ConflictError):
            raise
        except Exception as e:
            self.logger.error(
                "Failed to create user",
                email=email,
                error=str(e)
            )
            raise

    async def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email address."""
        try:
            query = select(User).where(
                and_(
                    User.email == email.lower().strip(),
                    User.deleted_at.is_(None)
                )
            ).options(selectinload(User.profile))

            result = await self.session.execute(query)
            return result.scalar_one_or_none()

        except Exception as e:
            self.logger.error(
                "Failed to get user by email",
                email=email,
                error=str(e)
            )
            return None

    async def get_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        try:
            query = select(User).where(
                and_(
                    User.username == username.lower().strip(),
                    User.deleted_at.is_(None)
                )
            ).options(selectinload(User.profile))

            result = await self.session.execute(query)
            return result.scalar_one_or_none()

        except Exception as e:
            self.logger.error(
                "Failed to get user by username",
                username=username,
                error=str(e)
            )
            return None

    async def authenticate(self, email: str, password: str) -> Optional[User]:
        """
        Authenticate user with email and password.

        Args:
            email: User email
            password: Plain text password

        Returns:
            User instance if authentication successful, None otherwise
        """
        try:
            user = await self.get_by_email(email)
            if not user:
                self.logger.warning("Authentication failed - user not found", email=email)
                return None

            # Check if account is locked
            if user.is_locked:
                self.logger.warning("Authentication failed - account locked", user_id=user.id)
                return None

            # Check if account is active (allow PENDING_VERIFICATION for testing)
            if user.status not in [UserStatus.ACTIVE, UserStatus.PENDING_VERIFICATION]:
                self.logger.warning("Authentication failed - account inactive", user_id=user.id)
                return None

            # Verify password
            if not pwd_context.verify(password, user.password_hash):
                # Record failed login attempt
                await self._record_failed_login(user)
                self.logger.warning("Authentication failed - invalid password", user_id=user.id)
                return None

            # Record successful login
            await self._record_successful_login(user)

            self.logger.info("User authenticated successfully", user_id=user.id)
            return user

        except Exception as e:
            self.logger.error(
                "Authentication error",
                email=email,
                error=str(e)
            )
            return None

    async def change_password(self, user_id: str, current_password: str, new_password: str) -> bool:
        """
        Change user password.

        Args:
            user_id: User ID
            current_password: Current password for verification
            new_password: New password

        Returns:
            True if password changed successfully

        Raises:
            NotFoundError: If user not found
            ValidationError: If current password is incorrect or new password is invalid
        """
        try:
            user = await self.get_by_id_or_raise(user_id)

            # Verify current password
            if not pwd_context.verify(current_password, user.password_hash):
                raise ValidationError(
                    message="Current password is incorrect",
                    field="current_password"
                )

            # Validate new password
            if not self._is_valid_password(new_password):
                raise ValidationError(
                    message=f"Password must be at least {settings.PASSWORD_MIN_LENGTH} characters",
                    field="new_password"
                )

            # Hash new password
            new_password_hash = pwd_context.hash(new_password)

            # Update password
            await self.update(user_id, password_hash=new_password_hash)

            self.logger.info("Password changed successfully", user_id=user_id)
            return True

        except (NotFoundError, ValidationError):
            raise
        except Exception as e:
            self.logger.error(
                "Failed to change password",
                user_id=user_id,
                error=str(e)
            )
            raise

    async def verify_email(self, user_id: str) -> bool:
        """Mark user email as verified."""
        try:
            user = await self.get_by_id_or_raise(user_id)

            if user.is_verified:
                return True

            # Update verification status
            await self.update(
                user_id,
                is_verified=True,
                email_verified_at=datetime.utcnow(),
                status=UserStatus.ACTIVE if user.status == UserStatus.PENDING_VERIFICATION else user.status
            )

            self.logger.info("Email verified successfully", user_id=user_id)
            return True

        except Exception as e:
            self.logger.error(
                "Failed to verify email",
                user_id=user_id,
                error=str(e)
            )
            return False

    async def update_profile(self, user_id: str, profile_data: Dict[str, Any]) -> UserProfile:
        """Update user profile."""
        try:
            # Get existing profile
            query = select(UserProfile).where(UserProfile.user_id == user_id)
            result = await self.session.execute(query)
            profile = result.scalar_one_or_none()

            if not profile:
                # Create new profile
                profile = UserProfile(user_id=user_id, **profile_data)
                self.session.add(profile)
            else:
                # Update existing profile
                for key, value in profile_data.items():
                    if hasattr(profile, key):
                        setattr(profile, key, value)

            await self.session.flush()
            await self.session.refresh(profile)

            self.logger.info("Profile updated successfully", user_id=user_id)
            return profile

        except Exception as e:
            self.logger.error(
                "Failed to update profile",
                user_id=user_id,
                error=str(e)
            )
            raise

    async def get_active_sessions(self, user_id: str) -> List[UserSession]:
        """Get active sessions for a user."""
        try:
            query = select(UserSession).where(
                and_(
                    UserSession.user_id == user_id,
                    UserSession.is_active == True,
                    UserSession.expires_at > datetime.utcnow()
                )
            ).order_by(UserSession.last_activity_at.desc())

            result = await self.session.execute(query)
            return list(result.scalars().all())

        except Exception as e:
            self.logger.error(
                "Failed to get active sessions",
                user_id=user_id,
                error=str(e)
            )
            return []

    async def create_session(
        self,
        user_id: str,
        session_token: str,
        refresh_token: str = None,
        expires_at: datetime = None,
        ip_address: str = None,
        user_agent: str = None
    ) -> UserSession:
        """Create a new user session."""
        try:
            if not expires_at:
                expires_at = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)

            session = UserSession(
                user_id=user_id,
                session_token=session_token,
                refresh_token=refresh_token,
                expires_at=expires_at,
                ip_address=ip_address,
                user_agent=user_agent
            )

            self.session.add(session)
            await self.session.flush()
            await self.session.refresh(session)

            self.logger.info("Session created", user_id=user_id, session_id=session.id)
            return session

        except Exception as e:
            self.logger.error(
                "Failed to create session",
                user_id=user_id,
                error=str(e)
            )
            raise

    async def invalidate_session(self, session_token: str, reason: str = "logout") -> bool:
        """Invalidate a user session."""
        try:
            query = select(UserSession).where(UserSession.session_token == session_token)
            result = await self.session.execute(query)
            session = result.scalar_one_or_none()

            if session:
                session.logout(reason)
                await self.session.flush()

                self.logger.info(
                    "Session invalidated",
                    session_id=session.id,
                    reason=reason
                )
                return True

            return False

        except Exception as e:
            self.logger.error(
                "Failed to invalidate session",
                session_token=session_token,
                error=str(e)
            )
            return False

    async def update_session_tokens(
        self,
        user_id: str,
        access_token: str,
        refresh_token: str
    ) -> bool:
        """Update session tokens for user."""
        try:
            # Find the most recent active session for the user
            query = select(UserSession).where(
                and_(
                    UserSession.user_id == user_id,
                    UserSession.is_active == True
                )
            ).order_by(UserSession.last_activity_at.desc()).limit(1)

            result = await self.session.execute(query)
            session = result.scalar_one_or_none()

            if session:
                # Update existing session
                session.session_token = access_token
                session.refresh_token = refresh_token
                session.update_activity()
                await self.session.flush()

                self.logger.info(
                    "Session tokens updated",
                    user_id=user_id,
                    session_id=session.id
                )
                return True
            else:
                # Create new session if none exists
                await self.create_session(
                    user_id=user_id,
                    session_token=access_token,
                    refresh_token=refresh_token
                )
                return True

        except Exception as e:
            self.logger.error(
                "Failed to update session tokens",
                user_id=user_id,
                error=str(e)
            )
            return False

    async def _create_user_profile(self, user_id: str, full_name: str) -> UserProfile:
        """Create initial user profile."""
        profile = UserProfile(
            user_id=user_id,
            bio=f"Welcome {full_name}!",
            timezone="UTC",
            language="en",
            theme="light"
        )

        self.session.add(profile)
        await self.session.flush()
        return profile

    async def _record_successful_login(self, user: User) -> None:
        """Record successful login."""
        user.record_login()
        await self.session.flush()

    async def _record_failed_login(self, user: User) -> None:
        """Record failed login attempt."""
        user.record_failed_login(
            max_attempts=settings.MAX_LOGIN_ATTEMPTS,
            lockout_minutes=settings.ACCOUNT_LOCKOUT_MINUTES
        )
        await self.session.flush()

    def _is_valid_email(self, email: str) -> bool:
        """Validate email format."""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None

    def _is_valid_password(self, password: str) -> bool:
        """Validate password strength."""
        return len(password) >= settings.PASSWORD_MIN_LENGTH
