"""
Authentication and authorization API endpoints.

Handles user registration, login, token management, and access control.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, EmailStr
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import jwt
import bcrypt

# Simple logger replacement
class SimpleLogger:
    def info(self, msg, **kwargs):
        print(f"INFO: {msg} {kwargs}")
    def debug(self, msg, **kwargs):
        print(f"DEBUG: {msg} {kwargs}")
    def warning(self, msg, **kwargs):
        print(f"WARNING: {msg} {kwargs}")
    def error(self, msg, **kwargs):
        print(f"ERROR: {msg} {kwargs}")

logger = SimpleLogger()

from app.api.dependencies import (
    get_user_repository,
    get_current_user_from_db,
    get_current_active_user_from_db,
    get_current_verified_user_from_db,
    get_db_session
)
from app.api.models.auth import (
    UserRegistrationRequest,
    UserLoginRequest,
    UserResponse,
    TokenResponse,
    TokenRefreshRequest,
    PasswordResetRequest,
    PasswordChangeRequest
)
from app.core.config import get_settings
from app.core.exceptions import ValidationError, ConflictError, AuthenticationError
from app.services.database.repositories import UserRepository
from app.models.users import User, UserRole

# Use the logger defined above
router = APIRouter()
security = HTTPBearer()
settings = get_settings()


@router.post("/register", response_model=UserResponse)
async def register_user(
    request: UserRegistrationRequest,
    user_repo: UserRepository = Depends(get_user_repository)
):
    """
    Register a new user account.

    Creates a new user with email verification required.
    """
    try:
        # Create user using repository
        user = await user_repo.create_user(
            email=request.email,
            password=request.password,
            full_name=request.full_name,
            username=getattr(request, 'username', None),
            role=UserRole.USER
        )

        # Auto-activate user for testing (in production, this should be done via email verification)
        await user_repo.verify_email(user.id)

        # Refresh user object to get updated status
        user = await user_repo.get_by_id(user.id)

        # Generate verification token
        verification_token = _generate_verification_token(user.id)

        # TODO: Send verification email

        logger.info(
            "User registered successfully",
            user_id=user.id,
            email=request.email,
            is_active=user.is_active
        )

        return UserResponse(
            user_id=user.id,
            email=user.email,
            full_name=user.full_name,
            is_active=user.is_active,
            is_verified=user.is_verified,
            role=user.role.value,
            created_at=user.created_at
        )

    except ConflictError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        logger.error("User registration failed", error=str(e), email=request.email)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )


@router.post("/login", response_model=TokenResponse)
async def login_user(
    request: UserLoginRequest,
    user_repo: UserRepository = Depends(get_user_repository)
):
    """
    Authenticate user and return access tokens.

    Returns JWT access token and refresh token for authenticated sessions.
    """
    try:
        # Authenticate user using repository
        user = await user_repo.authenticate(request.email, request.password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )

        # Generate tokens
        access_token = _generate_access_token_for_user(user)
        refresh_token = _generate_refresh_token_for_user(user)

        # Create session
        session = await user_repo.create_session(
            user_id=user.id,
            session_token=access_token,
            refresh_token=refresh_token
        )

        logger.info(
            "User logged in successfully",
            user_id=user.id,
            email=request.email,
            session_id=session.id
        )

        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            user=UserResponse(
                user_id=user.id,
                email=user.email,
                full_name=user.full_name,
                is_active=user.is_active,
                is_verified=user.is_verified,
                role=user.role.value,
                created_at=user.created_at
            )
        )

    except HTTPException:
        raise
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e)
        )
    except Exception as e:
        logger.error("User login failed", error=str(e), email=request.email)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    request: TokenRefreshRequest,
    user_repo: UserRepository = Depends(get_user_repository)
):
    """
    Refresh access token using refresh token.
    """
    try:
        # Verify refresh token
        payload = _verify_token(request.refresh_token, token_type="refresh")
        user_id = payload.get("sub")

        logger.info(
            "Token refresh attempt",
            user_id=user_id,
            token_type=payload.get("type")
        )

        # Get user from database
        user = await user_repo.get_by_id(user_id)

        if not user:
            logger.warning("User not found for refresh token", user_id=user_id)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )

        if not user.is_active:
            logger.warning("User not active for refresh token", user_id=user_id, is_active=user.is_active)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )

        # Generate new tokens
        access_token = _generate_access_token_for_user(user)
        refresh_token = _generate_refresh_token_for_user(user)

        # Update session with new tokens
        await user_repo.update_session_tokens(
            user_id=user.id,
            access_token=access_token,
            refresh_token=refresh_token
        )

        logger.info(
            "Token refreshed successfully",
            user_id=user.id,
            email=user.email
        )

        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            user=UserResponse(
                user_id=user.id,
                email=user.email,
                full_name=user.full_name,
                is_active=user.is_active,
                is_verified=user.is_verified,
                role=user.role.value,
                created_at=user.created_at
            )
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Token refresh failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type"
        )


@router.post("/logout")
async def logout_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db_session = Depends(get_db_session)
):
    """
    Logout user and invalidate tokens.
    """
    try:
        # Verify token
        payload = _verify_token(credentials.credentials)
        user_id = payload.get("sub")

        # TODO: Add token to blacklist

        logger.info("User logged out", user_id=user_id)

        return {"message": "Logged out successfully"}

    except Exception as e:
        logger.error("Logout failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_verified_user_from_db)
):
    """
    Get current user information.
    """
    return UserResponse(
        user_id=current_user.id,
        email=current_user.email,
        full_name=current_user.full_name,
        is_active=current_user.is_active,
        is_verified=current_user.is_verified,
        role=current_user.role.value,
        created_at=current_user.created_at
    )


@router.post("/password/reset")
async def request_password_reset(
    request: PasswordResetRequest,
    db_session = Depends(get_db_session)
):
    """
    Request password reset email.
    """
    try:
        # Check if user exists
        user = await _get_user_by_email(db_session, request.email)
        if not user:
            # Don't reveal if email exists or not
            return {"message": "If the email exists, a reset link has been sent"}

        # Generate reset token
        reset_token = _generate_password_reset_token(user["user_id"])

        # TODO: Send reset email

        logger.info("Password reset requested", user_id=user["user_id"], email=request.email)

        return {"message": "If the email exists, a reset link has been sent"}

    except Exception as e:
        logger.error("Password reset request failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password reset request failed"
        )


@router.post("/password/change")
async def change_password(
    request: PasswordChangeRequest,
    current_user: User = Depends(get_current_verified_user_from_db),
    user_repo: UserRepository = Depends(get_user_repository)
):
    """
    Change user password.
    """
    try:
        # Change password using repository
        success = await user_repo.change_password(
            user_id=current_user.id,
            current_password=request.current_password,
            new_password=request.new_password
        )

        if success:
            logger.info("Password changed successfully", user_id=current_user.id)
            return {"message": "Password changed successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password change failed"
            )

    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Password change failed", error=str(e), user_id=current_user.id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password change failed"
        )


# Helper functions

def _hash_password(password: str) -> str:
    """Hash password using bcrypt."""
    import bcrypt
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

def _verify_password(password: str, password_hash: str) -> bool:
    """Verify password against hash."""
    import bcrypt
    return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))

def _generate_access_token_for_user(user: User) -> str:
    """Generate JWT access token for user."""
    payload = {
        "sub": user.id,
        "email": user.email,
        "role": user.role.value,
        "type": "access",
        "exp": datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
        "iat": datetime.utcnow()
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm="HS256")

def _generate_refresh_token_for_user(user: User) -> str:
    """Generate JWT refresh token for user."""
    payload = {
        "sub": user.id,
        "type": "refresh",
        "exp": datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
        "iat": datetime.utcnow()
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm="HS256")

def _generate_access_token(user: dict) -> str:
    """Generate JWT access token."""
    payload = {
        "sub": user["user_id"],
        "email": user["email"],
        "role": user["role"],
        "type": "access",
        "exp": datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
        "iat": datetime.utcnow()
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm="HS256")

def _generate_refresh_token(user: dict) -> str:
    """Generate JWT refresh token."""
    payload = {
        "sub": user["user_id"],
        "type": "refresh",
        "exp": datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
        "iat": datetime.utcnow()
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm="HS256")

def _generate_verification_token(user_id: str) -> str:
    """Generate email verification token."""
    payload = {
        "sub": user_id,
        "type": "verification",
        "exp": datetime.utcnow() + timedelta(hours=24),
        "iat": datetime.utcnow()
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm="HS256")

def _generate_password_reset_token(user_id: str) -> str:
    """Generate password reset token."""
    payload = {
        "sub": user_id,
        "type": "password_reset",
        "exp": datetime.utcnow() + timedelta(hours=1),
        "iat": datetime.utcnow()
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm="HS256")

def _verify_token(token: str, token_type: str = "access") -> dict:
    """Verify JWT token."""
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
        if payload.get("type") != token_type:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type"
            )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

# Database helper functions (to be implemented)
async def _get_user_by_email(db_session, email: str) -> Optional[dict]:
    """Get user by email from database."""
    # TODO: Implement database query
    return None

async def _get_user_by_id(db_session, user_id: str) -> Optional[dict]:
    """Get user by ID from database."""
    # TODO: Implement database query
    return None

async def _create_user(db_session, user_data: dict) -> dict:
    """Create new user in database."""
    # TODO: Implement database insertion
    return user_data

async def _update_last_login(db_session, user_id: str):
    """Update user's last login timestamp."""
    # TODO: Implement database update
    pass

async def _update_user_password(db_session, user_id: str, password_hash: str):
    """Update user's password hash."""
    # TODO: Implement database update
    pass

# Import get_current_user function
def get_current_user_dependency():
    """Get current user dependency to avoid circular imports."""
    from app.api.dependencies import get_current_user
    return get_current_user
