"""
Test FastAPI application with mocked services.

This module provides a test version of the FastAPI app that uses
mock services instead of real database and external service connections.
"""

from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import uuid

from tests.test_config import (
    get_test_settings,
    get_mock_services,
    SAMPLE_PYTHON_CODE,
    SAMPLE_JAVASCRIPT_CODE
)

# Initialize test settings and services
test_settings = get_test_settings()
mock_services = get_mock_services()

# Create test FastAPI app
test_app = FastAPI(
    title="Cognify Test API",
    description="Test version of Cognify API with mocked services",
    version="1.0.0-test"
)

# Add CORS middleware
test_app.add_middleware(
    CORSMiddleware,
    allow_origins=test_settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mock database dependency
async def get_test_db():
    """Mock database session dependency."""
    async with mock_services["database"].get_session() as session:
        yield session

# Global session storage that persists between requests
class MockSessionManager:
    """Persistent session manager for testing."""

    def __init__(self):
        self.users = {}
        self.sessions = {}
        self.refresh_tokens = {}

    def clear_all(self):
        """Clear all sessions and users."""
        self.users.clear()
        self.sessions.clear()
        self.refresh_tokens.clear()

    def add_user(self, user_data):
        """Add a user to storage with duplicate checking."""
        user_id = user_data["user_id"]

        # Check for duplicate email
        for existing_user in self.users.values():
            if existing_user["email"] == user_data["email"]:
                return None  # Duplicate email

        # Check for duplicate username
        for existing_user in self.users.values():
            if existing_user["username"] == user_data["username"]:
                return None  # Duplicate username

        self.users[user_id] = user_data
        return user_id

    def get_user_by_email(self, email):
        """Get user by email."""
        for user in self.users.values():
            if user["email"] == email:
                return user
        return None

    def get_user_by_id(self, user_id):
        """Get user by ID."""
        return self.users.get(user_id)

    def create_session(self, user_id):
        """Create access and refresh tokens for user."""
        access_token = f"mock_access_token_{user_id}_{len(self.sessions)}"
        refresh_token = f"mock_refresh_token_{user_id}_{len(self.refresh_tokens)}"

        self.sessions[access_token] = user_id
        self.refresh_tokens[refresh_token] = user_id

        return access_token, refresh_token

    def validate_token(self, token):
        """Validate access token and return user."""
        if token in self.sessions:
            user_id = self.sessions[token]
            return self.users.get(user_id)
        return None

    def validate_refresh_token(self, token):
        """Validate refresh token and return user."""
        if token in self.refresh_tokens:
            user_id = self.refresh_tokens[token]
            return self.users.get(user_id)
        return None

    def refresh_session(self, refresh_token):
        """Create new tokens from refresh token."""
        user = self.validate_refresh_token(refresh_token)
        if user:
            # Remove old refresh token
            del self.refresh_tokens[refresh_token]
            # Create new session
            return self.create_session(user["user_id"])
        return None, None

    def revoke_session(self, token):
        """Revoke access token."""
        if token in self.sessions:
            del self.sessions[token]
            return True
        return False

# Global session manager instance
session_manager = MockSessionManager()

# Mock authentication dependency with improved validation
def get_current_user(request: Request):
    """Mock current user dependency with persistent session validation."""
    # Extract authorization header from request
    authorization = request.headers.get("Authorization")

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )

    token = authorization.replace("Bearer ", "")

    # Validate token using session manager
    user = session_manager.validate_token(token)
    if user:
        return user

    # Fallback for specific test tokens
    if token in ["mock_test_token_123", "test_token_123"]:
        return {
            "user_id": "test_user_123",
            "username": "testuser",
            "email": "test@example.com",
            "role": "user",
            "is_active": True
        }

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials"
    )

# ============================================================================
# Authentication Endpoints
# ============================================================================

@test_app.post("/api/v1/auth/register", status_code=status.HTTP_201_CREATED)
async def register_user(user_data: Dict[str, Any]):
    """Mock user registration endpoint with proper validation."""
    # Validate required fields
    required_fields = ["username", "email", "password"]
    for field in required_fields:
        if field not in user_data:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=[{"msg": f"Missing required field: {field}", "type": "missing"}]
            )

    # Mock email validation
    if "@" not in user_data["email"]:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=[{"msg": "Invalid email format", "type": "value_error"}]
        )

    # Mock password validation
    if len(user_data["password"]) < 8:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=[{"msg": "Password too weak", "type": "value_error"}]
        )

    # Create new user
    user_id = str(uuid.uuid4())
    user_response = {
        "user_id": user_id,
        "username": user_data["username"],
        "email": user_data["email"],
        "full_name": user_data.get("full_name"),
        "role": user_data.get("role", "user"),
        "is_active": user_data.get("is_active", True),  # Allow setting inactive users
        "is_verified": False,
        "created_at": datetime.utcnow().isoformat()
    }

    # Store user in session manager with duplicate checking
    result = session_manager.add_user(user_response)
    if result is None:
        # Check which field is duplicate
        existing_user = session_manager.get_user_by_email(user_data["email"])
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already taken"
            )

    return {
        "user": user_response,
        "message": "User registered successfully"
    }

@test_app.post("/api/v1/auth/login")
async def login_user(login_data: Dict[str, Any]):
    """Mock user login endpoint with proper session management."""
    # Validate required fields
    if "email" not in login_data or "password" not in login_data:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=[{"msg": "Missing email or password", "type": "missing"}]
        )

    # Find user by email using session manager
    user_found = session_manager.get_user_by_email(login_data["email"])

    # Check if user exists and password matches (mock validation)
    if user_found and login_data["password"] == "TestPassword123!":
        # Check if user is active
        if not user_found.get("is_active", True):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is inactive"
            )

        # Create session using session manager
        access_token, refresh_token = session_manager.create_session(user_found["user_id"])

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": 3600,
            "user": {
                "user_id": user_found["user_id"],
                "username": user_found["username"],
                "email": user_found["email"],
                "role": user_found["role"],
                "is_active": user_found["is_active"]
            }
        }
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )

@test_app.post("/api/v1/auth/refresh")
async def refresh_token_endpoint(refresh_data: Dict[str, Any] = None):
    """Mock token refresh endpoint with proper refresh token validation."""
    # Handle empty request body
    if not refresh_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token required"
        )

    # Extract refresh token from request body or Authorization header
    refresh_token = None

    if "refresh_token" in refresh_data:
        refresh_token = refresh_data["refresh_token"]
    elif "Authorization" in refresh_data:
        auth_header = refresh_data["Authorization"]
        if auth_header.startswith("Bearer "):
            refresh_token = auth_header.replace("Bearer ", "")

    if not refresh_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token required"
        )

    # Validate refresh token and create new session
    new_access_token, new_refresh_token = session_manager.refresh_session(refresh_token)

    if not new_access_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )

    return {
        "access_token": new_access_token,
        "refresh_token": new_refresh_token,
        "token_type": "bearer",
        "expires_in": 3600
    }

@test_app.get("/api/v1/auth/me")
async def get_current_user_info(request: Request):
    """Mock get current user endpoint with proper authorization handling."""
    # Extract authorization header
    authorization = request.headers.get("Authorization")

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )

    token = authorization.replace("Bearer ", "")

    # Validate token using session manager
    user = session_manager.validate_token(token)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )

    return {"user": user}

@test_app.post("/api/v1/auth/logout")
async def logout_user(logout_data: Dict[str, Any] = None):
    """Mock user logout endpoint with session revocation."""
    # Handle empty request body
    if not logout_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )

    # Extract authorization from request body
    authorization = logout_data.get("authorization")

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )

    token = authorization.replace("Bearer ", "")

    # Revoke session
    revoked = session_manager.revoke_session(token)

    if not revoked:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

    return {
        "message": "Logged out successfully",
        "logged_out_at": datetime.utcnow().isoformat()
    }

@test_app.post("/api/v1/auth/password/reset")
async def request_password_reset(reset_data: Dict[str, Any]):
    """Mock password reset request endpoint."""
    return {"message": "Password reset email sent"}

@test_app.post("/api/v1/auth/password/change")
async def change_password(change_data: Dict[str, Any]):
    """Mock password change endpoint with proper authorization."""
    # Extract authorization from request body
    authorization = change_data.get("authorization")

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )

    token = authorization.replace("Bearer ", "")

    # Validate token using session manager
    user = session_manager.validate_token(token)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )

    # Validate current password
    if change_data.get("current_password") != "TestPassword123!":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect"
        )

    return {"message": "Password changed successfully"}

# ============================================================================
# Chunking Endpoints
# ============================================================================

@test_app.get("/api/v1/chunk/health")
async def chunking_health_check():
    """Mock chunking health check endpoint."""
    return {
        "status": "healthy",
        "service": "chunking",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0-test"
    }

@test_app.get("/api/v1/chunk/supported-languages")
async def get_supported_languages():
    """Mock supported languages endpoint."""
    languages = mock_services["chunking"].get_supported_languages()
    return {
        "languages": languages,
        "total_count": len(languages)
    }

@test_app.post("/api/v1/chunk")
async def chunk_code(
    chunk_request: Dict[str, Any],
    request: Request
):
    """Mock code chunking endpoint."""
    # Get current user from request (authentication required)
    current_user = get_current_user(request)

    # Validate required fields
    required_fields = ["content", "language", "file_path", "purpose"]
    for field in required_fields:
        if field not in chunk_request:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Missing required field: {field}"
            )

    # Check supported language
    supported_languages = [lang["name"] for lang in mock_services["chunking"].get_supported_languages()]
    if chunk_request["language"] not in supported_languages:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported language: {chunk_request['language']}"
        )

    # Perform chunking
    result = await mock_services["chunking"].chunk_code(
        content=chunk_request["content"],
        language=chunk_request["language"],
        **chunk_request.get("options", {})
    )

    # Add request metadata
    result["request_id"] = str(uuid.uuid4())
    result["metadata"].update({
        "file_path": chunk_request["file_path"],
        "purpose": chunk_request["purpose"],
        "user_id": current_user["user_id"],
        "timestamp": datetime.utcnow().isoformat()
    })

    return result

@test_app.get("/api/v1/chunk/stats")
async def get_chunking_stats():
    """Mock chunking statistics endpoint."""
    return {
        "performance": {
            "avg_processing_time": 0.05,
            "total_requests": 1000,
            "success_rate": 99.5
        },
        "usage": {
            "total_chunks_created": 50000,
            "languages_processed": ["python", "javascript", "java", "cpp", "go"],
            "avg_chunks_per_file": 25
        },
        "timestamp": datetime.utcnow().isoformat()
    }

@test_app.post("/api/v1/chunk/test")
async def test_chunking():
    """Mock chunking test endpoint."""
    # Test with sample code
    test_result = await mock_services["chunking"].chunk_code(
        content=SAMPLE_PYTHON_CODE,
        language="python"
    )

    return {
        "status": "success",
        "test_results": {
            "chunks_created": len(test_result["chunks"]),
            "processing_time": test_result["metadata"]["processing_time"],
            "language": "python"
        }
    }

# ============================================================================
# System Endpoints
# ============================================================================

@test_app.get("/api/v1/system/health")
async def system_health_check():
    """Mock system health check endpoint."""
    return {
        "status": "healthy",
        "services": {
            "database": "healthy",
            "chunking": "healthy",
            "embedding": "healthy",
            "vectordb": "healthy"
        },
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0-test"
    }

# ============================================================================
# Error Handlers
# ============================================================================

@test_app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

@test_app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error"}
    )

# ============================================================================
# Test Session Management
# ============================================================================

@test_app.post("/api/v1/test/clear-sessions")
async def clear_test_sessions():
    """Clear all test sessions - for testing only."""
    session_manager.clear_all()
    return {"message": "All sessions cleared"}

# ============================================================================
# Startup/Shutdown Events
# ============================================================================

@test_app.on_event("startup")
async def startup_event():
    """Initialize test services on startup."""
    await mock_services["database"].initialize()

@test_app.on_event("shutdown")
async def shutdown_event():
    """Cleanup test services on shutdown."""
    await mock_services["database"].cleanup()

# Export the test app
__all__ = ["test_app"]
