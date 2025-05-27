"""
Comprehensive API Integration Tests for Authentication Endpoints.

This module provides complete test coverage for all authentication-related
API endpoints to achieve 60%+ API coverage target.
"""

import pytest
import asyncio
from fastapi import status
from fastapi.testclient import TestClient
from httpx import AsyncClient
import json
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Import the test app
try:
    from tests.test_app import test_app as app
    APP_AVAILABLE = True
except ImportError:
    # Fallback to main app if test app not available
    try:
        from app.main import app
        APP_AVAILABLE = True
    except ImportError:
        APP_AVAILABLE = False


@pytest.fixture
def test_client():
    """Create test client for API testing."""
    if not APP_AVAILABLE:
        pytest.skip("App not available for testing")
    return TestClient(app)


@pytest.fixture
async def async_client():
    """Create async test client for API testing."""
    if not APP_AVAILABLE:
        pytest.skip("App not available for testing")
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture
def sample_user_data():
    """Sample user data for testing."""
    return {
        "username": "testuser",
        "email": "test@example.com",
        "password": "TestPassword123!",
        "full_name": "Test User",
        "role": "user"
    }


@pytest.fixture
def sample_admin_data():
    """Sample admin data for testing."""
    return {
        "username": "admin",
        "email": "admin@example.com",
        "password": "AdminPassword123!",
        "full_name": "Admin User",
        "role": "admin"
    }


class TestUserRegistration:
    """Comprehensive tests for user registration endpoint."""

    def test_register_user_success(self, test_client, sample_user_data):
        """Test successful user registration."""
        response = test_client.post("/api/v1/auth/register", json=sample_user_data)

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()

        # Verify response structure
        assert "user" in data
        assert "message" in data

        # Verify user data
        user = data["user"]
        assert user["email"] == sample_user_data["email"]
        assert user["username"] == sample_user_data["username"]
        assert user["full_name"] == sample_user_data["full_name"]
        assert user["role"] == sample_user_data["role"]
        assert user["is_active"] is True
        assert "user_id" in user
        assert "created_at" in user

        # Verify password is not returned
        assert "password" not in user
        assert "hashed_password" not in user

    def test_register_user_duplicate_email(self, test_client, sample_user_data):
        """Test registration with duplicate email."""
        # First registration
        test_client.post("/api/v1/auth/register", json=sample_user_data)

        # Second registration with same email
        response = test_client.post("/api/v1/auth/register", json=sample_user_data)

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        assert "detail" in data
        assert "email" in data["detail"].lower()

    def test_register_user_invalid_email(self, test_client, sample_user_data):
        """Test registration with invalid email format."""
        sample_user_data["email"] = "invalid-email"

        response = test_client.post("/api/v1/auth/register", json=sample_user_data)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        data = response.json()
        # Check for either 'detail' or 'error' field depending on implementation
        assert "detail" in data or "error" in data

    def test_register_user_weak_password(self, test_client, sample_user_data):
        """Test registration with weak password."""
        sample_user_data["password"] = "123"

        response = test_client.post("/api/v1/auth/register", json=sample_user_data)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        data = response.json()
        assert "detail" in data or "error" in data

    def test_register_user_missing_fields(self, test_client):
        """Test registration with missing required fields."""
        incomplete_data = {"email": "test@example.com"}

        response = test_client.post("/api/v1/auth/register", json=incomplete_data)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        data = response.json()
        assert "detail" in data or "error" in data

    def test_register_admin_user(self, test_client, sample_admin_data):
        """Test admin user registration."""
        response = test_client.post("/api/v1/auth/register", json=sample_admin_data)

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()

        user = data["user"]
        assert user["role"] == "admin"
        assert user["email"] == sample_admin_data["email"]


class TestUserLogin:
    """Comprehensive tests for user login endpoint."""

    def test_login_success(self, test_client, sample_user_data):
        """Test successful user login."""
        # Register user first
        test_client.post("/api/v1/auth/register", json=sample_user_data)

        # Login
        login_data = {
            "email": sample_user_data["email"],
            "password": sample_user_data["password"]
        }
        response = test_client.post("/api/v1/auth/login", json=login_data)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Verify tokens
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
        assert "expires_in" in data

        # Verify user info
        assert "user" in data
        user = data["user"]
        assert user["email"] == sample_user_data["email"]
        assert user["is_active"] is True

    def test_login_invalid_credentials(self, test_client, sample_user_data):
        """Test login with invalid credentials."""
        # Register user first
        test_client.post("/api/v1/auth/register", json=sample_user_data)

        # Login with wrong password
        login_data = {
            "email": sample_user_data["email"],
            "password": "WrongPassword123!"
        }
        response = test_client.post("/api/v1/auth/login", json=login_data)

        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        data = response.json()
        assert "detail" in data

    def test_login_nonexistent_user(self, test_client):
        """Test login with non-existent user."""
        login_data = {
            "email": "nonexistent@example.com",
            "password": "Password123!"
        }
        response = test_client.post("/api/v1/auth/login", json=login_data)

        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        data = response.json()
        assert "detail" in data

    def test_login_inactive_user(self, test_client, sample_user_data):
        """Test login with inactive user account."""
        # Create inactive user data
        inactive_user_data = sample_user_data.copy()
        inactive_user_data["email"] = "inactive@example.com"
        inactive_user_data["username"] = "inactiveuser"
        inactive_user_data["is_active"] = False

        # Register inactive user
        test_client.post("/api/v1/auth/register", json=inactive_user_data)

        # Try to login with inactive user
        login_data = {
            "email": inactive_user_data["email"],
            "password": inactive_user_data["password"]
        }
        response = test_client.post("/api/v1/auth/login", json=login_data)

        # Should return 403 for inactive user
        assert response.status_code in [status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN]

    def test_login_missing_fields(self, test_client):
        """Test login with missing fields."""
        incomplete_data = {"email": "test@example.com"}

        response = test_client.post("/api/v1/auth/login", json=incomplete_data)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        data = response.json()
        assert "detail" in data or "error" in data


class TestTokenManagement:
    """Comprehensive tests for token management endpoints."""

    def test_token_refresh_success(self, test_client, sample_user_data):
        """Test successful token refresh."""
        # Register and login user
        test_client.post("/api/v1/auth/register", json=sample_user_data)
        login_response = test_client.post("/api/v1/auth/login", json={
            "email": sample_user_data["email"],
            "password": sample_user_data["password"]
        })

        tokens = login_response.json()
        refresh_token = tokens["refresh_token"]

        # Refresh token using request body
        refresh_data = {"refresh_token": refresh_token}
        response = test_client.post("/api/v1/auth/refresh", json=refresh_data)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
        assert "expires_in" in data

    def test_token_refresh_invalid_token(self, test_client):
        """Test token refresh with invalid token."""
        headers = {"Authorization": "Bearer invalid_token"}
        response = test_client.post("/api/v1/auth/refresh", headers=headers)

        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        data = response.json()
        assert "detail" in data

    def test_token_refresh_no_token(self, test_client):
        """Test token refresh without token."""
        response = test_client.post("/api/v1/auth/refresh")

        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        data = response.json()
        assert "detail" in data


class TestUserProfile:
    """Comprehensive tests for user profile endpoints."""

    def test_get_current_user_success(self, test_client, sample_user_data):
        """Test getting current user profile."""
        # Register and login user
        test_client.post("/api/v1/auth/register", json=sample_user_data)
        login_response = test_client.post("/api/v1/auth/login", json={
            "email": sample_user_data["email"],
            "password": sample_user_data["password"]
        })

        tokens = login_response.json()
        access_token = tokens["access_token"]

        # Get current user
        headers = {"Authorization": f"Bearer {access_token}"}
        response = test_client.get("/api/v1/auth/me", headers=headers)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert "user" in data
        user = data["user"]
        assert user["email"] == sample_user_data["email"]
        assert user["username"] == sample_user_data["username"]
        assert "user_id" in user
        assert "created_at" in user

    def test_get_current_user_unauthorized(self, test_client):
        """Test getting current user without authentication."""
        response = test_client.get("/api/v1/auth/me")

        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        data = response.json()
        assert "detail" in data

    def test_get_current_user_invalid_token(self, test_client):
        """Test getting current user with invalid token."""
        headers = {"Authorization": "Bearer invalid_token"}
        response = test_client.get("/api/v1/auth/me", headers=headers)

        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        data = response.json()
        assert "detail" in data


class TestUserLogout:
    """Comprehensive tests for user logout endpoint."""

    def test_logout_success(self, test_client, sample_user_data):
        """Test successful user logout."""
        # Register and login user
        test_client.post("/api/v1/auth/register", json=sample_user_data)
        login_response = test_client.post("/api/v1/auth/login", json={
            "email": sample_user_data["email"],
            "password": sample_user_data["password"]
        })

        tokens = login_response.json()
        access_token = tokens["access_token"]

        # Logout using authorization parameter
        logout_data = {"authorization": f"Bearer {access_token}"}
        response = test_client.post("/api/v1/auth/logout", json=logout_data)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "message" in data
        assert "logged_out_at" in data

    def test_logout_without_auth(self, test_client):
        """Test logout without authentication."""
        response = test_client.post("/api/v1/auth/logout")

        # Should require authentication
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        data = response.json()
        assert "detail" in data


class TestPasswordManagement:
    """Comprehensive tests for password management endpoints."""

    def test_password_reset_request(self, test_client, sample_user_data):
        """Test password reset request."""
        # Register user first
        test_client.post("/api/v1/auth/register", json=sample_user_data)

        # Request password reset
        reset_data = {"email": sample_user_data["email"]}
        response = test_client.post("/api/v1/auth/password/reset", json=reset_data)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "message" in data

    def test_password_change_success(self, test_client, sample_user_data):
        """Test successful password change."""
        # Register and login user
        test_client.post("/api/v1/auth/register", json=sample_user_data)
        login_response = test_client.post("/api/v1/auth/login", json={
            "email": sample_user_data["email"],
            "password": sample_user_data["password"]
        })

        tokens = login_response.json()
        access_token = tokens["access_token"]

        # Change password using authorization in request body
        change_data = {
            "current_password": sample_user_data["password"],
            "new_password": "NewPassword123!",
            "authorization": f"Bearer {access_token}"
        }
        response = test_client.post("/api/v1/auth/password/change", json=change_data)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "message" in data

    def test_password_change_wrong_current(self, test_client, sample_user_data):
        """Test password change with wrong current password."""
        # Register and login user
        test_client.post("/api/v1/auth/register", json=sample_user_data)
        login_response = test_client.post("/api/v1/auth/login", json={
            "email": sample_user_data["email"],
            "password": sample_user_data["password"]
        })

        tokens = login_response.json()
        access_token = tokens["access_token"]

        # Change password with wrong current password using authorization in request body
        change_data = {
            "current_password": "WrongPassword123!",
            "new_password": "NewPassword123!",
            "authorization": f"Bearer {access_token}"
        }
        response = test_client.post("/api/v1/auth/password/change", json=change_data)

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        assert "detail" in data


@pytest.mark.asyncio
class TestAsyncAuthEndpoints:
    """Async tests for authentication endpoints."""

    async def test_async_user_registration(self, async_client, sample_user_data):
        """Test async user registration."""
        # Create unique user data for async test
        async_user_data = sample_user_data.copy()
        async_user_data["email"] = "async@example.com"
        async_user_data["username"] = "asyncuser"

        response = await async_client.post("/api/v1/auth/register", json=async_user_data)

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert "user" in data
        assert data["user"]["email"] == async_user_data["email"]

    async def test_async_user_login(self, async_client, sample_user_data):
        """Test async user login."""
        # Register user first
        await async_client.post("/api/v1/auth/register", json=sample_user_data)

        # Login
        login_data = {
            "email": sample_user_data["email"],
            "password": sample_user_data["password"]
        }
        response = await async_client.post("/api/v1/auth/login", json=login_data)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "access_token" in data
        assert "user" in data

    async def test_concurrent_registrations(self, async_client):
        """Test concurrent user registrations."""
        users = [
            {"username": f"user{i}", "email": f"user{i}@example.com",
             "password": "Password123!", "full_name": f"User {i}", "role": "user"}
            for i in range(5)
        ]

        # Register users concurrently
        tasks = [async_client.post("/api/v1/auth/register", json=user) for user in users]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Check that all registrations succeeded
        success_count = 0
        for response in responses:
            if hasattr(response, 'status_code') and response.status_code == status.HTTP_201_CREATED:
                success_count += 1

        assert success_count == len(users)
