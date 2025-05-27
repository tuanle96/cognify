"""
Comprehensive unit tests for Authentication endpoints.

Tests all authentication-related functionality including registration,
login, logout, token management, and user profile operations.
"""

import pytest
from fastapi import status


class TestUserRegistration:
    """Test user registration functionality."""

    def test_register_user_success(self, mock_client, sample_user_data):
        """Test successful user registration."""
        response = mock_client.post("/api/v1/auth/register", json=sample_user_data)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Verify user data
        assert data["email"] == sample_user_data["email"]
        assert data["full_name"] == sample_user_data["full_name"]
        assert data["username"] == sample_user_data["username"]
        assert data["user_id"] == "test_user_123"
        
        # Verify default values
        assert data["is_active"] is True
        assert data["is_verified"] is False
        assert data["role"] == "user"
        assert data["created_at"] == "2024-01-01T00:00:00Z"
        assert data["last_login"] is None
        assert data["profile_picture"] is None

    def test_register_user_minimal_data(self, mock_client):
        """Test registration with minimal required data."""
        minimal_data = {
            "email": "minimal@example.com",
            "password": "MinimalPass123"
        }
        
        response = mock_client.post("/api/v1/auth/register", json=minimal_data)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["email"] == minimal_data["email"]
        assert data["user_id"] == "test_user_123"

    def test_register_user_with_custom_username(self, mock_client):
        """Test registration with custom username."""
        user_data = {
            "email": "custom@example.com",
            "password": "CustomPass123",
            "username": "customuser",
            "full_name": "Custom User"
        }
        
        response = mock_client.post("/api/v1/auth/register", json=user_data)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["username"] == "customuser"
        assert data["full_name"] == "Custom User"


class TestUserLogin:
    """Test user login functionality."""

    def test_login_success(self, mock_client):
        """Test successful user login."""
        login_data = {
            "email": "test@example.com",
            "password": "TestPassword123"
        }
        
        response = mock_client.post("/api/v1/auth/login", json=login_data)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Verify tokens
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
        assert data["expires_in"] == 3600
        
        # Verify user info
        assert "user" in data
        user = data["user"]
        assert user["email"] == login_data["email"]
        assert user["user_id"] == "test_user_123"
        assert user["is_active"] is True
        assert user["is_verified"] is True

    def test_login_with_username(self, mock_client):
        """Test login using username instead of email."""
        login_data = {
            "username": "testuser",
            "password": "TestPassword123"
        }
        
        response = mock_client.post("/api/v1/auth/login", json=login_data)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "access_token" in data
        assert data["user"]["username"] == "testuser"

    def test_login_response_structure(self, mock_client):
        """Test login response has correct structure."""
        login_data = {
            "email": "test@example.com",
            "password": "TestPassword123"
        }
        
        response = mock_client.post("/api/v1/auth/login", json=login_data)
        data = response.json()
        
        # Required fields
        required_fields = ["access_token", "refresh_token", "token_type", "expires_in", "user"]
        for field in required_fields:
            assert field in data
        
        # User object structure
        user_fields = ["user_id", "email", "full_name", "username", "is_active", "is_verified", "role"]
        for field in user_fields:
            assert field in data["user"]


class TestTokenManagement:
    """Test token refresh and management."""

    def test_refresh_token_success(self, mock_client):
        """Test successful token refresh."""
        response = mock_client.post("/api/v1/auth/refresh")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
        assert data["expires_in"] == 3600
        
        # Verify new tokens are different from mock login tokens
        assert data["access_token"] == "new_mock_access_token_def456"
        assert data["refresh_token"] == "new_mock_refresh_token_uvw012"

    def test_refresh_token_structure(self, mock_client):
        """Test refresh token response structure."""
        response = mock_client.post("/api/v1/auth/refresh")
        data = response.json()
        
        required_fields = ["access_token", "refresh_token", "token_type", "expires_in"]
        for field in required_fields:
            assert field in data


class TestUserLogout:
    """Test user logout functionality."""

    def test_logout_success(self, mock_client, auth_headers):
        """Test successful user logout."""
        response = mock_client.post("/api/v1/auth/logout", headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["message"] == "Logged out successfully"
        assert "logged_out_at" in data
        assert data["logged_out_at"] == "2024-01-01T12:30:00Z"

    def test_logout_without_auth(self, mock_client):
        """Test logout without authentication headers."""
        response = mock_client.post("/api/v1/auth/logout")
        
        # Mock implementation returns success regardless
        assert response.status_code == status.HTTP_200_OK


class TestUserProfile:
    """Test user profile management."""

    def test_get_current_user(self, mock_client, auth_headers):
        """Test getting current user profile."""
        response = mock_client.get("/api/v1/auth/me", headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Verify user data
        assert data["user_id"] == "test_user_123"
        assert data["email"] == "test@example.com"
        assert data["full_name"] == "Test User"
        assert data["username"] == "testuser"
        assert data["is_active"] is True
        assert data["is_verified"] is True
        assert data["role"] == "user"
        
        # Verify timestamps
        assert data["created_at"] == "2024-01-01T00:00:00Z"
        assert data["last_login"] == "2024-01-01T12:00:00Z"
        
        # Verify preferences
        assert "preferences" in data
        preferences = data["preferences"]
        assert preferences["theme"] == "light"
        assert preferences["language"] == "en"
        assert preferences["notifications"] is True

    def test_get_current_user_structure(self, mock_client, auth_headers):
        """Test current user response structure."""
        response = mock_client.get("/api/v1/auth/me", headers=auth_headers)
        data = response.json()
        
        required_fields = [
            "user_id", "email", "full_name", "username", "is_active", 
            "is_verified", "role", "created_at", "last_login", "preferences"
        ]
        for field in required_fields:
            assert field in data


class TestPasswordManagement:
    """Test password reset and change functionality."""

    def test_reset_password_request(self, mock_client):
        """Test password reset request."""
        reset_data = {
            "email": "test@example.com"
        }
        
        response = mock_client.post("/api/v1/auth/password/reset", json=reset_data)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["message"] == "If the email exists, a reset link has been sent"
        assert data["email"] == reset_data["email"]
        assert data["reset_token_sent"] is True
        assert data["expires_in"] == 3600

    def test_change_password(self, mock_client, auth_headers):
        """Test password change."""
        change_data = {
            "current_password": "OldPassword123",
            "new_password": "NewPassword456"
        }
        
        response = mock_client.post("/api/v1/auth/password/change", 
                                  json=change_data, headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["message"] == "Password changed successfully"
        assert "changed_at" in data
        assert data["requires_relogin"] is True


class TestUserStatistics:
    """Test user statistics and analytics."""

    def test_get_user_stats(self, mock_client, auth_headers):
        """Test getting user statistics."""
        response = mock_client.get("/api/v1/auth/stats", headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Verify statistics
        assert data["total_documents"] == 15
        assert data["total_collections"] == 3
        assert data["total_queries"] == 127
        assert data["storage_used"] == "2.5 MB"
        assert data["account_age_days"] == 30
        
        # Verify activity summary
        assert "activity_summary" in data
        activity = data["activity_summary"]
        assert activity["documents_this_week"] == 5
        assert activity["queries_this_week"] == 23
        assert activity["collections_created"] == 1
