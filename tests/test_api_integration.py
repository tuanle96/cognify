"""
Integration tests for API endpoints.
"""
import pytest
import asyncio
from httpx import AsyncClient
from fastapi.testclient import TestClient

# Use mock app for integration tests
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from test_main import test_app as app
from app.core.config import get_settings


@pytest.fixture(scope="function")
def event_loop():
    """Create an instance of the default event loop for each test function."""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


@pytest.fixture
def test_client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
async def async_client():
    """Create async test client."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_health_check(self, test_client):
        """Test basic health check."""
        response = test_client.get("/api/v1/system/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "services" in data

    def test_system_status(self, test_client):
        """Test system status endpoint."""
        response = test_client.get("/api/v1/system/status")
        assert response.status_code == 200

        data = response.json()
        assert "system" in data
        assert "services" in data
        assert "version" in data

    def test_metrics_endpoint(self, test_client):
        """Test metrics endpoint."""
        response = test_client.get("/api/v1/system/metrics")
        assert response.status_code == 200

        data = response.json()
        assert "performance" in data
        assert "usage" in data


class TestAuthenticationEndpoints:
    """Test authentication endpoints."""

    def test_register_user(self, test_client):
        """Test user registration."""
        user_data = {
            "username": "testuser",
            "email": "test@example.com",
            "password": "testpassword123"
        }

        response = test_client.post("/api/v1/auth/register", json=user_data)
        assert response.status_code == 201

        data = response.json()
        assert "user" in data
        assert data["user"]["username"] == "testuser"
        assert data["user"]["email"] == "test@example.com"
        assert "access_token" in data

    def test_login_user(self, test_client):
        """Test user login."""
        # First register a user
        user_data = {
            "username": "loginuser",
            "email": "login@example.com",
            "password": "loginpassword123"
        }
        test_client.post("/api/v1/auth/register", json=user_data)

        # Then login
        login_data = {
            "username": "loginuser",
            "password": "loginpassword123"
        }

        response = test_client.post("/api/v1/auth/login", json=login_data)
        assert response.status_code == 200

        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert "user" in data

    def test_refresh_token(self, test_client):
        """Test token refresh."""
        # Register and login
        user_data = {
            "username": "refreshuser",
            "email": "refresh@example.com",
            "password": "refreshpassword123"
        }
        test_client.post("/api/v1/auth/register", json=user_data)

        login_data = {
            "username": "refreshuser",
            "password": "refreshpassword123"
        }
        login_response = test_client.post("/api/v1/auth/login", json=login_data)
        refresh_token = login_response.json()["refresh_token"]

        # Refresh token
        refresh_data = {"refresh_token": refresh_token}
        response = test_client.post("/api/v1/auth/refresh", json=refresh_data)
        assert response.status_code == 200

        data = response.json()
        assert "access_token" in data


class TestDocumentEndpoints:
    """Test document management endpoints."""

    @pytest.fixture
    def auth_headers(self, test_client):
        """Get authentication headers."""
        # Register and login
        user_data = {
            "username": "docuser",
            "email": "doc@example.com",
            "password": "docpassword123"
        }
        test_client.post("/api/v1/auth/register", json=user_data)

        login_data = {
            "username": "docuser",
            "password": "docpassword123"
        }
        login_response = test_client.post("/api/v1/auth/login", json=login_data)
        access_token = login_response.json()["access_token"]

        return {"Authorization": f"Bearer {access_token}"}

    def test_upload_document(self, test_client, auth_headers):
        """Test document upload."""
        document_data = {
            "title": "Test Document",
            "content": "This is test content for the document.",
            "file_type": "text"
        }

        response = test_client.post(
            "/api/v1/documents/upload",
            json=document_data,
            headers=auth_headers
        )
        assert response.status_code == 201

        data = response.json()
        assert "document" in data
        assert data["document"]["title"] == "Test Document"
        assert "id" in data["document"]

    def test_get_document(self, test_client, auth_headers):
        """Test getting a document."""
        # First upload a document
        document_data = {
            "title": "Get Test Document",
            "content": "Content for get test.",
            "file_type": "text"
        }

        upload_response = test_client.post(
            "/api/v1/documents/upload",
            json=document_data,
            headers=auth_headers
        )
        document_id = upload_response.json()["document"]["id"]

        # Get the document
        response = test_client.get(
            f"/api/v1/documents/{document_id}",
            headers=auth_headers
        )
        assert response.status_code == 200

        data = response.json()
        assert data["title"] == "Get Test Document"
        assert data["content"] == "Content for get test."

    def test_list_documents(self, test_client, auth_headers):
        """Test listing documents."""
        # Upload multiple documents
        for i in range(3):
            document_data = {
                "title": f"List Document {i}",
                "content": f"Content for document {i}.",
                "file_type": "text"
            }
            test_client.post(
                "/api/v1/documents/upload",
                json=document_data,
                headers=auth_headers
            )

        # List documents
        response = test_client.get(
            "/api/v1/documents/",
            headers=auth_headers
        )
        assert response.status_code == 200

        data = response.json()
        assert "documents" in data
        assert len(data["documents"]) >= 3

    def test_batch_upload(self, test_client, auth_headers):
        """Test batch document upload."""
        documents_data = {
            "documents": [
                {
                    "title": "Batch Document 1",
                    "content": "Content 1",
                    "file_type": "text"
                },
                {
                    "title": "Batch Document 2",
                    "content": "Content 2",
                    "file_type": "text"
                }
            ]
        }

        response = test_client.post(
            "/api/v1/documents/batch-upload",
            json=documents_data,
            headers=auth_headers
        )
        assert response.status_code == 201

        data = response.json()
        assert "documents" in data
        assert len(data["documents"]) == 2


class TestQueryEndpoints:
    """Test query and search endpoints."""

    @pytest.fixture
    def auth_headers_with_docs(self, test_client):
        """Get authentication headers and upload test documents."""
        # Register and login
        user_data = {
            "username": "queryuser",
            "email": "query@example.com",
            "password": "querypassword123"
        }
        test_client.post("/api/v1/auth/register", json=user_data)

        login_data = {
            "username": "queryuser",
            "password": "querypassword123"
        }
        login_response = test_client.post("/api/v1/auth/login", json=login_data)
        access_token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {access_token}"}

        # Upload test documents
        test_docs = [
            {
                "title": "Python Functions",
                "content": "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
                "file_type": "python"
            },
            {
                "title": "JavaScript Async",
                "content": "async function fetchData() { const response = await fetch('/api/data'); return response.json(); }",
                "file_type": "javascript"
            }
        ]

        for doc in test_docs:
            test_client.post("/api/v1/documents/upload", json=doc, headers=headers)

        return headers

    def test_search_query(self, test_client, auth_headers_with_docs):
        """Test search functionality."""
        query_data = {
            "query": "fibonacci function",
            "search_type": "semantic",
            "limit": 10
        }

        response = test_client.post(
            "/api/v1/query/search",
            json=query_data,
            headers=auth_headers_with_docs
        )
        assert response.status_code == 200

        data = response.json()
        assert "results" in data
        assert "query_id" in data
        assert len(data["results"]) > 0

    def test_similar_documents(self, test_client, auth_headers_with_docs):
        """Test finding similar documents."""
        # First get a document ID
        docs_response = test_client.get(
            "/api/v1/documents/",
            headers=auth_headers_with_docs
        )
        document_id = docs_response.json()["documents"][0]["id"]

        # Find similar documents
        response = test_client.post(
            "/api/v1/query/similar",
            json={"document_id": document_id, "limit": 5},
            headers=auth_headers_with_docs
        )
        assert response.status_code == 200

        data = response.json()
        assert "results" in data
        assert "document_id" in data

    def test_query_suggestions(self, test_client, auth_headers_with_docs):
        """Test query suggestions."""
        response = test_client.get(
            "/api/v1/query/suggestions?partial=python func",
            headers=auth_headers_with_docs
        )
        assert response.status_code == 200

        data = response.json()
        assert "suggestions" in data
        assert isinstance(data["suggestions"], list)


class TestCollectionEndpoints:
    """Test collection management endpoints."""

    @pytest.fixture
    def auth_headers(self, test_client):
        """Get authentication headers."""
        user_data = {
            "username": "colluser",
            "email": "coll@example.com",
            "password": "collpassword123"
        }
        test_client.post("/api/v1/auth/register", json=user_data)

        login_data = {
            "username": "colluser",
            "password": "collpassword123"
        }
        login_response = test_client.post("/api/v1/auth/login", json=login_data)
        access_token = login_response.json()["access_token"]

        return {"Authorization": f"Bearer {access_token}"}

    def test_create_collection(self, test_client, auth_headers):
        """Test creating a collection."""
        collection_data = {
            "name": "My Test Collection",
            "description": "A collection for testing",
            "is_public": False
        }

        response = test_client.post(
            "/api/v1/collections/",
            json=collection_data,
            headers=auth_headers
        )
        assert response.status_code == 201

        data = response.json()
        assert "collection" in data
        assert data["collection"]["name"] == "My Test Collection"
        assert "id" in data["collection"]

    def test_get_collection(self, test_client, auth_headers):
        """Test getting a collection."""
        # Create collection first
        collection_data = {
            "name": "Get Test Collection",
            "description": "Collection for get test"
        }

        create_response = test_client.post(
            "/api/v1/collections/",
            json=collection_data,
            headers=auth_headers
        )
        collection_id = create_response.json()["collection"]["id"]

        # Get the collection
        response = test_client.get(
            f"/api/v1/collections/{collection_id}",
            headers=auth_headers
        )
        assert response.status_code == 200

        data = response.json()
        assert data["name"] == "Get Test Collection"

    def test_list_collections(self, test_client, auth_headers):
        """Test listing collections."""
        # Create multiple collections
        for i in range(3):
            collection_data = {
                "name": f"List Collection {i}",
                "description": f"Description {i}"
            }
            test_client.post(
                "/api/v1/collections/",
                json=collection_data,
                headers=auth_headers
            )

        # List collections
        response = test_client.get(
            "/api/v1/collections/",
            headers=auth_headers
        )
        assert response.status_code == 200

        data = response.json()
        assert "collections" in data
        assert len(data["collections"]) >= 3
