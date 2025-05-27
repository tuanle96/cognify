"""
Comprehensive API Integration Tests for Chunking Endpoints.

This module provides complete test coverage for all chunking-related
API endpoints to achieve 60%+ API coverage target.
"""

import pytest
import asyncio
import time
from fastapi import status
from fastapi.testclient import TestClient
from httpx import AsyncClient
import json
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
def sample_python_code():
    """Sample Python code for testing."""
    return '''
def fibonacci(n):
    """Calculate fibonacci number recursively."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

class Calculator:
    """Simple calculator class."""

    def __init__(self):
        self.history = []

    def add(self, a, b):
        """Add two numbers."""
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result

    def multiply(self, a, b):
        """Multiply two numbers."""
        result = a * b
        self.history.append(f"{a} * {b} = {result}")
        return result

    def get_history(self):
        """Get calculation history."""
        return self.history

def main():
    """Main function."""
    calc = Calculator()
    print(f"5 + 3 = {calc.add(5, 3)}")
    print(f"4 * 6 = {calc.multiply(4, 6)}")
    print(f"Fibonacci(10) = {fibonacci(10)}")
    print(f"History: {calc.get_history()}")

if __name__ == "__main__":
    main()
'''


@pytest.fixture
def sample_javascript_code():
    """Sample JavaScript code for testing."""
    return '''
function fibonacci(n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

class Calculator {
    constructor() {
        this.history = [];
    }

    add(a, b) {
        const result = a + b;
        this.history.push(`${a} + ${b} = ${result}`);
        return result;
    }

    multiply(a, b) {
        const result = a * b;
        this.history.push(`${a} * ${b} = ${result}`);
        return result;
    }

    getHistory() {
        return this.history;
    }
}

function main() {
    const calc = new Calculator();
    console.log(`5 + 3 = ${calc.add(5, 3)}`);
    console.log(`4 * 6 = ${calc.multiply(4, 6)}`);
    console.log(`Fibonacci(10) = ${fibonacci(10)}`);
    console.log(`History: ${calc.getHistory()}`);
}

main();
'''


@pytest.fixture
def auth_headers():
    """Mock authentication headers."""
    return {
        "Authorization": "Bearer mock_test_token_123",
        "Content-Type": "application/json"
    }


class TestChunkingHealthEndpoints:
    """Tests for chunking health check endpoints."""

    def test_chunking_health_check(self, test_client):
        """Test chunking service health check."""
        response = test_client.get("/api/v1/chunk/health")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Verify health check structure
        assert "status" in data
        assert "service" in data
        assert "timestamp" in data
        assert "version" in data

        # Verify service is healthy
        assert data["status"] in ["healthy", "degraded"]
        assert data["service"] == "chunking"

    def test_chunking_service_status(self, test_client):
        """Test chunking service detailed status."""
        response = test_client.get("/api/v1/chunk/health")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Verify detailed status information
        if "details" in data:
            details = data["details"]
            assert "agents" in details or "services" in details
            assert "performance" in details or "metrics" in details


class TestSupportedLanguages:
    """Tests for supported languages endpoint."""

    def test_get_supported_languages(self, test_client):
        """Test getting supported programming languages."""
        response = test_client.get("/api/v1/chunk/supported-languages")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Verify response structure
        assert "languages" in data
        assert "total_count" in data

        # Verify languages list
        languages = data["languages"]
        assert isinstance(languages, list)
        assert len(languages) > 0

        # Verify common languages are supported
        language_names = [lang["name"] for lang in languages]
        expected_languages = ["python", "javascript", "java", "cpp", "go"]

        for lang in expected_languages:
            assert lang in language_names

    def test_supported_languages_structure(self, test_client):
        """Test supported languages response structure."""
        response = test_client.get("/api/v1/chunk/supported-languages")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Verify each language entry structure
        languages = data["languages"]
        for language in languages[:3]:  # Check first 3 languages
            assert "name" in language
            assert "extensions" in language
            assert "description" in language
            assert isinstance(language["extensions"], list)


class TestChunkingBasicOperations:
    """Tests for basic chunking operations."""

    def test_chunk_python_code_basic(self, test_client, sample_python_code, auth_headers):
        """Test basic Python code chunking."""
        chunk_request = {
            "content": sample_python_code,
            "language": "python",
            "file_path": "test.py",
            "purpose": "general"
        }

        response = test_client.post("/api/v1/chunk", json=chunk_request, headers=auth_headers)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Verify response structure
        assert "chunks" in data
        assert "metadata" in data
        assert "request_id" in data

        # Verify chunks
        chunks = data["chunks"]
        assert isinstance(chunks, list)
        assert len(chunks) > 0

        # Verify chunk structure
        for chunk in chunks:
            assert "content" in chunk
            assert "chunk_type" in chunk
            assert "start_line" in chunk
            assert "end_line" in chunk
            assert "metadata" in chunk

    def test_chunk_javascript_code(self, test_client, sample_javascript_code, auth_headers):
        """Test JavaScript code chunking."""
        chunk_request = {
            "content": sample_javascript_code,
            "language": "javascript",
            "file_path": "test.js",
            "purpose": "general"
        }

        response = test_client.post("/api/v1/chunk", json=chunk_request, headers=auth_headers)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Verify response
        assert "chunks" in data
        assert "metadata" in data

        # Verify metadata
        metadata = data["metadata"]
        assert metadata["language"] == "javascript"
        assert metadata["file_path"] == "test.js"
        assert metadata["purpose"] == "general"

    def test_chunk_with_different_purposes(self, test_client, sample_python_code, auth_headers):
        """Test chunking with different purposes."""
        purposes = ["general", "code_review", "bug_detection", "documentation"]

        for purpose in purposes:
            chunk_request = {
                "content": sample_python_code,
                "language": "python",
                "file_path": "test.py",
                "purpose": purpose
            }

            response = test_client.post("/api/v1/chunk", json=chunk_request, headers=auth_headers)

            assert response.status_code == status.HTTP_200_OK
            data = response.json()

            # Verify purpose is reflected in metadata
            assert data["metadata"]["purpose"] == purpose

    def test_chunk_empty_content(self, test_client, auth_headers):
        """Test chunking with empty content."""
        chunk_request = {
            "content": "",
            "language": "python",
            "file_path": "empty.py",
            "purpose": "general"
        }

        response = test_client.post("/api/v1/chunk", json=chunk_request, headers=auth_headers)

        # Should handle empty content gracefully
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_400_BAD_REQUEST]

        if response.status_code == status.HTTP_200_OK:
            data = response.json()
            assert "chunks" in data
            # Empty content should result in empty chunks or single empty chunk
            assert len(data["chunks"]) == 0 or (len(data["chunks"]) == 1 and not data["chunks"][0]["content"].strip())

    def test_chunk_unsupported_language(self, test_client, sample_python_code, auth_headers):
        """Test chunking with unsupported language."""
        chunk_request = {
            "content": sample_python_code,
            "language": "unsupported_lang",
            "file_path": "test.unknown",
            "purpose": "general"
        }

        response = test_client.post("/api/v1/chunk", json=chunk_request, headers=auth_headers)

        # Should return error for unsupported language
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        assert "error" in data or "detail" in data


class TestChunkingAdvancedFeatures:
    """Tests for advanced chunking features."""

    def test_chunk_with_custom_options(self, test_client, sample_python_code, auth_headers):
        """Test chunking with custom options."""
        chunk_request = {
            "content": sample_python_code,
            "language": "python",
            "file_path": "test.py",
            "purpose": "general",
            "options": {
                "max_chunk_size": 500,
                "min_chunk_size": 50,
                "overlap_size": 20,
                "force_agentic": True
            }
        }

        response = test_client.post("/api/v1/chunk", json=chunk_request, headers=auth_headers)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Verify custom options are reflected
        metadata = data["metadata"]
        if "options" in metadata:
            assert metadata["options"]["force_agentic"] is True

    def test_chunk_with_metadata(self, test_client, sample_python_code, auth_headers):
        """Test chunking with additional metadata."""
        chunk_request = {
            "content": sample_python_code,
            "language": "python",
            "file_path": "test.py",
            "purpose": "general",
            "metadata": {
                "project": "test_project",
                "author": "test_user",
                "version": "1.0.0"
            }
        }

        response = test_client.post("/api/v1/chunk", json=chunk_request, headers=auth_headers)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Verify additional metadata is preserved
        metadata = data["metadata"]
        if "additional_metadata" in metadata:
            additional = metadata["additional_metadata"]
            assert additional["project"] == "test_project"
            assert additional["author"] == "test_user"

    def test_chunk_large_file(self, test_client, auth_headers):
        """Test chunking with large file content."""
        # Generate large Python code
        code_parts = []
        for i in range(100):
            code_parts.extend([
                f"def function_{i}():",
                f'    """Function number {i}."""',
                f"    return {i}",
                ""
            ])
        large_code = "\n".join(code_parts)

        chunk_request = {
            "content": large_code,
            "language": "python",
            "file_path": "large_file.py",
            "purpose": "general"
        }

        response = test_client.post("/api/v1/chunk", json=chunk_request, headers=auth_headers)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Verify large file is chunked appropriately
        chunks = data["chunks"]
        assert len(chunks) > 1  # Should be split into multiple chunks

        # Verify total content is preserved
        total_content = "".join(chunk["content"] for chunk in chunks)
        assert len(total_content) > 0


class TestChunkingPerformance:
    """Tests for chunking performance and statistics."""

    def test_chunking_performance_stats(self, test_client):
        """Test chunking performance statistics endpoint."""
        response = test_client.get("/api/v1/chunk/stats")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Verify stats structure
        assert "performance" in data
        assert "usage" in data
        assert "timestamp" in data

        # Verify performance metrics
        performance = data["performance"]
        expected_metrics = ["avg_processing_time", "total_requests", "success_rate"]
        for metric in expected_metrics:
            if metric in performance:
                assert isinstance(performance[metric], (int, float))

    def test_chunking_test_endpoint(self, test_client):
        """Test chunking test endpoint."""
        response = test_client.post("/api/v1/chunk/test")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Verify test response
        assert "status" in data
        assert "test_results" in data
        assert data["status"] == "success"

    def test_chunking_performance_timing(self, test_client, sample_python_code, auth_headers):
        """Test chunking performance timing."""
        chunk_request = {
            "content": sample_python_code,
            "language": "python",
            "file_path": "test.py",
            "purpose": "general"
        }

        start_time = time.time()
        response = test_client.post("/api/v1/chunk", json=chunk_request, headers=auth_headers)
        end_time = time.time()

        assert response.status_code == status.HTTP_200_OK

        # Verify response time is reasonable (< 5 seconds)
        processing_time = end_time - start_time
        assert processing_time < 5.0

        # Check if processing time is included in response
        data = response.json()
        if "metadata" in data and "processing_time" in data["metadata"]:
            reported_time = data["metadata"]["processing_time"]
            assert isinstance(reported_time, (int, float))
            assert reported_time > 0


class TestChunkingErrorHandling:
    """Tests for chunking error handling."""

    def test_chunk_without_auth(self, test_client, sample_python_code):
        """Test chunking without authentication."""
        chunk_request = {
            "content": sample_python_code,
            "language": "python",
            "file_path": "test.py",
            "purpose": "general"
        }

        response = test_client.post("/api/v1/chunk", json=chunk_request)

        # Should require authentication
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        data = response.json()
        assert "error" in data or "detail" in data

    def test_chunk_invalid_request_format(self, test_client, auth_headers):
        """Test chunking with invalid request format."""
        invalid_request = {
            "invalid_field": "invalid_value"
        }

        response = test_client.post("/api/v1/chunk", json=invalid_request, headers=auth_headers)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        data = response.json()
        assert "detail" in data

    def test_chunk_missing_required_fields(self, test_client, auth_headers):
        """Test chunking with missing required fields."""
        incomplete_request = {
            "content": "print('hello')"
            # Missing language, file_path, purpose
        }

        response = test_client.post("/api/v1/chunk", json=incomplete_request, headers=auth_headers)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        data = response.json()
        assert "detail" in data

    def test_chunk_malformed_content(self, test_client, auth_headers):
        """Test chunking with malformed content."""
        chunk_request = {
            "content": "def invalid_syntax(\n    # Malformed Python code",
            "language": "python",
            "file_path": "malformed.py",
            "purpose": "general"
        }

        response = test_client.post("/api/v1/chunk", json=chunk_request, headers=auth_headers)

        # Should handle malformed content gracefully
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_400_BAD_REQUEST]

        if response.status_code == status.HTTP_200_OK:
            data = response.json()
            # Should still return chunks, possibly with error information
            assert "chunks" in data


@pytest.mark.asyncio
class TestAsyncChunkingOperations:
    """Async tests for chunking operations."""

    async def test_async_chunking(self, async_client, sample_python_code, auth_headers):
        """Test async chunking operation."""
        chunk_request = {
            "content": sample_python_code,
            "language": "python",
            "file_path": "test.py",
            "purpose": "general"
        }

        response = await async_client.post("/api/v1/chunk", json=chunk_request, headers=auth_headers)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "chunks" in data
        assert "metadata" in data

    async def test_concurrent_chunking_requests(self, async_client, auth_headers):
        """Test concurrent chunking requests."""
        code_samples = [
            ("python", "def func1(): pass"),
            ("javascript", "function func2() {}"),
            ("python", "class TestClass: pass"),
            ("javascript", "const test = () => {};"),
        ]

        # Create concurrent requests
        tasks = []
        for language, content in code_samples:
            chunk_request = {
                "content": content,
                "language": language,
                "file_path": f"test.{language}",
                "purpose": "general"
            }
            task = async_client.post("/api/v1/chunk", json=chunk_request, headers=auth_headers)
            tasks.append(task)

        # Execute concurrently
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Verify all requests succeeded
        success_count = 0
        for response in responses:
            if hasattr(response, 'status_code') and response.status_code == status.HTTP_200_OK:
                success_count += 1

        assert success_count == len(code_samples)

    async def test_async_health_check(self, async_client):
        """Test async health check."""
        response = await async_client.get("/api/v1/chunk/health")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "status" in data
        assert "service" in data
