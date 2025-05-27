"""
Comprehensive unit tests for Chunking endpoints.

Tests all chunking-related functionality including content processing,
health checks, performance statistics, and language support.
"""

import pytest
from fastapi import status


class TestContentChunking:
    """Test content chunking functionality."""

    def test_chunk_content_success(self, mock_client, sample_python_code):
        """Test successful content chunking."""
        chunking_data = {
            "content": sample_python_code,
            "language": "python",
            "strategy": "hybrid",
            "max_chunk_size": 1000,
            "overlap_size": 100,
            "include_metadata": True
        }
        
        response = mock_client.post("/api/v1/chunking/", json=chunking_data)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Verify chunking results
        assert data["strategy_used"] == "hybrid"
        assert data["language_detected"] == "python"
        assert data["total_chunks"] > 0
        assert "processing_time" in data
        assert "chunks" in data
        
        # Verify metadata
        assert "metadata" in data
        metadata = data["metadata"]
        assert "original_length" in metadata
        assert "avg_chunk_size" in metadata
        assert "overlap_used" in metadata
        assert metadata["parser_used"] == "mock_parser"

    def test_chunk_content_minimal_data(self, mock_client):
        """Test chunking with minimal data."""
        chunking_data = {
            "content": "def test(): pass"
        }
        
        response = mock_client.post("/api/v1/chunking/", json=chunking_data)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["strategy_used"] == "hybrid"  # Default
        assert data["language_detected"] == "python"  # Default
        assert data["total_chunks"] >= 1

    def test_chunk_different_languages(self, mock_client):
        """Test chunking with different programming languages."""
        languages = ["python", "javascript", "typescript", "java", "cpp"]
        
        for language in languages:
            chunking_data = {
                "content": f"// {language} code\nfunction test() {{ return true; }}",
                "language": language,
                "strategy": "hybrid"
            }
            
            response = mock_client.post("/api/v1/chunking/", json=chunking_data)
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["language_detected"] == language

    def test_chunk_different_strategies(self, mock_client):
        """Test chunking with different strategies."""
        strategies = ["ast", "hybrid", "agentic"]
        
        for strategy in strategies:
            chunking_data = {
                "content": "def test_function(): return 'test'",
                "language": "python",
                "strategy": strategy
            }
            
            response = mock_client.post("/api/v1/chunking/", json=chunking_data)
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["strategy_used"] == strategy

    def test_chunk_response_structure(self, mock_client):
        """Test chunking response structure."""
        chunking_data = {
            "content": "def example(): pass",
            "language": "python"
        }
        
        response = mock_client.post("/api/v1/chunking/", json=chunking_data)
        data = response.json()
        
        required_fields = [
            "chunks", "total_chunks", "processing_time", "strategy_used",
            "language_detected", "metadata"
        ]
        for field in required_fields:
            assert field in data
        
        # Verify chunk structure
        if data["chunks"]:
            chunk = data["chunks"][0]
            chunk_fields = ["content", "metadata", "chunk_id"]
            for field in chunk_fields:
                assert field in chunk
            
            # Verify chunk metadata
            chunk_metadata = chunk["metadata"]
            metadata_fields = [
                "start_line", "end_line", "chunk_type", "language", 
                "complexity", "dependencies"
            ]
            for field in metadata_fields:
                assert field in chunk_metadata

    def test_chunk_with_custom_parameters(self, mock_client):
        """Test chunking with custom parameters."""
        chunking_data = {
            "content": "def test(): pass\nclass TestClass: pass",
            "language": "python",
            "strategy": "ast",
            "max_chunk_size": 500,
            "overlap_size": 50,
            "include_metadata": True
        }
        
        response = mock_client.post("/api/v1/chunking/", json=chunking_data)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Verify parameters were used
        assert data["strategy_used"] == "ast"
        assert data["metadata"]["overlap_used"] == 50


class TestChunkingHealth:
    """Test chunking service health functionality."""

    def test_chunking_health_check(self, mock_client):
        """Test chunking service health check."""
        response = mock_client.get("/api/v1/chunking/health")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Verify health status
        assert data["status"] == "healthy"
        assert data["service"] == "chunking"
        assert data["version"] == "1.0.0-test"
        assert data["last_check"] == "2024-01-01T12:00:00Z"
        
        # Verify capabilities
        assert "strategies_available" in data
        strategies = data["strategies_available"]
        assert "ast" in strategies
        assert "hybrid" in strategies
        assert "agentic" in strategies
        
        assert data["languages_supported"] == 25

    def test_health_performance_metrics(self, mock_client):
        """Test health check performance metrics."""
        response = mock_client.get("/api/v1/chunking/health")
        data = response.json()
        
        assert "performance" in data
        performance = data["performance"]
        
        required_metrics = [
            "avg_processing_time", "requests_processed", "success_rate"
        ]
        for metric in required_metrics:
            assert metric in performance
            assert isinstance(performance[metric], (int, float))

    def test_health_check_structure(self, mock_client):
        """Test health check response structure."""
        response = mock_client.get("/api/v1/chunking/health")
        data = response.json()
        
        required_fields = [
            "status", "service", "version", "strategies_available",
            "languages_supported", "last_check", "performance"
        ]
        for field in required_fields:
            assert field in data


class TestChunkingStatistics:
    """Test chunking performance statistics."""

    def test_get_chunking_stats(self, mock_client):
        """Test getting chunking performance statistics."""
        response = mock_client.get("/api/v1/chunking/stats")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Verify basic stats
        assert data["total_requests"] == 1247
        assert data["avg_processing_time"] == 0.125
        assert data["success_rate"] == 99.2
        
        # Verify supported languages
        assert "supported_languages" in data
        languages = data["supported_languages"]
        assert len(languages) > 20
        assert "python" in languages
        assert "javascript" in languages
        
        # Verify active strategies
        assert "active_strategies" in data
        strategies = data["active_strategies"]
        assert "ast" in strategies
        assert "hybrid" in strategies
        assert "agentic" in strategies

    def test_performance_by_strategy(self, mock_client):
        """Test performance statistics by strategy."""
        response = mock_client.get("/api/v1/chunking/stats")
        data = response.json()
        
        assert "performance_by_strategy" in data
        strategy_stats = data["performance_by_strategy"]
        
        # Verify each strategy has stats
        for strategy in ["ast", "hybrid", "agentic"]:
            assert strategy in strategy_stats
            stats = strategy_stats[strategy]
            
            required_fields = ["avg_time", "success_rate", "requests"]
            for field in required_fields:
                assert field in stats
                assert isinstance(stats[field], (int, float))

    def test_performance_by_language(self, mock_client):
        """Test performance statistics by language."""
        response = mock_client.get("/api/v1/chunking/stats")
        data = response.json()
        
        assert "performance_by_language" in data
        language_stats = data["performance_by_language"]
        
        # Verify popular languages have stats
        popular_languages = ["python", "javascript", "typescript", "java"]
        for language in popular_languages:
            if language in language_stats:
                stats = language_stats[language]
                assert "avg_time" in stats
                assert "requests" in stats
                assert isinstance(stats["avg_time"], float)
                assert isinstance(stats["requests"], int)

    def test_stats_response_structure(self, mock_client):
        """Test statistics response structure."""
        response = mock_client.get("/api/v1/chunking/stats")
        data = response.json()
        
        required_fields = [
            "total_requests", "avg_processing_time", "success_rate",
            "supported_languages", "active_strategies", "performance_by_strategy",
            "performance_by_language"
        ]
        for field in required_fields:
            assert field in data


class TestLanguageSupport:
    """Test language support functionality."""

    def test_get_supported_languages(self, mock_client):
        """Test getting supported programming languages."""
        response = mock_client.get("/api/v1/chunking/supported-languages")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Verify basic language support
        assert "supported_languages" in data
        languages = data["supported_languages"]
        assert len(languages) > 20
        
        # Verify specific languages
        expected_languages = [
            "python", "javascript", "typescript", "java", "cpp", "c",
            "csharp", "go", "rust", "php", "ruby", "swift", "kotlin"
        ]
        for language in expected_languages:
            assert language in languages

    def test_language_features(self, mock_client):
        """Test language-specific features."""
        response = mock_client.get("/api/v1/chunking/supported-languages")
        data = response.json()
        
        # Verify auto detection
        assert data["auto_detection"] is True
        
        # Verify parsers
        assert "custom_parsers" in data
        parsers = data["custom_parsers"]
        assert "ast" in parsers
        assert "tree-sitter" in parsers
        assert "regex" in parsers
        
        # Verify strategies
        assert "strategies" in data
        strategies = data["strategies"]
        assert "ast" in strategies
        assert "hybrid" in strategies
        assert "agentic" in strategies

    def test_language_details(self, mock_client):
        """Test detailed language information."""
        response = mock_client.get("/api/v1/chunking/supported-languages")
        data = response.json()
        
        assert "language_details" in data
        details = data["language_details"]
        
        # Verify Python details
        if "python" in details:
            python_details = details["python"]
            assert python_details["parser"] == "ast"
            assert "strategies" in python_details
            assert "features" in python_details
            assert "functions" in python_details["features"]
            assert "classes" in python_details["features"]

    def test_supported_languages_structure(self, mock_client):
        """Test supported languages response structure."""
        response = mock_client.get("/api/v1/chunking/supported-languages")
        data = response.json()
        
        required_fields = [
            "supported_languages", "auto_detection", "custom_parsers",
            "strategies", "language_details"
        ]
        for field in required_fields:
            assert field in data


class TestChunkingTesting:
    """Test chunking test functionality."""

    def test_chunking_test_endpoint(self, mock_client):
        """Test chunking test endpoint."""
        response = mock_client.post("/api/v1/chunking/test")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Verify test status
        assert data["test_status"] == "success"
        assert data["chunks_created"] == 3
        assert data["processing_time"] == 0.08
        assert data["strategy_used"] == "hybrid"
        assert data["language_detected"] == "python"
        
        # Verify sample code
        assert "sample_code" in data
        sample_code = data["sample_code"]
        assert "fibonacci" in sample_code
        assert "Calculator" in sample_code

    def test_test_chunks_structure(self, mock_client):
        """Test structure of test chunks."""
        response = mock_client.post("/api/v1/chunking/test")
        data = response.json()
        
        assert "chunks" in data
        chunks = data["chunks"]
        assert len(chunks) == 3
        
        # Verify chunk structure
        for chunk in chunks:
            assert "chunk_id" in chunk
            assert "content" in chunk
            assert "metadata" in chunk
            
            # Verify metadata
            metadata = chunk["metadata"]
            assert "chunk_type" in metadata
            assert "start_line" in metadata
            assert "end_line" in metadata

    def test_test_endpoint_consistency(self, mock_client):
        """Test test endpoint returns consistent results."""
        responses = []
        for _ in range(3):
            response = mock_client.post("/api/v1/chunking/test")
            responses.append(response.json())
        
        # All responses should be identical (predictable mock data)
        for response in responses[1:]:
            assert response["test_status"] == responses[0]["test_status"]
            assert response["chunks_created"] == responses[0]["chunks_created"]
            assert response["strategy_used"] == responses[0]["strategy_used"]


class TestChunkingValidation:
    """Test chunking validation and error handling."""

    def test_chunk_with_empty_content(self, mock_client):
        """Test chunking with empty content."""
        chunking_data = {
            "content": "",
            "language": "python"
        }
        
        response = mock_client.post("/api/v1/chunking/", json=chunking_data)
        
        # Mock should handle gracefully
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total_chunks"] >= 1  # Mock returns at least 1 chunk

    def test_chunk_with_invalid_language(self, mock_client):
        """Test chunking with unsupported language."""
        chunking_data = {
            "content": "some code content",
            "language": "unsupported_language"
        }
        
        response = mock_client.post("/api/v1/chunking/", json=chunking_data)
        
        # Mock should handle gracefully
        assert response.status_code == status.HTTP_200_OK

    def test_chunk_parameter_validation(self, mock_client):
        """Test chunking parameter validation."""
        chunking_data = {
            "content": "def test(): pass",
            "max_chunk_size": 100,
            "overlap_size": 50
        }
        
        response = mock_client.post("/api/v1/chunking/", json=chunking_data)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Verify parameters are reflected in metadata
        assert "metadata" in data
        assert "overlap_used" in data["metadata"]


class TestChunkingPerformance:
    """Test chunking performance characteristics."""

    def test_chunking_performance(self, mock_client, performance_timer):
        """Test chunking performance."""
        chunking_data = {
            "content": "def test(): pass\nclass Test: pass",
            "language": "python",
            "strategy": "hybrid"
        }
        
        performance_timer.start()
        response = mock_client.post("/api/v1/chunking/", json=chunking_data)
        performance_timer.stop()
        
        assert response.status_code == status.HTTP_200_OK
        assert performance_timer.elapsed < 0.1  # Should be very fast for mock

    def test_health_check_performance(self, mock_client, performance_timer):
        """Test health check performance."""
        performance_timer.start()
        response = mock_client.get("/api/v1/chunking/health")
        performance_timer.stop()
        
        assert response.status_code == status.HTTP_200_OK
        assert performance_timer.elapsed < 0.1  # Should be very fast for mock
