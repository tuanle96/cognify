"""
Comprehensive unit tests for Query endpoints.

Tests all query-related functionality including search, suggestions,
analytics, and query management operations.
"""

import pytest
from fastapi import status


class TestQuerySubmission:
    """Test query submission functionality."""

    def test_submit_query_success(self, mock_client):
        """Test successful query submission."""
        query_data = {
            "query": "test search query",
            "query_type": "semantic"
        }

        response = mock_client.post("/api/v1/query/", json=query_data)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Verify query data
        assert data["query_id"] == "query_test_123"
        assert data["query"] == query_data["query"]
        assert data["query_type"] == query_data["query_type"]
        assert data["status"] == "completed"
        assert data["results_count"] == 5

        # Verify timestamps
        assert data["submitted_at"] == "2024-01-01T12:00:00Z"
        assert data["completed_at"] == "2024-01-01T12:00:02Z"
        assert data["processing_time"] == 0.15

    def test_submit_query_minimal_data(self, mock_client):
        """Test query submission with minimal data."""
        query_data = {
            "query": "minimal query"
        }

        response = mock_client.post("/api/v1/query/", json=query_data)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert data["query"] == query_data["query"]
        assert data["query_type"] == "semantic"  # Default type

    def test_submit_query_different_types(self, mock_client):
        """Test query submission with different query types."""
        query_types = ["semantic", "keyword", "hybrid", "vector"]

        for query_type in query_types:
            query_data = {
                "query": f"test {query_type} query",
                "query_type": query_type
            }

            response = mock_client.post("/api/v1/query/", json=query_data)

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["query_type"] == query_type

    def test_submit_query_response_structure(self, mock_client):
        """Test query submission response structure."""
        query_data = {
            "query": "structure test query",
            "query_type": "semantic"
        }

        response = mock_client.post("/api/v1/query/", json=query_data)
        data = response.json()

        required_fields = [
            "query_id", "query", "query_type", "status", "submitted_at",
            "completed_at", "processing_time", "results_count"
        ]
        for field in required_fields:
            assert field in data


class TestDocumentSearch:
    """Test document search functionality."""

    def test_search_documents_success(self, mock_client):
        """Test successful document search."""
        search_data = {
            "query": "test search query",
            "limit": 5
        }

        response = mock_client.post("/api/v1/query/search", json=search_data)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Verify search metadata
        assert data["query"] == search_data["query"]
        assert data["query_type"] == "semantic"
        assert data["total_results"] == 25
        assert data["search_time"] == 0.08
        assert data["page"] == 1
        assert data["per_page"] == search_data["limit"]
        assert data["has_next"] is True

        # Verify results
        assert len(data["results"]) == search_data["limit"]
        for i, result in enumerate(data["results"], 1):
            assert result["document_id"] == f"doc_test_{i}"
            assert result["chunk_id"] == f"chunk_test_{i}"
            assert result["title"] == f"Test Document {i}"
            assert result["score"] == 0.95 - (i * 0.1)
            assert result["collection_id"] == "col_test_1"

    def test_search_with_filters(self, mock_client):
        """Test search with filters."""
        search_data = {
            "query": "filtered search",
            "collection_id": "col_test_1",
            "document_types": ["text", "markdown"],
            "tags": ["important", "recent"],
            "limit": 10
        }

        response = mock_client.post("/api/v1/query/search", json=search_data)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Verify filters applied
        assert "filters_applied" in data
        filters = data["filters_applied"]
        assert filters["collection_id"] == search_data["collection_id"]
        assert filters["document_types"] == search_data["document_types"]
        assert filters["tags"] == search_data["tags"]

    def test_search_result_structure(self, mock_client):
        """Test search result structure."""
        search_data = {
            "query": "structure test",
            "limit": 3
        }

        response = mock_client.post("/api/v1/query/search", json=search_data)
        data = response.json()

        # Verify top-level structure
        required_fields = [
            "query", "query_type", "results", "total_results", "search_time",
            "page", "per_page", "has_next", "filters_applied", "suggestions", "executed_at"
        ]
        for field in required_fields:
            assert field in data

        # Verify result item structure
        if data["results"]:
            result = data["results"][0]
            result_fields = [
                "document_id", "chunk_id", "title", "content", "score",
                "document_type", "collection_id", "metadata", "highlights",
                "chunk_index", "created_at"
            ]
            for field in result_fields:
                assert field in result

    def test_search_suggestions(self, mock_client):
        """Test search suggestions in results."""
        search_data = {
            "query": "test query for suggestions"
        }

        response = mock_client.post("/api/v1/query/search", json=search_data)
        data = response.json()

        assert "suggestions" in data
        suggestions = data["suggestions"]
        assert len(suggestions) == 3
        assert "test query for suggestions examples" in suggestions
        assert "test query for suggestions tutorial" in suggestions
        assert "how to test query for suggestions" in suggestions

    def test_search_with_different_limits(self, mock_client):
        """Test search with different result limits."""
        limits = [1, 3, 5, 10]

        for limit in limits:
            search_data = {
                "query": f"limit test {limit}",
                "limit": limit
            }

            response = mock_client.post("/api/v1/query/search", json=search_data)
            data = response.json()

            assert len(data["results"]) == min(limit, 5)  # Mock returns max 5
            assert data["per_page"] == limit


class TestQuerySuggestions:
    """Test query suggestions functionality."""

    def test_get_query_suggestions(self, mock_client):
        """Test getting query suggestions."""
        response = mock_client.get("/api/v1/query/suggestions")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Verify suggestions structure
        assert "suggestions" in data
        assert "popular_queries" in data
        assert "related_queries" in data
        assert "autocomplete" in data

        # Verify content
        assert len(data["suggestions"]) == 5
        assert len(data["popular_queries"]) == 5
        assert len(data["related_queries"]) == 3
        assert len(data["autocomplete"]) == 5

        # Verify specific suggestions
        assert "How to implement authentication?" in data["suggestions"]
        assert "authentication implementation" in data["popular_queries"]
        assert "user management" in data["related_queries"]
        assert "authentication" in data["autocomplete"]

    def test_suggestions_content_quality(self, mock_client):
        """Test quality of suggestion content."""
        response = mock_client.get("/api/v1/query/suggestions")
        data = response.json()

        # Verify suggestions are meaningful
        for suggestion in data["suggestions"]:
            assert len(suggestion) > 10  # Reasonable length
            # Some suggestions are questions, some are statements
            assert isinstance(suggestion, str)  # Should be strings

        # Verify popular queries are keywords/phrases
        for query in data["popular_queries"]:
            assert len(query) > 3  # Not too short
            assert isinstance(query, str)  # Should be strings


class TestQueryHistory:
    """Test query history functionality."""

    def test_get_query_history(self, mock_client):
        """Test getting query history."""
        response = mock_client.get("/api/v1/query/history")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Verify pagination
        assert data["total"] == 10
        assert data["page"] == 1
        assert data["per_page"] == 10
        assert data["has_next"] is False

        # Verify queries
        assert len(data["queries"]) == 10
        for i, query in enumerate(data["queries"], 1):
            assert query["query_id"] == f"query_test_{i}"
            assert query["query"] == f"Test query {i}"
            assert query["query_type"] == "semantic"
            assert query["results_count"] == 10 - i
            assert "search_time" in query
            assert "executed_at" in query

    def test_query_history_structure(self, mock_client):
        """Test query history response structure."""
        response = mock_client.get("/api/v1/query/history")
        data = response.json()

        required_fields = ["queries", "total", "page", "per_page", "has_next"]
        for field in required_fields:
            assert field in data

        # Query item structure
        if data["queries"]:
            query = data["queries"][0]
            query_fields = [
                "query_id", "query", "query_type", "results_count",
                "search_time", "executed_at", "collection_id"
            ]
            for field in query_fields:
                assert field in query


class TestQueryAnalytics:
    """Test query analytics functionality."""

    def test_get_query_analytics(self, mock_client):
        """Test getting query analytics."""
        response = mock_client.get("/api/v1/query/analytics")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Verify basic metrics
        assert data["total_queries"] == 1247
        assert data["unique_queries"] == 892
        assert data["avg_search_time"] == 0.12

        # Verify popular queries
        assert "popular_queries" in data
        popular = data["popular_queries"]
        assert len(popular) == 5

        for query_stat in popular:
            assert "query" in query_stat
            assert "count" in query_stat
            assert "avg_score" in query_stat
            assert isinstance(query_stat["count"], int)
            assert isinstance(query_stat["avg_score"], float)

    def test_query_trends_analysis(self, mock_client):
        """Test query trends in analytics."""
        response = mock_client.get("/api/v1/query/analytics")
        data = response.json()

        assert "query_trends" in data
        trends = data["query_trends"]
        assert len(trends) == 5

        for trend in trends:
            assert "date" in trend
            assert "count" in trend
            assert isinstance(trend["count"], int)

    def test_performance_metrics(self, mock_client):
        """Test performance metrics in analytics."""
        response = mock_client.get("/api/v1/query/analytics")
        data = response.json()

        assert "performance_metrics" in data
        metrics = data["performance_metrics"]

        required_metrics = [
            "avg_response_time", "p95_response_time",
            "success_rate", "cache_hit_rate"
        ]
        for metric in required_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))

    def test_analytics_date_range(self, mock_client):
        """Test analytics date range information."""
        response = mock_client.get("/api/v1/query/analytics")
        data = response.json()

        assert "date_range" in data
        date_range = data["date_range"]
        assert "from" in date_range
        assert "to" in date_range


class TestSavedQueries:
    """Test saved queries functionality."""

    def test_save_query(self, mock_client):
        """Test saving a query."""
        query_data = {
            "name": "Test Saved Query",
            "query": "authentication best practices",
            "query_type": "semantic",
            "description": "Query about authentication best practices",
            "collection_id": "col_test_123",
            "filters": {
                "document_types": ["text", "markdown"],
                "tags": ["security", "auth"]
            },
            "is_public": False
        }

        response = mock_client.post("/api/v1/query/saved", json=query_data)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert data["query_id"] == "saved_query_test_123"
        assert data["name"] == query_data["name"]
        assert data["query"] == query_data["query"]
        assert data["query_type"] == query_data["query_type"]
        assert data["description"] == query_data["description"]
        assert data["collection_id"] == query_data["collection_id"]
        assert data["is_public"] == query_data["is_public"]
        assert data["owner_id"] == "test_user_123"
        assert data["usage_count"] == 0

    def test_save_query_minimal_data(self, mock_client):
        """Test saving query with minimal data."""
        query_data = {
            "name": "Minimal Saved Query",
            "query": "minimal query"
        }

        response = mock_client.post("/api/v1/query/saved", json=query_data)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert data["name"] == query_data["name"]
        assert data["query"] == query_data["query"]
        assert data["query_type"] == "semantic"  # Default


class TestQueryPerformance:
    """Test query performance characteristics."""

    def test_search_performance(self, mock_client, performance_timer):
        """Test search performance."""
        search_data = {
            "query": "performance test query",
            "limit": 10
        }

        performance_timer.start()
        response = mock_client.post("/api/v1/query/search", json=search_data)
        performance_timer.stop()

        assert response.status_code == status.HTTP_200_OK
        assert performance_timer.elapsed < 0.1  # Should be very fast for mock

    def test_suggestions_performance(self, mock_client, performance_timer):
        """Test suggestions performance."""
        performance_timer.start()
        response = mock_client.get("/api/v1/query/suggestions")
        performance_timer.stop()

        assert response.status_code == status.HTTP_200_OK
        assert performance_timer.elapsed < 0.1  # Should be very fast for mock
