"""
Comprehensive unit tests for Collections endpoints.

Tests all collection-related functionality including creation, management,
statistics, and document operations.
"""

import pytest
from fastapi import status


class TestCollectionCreation:
    """Test collection creation functionality."""

    def test_create_collection_success(self, mock_client, sample_collection_data):
        """Test successful collection creation."""
        response = mock_client.post("/api/v1/collections/", json=sample_collection_data)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Verify collection data
        assert data["collection_id"] == "col_test_123"
        assert data["name"] == sample_collection_data["name"]
        assert data["description"] == sample_collection_data["description"]
        assert data["visibility"] == sample_collection_data["visibility"]
        assert data["tags"] == sample_collection_data["tags"]
        assert data["owner_id"] == "test_user_123"
        
        # Verify default values
        assert data["document_count"] == 0
        assert data["total_chunks"] == 0
        assert data["storage_size"] == "0 B"
        assert data["embedding_dimension"] == 384
        assert data["distance_metric"] == "cosine"
        assert data["is_public"] is False

    def test_create_collection_minimal_data(self, mock_client):
        """Test collection creation with minimal data."""
        minimal_data = {
            "name": "Minimal Collection"
        }
        
        response = mock_client.post("/api/v1/collections/", json=minimal_data)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["name"] == minimal_data["name"]
        assert data["description"] == "A test collection for unit testing"  # Default
        assert data["visibility"] == "private"  # Default

    def test_create_collection_with_custom_metadata(self, mock_client):
        """Test collection creation with custom metadata."""
        collection_data = {
            "name": "Custom Metadata Collection",
            "description": "Collection with custom metadata",
            "metadata": {
                "project": "test_project",
                "version": "1.0",
                "environment": "testing"
            }
        }
        
        response = mock_client.post("/api/v1/collections/", json=collection_data)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["name"] == collection_data["name"]
        assert "metadata" in data

    def test_create_collection_response_structure(self, mock_client, sample_collection_data):
        """Test collection creation response structure."""
        response = mock_client.post("/api/v1/collections/", json=sample_collection_data)
        data = response.json()
        
        required_fields = [
            "collection_id", "name", "description", "visibility", "tags", "metadata",
            "document_count", "total_chunks", "storage_size", "embedding_dimension",
            "distance_metric", "created_at", "updated_at", "owner_id", "is_public"
        ]
        for field in required_fields:
            assert field in data


class TestCollectionRetrieval:
    """Test collection retrieval functionality."""

    def test_list_collections(self, mock_client):
        """Test listing all collections."""
        response = mock_client.get("/api/v1/collections/")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Verify pagination
        assert data["total"] == 5
        assert data["page"] == 1
        assert data["per_page"] == 10
        assert data["has_next"] is False
        
        # Verify collections
        assert len(data["collections"]) == 5
        for i, collection in enumerate(data["collections"], 1):
            assert collection["collection_id"] == f"col_test_{i}"
            assert collection["name"] == f"Test Collection {i}"
            assert collection["visibility"] == "private"
            assert collection["document_count"] == i * 3
            assert collection["owner_id"] == "test_user_123"

    def test_list_collections_structure(self, mock_client):
        """Test list collections response structure."""
        response = mock_client.get("/api/v1/collections/")
        data = response.json()
        
        required_fields = ["collections", "total", "page", "per_page", "has_next"]
        for field in required_fields:
            assert field in data
        
        # Collection structure
        if data["collections"]:
            collection = data["collections"][0]
            collection_fields = [
                "collection_id", "name", "description", "visibility",
                "document_count", "total_chunks", "storage_size",
                "created_at", "owner_id", "is_public"
            ]
            for field in collection_fields:
                assert field in collection

    def test_get_specific_collection(self, mock_client):
        """Test getting a specific collection."""
        collection_id = "col_test_123"
        response = mock_client.get(f"/api/v1/collections/{collection_id}")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Verify collection data
        assert data["collection_id"] == collection_id
        assert data["name"] == f"Test Collection {collection_id}"
        assert data["description"] == "A comprehensive test collection for unit testing"
        assert data["visibility"] == "private"
        assert data["document_count"] == 12
        assert data["total_chunks"] == 84
        assert data["storage_size"] == "5.2 MB"
        
        # Verify documents list
        assert "documents" in data
        assert len(data["documents"]) == 5
        for i, doc in enumerate(data["documents"], 1):
            assert doc["document_id"] == f"doc_test_{i}"
            assert doc["title"] == f"Test Document {i}"
            assert doc["chunks"] == 7

    def test_get_collection_with_different_ids(self, mock_client):
        """Test getting collections with different IDs."""
        test_ids = ["col_1", "col_abc", "col_xyz_123"]
        
        for collection_id in test_ids:
            response = mock_client.get(f"/api/v1/collections/{collection_id}")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["collection_id"] == collection_id


class TestCollectionManagement:
    """Test collection management operations."""

    def test_update_collection(self, mock_client):
        """Test collection update."""
        collection_id = "col_test_123"
        update_data = {
            "name": "Updated Collection Name",
            "description": "Updated description",
            "visibility": "public",
            "tags": ["updated", "test", "modified"]
        }
        
        response = mock_client.put(f"/api/v1/collections/{collection_id}", json=update_data)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["collection_id"] == collection_id
        assert data["name"] == update_data["name"]
        assert data["description"] == update_data["description"]
        assert data["visibility"] == update_data["visibility"]
        assert data["tags"] == update_data["tags"]
        assert data["updated_at"] == "2024-01-01T12:45:00Z"

    def test_update_collection_partial(self, mock_client):
        """Test partial collection update."""
        collection_id = "col_test_123"
        update_data = {
            "name": "Partially Updated Collection"
        }
        
        response = mock_client.put(f"/api/v1/collections/{collection_id}", json=update_data)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["name"] == update_data["name"]
        assert data["collection_id"] == collection_id

    def test_delete_collection(self, mock_client):
        """Test collection deletion."""
        collection_id = "col_test_123"
        response = mock_client.delete(f"/api/v1/collections/{collection_id}")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["message"] == f"Collection {collection_id} deleted successfully"
        assert data["collection_id"] == collection_id
        assert data["deleted_at"] == "2024-01-01T12:50:00Z"
        assert data["documents_removed"] == 12
        assert data["chunks_removed"] == 84

    def test_add_document_to_collection(self, mock_client):
        """Test adding document to collection."""
        collection_id = "col_test_123"
        document_data = {
            "document_id": "doc_test_new"
        }
        
        response = mock_client.post(f"/api/v1/collections/{collection_id}/documents", 
                                  json=document_data)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["message"] == f"Document doc_test_new added to collection {collection_id}"
        assert data["collection_id"] == collection_id
        assert data["document_id"] == "doc_test_new"
        assert data["added_at"] == "2024-01-01T13:00:00Z"


class TestCollectionStatistics:
    """Test collection statistics and analytics."""

    def test_get_collection_stats(self, mock_client):
        """Test getting collection statistics."""
        collection_id = "col_test_123"
        response = mock_client.get(f"/api/v1/collections/{collection_id}/stats")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Verify basic stats
        assert data["collection_id"] == collection_id
        assert data["document_count"] == 12
        assert data["total_chunks"] == 84
        assert data["total_tokens"] == 15680
        assert data["storage_size_bytes"] == 5242880
        assert data["storage_size_human"] == "5.2 MB"
        
        # Verify document types distribution
        assert "document_types" in data
        doc_types = data["document_types"]
        assert doc_types["text"] == 8
        assert doc_types["markdown"] == 3
        assert doc_types["code"] == 1
        
        # Verify language distribution
        assert "language_distribution" in data
        lang_dist = data["language_distribution"]
        assert lang_dist["en"] == 10
        assert lang_dist["es"] == 1
        assert lang_dist["fr"] == 1

    def test_collection_stats_structure(self, mock_client):
        """Test collection statistics response structure."""
        collection_id = "col_test_123"
        response = mock_client.get(f"/api/v1/collections/{collection_id}/stats")
        data = response.json()
        
        required_fields = [
            "collection_id", "document_count", "total_chunks", "total_tokens",
            "storage_size_bytes", "storage_size_human", "avg_document_size",
            "document_types", "language_distribution", "recent_activity",
            "created_at", "last_updated"
        ]
        for field in required_fields:
            assert field in data

    def test_collection_recent_activity(self, mock_client):
        """Test collection recent activity tracking."""
        collection_id = "col_test_123"
        response = mock_client.get(f"/api/v1/collections/{collection_id}/stats")
        data = response.json()
        
        assert "recent_activity" in data
        activity = data["recent_activity"]
        
        # Verify activity structure
        assert len(activity) == 2
        for item in activity:
            assert "action" in item
            assert "timestamp" in item
            
        # Verify specific activities
        assert activity[0]["action"] == "document_added"
        assert activity[0]["document_id"] == "doc_test_12"
        assert activity[1]["action"] == "document_updated"
        assert activity[1]["document_id"] == "doc_test_8"


class TestCollectionValidation:
    """Test collection validation and error handling."""

    def test_create_collection_with_different_visibilities(self, mock_client):
        """Test collection creation with different visibility settings."""
        visibilities = ["private", "public", "shared"]
        
        for visibility in visibilities:
            collection_data = {
                "name": f"Test {visibility.title()} Collection",
                "visibility": visibility
            }
            
            response = mock_client.post("/api/v1/collections/", json=collection_data)
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["visibility"] == visibility

    def test_collection_tags_validation(self, mock_client):
        """Test collection tags validation."""
        collection_data = {
            "name": "Tags Test Collection",
            "tags": ["tag1", "tag2", "tag3", "special-tag", "tag_with_underscore"]
        }
        
        response = mock_client.post("/api/v1/collections/", json=collection_data)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["tags"] == collection_data["tags"]

    def test_collection_metadata_types(self, mock_client):
        """Test collection metadata with different data types."""
        collection_data = {
            "name": "Metadata Types Collection",
            "metadata": {
                "string_field": "test_string",
                "number_field": 42,
                "boolean_field": True,
                "array_field": ["item1", "item2"],
                "nested_object": {
                    "nested_string": "nested_value",
                    "nested_number": 123
                }
            }
        }
        
        response = mock_client.post("/api/v1/collections/", json=collection_data)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "metadata" in data


class TestCollectionPerformance:
    """Test collection performance characteristics."""

    def test_create_collection_performance(self, mock_client, performance_timer):
        """Test collection creation performance."""
        collection_data = {
            "name": "Performance Test Collection",
            "description": "Testing collection creation performance"
        }
        
        performance_timer.start()
        response = mock_client.post("/api/v1/collections/", json=collection_data)
        performance_timer.stop()
        
        assert response.status_code == status.HTTP_200_OK
        assert performance_timer.elapsed < 0.1  # Should be very fast for mock

    def test_list_collections_performance(self, mock_client, performance_timer):
        """Test list collections performance."""
        performance_timer.start()
        response = mock_client.get("/api/v1/collections/")
        performance_timer.stop()
        
        assert response.status_code == status.HTTP_200_OK
        assert performance_timer.elapsed < 0.1  # Should be very fast for mock

    def test_collection_stats_performance(self, mock_client, performance_timer):
        """Test collection statistics performance."""
        collection_id = "col_test_123"
        
        performance_timer.start()
        response = mock_client.get(f"/api/v1/collections/{collection_id}/stats")
        performance_timer.stop()
        
        assert response.status_code == status.HTTP_200_OK
        assert performance_timer.elapsed < 0.1  # Should be very fast for mock
