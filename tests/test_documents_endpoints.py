"""
Comprehensive unit tests for Documents endpoints.

Tests all document-related functionality including upload, processing,
management, and retrieval operations.
"""

import pytest
from fastapi import status


class TestDocumentUpload:
    """Test document upload functionality."""

    def test_upload_document_success(self, mock_client, sample_document_data):
        """Test successful document upload."""
        response = mock_client.post("/api/v1/documents/upload", json=sample_document_data)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Verify document data
        assert data["document_id"] == "doc_test_123"
        assert data["title"] == sample_document_data["title"]
        assert data["document_type"] == sample_document_data["document_type"]
        assert data["status"] == "indexed"
        assert data["chunks_created"] == 8
        assert data["owner_id"] == "test_user_123"
        
        # Verify processing info
        assert "processing_time" in data
        assert data["processing_time"] == 0.25
        assert data["created_at"] == "2024-01-01T12:00:00Z"

    def test_upload_document_with_metadata(self, mock_client):
        """Test document upload with custom metadata."""
        document_data = {
            "title": "Document with Metadata",
            "content": "This document has custom metadata.",
            "document_type": "text",
            "metadata": {
                "author": "Test Author",
                "category": "testing",
                "priority": "high"
            }
        }
        
        response = mock_client.post("/api/v1/documents/upload", json=document_data)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["title"] == document_data["title"]
        assert "metadata" in data
        assert data["metadata"]["file_size"] == 2048
        assert data["metadata"]["word_count"] == 350

    def test_upload_document_minimal_data(self, mock_client):
        """Test upload with minimal required data."""
        minimal_data = {
            "title": "Minimal Document",
            "content": "Minimal content for testing."
        }
        
        response = mock_client.post("/api/v1/documents/upload", json=minimal_data)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["title"] == minimal_data["title"]
        assert data["document_type"] == "text"  # Default type

    def test_upload_document_response_structure(self, mock_client, sample_document_data):
        """Test upload response has correct structure."""
        response = mock_client.post("/api/v1/documents/upload", json=sample_document_data)
        data = response.json()
        
        required_fields = [
            "document_id", "title", "file_path", "document_type", "status",
            "chunks_created", "processing_time", "metadata", "created_at", "owner_id"
        ]
        for field in required_fields:
            assert field in data
        
        # Metadata structure
        metadata_fields = ["file_size", "word_count", "language", "encoding"]
        for field in metadata_fields:
            assert field in data["metadata"]


class TestBatchUpload:
    """Test batch document upload functionality."""

    def test_batch_upload_success(self, mock_client):
        """Test successful batch upload."""
        batch_data = {
            "files": [
                {"title": "Doc 1", "content": "Content 1"},
                {"title": "Doc 2", "content": "Content 2"},
                {"title": "Doc 3", "content": "Content 3"}
            ]
        }
        
        response = mock_client.post("/api/v1/documents/upload/batch", json=batch_data)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Verify batch info
        assert data["batch_id"] == "batch_test_456"
        assert data["total_files"] == 3
        assert data["processed"] == 3
        assert data["successful"] == 3
        assert data["failed"] == 0
        
        # Verify documents
        assert len(data["documents"]) == 3
        for i, doc in enumerate(data["documents"], 1):
            assert doc["document_id"] == f"doc_test_{i}"
            assert doc["title"] == f"Test Document {i}"
            assert doc["status"] == "indexed"

    def test_batch_upload_empty_list(self, mock_client):
        """Test batch upload with empty file list."""
        batch_data = {"files": []}
        
        response = mock_client.post("/api/v1/documents/upload/batch", json=batch_data)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Should default to 3 files in mock
        assert data["total_files"] == 3

    def test_batch_upload_response_structure(self, mock_client):
        """Test batch upload response structure."""
        batch_data = {"files": [{"title": "Test", "content": "Test"}]}
        
        response = mock_client.post("/api/v1/documents/upload/batch", json=batch_data)
        data = response.json()
        
        required_fields = [
            "batch_id", "total_files", "processed", "successful", "failed",
            "documents", "errors", "started_at", "completed_at"
        ]
        for field in required_fields:
            assert field in data


class TestDocumentRetrieval:
    """Test document retrieval functionality."""

    def test_list_documents(self, mock_client):
        """Test listing all documents."""
        response = mock_client.get("/api/v1/documents/")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Verify pagination
        assert data["total"] == 5
        assert data["page"] == 1
        assert data["per_page"] == 10
        assert data["has_next"] is False
        
        # Verify documents
        assert len(data["documents"]) == 5
        for i, doc in enumerate(data["documents"], 1):
            assert doc["document_id"] == f"doc_test_{i}"
            assert doc["title"] == f"Test Document {i}"
            assert doc["status"] == "indexed"
            assert doc["owner_id"] == "test_user_123"

    def test_list_documents_structure(self, mock_client):
        """Test list documents response structure."""
        response = mock_client.get("/api/v1/documents/")
        data = response.json()
        
        required_fields = ["documents", "total", "page", "per_page", "has_next"]
        for field in required_fields:
            assert field in data
        
        # Document structure
        if data["documents"]:
            doc = data["documents"][0]
            doc_fields = [
                "document_id", "title", "document_type", "status",
                "chunks_created", "created_at", "owner_id"
            ]
            for field in doc_fields:
                assert field in doc

    def test_get_specific_document(self, mock_client):
        """Test getting a specific document."""
        document_id = "doc_test_123"
        response = mock_client.get(f"/api/v1/documents/{document_id}")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Verify document data
        assert data["document_id"] == document_id
        assert data["title"] == f"Test Document {document_id}"
        assert data["status"] == "indexed"
        assert data["chunks_created"] == 7
        assert data["owner_id"] == "test_user_123"
        
        # Verify content and metadata
        assert "content" in data
        assert "metadata" in data
        assert data["metadata"]["file_size"] == 2048
        assert data["metadata"]["word_count"] == 350

    def test_get_document_with_different_ids(self, mock_client):
        """Test getting documents with different IDs."""
        test_ids = ["doc_1", "doc_abc", "doc_xyz_123"]
        
        for doc_id in test_ids:
            response = mock_client.get(f"/api/v1/documents/{doc_id}")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["document_id"] == doc_id


class TestDocumentManagement:
    """Test document management operations."""

    def test_delete_document(self, mock_client):
        """Test document deletion."""
        document_id = "doc_test_123"
        response = mock_client.delete(f"/api/v1/documents/{document_id}")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["message"] == f"Document {document_id} deleted successfully"
        assert data["document_id"] == document_id
        assert data["deleted_at"] == "2024-01-01T12:30:00Z"
        assert data["chunks_removed"] == 7

    def test_reindex_document(self, mock_client):
        """Test document reindexing."""
        document_id = "doc_test_123"
        response = mock_client.post(f"/api/v1/documents/{document_id}/reindex")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["document_id"] == document_id
        assert data["status"] == "reindexed"
        assert data["chunks_updated"] == 7
        assert data["chunks_added"] == 2
        assert data["chunks_removed"] == 1
        assert "processing_time" in data
        assert data["reindexed_at"] == "2024-01-01T12:45:00Z"

    def test_get_document_chunks(self, mock_client):
        """Test getting document chunks."""
        document_id = "doc_test_123"
        response = mock_client.get(f"/api/v1/documents/{document_id}/chunks")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["document_id"] == document_id
        assert data["total_chunks"] == 5
        assert len(data["chunks"]) == 5
        
        # Verify chunk structure
        for i, chunk in enumerate(data["chunks"], 1):
            assert chunk["chunk_id"] == f"chunk_{document_id}_{i}"
            assert "content" in chunk
            assert chunk["chunk_index"] == i
            assert "metadata" in chunk
            
            # Verify chunk metadata
            metadata = chunk["metadata"]
            assert metadata["chunk_type"] == "paragraph"
            assert metadata["language"] == "en"
            assert "complexity" in metadata


class TestDocumentValidation:
    """Test document validation and error handling."""

    def test_upload_document_types(self, mock_client):
        """Test upload with different document types."""
        document_types = ["text", "markdown", "code", "pdf", "docx"]
        
        for doc_type in document_types:
            document_data = {
                "title": f"Test {doc_type} Document",
                "content": f"Content for {doc_type} document",
                "document_type": doc_type
            }
            
            response = mock_client.post("/api/v1/documents/upload", json=document_data)
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["document_type"] == doc_type

    def test_document_metadata_validation(self, mock_client):
        """Test document metadata validation."""
        response = mock_client.post("/api/v1/documents/upload", json={
            "title": "Metadata Test",
            "content": "Test content"
        })
        
        data = response.json()
        metadata = data["metadata"]
        
        # Verify metadata types
        assert isinstance(metadata["file_size"], int)
        assert isinstance(metadata["word_count"], int)
        assert isinstance(metadata["language"], str)
        assert isinstance(metadata["encoding"], str)


class TestDocumentPerformance:
    """Test document processing performance."""

    def test_upload_processing_time(self, mock_client, performance_timer):
        """Test document upload processing time."""
        document_data = {
            "title": "Performance Test Document",
            "content": "Content for performance testing."
        }
        
        performance_timer.start()
        response = mock_client.post("/api/v1/documents/upload", json=document_data)
        performance_timer.stop()
        
        assert response.status_code == status.HTTP_200_OK
        assert performance_timer.elapsed < 0.1  # Should be very fast for mock

    def test_batch_upload_performance(self, mock_client, performance_timer):
        """Test batch upload performance."""
        batch_data = {
            "files": [
                {"title": f"Doc {i}", "content": f"Content {i}"}
                for i in range(10)
            ]
        }
        
        performance_timer.start()
        response = mock_client.post("/api/v1/documents/upload/batch", json=batch_data)
        performance_timer.stop()
        
        assert response.status_code == status.HTTP_200_OK
        assert performance_timer.elapsed < 0.1  # Should be very fast for mock
