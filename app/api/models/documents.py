"""
Document management API models.

Pydantic models for document upload, indexing, and management endpoints.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from enum import Enum


class DocumentType(str, Enum):
    """Document type enumeration."""
    TEXT = "text"
    CODE = "code"
    MARKDOWN = "markdown"
    PDF = "pdf"
    DOCX = "docx"
    HTML = "html"
    JSON = "json"
    CSV = "csv"
    XML = "xml"


class DocumentStatus(str, Enum):
    """Document processing status."""
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"
    DELETED = "deleted"


class DocumentUploadRequest(BaseModel):
    """Document upload request model."""
    title: str = Field(..., min_length=1, max_length=200, description="Document title")
    content: Optional[str] = Field(None, description="Document content (for text uploads)")
    file_path: Optional[str] = Field(None, description="File path (for file uploads)")
    document_type: DocumentType = Field(DocumentType.TEXT, description="Document type")
    collection_id: Optional[str] = Field(None, description="Collection ID to add document to")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    tags: Optional[List[str]] = Field(None, description="Document tags")
    language: Optional[str] = Field("auto", description="Document language (auto-detect if not specified)")

    @validator('content', 'file_path')
    def content_or_file_required(cls, v, values):
        """Validate that either content or file_path is provided."""
        if not v and not values.get('file_path') and not values.get('content'):
            raise ValueError('Either content or file_path must be provided')
        return v


class DocumentMetadata(BaseModel):
    """Document metadata model."""
    file_size: Optional[int] = Field(None, description="File size in bytes")
    file_type: Optional[str] = Field(None, description="MIME type")
    encoding: Optional[str] = Field(None, description="Text encoding")
    language: Optional[str] = Field(None, description="Detected language")
    word_count: Optional[int] = Field(None, description="Word count")
    line_count: Optional[int] = Field(None, description="Line count")
    character_count: Optional[int] = Field(None, description="Character count")
    checksum: Optional[str] = Field(None, description="File checksum")
    extracted_entities: Optional[List[str]] = Field(None, description="Extracted entities")
    keywords: Optional[List[str]] = Field(None, description="Extracted keywords")


class DocumentResponse(BaseModel):
    """Document response model."""
    document_id: str = Field(..., description="Document ID")
    filename: str = Field(..., description="Document filename")
    document_type: str = Field(..., description="Document type")
    status: str = Field(..., description="Processing status")
    collection_name: Optional[str] = Field(None, description="Collection name")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Document metadata")
    upload_time: datetime = Field(..., description="Upload timestamp")


class DocumentListResponse(BaseModel):
    """Document list response model."""
    documents: List[DocumentResponse] = Field(..., description="List of documents")
    total: int = Field(..., description="Total number of documents")
    page: int = Field(..., description="Current page number")
    per_page: int = Field(..., description="Documents per page")
    has_next: bool = Field(..., description="Whether there are more pages")


class IndexingJobResponse(BaseModel):
    """Document indexing job response model."""
    job_id: str = Field(..., description="Indexing job ID")
    document_id: str = Field(..., description="Document ID")
    status: str = Field(..., description="Job status")
    progress: float = Field(..., description="Job progress (0.0 to 1.0)")
    chunks_created: Optional[int] = Field(None, description="Number of chunks created")
    chunks_indexed: Optional[int] = Field(None, description="Number of chunks indexed")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    started_at: datetime = Field(..., description="Job start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Job completion timestamp")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")


class BatchUploadResponse(BaseModel):
    """Batch upload response model."""
    batch_id: str = Field(..., description="Batch upload ID")
    total_files: int = Field(..., description="Total number of files")
    processed_files: int = Field(..., description="Number of files processed")
    failed_files: int = Field(..., description="Number of failed uploads")
    documents: List[DocumentResponse] = Field(..., description="List of uploaded documents")
    status: str = Field(..., description="Batch status")
    errors: Optional[List[Dict[str, str]]] = Field(None, description="List of errors")


class DocumentChunkResponse(BaseModel):
    """Document chunk response model."""
    chunk_id: str = Field(..., description="Chunk ID")
    document_id: str = Field(..., description="Document ID")
    content: str = Field(..., description="Chunk content")
    chunk_index: int = Field(..., description="Chunk index in document")
    start_position: Optional[int] = Field(None, description="Start position in original document")
    end_position: Optional[int] = Field(None, description="End position in original document")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Chunk metadata")
    embedding_vector: Optional[List[float]] = Field(None, description="Embedding vector")
    created_at: datetime = Field(..., description="Chunk creation timestamp")


class DocumentSearchRequest(BaseModel):
    """Document search request model."""
    query: str = Field(..., min_length=1, description="Search query")
    collection_id: Optional[str] = Field(None, description="Collection to search in")
    document_types: Optional[List[DocumentType]] = Field(None, description="Document types to include")
    tags: Optional[List[str]] = Field(None, description="Tags to filter by")
    limit: Optional[int] = Field(10, ge=1, le=100, description="Maximum number of results")
    offset: Optional[int] = Field(0, ge=0, description="Result offset for pagination")


class DocumentSearchResponse(BaseModel):
    """Document search response model."""
    query: str = Field(..., description="Search query")
    results: List[DocumentResponse] = Field(..., description="Search results")
    total: int = Field(..., description="Total number of matching documents")
    search_time: float = Field(..., description="Search time in seconds")
    page: int = Field(..., description="Current page number")
    per_page: int = Field(..., description="Results per page")
