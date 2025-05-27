"""
Collection management API models.

Pydantic models for collection creation, management, and organization endpoints.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class CollectionVisibility(str, Enum):
    """Collection visibility enumeration."""
    PRIVATE = "private"
    PUBLIC = "public"
    SHARED = "shared"


class CollectionCreateRequest(BaseModel):
    """Collection creation request model."""
    name: str = Field(..., min_length=1, max_length=100, description="Collection name")
    description: Optional[str] = Field(None, max_length=500, description="Collection description")
    visibility: CollectionVisibility = Field(CollectionVisibility.PRIVATE, description="Collection visibility")
    tags: Optional[List[str]] = Field(None, description="Collection tags")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    embedding_dimension: Optional[int] = Field(384, ge=1, le=4096, description="Embedding vector dimension")
    distance_metric: Optional[str] = Field("cosine", description="Distance metric for similarity search")
    
    @validator('name')
    def validate_name(cls, v):
        """Validate collection name."""
        if not v.strip():
            raise ValueError('Collection name cannot be empty')
        # Check for invalid characters
        invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
        if any(char in v for char in invalid_chars):
            raise ValueError(f'Collection name cannot contain: {", ".join(invalid_chars)}')
        return v.strip()


class CollectionUpdateRequest(BaseModel):
    """Collection update request model."""
    name: Optional[str] = Field(None, min_length=1, max_length=100, description="Collection name")
    description: Optional[str] = Field(None, max_length=500, description="Collection description")
    visibility: Optional[CollectionVisibility] = Field(None, description="Collection visibility")
    tags: Optional[List[str]] = Field(None, description="Collection tags")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class CollectionResponse(BaseModel):
    """Collection response model."""
    collection_id: str = Field(..., description="Collection ID")
    name: str = Field(..., description="Collection name")
    description: Optional[str] = Field(None, description="Collection description")
    visibility: CollectionVisibility = Field(..., description="Collection visibility")
    tags: Optional[List[str]] = Field(None, description="Collection tags")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Collection metadata")
    document_count: int = Field(..., description="Number of documents in collection")
    total_chunks: Optional[int] = Field(None, description="Total number of chunks")
    storage_size: Optional[str] = Field(None, description="Storage size used")
    embedding_dimension: int = Field(..., description="Embedding vector dimension")
    distance_metric: str = Field(..., description="Distance metric")
    created_at: datetime = Field(..., description="Collection creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    owner_id: str = Field(..., description="Collection owner ID")
    is_public: bool = Field(..., description="Whether collection is public")


class CollectionListResponse(BaseModel):
    """Collection list response model."""
    collections: List[CollectionResponse] = Field(..., description="List of collections")
    total: int = Field(..., description="Total number of collections")
    page: int = Field(..., description="Current page number")
    per_page: int = Field(..., description="Collections per page")
    has_next: bool = Field(..., description="Whether there are more pages")


class CollectionStatsResponse(BaseModel):
    """Collection statistics response model."""
    collection_id: str = Field(..., description="Collection ID")
    document_count: int = Field(..., description="Number of documents")
    total_chunks: int = Field(..., description="Total number of chunks")
    total_tokens: Optional[int] = Field(None, description="Total number of tokens")
    storage_size_bytes: int = Field(..., description="Storage size in bytes")
    storage_size_human: str = Field(..., description="Human-readable storage size")
    avg_document_size: Optional[float] = Field(None, description="Average document size")
    document_types: Dict[str, int] = Field(..., description="Document type distribution")
    language_distribution: Optional[Dict[str, int]] = Field(None, description="Language distribution")
    recent_activity: List[Dict[str, Any]] = Field(..., description="Recent activity in collection")
    created_at: datetime = Field(..., description="Collection creation timestamp")
    last_updated: Optional[datetime] = Field(None, description="Last update timestamp")


class CollectionMemberRequest(BaseModel):
    """Collection member management request model."""
    user_id: str = Field(..., description="User ID to add/remove")
    permission: str = Field(..., description="Permission level (read, write, admin)")


class CollectionMemberResponse(BaseModel):
    """Collection member response model."""
    user_id: str = Field(..., description="User ID")
    username: str = Field(..., description="Username")
    full_name: str = Field(..., description="User full name")
    permission: str = Field(..., description="Permission level")
    added_at: datetime = Field(..., description="When user was added to collection")
    added_by: str = Field(..., description="Who added the user")


class CollectionPermissionRequest(BaseModel):
    """Collection permission update request model."""
    user_id: str = Field(..., description="User ID")
    permission: str = Field(..., description="New permission level")
    
    @validator('permission')
    def validate_permission(cls, v):
        """Validate permission level."""
        valid_permissions = ['read', 'write', 'admin']
        if v not in valid_permissions:
            raise ValueError(f'Permission must be one of: {", ".join(valid_permissions)}')
        return v


class CollectionSearchRequest(BaseModel):
    """Collection search request model."""
    query: Optional[str] = Field(None, description="Search query")
    visibility: Optional[CollectionVisibility] = Field(None, description="Filter by visibility")
    tags: Optional[List[str]] = Field(None, description="Filter by tags")
    owner_id: Optional[str] = Field(None, description="Filter by owner")
    limit: Optional[int] = Field(20, ge=1, le=100, description="Maximum number of results")
    offset: Optional[int] = Field(0, ge=0, description="Result offset for pagination")
    sort_by: Optional[str] = Field("created_at", description="Sort field")
    sort_order: Optional[str] = Field("desc", description="Sort order (asc/desc)")


class CollectionDocumentRequest(BaseModel):
    """Add document to collection request model."""
    document_id: str = Field(..., description="Document ID to add")


class CollectionBulkOperationRequest(BaseModel):
    """Bulk operation request model."""
    document_ids: List[str] = Field(..., description="List of document IDs")
    operation: str = Field(..., description="Operation to perform (add/remove)")
    
    @validator('operation')
    def validate_operation(cls, v):
        """Validate operation type."""
        valid_operations = ['add', 'remove']
        if v not in valid_operations:
            raise ValueError(f'Operation must be one of: {", ".join(valid_operations)}')
        return v


class CollectionBulkOperationResponse(BaseModel):
    """Bulk operation response model."""
    operation: str = Field(..., description="Operation performed")
    total_requested: int = Field(..., description="Total documents requested")
    successful: int = Field(..., description="Successfully processed documents")
    failed: int = Field(..., description="Failed documents")
    errors: Optional[List[Dict[str, str]]] = Field(None, description="List of errors")
    processed_at: datetime = Field(..., description="Processing timestamp")
