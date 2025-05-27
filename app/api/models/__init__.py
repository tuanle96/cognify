"""
API Models for Cognify

This package contains Pydantic models for API request/response validation.
"""

from .auth import (
    UserRegistrationRequest,
    UserLoginRequest,
    UserResponse,
    TokenResponse,
    PasswordResetRequest,
    PasswordChangeRequest
)

from .documents import (
    DocumentUploadRequest,
    DocumentResponse,
    DocumentListResponse,
    DocumentMetadata,
    IndexingJobResponse,
    BatchUploadResponse
)

from .collections import (
    CollectionCreateRequest,
    CollectionResponse,
    CollectionListResponse,
    CollectionStatsResponse,
    CollectionUpdateRequest
)

from .query import (
    QueryRequest,
    QueryResponse,
    SearchRequest,
    SearchResponse,
    QuerySuggestionResponse,
    QueryHistoryResponse
)

from .system import (
    HealthCheckResponse,
    SystemStatsResponse,
    SystemMetricsResponse,
    LogsResponse,
    SystemAlertsResponse
)

__all__ = [
    # Auth models
    "UserRegistrationRequest",
    "UserLoginRequest", 
    "UserResponse",
    "TokenResponse",
    "PasswordResetRequest",
    "PasswordChangeRequest",
    
    # Document models
    "DocumentUploadRequest",
    "DocumentResponse",
    "DocumentListResponse",
    "DocumentMetadata",
    "IndexingJobResponse",
    "BatchUploadResponse",
    
    # Collection models
    "CollectionCreateRequest",
    "CollectionResponse",
    "CollectionListResponse",
    "CollectionStatsResponse",
    "CollectionUpdateRequest",
    
    # Query models
    "QueryRequest",
    "QueryResponse",
    "SearchRequest",
    "SearchResponse",
    "QuerySuggestionResponse",
    "QueryHistoryResponse",
    
    # System models
    "HealthCheckResponse",
    "SystemStatsResponse",
    "SystemMetricsResponse",
    "LogsResponse",
    "SystemAlertsResponse",
]
