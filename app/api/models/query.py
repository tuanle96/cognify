"""
Query and search API models.

Pydantic models for RAG queries, search operations, and result processing.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from enum import Enum


class QueryType(str, Enum):
    """Query type enumeration."""
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"
    RAG = "rag"
    SIMILARITY = "similarity"


class SearchScope(str, Enum):
    """Search scope enumeration."""
    ALL = "all"
    COLLECTION = "collection"
    DOCUMENT = "document"
    USER = "user"


class QueryRequest(BaseModel):
    """Query request model."""
    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    query_type: str = Field("semantic", description="Type of query to perform")
    collection_name: Optional[str] = Field(None, description="Collection name to search in")
    collection_id: Optional[str] = Field(None, description="Collection ID to search in")
    max_results: Optional[int] = Field(10, ge=1, le=100, description="Maximum number of results")
    min_score: Optional[float] = Field(0.0, ge=0.0, le=1.0, description="Minimum similarity score")
    rerank_results: Optional[bool] = Field(True, description="Whether to rerank results")
    include_metadata: Optional[bool] = Field(True, description="Include result metadata")
    document_ids: Optional[List[str]] = Field(None, description="Specific documents to search")
    scope: SearchScope = Field(SearchScope.ALL, description="Search scope")
    limit: Optional[int] = Field(10, ge=1, le=100, description="Maximum number of results")
    offset: Optional[int] = Field(0, ge=0, description="Result offset for pagination")
    include_content: Optional[bool] = Field(True, description="Include content in results")
    similarity_threshold: Optional[float] = Field(0.7, ge=0.0, le=1.0, description="Minimum similarity score")
    filters: Optional[Dict[str, Any]] = Field(None, description="Additional filters")

    @validator('query')
    def validate_query(cls, v):
        """Validate query string."""
        if not v.strip():
            raise ValueError('Query cannot be empty')
        return v.strip()


class SearchRequest(BaseModel):
    """Search request model."""
    query: str = Field(..., min_length=1, description="Search query")
    collection_name: Optional[str] = Field(None, description="Collection name to search in")
    collection_id: Optional[str] = Field(None, description="Collection ID to search in")
    max_results: Optional[int] = Field(10, ge=1, le=100, description="Maximum number of results")
    min_score: Optional[float] = Field(0.0, ge=0.0, le=1.0, description="Minimum similarity score")
    document_types: Optional[List[str]] = Field(None, description="Document types to include")
    tags: Optional[List[str]] = Field(None, description="Tags to filter by")
    date_from: Optional[datetime] = Field(None, description="Filter documents from this date")
    date_to: Optional[datetime] = Field(None, description="Filter documents to this date")
    limit: Optional[int] = Field(10, ge=1, le=100, description="Maximum number of results")
    offset: Optional[int] = Field(0, ge=0, description="Result offset for pagination")
    sort_by: Optional[str] = Field("relevance", description="Sort field")
    sort_order: Optional[str] = Field("desc", description="Sort order")


class QueryResult(BaseModel):
    """Individual query result model."""
    document_id: str = Field(..., description="Document ID")
    chunk_id: Optional[str] = Field(None, description="Chunk ID")
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Relevant content")
    score: float = Field(..., description="Relevance score")
    document_type: str = Field(..., description="Document type")
    collection_id: Optional[str] = Field(None, description="Collection ID")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Result metadata")
    highlights: Optional[List[str]] = Field(None, description="Highlighted text snippets")
    chunk_index: Optional[int] = Field(None, description="Chunk index in document")
    created_at: datetime = Field(..., description="Document creation timestamp")


class QueryResponse(BaseModel):
    """Query response model."""
    query_id: str = Field(..., description="Query ID")
    query: str = Field(..., description="Original query")
    collection_name: str = Field(..., description="Collection searched")
    results: List[Dict[str, Any]] = Field(..., description="Query results")
    result_count: int = Field(..., description="Number of results returned")
    processing_time: float = Field(..., description="Processing time in seconds")
    query_type: str = Field(..., description="Query type used")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Query metadata")
    timestamp: datetime = Field(..., description="Query execution timestamp")


class SearchResponse(BaseModel):
    """Search response model."""
    query: str = Field(..., description="Search query")
    results: List[QueryResult] = Field(..., description="Search results")
    total_results: int = Field(..., description="Total number of results")
    search_time: float = Field(..., description="Search time in seconds")
    page: int = Field(..., description="Current page number")
    per_page: int = Field(..., description="Results per page")
    facets: Optional[Dict[str, List[Dict[str, Any]]]] = Field(None, description="Search facets")


class QuerySuggestionResponse(BaseModel):
    """Query suggestion response model."""
    suggestions: List[str] = Field(..., description="Query suggestions")
    popular_queries: Optional[List[str]] = Field(None, description="Popular queries")
    related_queries: Optional[List[str]] = Field(None, description="Related queries")
    autocomplete: Optional[List[str]] = Field(None, description="Autocomplete suggestions")


class QueryHistoryItem(BaseModel):
    """Query history item model."""
    query_id: str = Field(..., description="Query ID")
    query: str = Field(..., description="Query text")
    query_type: QueryType = Field(..., description="Query type")
    results_count: int = Field(..., description="Number of results returned")
    search_time: float = Field(..., description="Search time in seconds")
    executed_at: datetime = Field(..., description="Query execution timestamp")
    collection_id: Optional[str] = Field(None, description="Collection searched")


class QueryHistoryResponse(BaseModel):
    """Query history response model."""
    queries: List[QueryHistoryItem] = Field(..., description="Query history")
    total: int = Field(..., description="Total number of queries")
    page: int = Field(..., description="Current page number")
    per_page: int = Field(..., description="Queries per page")
    has_next: bool = Field(..., description="Whether there are more queries")


class QueryAnalyticsRequest(BaseModel):
    """Query analytics request model."""
    date_from: Optional[datetime] = Field(None, description="Analytics from this date")
    date_to: Optional[datetime] = Field(None, description="Analytics to this date")
    collection_id: Optional[str] = Field(None, description="Filter by collection")
    query_type: Optional[QueryType] = Field(None, description="Filter by query type")
    group_by: Optional[str] = Field("day", description="Group results by (hour/day/week/month)")


class QueryAnalyticsResponse(BaseModel):
    """Query analytics response model."""
    total_queries: int = Field(..., description="Total number of queries")
    unique_queries: int = Field(..., description="Number of unique queries")
    avg_search_time: float = Field(..., description="Average search time")
    popular_queries: List[Dict[str, Any]] = Field(..., description="Most popular queries")
    query_trends: List[Dict[str, Any]] = Field(..., description="Query trends over time")
    performance_metrics: Dict[str, float] = Field(..., description="Performance metrics")
    date_range: Dict[str, datetime] = Field(..., description="Date range analyzed")


class SavedQueryRequest(BaseModel):
    """Saved query request model."""
    name: str = Field(..., min_length=1, max_length=100, description="Query name")
    query: str = Field(..., description="Query text")
    query_type: QueryType = Field(..., description="Query type")
    description: Optional[str] = Field(None, description="Query description")
    collection_id: Optional[str] = Field(None, description="Default collection")
    filters: Optional[Dict[str, Any]] = Field(None, description="Default filters")
    is_public: Optional[bool] = Field(False, description="Whether query is public")


class SavedQueryResponse(BaseModel):
    """Saved query response model."""
    query_id: str = Field(..., description="Saved query ID")
    name: str = Field(..., description="Query name")
    query: str = Field(..., description="Query text")
    query_type: QueryType = Field(..., description="Query type")
    description: Optional[str] = Field(None, description="Query description")
    collection_id: Optional[str] = Field(None, description="Default collection")
    filters: Optional[Dict[str, Any]] = Field(None, description="Default filters")
    is_public: bool = Field(..., description="Whether query is public")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    owner_id: str = Field(..., description="Query owner ID")
    usage_count: int = Field(..., description="Number of times used")
