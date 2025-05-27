"""
Base interfaces and models for document retrieval services.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import time

class QueryType(Enum):
    """Types of queries for retrieval."""
    SEMANTIC = "semantic"  # Vector similarity search
    KEYWORD = "keyword"    # Traditional keyword search
    HYBRID = "hybrid"      # Combination of semantic and keyword
    CODE = "code"          # Code-specific search
    QUESTION = "question"  # Question answering
    SUMMARY = "summary"    # Document summarization

class RetrievalStrategy(Enum):
    """Retrieval strategies."""
    VECTOR_ONLY = "vector_only"
    KEYWORD_ONLY = "keyword_only"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"  # Automatically choose best strategy

@dataclass
class RetrievalConfig:
    """Configuration for retrieval operations."""
    # Search settings
    max_results: int = 10
    min_score: float = 0.0
    collection_name: str = "documents"
    
    # Vector search settings
    vector_weight: float = 0.7
    enable_vector_search: bool = True
    
    # Keyword search settings
    keyword_weight: float = 0.3
    enable_keyword_search: bool = True
    
    # Hybrid settings
    fusion_method: str = "rrf"  # reciprocal rank fusion
    
    # Re-ranking settings
    enable_reranking: bool = True
    rerank_top_k: int = 50
    final_top_k: int = 10
    
    # Query processing
    enable_query_expansion: bool = True
    enable_query_understanding: bool = True
    
    # Filtering
    metadata_filters: Optional[Dict[str, Any]] = None
    language_filter: Optional[str] = None
    document_type_filter: Optional[str] = None
    
    # Performance
    timeout: float = 30.0
    enable_caching: bool = True
    cache_ttl: int = 3600
    
    def __post_init__(self):
        if self.max_results <= 0:
            raise ValueError("max_results must be positive")
        if not (0 <= self.min_score <= 1):
            raise ValueError("min_score must be between 0 and 1")
        if not (0 <= self.vector_weight <= 1):
            raise ValueError("vector_weight must be between 0 and 1")
        if not (0 <= self.keyword_weight <= 1):
            raise ValueError("keyword_weight must be between 0 and 1")

@dataclass
class RetrievalRequest:
    """Request for document retrieval."""
    query: str
    query_type: QueryType = QueryType.SEMANTIC
    strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    config: Optional[RetrievalConfig] = None
    
    # Context for better retrieval
    context: Optional[str] = None
    conversation_history: Optional[List[str]] = None
    
    # Additional metadata
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    
    def __post_init__(self):
        if not self.query or not self.query.strip():
            raise ValueError("Query cannot be empty")
        if self.config is None:
            self.config = RetrievalConfig()

@dataclass
class RetrievalResult:
    """A single retrieval result."""
    # Content information
    document_id: str
    chunk_id: str
    content: str
    
    # Scoring information
    score: float
    vector_score: Optional[float] = None
    keyword_score: Optional[float] = None
    rerank_score: Optional[float] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Source information
    source_file: Optional[str] = None
    chunk_index: Optional[int] = None
    language: Optional[str] = None
    document_type: Optional[str] = None
    
    # Highlighting
    highlighted_content: Optional[str] = None
    matched_keywords: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.document_id:
            raise ValueError("document_id cannot be empty")
        if not self.chunk_id:
            raise ValueError("chunk_id cannot be empty")
        if not (0 <= self.score <= 1):
            raise ValueError("score must be between 0 and 1")

@dataclass
class RetrievalResponse:
    """Response from retrieval operations."""
    # Results
    results: List[RetrievalResult]
    
    # Query information
    original_query: str
    processed_query: Optional[str] = None
    query_type: QueryType = QueryType.SEMANTIC
    strategy_used: RetrievalStrategy = RetrievalStrategy.HYBRID
    
    # Performance metrics
    total_results: int = 0
    processing_time: float = 0.0
    vector_search_time: float = 0.0
    keyword_search_time: float = 0.0
    rerank_time: float = 0.0
    
    # Search statistics
    vector_results_count: int = 0
    keyword_results_count: int = 0
    fusion_results_count: int = 0
    
    # Metadata
    collection_searched: Optional[str] = None
    filters_applied: Optional[Dict[str, Any]] = None
    
    # Status
    success: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Caching info
    from_cache: bool = False
    cache_key: Optional[str] = None
    
    @property
    def result_count(self) -> int:
        """Get the number of results."""
        return len(self.results)
    
    @property
    def has_results(self) -> bool:
        """Check if there are any results."""
        return len(self.results) > 0
    
    @property
    def top_score(self) -> float:
        """Get the highest score among results."""
        if not self.results:
            return 0.0
        return max(result.score for result in self.results)

class RetrievalError(Exception):
    """Base exception for retrieval operations."""
    pass

class QueryProcessingError(RetrievalError):
    """Exception for query processing errors."""
    def __init__(self, message: str, query: str, original_error: Optional[Exception] = None):
        self.query = query
        self.original_error = original_error
        super().__init__(message)

class SearchError(RetrievalError):
    """Exception for search operation errors."""
    def __init__(self, message: str, search_type: str, original_error: Optional[Exception] = None):
        self.search_type = search_type
        self.original_error = original_error
        super().__init__(message)

class ReRankingError(RetrievalError):
    """Exception for re-ranking errors."""
    def __init__(self, message: str, reranker_type: str, original_error: Optional[Exception] = None):
        self.reranker_type = reranker_type
        self.original_error = original_error
        super().__init__(message)
