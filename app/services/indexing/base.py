"""
Base interfaces and models for document indexing services.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import time
import uuid

class IndexingStatus(Enum):
    """Status of indexing operations."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"

@dataclass
class IndexingConfig:
    """Configuration for indexing operations."""
    # Collection settings
    collection_name: str = "documents"
    vector_dimension: int = 1536
    distance_metric: str = "cosine"
    
    # Processing settings
    batch_size: int = 100
    max_concurrent: int = 5
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Embedding settings
    embedding_model: Optional[str] = None
    embedding_type: str = "text"
    
    # Parsing settings
    extract_metadata: bool = True
    extract_sections: bool = True
    
    # Performance settings
    enable_caching: bool = True
    timeout: float = 300.0
    retry_attempts: int = 3
    retry_delay: float = 1.0
    
    # Storage settings
    auto_create_collection: bool = True
    overwrite_existing: bool = False
    
    def __post_init__(self):
        if not self.collection_name:
            raise ValueError("Collection name cannot be empty")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if self.vector_dimension <= 0:
            raise ValueError("Vector dimension must be positive")

@dataclass
class IndexingRequest:
    """Request for document indexing."""
    # Source specification
    content: Optional[str] = None
    file_path: Optional[str] = None
    file_data: Optional[bytes] = None
    directory_path: Optional[str] = None
    
    # Processing options
    document_id: Optional[str] = None
    document_type: Optional[str] = None
    language: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    # Configuration overrides
    config: Optional[IndexingConfig] = None
    
    def __post_init__(self):
        # Must have at least one source
        if not any([self.content, self.file_path, self.file_data, self.directory_path]):
            raise ValueError("Must provide content, file_path, file_data, or directory_path")
        
        # Generate document ID if not provided
        if not self.document_id:
            if self.file_path:
                self.document_id = str(Path(self.file_path).name)
            else:
                self.document_id = str(uuid.uuid4())
        
        if self.metadata is None:
            self.metadata = {}

@dataclass
class IndexedDocument:
    """A document that has been indexed."""
    document_id: str
    content: str
    chunks: List[Dict[str, Any]]
    embeddings: List[List[float]]
    metadata: Dict[str, Any]
    
    # Processing info
    parsing_time: float = 0.0
    chunking_time: float = 0.0
    embedding_time: float = 0.0
    storage_time: float = 0.0
    total_time: float = 0.0
    
    # Status info
    status: IndexingStatus = IndexingStatus.COMPLETED
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if not self.document_id:
            raise ValueError("Document ID cannot be empty")
        if not self.chunks:
            raise ValueError("Document must have at least one chunk")
        if len(self.chunks) != len(self.embeddings):
            raise ValueError("Number of chunks must match number of embeddings")

@dataclass
class IndexingProgress:
    """Progress tracking for indexing operations."""
    job_id: str
    status: IndexingStatus
    
    # Progress counters
    total_documents: int = 0
    processed_documents: int = 0
    successful_documents: int = 0
    failed_documents: int = 0
    
    # Time tracking
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    estimated_completion: Optional[float] = None
    
    # Current operation
    current_document: Optional[str] = None
    current_operation: Optional[str] = None
    
    # Statistics
    total_chunks: int = 0
    total_embeddings: int = 0
    total_processing_time: float = 0.0
    
    # Error tracking
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    @property
    def progress_percentage(self) -> float:
        """Calculate progress percentage."""
        if self.total_documents == 0:
            return 0.0
        return (self.processed_documents / self.total_documents) * 100
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.processed_documents == 0:
            return 0.0
        return (self.successful_documents / self.processed_documents) * 100
    
    @property
    def elapsed_time(self) -> float:
        """Calculate elapsed time."""
        end = self.end_time or time.time()
        return end - self.start_time
    
    @property
    def estimated_remaining_time(self) -> Optional[float]:
        """Estimate remaining time."""
        if self.processed_documents == 0 or self.total_documents == 0:
            return None
        
        elapsed = self.elapsed_time
        rate = self.processed_documents / elapsed
        remaining_docs = self.total_documents - self.processed_documents
        
        return remaining_docs / rate if rate > 0 else None

@dataclass
class IndexingResponse:
    """Response from indexing operations."""
    job_id: str
    status: IndexingStatus
    progress: IndexingProgress
    
    # Results
    indexed_documents: List[IndexedDocument] = field(default_factory=list)
    collection_name: Optional[str] = None
    
    # Summary statistics
    total_processing_time: float = 0.0
    total_chunks_created: int = 0
    total_embeddings_generated: int = 0
    total_vectors_stored: int = 0
    
    # Configuration used
    config: Optional[IndexingConfig] = None
    
    # Error information
    success: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

class IndexingError(Exception):
    """Base exception for indexing operations."""
    pass

class IndexingConfigError(IndexingError):
    """Exception for configuration errors."""
    def __init__(self, message: str, config_field: Optional[str] = None):
        self.config_field = config_field
        super().__init__(message)

class IndexingTimeoutError(IndexingError):
    """Exception for indexing timeouts."""
    def __init__(self, message: str, timeout: float):
        self.timeout = timeout
        super().__init__(message)

class IndexingResourceError(IndexingError):
    """Exception for resource-related errors."""
    def __init__(self, message: str, resource_type: str):
        self.resource_type = resource_type
        super().__init__(message)

# Progress callback type
ProgressCallback = Callable[[IndexingProgress], None]
