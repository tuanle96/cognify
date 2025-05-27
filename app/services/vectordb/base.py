"""
Base interfaces and models for vector database services.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import uuid

class VectorDBProvider(Enum):
    """Supported vector database providers."""
    QDRANT = "qdrant"
    MILVUS = "milvus"
    CHROMA = "chroma"
    PINECONE = "pinecone"

@dataclass
class VectorPoint:
    """A point in vector space with metadata."""
    id: str
    vector: List[float]
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.vector:
            raise ValueError("vector cannot be empty")
        if self.metadata is None:
            self.metadata = {}

@dataclass
class SearchRequest:
    """Request for vector similarity search."""
    vector: List[float]
    limit: int = 10
    score_threshold: Optional[float] = None
    filter_conditions: Optional[Dict[str, Any]] = None
    include_metadata: bool = True
    include_vectors: bool = False

    def __post_init__(self):
        if not self.vector:
            raise ValueError("search vector cannot be empty")
        if self.limit <= 0:
            raise ValueError("limit must be positive")
        if self.score_threshold is not None and not (0 <= self.score_threshold <= 1):
            raise ValueError("score_threshold must be between 0 and 1")

@dataclass
class SearchResult:
    """Result from vector similarity search."""
    id: str
    score: float
    vector: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if not self.id:
            raise ValueError("result id cannot be empty")
        if not (0 <= self.score <= 1):
            raise ValueError("score must be between 0 and 1")

@dataclass
class CollectionInfo:
    """Information about a vector collection."""
    name: str
    dimension: int
    vector_count: int
    distance_metric: str = "cosine"
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if not self.name:
            raise ValueError("collection name cannot be empty")
        if self.dimension <= 0:
            raise ValueError("dimension must be positive")
        if self.vector_count < 0:
            raise ValueError("vector_count cannot be negative")

@dataclass
class VectorDBConfig:
    """Configuration for vector database."""
    provider: VectorDBProvider
    host: str = "localhost"
    port: int = 6333
    api_key: Optional[str] = None
    timeout: int = 30
    max_retries: int = 3
    metadata: Optional[Dict[str, Any]] = None

class VectorDBClient(ABC):
    """Abstract base class for vector database clients."""

    def __init__(self, host: str, port: int, **kwargs):
        self.host = host
        self.port = port
        self.config = kwargs
        self._initialized = False

    @property
    @abstractmethod
    def provider(self) -> VectorDBProvider:
        """Get the provider type."""
        pass

    @property
    @abstractmethod
    def max_batch_size(self) -> int:
        """Get maximum batch size for operations."""
        pass

    @property
    @abstractmethod
    def supported_distance_metrics(self) -> List[str]:
        """Get supported distance metrics."""
        pass

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the client connection."""
        pass

    @abstractmethod
    async def create_collection(
        self,
        name: str,
        dimension: int,
        distance_metric: str = "cosine",
        **kwargs
    ) -> bool:
        """Create a new collection."""
        pass

    @abstractmethod
    async def delete_collection(self, name: str) -> bool:
        """Delete a collection."""
        pass

    @abstractmethod
    async def collection_exists(self, name: str) -> bool:
        """Check if collection exists."""
        pass

    @abstractmethod
    async def get_collection_info(self, name: str) -> CollectionInfo:
        """Get information about a collection."""
        pass

    @abstractmethod
    async def list_collections(self) -> List[str]:
        """List all collections."""
        pass

    @abstractmethod
    async def insert_points(self, collection_name: str, points: List[VectorPoint]) -> bool:
        """Insert points into a collection."""
        pass

    @abstractmethod
    async def update_points(self, collection_name: str, points: List[VectorPoint]) -> bool:
        """Update existing points in a collection."""
        pass

    @abstractmethod
    async def delete_points(self, collection_name: str, point_ids: List[str]) -> bool:
        """Delete points from a collection."""
        pass

    @abstractmethod
    async def search(self, collection_name: str, request: SearchRequest) -> List[SearchResult]:
        """Search for similar vectors."""
        pass

    @abstractmethod
    async def get_point(self, collection_name: str, point_id: str) -> Optional[VectorPoint]:
        """Get a specific point by ID."""
        pass

    @abstractmethod
    async def count_points(self, collection_name: str) -> int:
        """Count points in a collection."""
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Check the health of the vector database."""
        pass

    async def cleanup(self) -> None:
        """Cleanup resources."""
        pass

    def _validate_collection_name(self, name: str) -> None:
        """Validate collection name."""
        if not name or not name.strip():
            raise ValueError("Collection name cannot be empty")
        if len(name) > 255:
            raise ValueError("Collection name too long (max 255 characters)")
        # Basic validation - can be extended per provider
        if not name.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Collection name must be alphanumeric with underscores/hyphens")

    def _validate_dimension(self, dimension: int) -> None:
        """Validate vector dimension."""
        if dimension <= 0:
            raise ValueError("Dimension must be positive")
        if dimension > 65536:  # Reasonable upper limit
            raise ValueError("Dimension too large (max 65536)")

    def _validate_distance_metric(self, metric: str) -> None:
        """Validate distance metric."""
        if metric not in self.supported_distance_metrics:
            raise ValueError(f"Unsupported distance metric: {metric}")

class VectorDBError(Exception):
    """Base exception for vector database operations."""
    pass

class VectorDBConnectionError(VectorDBError):
    """Exception for connection errors."""
    def __init__(self, provider: VectorDBProvider, message: str, original_error: Optional[Exception] = None):
        self.provider = provider
        self.original_error = original_error
        super().__init__(f"{provider.value}: {message}")

class VectorDBOperationError(VectorDBError):
    """Exception for operation errors."""
    def __init__(self, provider: VectorDBProvider, operation: str, message: str):
        self.provider = provider
        self.operation = operation
        super().__init__(f"{provider.value} {operation}: {message}")

class CollectionNotFoundError(VectorDBError):
    """Exception for collection not found."""
    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        super().__init__(f"Collection '{collection_name}' not found")

class CollectionAlreadyExistsError(VectorDBError):
    """Exception for collection already exists."""
    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        super().__init__(f"Collection '{collection_name}' already exists")

# Alias for backward compatibility
VectorDBService = VectorDBClient
