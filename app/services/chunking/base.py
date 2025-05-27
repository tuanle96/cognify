"""
Base classes and interfaces for chunking services.

Defines the core abstractions for intelligent chunking operations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import structlog

logger = structlog.get_logger(__name__)


class ChunkingStrategy(Enum):
    """Available chunking strategies."""
    AGENTIC = "agentic"
    AST_FALLBACK = "ast_fallback"
    HYBRID = "hybrid"


class ChunkType(Enum):
    """Types of code chunks."""
    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    MODULE = "module"
    IMPORT = "import"
    COMMENT = "comment"
    SEMANTIC_BLOCK = "semantic_block"
    # Additional types that LLM might return
    CLASS_DEFINITION = "class_definition"
    FUNCTION_DEFINITION = "function_definition"
    METHOD_DEFINITION = "method_definition"
    IMPORTS = "imports"
    VARIABLE = "variable"
    CONSTANT = "constant"
    INTERFACE = "interface"
    STRUCT = "struct"
    ENUM = "enum"
    NAMESPACE = "namespace"
    PACKAGE = "package"
    CONFIGURATION = "configuration"
    DOCUMENTATION = "documentation"
    TEST = "test"
    UTILITY = "utility"
    HELPER = "helper"
    MAIN = "main"
    INITIALIZATION = "initialization"
    UNKNOWN = "unknown"


@dataclass
class ChunkMetadata:
    """Metadata associated with a code chunk."""
    complexity: Optional[int] = None
    dependencies: List[str] = None
    author: Optional[str] = None
    last_modified: Optional[str] = None
    test_coverage: Optional[float] = None
    documentation_score: Optional[float] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class AgenticChunk:
    """
    Enhanced chunk with agentic processing metadata.
    """
    id: str
    content: str
    language: str
    chunk_type: ChunkType
    name: str
    start_line: int
    end_line: int
    file_path: str
    dependencies: List[str]
    semantic_relationships: List[str]
    purpose_optimization: Dict[str, Any]
    quality_score: float
    metadata: ChunkMetadata
    embedding: Optional[List[float]] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.semantic_relationships is None:
            self.semantic_relationships = []
        if self.purpose_optimization is None:
            self.purpose_optimization = {}
        if self.metadata is None:
            self.metadata = ChunkMetadata()

    @property
    def size_lines(self) -> int:
        """Get chunk size in lines."""
        return self.end_line - self.start_line + 1

    @property
    def size_chars(self) -> int:
        """Get chunk size in characters."""
        return len(self.content)

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary representation."""
        return {
            "id": self.id,
            "content": self.content,
            "language": self.language,
            "chunk_type": self.chunk_type.value,
            "name": self.name,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "file_path": self.file_path,
            "dependencies": self.dependencies,
            "semantic_relationships": self.semantic_relationships,
            "purpose_optimization": self.purpose_optimization,
            "quality_score": self.quality_score,
            "metadata": {
                "complexity": self.metadata.complexity,
                "dependencies": self.metadata.dependencies,
                "author": self.metadata.author,
                "last_modified": self.metadata.last_modified,
                "test_coverage": self.metadata.test_coverage,
                "documentation_score": self.metadata.documentation_score,
            },
            "size_lines": self.size_lines,
            "size_chars": self.size_chars,
        }


@dataclass
class ChunkingRequest:
    """Request for chunking operation."""
    content: str
    language: str
    file_path: str
    purpose: str = "general"
    quality_threshold: float = 0.8
    max_processing_time: int = 30
    force_agentic: bool = False
    context: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.context is None:
            self.context = {}


@dataclass
class ChunkingResult:
    """Result of chunking operation."""
    chunks: List[AgenticChunk]
    strategy_used: ChunkingStrategy
    quality_score: float
    processing_time: float
    metadata: Dict[str, Any]
    language_detected: Optional[str] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    @property
    def chunk_count(self) -> int:
        """Get number of chunks."""
        return len(self.chunks)

    @property
    def average_chunk_size(self) -> float:
        """Get average chunk size in lines."""
        if not self.chunks:
            return 0.0
        return sum(chunk.size_lines for chunk in self.chunks) / len(self.chunks)

    @property
    def total_lines(self) -> int:
        """Get total lines across all chunks."""
        return sum(chunk.size_lines for chunk in self.chunks)


class BaseChunker(ABC):
    """
    Abstract base class for chunking implementations.
    """

    def __init__(self, name: str):
        self.name = name
        self.logger = structlog.get_logger(f"chunker.{name}")

    @abstractmethod
    async def chunk(self, request: ChunkingRequest) -> ChunkingResult:
        """
        Perform chunking operation.

        Args:
            request: Chunking request

        Returns:
            ChunkingResult: Chunking results
        """
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check for the chunker.

        Returns:
            dict: Health check results
        """
        pass

    def log_performance(self, operation: str, duration: float, **kwargs):
        """Log performance metrics."""
        self.logger.info(
            f"Chunker performance: {operation}",
            duration_ms=duration * 1000,
            chunker=self.name,
            **kwargs
        )


class BaseQualityEvaluator(ABC):
    """
    Abstract base class for quality evaluation.
    """

    @abstractmethod
    async def evaluate_chunks(
        self,
        chunks: List[AgenticChunk],
        original_content: str,
        purpose: str
    ) -> Dict[str, float]:
        """
        Evaluate the quality of chunks.

        Args:
            chunks: List of chunks to evaluate
            original_content: Original content before chunking
            purpose: Chunking purpose

        Returns:
            dict: Quality scores by dimension
        """
        pass

    @abstractmethod
    async def calculate_overall_score(self, dimension_scores: Dict[str, float]) -> float:
        """
        Calculate overall quality score from dimension scores.

        Args:
            dimension_scores: Scores for each quality dimension

        Returns:
            float: Overall quality score
        """
        pass


class BaseCacheManager(ABC):
    """
    Abstract base class for chunking cache management.
    """

    @abstractmethod
    async def get_cached_result(
        self,
        content_hash: str,
        purpose: str,
        language: str
    ) -> Optional[ChunkingResult]:
        """
        Get cached chunking result.

        Args:
            content_hash: Hash of the content
            purpose: Chunking purpose
            language: Programming language

        Returns:
            ChunkingResult or None if not cached
        """
        pass

    @abstractmethod
    async def cache_result(
        self,
        content_hash: str,
        purpose: str,
        language: str,
        result: ChunkingResult
    ) -> None:
        """
        Cache chunking result.

        Args:
            content_hash: Hash of the content
            purpose: Chunking purpose
            language: Programming language
            result: Chunking result to cache
        """
        pass

    @abstractmethod
    async def invalidate_cache(self, pattern: str = None) -> int:
        """
        Invalidate cache entries.

        Args:
            pattern: Pattern to match for invalidation (optional)

        Returns:
            int: Number of entries invalidated
        """
        pass


# Utility functions

def generate_chunk_id(file_path: str, start_line: int, end_line: int) -> str:
    """
    Generate a unique chunk ID.

    Args:
        file_path: Path to the source file
        start_line: Starting line number
        end_line: Ending line number

    Returns:
        str: Unique chunk ID
    """
    import hashlib

    content = f"{file_path}:{start_line}:{end_line}"
    return hashlib.md5(content.encode()).hexdigest()[:12]


def calculate_content_hash(content: str) -> str:
    """
    Calculate hash of content for caching.

    Args:
        content: Content to hash

    Returns:
        str: Content hash
    """
    import hashlib

    return hashlib.sha256(content.encode()).hexdigest()


def normalize_chunk_type(chunk_type_str: str) -> ChunkType:
    """
    Normalize chunk type string to ChunkType enum.

    Args:
        chunk_type_str: String representation of chunk type

    Returns:
        ChunkType: Normalized chunk type
    """
    # Convert to lowercase and replace spaces/hyphens with underscores
    normalized = chunk_type_str.lower().replace(' ', '_').replace('-', '_')

    # Try direct mapping first
    try:
        return ChunkType(normalized)
    except ValueError:
        pass

    # Try common mappings
    mappings = {
        'func': ChunkType.FUNCTION,
        'fn': ChunkType.FUNCTION,
        'def': ChunkType.FUNCTION,
        'cls': ChunkType.CLASS,
        'klass': ChunkType.CLASS,
        'meth': ChunkType.METHOD,
        'mod': ChunkType.MODULE,
        'imp': ChunkType.IMPORT,
        'block': ChunkType.SEMANTIC_BLOCK,
        'code_block': ChunkType.SEMANTIC_BLOCK,
        'semantic': ChunkType.SEMANTIC_BLOCK,
    }

    if normalized in mappings:
        return mappings[normalized]

    # Fallback to UNKNOWN
    return ChunkType.UNKNOWN


def validate_chunk(chunk: AgenticChunk) -> List[str]:
    """
    Validate chunk data integrity.

    Args:
        chunk: Chunk to validate

    Returns:
        list: List of validation errors (empty if valid)
    """
    errors = []

    if not chunk.id:
        errors.append("Chunk ID is required")

    if not chunk.content:
        errors.append("Chunk content is required")

    if chunk.start_line < 1:
        errors.append("Start line must be >= 1")

    if chunk.end_line < chunk.start_line:
        errors.append("End line must be >= start line")

    if not (0 <= chunk.quality_score <= 1):
        errors.append("Quality score must be between 0 and 1")

    return errors


# Alias for backward compatibility
ChunkingResponse = ChunkingResult
