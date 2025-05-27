"""
Chunking API endpoints for V1.

Provides intelligent code chunking operations as part of the V1 API.
This module handles parsing, analyzing, and breaking down source code
into semantically meaningful chunks using various strategies.
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

from app.services.chunking.service import ChunkingService
from app.services.chunking.base import ChunkingRequest, ChunkingResult
from app.services.embedding.service import EmbeddingService

# Simple logger replacement
class SimpleLogger:
    def info(self, msg, **kwargs):
        print(f"INFO: {msg} {kwargs}")
    def debug(self, msg, **kwargs):
        print(f"DEBUG: {msg} {kwargs}")
    def warning(self, msg, **kwargs):
        print(f"WARNING: {msg} {kwargs}")
    def error(self, msg, **kwargs):
        print(f"ERROR: {msg} {kwargs}")

logger = SimpleLogger()

# Create chunking router for V1
router = APIRouter()


# Request/Response models

class ChunkingRequestModel(BaseModel):
    """Request model for chunking operations."""
    content: str = Field(..., description="Source code content to chunk")
    language: Optional[str] = Field(None, description="Programming language (auto-detect if not provided)")
    strategy: Optional[str] = Field("hybrid", description="Chunking strategy: ast, hybrid, or agentic")
    max_chunk_size: Optional[int] = Field(1000, description="Maximum chunk size in characters")
    overlap_size: Optional[int] = Field(100, description="Overlap size between chunks")
    include_metadata: Optional[bool] = Field(True, description="Include metadata in response")
    include_embeddings: Optional[bool] = Field(False, description="Include embeddings for each chunk")
    embedding_provider: Optional[str] = Field("openai", description="Embedding provider (openai, voyage, cohere)")


class ChunkMetadata(BaseModel):
    """Metadata for a code chunk."""
    start_line: int = Field(..., description="Starting line number")
    end_line: int = Field(..., description="Ending line number")
    chunk_type: str = Field(..., description="Type of chunk (function, class, etc.)")
    language: str = Field(..., description="Programming language")
    complexity: Optional[float] = Field(None, description="Code complexity score")
    dependencies: Optional[List[str]] = Field(None, description="Dependencies identified")


class ChunkModel(BaseModel):
    """Model for a single code chunk."""
    content: str = Field(..., description="Chunk content")
    metadata: ChunkMetadata = Field(..., description="Chunk metadata")
    chunk_id: str = Field(..., description="Unique chunk identifier")
    embedding: Optional[List[float]] = Field(None, description="Embedding vector for the chunk")


class ChunkingResponseModel(BaseModel):
    """Response model for chunking operations."""
    chunks: List[ChunkModel] = Field(..., description="Generated chunks")
    total_chunks: int = Field(..., description="Total number of chunks")
    processing_time: float = Field(..., description="Processing time in seconds")
    strategy_used: str = Field(..., description="Chunking strategy that was used")
    language_detected: str = Field(..., description="Detected programming language")
    metadata: Dict[str, Any] = Field(..., description="Additional processing metadata")


class PerformanceStatsResponse(BaseModel):
    """Response model for performance statistics."""
    total_requests: int = Field(..., description="Total chunking requests processed")
    avg_processing_time: float = Field(..., description="Average processing time in seconds")
    success_rate: float = Field(..., description="Success rate percentage")
    supported_languages: List[str] = Field(..., description="List of supported languages")
    active_strategies: List[str] = Field(..., description="Available chunking strategies")


# Dependency injection
async def get_chunking_service(request: Request) -> ChunkingService:
    """Get chunking service instance from app state."""
    try:
        if hasattr(request.app.state, 'chunking_service'):
            return request.app.state.chunking_service

        # Fallback: create new instance
        service = ChunkingService()
        if not service._initialized:
            await service.initialize()
        return service
    except Exception as e:
        logger.error("Failed to get chunking service", error=str(e))
        raise HTTPException(status_code=503, detail="Chunking service not initialized")


# Chunking endpoints

@router.post("/", response_model=ChunkingResponseModel)
async def chunk_content(
    chunk_request: ChunkingRequestModel,
    request: Request,
    chunking_service: ChunkingService = Depends(get_chunking_service)
):
    """
    Chunk source code content into semantically meaningful pieces.

    This endpoint accepts source code and returns it broken down into chunks
    using the specified strategy (AST, Hybrid, or Agentic).
    """
    try:
        logger.info("Processing chunking request",
                   language=chunk_request.language,
                   strategy=chunk_request.strategy,
                   content_length=len(chunk_request.content))

        # Create chunking request
        chunking_request = ChunkingRequest(
            content=chunk_request.content,
            language=chunk_request.language or "python",
            file_path="<api_request>",
            purpose="api_chunking"
        )

        # Process chunking
        result = await chunking_service.chunk_content(chunking_request)

        # Generate embeddings if requested
        if chunk_request.include_embeddings:
            try:
                from app.services.embedding.service import embedding_service

                # Initialize embedding service
                await embedding_service.initialize()

                # Generate embeddings for each chunk
                chunk_texts = [chunk.content if hasattr(chunk, 'content') else chunk.get('content', '')
                              for chunk in result.chunks]

                if chunk_texts:
                    # Use LiteLLM embedding service
                    chunk_embeddings = await embedding_service.get_embeddings(chunk_texts)
                else:
                    chunk_embeddings = []

            except Exception as e:
                logger.warning("Failed to generate embeddings", error=str(e))
                chunk_embeddings = []
        else:
            chunk_embeddings = []

        # Convert to response model
        chunks = []
        for i, chunk in enumerate(result.chunks):
            # Handle both dict and object chunk formats
            if hasattr(chunk, 'content'):
                # AgenticChunk object format
                # Handle chunk_type enum
                chunk_type_value = getattr(chunk, 'chunk_type', 'unknown')
                if hasattr(chunk_type_value, 'value'):
                    chunk_type_value = chunk_type_value.value

                # Handle metadata object
                complexity = None
                if hasattr(chunk, 'metadata') and chunk.metadata:
                    complexity = getattr(chunk.metadata, 'complexity', None)

                chunk_metadata = ChunkMetadata(
                    start_line=getattr(chunk, 'start_line', 0),
                    end_line=getattr(chunk, 'end_line', 0),
                    chunk_type=str(chunk_type_value),
                    language=chunk_request.language or 'unknown',
                    complexity=complexity,
                    dependencies=getattr(chunk, 'dependencies', [])
                )
                chunk_content = chunk.content
            else:
                # Dictionary format
                chunk_metadata = ChunkMetadata(
                    start_line=chunk.get('start_line', 0),
                    end_line=chunk.get('end_line', 0),
                    chunk_type=chunk.get('type', 'unknown'),
                    language=chunk_request.language or 'unknown',
                    complexity=chunk.get('complexity'),
                    dependencies=chunk.get('dependencies', [])
                )
                chunk_content = chunk.get('content', '')

            # Get embedding for this chunk if available
            chunk_embedding = None
            if chunk_embeddings and i < len(chunk_embeddings):
                chunk_embedding = chunk_embeddings[i]

            chunk_model = ChunkModel(
                content=chunk_content,
                metadata=chunk_metadata,
                chunk_id=f"chunk_{i}",
                embedding=chunk_embedding
            )
            chunks.append(chunk_model)

        response = ChunkingResponseModel(
            chunks=chunks,
            total_chunks=len(chunks),
            processing_time=result.processing_time,
            strategy_used=result.strategy_used.value if hasattr(result.strategy_used, 'value') else str(result.strategy_used),
            language_detected=chunk_request.language or 'unknown',
            metadata=result.metadata or {}
        )

        logger.info("Chunking completed successfully",
                   chunks_created=len(chunks),
                   processing_time=result.processing_time)

        return response

    except ValueError as e:
        logger.warning("Invalid chunking request", error=str(e))
        raise HTTPException(status_code=400, detail=f"Invalid request: {str(e)}")
    except Exception as e:
        logger.error("Chunking failed", error=str(e))
        if "timeout" in str(e).lower():
            raise HTTPException(status_code=408, detail=f"Chunking timeout: {str(e)}")
        else:
            raise HTTPException(status_code=500, detail=f"Chunking failed: {str(e)}")


@router.get("/health", response_model=Dict[str, Any])
async def chunking_health_check(
    request: Request,
    chunking_service: ChunkingService = Depends(get_chunking_service)
):
    """
    Check the health status of the chunking service.
    """
    try:
        # Perform a simple health check
        health_status = await chunking_service.health_check()
        return health_status
    except Exception as e:
        logger.error("Chunking health check failed", error=str(e))
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")


@router.get("/stats", response_model=PerformanceStatsResponse)
async def chunking_performance_stats(
    request: Request,
    chunking_service: ChunkingService = Depends(get_chunking_service)
):
    """
    Get performance statistics for the chunking service.
    """
    try:
        stats = await chunking_service.get_performance_stats()

        # Map service stats to response model
        total_requests = stats.get("total_requests", 0)
        success_rate = 100.0 if total_requests > 0 else 0.0  # Assume all requests succeed for now

        response_stats = PerformanceStatsResponse(
            total_requests=total_requests,
            avg_processing_time=stats.get("average_processing_time", 0.0),
            success_rate=success_rate,
            supported_languages=[
                "python", "javascript", "typescript", "java", "cpp", "c",
                "csharp", "go", "rust", "php", "ruby", "swift", "kotlin",
                "scala", "r", "matlab", "sql", "html", "css", "xml", "json",
                "yaml", "markdown", "shell", "dockerfile", "terraform"
            ],
            active_strategies=["ast", "hybrid", "agentic"]
        )

        return response_stats
    except Exception as e:
        logger.error("Failed to get chunking stats", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@router.get("/supported-languages")
async def get_supported_languages():
    """
    Get list of supported programming languages for chunking.
    """
    return {
        "supported_languages": [
            "python", "javascript", "typescript", "java", "cpp", "c",
            "csharp", "go", "rust", "php", "ruby", "swift", "kotlin",
            "scala", "r", "matlab", "sql", "html", "css", "xml", "json",
            "yaml", "markdown", "shell", "dockerfile", "terraform"
        ],
        "auto_detection": True,
        "custom_parsers": ["ast", "tree-sitter", "regex"],
        "strategies": ["ast", "hybrid", "agentic"]
    }


@router.get("/supported-strategies")
async def get_supported_strategies():
    """
    Get list of supported chunking strategies.
    """
    return {
        "strategies": ["ast", "hybrid", "agentic"],
        "default_strategy": "hybrid",
        "strategy_descriptions": {
            "ast": "AST-based chunking using abstract syntax tree analysis",
            "hybrid": "Hybrid approach combining AST and heuristic methods",
            "agentic": "AI-powered intelligent chunking with context awareness"
        },
        "fallback_order": ["ast", "hybrid", "agentic"]
    }


@router.post("/test")
async def test_chunking(
    request: Request,
    chunking_service: ChunkingService = Depends(get_chunking_service)
):
    """
    Test endpoint for chunking service with sample code.
    """
    sample_code = '''
def fibonacci(n):
    """Calculate fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

class Calculator:
    """Simple calculator class."""

    def add(self, a, b):
        return a + b

    def multiply(self, a, b):
        return a * b
'''

    try:
        request = ChunkingRequest(
            content=sample_code,
            language="python",
            file_path="<test_request>",
            purpose="test_chunking"
        )

        result = await chunking_service.chunk_content(request)

        return {
            "test_status": "success",
            "sample_code": sample_code,
            "chunks_created": len(result.chunks),
            "processing_time": result.processing_time,
            "strategy_used": result.strategy_used,
            "language_detected": result.language_detected
        }
    except Exception as e:
        logger.error("Test chunking failed", error=str(e))
        return {
            "test_status": "failed",
            "error": str(e)
        }
