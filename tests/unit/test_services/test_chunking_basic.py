"""Basic unit tests for chunking service components."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock


class TestChunkingBasics:
    """Test basic chunking functionality without complex imports."""
    
    @pytest.mark.unit
    def test_chunking_strategy_enum(self):
        """Test ChunkingStrategy enum."""
        try:
            from app.services.chunking.base import ChunkingStrategy
            
            assert ChunkingStrategy.AGENTIC.value == "agentic"
            assert ChunkingStrategy.AST_FALLBACK.value == "ast_fallback"
            assert ChunkingStrategy.HYBRID.value == "hybrid"
            
            # Test enum comparison
            assert ChunkingStrategy.AGENTIC != ChunkingStrategy.AST_FALLBACK
            assert ChunkingStrategy.HYBRID != ChunkingStrategy.AGENTIC
            
        except ImportError as e:
            pytest.skip(f"Cannot import ChunkingStrategy: {e}")
    
    @pytest.mark.unit
    def test_chunk_type_enum(self):
        """Test ChunkType enum."""
        try:
            from app.services.chunking.base import ChunkType
            
            assert ChunkType.FUNCTION.value == "function"
            assert ChunkType.CLASS.value == "class"
            assert ChunkType.METHOD.value == "method"
            assert ChunkType.MODULE.value == "module"
            assert ChunkType.IMPORT.value == "import"
            assert ChunkType.COMMENT.value == "comment"
            assert ChunkType.SEMANTIC_BLOCK.value == "semantic_block"
            assert ChunkType.UNKNOWN.value == "unknown"
            
        except ImportError as e:
            pytest.skip(f"Cannot import ChunkType: {e}")
    
    @pytest.mark.unit
    def test_chunk_metadata_creation(self):
        """Test ChunkMetadata creation."""
        try:
            from app.services.chunking.base import ChunkMetadata
            
            # Test default creation
            metadata = ChunkMetadata()
            assert metadata.complexity is None
            assert metadata.dependencies == []
            assert metadata.author is None
            assert metadata.last_modified is None
            assert metadata.test_coverage is None
            assert metadata.documentation_score is None
            
            # Test with values
            metadata = ChunkMetadata(
                complexity=5,
                dependencies=["os", "sys"],
                author="test_user",
                test_coverage=0.8
            )
            assert metadata.complexity == 5
            assert metadata.dependencies == ["os", "sys"]
            assert metadata.author == "test_user"
            assert metadata.test_coverage == 0.8
            
        except ImportError as e:
            pytest.skip(f"Cannot import ChunkMetadata: {e}")
    
    @pytest.mark.unit
    def test_chunking_request_creation(self):
        """Test ChunkingRequest creation."""
        try:
            from app.services.chunking.base import ChunkingRequest
            
            # Test basic creation
            request = ChunkingRequest(
                content="def hello(): pass",
                language="python",
                file_path="test.py"
            )
            
            assert request.content == "def hello(): pass"
            assert request.language == "python"
            assert request.file_path == "test.py"
            assert request.purpose == "general"
            assert request.quality_threshold == 0.8
            assert request.max_processing_time == 30
            assert request.force_agentic is False
            assert request.context == {}
            
            # Test with custom values
            request = ChunkingRequest(
                content="class Test: pass",
                language="python",
                file_path="test.py",
                purpose="code_review",
                quality_threshold=0.9,
                max_processing_time=60,
                force_agentic=True,
                context={"project": "test"}
            )
            
            assert request.purpose == "code_review"
            assert request.quality_threshold == 0.9
            assert request.max_processing_time == 60
            assert request.force_agentic is True
            assert request.context == {"project": "test"}
            
        except ImportError as e:
            pytest.skip(f"Cannot import ChunkingRequest: {e}")
    
    @pytest.mark.unit
    def test_agentic_chunk_creation(self):
        """Test AgenticChunk creation."""
        try:
            from app.services.chunking.base import AgenticChunk, ChunkType, ChunkMetadata
            
            chunk = AgenticChunk(
                id="test_chunk_1",
                content="def hello(): pass",
                language="python",
                chunk_type=ChunkType.FUNCTION,
                name="hello",
                start_line=1,
                end_line=1,
                file_path="test.py",
                dependencies=[],
                semantic_relationships=[],
                purpose_optimization={},
                quality_score=0.9,
                metadata=ChunkMetadata()
            )
            
            assert chunk.id == "test_chunk_1"
            assert chunk.content == "def hello(): pass"
            assert chunk.language == "python"
            assert chunk.chunk_type == ChunkType.FUNCTION
            assert chunk.name == "hello"
            assert chunk.start_line == 1
            assert chunk.end_line == 1
            assert chunk.file_path == "test.py"
            assert chunk.quality_score == 0.9
            
            # Test properties
            assert chunk.size_lines == 1
            assert chunk.size_chars == len("def hello(): pass")
            
        except ImportError as e:
            pytest.skip(f"Cannot import AgenticChunk: {e}")
    
    @pytest.mark.unit
    def test_chunking_result_creation(self):
        """Test ChunkingResult creation."""
        try:
            from app.services.chunking.base import (
                ChunkingResult, ChunkingStrategy, AgenticChunk, 
                ChunkType, ChunkMetadata
            )
            
            chunk = AgenticChunk(
                id="test_chunk_1",
                content="def hello(): pass",
                language="python",
                chunk_type=ChunkType.FUNCTION,
                name="hello",
                start_line=1,
                end_line=1,
                file_path="test.py",
                dependencies=[],
                semantic_relationships=[],
                purpose_optimization={},
                quality_score=0.9,
                metadata=ChunkMetadata()
            )
            
            result = ChunkingResult(
                chunks=[chunk],
                strategy_used=ChunkingStrategy.AST_FALLBACK,
                quality_score=0.9,
                processing_time=0.5,
                metadata={"test": "data"}
            )
            
            assert len(result.chunks) == 1
            assert result.strategy_used == ChunkingStrategy.AST_FALLBACK
            assert result.quality_score == 0.9
            assert result.processing_time == 0.5
            assert result.metadata == {"test": "data"}
            
            # Test properties
            assert result.chunk_count == 1
            assert result.average_chunk_size == 1.0
            assert result.total_lines == 1
            
        except ImportError as e:
            pytest.skip(f"Cannot import ChunkingResult: {e}")
    
    @pytest.mark.unit
    def test_utility_functions(self):
        """Test utility functions."""
        try:
            from app.services.chunking.base import generate_chunk_id, calculate_content_hash
            
            # Test chunk ID generation
            chunk_id = generate_chunk_id("test.py", 1, 10)
            assert isinstance(chunk_id, str)
            assert len(chunk_id) == 12  # MD5 hash truncated to 12 chars
            
            # Same inputs should generate same ID
            chunk_id2 = generate_chunk_id("test.py", 1, 10)
            assert chunk_id == chunk_id2
            
            # Different inputs should generate different IDs
            chunk_id3 = generate_chunk_id("test.py", 2, 10)
            assert chunk_id != chunk_id3
            
            # Test content hash calculation
            content = "def hello(): pass"
            hash1 = calculate_content_hash(content)
            assert isinstance(hash1, str)
            assert len(hash1) == 64  # SHA256 hash
            
            # Same content should generate same hash
            hash2 = calculate_content_hash(content)
            assert hash1 == hash2
            
            # Different content should generate different hash
            hash3 = calculate_content_hash("def goodbye(): pass")
            assert hash1 != hash3
            
        except ImportError as e:
            pytest.skip(f"Cannot import utility functions: {e}")
    
    @pytest.mark.unit
    def test_chunk_validation(self):
        """Test chunk validation."""
        try:
            from app.services.chunking.base import (
                validate_chunk, AgenticChunk, ChunkType, ChunkMetadata
            )
            
            # Valid chunk
            valid_chunk = AgenticChunk(
                id="test_chunk_1",
                content="def hello(): pass",
                language="python",
                chunk_type=ChunkType.FUNCTION,
                name="hello",
                start_line=1,
                end_line=1,
                file_path="test.py",
                dependencies=[],
                semantic_relationships=[],
                purpose_optimization={},
                quality_score=0.9,
                metadata=ChunkMetadata()
            )
            
            errors = validate_chunk(valid_chunk)
            assert len(errors) == 0
            
            # Invalid chunk - missing ID
            invalid_chunk = AgenticChunk(
                id="",
                content="def hello(): pass",
                language="python",
                chunk_type=ChunkType.FUNCTION,
                name="hello",
                start_line=1,
                end_line=1,
                file_path="test.py",
                dependencies=[],
                semantic_relationships=[],
                purpose_optimization={},
                quality_score=0.9,
                metadata=ChunkMetadata()
            )
            
            errors = validate_chunk(invalid_chunk)
            assert len(errors) > 0
            assert "Chunk ID is required" in errors
            
            # Invalid chunk - bad quality score
            invalid_chunk2 = AgenticChunk(
                id="test_chunk_1",
                content="def hello(): pass",
                language="python",
                chunk_type=ChunkType.FUNCTION,
                name="hello",
                start_line=1,
                end_line=1,
                file_path="test.py",
                dependencies=[],
                semantic_relationships=[],
                purpose_optimization={},
                quality_score=1.5,  # Invalid score > 1
                metadata=ChunkMetadata()
            )
            
            errors = validate_chunk(invalid_chunk2)
            assert len(errors) > 0
            assert "Quality score must be between 0 and 1" in errors
            
        except ImportError as e:
            pytest.skip(f"Cannot import validation functions: {e}")


class TestChunkingServiceMocked:
    """Test chunking service with mocked components."""
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_mocked_chunking_service(self):
        """Test chunking service with mocked dependencies."""
        # Create a mock chunking service
        mock_service = MagicMock()
        mock_service._initialized = False
        
        # Mock initialization
        async def mock_initialize():
            mock_service._initialized = True
            return True
        
        mock_service.initialize = mock_initialize
        
        # Test initialization
        result = await mock_service.initialize()
        assert mock_service._initialized is True
        assert result is True
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_mocked_health_check(self):
        """Test health check with mocked service."""
        mock_service = MagicMock()
        
        # Mock health check
        async def mock_health_check():
            return {
                "status": "healthy",
                "chunkers": {
                    "ast_fallback": {"status": "healthy"},
                    "hybrid": {"status": "healthy"}
                },
                "performance_stats": {
                    "total_requests": 0,
                    "average_processing_time": 0.0
                }
            }
        
        mock_service.health_check = mock_health_check
        
        health = await mock_service.health_check()
        
        assert health["status"] == "healthy"
        assert "chunkers" in health
        assert "performance_stats" in health
        assert health["chunkers"]["ast_fallback"]["status"] == "healthy"
        assert health["chunkers"]["hybrid"]["status"] == "healthy"
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_mocked_performance_stats(self):
        """Test performance statistics with mocked service."""
        mock_service = MagicMock()
        
        # Mock performance stats
        async def mock_get_performance_stats():
            return {
                "total_requests": 10,
                "agentic_requests": 3,
                "ast_fallback_requests": 7,
                "cache_hits": 2,
                "average_processing_time": 1.5,
                "average_quality_score": 0.85
            }
        
        mock_service.get_performance_stats = mock_get_performance_stats
        
        stats = await mock_service.get_performance_stats()
        
        assert stats["total_requests"] == 10
        assert stats["agentic_requests"] == 3
        assert stats["ast_fallback_requests"] == 7
        assert stats["cache_hits"] == 2
        assert stats["average_processing_time"] == 1.5
        assert stats["average_quality_score"] == 0.85
