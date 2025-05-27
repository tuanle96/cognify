"""Unit tests for ChunkingService."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from app.services.chunking.service import ChunkingService
from app.services.chunking.base import ChunkingRequest, ChunkingResult, AgenticChunk
from tests.utils.assertions import (
    assert_chunking_response_valid,
    assert_chunk_valid,
    assert_performance_acceptable
)
from tests.utils.test_helpers import create_chunking_request, generate_test_code


class TestChunkingService:
    """Test cases for ChunkingService."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_service_initialization(self):
        """Test that ChunkingService initializes correctly."""
        service = ChunkingService()
        await service.initialize()

        assert service is not None
        assert hasattr(service, 'hybrid_chunker')
        assert hasattr(service, 'ast_chunker')

        # Test health check
        health = await service.health_check()
        assert health['status'] == 'healthy'
        assert 'chunking_service' in health

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_chunk_content_basic(self, chunking_service, sample_python_code):
        """Test basic content chunking functionality."""
        request = ChunkingRequest(
            content=sample_python_code,
            language="python",
            file_path="test.py",
            purpose="general"
        )

        response = await chunking_service.chunk_content(request)

        # Validate response structure
        assert_chunking_response_valid(response)

        # Validate content
        assert len(response.chunks) > 0
        assert response.metadata['total_chunks'] > 0
        assert response.metadata['strategy_used'] in ['agentic', 'ast', 'hybrid']

        # Validate individual chunks
        for chunk in response.chunks:
            assert_chunk_valid(chunk)
            assert len(chunk.content.strip()) > 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_chunk_content_different_purposes(self, chunking_service, sample_python_code):
        """Test chunking with different purposes."""
        purposes = ["general", "code_review", "bug_detection", "documentation"]

        for purpose in purposes:
            request = ChunkingRequest(
                content=sample_python_code,
                language="python",
                file_path="test.py",
                purpose=purpose
            )

            response = await chunking_service.chunk_content(request)

            assert_chunking_response_valid(response)
            assert response.metadata['purpose'] == purpose
            assert len(response.chunks) > 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_chunk_content_force_agentic(self, chunking_service, sample_python_code):
        """Test forcing agentic chunking strategy."""
        request = ChunkingRequest(
            content=sample_python_code,
            language="python",
            file_path="test.py",
            purpose="general",
            force_agentic=True
        )

        response = await chunking_service.chunk_content(request)

        assert_chunking_response_valid(response)
        assert response.metadata['strategy_used'] == 'agentic'
        assert 'agent_decisions' in response.metadata

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_chunk_content_force_ast(self, chunking_service, sample_python_code):
        """Test forcing AST chunking strategy."""
        request = ChunkingRequest(
            content=sample_python_code,
            language="python",
            file_path="test.py",
            purpose="general",
            force_agentic=False
        )

        response = await chunking_service.chunk_content(request)

        assert_chunking_response_valid(response)
        assert response.metadata['strategy_used'] == 'ast'

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_chunk_content_empty_content(self, chunking_service):
        """Test chunking with empty content."""
        request = ChunkingRequest(
            content="",
            language="python",
            file_path="empty.py",
            purpose="general"
        )

        response = await chunking_service.chunk_content(request)

        assert_chunking_response_valid(response)
        assert len(response.chunks) == 0
        assert response.metadata['total_chunks'] == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_chunk_content_unsupported_language(self, chunking_service):
        """Test chunking with unsupported language."""
        request = ChunkingRequest(
            content="print('hello')",
            language="unsupported_lang",
            file_path="test.unknown",
            purpose="general"
        )

        with pytest.raises(ValueError, match="Unsupported language"):
            await chunking_service.chunk_content(request)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_chunk_content_invalid_purpose(self, chunking_service, sample_python_code):
        """Test chunking with invalid purpose."""
        request = ChunkingRequest(
            content=sample_python_code,
            language="python",
            file_path="test.py",
            purpose="invalid_purpose"
        )

        with pytest.raises(ValueError, match="Invalid purpose"):
            await chunking_service.chunk_content(request)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_chunk_content_performance(self, chunking_service):
        """Test chunking performance with different content sizes."""
        # Test with small content
        small_content = generate_test_code("python", "simple")
        request = create_chunking_request(content=small_content)

        response = await chunking_service.chunk_content(request)
        assert_performance_acceptable(response.metadata['processing_time'], max_time=5.0)

        # Test with medium content
        medium_content = generate_test_code("python", "medium")
        request = create_chunking_request(content=medium_content)

        response = await chunking_service.chunk_content(request)
        assert_performance_acceptable(response.metadata['processing_time'], max_time=10.0)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_health_check(self, chunking_service):
        """Test service health check."""
        health = await chunking_service.health_check()

        assert isinstance(health, dict)
        assert health['status'] == 'healthy'
        assert 'chunking_service' in health
        assert 'hybrid_chunker' in health['chunking_service']
        assert 'ast_chunker' in health['chunking_service']

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_supported_languages(self, chunking_service):
        """Test getting supported languages."""
        languages = await chunking_service.get_supported_languages()

        assert isinstance(languages, list)
        assert 'python' in languages
        assert len(languages) > 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_supported_purposes(self, chunking_service):
        """Test getting supported purposes."""
        purposes = await chunking_service.get_supported_purposes()

        assert isinstance(purposes, list)
        assert 'general' in purposes
        assert 'code_review' in purposes
        assert 'bug_detection' in purposes
        assert 'documentation' in purposes

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_chunk_content_with_metadata(self, chunking_service, sample_python_code):
        """Test chunking with additional metadata."""
        request = ChunkingRequest(
            content=sample_python_code,
            language="python",
            file_path="test.py",
            purpose="general",
            metadata={
                "author": "test_user",
                "project": "test_project",
                "version": "1.0.0"
            }
        )

        response = await chunking_service.chunk_content(request)

        assert_chunking_response_valid(response)
        assert 'author' in response.metadata
        assert 'project' in response.metadata
        assert 'version' in response.metadata

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_chunk_content_concurrent_requests(self, chunking_service, sample_python_code):
        """Test handling concurrent chunking requests."""
        import asyncio

        # Create multiple requests
        requests = [
            ChunkingRequest(
                content=sample_python_code,
                language="python",
                file_path=f"test_{i}.py",
                purpose="general"
            )
            for i in range(5)
        ]

        # Process concurrently
        tasks = [
            chunking_service.chunk_content(request)
            for request in requests
        ]

        responses = await asyncio.gather(*tasks)

        # Validate all responses
        assert len(responses) == 5
        for response in responses:
            assert_chunking_response_valid(response)
            assert len(response.chunks) > 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_error_handling_llm_failure(self, chunking_service, sample_python_code):
        """Test error handling when LLM service fails."""
        with patch('app.services.chunking.service.ChunkingService._get_llm_service') as mock_llm:
            # Mock LLM service to raise an exception
            mock_llm.return_value.generate.side_effect = Exception("LLM service unavailable")

            request = ChunkingRequest(
                content=sample_python_code,
                language="python",
                file_path="test.py",
                purpose="general",
                force_agentic=True  # Force agentic to trigger LLM usage
            )

            # Should fallback to AST chunking
            response = await chunking_service.chunk_content(request)

            assert_chunking_response_valid(response)
            assert response.metadata['strategy_used'] == 'ast'  # Fallback strategy

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_cleanup(self, chunking_service):
        """Test service cleanup."""
        # Service should be healthy before cleanup
        health = await chunking_service.health_check()
        assert health['status'] == 'healthy'

        # Cleanup should not raise exceptions
        if hasattr(chunking_service, 'cleanup'):
            await chunking_service.cleanup()

        # After cleanup, service might not be healthy (depending on implementation)
        # This test mainly ensures cleanup doesn't crash
