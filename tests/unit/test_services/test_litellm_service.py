"""Unit tests for LiteLLM service."""

import pytest
import os
from unittest.mock import AsyncMock, patch, MagicMock
import aiohttp

from app.services.llm.litellm_service import LiteLLMService
from app.services.llm.base import LLMConfig, LLMProvider, LLMMessage, LLMResponse


class TestLiteLLMService:
    """Test cases for LiteLLMService."""

    @pytest.fixture
    def litellm_config(self):
        """Create LiteLLM configuration for testing."""
        return LLMConfig(
            provider=LLMProvider.LITELLM,
            model="grok-3-beta",
            api_key="test-litellm-key",
            temperature=0.5,
            max_tokens=4000,
            timeout=30,
            retry_attempts=3
        )

    @pytest.fixture
    def litellm_service(self, litellm_config):
        """Create LiteLLM service instance for testing."""
        with patch.dict(os.environ, {
            'LITELLM_API_KEY': 'test-litellm-key',
            'LITELLM_BASE_URL': 'https://test.api.com/v1',
            'LITELLM_CHAT_MODEL': 'grok-3-beta',
            'LITELLM_EMBEDDING_MODEL': 'text-embedding-004'
        }):
            return LiteLLMService(litellm_config)

    @pytest.mark.unit
    def test_service_initialization(self, litellm_service):
        """Test LiteLLM service initialization."""
        assert litellm_service is not None
        assert litellm_service.provider == LLMProvider.LITELLM
        assert litellm_service.chat_model == "grok-3-beta"
        assert litellm_service.api_key == "test-litellm-key"
        assert not litellm_service.is_initialized

    @pytest.mark.unit
    def test_service_initialization_no_api_key(self):
        """Test LiteLLM service initialization without API key."""
        config = LLMConfig(
            provider=LLMProvider.LITELLM,
            model="grok-3-beta",
            api_key=None
        )

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="LiteLLM API key not found"):
                LiteLLMService(config)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_health_check_success(self, litellm_service):
        """Test successful health check."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "choices": [{"message": {"content": "test"}}],
            "usage": {"total_tokens": 1}
        })

        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response

            health = await litellm_service.health_check()

            assert health["status"] == "healthy"
            assert health["provider"] == "litellm"
            assert health["model"] == "grok-3-beta"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_health_check_failure(self, litellm_service):
        """Test failed health check."""
        mock_response = MagicMock()
        mock_response.status = 401

        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response

            health = await litellm_service.health_check()

            assert health["status"] == "unhealthy"
            assert health["provider"] == "litellm"
            assert "HTTP 401" in health["error"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_generate_success(self, litellm_service):
        """Test successful text generation."""
        messages = [LLMMessage(role="user", content="Hello")]

        mock_response_data = {
            "choices": [{
                "message": {"content": "Hello! How can I help you?"},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 10,
                "total_tokens": 15
            },
            "model": "grok-3-beta"
        }

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_response_data)

        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response

            response = await litellm_service.generate(messages)

            assert isinstance(response, LLMResponse)
            assert response.content == "Hello! How can I help you?"
            assert response.model == "grok-3-beta"
            assert response.provider == "litellm"
            assert response.usage["total_tokens"] == 15

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_generate_with_parameters(self, litellm_service):
        """Test text generation with custom parameters."""
        messages = [LLMMessage(role="user", content="Hello")]

        mock_response_data = {
            "choices": [{
                "message": {"content": "Response"},
                "finish_reason": "stop"
            }],
            "usage": {"total_tokens": 10},
            "model": "grok-3-beta"
        }

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_response_data)

        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response

            response = await litellm_service.generate(
                messages,
                temperature=0.8,
                max_tokens=2000
            )

            # Verify the request was made with correct parameters
            call_args = mock_post.call_args
            request_data = call_args[1]['json']

            assert request_data['temperature'] == 0.8
            assert request_data['max_tokens'] == 2000
            assert request_data['model'] == "grok-3-beta"
            assert isinstance(response, LLMResponse)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_generate_api_error(self, litellm_service):
        """Test text generation with API error."""
        messages = [LLMMessage(role="user", content="Hello")]

        mock_response = MagicMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal Server Error")

        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response

            with pytest.raises(Exception, match="LiteLLM API error 500"):
                await litellm_service.generate(messages)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_generate_rate_limit_retry(self, litellm_service):
        """Test text generation with rate limit and retry."""
        messages = [LLMMessage(role="user", content="Hello")]

        # First call returns 429, second call succeeds
        mock_response_429 = MagicMock()
        mock_response_429.status = 429
        mock_response_429.text = AsyncMock(return_value="Rate limit exceeded")

        mock_response_200 = MagicMock()
        mock_response_200.status = 200
        mock_response_200.json = AsyncMock(return_value={
            "choices": [{"message": {"content": "Success"}}],
            "usage": {"total_tokens": 5},
            "model": "grok-3-beta"
        })

        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.side_effect = [
                mock_response_429,
                mock_response_200
            ]

            with patch('asyncio.sleep'):  # Mock sleep to speed up test
                response = await litellm_service.generate(messages)

                assert response.content == "Success"
                assert mock_post.call_count == 2  # Verify retry happened

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_stream_generate(self, litellm_service):
        """Test streaming text generation."""
        messages = [LLMMessage(role="user", content="Hello")]

        # Mock streaming response
        stream_data = [
            'data: {"choices":[{"delta":{"content":"Hello"}}]}\n',
            'data: {"choices":[{"delta":{"content":" there"}}]}\n',
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n',
            'data: [DONE]\n'
        ]

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.content.__aiter__.return_value = [
            line.encode('utf-8') for line in stream_data
        ]

        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response

            chunks = []
            async for chunk in litellm_service.stream_generate(messages):
                chunks.append(chunk)

            assert len(chunks) >= 2
            assert chunks[0]["content"] == "Hello"
            assert chunks[1]["content"] == " there"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_cleanup(self, litellm_service):
        """Test service cleanup."""
        await litellm_service.cleanup()
        # Should complete without error
        assert True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_initialize_success(self, litellm_service):
        """Test successful service initialization."""
        # Mock successful health check
        with patch.object(litellm_service, 'health_check') as mock_health:
            mock_health.return_value = {"status": "healthy"}

            await litellm_service.initialize()

            assert litellm_service.is_initialized
            mock_health.assert_called_once()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_initialize_failure(self, litellm_service):
        """Test failed service initialization."""
        # Mock failed health check
        with patch.object(litellm_service, 'health_check') as mock_health:
            mock_health.return_value = {"status": "unhealthy", "error": "Connection failed"}

            with pytest.raises(Exception, match="LiteLLM service health check failed"):
                await litellm_service.initialize()

            assert not litellm_service.is_initialized


class TestLiteLLMEmbeddingClient:
    """Test cases for LiteLLM embedding client."""

    @pytest.fixture
    def embedding_client(self):
        """Create LiteLLM embedding client for testing."""
        # Clear cache first to ensure fresh settings
        from app.core.config import get_settings
        get_settings.cache_clear()

        with patch.dict(os.environ, {
            'SECRET_KEY': 'test-secret-key',
            'DATABASE_URL': 'postgresql://test:test@localhost/test',
            'ENVIRONMENT': 'testing',
            'LITELLM_API_KEY': 'test-litellm-key',
            'LITELLM_BASE_URL': 'https://test.api.com/v1',
            'LITELLM_EMBEDDING_MODEL': 'text-embedding-004'
        }):
            from app.services.llm.litellm_service import LiteLLMEmbeddingClient
            return LiteLLMEmbeddingClient()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_embed_texts_success(self, embedding_client):
        """Test successful text embedding."""
        texts = ["Hello world", "How are you?"]

        mock_response_data = {
            "data": [
                {"embedding": [0.1, 0.2, 0.3]},
                {"embedding": [0.4, 0.5, 0.6]}
            ]
        }

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_response_data)

        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response

            embeddings = await embedding_client.embed_texts(texts)

            assert len(embeddings) == 2
            assert embeddings[0] == [0.1, 0.2, 0.3]
            assert embeddings[1] == [0.4, 0.5, 0.6]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_embed_single_text(self, embedding_client):
        """Test single text embedding."""
        text = "Hello world"

        mock_response_data = {
            "data": [{"embedding": [0.1, 0.2, 0.3]}]
        }

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_response_data)

        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response

            embedding = await embedding_client.embed_single(text)

            assert embedding == [0.1, 0.2, 0.3]
