"""Unit tests for configuration management."""

import pytest
import os
from unittest.mock import patch

from app.core.config import Settings, get_settings


class TestSettings:
    """Test cases for Settings class."""

    @pytest.mark.unit
    def test_default_settings(self):
        """Test default settings values."""
        with patch.dict(os.environ, {}, clear=True):
            # Set required environment variables
            with patch.dict(os.environ, {
                'SECRET_KEY': 'test-secret-key',
                'DATABASE_URL': 'postgresql://test:test@localhost/test',
                'ENVIRONMENT': 'testing'
            }):
                settings = Settings()

                assert settings.APP_NAME == "Cognify"
                assert settings.VERSION == "0.1.0"
                assert settings.ENVIRONMENT == "testing"
                assert settings.HOST == "0.0.0.0"
                assert settings.PORT == 8000

    @pytest.mark.unit
    def test_environment_validation(self):
        """Test environment validation."""
        with patch.dict(os.environ, {
            'SECRET_KEY': 'test-secret-key',
            'DATABASE_URL': 'postgresql://test:test@localhost/test',
            'ENVIRONMENT': 'invalid_env'
        }):
            with pytest.raises(ValueError, match="Environment must be one of"):
                Settings()

    @pytest.mark.unit
    def test_testing_environment_allowed(self):
        """Test that testing environment is allowed."""
        with patch.dict(os.environ, {
            'SECRET_KEY': 'test-secret-key',
            'DATABASE_URL': 'postgresql://test:test@localhost/test',
            'ENVIRONMENT': 'testing'
        }):
            settings = Settings()
            assert settings.ENVIRONMENT == "testing"

    @pytest.mark.unit
    def test_llm_config_with_api_key(self):
        """Test LLM config when API key is provided."""
        with patch.dict(os.environ, {
            'SECRET_KEY': 'test-secret-key',
            'DATABASE_URL': 'postgresql://test:test@localhost/test',
            'ENVIRONMENT': 'testing',
            'OPENAI_API_KEY': 'test-openai-key'
        }):
            settings = Settings()
            llm_config = settings.get_llm_config()

            assert llm_config['provider'] == 'openai'
            assert llm_config['api_key'] == 'test-openai-key'
            assert llm_config['model'] == 'gpt-4'

    @pytest.mark.unit
    def test_llm_config_with_litellm_api_key(self):
        """Test LLM config when LiteLLM API key is provided (priority provider)."""
        with patch.dict(os.environ, {
            'SECRET_KEY': 'test-secret-key',
            'DATABASE_URL': 'postgresql://test:test@localhost/test',
            'ENVIRONMENT': 'testing',
            'LITELLM_API_KEY': 'test-litellm-key'
        }):
            settings = Settings()
            llm_config = settings.get_llm_config()

            assert llm_config['provider'] == 'litellm'
            assert llm_config['api_key'] == 'test-litellm-key'
            assert llm_config['model'] == 'grok-3-beta'
            assert llm_config['base_url'] == 'https://ai.earnbase.io/v1'

    @pytest.mark.unit
    def test_llm_config_without_api_key(self):
        """Test LLM config when no API key is provided (should use mock)."""
        with patch.dict(os.environ, {
            'SECRET_KEY': 'test-secret-key',
            'DATABASE_URL': 'postgresql://test:test@localhost/test',
            'ENVIRONMENT': 'testing'
        }, clear=True):
            settings = Settings()
            llm_config = settings.get_llm_config()

            assert llm_config['provider'] == 'mock'
            assert llm_config['api_key'] is None
            assert llm_config['model'] == 'mock-model'

    @pytest.mark.unit
    def test_is_development_property(self):
        """Test is_development property."""
        with patch.dict(os.environ, {
            'SECRET_KEY': 'test-secret-key',
            'DATABASE_URL': 'postgresql://test:test@localhost/test',
            'ENVIRONMENT': 'development'
        }):
            settings = Settings()
            assert settings.is_development is True
            assert settings.is_production is False

    @pytest.mark.unit
    def test_is_production_property(self):
        """Test is_production property."""
        with patch.dict(os.environ, {
            'SECRET_KEY': 'test-secret-key',
            'DATABASE_URL': 'postgresql://test:test@localhost/test',
            'ENVIRONMENT': 'production'
        }):
            settings = Settings()
            assert settings.is_production is True
            assert settings.is_development is False

    @pytest.mark.unit
    def test_database_url_async(self):
        """Test async database URL conversion."""
        with patch.dict(os.environ, {
            'SECRET_KEY': 'test-secret-key',
            'DATABASE_URL': 'postgresql://test:test@localhost/test',
            'ENVIRONMENT': 'testing'
        }):
            settings = Settings()
            async_url = settings.database_url_async

            assert async_url.startswith('postgresql+asyncpg://')

    @pytest.mark.unit
    def test_chunking_quality_threshold_validation(self):
        """Test chunking quality threshold validation."""
        with patch.dict(os.environ, {
            'SECRET_KEY': 'test-secret-key',
            'DATABASE_URL': 'postgresql://test:test@localhost/test',
            'ENVIRONMENT': 'testing',
            'CHUNKING_QUALITY_THRESHOLD': '1.5'  # Invalid value > 1
        }):
            with pytest.raises(ValueError, match="Quality threshold must be between 0 and 1"):
                Settings()

    @pytest.mark.unit
    def test_log_level_validation(self):
        """Test log level validation."""
        with patch.dict(os.environ, {
            'SECRET_KEY': 'test-secret-key',
            'DATABASE_URL': 'postgresql://test:test@localhost/test',
            'ENVIRONMENT': 'testing',
            'LOG_LEVEL': 'INVALID_LEVEL'
        }):
            with pytest.raises(ValueError, match="Log level must be one of"):
                Settings()

    @pytest.mark.unit
    def test_vector_db_type_validation(self):
        """Test vector database type validation."""
        with patch.dict(os.environ, {
            'SECRET_KEY': 'test-secret-key',
            'DATABASE_URL': 'postgresql://test:test@localhost/test',
            'ENVIRONMENT': 'testing',
            'VECTOR_DB_TYPE': 'invalid_db'
        }):
            with pytest.raises(ValueError, match="Vector DB type must be one of"):
                Settings()

    @pytest.mark.unit
    def test_chunking_strategy_validation(self):
        """Test chunking strategy validation."""
        with patch.dict(os.environ, {
            'SECRET_KEY': 'test-secret-key',
            'DATABASE_URL': 'postgresql://test:test@localhost/test',
            'ENVIRONMENT': 'testing',
            'CHUNKING_STRATEGY': 'invalid_strategy'
        }):
            with pytest.raises(ValueError, match="Chunking strategy must be one of"):
                Settings()


class TestGetSettings:
    """Test cases for get_settings function."""

    @pytest.mark.unit
    def test_get_settings_caching(self):
        """Test that get_settings returns cached instance."""
        with patch.dict(os.environ, {
            'SECRET_KEY': 'test-secret-key',
            'DATABASE_URL': 'postgresql://test:test@localhost/test',
            'ENVIRONMENT': 'testing'
        }):
            settings1 = get_settings()
            settings2 = get_settings()

            # Should return the same instance due to lru_cache
            assert settings1 is settings2

    @pytest.mark.unit
    def test_settings_module_import(self):
        """Test that settings can be imported from module."""
        # This test ensures the module-level settings import works
        from app.core.config import settings

        assert settings is not None
        assert hasattr(settings, 'APP_NAME')
        assert hasattr(settings, 'ENVIRONMENT')
