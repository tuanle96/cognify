"""
Configuration management for Cognify application.

Handles environment variables, settings validation, and configuration loading.
"""

import os
from functools import lru_cache
from typing import List, Optional, Union

from pydantic import Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings with environment variable support.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

    # Application settings
    APP_NAME: str = "Cognify"
    VERSION: str = "0.1.0"
    ENVIRONMENT: str = Field(default="development", description="Environment: development, staging, production")
    DEBUG: bool = Field(default=False, description="Enable debug mode")

    # Server settings
    HOST: str = Field(default="0.0.0.0", description="Server host")
    PORT: int = Field(default=8000, description="Server port")
    WORKERS: int = Field(default=1, description="Number of worker processes")

    # Security settings
    SECRET_KEY: str = Field(default="dev-secret-key-change-in-production", description="Secret key for JWT tokens")
    ALGORITHM: str = Field(default="HS256", description="JWT algorithm")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, description="Access token expiration in minutes")

    # CORS settings
    ALLOWED_ORIGINS: List[str] = Field(default=["*"], description="Allowed CORS origins")
    ALLOWED_HOSTS: List[str] = Field(default=["*"], description="Allowed hosts")

    # Database settings
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://cognify_test:test_password@localhost:5433/cognify_test",
        description="PostgreSQL database URL"
    )
    DATABASE_POOL_SIZE: int = Field(default=10, description="Database connection pool size")
    DATABASE_MAX_OVERFLOW: int = Field(default=20, description="Database max overflow connections")
    DATABASE_ECHO: bool = Field(default=False, description="Enable SQLAlchemy query logging")
    DATABASE_POOL_RECYCLE: int = Field(default=3600, description="Database pool recycle time in seconds")
    DATABASE_POOL_PRE_PING: bool = Field(default=True, description="Enable database pool pre-ping")

    # Authentication settings
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=7, description="Refresh token expiration in days")
    PASSWORD_MIN_LENGTH: int = Field(default=8, description="Minimum password length")
    MAX_LOGIN_ATTEMPTS: int = Field(default=5, description="Maximum failed login attempts")
    ACCOUNT_LOCKOUT_MINUTES: int = Field(default=30, description="Account lockout duration in minutes")

    # Vector Database settings
    VECTOR_DB_TYPE: str = Field(default="qdrant", description="Vector database type: qdrant, milvus")
    QDRANT_URL: str = Field(default="http://localhost:6333", description="Qdrant server URL")
    QDRANT_API_KEY: Optional[str] = Field(default=None, description="Qdrant API key")
    QDRANT_COLLECTION_NAME: str = Field(default="cognify_chunks", description="Default Qdrant collection name")
    VECTOR_DIMENSION: int = Field(default=1536, description="Vector embedding dimension")

    # Redis settings
    REDIS_URL: str = Field(default="redis://localhost:6379", description="Redis server URL")
    REDIS_DB: int = Field(default=0, description="Redis database number")
    REDIS_PASSWORD: Optional[str] = Field(default=None, description="Redis password")

    # AI/LLM settings
    OPENAI_API_KEY: Optional[str] = Field(default=None, description="OpenAI API key")
    OPENAI_BASE_URL: Optional[str] = Field(default=None, description="OpenAI base URL (for proxies)")
    ANTHROPIC_API_KEY: Optional[str] = Field(default=None, description="Anthropic API key")
    VOYAGE_API_KEY: Optional[str] = Field(default=None, description="Voyage AI API key")
    COHERE_API_KEY: Optional[str] = Field(default=None, description="Cohere API key")

    # LiteLLM settings
    LITELLM_API_KEY: Optional[str] = Field(default=None, description="LiteLLM API key")
    LITELLM_BASE_URL: str = Field(default="https://ai.earnbase.io/v1", description="LiteLLM base URL")
    LITELLM_CHAT_MODEL: str = Field(default="grok-3-beta", description="LiteLLM chat model")
    LITELLM_EMBEDDING_MODEL: str = Field(default="text-embedding-004", description="LiteLLM embedding model")
    LITELLM_MAX_TOKENS: int = Field(default=4000, description="LiteLLM max tokens")
    LITELLM_TEMPERATURE: float = Field(default=0.5, description="LiteLLM temperature")

    # Default LLM settings
    DEFAULT_LLM_PROVIDER: str = Field(default="openai", description="Default LLM provider")
    DEFAULT_LLM_MODEL: str = Field(default="gpt-4o-mini", description="Default LLM model")
    DEFAULT_EMBEDDING_PROVIDER: str = Field(default="openai", description="Default embedding provider")
    DEFAULT_EMBEDDING_MODEL: str = Field(default="text-embedding-004", description="Default embedding model")

    # Chunking settings
    CHUNKING_STRATEGY: str = Field(default="hybrid", description="Chunking strategy: agentic, ast_fallback, hybrid")
    CHUNKING_QUALITY_THRESHOLD: float = Field(default=0.8, description="Minimum quality threshold for chunks")
    CHUNKING_MAX_PROCESSING_TIME: int = Field(default=30, description="Max processing time for chunking in seconds")
    CHUNKING_CACHE_TTL: int = Field(default=3600, description="Chunking cache TTL in seconds")

    # Agent settings
    AGENT_MAX_ITERATIONS: int = Field(default=5, description="Maximum agent iterations")
    AGENT_TIMEOUT: int = Field(default=60, description="Agent timeout in seconds")
    AGENT_MEMORY_ENABLED: bool = Field(default=True, description="Enable agent memory")

    # Monitoring and observability
    ENABLE_METRICS: bool = Field(default=True, description="Enable Prometheus metrics")
    SENTRY_DSN: Optional[str] = Field(default=None, description="Sentry DSN for error tracking")
    SENTRY_TRACES_SAMPLE_RATE: float = Field(default=0.1, description="Sentry traces sample rate")

    # Logging settings
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    LOG_FORMAT: str = Field(default="json", description="Log format: json, text")

    # File processing settings
    MAX_FILE_SIZE: int = Field(default=10 * 1024 * 1024, description="Maximum file size in bytes (10MB)")
    SUPPORTED_LANGUAGES: List[str] = Field(
        default=["python", "javascript", "typescript", "go", "java", "rust", "cpp"],
        description="Supported programming languages"
    )

    # Rate limiting
    RATE_LIMIT_REQUESTS: int = Field(default=100, description="Rate limit requests per minute")
    RATE_LIMIT_BURST: int = Field(default=200, description="Rate limit burst capacity")

    @validator("ENVIRONMENT")
    def validate_environment(cls, v):
        """Validate environment setting."""
        allowed_environments = ["development", "staging", "production", "testing"]
        if v not in allowed_environments:
            raise ValueError(f"Environment must be one of: {allowed_environments}")
        return v

    @validator("VECTOR_DB_TYPE")
    def validate_vector_db_type(cls, v):
        """Validate vector database type."""
        allowed_types = ["qdrant", "milvus"]
        if v not in allowed_types:
            raise ValueError(f"Vector DB type must be one of: {allowed_types}")
        return v

    @validator("CHUNKING_STRATEGY")
    def validate_chunking_strategy(cls, v):
        """Validate chunking strategy."""
        allowed_strategies = ["agentic", "ast_fallback", "hybrid"]
        if v not in allowed_strategies:
            raise ValueError(f"Chunking strategy must be one of: {allowed_strategies}")
        return v

    @validator("CHUNKING_QUALITY_THRESHOLD")
    def validate_quality_threshold(cls, v):
        """Validate quality threshold is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("Quality threshold must be between 0 and 1")
        return v

    @validator("LOG_LEVEL")
    def validate_log_level(cls, v):
        """Validate log level."""
        allowed_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in allowed_levels:
            raise ValueError(f"Log level must be one of: {allowed_levels}")
        return v.upper()

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.ENVIRONMENT == "development"

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.ENVIRONMENT == "production"

    @property
    def database_url_async(self) -> str:
        """Get async database URL."""
        if self.DATABASE_URL.startswith("postgresql://"):
            return self.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
        return self.DATABASE_URL

    def get_llm_config(self) -> dict:
        """Get LLM configuration with auto-provider selection."""
        # Priority order: LiteLLM -> OpenAI -> Mock
        if self.LITELLM_API_KEY:
            return {
                "provider": "litellm",
                "model": self.LITELLM_CHAT_MODEL,
                "api_key": self.LITELLM_API_KEY,
                "base_url": self.LITELLM_BASE_URL,
                "temperature": self.LITELLM_TEMPERATURE,
                "max_tokens": self.LITELLM_MAX_TOKENS,
            }
        elif self.OPENAI_API_KEY:
            return {
                "provider": "openai",
                "model": self.DEFAULT_LLM_MODEL,
                "api_key": self.OPENAI_API_KEY,
                "base_url": self.OPENAI_BASE_URL,
                "temperature": 0.1,
                "max_tokens": 4000,
            }
        else:
            # For development/testing, return mock config if no API key
            return {
                "provider": "mock",
                "model": "mock-model",
                "api_key": None,
                "temperature": 0.1,
                "max_tokens": 4000,
            }

    def get_embedding_config(self) -> dict:
        """Get embedding configuration."""
        return {
            "provider": self.DEFAULT_EMBEDDING_PROVIDER,
            "model": self.DEFAULT_EMBEDDING_MODEL,
            "api_key": getattr(self, f"{self.DEFAULT_EMBEDDING_PROVIDER.upper()}_API_KEY"),
            "dimension": self.VECTOR_DIMENSION,
        }


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached application settings.

    Returns:
        Settings: Application settings instance
    """
    return Settings()


# Export settings instance for convenience
settings = get_settings()
