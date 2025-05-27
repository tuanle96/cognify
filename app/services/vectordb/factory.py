"""
Factory for creating vector database clients.
"""

from typing import Dict, Any, Optional, Type
import logging
import os

from .base import VectorDBClient, VectorDBProvider, VectorDBError
from .qdrant_client import QdrantClient
from .milvus_client import MilvusClient

logger = logging.getLogger(__name__)

class VectorDBFactory:
    """Factory for creating and managing vector database clients."""
    
    # Registry of available clients
    _clients: Dict[VectorDBProvider, Type[VectorDBClient]] = {
        VectorDBProvider.QDRANT: QdrantClient,
        VectorDBProvider.MILVUS: MilvusClient,
    }
    
    # Default configurations for each provider
    _default_configs = {
        VectorDBProvider.QDRANT: {
            "host": "localhost",
            "port": 6333,
            "timeout": 30.0
        },
        VectorDBProvider.MILVUS: {
            "host": "localhost", 
            "port": 19530,
            "timeout": 30.0
        }
    }
    
    @classmethod
    def create_client(
        self,
        provider: VectorDBProvider,
        host: Optional[str] = None,
        port: Optional[int] = None,
        **kwargs
    ) -> VectorDBClient:
        """
        Create a vector database client for the specified provider.
        
        Args:
            provider: The vector database provider to use
            host: Database host (if None, will use default)
            port: Database port (if None, will use default)
            **kwargs: Additional configuration options
        
        Returns:
            Configured vector database client
        
        Raises:
            VectorDBError: If provider is not supported or configuration is invalid
        """
        if provider not in self._clients:
            raise VectorDBError(f"Unsupported vector database provider: {provider}")
        
        # Get default configuration
        default_config = self._default_configs.get(provider, {}).copy()
        
        # Override with provided values
        if host:
            default_config["host"] = host
        if port:
            default_config["port"] = port
        default_config.update(kwargs)
        
        # Create client
        client_class = self._clients[provider]
        try:
            return client_class(**default_config)
        except Exception as e:
            raise VectorDBError(f"Failed to create {provider.value} client: {e}") from e
    
    @classmethod
    def create_qdrant_client(
        self,
        host: str = "localhost",
        port: int = 6333,
        api_key: Optional[str] = None,
        **kwargs
    ) -> QdrantClient:
        """Create a Qdrant vector database client."""
        return self.create_client(
            VectorDBProvider.QDRANT, 
            host=host, 
            port=port, 
            api_key=api_key,
            **kwargs
        )
    
    @classmethod
    def create_milvus_client(
        self,
        host: str = "localhost",
        port: int = 19530,
        user: Optional[str] = None,
        password: Optional[str] = None,
        **kwargs
    ) -> MilvusClient:
        """Create a Milvus vector database client."""
        return self.create_client(
            VectorDBProvider.MILVUS,
            host=host,
            port=port,
            user=user,
            password=password,
            **kwargs
        )
    
    @classmethod
    def get_available_providers(self) -> list[VectorDBProvider]:
        """Get list of available vector database providers."""
        return list(self._clients.keys())
    
    @classmethod
    def get_default_config(self, provider: VectorDBProvider) -> Dict[str, Any]:
        """Get default configuration for a provider."""
        return self._default_configs.get(provider, {}).copy()
    
    @classmethod
    def register_client(
        self,
        provider: VectorDBProvider,
        client_class: Type[VectorDBClient],
        default_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register a new vector database client.
        
        Args:
            provider: The provider enum
            client_class: The client class
            default_config: Default configuration for the client
        """
        self._clients[provider] = client_class
        if default_config:
            self._default_configs[provider] = default_config
        
        logger.info(f"Registered vector database client for {provider.value}")
    
    @classmethod
    def create_best_available_client(
        self,
        preferred_providers: Optional[list[VectorDBProvider]] = None,
        **kwargs
    ) -> VectorDBClient:
        """
        Create the best available vector database client.
        
        Args:
            preferred_providers: List of providers in order of preference
            **kwargs: Additional configuration options
        
        Returns:
            The first available vector database client
        
        Raises:
            VectorDBError: If no providers are available
        """
        if preferred_providers is None:
            # Default preference order: Qdrant -> Milvus
            preferred_providers = [
                VectorDBProvider.QDRANT,
                VectorDBProvider.MILVUS
            ]
        
        for provider in preferred_providers:
            try:
                client = self.create_client(provider, **kwargs)
                logger.info(f"Created {provider.value} vector database client")
                return client
            except Exception as e:
                logger.warning(f"Failed to create {provider.value} client: {e}")
                continue
        
        raise VectorDBError("No vector database providers available")
    
    @classmethod
    def get_connection_info_from_env(self, provider: VectorDBProvider) -> Dict[str, Any]:
        """Get connection information from environment variables."""
        env_configs = {
            VectorDBProvider.QDRANT: {
                "host": os.getenv("QDRANT_HOST", "localhost"),
                "port": int(os.getenv("QDRANT_PORT", "6333")),
                "api_key": os.getenv("QDRANT_API_KEY"),
                "https": os.getenv("QDRANT_HTTPS", "false").lower() == "true"
            },
            VectorDBProvider.MILVUS: {
                "host": os.getenv("MILVUS_HOST", "localhost"),
                "port": int(os.getenv("MILVUS_PORT", "19530")),
                "user": os.getenv("MILVUS_USER"),
                "password": os.getenv("MILVUS_PASSWORD"),
                "secure": os.getenv("MILVUS_SECURE", "false").lower() == "true"
            }
        }
        
        config = env_configs.get(provider, {})
        # Remove None values
        return {k: v for k, v in config.items() if v is not None}
    
    @classmethod
    def create_from_env(
        self,
        provider: Optional[VectorDBProvider] = None,
        **kwargs
    ) -> VectorDBClient:
        """
        Create a vector database client using environment variables.
        
        Args:
            provider: Specific provider to use (if None, will try to detect)
            **kwargs: Additional configuration options
        
        Returns:
            Configured vector database client
        """
        if provider is None:
            # Try to detect from environment
            if os.getenv("QDRANT_HOST") or os.getenv("QDRANT_PORT"):
                provider = VectorDBProvider.QDRANT
            elif os.getenv("MILVUS_HOST") or os.getenv("MILVUS_PORT"):
                provider = VectorDBProvider.MILVUS
            else:
                # Default to Qdrant
                provider = VectorDBProvider.QDRANT
        
        # Get configuration from environment
        env_config = self.get_connection_info_from_env(provider)
        env_config.update(kwargs)
        
        return self.create_client(provider, **env_config)

# Global factory instance
vectordb_factory = VectorDBFactory()
