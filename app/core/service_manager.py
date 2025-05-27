"""
Service manager for handling service dependencies and initialization.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

from .config import get_settings

logger = logging.getLogger(__name__)

class ServiceManager:
    """
    Manages service lifecycle and dependencies.
    """

    def __init__(self):
        self.settings = get_settings()
        self._services: Dict[str, Any] = {}
        self._initialized: Dict[str, bool] = {}
        self._health_status: Dict[str, Dict[str, Any]] = {}

    async def initialize_all_services(self) -> None:
        """Initialize all services in correct dependency order."""
        logger.info("Initializing all services...")

        # Service initialization order (dependencies first)
        service_order = [
            "embedding",
            "vectordb",
            "llm",
            "parsing",
            "chunking",
            "indexing",
            "retrieval"
        ]

        for service_name in service_order:
            try:
                await self._initialize_service(service_name)
                logger.info(f"✅ {service_name} service initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize {service_name} service: {e}")
                # Continue with other services for graceful degradation

        logger.info("Service initialization completed")

    async def _initialize_service(self, service_name: str) -> None:
        """Initialize a specific service."""
        if service_name in self._initialized and self._initialized[service_name]:
            return

        try:
            if service_name == "embedding":
                await self._init_embedding_service()
            elif service_name == "vectordb":
                await self._init_vectordb_service()
            elif service_name == "llm":
                await self._init_llm_service()
            elif service_name == "parsing":
                await self._init_parsing_service()
            elif service_name == "chunking":
                await self._init_chunking_service()
            elif service_name == "indexing":
                await self._init_indexing_service()
            elif service_name == "retrieval":
                await self._init_retrieval_service()

            self._initialized[service_name] = True

        except Exception as e:
            self._initialized[service_name] = False
            raise e

    async def _init_embedding_service(self) -> None:
        """Initialize embedding service."""
        from ..services.embedding.service import embedding_service

        # Configure for test environment
        if self.settings.ENVIRONMENT == "testing" or not self.settings.OPENAI_API_KEY:
            # Use mock embedding service
            from ..services.mocks.mock_embedding_service import MockEmbeddingService
            mock_service = MockEmbeddingService()
            await mock_service.initialize()
            self._services["embedding"] = mock_service
            logger.info("Using mock embedding service for testing")
        else:
            await embedding_service.initialize()
            self._services["embedding"] = embedding_service

    async def _init_vectordb_service(self) -> None:
        """Initialize vector database service."""
        from ..services.vectordb.service import vectordb_service

        # Try to connect to real Qdrant, fallback to mock if needed
        try:
            await vectordb_service.initialize()
            # Test connection
            health = await vectordb_service.health_check()
            if health.get("status") == "healthy":
                self._services["vectordb"] = vectordb_service
                logger.info("Connected to Qdrant vector database")
            else:
                raise Exception("Qdrant health check failed")
        except Exception as e:
            logger.warning(f"Failed to connect to Qdrant: {e}")
            # Use mock vector database
            from ..services.mocks.mock_vectordb_service import MockVectorDBService
            mock_service = MockVectorDBService()
            await mock_service.initialize()
            self._services["vectordb"] = mock_service
            logger.info("Using mock vector database service")

    async def _init_llm_service(self) -> None:
        """Initialize LLM service."""
        from ..services.llm.service import llm_service

        # Configure based on available API keys
        if self.settings.ENVIRONMENT == "testing" or not self.settings.LITELLM_API_KEY:
            # Use mock LLM service
            from ..services.mocks.mock_llm_service import MockLLMService
            mock_service = MockLLMService()
            await mock_service.initialize()
            self._services["llm"] = mock_service
            logger.info("Using mock LLM service for testing")
        else:
            await llm_service.initialize()
            self._services["llm"] = llm_service

    async def _init_parsing_service(self) -> None:
        """Initialize parsing service."""
        from ..services.parsers.service import parsing_service
        await parsing_service.initialize()
        self._services["parsing"] = parsing_service

    async def _init_chunking_service(self) -> None:
        """Initialize chunking service."""
        from ..services.chunking.service import chunking_service
        await chunking_service.initialize()
        self._services["chunking"] = chunking_service

    async def _init_indexing_service(self) -> None:
        """Initialize indexing service."""
        # Import locally to avoid early import
        from ..services.indexing.service import IndexingService
        from ..services.indexing.base import IndexingConfig

        # Create new instance instead of using global
        config = IndexingConfig()
        indexing_service = IndexingService(config)

        # Inject mock services if in test mode
        if self.settings.ENVIRONMENT == "testing":
            # Replace dependencies with mock services
            indexing_service._embedding_service = self._services.get("embedding")
            indexing_service._vectordb_service = self._services.get("vectordb")
            indexing_service._chunking_service = self._services.get("chunking")
            indexing_service._parsing_service = self._services.get("parsing")

        await indexing_service.initialize()
        self._services["indexing"] = indexing_service

    async def _init_retrieval_service(self) -> None:
        """Initialize retrieval service."""
        # Import locally to avoid early import
        from ..services.retrieval.service import RetrievalService
        from ..services.retrieval.base import RetrievalConfig

        # Create new instance instead of using global
        config = RetrievalConfig()
        retrieval_service = RetrievalService(config)

        # Inject mock services if in test mode
        if self.settings.ENVIRONMENT == "testing":
            # Replace dependencies with mock services
            retrieval_service._embedding_service = self._services.get("embedding")
            retrieval_service._vectordb_service = self._services.get("vectordb")

        await retrieval_service.initialize()
        self._services["retrieval"] = retrieval_service

    def get_service(self, service_name: str) -> Optional[Any]:
        """Get a service instance."""
        return self._services.get(service_name)

    def is_service_initialized(self, service_name: str) -> bool:
        """Check if a service is initialized."""
        return self._initialized.get(service_name, False)

    async def health_check_all(self) -> Dict[str, Any]:
        """Perform health check on all services."""
        health_results = {}

        for service_name, service in self._services.items():
            try:
                if hasattr(service, 'health_check'):
                    health = await service.health_check()
                    health_results[service_name] = health
                else:
                    health_results[service_name] = {
                        "status": "healthy" if self._initialized.get(service_name, False) else "unknown",
                        "initialized": self._initialized.get(service_name, False)
                    }
            except Exception as e:
                health_results[service_name] = {
                    "status": "unhealthy",
                    "error": str(e),
                    "initialized": self._initialized.get(service_name, False)
                }

        # Overall health status
        healthy_services = sum(1 for h in health_results.values() if h.get("status") == "healthy")
        total_services = len(health_results)

        overall_status = "healthy" if healthy_services == total_services else (
            "degraded" if healthy_services > 0 else "unhealthy"
        )

        return {
            "status": overall_status,
            "services": health_results,
            "summary": {
                "total_services": total_services,
                "healthy_services": healthy_services,
                "unhealthy_services": total_services - healthy_services,
                "health_percentage": (healthy_services / total_services * 100) if total_services > 0 else 0
            }
        }

    async def cleanup_all(self) -> None:
        """Cleanup all services."""
        logger.info("Cleaning up all services...")

        for service_name, service in self._services.items():
            try:
                if hasattr(service, 'cleanup'):
                    await service.cleanup()
                logger.info(f"✅ {service_name} service cleaned up")
            except Exception as e:
                logger.error(f"❌ Failed to cleanup {service_name} service: {e}")

        self._services.clear()
        self._initialized.clear()
        logger.info("Service cleanup completed")

    def get_service_stats(self) -> Dict[str, Any]:
        """Get statistics for all services."""
        stats = {}

        for service_name, service in self._services.items():
            try:
                if hasattr(service, 'get_stats'):
                    stats[service_name] = service.get_stats()
                else:
                    stats[service_name] = {
                        "initialized": self._initialized.get(service_name, False),
                        "type": type(service).__name__
                    }
            except Exception as e:
                stats[service_name] = {
                    "error": str(e),
                    "initialized": self._initialized.get(service_name, False)
                }

        return stats

# Global service manager instance
_service_manager: Optional[ServiceManager] = None

def get_service_manager() -> ServiceManager:
    """Get global service manager instance."""
    global _service_manager
    if _service_manager is None:
        _service_manager = ServiceManager()
    return _service_manager

@asynccontextmanager
async def service_lifespan():
    """Context manager for service lifecycle."""
    manager = get_service_manager()
    try:
        await manager.initialize_all_services()
        yield manager
    finally:
        await manager.cleanup_all()

async def initialize_services_for_testing() -> ServiceManager:
    """Initialize services specifically for testing."""
    import os

    # Set test environment
    os.environ["ENVIRONMENT"] = "testing"
    os.environ["EMBEDDING_PROVIDER"] = "mock"
    os.environ["LLM_PROVIDER"] = "mock"
    os.environ["MOCK_EXTERNAL_SERVICES"] = "true"

    manager = get_service_manager()
    await manager.initialize_all_services()
    return manager
