"""
Simple embedding service using LiteLLM only.
Universal embedding service that works with all providers through LiteLLM.
"""

import time
from typing import List, Dict, Any, Optional
import logging

from app.services.settings.settings_service import settings_service, SettingScope

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Simple embedding service using LiteLLM for all providers"""

    def __init__(self):
        self.litellm = None
        self._initialized = False
        self._stats = {
            "requests": 0,
            "total_processing_time": 0.0,
            "total_tokens": 0
        }

    async def initialize(self):
        """Initialize LiteLLM for embeddings"""
        if self._initialized:
            return

        try:
            import litellm
            self.litellm = litellm

            # Enable logging for debugging
            litellm.set_verbose = True

            await settings_service.initialize()
            self._initialized = True
            logger.info("✅ Embedding service initialized with LiteLLM")

        except ImportError:
            logger.error("❌ LiteLLM not installed. Run: pip install litellm")
            raise
        except Exception as e:
            logger.error(f"❌ Error initializing embedding service: {e}")
            raise

    async def get_config(self,
                       organization_id: Optional[str] = None,
                       workspace_id: Optional[str] = None,
                       user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get embedding configuration from database (simplified)"""

        try:
            # Get embedding settings - LiteLLM auto-detects provider from model name
            model = await settings_service.get_setting('llm_embedding_model', SettingScope.GLOBAL, organization_id, workspace_id, user_id) or 'text-embedding-004'
            api_key = await settings_service.get_setting('llm_api_key', SettingScope.GLOBAL, organization_id, workspace_id, user_id)
            base_url = await settings_service.get_setting('llm_base_url', SettingScope.GLOBAL, organization_id, workspace_id, user_id)

            config = {
                'model': model,
                'api_key': api_key,
                'api_base': base_url,
                'timeout': 30
            }

            logger.info(f"🔍 DEBUG: Embedding config - Model: {model}, Base URL: {base_url}, API Key: {'***' + api_key[-10:] if api_key else 'None'}")

            # Force debug print to console
            print(f"🔍 EMBEDDING DEBUG: Model: {model}")
            print(f"🔍 EMBEDDING DEBUG: Base URL: {base_url}")
            print(f"🔍 EMBEDDING DEBUG: API Key: {'***' + api_key[-10:] if api_key else 'None'}")

            return config

        except Exception as e:
            logger.error(f"❌ Error getting embedding config: {e}")
            return {'model': 'text-embedding-004', 'api_key': None}

    async def get_embeddings(self,
                           texts: List[str],
                           organization_id: Optional[str] = None,
                           workspace_id: Optional[str] = None,
                           user_id: Optional[str] = None,
                           **kwargs) -> List[List[float]]:
        """Generate embeddings using LiteLLM"""

        try:
            if not self._initialized:
                await self.initialize()

            # Get configuration
            config = await self.get_config(organization_id, workspace_id, user_id)

            # Prepare LiteLLM parameters
            model_name = config['model']

            # Fix LiteLLM routing: text-embedding-004 is treated as Vertex AI by default
            # Need to prefix with openai/ or set custom_llm_provider when using OpenAI proxy
            if model_name == 'text-embedding-004' and config.get('api_base'):
                model_name = f"openai/{model_name}"

            litellm_kwargs = {
                'model': model_name,
                'input': texts,
                'timeout': config.get('timeout', 30)
            }

            # Add API key and base URL if configured
            if config.get('api_key'):
                litellm_kwargs['api_key'] = config['api_key']
            if config.get('api_base'):
                litellm_kwargs['api_base'] = config['api_base']

            logger.info(f"🔍 DEBUG: Making embedding request with model: {model_name} (original: {config['model']})")

            # Make the request
            start_time = time.time()
            response = await self.litellm.aembedding(**litellm_kwargs)
            processing_time = time.time() - start_time

            # Extract embeddings
            embeddings = []
            for item in response.data:
                if hasattr(item, 'embedding'):
                    embeddings.append(item.embedding)
                elif isinstance(item, dict) and 'embedding' in item:
                    embeddings.append(item['embedding'])
                else:
                    raise ValueError(f"Unexpected response format: {type(item)}")

            # Update stats
            self._stats["requests"] += 1
            self._stats["total_processing_time"] += processing_time
            if hasattr(response, 'usage') and response.usage:
                self._stats["total_tokens"] += getattr(response.usage, 'total_tokens', len(texts))

            logger.info(f"✅ Generated {len(embeddings)} embeddings in {processing_time:.2f}s")
            return embeddings

        except Exception as e:
            logger.error(f"❌ Error in embeddings: {e}")
            raise

    async def get_single_embedding(self,
                                 text: str,
                                 organization_id: Optional[str] = None,
                                 workspace_id: Optional[str] = None,
                                 user_id: Optional[str] = None,
                                 **kwargs) -> List[float]:
        """Generate single embedding"""

        embeddings = await self.get_embeddings(
            [text], organization_id, workspace_id, user_id, **kwargs
        )
        return embeddings[0] if embeddings else []

    async def health_check(self) -> Dict[str, Any]:
        """Check service health"""
        try:
            if not self._initialized:
                return {"status": "unhealthy", "error": "Service not initialized"}

            return {
                "status": "healthy",
                "service": "litellm_embeddings",
                "stats": self._stats
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def cleanup(self):
        """Cleanup service resources"""
        self._initialized = False
        logger.info("Embedding service cleaned up")

    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics"""
        return self._stats.copy()


# Global embedding service instance
embedding_service = EmbeddingService()
