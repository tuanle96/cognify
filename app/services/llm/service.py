"""
LiteLLM Service
Universal LLM service using LiteLLM with database-driven configuration
Supports all major LLM providers: OpenAI, Anthropic, Google, Azure, AWS, etc.
"""

from typing import List, Dict, Any, Optional
import logging
import asyncio

from app.services.settings.settings_service import settings_service, SettingScope

logger = logging.getLogger(__name__)

class LiteLLMService:
    """Universal LLM service using LiteLLM with dynamic configuration"""

    def __init__(self):
        self.litellm = None
        self._initialized = False

    async def initialize(self):
        """Initialize LiteLLM"""
        if self._initialized:
            return

        try:
            import litellm
            self.litellm = litellm

            # Enable logging for debugging
            litellm.set_verbose = True

            await settings_service.initialize()
            self._initialized = True
            logger.info("✅ LiteLLM service initialized")

        except ImportError:
            logger.error("❌ LiteLLM not installed. Run: pip install litellm")
            raise
        except Exception as e:
            logger.error(f"❌ Error initializing LiteLLM: {e}")
            raise

    async def get_config(self,
                       organization_id: Optional[str] = None,
                       workspace_id: Optional[str] = None,
                       user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get LLM configuration from database (simplified)"""

        try:
            # Get basic settings - LiteLLM auto-detects provider from model name
            model = await settings_service.get_setting('llm_chat_model', SettingScope.GLOBAL, organization_id, workspace_id, user_id) or 'gpt-4o-mini'
            api_key = await settings_service.get_setting('llm_api_key', SettingScope.GLOBAL, organization_id, workspace_id, user_id)
            base_url = await settings_service.get_setting('llm_base_url', SettingScope.GLOBAL, organization_id, workspace_id, user_id)

            config = {
                'model': model,
                'api_key': api_key,
                'api_base': base_url,
                'temperature': 0.1,
                'max_tokens': 4000,
                'timeout': 30,
                'max_retries': 3
            }

            return config

        except Exception as e:
            logger.error(f"❌ Error getting config: {e}")
            return {'model': 'gpt-4o-mini', 'api_key': None}

    async def get_embedding_config(self,
                                 organization_id: Optional[str] = None,
                                 workspace_id: Optional[str] = None,
                                 user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get embedding configuration (simplified)"""

        try:
            # Get embedding settings - LiteLLM auto-detects provider from model name
            model = await settings_service.get_setting('llm_embedding_model', SettingScope.GLOBAL, organization_id, workspace_id, user_id) or 'text-embedding-004'
            api_key = await settings_service.get_setting('llm_api_key', SettingScope.GLOBAL, organization_id, workspace_id, user_id)
            base_url = await settings_service.get_setting('llm_base_url', SettingScope.GLOBAL, organization_id, workspace_id, user_id)

            config = {
                'model': model,
                'api_key': api_key,
                'api_base': base_url
            }

            return config

        except Exception as e:
            logger.error(f"❌ Error getting embedding config: {e}")
            return {'model': 'text-embedding-004', 'api_key': None}

    async def chat_completion(self,
                            messages: List[Dict[str, str]],
                            organization_id: Optional[str] = None,
                            workspace_id: Optional[str] = None,
                            user_id: Optional[str] = None,
                            **kwargs) -> str:
        """Generate chat completion using LiteLLM"""

        try:
            if not self._initialized:
                await self.initialize()

            # Get configuration
            config = await self.get_config(organization_id, workspace_id, user_id)

            # Prepare LiteLLM parameters
            litellm_kwargs = {
                'model': config['model'],
                'messages': messages,
                'temperature': kwargs.get('temperature', config.get('temperature', 0.1)),
                'max_tokens': kwargs.get('max_tokens', config.get('max_tokens', 4000)),
                'timeout': config.get('timeout', 30)
            }

            # Add API key and base URL if configured
            if config.get('api_key'):
                litellm_kwargs['api_key'] = config['api_key']
            if config.get('api_base'):
                litellm_kwargs['api_base'] = config['api_base']

            # Make the request with retry logic
            max_retries = config.get('max_retries', 3)

            for attempt in range(max_retries + 1):
                try:
                    response = await self.litellm.acompletion(**litellm_kwargs)
                    return response.choices[0].message.content

                except Exception as e:
                    if attempt == max_retries:
                        raise e
                    logger.warning(f"LiteLLM request failed (attempt {attempt + 1}), retrying: {e}")
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff

        except Exception as e:
            logger.error(f"❌ Error in chat completion: {e}")
            raise

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

            # Get embedding configuration
            config = await self.get_embedding_config(organization_id, workspace_id, user_id)

            # Prepare LiteLLM parameters
            litellm_kwargs = {
                'model': config['model'],
                'input': texts
            }

            # Add provider-specific parameters
            if config.get('api_key'):
                litellm_kwargs['api_key'] = config['api_key']
            if config.get('api_base'):
                litellm_kwargs['api_base'] = config['api_base']

            # Make the request
            response = await self.litellm.aembedding(**litellm_kwargs)

            return [embedding['embedding'] for embedding in response['data']]

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

            return {"status": "healthy", "service": "litellm"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def cleanup(self):
        """Cleanup service resources"""
        self._initialized = False
        logger.info("LiteLLM service cleaned up")

# Global LiteLLM service instance
llm_service = LiteLLMService()
