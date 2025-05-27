"""
Admin Settings API
Manage system settings, LLM providers, and configurations
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
import logging

from app.core.auth import get_current_user, require_admin
from app.models.user import User
from app.services.settings.settings_service import (
    settings_service, SettingScope, SettingCategory
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Pydantic models
class SettingValue(BaseModel):
    value: Any
    change_reason: Optional[str] = None

class LLMProviderConfig(BaseModel):
    provider: str = Field(..., description="LLM provider name")
    api_key: Optional[str] = Field(None, description="API key")
    base_url: Optional[str] = Field(None, description="Base URL")
    model: Optional[str] = Field(None, description="Model name")
    max_tokens: Optional[int] = Field(None, description="Max tokens")
    temperature: Optional[float] = Field(None, description="Temperature")

class LiteLLMConfig(BaseModel):
    base_url: Optional[str] = Field(None, description="LiteLLM proxy base URL")
    api_key: Optional[str] = Field(None, description="LiteLLM proxy API key")
    timeout: Optional[int] = Field(None, description="Request timeout in seconds")
    max_retries: Optional[int] = Field(None, description="Maximum retry attempts")

class VectorDBConfig(BaseModel):
    type: str = Field(..., description="Vector DB type")
    url: Optional[str] = Field(None, description="Database URL")
    api_key: Optional[str] = Field(None, description="API key")
    dimension: Optional[int] = Field(None, description="Vector dimension")

class SecurityConfig(BaseModel):
    jwt_expiration_hours: Optional[int] = None
    max_login_attempts: Optional[int] = None
    password_min_length: Optional[int] = None
    rate_limit_requests_per_minute: Optional[int] = None

# Get all settings by category
@router.get("/settings/{category}")
async def get_settings_by_category(
    category: SettingCategory,
    scope: SettingScope = Query(SettingScope.GLOBAL),
    organization_id: Optional[str] = Query(None),
    workspace_id: Optional[str] = Query(None),
    current_user: User = Depends(require_admin)
):
    """Get all settings in a category"""

    try:
        await settings_service.initialize()

        settings = await settings_service.get_settings_by_category(
            category=category,
            scope=scope,
            organization_id=organization_id,
            workspace_id=workspace_id,
            user_id=str(current_user.id)
        )

        return {
            "category": category.value,
            "scope": scope.value,
            "settings": settings,
            "total": len(settings)
        }

    except Exception as e:
        logger.error(f"Error getting settings for category {category}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get settings")

# Get specific setting
@router.get("/settings/{category}/{key}")
async def get_setting(
    category: SettingCategory,
    key: str,
    scope: SettingScope = Query(SettingScope.GLOBAL),
    organization_id: Optional[str] = Query(None),
    workspace_id: Optional[str] = Query(None),
    current_user: User = Depends(require_admin)
):
    """Get specific setting value"""

    try:
        await settings_service.initialize()

        value = await settings_service.get_setting(
            key=key,
            scope=scope,
            organization_id=organization_id,
            workspace_id=workspace_id,
            user_id=str(current_user.id)
        )

        return {
            "key": key,
            "category": category.value,
            "scope": scope.value,
            "value": value
        }

    except Exception as e:
        logger.error(f"Error getting setting {key}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get setting")

# Update specific setting
@router.put("/settings/{category}/{key}")
async def update_setting(
    category: SettingCategory,
    key: str,
    setting_value: SettingValue,
    scope: SettingScope = Query(SettingScope.GLOBAL),
    organization_id: Optional[str] = Query(None),
    workspace_id: Optional[str] = Query(None),
    current_user: User = Depends(require_admin)
):
    """Update specific setting value"""

    try:
        await settings_service.initialize()

        success = await settings_service.set_setting(
            key=key,
            value=setting_value.value,
            scope=scope,
            organization_id=organization_id,
            workspace_id=workspace_id,
            changed_by=str(current_user.id),
            change_reason=setting_value.change_reason
        )

        if success:
            return {
                "message": f"Setting {key} updated successfully",
                "key": key,
                "value": setting_value.value
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to update setting")

    except Exception as e:
        logger.error(f"Error updating setting {key}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update setting")

# Get LLM configuration
@router.get("/llm/config")
async def get_llm_config(
    organization_id: Optional[str] = Query(None),
    workspace_id: Optional[str] = Query(None),
    current_user: User = Depends(require_admin)
):
    """Get complete LLM configuration"""

    try:
        await settings_service.initialize()

        config = await settings_service.get_llm_config(
            organization_id=organization_id,
            workspace_id=workspace_id,
            user_id=str(current_user.id)
        )

        # Mask sensitive values
        if 'openai' in config and 'api_key' in config['openai']:
            if config['openai']['api_key']:
                config['openai']['api_key'] = '***' + config['openai']['api_key'][-4:]

        if 'anthropic' in config and 'api_key' in config['anthropic']:
            if config['anthropic']['api_key']:
                config['anthropic']['api_key'] = '***' + config['anthropic']['api_key'][-4:]

        return config

    except Exception as e:
        logger.error(f"Error getting LLM config: {e}")
        raise HTTPException(status_code=500, detail="Failed to get LLM configuration")

# Update LLM provider
@router.put("/llm/provider")
async def update_llm_provider(
    config: LLMProviderConfig,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(require_admin)
):
    """Update LLM provider configuration"""

    try:
        await settings_service.initialize()

        # Prepare config dict
        provider_config = {}
        if config.api_key:
            provider_config['api_key'] = config.api_key
        if config.base_url:
            provider_config['base_url'] = config.base_url
        if config.model:
            provider_config['model'] = config.model
        if config.max_tokens:
            provider_config['max_tokens'] = config.max_tokens
        if config.temperature is not None:
            provider_config['temperature'] = config.temperature

        success = await settings_service.update_llm_provider(
            provider=config.provider,
            config=provider_config,
            changed_by=str(current_user.id),
            organization_id=organization_id
        )

        if success:
            return {
                "message": f"LLM provider updated to {config.provider}",
                "provider": config.provider
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to update LLM provider")

    except Exception as e:
        logger.error(f"Error updating LLM provider: {e}")
        raise HTTPException(status_code=500, detail="Failed to update LLM provider")

# Test LLM connection
@router.post("/llm/test")
async def test_llm_connection(
    organization_id: Optional[str] = Query(None),
    workspace_id: Optional[str] = Query(None),
    current_user: User = Depends(require_admin)
):
    """Test LLM provider connection"""

    try:
        await settings_service.initialize()

        # Get LLM config
        config = await settings_service.get_llm_config(
            organization_id=organization_id,
            workspace_id=workspace_id,
            user_id=str(current_user.id)
        )

        # Test connection based on provider
        provider = config.get('provider', 'openai')

        # Use dynamic LiteLLM service for testing
        from app.services.llm.dynamic_litellm_service import dynamic_litellm_service

        try:
            # Test connection using LiteLLM
            validation_result = await dynamic_litellm_service.validate_configuration(
                organization_id=organization_id,
                workspace_id=workspace_id,
                user_id=str(current_user.id)
            )

            return validation_result

        except Exception as e:
            return {
                "status": "error",
                "message": f"Connection test failed: {str(e)}"
            }

    except Exception as e:
        logger.error(f"Error testing LLM connection: {e}")
        return {
            "status": "error",
            "message": f"Connection failed: {str(e)}"
        }

# Get vector database configuration
@router.get("/vector-db/config")
async def get_vector_db_config(
    organization_id: Optional[str] = Query(None),
    workspace_id: Optional[str] = Query(None),
    current_user: User = Depends(require_admin)
):
    """Get vector database configuration"""

    try:
        await settings_service.initialize()

        config = await settings_service.get_vector_db_config(
            organization_id=organization_id,
            workspace_id=workspace_id,
            user_id=str(current_user.id)
        )

        # Mask API key
        if config.get('qdrant_api_key'):
            config['qdrant_api_key'] = '***' + config['qdrant_api_key'][-4:]

        return config

    except Exception as e:
        logger.error(f"Error getting vector DB config: {e}")
        raise HTTPException(status_code=500, detail="Failed to get vector DB configuration")

# Update vector database configuration
@router.put("/vector-db/config")
async def update_vector_db_config(
    config: VectorDBConfig,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(require_admin)
):
    """Update vector database configuration"""

    try:
        await settings_service.initialize()

        # Update settings
        updates = [
            ('vector_db_type', config.type),
        ]

        if config.url:
            updates.append(('qdrant_url', config.url))
        if config.api_key:
            updates.append(('qdrant_api_key', config.api_key))
        if config.dimension:
            updates.append(('vector_dimension', config.dimension))

        success_count = 0
        for key, value in updates:
            success = await settings_service.set_setting(
                key=key,
                value=value,
                scope=SettingScope.GLOBAL,
                organization_id=organization_id,
                changed_by=str(current_user.id),
                change_reason=f"Updated vector DB {key}"
            )
            if success:
                success_count += 1

        if success_count > 0:
            return {
                "message": f"Vector DB configuration updated ({success_count} settings)",
                "type": config.type
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to update vector DB configuration")

    except Exception as e:
        logger.error(f"Error updating vector DB config: {e}")
        raise HTTPException(status_code=500, detail="Failed to update vector DB configuration")

# Get settings history
@router.get("/settings/history")
async def get_settings_history(
    key: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    current_user: User = Depends(require_admin)
):
    """Get settings change history"""

    try:
        await settings_service.initialize()

        history = await settings_service.get_settings_history(
            key=key,
            limit=limit
        )

        return {
            "history": history,
            "total": len(history),
            "key_filter": key
        }

    except Exception as e:
        logger.error(f"Error getting settings history: {e}")
        raise HTTPException(status_code=500, detail="Failed to get settings history")

# Get LiteLLM configuration
@router.get("/litellm/config")
async def get_litellm_config(
    organization_id: Optional[str] = Query(None),
    workspace_id: Optional[str] = Query(None),
    current_user: User = Depends(require_admin)
):
    """Get LiteLLM proxy configuration"""

    try:
        await settings_service.initialize()

        config = {
            'base_url': await settings_service.get_setting('litellm_base_url', SettingScope.GLOBAL, organization_id, workspace_id, str(current_user.id)),
            'api_key': await settings_service.get_setting('litellm_api_key', SettingScope.GLOBAL, organization_id, workspace_id, str(current_user.id)),
            'timeout': await settings_service.get_setting('litellm_timeout', SettingScope.GLOBAL, organization_id, workspace_id, str(current_user.id)),
            'max_retries': await settings_service.get_setting('litellm_max_retries', SettingScope.GLOBAL, organization_id, workspace_id, str(current_user.id))
        }

        # Mask API key
        if config.get('api_key'):
            config['api_key'] = '***' + config['api_key'][-4:] if len(config['api_key']) > 4 else '***'

        return config

    except Exception as e:
        logger.error(f"Error getting LiteLLM config: {e}")
        raise HTTPException(status_code=500, detail="Failed to get LiteLLM configuration")

# Update LiteLLM configuration
@router.put("/litellm/config")
async def update_litellm_config(
    config: LiteLLMConfig,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(require_admin)
):
    """Update LiteLLM proxy configuration"""

    try:
        await settings_service.initialize()

        # Update LiteLLM settings
        updates = []
        if config.base_url is not None:
            updates.append(('litellm_base_url', config.base_url))
        if config.api_key is not None:
            updates.append(('litellm_api_key', config.api_key))
        if config.timeout is not None:
            updates.append(('litellm_timeout', config.timeout))
        if config.max_retries is not None:
            updates.append(('litellm_max_retries', config.max_retries))

        success_count = 0
        for key, value in updates:
            success = await settings_service.set_setting(
                key=key,
                value=value,
                scope=SettingScope.GLOBAL,
                organization_id=organization_id,
                changed_by=str(current_user.id),
                change_reason=f"Updated LiteLLM {key}"
            )
            if success:
                success_count += 1

        if success_count > 0:
            return {
                "message": f"LiteLLM configuration updated ({success_count} settings)",
                "updated_settings": success_count
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to update LiteLLM configuration")

    except Exception as e:
        logger.error(f"Error updating LiteLLM config: {e}")
        raise HTTPException(status_code=500, detail="Failed to update LiteLLM configuration")

# Get model configuration by type
@router.get("/models/{model_type}")
async def get_model_config_by_type(
    model_type: str,
    organization_id: Optional[str] = Query(None),
    workspace_id: Optional[str] = Query(None),
    current_user: User = Depends(require_admin)
):
    """Get model configuration by type (chat, completion, embedding, vision)"""

    try:
        from app.services.llm.dynamic_litellm_service import dynamic_litellm_service

        await dynamic_litellm_service.initialize()

        config = await dynamic_litellm_service.get_model_config_by_type(
            model_type=model_type,
            organization_id=organization_id,
            workspace_id=workspace_id,
            user_id=str(current_user.id)
        )

        # Mask sensitive values
        if config.get('api_key'):
            config['api_key'] = '***' + config['api_key'][-4:] if len(config['api_key']) > 4 else '***'

        return {
            "model_type": model_type,
            "config": config
        }

    except Exception as e:
        logger.error(f"Error getting {model_type} config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get {model_type} configuration")

# Get available model types and providers
@router.get("/providers/summary")
async def get_providers_summary(
    current_user: User = Depends(require_admin)
):
    """Get summary of available providers by model type"""

    return {
        "model_types": {
            "chat": {
                "description": "Chat completion models for conversational AI",
                "providers": ["openai", "anthropic", "google", "azure", "aws"],
                "default": "openai"
            },
            "completion": {
                "description": "Text completion models for text generation",
                "providers": ["openai", "anthropic", "google", "azure", "aws"],
                "default": "openai"
            },
            "embedding": {
                "description": "Text embedding models for vector representations",
                "providers": ["openai", "voyage", "cohere", "azure"],
                "default": "openai"
            },
            "vision": {
                "description": "Vision models for image understanding",
                "providers": ["openai", "anthropic", "google", "azure"],
                "default": "openai"
            }
        },
        "providers": {
            "openai": {
                "name": "OpenAI",
                "supports": ["chat", "completion", "embedding", "vision"],
                "latest_models": {
                    "chat": "gpt-4o",
                    "embedding": "text-embedding-3-small",
                    "vision": "gpt-4o"
                }
            },
            "anthropic": {
                "name": "Anthropic",
                "supports": ["chat", "completion", "vision"],
                "latest_models": {
                    "chat": "claude-3-5-sonnet-20241022",
                    "vision": "claude-3-5-sonnet-20241022"
                }
            },
            "google": {
                "name": "Google Gemini",
                "supports": ["chat", "vision"],
                "latest_models": {
                    "chat": "gemini-2.0-flash-exp",
                    "vision": "gemini-2.0-flash-exp"
                }
            },
            "azure": {
                "name": "Azure OpenAI",
                "supports": ["chat", "embedding"],
                "latest_models": {
                    "chat": "gpt-4o",
                    "embedding": "text-embedding-3-small"
                }
            },
            "aws": {
                "name": "AWS Bedrock",
                "supports": ["chat"],
                "latest_models": {
                    "chat": "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0"
                }
            },
            "voyage": {
                "name": "Voyage AI",
                "supports": ["embedding"],
                "latest_models": {
                    "embedding": "voyage-3"
                }
            },
            "cohere": {
                "name": "Cohere",
                "supports": ["embedding"],
                "latest_models": {
                    "embedding": "embed-english-v3.0"
                }
            }
        },
        "vector_databases": {
            "qdrant": {
                "name": "Qdrant",
                "description": "Open-source vector database with excellent performance"
            },
            "pinecone": {
                "name": "Pinecone",
                "description": "Managed vector database with serverless scaling"
            },
            "weaviate": {
                "name": "Weaviate",
                "description": "Open-source vector database with GraphQL API"
            }
        }
    }
