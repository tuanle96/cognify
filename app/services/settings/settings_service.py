"""
Dynamic Settings Service
Manage system settings stored in database with caching and validation
"""

from typing import Any, Dict, List, Optional, Union
from uuid import UUID
import json
import logging
from datetime import datetime, timedelta
from enum import Enum

from sqlalchemy import text
from app.core.database import database_manager
from app.core.cache import get_cache

logger = logging.getLogger(__name__)

class SettingScope(str, Enum):
    GLOBAL = "global"
    ORGANIZATION = "organization"
    WORKSPACE = "workspace"
    USER = "user"

class SettingCategory(str, Enum):
    LLM_PROVIDER = "llm_provider"
    EMBEDDING_PROVIDER = "embedding_provider"
    VECTOR_DATABASE = "vector_database"
    SECURITY = "security"
    PERFORMANCE = "performance"
    FEATURES = "features"
    NOTIFICATIONS = "notifications"
    INTEGRATIONS = "integrations"

class SettingDataType(str, Enum):
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    JSON = "json"
    ENCRYPTED = "encrypted"

class SettingsService:
    """Service for managing dynamic system settings"""

    def __init__(self):
        self.db = None
        self.cache = None
        self.cache_ttl = 300  # 5 minutes
        self.cache_prefix = "settings:"

    async def initialize(self):
        """Initialize the service"""
        # Initialize database manager if not already done
        if not database_manager.is_initialized:
            await database_manager.initialize()

        # Use database manager's engine for raw queries
        self.db = database_manager.engine

        # Initialize cache
        try:
            self.cache = await get_cache()
            logger.info("Settings cache initialized successfully")
        except Exception as e:
            # Cache is optional, continue without it
            logger.warning(f"Failed to initialize settings cache: {e}")
            self.cache = None

    def _get_cache_key(self, key: str, scope: SettingScope,
                      organization_id: Optional[str] = None,
                      workspace_id: Optional[str] = None,
                      user_id: Optional[str] = None) -> str:
        """Generate cache key for setting"""
        parts = [self.cache_prefix, key, scope.value]
        if organization_id:
            parts.append(f"org:{organization_id}")
        if workspace_id:
            parts.append(f"ws:{workspace_id}")
        if user_id:
            parts.append(f"user:{user_id}")
        return ":".join(parts)

    async def get_setting(self,
                         key: str,
                         scope: SettingScope = SettingScope.GLOBAL,
                         organization_id: Optional[str] = None,
                         workspace_id: Optional[str] = None,
                         user_id: Optional[str] = None,
                         use_cache: bool = True) -> Optional[Any]:
        """Get setting value with scope hierarchy and caching"""

        try:
            # Check cache first
            if use_cache and self.cache:
                cache_key = self._get_cache_key(key, scope, organization_id, workspace_id, user_id)
                cached_value = await self.cache.get(cache_key)
                if cached_value is not None:
                    return self._deserialize_value(cached_value)

            # Get from database with scope hierarchy
            async with self.db.connect() as conn:
                # Try to get setting value with scope hierarchy
                result = await conn.execute(text("""
                    SELECT value, data_type, is_sensitive FROM system_settings
                    WHERE key = :key AND scope = :scope
                    AND (
                        (scope = 'global') OR
                        (scope = 'organization' AND organization_id = :org_id) OR
                        (scope = 'workspace' AND workspace_id = :workspace_id) OR
                        (scope = 'user' AND user_id = :user_id)
                    )
                    ORDER BY
                        CASE scope
                            WHEN 'user' THEN 1
                            WHEN 'workspace' THEN 2
                            WHEN 'organization' THEN 3
                            WHEN 'global' THEN 4
                        END
                    LIMIT 1
                """), {
                    "key": key,
                    "scope": scope.value,
                    "org_id": organization_id,
                    "workspace_id": workspace_id,
                    "user_id": user_id
                })
                setting_row = result.first()

                if setting_row and setting_row[0] is not None:
                    deserialized_value = self._deserialize_value(setting_row[0], setting_row[1])

                    # Cache the result
                    if use_cache and self.cache:
                        cache_key = self._get_cache_key(key, scope, organization_id, workspace_id, user_id)
                        await self.cache.set(cache_key, setting_row[0], ttl=self.cache_ttl)

                    return deserialized_value

            return None

        except Exception as e:
            logger.error(f"Error getting setting {key}: {e}")
            return None

    async def set_setting(self,
                         key: str,
                         value: Any,
                         scope: SettingScope = SettingScope.GLOBAL,
                         organization_id: Optional[str] = None,
                         workspace_id: Optional[str] = None,
                         user_id: Optional[str] = None,
                         changed_by: Optional[str] = None,
                         change_reason: Optional[str] = None) -> bool:
        """Set setting value with validation and history tracking"""

        try:
            # Serialize value
            serialized_value = self._serialize_value(value)

            # Validate value
            if not await self._validate_setting_value(key, value, scope):
                logger.error(f"Invalid value for setting {key}: {value}")
                return False

            # Set in database
            async with self.db.connect() as conn:
                # Check if setting exists
                result = await conn.execute(text("""
                    SELECT id FROM system_settings
                    WHERE key = :key AND scope = :scope
                    AND (
                        (scope = 'global' AND organization_id IS NULL AND workspace_id IS NULL AND user_id IS NULL) OR
                        (scope = 'organization' AND organization_id = :org_id) OR
                        (scope = 'workspace' AND workspace_id = :workspace_id) OR
                        (scope = 'user' AND user_id = :user_id)
                    )
                """), {
                    "key": key,
                    "scope": scope.value,
                    "org_id": organization_id,
                    "workspace_id": workspace_id,
                    "user_id": user_id
                })
                setting_row = result.first()

                if setting_row:
                    # Update existing setting
                    await conn.execute(text("""
                        UPDATE system_settings
                        SET value = :value, updated_at = NOW(), updated_by = :changed_by
                        WHERE id = :setting_id
                    """), {
                        "value": serialized_value,
                        "changed_by": changed_by,
                        "setting_id": setting_row[0]
                    })
                    setting_id = setting_row[0]
                else:
                    # Setting doesn't exist, cannot create new ones dynamically
                    logger.error(f"Setting {key} with scope {scope.value} not found")
                    return False

            if setting_id:
                # Invalidate cache
                if self.cache:
                    cache_key = self._get_cache_key(key, scope, organization_id, workspace_id, user_id)
                    await self.cache.delete(cache_key)

                logger.info(f"Setting {key} updated successfully")
                return True

            return False

        except Exception as e:
            logger.error(f"Error setting {key}: {e}")
            return False

    async def get_settings_by_category(self,
                                     category: SettingCategory,
                                     scope: SettingScope = SettingScope.GLOBAL,
                                     organization_id: Optional[str] = None,
                                     workspace_id: Optional[str] = None,
                                     user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get all settings in a category"""

        try:
            # Get settings metadata
            async with self.db.connect() as conn:
                result = await conn.execute(text("""
                    SELECT key, name, description, data_type, default_value, is_sensitive
                    FROM system_settings
                    WHERE category = :category AND scope = :scope
                    ORDER BY name
                """), {"category": category.value, "scope": scope.value})
                settings_meta = result.fetchall()

            result = {}
            for setting in settings_meta:
                value = await self.get_setting(
                    setting['key'], scope, organization_id, workspace_id, user_id
                )

                result[setting['key']] = {
                    'name': setting['name'],
                    'description': setting['description'],
                    'value': value if not setting['is_sensitive'] else '***' if value else None,
                    'default_value': setting['default_value'],
                    'data_type': setting['data_type'],
                    'is_sensitive': setting['is_sensitive']
                }

            return result

        except Exception as e:
            logger.error(f"Error getting settings for category {category}: {e}")
            return {}

    async def get_llm_config(self,
                           organization_id: Optional[str] = None,
                           workspace_id: Optional[str] = None,
                           user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get complete LLM configuration"""

        try:
            # Get default provider
            default_provider = await self.get_setting(
                'default_llm_provider', SettingScope.GLOBAL,
                organization_id, workspace_id, user_id
            ) or 'openai'

            config = {
                'provider': default_provider,
                'openai': {},
                'anthropic': {},
                'embedding': {}
            }

            # OpenAI settings
            if default_provider == 'openai':
                config['openai'] = {
                    'api_key': await self.get_setting('openai_api_key', SettingScope.GLOBAL, organization_id, workspace_id, user_id),
                    'base_url': await self.get_setting('openai_base_url', SettingScope.GLOBAL, organization_id, workspace_id, user_id),
                    'model': await self.get_setting('openai_model', SettingScope.GLOBAL, organization_id, workspace_id, user_id),
                    'max_tokens': await self.get_setting('openai_max_tokens', SettingScope.GLOBAL, organization_id, workspace_id, user_id),
                    'temperature': await self.get_setting('openai_temperature', SettingScope.GLOBAL, organization_id, workspace_id, user_id)
                }

            # Anthropic settings
            elif default_provider == 'anthropic':
                config['anthropic'] = {
                    'api_key': await self.get_setting('anthropic_api_key', SettingScope.GLOBAL, organization_id, workspace_id, user_id),
                    'model': await self.get_setting('anthropic_model', SettingScope.GLOBAL, organization_id, workspace_id, user_id),
                    'max_tokens': await self.get_setting('anthropic_max_tokens', SettingScope.GLOBAL, organization_id, workspace_id, user_id)
                }

            # Embedding settings
            embedding_provider = await self.get_setting('default_embedding_provider', SettingScope.GLOBAL, organization_id, workspace_id, user_id) or 'openai'
            config['embedding'] = {
                'provider': embedding_provider,
                'model': await self.get_setting('openai_embedding_model', SettingScope.GLOBAL, organization_id, workspace_id, user_id)
            }

            return config

        except Exception as e:
            logger.error(f"Error getting LLM config: {e}")
            return {}

    async def get_vector_db_config(self,
                                 organization_id: Optional[str] = None,
                                 workspace_id: Optional[str] = None,
                                 user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get vector database configuration"""

        try:
            return {
                'type': await self.get_setting('vector_db_type', SettingScope.GLOBAL, organization_id, workspace_id, user_id),
                'qdrant_url': await self.get_setting('qdrant_url', SettingScope.GLOBAL, organization_id, workspace_id, user_id),
                'qdrant_api_key': await self.get_setting('qdrant_api_key', SettingScope.GLOBAL, organization_id, workspace_id, user_id),
                'dimension': await self.get_setting('vector_dimension', SettingScope.GLOBAL, organization_id, workspace_id, user_id)
            }

        except Exception as e:
            logger.error(f"Error getting vector DB config: {e}")
            return {}

    async def update_llm_provider(self,
                                provider: str,
                                config: Dict[str, Any],
                                changed_by: str,
                                organization_id: Optional[str] = None) -> bool:
        """Update LLM provider configuration"""

        try:
            # Update default provider
            await self.set_setting(
                'default_llm_provider', provider, SettingScope.GLOBAL,
                organization_id=organization_id, changed_by=changed_by,
                change_reason=f"Updated LLM provider to {provider}"
            )

            # Update provider-specific settings
            if provider == 'openai':
                for key, value in config.items():
                    await self.set_setting(
                        f'openai_{key}', value, SettingScope.GLOBAL,
                        organization_id=organization_id, changed_by=changed_by,
                        change_reason=f"Updated OpenAI {key}"
                    )

            elif provider == 'anthropic':
                for key, value in config.items():
                    await self.set_setting(
                        f'anthropic_{key}', value, SettingScope.GLOBAL,
                        organization_id=organization_id, changed_by=changed_by,
                        change_reason=f"Updated Anthropic {key}"
                    )

            return True

        except Exception as e:
            logger.error(f"Error updating LLM provider: {e}")
            return False

    async def _validate_setting_value(self, key: str, value: Any, scope: SettingScope) -> bool:
        """Validate setting value against rules"""

        try:
            # Get validation rules
            async with self.db.connect() as conn:
                result = await conn.execute(text("""
                    SELECT data_type, validation_rules, is_required
                    FROM system_settings
                    WHERE key = :key AND scope = :scope
                """), {"key": key, "scope": scope.value})
                setting_info = result.first()

            if not setting_info:
                return True  # No validation rules

            data_type = setting_info[0]
            rules = setting_info[1] or {}
            is_required = setting_info[2]

            # Check required
            if is_required and (value is None or value == ''):
                return False

            # Type validation
            if value is not None:
                if data_type == 'integer' and not isinstance(value, int):
                    return False
                elif data_type == 'float' and not isinstance(value, (int, float)):
                    return False
                elif data_type == 'boolean' and not isinstance(value, bool):
                    return False

                # Rule validation
                if 'min' in rules and value < rules['min']:
                    return False
                if 'max' in rules and value > rules['max']:
                    return False
                if 'enum' in rules and value not in rules['enum']:
                    return False
                if 'min_length' in rules and len(str(value)) < rules['min_length']:
                    return False

            return True

        except Exception as e:
            logger.error(f"Error validating setting {key}: {e}")
            return False

    def _serialize_value(self, value: Any) -> str:
        """Serialize value for storage"""
        if value is None:
            return ''
        elif isinstance(value, (dict, list)):
            return json.dumps(value)
        elif isinstance(value, bool):
            return 'true' if value else 'false'
        else:
            return str(value)

    def _deserialize_value(self, value: str, data_type: str = 'string') -> Any:
        """Deserialize value from storage"""
        if not value:
            return None

        try:
            if data_type == 'integer':
                return int(value)
            elif data_type == 'float':
                return float(value)
            elif data_type == 'boolean':
                return value.lower() in ('true', '1', 'yes', 'on')
            elif data_type == 'json':
                return json.loads(value)
            else:
                return value
        except (ValueError, json.JSONDecodeError):
            return value

    async def get_settings_history(self,
                                 key: Optional[str] = None,
                                 limit: int = 100) -> List[Dict[str, Any]]:
        """Get settings change history"""

        try:
            query = """
                SELECT sh.*, ss.key, ss.name, u.email as changed_by_email
                FROM settings_history sh
                JOIN system_settings ss ON sh.setting_id = ss.id
                LEFT JOIN users u ON sh.changed_by = u.id
            """
            params = []

            if key:
                query += " WHERE ss.key = $1"
                params.append(key)

            query += " ORDER BY sh.changed_at DESC LIMIT $" + str(len(params) + 1)
            params.append(limit)

            async with self.db.connect() as conn:
                result = await conn.execute(text(query), params)
                rows = result.fetchall()

            return [
                {
                    'id': str(row['id']),
                    'setting_key': row['key'],
                    'setting_name': row['name'],
                    'old_value': row['old_value'],
                    'new_value': row['new_value'],
                    'change_reason': row['change_reason'],
                    'changed_by': str(row['changed_by']) if row['changed_by'] else None,
                    'changed_by_email': row['changed_by_email'],
                    'changed_at': row['changed_at'].isoformat(),
                    'metadata': row['metadata']
                }
                for row in rows
            ]

        except Exception as e:
            logger.error(f"Error getting settings history: {e}")
            return []

# Global settings service instance
settings_service = SettingsService()
