"""
Cognify API Keys Service
Manage user API keys for authenticating with Cognify API
"""

from typing import List, Dict, Any, Optional, Tuple
from uuid import UUID
import logging
from datetime import datetime, timedelta
from enum import Enum

from app.core.database import get_database
from app.core.cache import get_cache

logger = logging.getLogger(__name__)

class APIKeyStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    REVOKED = "revoked"
    EXPIRED = "expired"

class CognifyAPIKeysService:
    """Service for managing Cognify API keys"""
    
    def __init__(self):
        self.db = None
        self.cache = None
        self.cache_ttl = 300  # 5 minutes
        self.cache_prefix = "cognify_api_key:"
    
    async def initialize(self):
        """Initialize the service"""
        self.db = await get_database()
        self.cache = await get_cache()
    
    async def create_api_key(self,
                           user_id: str,
                           key_name: str,
                           description: Optional[str] = None,
                           organization_id: Optional[str] = None,
                           workspace_id: Optional[str] = None,
                           permissions: Optional[Dict[str, Any]] = None,
                           rate_limits: Optional[Dict[str, Any]] = None,
                           expires_at: Optional[datetime] = None,
                           created_from_ip: Optional[str] = None) -> Dict[str, Any]:
        """Create a new Cognify API key"""
        
        try:
            # Generate API key
            api_key = await self.db.fetchval("SELECT generate_cognify_api_key()")
            key_hash = await self.db.fetchval("SELECT hash_cognify_api_key($1)", api_key)
            key_prefix = await self.db.fetchval("SELECT get_api_key_prefix($1)", api_key)
            
            # Set default permissions if not provided
            if permissions is None:
                permissions = {
                    "endpoints": ["*"],  # All endpoints
                    "methods": ["GET", "POST", "PUT", "DELETE"],
                    "scopes": ["read", "write"]
                }
            
            # Set default rate limits if not provided
            if rate_limits is None:
                rate_limits = {
                    "requests_per_minute": 60,
                    "requests_per_hour": 1000,
                    "requests_per_day": 10000
                }
            
            # Insert the API key
            key_id = await self.db.fetchval("""
                INSERT INTO cognify_api_keys (
                    user_id, organization_id, workspace_id, key_name, 
                    key_description, key_prefix, key_hash, permissions,
                    rate_limits, expires_at, created_from_ip, status
                ) VALUES ($1::UUID, $2::UUID, $3::UUID, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                RETURNING id
            """, 
                UUID(user_id),
                UUID(organization_id) if organization_id else None,
                UUID(workspace_id) if workspace_id else None,
                key_name,
                description,
                key_prefix,
                key_hash,
                permissions,
                rate_limits,
                expires_at,
                created_from_ip,
                APIKeyStatus.ACTIVE.value
            )
            
            logger.info(f"Created Cognify API key for user {user_id}: {key_name}")
            
            return {
                "id": str(key_id),
                "api_key": api_key,  # Return full key only once
                "key_name": key_name,
                "key_prefix": key_prefix,
                "permissions": permissions,
                "rate_limits": rate_limits,
                "expires_at": expires_at.isoformat() if expires_at else None,
                "created_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error creating Cognify API key: {e}")
            raise
    
    async def get_user_api_keys(self,
                              user_id: str,
                              organization_id: Optional[str] = None,
                              workspace_id: Optional[str] = None,
                              include_inactive: bool = False) -> List[Dict[str, Any]]:
        """Get user's Cognify API keys"""
        
        try:
            # Build query conditions
            conditions = ["user_id = $1"]
            params = [UUID(user_id)]
            param_count = 1
            
            if organization_id:
                param_count += 1
                conditions.append(f"organization_id = ${param_count}::UUID")
                params.append(UUID(organization_id))
            
            if workspace_id:
                param_count += 1
                conditions.append(f"workspace_id = ${param_count}::UUID")
                params.append(UUID(workspace_id))
            
            if not include_inactive:
                param_count += 1
                conditions.append(f"status = ${param_count}")
                params.append(APIKeyStatus.ACTIVE.value)
            
            query = f"""
                SELECT 
                    id, key_name, key_description, key_prefix, permissions,
                    rate_limits, status, expires_at, last_used_at, usage_count,
                    created_at, updated_at
                FROM cognify_api_keys
                WHERE {' AND '.join(conditions)}
                ORDER BY created_at DESC
            """
            
            rows = await self.db.fetch(query, *params)
            
            return [
                {
                    "id": str(row["id"]),
                    "key_name": row["key_name"],
                    "description": row["key_description"],
                    "key_prefix": row["key_prefix"],
                    "permissions": row["permissions"],
                    "rate_limits": row["rate_limits"],
                    "status": row["status"],
                    "expires_at": row["expires_at"].isoformat() if row["expires_at"] else None,
                    "last_used_at": row["last_used_at"].isoformat() if row["last_used_at"] else None,
                    "usage_count": row["usage_count"],
                    "created_at": row["created_at"].isoformat(),
                    "updated_at": row["updated_at"].isoformat()
                }
                for row in rows
            ]
            
        except Exception as e:
            logger.error(f"Error getting user API keys: {e}")
            raise
    
    async def authenticate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Authenticate API key and return user info"""
        
        try:
            # Check cache first
            cache_key = f"{self.cache_prefix}auth:{api_key}"
            if self.cache:
                cached_result = await self.cache.get(cache_key)
                if cached_result:
                    return cached_result
            
            # Authenticate using database function
            result = await self.db.fetchrow("""
                SELECT * FROM authenticate_cognify_api_key($1)
            """, api_key)
            
            if result:
                auth_data = {
                    "api_key_id": str(result["api_key_id"]),
                    "user_id": str(result["user_id"]),
                    "organization_id": str(result["organization_id"]) if result["organization_id"] else None,
                    "workspace_id": str(result["workspace_id"]) if result["workspace_id"] else None,
                    "permissions": result["permissions"],
                    "rate_limits": result["rate_limits"]
                }
                
                # Cache the result
                if self.cache:
                    await self.cache.set(cache_key, auth_data, ttl=self.cache_ttl)
                
                return auth_data
            
            return None
            
        except Exception as e:
            logger.error(f"Error authenticating API key: {e}")
            return None
    
    async def check_rate_limit(self, api_key_id: str, endpoint: str) -> bool:
        """Check if API key is within rate limits"""
        
        try:
            result = await self.db.fetchval("""
                SELECT check_cognify_api_key_rate_limit($1::UUID, $2)
            """, UUID(api_key_id), endpoint)
            
            return result or False
            
        except Exception as e:
            logger.error(f"Error checking rate limit: {e}")
            return False
    
    async def track_usage(self,
                        api_key_id: str,
                        user_id: str,
                        endpoint: str,
                        method: str,
                        status_code: int,
                        request_id: Optional[str] = None,
                        user_agent: Optional[str] = None,
                        ip_address: Optional[str] = None,
                        response_time_ms: Optional[int] = None,
                        request_size_bytes: Optional[int] = None,
                        response_size_bytes: Optional[int] = None,
                        organization_id: Optional[str] = None,
                        workspace_id: Optional[str] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> str:
        """Track API key usage"""
        
        try:
            usage_id = await self.db.fetchval("""
                SELECT track_cognify_api_key_usage(
                    $1::UUID, $2::UUID, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12::UUID, $13::UUID, $14
                )
            """,
                UUID(api_key_id),
                UUID(user_id),
                endpoint,
                method,
                status_code,
                request_id,
                user_agent,
                ip_address,
                response_time_ms,
                request_size_bytes,
                response_size_bytes,
                UUID(organization_id) if organization_id else None,
                UUID(workspace_id) if workspace_id else None,
                metadata or {}
            )
            
            return str(usage_id)
            
        except Exception as e:
            logger.error(f"Error tracking API key usage: {e}")
            raise
    
    async def update_api_key(self,
                           key_id: str,
                           user_id: str,
                           key_name: Optional[str] = None,
                           description: Optional[str] = None,
                           permissions: Optional[Dict[str, Any]] = None,
                           rate_limits: Optional[Dict[str, Any]] = None,
                           status: Optional[APIKeyStatus] = None,
                           expires_at: Optional[datetime] = None) -> bool:
        """Update an API key"""
        
        try:
            # Build update query
            updates = []
            params = []
            param_count = 0
            
            if key_name is not None:
                param_count += 1
                updates.append(f"key_name = ${param_count}")
                params.append(key_name)
            
            if description is not None:
                param_count += 1
                updates.append(f"key_description = ${param_count}")
                params.append(description)
            
            if permissions is not None:
                param_count += 1
                updates.append(f"permissions = ${param_count}")
                params.append(permissions)
            
            if rate_limits is not None:
                param_count += 1
                updates.append(f"rate_limits = ${param_count}")
                params.append(rate_limits)
            
            if status is not None:
                param_count += 1
                updates.append(f"status = ${param_count}")
                params.append(status.value)
            
            if expires_at is not None:
                param_count += 1
                updates.append(f"expires_at = ${param_count}")
                params.append(expires_at)
            
            if not updates:
                return True  # Nothing to update
            
            # Add WHERE clause parameters
            param_count += 1
            params.append(UUID(key_id))
            param_count += 1
            params.append(UUID(user_id))
            
            query = f"""
                UPDATE cognify_api_keys 
                SET {', '.join(updates)}, updated_at = NOW()
                WHERE id = ${param_count - 1} AND user_id = ${param_count}
            """
            
            result = await self.db.execute(query, *params)
            
            # Clear cache
            await self._clear_cache_for_key(key_id)
            
            return result == "UPDATE 1"
            
        except Exception as e:
            logger.error(f"Error updating API key: {e}")
            return False
    
    async def delete_api_key(self, key_id: str, user_id: str) -> bool:
        """Delete an API key"""
        
        try:
            result = await self.db.execute("""
                DELETE FROM cognify_api_keys 
                WHERE id = $1 AND user_id = $2
            """, UUID(key_id), UUID(user_id))
            
            # Clear cache
            await self._clear_cache_for_key(key_id)
            
            logger.info(f"Deleted Cognify API key {key_id} for user {user_id}")
            return result == "DELETE 1"
            
        except Exception as e:
            logger.error(f"Error deleting API key: {e}")
            return False
    
    async def get_usage_statistics(self,
                                 api_key_id: str,
                                 user_id: str,
                                 days: int = 30) -> Dict[str, Any]:
        """Get API key usage statistics"""
        
        try:
            # Verify key belongs to user
            key_exists = await self.db.fetchval("""
                SELECT EXISTS(SELECT 1 FROM cognify_api_keys WHERE id = $1 AND user_id = $2)
            """, UUID(api_key_id), UUID(user_id))
            
            if not key_exists:
                raise ValueError("API key not found")
            
            # Get usage statistics
            usage_stats = await self.db.fetchrow("""
                SELECT 
                    COUNT(*) as total_requests,
                    COUNT(CASE WHEN status_code >= 200 AND status_code < 300 THEN 1 END) as successful_requests,
                    COUNT(CASE WHEN status_code >= 400 THEN 1 END) as failed_requests,
                    COUNT(CASE WHEN used_at >= CURRENT_DATE THEN 1 END) as requests_today,
                    AVG(response_time_ms) as avg_response_time,
                    MAX(used_at) as last_request_at
                FROM cognify_api_key_usage
                WHERE api_key_id = $1 AND used_at >= NOW() - INTERVAL '%s days'
            """ % days, UUID(api_key_id))
            
            # Get most used endpoints
            top_endpoints = await self.db.fetch("""
                SELECT 
                    endpoint,
                    method,
                    COUNT(*) as request_count,
                    AVG(response_time_ms) as avg_response_time
                FROM cognify_api_key_usage
                WHERE api_key_id = $1 AND used_at >= NOW() - INTERVAL '%s days'
                GROUP BY endpoint, method
                ORDER BY request_count DESC
                LIMIT 10
            """ % days, UUID(api_key_id))
            
            return {
                "total_requests": usage_stats["total_requests"],
                "successful_requests": usage_stats["successful_requests"],
                "failed_requests": usage_stats["failed_requests"],
                "requests_today": usage_stats["requests_today"],
                "success_rate": (usage_stats["successful_requests"] / max(usage_stats["total_requests"], 1)) * 100,
                "avg_response_time_ms": float(usage_stats["avg_response_time"]) if usage_stats["avg_response_time"] else 0,
                "last_request_at": usage_stats["last_request_at"].isoformat() if usage_stats["last_request_at"] else None,
                "top_endpoints": [
                    {
                        "endpoint": row["endpoint"],
                        "method": row["method"],
                        "requests": row["request_count"],
                        "avg_response_time_ms": float(row["avg_response_time"]) if row["avg_response_time"] else 0
                    }
                    for row in top_endpoints
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting usage statistics: {e}")
            raise
    
    async def _clear_cache_for_key(self, key_id: str):
        """Clear cache for specific API key"""
        if self.cache:
            # We can't easily clear by key_id since we cache by actual key value
            # In a real implementation, you might maintain a reverse mapping
            pass

# Global service instance
cognify_api_keys_service = CognifyAPIKeysService()
