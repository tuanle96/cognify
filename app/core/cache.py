"""
Cache module for Cognify application.

Provides Redis-based caching functionality with fallback to in-memory cache.
"""

import json
import logging
import asyncio
from typing import Any, Optional, Dict, Union
from datetime import datetime, timedelta

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from app.core.config import settings

logger = logging.getLogger(__name__)


class InMemoryCache:
    """Simple in-memory cache implementation as fallback."""
    
    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        async with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                if entry['expires_at'] is None or entry['expires_at'] > datetime.now():
                    return entry['value']
                else:
                    # Expired, remove it
                    del self._cache[key]
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        async with self._lock:
            expires_at = None
            if ttl:
                expires_at = datetime.now() + timedelta(seconds=ttl)
            
            self._cache[key] = {
                'value': value,
                'expires_at': expires_at
            }
            return True
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    async def clear(self) -> bool:
        """Clear all cache entries."""
        async with self._lock:
            self._cache.clear()
            return True
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        return await self.get(key) is not None


class RedisCache:
    """Redis-based cache implementation."""
    
    def __init__(self, redis_client):
        self.redis = redis_client
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache."""
        try:
            value = await self.redis.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Redis get error for key {key}: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Redis cache."""
        try:
            serialized_value = json.dumps(value, default=str)
            if ttl:
                await self.redis.setex(key, ttl, serialized_value)
            else:
                await self.redis.set(key, serialized_value)
            return True
        except Exception as e:
            logger.error(f"Redis set error for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from Redis cache."""
        try:
            result = await self.redis.delete(key)
            return result > 0
        except Exception as e:
            logger.error(f"Redis delete error for key {key}: {e}")
            return False
    
    async def clear(self) -> bool:
        """Clear all cache entries."""
        try:
            await self.redis.flushdb()
            return True
        except Exception as e:
            logger.error(f"Redis clear error: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis cache."""
        try:
            result = await self.redis.exists(key)
            return result > 0
        except Exception as e:
            logger.error(f"Redis exists error for key {key}: {e}")
            return False


class CacheManager:
    """Cache manager that handles Redis connection and fallback."""
    
    def __init__(self):
        self._cache = None
        self._redis_client = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize cache connection."""
        if self._initialized:
            return
        
        if REDIS_AVAILABLE:
            try:
                # Try to connect to Redis
                redis_url = getattr(settings, 'REDIS_URL', 'redis://localhost:6379/0')
                self._redis_client = redis.from_url(redis_url, decode_responses=True)
                
                # Test connection
                await self._redis_client.ping()
                self._cache = RedisCache(self._redis_client)
                logger.info("Redis cache initialized successfully")
                
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}. Using in-memory cache.")
                self._cache = InMemoryCache()
        else:
            logger.warning("Redis not available. Using in-memory cache.")
            self._cache = InMemoryCache()
        
        self._initialized = True
    
    async def get_cache(self):
        """Get cache instance."""
        if not self._initialized:
            await self.initialize()
        return self._cache
    
    async def close(self) -> None:
        """Close cache connections."""
        if self._redis_client:
            await self._redis_client.close()
        self._initialized = False


# Global cache manager instance
cache_manager = CacheManager()


async def get_cache():
    """Get cache instance."""
    return await cache_manager.get_cache()


async def initialize_cache():
    """Initialize cache system."""
    await cache_manager.initialize()


async def close_cache():
    """Close cache connections."""
    await cache_manager.close()
