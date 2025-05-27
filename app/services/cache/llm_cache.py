"""
LLM response caching service for performance optimization.

Provides intelligent caching of LLM responses to reduce API calls and improve performance.
"""

import asyncio
import hashlib
import json
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

import structlog

from app.core.config import get_settings
# Simple classes to replace LLMMessage and LLMResponse
class LLMMessage:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

class LLMResponse:
    def __init__(self, content: str, usage: dict = None, model: str = "", provider: str = "", metadata: dict = None):
        self.content = content
        self.usage = usage or {}
        self.model = model
        self.provider = provider
        self.metadata = metadata or {}

logger = structlog.get_logger(__name__)
settings = get_settings()


@dataclass
class CacheEntry:
    """Cache entry for LLM responses."""
    response: LLMResponse
    created_at: datetime
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    cache_key: str = ""

    def is_expired(self, ttl_seconds: int) -> bool:
        """Check if cache entry is expired."""
        if ttl_seconds <= 0:
            return False
        return (datetime.now() - self.created_at).total_seconds() > ttl_seconds

    def access(self) -> None:
        """Mark entry as accessed."""
        self.access_count += 1
        self.last_accessed = datetime.now()


class LLMCache:
    """
    High-performance LLM response cache with intelligent eviction.
    """

    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: List[str] = []  # For LRU eviction
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "total_requests": 0,
            "cache_size": 0,
            "memory_saved_tokens": 0
        }

        logger.info("LLM cache initialized", max_size=max_size, default_ttl=default_ttl)

    def _generate_cache_key(self, messages: List[LLMMessage], model: str, **kwargs) -> str:
        """Generate deterministic cache key for LLM request."""
        # Create normalized request representation
        request_data = {
            "messages": [{"role": msg.role, "content": msg.content} for msg in messages],
            "model": model,
            "temperature": kwargs.get("temperature", 0.1),
            "max_tokens": kwargs.get("max_tokens", 4000)
        }

        # Sort for consistency
        request_json = json.dumps(request_data, sort_keys=True)

        # Generate hash
        return hashlib.sha256(request_json.encode()).hexdigest()[:16]

    def _evict_expired(self) -> int:
        """Remove expired entries from cache."""
        expired_keys = []
        current_time = datetime.now()

        for key, entry in self._cache.items():
            if entry.is_expired(self.default_ttl):
                expired_keys.append(key)

        for key in expired_keys:
            self._remove_entry(key)

        if expired_keys:
            logger.debug("Evicted expired cache entries", count=len(expired_keys))

        return len(expired_keys)

    def _evict_lru(self, count: int = 1) -> int:
        """Evict least recently used entries."""
        evicted = 0

        # Sort by last access time (oldest first)
        sorted_keys = sorted(
            self._cache.keys(),
            key=lambda k: self._cache[k].last_accessed or self._cache[k].created_at
        )

        for key in sorted_keys[:count]:
            self._remove_entry(key)
            evicted += 1

        if evicted > 0:
            logger.debug("Evicted LRU cache entries", count=evicted)
            self._stats["evictions"] += evicted

        return evicted

    def _remove_entry(self, key: str) -> None:
        """Remove entry from cache and access order."""
        if key in self._cache:
            del self._cache[key]
        if key in self._access_order:
            self._access_order.remove(key)
        self._stats["cache_size"] = len(self._cache)

    def _ensure_capacity(self) -> None:
        """Ensure cache doesn't exceed max size."""
        # First, remove expired entries
        self._evict_expired()

        # If still over capacity, use LRU eviction
        if len(self._cache) >= self.max_size:
            evict_count = len(self._cache) - self.max_size + 1
            self._evict_lru(evict_count)

    async def get(self, messages: List[LLMMessage], model: str, **kwargs) -> Optional[LLMResponse]:
        """
        Get cached LLM response if available.

        Args:
            messages: LLM messages
            model: Model name
            **kwargs: Additional parameters

        Returns:
            Cached LLMResponse or None if not found
        """
        self._stats["total_requests"] += 1

        cache_key = self._generate_cache_key(messages, model, **kwargs)

        # Check if entry exists and is not expired
        if cache_key in self._cache:
            entry = self._cache[cache_key]

            if not entry.is_expired(self.default_ttl):
                # Cache hit
                entry.access()
                self._update_access_order(cache_key)
                self._stats["hits"] += 1
                self._stats["memory_saved_tokens"] += entry.response.usage.get("total_tokens", 0)

                logger.debug(
                    "Cache hit",
                    cache_key=cache_key,
                    access_count=entry.access_count,
                    age_seconds=(datetime.now() - entry.created_at).total_seconds()
                )

                return entry.response
            else:
                # Expired entry
                self._remove_entry(cache_key)

        # Cache miss
        self._stats["misses"] += 1
        logger.debug("Cache miss", cache_key=cache_key)
        return None

    async def put(self, messages: List[LLMMessage], model: str, response: LLMResponse, **kwargs) -> None:
        """
        Store LLM response in cache.

        Args:
            messages: LLM messages
            model: Model name
            response: LLM response to cache
            **kwargs: Additional parameters
        """
        cache_key = self._generate_cache_key(messages, model, **kwargs)

        # Ensure we have capacity
        self._ensure_capacity()

        # Create cache entry
        entry = CacheEntry(
            response=response,
            created_at=datetime.now(),
            cache_key=cache_key
        )
        entry.access()  # Mark as accessed

        # Store in cache
        self._cache[cache_key] = entry
        self._update_access_order(cache_key)
        self._stats["cache_size"] = len(self._cache)

        logger.debug(
            "Cached LLM response",
            cache_key=cache_key,
            response_tokens=response.usage.get("total_tokens", 0),
            cache_size=len(self._cache)
        )

    def _update_access_order(self, key: str) -> None:
        """Update access order for LRU tracking."""
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

    def clear(self) -> None:
        """Clear all cache entries."""
        cleared_count = len(self._cache)
        self._cache.clear()
        self._access_order.clear()
        self._stats["cache_size"] = 0

        logger.info("Cache cleared", entries_removed=cleared_count)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self._stats["total_requests"]
        hit_rate = (self._stats["hits"] / total_requests * 100) if total_requests > 0 else 0

        return {
            "cache_size": len(self._cache),
            "max_size": self.max_size,
            "hit_rate_percent": round(hit_rate, 2),
            "total_requests": total_requests,
            "cache_hits": self._stats["hits"],
            "cache_misses": self._stats["misses"],
            "evictions": self._stats["evictions"],
            "memory_saved_tokens": self._stats["memory_saved_tokens"],
            "utilization_percent": round(len(self._cache) / self.max_size * 100, 2)
        }

    def get_detailed_stats(self) -> Dict[str, Any]:
        """Get detailed cache statistics."""
        stats = self.get_stats()

        # Add entry-level statistics
        if self._cache:
            access_counts = [entry.access_count for entry in self._cache.values()]
            ages = [(datetime.now() - entry.created_at).total_seconds() for entry in self._cache.values()]

            stats.update({
                "avg_access_count": round(sum(access_counts) / len(access_counts), 2),
                "max_access_count": max(access_counts),
                "avg_age_seconds": round(sum(ages) / len(ages), 2),
                "oldest_entry_seconds": max(ages),
                "newest_entry_seconds": min(ages)
            })

        return stats

    async def cleanup(self) -> Dict[str, int]:
        """Perform cache maintenance and cleanup."""
        initial_size = len(self._cache)

        # Remove expired entries
        expired_count = self._evict_expired()

        # Optionally compact if cache is getting full
        if len(self._cache) > self.max_size * 0.8:
            lru_count = self._evict_lru(int(self.max_size * 0.1))
        else:
            lru_count = 0

        final_size = len(self._cache)

        cleanup_stats = {
            "initial_size": initial_size,
            "final_size": final_size,
            "expired_removed": expired_count,
            "lru_removed": lru_count,
            "total_removed": expired_count + lru_count
        }

        if cleanup_stats["total_removed"] > 0:
            logger.info("Cache cleanup completed", **cleanup_stats)

        return cleanup_stats


# Global cache instance
_llm_cache: Optional[LLMCache] = None


def get_llm_cache() -> LLMCache:
    """Get global LLM cache instance."""
    global _llm_cache
    if _llm_cache is None:
        _llm_cache = LLMCache(
            max_size=settings.CHUNKING_CACHE_TTL,  # Reuse config value
            default_ttl=3600  # 1 hour default TTL
        )
    return _llm_cache


async def cached_llm_generate(
    llm_service,
    messages: List[LLMMessage],
    use_cache: bool = True,
    **kwargs
) -> LLMResponse:
    """
    Generate LLM response with caching.

    Args:
        llm_service: LLM service instance
        messages: Messages to send
        use_cache: Whether to use caching
        **kwargs: Additional generation parameters

    Returns:
        LLM response (cached or fresh)
    """
    if not use_cache:
        return await llm_service.generate(messages, **kwargs)

    cache = get_llm_cache()

    # Get model name from service config
    if hasattr(llm_service, 'config') and hasattr(llm_service.config, 'model'):
        model = llm_service.config.model
    else:
        model = 'unknown'

    # Try to get from cache first
    cached_response = await cache.get(messages, model, **kwargs)
    if cached_response:
        return cached_response

    # Generate fresh response
    response = await llm_service.generate(messages, **kwargs)

    # Cache the response
    await cache.put(messages, model, response, **kwargs)

    return response


# Utility functions for cache management

async def warm_cache_for_common_patterns():
    """Pre-warm cache with common chunking patterns."""
    # This could be implemented to pre-populate cache with common code patterns
    pass


async def get_cache_health() -> Dict[str, Any]:
    """Get cache health information."""
    cache = get_llm_cache()
    stats = cache.get_detailed_stats()

    # Determine health status
    hit_rate = stats.get("hit_rate_percent", 0)
    utilization = stats.get("utilization_percent", 0)

    if hit_rate > 70 and utilization < 90:
        status = "healthy"
    elif hit_rate > 50 and utilization < 95:
        status = "good"
    elif hit_rate > 30:
        status = "fair"
    else:
        status = "poor"

    return {
        "status": status,
        "statistics": stats,
        "recommendations": _get_cache_recommendations(stats)
    }


def _get_cache_recommendations(stats: Dict[str, Any]) -> List[str]:
    """Generate cache optimization recommendations."""
    recommendations = []

    hit_rate = stats.get("hit_rate_percent", 0)
    utilization = stats.get("utilization_percent", 0)

    if hit_rate < 30:
        recommendations.append("Consider increasing cache size or TTL")

    if utilization > 90:
        recommendations.append("Cache is near capacity, consider increasing max_size")

    if stats.get("evictions", 0) > stats.get("cache_hits", 0):
        recommendations.append("High eviction rate, consider optimizing cache parameters")

    if not recommendations:
        recommendations.append("Cache performance is optimal")

    return recommendations
