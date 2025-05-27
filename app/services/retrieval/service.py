"""
Main retrieval service implementing intelligent document retrieval.
"""

import asyncio
import time
import hashlib
from typing import List, Dict, Any, Optional
import logging

from .base import (
    RetrievalRequest, RetrievalResponse, RetrievalResult, QueryType,
    RetrievalStrategy, RetrievalConfig, RetrievalError, SearchError
)
from .query_processor import query_processor
from .reranker import reranker
# from ..embedding.service import embedding_service  # Commented out to avoid early import
from ..vectordb.service import vectordb_service
from ..vectordb.base import SearchRequest as VectorSearchRequest

logger = logging.getLogger(__name__)

class RetrievalService:
    """
    Main retrieval service implementing intelligent document retrieval.

    Features:
    - Vector similarity search
    - Keyword search (basic implementation)
    - Hybrid retrieval with fusion
    - Query processing and understanding
    - Result re-ranking
    - Caching for performance
    """

    def __init__(self, config: Optional[RetrievalConfig] = None):
        self.config = config or RetrievalConfig()
        self._initialized = False
        self._cache: Dict[str, RetrievalResponse] = {}
        self._stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "vector_searches": 0,
            "keyword_searches": 0,
            "hybrid_searches": 0,
            "total_processing_time": 0.0,
            "avg_results_per_query": 0.0
        }

    async def initialize(self) -> None:
        """Initialize the retrieval service and all dependencies."""
        try:
            # Import embedding service locally to avoid early import
            from ..embedding.service import embedding_service

            # Initialize required services
            services_to_init = [
                ("embedding", embedding_service),
                ("vectordb", vectordb_service),
                ("query_processor", query_processor),
                ("reranker", reranker)
            ]

            for service_name, service in services_to_init:
                if not getattr(service, '_initialized', False):
                    logger.info(f"Initializing {service_name} service...")
                    await service.initialize()

            self._initialized = True
            logger.info("Retrieval service initialized")

        except Exception as e:
            raise RetrievalError(f"Failed to initialize retrieval service: {e}") from e

    async def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        """
        Retrieve relevant documents for a query.

        Args:
            request: Retrieval request with query and configuration

        Returns:
            Retrieval response with ranked results
        """
        if not self._initialized:
            await self.initialize()

        start_time = time.time()
        self._stats["total_queries"] += 1

        try:
            # Use request config or service default
            config = request.config or self.config

            # Check cache first
            cache_key = None
            if config.enable_caching:
                cache_key = self._generate_cache_key(request)
                if cache_key in self._cache:
                    cached_response = self._cache[cache_key]
                    cached_response.from_cache = True
                    cached_response.cache_key = cache_key
                    self._stats["cache_hits"] += 1
                    logger.debug(f"Cache hit for query: {request.query[:50]}...")
                    return cached_response
                else:
                    self._stats["cache_misses"] += 1

            # Process query
            query_info = await query_processor.process_query(
                query=request.query,
                query_type=request.query_type,
                context=request.context,
                expand_query=config.enable_query_expansion
            )

            # Determine retrieval strategy
            strategy = self._determine_strategy(request.strategy, query_info)

            # Perform retrieval based on strategy
            if strategy == RetrievalStrategy.VECTOR_ONLY:
                results = await self._vector_search(query_info, config)
                self._stats["vector_searches"] += 1
            elif strategy == RetrievalStrategy.KEYWORD_ONLY:
                results = await self._keyword_search(query_info, config)
                self._stats["keyword_searches"] += 1
            elif strategy == RetrievalStrategy.HYBRID:
                results = await self._hybrid_search(query_info, config)
                self._stats["hybrid_searches"] += 1
            else:  # ADAPTIVE
                results = await self._adaptive_search(query_info, config)
                self._stats["hybrid_searches"] += 1

            # Re-rank results if enabled
            rerank_time = 0.0
            if config.enable_reranking and results:
                rerank_start = time.time()
                results = await reranker.rerank_results(
                    results=results[:config.rerank_top_k],
                    query=request.query,
                    query_type=query_info["query_type"],
                    top_k=config.final_top_k
                )
                rerank_time = time.time() - rerank_start

            # Apply final filtering
            results = self._apply_final_filters(results, config)

            # Create response
            processing_time = time.time() - start_time

            response = RetrievalResponse(
                results=results,
                original_query=request.query,
                processed_query=query_info.get("processed_query"),
                query_type=query_info["query_type"],
                strategy_used=strategy,
                total_results=len(results),
                processing_time=processing_time,
                rerank_time=rerank_time,
                collection_searched=config.collection_name,
                filters_applied=config.metadata_filters,
                success=True,
                from_cache=False,
                cache_key=cache_key
            )

            # Cache successful response
            if config.enable_caching and cache_key:
                self._cache[cache_key] = response

            # Update stats
            self._stats["successful_queries"] += 1
            self._stats["total_processing_time"] += processing_time
            self._update_avg_results(len(results))

            logger.debug(f"Retrieved {len(results)} results for query: {request.query[:50]}...")
            return response

        except Exception as e:
            # Update stats on error
            self._stats["failed_queries"] += 1
            processing_time = time.time() - start_time

            logger.error(f"Retrieval failed for query '{request.query}': {e}")

            # Create error response
            return RetrievalResponse(
                results=[],
                original_query=request.query,
                query_type=request.query_type,
                strategy_used=request.strategy,
                processing_time=processing_time,
                success=False,
                errors=[str(e)]
            )

    async def _vector_search(
        self,
        query_info: Dict[str, Any],
        config: RetrievalConfig
    ) -> List[RetrievalResult]:
        """Perform vector similarity search."""
        try:
            # Import embedding service locally to avoid early import
            from ..embedding.service import embedding_service

            # Generate query embedding
            query_embedding = await embedding_service.embed_single(
                text=query_info["processed_query"],
                embedding_type="text"
            )

            # Prepare search request
            search_request = VectorSearchRequest(
                vector=query_embedding,
                limit=config.max_results,
                score_threshold=config.min_score,
                filter_conditions=config.metadata_filters,
                include_metadata=True,
                include_vectors=False
            )

            # Perform vector search
            vector_results = await vectordb_service.search(
                collection_name=config.collection_name,
                request=search_request
            )

            # Convert to retrieval results
            results = []
            for vector_result in vector_results:
                result = RetrievalResult(
                    document_id=vector_result.metadata.get("document_id", "unknown"),
                    chunk_id=vector_result.id,
                    content=vector_result.metadata.get("content", ""),
                    score=vector_result.score,
                    vector_score=vector_result.score,
                    metadata=vector_result.metadata,
                    source_file=vector_result.metadata.get("source_file"),
                    chunk_index=vector_result.metadata.get("chunk_index"),
                    language=vector_result.metadata.get("language"),
                    document_type=vector_result.metadata.get("document_type")
                )
                results.append(result)

            return results

        except Exception as e:
            raise SearchError(f"Vector search failed: {e}", "vector", e)

    async def _keyword_search(
        self,
        query_info: Dict[str, Any],
        config: RetrievalConfig
    ) -> List[RetrievalResult]:
        """Perform keyword-based search."""
        # This is a simplified keyword search implementation
        # In a production system, you would use a proper text search engine like Elasticsearch

        try:
            # For now, we'll use vector search with keyword-optimized query
            # and apply keyword-based scoring

            keywords = query_info.get("keywords", [])
            if not keywords:
                return []

            # Use keyword query for embedding
            keyword_query = query_info.get("keyword_query", query_info["cleaned_query"])

            # Import embedding service locally to avoid early import
            from ..embedding.service import embedding_service

            # Generate embedding for keyword query
            query_embedding = await embedding_service.embed_single(
                text=keyword_query,
                embedding_type="text"
            )

            # Search with relaxed threshold for keyword search
            search_request = VectorSearchRequest(
                vector=query_embedding,
                limit=config.max_results * 2,  # Get more results for keyword filtering
                score_threshold=max(0.1, config.min_score - 0.2),  # Lower threshold
                filter_conditions=config.metadata_filters,
                include_metadata=True,
                include_vectors=False
            )

            # Perform search
            vector_results = await vectordb_service.search(
                collection_name=config.collection_name,
                request=search_request
            )

            # Apply keyword scoring
            results = []
            for vector_result in vector_results:
                content = vector_result.metadata.get("content", "")

                # Calculate keyword match score
                keyword_score = self._calculate_keyword_score(content, keywords)

                # Skip results with very low keyword match
                if keyword_score < 0.1:
                    continue

                result = RetrievalResult(
                    document_id=vector_result.metadata.get("document_id", "unknown"),
                    chunk_id=vector_result.id,
                    content=content,
                    score=keyword_score,
                    keyword_score=keyword_score,
                    vector_score=vector_result.score,
                    metadata=vector_result.metadata,
                    source_file=vector_result.metadata.get("source_file"),
                    chunk_index=vector_result.metadata.get("chunk_index"),
                    language=vector_result.metadata.get("language"),
                    document_type=vector_result.metadata.get("document_type"),
                    matched_keywords=self._find_matched_keywords(content, keywords)
                )
                results.append(result)

            # Sort by keyword score and limit results
            results.sort(key=lambda x: x.keyword_score, reverse=True)
            return results[:config.max_results]

        except Exception as e:
            raise SearchError(f"Keyword search failed: {e}", "keyword", e)

    async def _hybrid_search(
        self,
        query_info: Dict[str, Any],
        config: RetrievalConfig
    ) -> List[RetrievalResult]:
        """Perform hybrid search combining vector and keyword search."""
        try:
            # Perform both searches in parallel
            vector_task = self._vector_search(query_info, config)
            keyword_task = self._keyword_search(query_info, config)

            vector_results, keyword_results = await asyncio.gather(
                vector_task, keyword_task, return_exceptions=True
            )

            # Handle exceptions
            if isinstance(vector_results, Exception):
                logger.warning(f"Vector search failed in hybrid: {vector_results}")
                vector_results = []

            if isinstance(keyword_results, Exception):
                logger.warning(f"Keyword search failed in hybrid: {keyword_results}")
                keyword_results = []

            # Fuse results
            fused_results = self._fuse_results(
                vector_results=vector_results,
                keyword_results=keyword_results,
                vector_weight=config.vector_weight,
                keyword_weight=config.keyword_weight,
                fusion_method=config.fusion_method
            )

            return fused_results[:config.max_results]

        except Exception as e:
            raise SearchError(f"Hybrid search failed: {e}", "hybrid", e)

    async def _adaptive_search(
        self,
        query_info: Dict[str, Any],
        config: RetrievalConfig
    ) -> List[RetrievalResult]:
        """Perform adaptive search that chooses the best strategy."""
        query_type = query_info["query_type"]
        complexity = query_info.get("complexity", "medium")

        # Choose strategy based on query characteristics
        if query_type == QueryType.CODE:
            # Code queries benefit from hybrid search
            return await self._hybrid_search(query_info, config)
        elif query_type == QueryType.KEYWORD:
            # Keyword queries prefer keyword search
            return await self._keyword_search(query_info, config)
        elif complexity == "simple":
            # Simple queries can use vector search
            return await self._vector_search(query_info, config)
        else:
            # Default to hybrid for complex queries
            return await self._hybrid_search(query_info, config)

    def _determine_strategy(
        self,
        requested_strategy: RetrievalStrategy,
        query_info: Dict[str, Any]
    ) -> RetrievalStrategy:
        """Determine the actual retrieval strategy to use."""
        if requested_strategy != RetrievalStrategy.ADAPTIVE:
            return requested_strategy

        # Adaptive strategy selection
        query_type = query_info["query_type"]

        if query_type == QueryType.CODE:
            return RetrievalStrategy.HYBRID
        elif query_type == QueryType.KEYWORD:
            return RetrievalStrategy.KEYWORD_ONLY
        else:
            return RetrievalStrategy.HYBRID

    def _fuse_results(
        self,
        vector_results: List[RetrievalResult],
        keyword_results: List[RetrievalResult],
        vector_weight: float,
        keyword_weight: float,
        fusion_method: str
    ) -> List[RetrievalResult]:
        """Fuse vector and keyword search results."""
        if fusion_method == "rrf":  # Reciprocal Rank Fusion
            return self._reciprocal_rank_fusion(vector_results, keyword_results)
        elif fusion_method == "weighted":
            return self._weighted_fusion(vector_results, keyword_results, vector_weight, keyword_weight)
        else:
            # Default to simple combination
            return self._simple_fusion(vector_results, keyword_results)

    def _reciprocal_rank_fusion(
        self,
        vector_results: List[RetrievalResult],
        keyword_results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Implement Reciprocal Rank Fusion (RRF)."""
        k = 60  # RRF parameter

        # Create mapping of chunk_id to combined score
        scores = {}
        result_map = {}

        # Add vector results
        for rank, result in enumerate(vector_results):
            chunk_id = result.chunk_id
            rrf_score = 1 / (k + rank + 1)
            scores[chunk_id] = scores.get(chunk_id, 0) + rrf_score
            result_map[chunk_id] = result

        # Add keyword results
        for rank, result in enumerate(keyword_results):
            chunk_id = result.chunk_id
            rrf_score = 1 / (k + rank + 1)
            scores[chunk_id] = scores.get(chunk_id, 0) + rrf_score

            # Update result if not already present
            if chunk_id not in result_map:
                result_map[chunk_id] = result
            else:
                # Combine scores in existing result
                existing = result_map[chunk_id]
                existing.keyword_score = result.keyword_score
                existing.matched_keywords.extend(result.matched_keywords)

        # Create fused results
        fused_results = []
        for chunk_id, combined_score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            result = result_map[chunk_id]
            result.score = combined_score
            fused_results.append(result)

        return fused_results

    def _weighted_fusion(
        self,
        vector_results: List[RetrievalResult],
        keyword_results: List[RetrievalResult],
        vector_weight: float,
        keyword_weight: float
    ) -> List[RetrievalResult]:
        """Implement weighted score fusion."""
        result_map = {}

        # Add vector results
        for result in vector_results:
            chunk_id = result.chunk_id
            weighted_score = (result.vector_score or result.score) * vector_weight
            result.score = weighted_score
            result_map[chunk_id] = result

        # Add/update with keyword results
        for result in keyword_results:
            chunk_id = result.chunk_id
            keyword_weighted = (result.keyword_score or result.score) * keyword_weight

            if chunk_id in result_map:
                # Combine scores
                existing = result_map[chunk_id]
                existing.score += keyword_weighted
                existing.keyword_score = result.keyword_score
                existing.matched_keywords.extend(result.matched_keywords)
            else:
                # New result from keyword search only
                result.score = keyword_weighted
                result_map[chunk_id] = result

        # Sort by combined score
        fused_results = list(result_map.values())
        fused_results.sort(key=lambda x: x.score, reverse=True)

        return fused_results

    def _simple_fusion(
        self,
        vector_results: List[RetrievalResult],
        keyword_results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Simple fusion by combining and deduplicating results."""
        result_map = {}

        # Add all results, preferring vector results for duplicates
        for result in vector_results + keyword_results:
            chunk_id = result.chunk_id
            if chunk_id not in result_map:
                result_map[chunk_id] = result

        # Sort by original scores
        fused_results = list(result_map.values())
        fused_results.sort(key=lambda x: x.score, reverse=True)

        return fused_results

    def _calculate_keyword_score(self, content: str, keywords: List[str]) -> float:
        """Calculate keyword match score for content."""
        if not keywords:
            return 0.0

        content_lower = content.lower()
        matches = 0
        total_weight = 0

        for keyword in keywords:
            keyword_lower = keyword.lower()

            # Exact match
            if keyword_lower in content_lower:
                matches += 1
                total_weight += 1

            # Partial match (substring)
            elif any(keyword_lower in word for word in content_lower.split()):
                matches += 0.5
                total_weight += 1

        return matches / len(keywords) if keywords else 0.0

    def _find_matched_keywords(self, content: str, keywords: List[str]) -> List[str]:
        """Find which keywords match in the content."""
        content_lower = content.lower()
        matched = []

        for keyword in keywords:
            if keyword.lower() in content_lower:
                matched.append(keyword)

        return matched

    def _apply_final_filters(
        self,
        results: List[RetrievalResult],
        config: RetrievalConfig
    ) -> List[RetrievalResult]:
        """Apply final filtering to results."""
        filtered_results = []

        for result in results:
            # Apply score threshold
            if result.score < config.min_score:
                continue

            # Apply language filter
            if config.language_filter and result.language != config.language_filter:
                continue

            # Apply document type filter
            if config.document_type_filter and result.document_type != config.document_type_filter:
                continue

            filtered_results.append(result)

        return filtered_results[:config.max_results]

    def _generate_cache_key(self, request: RetrievalRequest) -> str:
        """Generate cache key for the request."""
        # Create a string representation of the request
        key_parts = [
            request.query,
            request.query_type.value,
            request.strategy.value,
            str(request.config.max_results if request.config else self.config.max_results),
            str(request.config.min_score if request.config else self.config.min_score),
            str(request.config.collection_name if request.config else self.config.collection_name)
        ]

        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _update_avg_results(self, result_count: int) -> None:
        """Update average results per query."""
        total_queries = self._stats["successful_queries"]
        if total_queries > 0:
            current_avg = self._stats["avg_results_per_query"]
            new_avg = ((current_avg * (total_queries - 1)) + result_count) / total_queries
            self._stats["avg_results_per_query"] = new_avg

    def get_stats(self) -> Dict[str, Any]:
        """Get retrieval service statistics."""
        stats = self._stats.copy()

        # Calculate derived metrics
        if stats["total_queries"] > 0:
            stats["success_rate"] = stats["successful_queries"] / stats["total_queries"]
            stats["cache_hit_rate"] = stats["cache_hits"] / stats["total_queries"]
            stats["avg_processing_time"] = stats["total_processing_time"] / stats["total_queries"]
        else:
            stats["success_rate"] = 0.0
            stats["cache_hit_rate"] = 0.0
            stats["avg_processing_time"] = 0.0

        stats["cache_size"] = len(self._cache)

        return stats

    async def health_check(self) -> Dict[str, Any]:
        """Check the health of the retrieval service."""
        if not self._initialized:
            try:
                await self.initialize()
            except Exception as e:
                return {
                    "status": "unhealthy",
                    "error": f"Initialization failed: {e}",
                    "stats": self.get_stats()
                }

        health_status = {
            "status": "healthy",
            "services": {},
            "stats": self.get_stats(),
            "config": {
                "collection_name": self.config.collection_name,
                "max_results": self.config.max_results,
                "enable_reranking": self.config.enable_reranking
            }
        }

        # Import embedding service locally to avoid early import
        from ..embedding.service import embedding_service

        # Check dependent services
        services = [
            ("embedding", embedding_service),
            ("vectordb", vectordb_service),
            ("query_processor", query_processor),
            ("reranker", reranker)
        ]

        for service_name, service in services:
            try:
                if hasattr(service, 'health_check'):
                    service_health = await service.health_check()
                    health_status["services"][service_name] = service_health
                else:
                    health_status["services"][service_name] = {
                        "status": "healthy" if getattr(service, '_initialized', False) else "unknown",
                        "initialized": getattr(service, '_initialized', False)
                    }
            except Exception as e:
                health_status["services"][service_name] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health_status["status"] = "degraded"

        return health_status

    async def cleanup(self) -> None:
        """Cleanup all resources."""
        self._cache.clear()
        self._initialized = False
        logger.info("Retrieval service cleaned up")

# Global service instance
retrieval_service = RetrievalService()
