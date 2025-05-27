"""
Main vector database service with multi-provider support and intelligent management.
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Union
import logging
from dataclasses import dataclass

from .base import (
    VectorDBClient, VectorPoint, SearchResult, SearchRequest, CollectionInfo,
    VectorDBProvider, VectorDBError
)
from .factory import vectordb_factory

logger = logging.getLogger(__name__)

@dataclass
class VectorDBServiceConfig:
    """Configuration for the vector database service."""
    provider: VectorDBProvider = VectorDBProvider.QDRANT
    host: str = "localhost"
    port: int = 6333
    auto_create_collections: bool = True
    default_dimension: int = 1536
    default_distance_metric: str = "cosine"
    connection_pool_size: int = 10
    retry_attempts: int = 3
    retry_delay: float = 1.0

class VectorDBService:
    """
    Main vector database service with multi-provider support and management features.
    
    Features:
    - Multi-provider support (Qdrant, Milvus)
    - Collection management with auto-creation
    - Batch operations with optimization
    - Connection pooling and retry logic
    - Performance monitoring and metrics
    """
    
    def __init__(self, config: Optional[VectorDBServiceConfig] = None):
        self.config = config or VectorDBServiceConfig()
        self.client: Optional[VectorDBClient] = None
        self._initialized = False
        self._stats = {
            "operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "collections_created": 0,
            "points_inserted": 0,
            "searches_performed": 0,
            "total_processing_time": 0.0
        }
    
    async def initialize(self) -> None:
        """Initialize the vector database service."""
        try:
            # Create client using factory
            self.client = vectordb_factory.create_client(
                provider=self.config.provider,
                host=self.config.host,
                port=self.config.port
            )
            
            # Initialize client connection
            await self.client.initialize()
            
            self._initialized = True
            logger.info(f"Vector database service initialized with {self.config.provider.value}")
            
        except Exception as e:
            raise VectorDBError(f"Failed to initialize vector database service: {e}") from e
    
    async def create_collection(
        self,
        name: str,
        dimension: Optional[int] = None,
        distance_metric: Optional[str] = None,
        **kwargs
    ) -> bool:
        """
        Create a new collection.
        
        Args:
            name: Collection name
            dimension: Vector dimension (uses default if None)
            distance_metric: Distance metric (uses default if None)
            **kwargs: Additional collection parameters
        
        Returns:
            True if collection was created successfully
        """
        if not self._initialized:
            await self.initialize()
        
        dimension = dimension or self.config.default_dimension
        distance_metric = distance_metric or self.config.default_distance_metric
        
        start_time = time.time()
        
        try:
            result = await self.client.create_collection(
                name=name,
                dimension=dimension,
                distance_metric=distance_metric,
                **kwargs
            )
            
            self._stats["collections_created"] += 1
            self._update_operation_stats(True, time.time() - start_time)
            
            logger.info(f"Created collection: {name} (dim={dimension})")
            return result
            
        except Exception as e:
            self._update_operation_stats(False, time.time() - start_time)
            logger.error(f"Failed to create collection {name}: {e}")
            raise
    
    async def ensure_collection(
        self,
        name: str,
        dimension: Optional[int] = None,
        distance_metric: Optional[str] = None,
        **kwargs
    ) -> bool:
        """
        Ensure collection exists, create if it doesn't.
        
        Args:
            name: Collection name
            dimension: Vector dimension
            distance_metric: Distance metric
            **kwargs: Additional collection parameters
        
        Returns:
            True if collection exists or was created
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Check if collection exists
            if await self.client.collection_exists(name):
                logger.debug(f"Collection {name} already exists")
                return True
            
            # Create collection if auto-creation is enabled
            if self.config.auto_create_collections:
                return await self.create_collection(name, dimension, distance_metric, **kwargs)
            else:
                logger.warning(f"Collection {name} does not exist and auto-creation is disabled")
                return False
                
        except Exception as e:
            logger.error(f"Failed to ensure collection {name}: {e}")
            raise
    
    async def insert_points(
        self,
        collection_name: str,
        points: List[VectorPoint],
        ensure_collection: bool = True
    ) -> bool:
        """
        Insert points into a collection.
        
        Args:
            collection_name: Name of the collection
            points: List of points to insert
            ensure_collection: Whether to create collection if it doesn't exist
        
        Returns:
            True if points were inserted successfully
        """
        if not self._initialized:
            await self.initialize()
        
        if not points:
            return True
        
        start_time = time.time()
        
        try:
            # Ensure collection exists
            if ensure_collection:
                # Infer dimension from first point
                dimension = len(points[0].vector) if points else self.config.default_dimension
                await self.ensure_collection(collection_name, dimension=dimension)
            
            # Insert points
            result = await self.client.insert_points(collection_name, points)
            
            self._stats["points_inserted"] += len(points)
            self._update_operation_stats(True, time.time() - start_time)
            
            logger.debug(f"Inserted {len(points)} points into {collection_name}")
            return result
            
        except Exception as e:
            self._update_operation_stats(False, time.time() - start_time)
            logger.error(f"Failed to insert points into {collection_name}: {e}")
            raise
    
    async def insert_batch(
        self,
        collection_name: str,
        points: List[VectorPoint],
        batch_size: Optional[int] = None
    ) -> bool:
        """
        Insert points in batches for better performance.
        
        Args:
            collection_name: Name of the collection
            points: List of points to insert
            batch_size: Batch size (uses client max if None)
        
        Returns:
            True if all batches were inserted successfully
        """
        if not points:
            return True
        
        # Determine batch size
        if batch_size is None:
            batch_size = self.client.max_batch_size if self.client else 1000
        
        # Process in batches
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            await self.insert_points(collection_name, batch, ensure_collection=(i == 0))
        
        logger.info(f"Inserted {len(points)} points in batches of {batch_size}")
        return True
    
    async def search(
        self,
        collection_name: str,
        vector: List[float],
        limit: int = 10,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[SearchResult]:
        """
        Search for similar vectors.
        
        Args:
            collection_name: Name of the collection to search
            vector: Query vector
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            filter_conditions: Metadata filter conditions
            **kwargs: Additional search parameters
        
        Returns:
            List of search results
        """
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # Create search request
            request = SearchRequest(
                vector=vector,
                limit=limit,
                score_threshold=score_threshold,
                filter_conditions=filter_conditions,
                **kwargs
            )
            
            # Perform search
            results = await self.client.search(collection_name, request)
            
            self._stats["searches_performed"] += 1
            self._update_operation_stats(True, time.time() - start_time)
            
            logger.debug(f"Search in {collection_name} returned {len(results)} results")
            return results
            
        except Exception as e:
            self._update_operation_stats(False, time.time() - start_time)
            logger.error(f"Search failed in {collection_name}: {e}")
            raise
    
    async def get_collection_info(self, name: str) -> CollectionInfo:
        """Get information about a collection."""
        if not self._initialized:
            await self.initialize()
        
        return await self.client.get_collection_info(name)
    
    async def list_collections(self) -> List[str]:
        """List all collections."""
        if not self._initialized:
            await self.initialize()
        
        return await self.client.list_collections()
    
    async def delete_collection(self, name: str) -> bool:
        """Delete a collection."""
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            result = await self.client.delete_collection(name)
            self._update_operation_stats(True, time.time() - start_time)
            logger.info(f"Deleted collection: {name}")
            return result
        except Exception as e:
            self._update_operation_stats(False, time.time() - start_time)
            logger.error(f"Failed to delete collection {name}: {e}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Check the health of the vector database service."""
        if not self._initialized:
            try:
                await self.initialize()
            except Exception as e:
                return {
                    "status": "unhealthy",
                    "error": f"Initialization failed: {e}",
                    "stats": self.get_stats()
                }
        
        try:
            # Get client health
            client_health = await self.client.health_check()
            
            # Get service stats
            stats = self.get_stats()
            
            return {
                "status": client_health.get("status", "unknown"),
                "provider": self.config.provider.value,
                "client_health": client_health,
                "service_stats": stats,
                "config": {
                    "host": self.config.host,
                    "port": self.config.port,
                    "auto_create_collections": self.config.auto_create_collections
                }
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "stats": self.get_stats()
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        stats = self._stats.copy()
        
        # Calculate derived metrics
        if stats["operations"] > 0:
            stats["success_rate"] = stats["successful_operations"] / stats["operations"]
            stats["avg_processing_time"] = stats["total_processing_time"] / stats["operations"]
        else:
            stats["success_rate"] = 0.0
            stats["avg_processing_time"] = 0.0
        
        return stats
    
    async def cleanup(self) -> None:
        """Cleanup all resources."""
        if self.client:
            try:
                await self.client.cleanup()
            except Exception as e:
                logger.warning(f"Error cleaning up vector database client: {e}")
        
        self.client = None
        self._initialized = False
        logger.info("Vector database service cleaned up")
    
    # Private methods
    
    def _update_operation_stats(self, success: bool, processing_time: float) -> None:
        """Update operation statistics."""
        self._stats["operations"] += 1
        self._stats["total_processing_time"] += processing_time
        
        if success:
            self._stats["successful_operations"] += 1
        else:
            self._stats["failed_operations"] += 1

# Global service instance
vectordb_service = VectorDBService()
