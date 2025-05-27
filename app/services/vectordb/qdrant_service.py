"""
Qdrant vector database service implementation.
"""

import asyncio
from typing import List, Dict, Any, Optional
import structlog

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import ResponseHandlingException

from .base import VectorDBService, VectorPoint, SearchResult, VectorDBConfig
from app.core.config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()


class QdrantService(VectorDBService):
    """
    Qdrant vector database service implementation.
    
    Provides high-performance vector storage and similarity search
    using Qdrant vector database.
    """
    
    def __init__(self, config: VectorDBConfig = None):
        super().__init__(config)
        self.client: Optional[QdrantClient] = None
        self._connection_params = self._get_connection_params()
    
    async def initialize(self) -> None:
        """Initialize Qdrant client and connection."""
        if self._initialized:
            return
        
        try:
            # Create Qdrant client
            self.client = QdrantClient(**self._connection_params)
            
            # Test connection
            info = self.client.get_collections()
            logger.info(
                "Qdrant service initialized",
                collections_count=len(info.collections),
                host=self._connection_params.get("host", "localhost"),
                port=self._connection_params.get("port", 6333)
            )
            
            self._initialized = True
            
        except Exception as e:
            logger.error("Failed to initialize Qdrant service", error=str(e))
            raise
    
    async def create_collection(self, collection_name: str, vector_dimension: int, distance_metric: str = "cosine") -> bool:
        """Create a new collection in Qdrant."""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Map distance metric
            distance_map = {
                "cosine": models.Distance.COSINE,
                "euclidean": models.Distance.EUCLID,
                "dot": models.Distance.DOT,
                "manhattan": models.Distance.MANHATTAN
            }
            
            distance = distance_map.get(distance_metric.lower(), models.Distance.COSINE)
            
            # Create collection
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_dimension,
                    distance=distance
                ),
                optimizers_config=models.OptimizersConfig(
                    default_segment_number=2,
                    max_segment_size=20000,
                    memmap_threshold=20000,
                    indexing_threshold=20000,
                    flush_interval_sec=5,
                    max_optimization_threads=1
                ),
                hnsw_config=models.HnswConfig(
                    m=16,
                    ef_construct=100,
                    full_scan_threshold=10000,
                    max_indexing_threads=0,
                    on_disk=False
                )
            )
            
            logger.info(
                "Qdrant collection created",
                collection=collection_name,
                dimension=vector_dimension,
                distance=distance_metric
            )
            
            return True
            
        except ResponseHandlingException as e:
            if "already exists" in str(e).lower():
                logger.warning("Collection already exists", collection=collection_name)
                return True
            logger.error("Failed to create Qdrant collection", error=str(e), collection=collection_name)
            return False
        except Exception as e:
            logger.error("Failed to create Qdrant collection", error=str(e), collection=collection_name)
            return False
    
    async def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection from Qdrant."""
        if not self._initialized:
            await self.initialize()
        
        try:
            self.client.delete_collection(collection_name)
            logger.info("Qdrant collection deleted", collection=collection_name)
            return True
            
        except Exception as e:
            logger.error("Failed to delete Qdrant collection", error=str(e), collection=collection_name)
            return False
    
    async def insert_points(self, collection_name: str, points: List[VectorPoint]) -> bool:
        """Insert vector points into Qdrant collection."""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Convert to Qdrant points
            qdrant_points = []
            for point in points:
                qdrant_point = models.PointStruct(
                    id=point.id,
                    vector=point.vector,
                    payload=point.metadata or {}
                )
                qdrant_points.append(qdrant_point)
            
            # Insert points
            operation_info = self.client.upsert(
                collection_name=collection_name,
                points=qdrant_points,
                wait=True
            )
            
            logger.info(
                "Points inserted into Qdrant",
                collection=collection_name,
                points_count=len(points),
                operation_id=operation_info.operation_id if operation_info else None
            )
            
            return True
            
        except Exception as e:
            logger.error(
                "Failed to insert points into Qdrant",
                error=str(e),
                collection=collection_name,
                points_count=len(points)
            )
            return False
    
    async def search(
        self,
        collection_name: str,
        vector: List[float],
        limit: int = 10,
        score_threshold: float = 0.0,
        filters: Dict[str, Any] = None
    ) -> List[SearchResult]:
        """Search for similar vectors in Qdrant collection."""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Build filter conditions
            filter_conditions = None
            if filters:
                filter_conditions = self._build_filter_conditions(filters)
            
            # Perform search
            search_results = self.client.search(
                collection_name=collection_name,
                query_vector=vector,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=filter_conditions,
                with_payload=True,
                with_vectors=False
            )
            
            # Convert to SearchResult objects
            results = []
            for result in search_results:
                search_result = SearchResult(
                    id=str(result.id),
                    score=float(result.score),
                    metadata=result.payload or {}
                )
                results.append(search_result)
            
            logger.debug(
                "Qdrant search completed",
                collection=collection_name,
                results_count=len(results),
                limit=limit,
                score_threshold=score_threshold
            )
            
            return results
            
        except Exception as e:
            logger.error(
                "Failed to search Qdrant collection",
                error=str(e),
                collection=collection_name,
                limit=limit
            )
            return []
    
    async def delete_points(self, collection_name: str, point_ids: List[str]) -> bool:
        """Delete specific points from Qdrant collection."""
        if not self._initialized:
            await self.initialize()
        
        try:
            operation_info = self.client.delete(
                collection_name=collection_name,
                points_selector=models.PointIdsList(
                    points=point_ids
                ),
                wait=True
            )
            
            logger.info(
                "Points deleted from Qdrant",
                collection=collection_name,
                points_count=len(point_ids),
                operation_id=operation_info.operation_id if operation_info else None
            )
            
            return True
            
        except Exception as e:
            logger.error(
                "Failed to delete points from Qdrant",
                error=str(e),
                collection=collection_name,
                points_count=len(point_ids)
            )
            return False
    
    async def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get information about a Qdrant collection."""
        if not self._initialized:
            await self.initialize()
        
        try:
            collection_info = self.client.get_collection(collection_name)
            
            return {
                "name": collection_name,
                "status": collection_info.status.value,
                "vectors_count": collection_info.vectors_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count,
                "points_count": collection_info.points_count,
                "segments_count": collection_info.segments_count,
                "config": {
                    "vector_size": collection_info.config.params.vectors.size,
                    "distance": collection_info.config.params.vectors.distance.value,
                    "hnsw_config": {
                        "m": collection_info.config.hnsw_config.m,
                        "ef_construct": collection_info.config.hnsw_config.ef_construct,
                        "full_scan_threshold": collection_info.config.hnsw_config.full_scan_threshold
                    }
                }
            }
            
        except Exception as e:
            logger.error("Failed to get Qdrant collection info", error=str(e), collection=collection_name)
            return {}
    
    async def list_collections(self) -> List[str]:
        """List all collections in Qdrant."""
        if not self._initialized:
            await self.initialize()
        
        try:
            collections_response = self.client.get_collections()
            collection_names = [col.name for col in collections_response.collections]
            
            logger.debug("Listed Qdrant collections", count=len(collection_names))
            return collection_names
            
        except Exception as e:
            logger.error("Failed to list Qdrant collections", error=str(e))
            return []
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Qdrant service health."""
        if not self._initialized:
            return {"status": "unhealthy", "error": "Not initialized"}
        
        try:
            # Get cluster info
            cluster_info = self.client.get_cluster_info()
            collections = self.client.get_collections()
            
            return {
                "status": "healthy",
                "service": "qdrant",
                "peer_id": cluster_info.peer_id,
                "raft_info": {
                    "term": cluster_info.raft_info.term,
                    "commit": cluster_info.raft_info.commit,
                    "pending_operations": cluster_info.raft_info.pending_operations,
                    "leader": cluster_info.raft_info.leader,
                    "role": cluster_info.raft_info.role.value if cluster_info.raft_info.role else None
                },
                "collections_count": len(collections.collections),
                "connection": self._connection_params
            }
            
        except Exception as e:
            logger.error("Qdrant health check failed", error=str(e))
            return {
                "status": "unhealthy",
                "service": "qdrant",
                "error": str(e)
            }
    
    async def cleanup(self) -> None:
        """Cleanup Qdrant client resources."""
        if self.client:
            try:
                self.client.close()
                logger.info("Qdrant client closed")
            except Exception as e:
                logger.error("Error closing Qdrant client", error=str(e))
        
        self._initialized = False
    
    def _get_connection_params(self) -> Dict[str, Any]:
        """Get Qdrant connection parameters from settings."""
        # Try environment variables first
        import os
        
        params = {
            "host": os.getenv("QDRANT_HOST", "localhost"),
            "port": int(os.getenv("QDRANT_PORT", "6333")),
            "grpc_port": int(os.getenv("QDRANT_GRPC_PORT", "6334")),
            "prefer_grpc": os.getenv("QDRANT_PREFER_GRPC", "false").lower() == "true",
            "https": os.getenv("QDRANT_HTTPS", "false").lower() == "true",
            "timeout": float(os.getenv("QDRANT_TIMEOUT", "60.0"))
        }
        
        # Add API key if provided
        api_key = os.getenv("QDRANT_API_KEY")
        if api_key:
            params["api_key"] = api_key
        
        # Add URL if provided (overrides host/port)
        url = os.getenv("QDRANT_URL")
        if url:
            params = {"url": url, "timeout": params["timeout"]}
            if api_key:
                params["api_key"] = api_key
        
        return params
    
    def _build_filter_conditions(self, filters: Dict[str, Any]) -> models.Filter:
        """Build Qdrant filter conditions from dictionary."""
        conditions = []
        
        for key, value in filters.items():
            if isinstance(value, (str, int, float, bool)):
                condition = models.FieldCondition(
                    key=key,
                    match=models.MatchValue(value=value)
                )
            elif isinstance(value, list):
                condition = models.FieldCondition(
                    key=key,
                    match=models.MatchAny(any=value)
                )
            elif isinstance(value, dict):
                # Handle range conditions
                if "gte" in value or "lte" in value or "gt" in value or "lt" in value:
                    range_condition = models.Range()
                    if "gte" in value:
                        range_condition.gte = value["gte"]
                    if "lte" in value:
                        range_condition.lte = value["lte"]
                    if "gt" in value:
                        range_condition.gt = value["gt"]
                    if "lt" in value:
                        range_condition.lt = value["lt"]
                    
                    condition = models.FieldCondition(
                        key=key,
                        range=range_condition
                    )
                else:
                    # Nested object match
                    condition = models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value)
                    )
            else:
                continue
            
            conditions.append(condition)
        
        if conditions:
            return models.Filter(must=conditions)
        
        return None
