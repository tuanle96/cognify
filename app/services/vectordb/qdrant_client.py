"""
Qdrant vector database client implementation.
"""

import asyncio
import time
from typing import List, Dict, Any, Optional
import logging

try:
    from qdrant_client import QdrantClient as QdrantSyncClient
    from qdrant_client.async_qdrant_client import AsyncQdrantClient
    from qdrant_client.models import (
        Distance, VectorParams, CreateCollection, PointStruct,
        Filter, FieldCondition, SearchRequest as QdrantSearchRequest,
        UpdateResult, CollectionInfo as QdrantCollectionInfo
    )
    from qdrant_client.http.exceptions import UnexpectedResponse
except ImportError:
    AsyncQdrantClient = None
    QdrantSyncClient = None
    Distance = None
    VectorParams = None
    CreateCollection = None
    PointStruct = None
    Filter = None
    FieldCondition = None
    QdrantSearchRequest = None
    UpdateResult = None
    QdrantCollectionInfo = None
    UnexpectedResponse = None

from .base import (
    VectorDBClient, VectorPoint, SearchResult, SearchRequest, CollectionInfo,
    VectorDBProvider, VectorDBConnectionError, VectorDBOperationError,
    CollectionNotFoundError, CollectionAlreadyExistsError
)

logger = logging.getLogger(__name__)

class QdrantClient(VectorDBClient):
    """Qdrant vector database client with async support."""

    # Distance metric mapping
    @property
    def DISTANCE_METRICS(self):
        if Distance is None:
            return {
                "cosine": "COSINE",
                "euclidean": "EUCLID",
                "dot": "DOT",
                "manhattan": "MANHATTAN"
            }
        return {
            "cosine": Distance.COSINE,
            "euclidean": Distance.EUCLID,
            "dot": Distance.DOT,
            "manhattan": Distance.MANHATTAN
        }

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        grpc_port: Optional[int] = None,
        api_key: Optional[str] = None,
        https: bool = False,
        timeout: float = 30.0,
        **kwargs
    ):
        if not AsyncQdrantClient:
            raise ImportError("qdrant-client package is required for Qdrant client")

        super().__init__(host, port, **kwargs)
        self.grpc_port = grpc_port
        self.api_key = api_key
        self.https = https
        self.timeout = timeout
        self.client = None

    @property
    def provider(self) -> VectorDBProvider:
        return VectorDBProvider.QDRANT

    @property
    def max_batch_size(self) -> int:
        return 1000  # Qdrant's recommended batch size

    @property
    def supported_distance_metrics(self) -> List[str]:
        return list(self.DISTANCE_METRICS.keys())

    async def initialize(self) -> None:
        """Initialize the Qdrant client."""
        try:
            # Build connection URL
            protocol = "https" if self.https else "http"
            url = f"{protocol}://{self.host}:{self.port}"

            # Create async client
            self.client = AsyncQdrantClient(
                url=url,
                api_key=self.api_key,
                timeout=self.timeout,
                grpc_port=self.grpc_port
            )

            # Test connection
            await self.health_check()

            self._initialized = True
            logger.info(f"Qdrant client initialized: {url}")

        except Exception as e:
            raise VectorDBConnectionError(self.provider, f"Failed to initialize: {e}", e)

    async def create_collection(
        self,
        name: str,
        dimension: int,
        distance_metric: str = "cosine",
        **kwargs
    ) -> bool:
        """Create a new collection in Qdrant."""
        if not self._initialized:
            await self.initialize()

        self._validate_collection_name(name)
        self._validate_dimension(dimension)
        self._validate_distance_metric(distance_metric)

        try:
            # Check if collection already exists
            if await self.collection_exists(name):
                raise CollectionAlreadyExistsError(name)

            # Map distance metric
            qdrant_distance = self.DISTANCE_METRICS[distance_metric]

            # Create collection
            await self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=dimension,
                    distance=qdrant_distance
                ),
                **kwargs
            )

            logger.info(f"Created Qdrant collection: {name} (dim={dimension}, metric={distance_metric})")
            return True

        except CollectionAlreadyExistsError:
            raise
        except Exception as e:
            raise VectorDBOperationError(self.provider, "create_collection", str(e))

    async def delete_collection(self, name: str) -> bool:
        """Delete a collection from Qdrant."""
        if not self._initialized:
            await self.initialize()

        try:
            # Check if collection exists
            if not await self.collection_exists(name):
                raise CollectionNotFoundError(name)

            # Delete collection
            await self.client.delete_collection(collection_name=name)

            logger.info(f"Deleted Qdrant collection: {name}")
            return True

        except CollectionNotFoundError:
            raise
        except Exception as e:
            raise VectorDBOperationError(self.provider, "delete_collection", str(e))

    async def collection_exists(self, name: str) -> bool:
        """Check if collection exists in Qdrant."""
        if not self._initialized:
            await self.initialize()

        try:
            collections = await self.client.get_collections()
            return any(col.name == name for col in collections.collections)
        except Exception as e:
            logger.warning(f"Error checking collection existence: {e}")
            return False

    async def get_collection_info(self, name: str) -> CollectionInfo:
        """Get information about a Qdrant collection."""
        if not self._initialized:
            await self.initialize()

        try:
            # Check if collection exists
            if not await self.collection_exists(name):
                raise CollectionNotFoundError(name)

            # Get collection info
            info = await self.client.get_collection(collection_name=name)

            # Map distance metric back
            distance_metric = "cosine"  # default
            for metric, qdrant_dist in self.DISTANCE_METRICS.items():
                if info.config.params.vectors.distance == qdrant_dist:
                    distance_metric = metric
                    break

            return CollectionInfo(
                name=name,
                dimension=info.config.params.vectors.size,
                vector_count=info.points_count or 0,
                distance_metric=distance_metric,
                metadata={
                    "status": info.status.value,
                    "optimizer_status": info.optimizer_status,
                    "indexed_vectors_count": info.indexed_vectors_count
                }
            )

        except CollectionNotFoundError:
            raise
        except Exception as e:
            raise VectorDBOperationError(self.provider, "get_collection_info", str(e))

    async def list_collections(self) -> List[str]:
        """List all collections in Qdrant."""
        if not self._initialized:
            await self.initialize()

        try:
            collections = await self.client.get_collections()
            return [col.name for col in collections.collections]
        except Exception as e:
            raise VectorDBOperationError(self.provider, "list_collections", str(e))

    async def insert_points(self, collection_name: str, points: List[VectorPoint]) -> bool:
        """Insert points into a Qdrant collection."""
        if not self._initialized:
            await self.initialize()

        if not points:
            return True

        try:
            # Check if collection exists
            if not await self.collection_exists(collection_name):
                raise CollectionNotFoundError(collection_name)

            # Convert to Qdrant points
            qdrant_points = [
                PointStruct(
                    id=point.id,
                    vector=point.vector,
                    payload=point.metadata or {}
                )
                for point in points
            ]

            # Insert points
            result = await self.client.upsert(
                collection_name=collection_name,
                points=qdrant_points
            )

            logger.debug(f"Inserted {len(points)} points into {collection_name}")
            return result.status.name == "COMPLETED"

        except CollectionNotFoundError:
            raise
        except Exception as e:
            raise VectorDBOperationError(self.provider, "insert_points", str(e))

    async def update_points(self, collection_name: str, points: List[VectorPoint]) -> bool:
        """Update existing points in a Qdrant collection."""
        # Qdrant upsert handles both insert and update
        return await self.insert_points(collection_name, points)

    async def delete_points(self, collection_name: str, point_ids: List[str]) -> bool:
        """Delete points from a Qdrant collection."""
        if not self._initialized:
            await self.initialize()

        if not point_ids:
            return True

        try:
            # Check if collection exists
            if not await self.collection_exists(collection_name):
                raise CollectionNotFoundError(collection_name)

            # Delete points
            result = await self.client.delete(
                collection_name=collection_name,
                points_selector=point_ids
            )

            logger.debug(f"Deleted {len(point_ids)} points from {collection_name}")
            return result.status.name == "COMPLETED"

        except CollectionNotFoundError:
            raise
        except Exception as e:
            raise VectorDBOperationError(self.provider, "delete_points", str(e))

    async def search(self, collection_name: str, request: SearchRequest) -> List[SearchResult]:
        """Search for similar vectors in Qdrant."""
        if not self._initialized:
            await self.initialize()

        try:
            # Check if collection exists
            if not await self.collection_exists(collection_name):
                raise CollectionNotFoundError(collection_name)

            # Build filter if provided
            qdrant_filter = None
            if request.filter_conditions:
                qdrant_filter = self._build_filter(request.filter_conditions)

            # Perform search
            search_result = await self.client.search(
                collection_name=collection_name,
                query_vector=request.vector,
                limit=request.limit,
                score_threshold=request.score_threshold,
                query_filter=qdrant_filter,
                with_payload=request.include_metadata,
                with_vectors=request.include_vectors
            )

            # Convert results
            results = []
            for hit in search_result:
                result = SearchResult(
                    id=str(hit.id),
                    score=hit.score,
                    vector=hit.vector if request.include_vectors else None,
                    metadata=hit.payload if request.include_metadata else None
                )
                results.append(result)

            logger.debug(f"Search in {collection_name} returned {len(results)} results")
            return results

        except CollectionNotFoundError:
            raise
        except Exception as e:
            raise VectorDBOperationError(self.provider, "search", str(e))

    async def get_point(self, collection_name: str, point_id: str) -> Optional[VectorPoint]:
        """Get a specific point by ID from Qdrant."""
        if not self._initialized:
            await self.initialize()

        try:
            # Check if collection exists
            if not await self.collection_exists(collection_name):
                raise CollectionNotFoundError(collection_name)

            # Get point
            points = await self.client.retrieve(
                collection_name=collection_name,
                ids=[point_id],
                with_payload=True,
                with_vectors=True
            )

            if not points:
                return None

            point = points[0]
            return VectorPoint(
                id=str(point.id),
                vector=point.vector,
                metadata=point.payload or {}
            )

        except CollectionNotFoundError:
            raise
        except Exception as e:
            raise VectorDBOperationError(self.provider, "get_point", str(e))

    async def count_points(self, collection_name: str) -> int:
        """Count points in a Qdrant collection."""
        if not self._initialized:
            await self.initialize()

        try:
            # Check if collection exists
            if not await self.collection_exists(collection_name):
                raise CollectionNotFoundError(collection_name)

            # Get collection info for count
            info = await self.client.get_collection(collection_name=collection_name)
            return info.points_count or 0

        except CollectionNotFoundError:
            raise
        except Exception as e:
            raise VectorDBOperationError(self.provider, "count_points", str(e))

    async def health_check(self) -> Dict[str, Any]:
        """Check the health of the Qdrant database."""
        try:
            if not self._initialized:
                await self.initialize()

            # Get cluster info
            cluster_info = await self.client.get_cluster_info()

            return {
                "status": "healthy",
                "provider": self.provider.value,
                "host": self.host,
                "port": self.port,
                "cluster_status": cluster_info.status.value if cluster_info else "unknown",
                "peer_count": len(cluster_info.peers) if cluster_info and cluster_info.peers else 0
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": self.provider.value,
                "error": str(e)
            }

    async def cleanup(self) -> None:
        """Cleanup Qdrant client resources."""
        if self.client:
            await self.client.close()
            self.client = None
        self._initialized = False
        logger.info("Qdrant client cleaned up")

    def _build_filter(self, conditions: Dict[str, Any]):
        """Build Qdrant filter from conditions."""
        # Simple implementation - can be extended for complex filters
        must_conditions = []

        for field, value in conditions.items():
            if isinstance(value, dict):
                # Handle operators like {"$eq": "value"}, {"$in": ["val1", "val2"]}
                for op, op_value in value.items():
                    if op == "$eq":
                        must_conditions.append(FieldCondition(key=field, match={"value": op_value}))
                    elif op == "$in":
                        must_conditions.append(FieldCondition(key=field, match={"any": op_value}))
                    # Add more operators as needed
            else:
                # Direct equality
                must_conditions.append(FieldCondition(key=field, match={"value": value}))

        return Filter(must=must_conditions) if must_conditions else None
