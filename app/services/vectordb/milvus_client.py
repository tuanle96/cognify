"""
Milvus vector database client implementation.
"""

import asyncio
import time
from typing import List, Dict, Any, Optional
import logging

try:
    from pymilvus import connections, Collection, utility, DataType, FieldSchema, CollectionSchema
    from pymilvus.exceptions import MilvusException
except ImportError:
    connections = None
    Collection = None
    utility = None
    DataType = None
    FieldSchema = None
    CollectionSchema = None
    MilvusException = None

from .base import (
    VectorDBClient, VectorPoint, SearchResult, SearchRequest, CollectionInfo,
    VectorDBProvider, VectorDBConnectionError, VectorDBOperationError,
    CollectionNotFoundError, CollectionAlreadyExistsError
)

logger = logging.getLogger(__name__)

class MilvusClient(VectorDBClient):
    """Milvus vector database client with async support."""

    # Distance metric mapping
    DISTANCE_METRICS = {
        "cosine": "COSINE",
        "euclidean": "L2",
        "manhattan": "L1",
        "dot": "IP"  # Inner Product
    }

    def __init__(
        self,
        host: str = "localhost",
        port: int = 19530,
        user: Optional[str] = None,
        password: Optional[str] = None,
        secure: bool = False,
        timeout: float = 30.0,
        **kwargs
    ):
        if not connections:
            raise ImportError("pymilvus package is required for Milvus client")

        super().__init__(host, port, **kwargs)
        self.user = user
        self.password = password
        self.secure = secure
        self.timeout = timeout
        self.connection_alias = f"milvus_{id(self)}"
        self._collections_cache = {}

    @property
    def provider(self) -> VectorDBProvider:
        return VectorDBProvider.MILVUS

    @property
    def max_batch_size(self) -> int:
        return 1000  # Milvus recommended batch size

    @property
    def supported_distance_metrics(self) -> List[str]:
        return list(self.DISTANCE_METRICS.keys())

    async def initialize(self) -> None:
        """Initialize the Milvus client."""
        try:
            # Connect to Milvus
            connections.connect(
                alias=self.connection_alias,
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                secure=self.secure,
                timeout=self.timeout
            )

            # Test connection
            await self.health_check()

            self._initialized = True
            logger.info(f"Milvus client initialized: {self.host}:{self.port}")

        except Exception as e:
            raise VectorDBConnectionError(self.provider, f"Failed to initialize: {e}", e)

    async def create_collection(
        self,
        name: str,
        dimension: int,
        distance_metric: str = "cosine",
        **kwargs
    ) -> bool:
        """Create a new collection in Milvus."""
        if not self._initialized:
            await self.initialize()

        self._validate_collection_name(name)
        self._validate_dimension(dimension)
        self._validate_distance_metric(distance_metric)

        try:
            # Check if collection already exists
            if await self.collection_exists(name):
                raise CollectionAlreadyExistsError(name)

            # Define collection schema
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=255),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dimension),
                FieldSchema(name="metadata", dtype=DataType.JSON)
            ]

            schema = CollectionSchema(
                fields=fields,
                description=f"Collection {name} with {dimension}D vectors"
            )

            # Create collection
            collection = Collection(
                name=name,
                schema=schema,
                using=self.connection_alias
            )

            # Create index for vector field
            index_params = {
                "metric_type": self.DISTANCE_METRICS[distance_metric],
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }

            collection.create_index(
                field_name="vector",
                index_params=index_params
            )

            # Cache collection
            self._collections_cache[name] = collection

            logger.info(f"Created Milvus collection: {name} (dim={dimension}, metric={distance_metric})")
            return True

        except CollectionAlreadyExistsError:
            raise
        except Exception as e:
            raise VectorDBOperationError(self.provider, "create_collection", str(e))

    async def delete_collection(self, name: str) -> bool:
        """Delete a collection from Milvus."""
        if not self._initialized:
            await self.initialize()

        try:
            # Check if collection exists
            if not await self.collection_exists(name):
                raise CollectionNotFoundError(name)

            # Drop collection
            utility.drop_collection(name, using=self.connection_alias)

            # Remove from cache
            self._collections_cache.pop(name, None)

            logger.info(f"Deleted Milvus collection: {name}")
            return True

        except CollectionNotFoundError:
            raise
        except Exception as e:
            raise VectorDBOperationError(self.provider, "delete_collection", str(e))

    async def collection_exists(self, name: str) -> bool:
        """Check if collection exists in Milvus."""
        if not self._initialized:
            await self.initialize()

        try:
            return utility.has_collection(name, using=self.connection_alias)
        except Exception as e:
            logger.warning(f"Error checking collection existence: {e}")
            return False

    async def get_collection_info(self, name: str) -> CollectionInfo:
        """Get information about a Milvus collection."""
        if not self._initialized:
            await self.initialize()

        try:
            # Check if collection exists
            if not await self.collection_exists(name):
                raise CollectionNotFoundError(name)

            # Get collection
            collection = self._get_collection(name)

            # Get collection stats
            collection.load()
            stats = collection.get_stats()

            # Extract dimension from schema
            dimension = 0
            for field in collection.schema.fields:
                if field.dtype == DataType.FLOAT_VECTOR:
                    dimension = field.params.get("dim", 0)
                    break

            return CollectionInfo(
                name=name,
                dimension=dimension,
                vector_count=int(stats.get("row_count", 0)),
                distance_metric="cosine",  # Default, would need to parse index info
                metadata={
                    "description": collection.description,
                    "is_loaded": utility.load_state(name, using=self.connection_alias).name,
                    "stats": stats
                }
            )

        except CollectionNotFoundError:
            raise
        except Exception as e:
            raise VectorDBOperationError(self.provider, "get_collection_info", str(e))

    async def list_collections(self) -> List[str]:
        """List all collections in Milvus."""
        if not self._initialized:
            await self.initialize()

        try:
            return utility.list_collections(using=self.connection_alias)
        except Exception as e:
            raise VectorDBOperationError(self.provider, "list_collections", str(e))

    async def insert_points(self, collection_name: str, points: List[VectorPoint]) -> bool:
        """Insert points into a Milvus collection."""
        if not self._initialized:
            await self.initialize()

        if not points:
            return True

        try:
            # Check if collection exists
            if not await self.collection_exists(collection_name):
                raise CollectionNotFoundError(collection_name)

            # Get collection
            collection = self._get_collection(collection_name)

            # Prepare data
            ids = [point.id for point in points]
            vectors = [point.vector for point in points]
            metadata = [point.metadata or {} for point in points]

            # Insert data
            data = [ids, vectors, metadata]
            collection.insert(data)

            # Flush to ensure data is written
            collection.flush()

            logger.debug(f"Inserted {len(points)} points into {collection_name}")
            return True

        except CollectionNotFoundError:
            raise
        except Exception as e:
            raise VectorDBOperationError(self.provider, "insert_points", str(e))

    async def update_points(self, collection_name: str, points: List[VectorPoint]) -> bool:
        """Update existing points in a Milvus collection."""
        # Milvus doesn't have direct update - need to delete and insert
        try:
            # Delete existing points
            point_ids = [point.id for point in points]
            await self.delete_points(collection_name, point_ids)

            # Insert new points
            return await self.insert_points(collection_name, points)

        except Exception as e:
            raise VectorDBOperationError(self.provider, "update_points", str(e))

    async def delete_points(self, collection_name: str, point_ids: List[str]) -> bool:
        """Delete points from a Milvus collection."""
        if not self._initialized:
            await self.initialize()

        if not point_ids:
            return True

        try:
            # Check if collection exists
            if not await self.collection_exists(collection_name):
                raise CollectionNotFoundError(collection_name)

            # Get collection
            collection = self._get_collection(collection_name)

            # Delete points
            expr = f"id in {point_ids}"
            collection.delete(expr)

            logger.debug(f"Deleted {len(point_ids)} points from {collection_name}")
            return True

        except CollectionNotFoundError:
            raise
        except Exception as e:
            raise VectorDBOperationError(self.provider, "delete_points", str(e))

    async def search(self, collection_name: str, request: SearchRequest) -> List[SearchResult]:
        """Search for similar vectors in Milvus."""
        if not self._initialized:
            await self.initialize()

        try:
            # Check if collection exists
            if not await self.collection_exists(collection_name):
                raise CollectionNotFoundError(collection_name)

            # Get collection
            collection = self._get_collection(collection_name)

            # Ensure collection is loaded
            collection.load()

            # Build search parameters
            search_params = {
                "metric_type": "COSINE",  # Default
                "params": {"nprobe": 10}
            }

            # Build filter expression if provided
            expr = None
            if request.filter_conditions:
                expr = self._build_filter_expression(request.filter_conditions)

            # Perform search
            search_result = collection.search(
                data=[request.vector],
                anns_field="vector",
                param=search_params,
                limit=request.limit,
                expr=expr,
                output_fields=["metadata"] if request.include_metadata else None
            )

            # Convert results
            results = []
            for hits in search_result:
                for hit in hits:
                    result = SearchResult(
                        id=str(hit.id),
                        score=float(hit.score),
                        vector=None,  # Milvus doesn't return vectors by default
                        metadata=hit.entity.get("metadata") if request.include_metadata else None
                    )
                    results.append(result)

            logger.debug(f"Search in {collection_name} returned {len(results)} results")
            return results

        except CollectionNotFoundError:
            raise
        except Exception as e:
            raise VectorDBOperationError(self.provider, "search", str(e))

    async def get_point(self, collection_name: str, point_id: str) -> Optional[VectorPoint]:
        """Get a specific point by ID from Milvus."""
        if not self._initialized:
            await self.initialize()

        try:
            # Check if collection exists
            if not await self.collection_exists(collection_name):
                raise CollectionNotFoundError(collection_name)

            # Get collection
            collection = self._get_collection(collection_name)

            # Ensure collection is loaded
            collection.load()

            # Query for the point
            expr = f"id == '{point_id}'"
            result = collection.query(
                expr=expr,
                output_fields=["id", "vector", "metadata"]
            )

            if not result:
                return None

            point_data = result[0]
            return VectorPoint(
                id=point_data["id"],
                vector=point_data["vector"],
                metadata=point_data.get("metadata", {})
            )

        except CollectionNotFoundError:
            raise
        except Exception as e:
            raise VectorDBOperationError(self.provider, "get_point", str(e))

    async def count_points(self, collection_name: str) -> int:
        """Count points in a Milvus collection."""
        if not self._initialized:
            await self.initialize()

        try:
            # Check if collection exists
            if not await self.collection_exists(collection_name):
                raise CollectionNotFoundError(collection_name)

            # Get collection
            collection = self._get_collection(collection_name)

            # Get count from stats
            stats = collection.get_stats()
            return int(stats.get("row_count", 0))

        except CollectionNotFoundError:
            raise
        except Exception as e:
            raise VectorDBOperationError(self.provider, "count_points", str(e))

    async def health_check(self) -> Dict[str, Any]:
        """Check the health of the Milvus database."""
        try:
            if not self._initialized:
                await self.initialize()

            # Check if connection is alive
            is_connected = connections.get_connection_addr(self.connection_alias)

            return {
                "status": "healthy" if is_connected else "unhealthy",
                "provider": self.provider.value,
                "host": self.host,
                "port": self.port,
                "connection_alias": self.connection_alias,
                "connected": bool(is_connected)
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": self.provider.value,
                "error": str(e)
            }

    async def cleanup(self) -> None:
        """Cleanup Milvus client resources."""
        try:
            if self.connection_alias in connections.list_connections():
                connections.disconnect(self.connection_alias)
            self._collections_cache.clear()
            self._initialized = False
            logger.info("Milvus client cleaned up")
        except Exception as e:
            logger.warning(f"Error during Milvus cleanup: {e}")

    def _get_collection(self, name: str):
        """Get collection instance (with caching)."""
        if name not in self._collections_cache:
            self._collections_cache[name] = Collection(name, using=self.connection_alias)
        return self._collections_cache[name]

    def _build_filter_expression(self, conditions: Dict[str, Any]) -> str:
        """Build Milvus filter expression from conditions."""
        # Simple implementation - can be extended for complex filters
        expressions = []

        for field, value in conditions.items():
            if isinstance(value, dict):
                # Handle operators
                for op, op_value in value.items():
                    if op == "$eq":
                        expressions.append(f"metadata['{field}'] == '{op_value}'")
                    elif op == "$in":
                        values_str = ", ".join([f"'{v}'" for v in op_value])
                        expressions.append(f"metadata['{field}'] in [{values_str}]")
                    # Add more operators as needed
            else:
                # Direct equality
                expressions.append(f"metadata['{field}'] == '{value}'")

        return " and ".join(expressions) if expressions else None
