"""
Tenant-Isolated Vector Database Service
CRITICAL: Security fix for multi-tenant data isolation
"""

from typing import List, Dict, Any, Optional, Union
from uuid import UUID
import logging
from datetime import datetime

from app.services.vector.base import BaseVectorService, VectorPoint, SearchResult
from app.core.database import get_database
from app.models.workspace import Workspace
from app.models.user import User

logger = logging.getLogger(__name__)

class TenantIsolatedVectorService(BaseVectorService):
    """Vector service with proper tenant isolation and security"""
    
    def __init__(self, vector_service: BaseVectorService):
        """Initialize with existing vector service"""
        self.vector_service = vector_service
        self.db = None
    
    async def initialize(self):
        """Initialize the service"""
        await self.vector_service.initialize()
        self.db = await get_database()
    
    def _get_tenant_collection_name(self, workspace_id: str, collection_name: str) -> str:
        """Generate tenant-isolated collection name"""
        # Format: ws_<workspace_id>_<collection_name>
        clean_workspace_id = str(workspace_id).replace('-', '')[:16]  # Limit length
        clean_collection_name = collection_name.replace('-', '_').lower()
        return f"ws_{clean_workspace_id}_{clean_collection_name}"
    
    async def _validate_workspace_access(
        self, 
        user_id: str, 
        workspace_id: str, 
        permission: str = "read"
    ) -> bool:
        """Validate user has access to workspace"""
        try:
            query = """
                SELECT validate_workspace_access($1::UUID, $2::UUID, $3)
            """
            result = await self.db.fetchval(query, UUID(user_id), UUID(workspace_id), permission)
            return result or False
        except Exception as e:
            logger.error(f"Error validating workspace access: {e}")
            return False
    
    async def _log_vector_access(
        self,
        user_id: str,
        workspace_id: str,
        action: str,
        collection_name: str,
        metadata: Dict[str, Any] = None
    ):
        """Log vector database access for audit"""
        try:
            query = """
                SELECT log_data_access($1::UUID, $2::UUID, 
                    (SELECT organization_id FROM workspaces WHERE id = $2::UUID),
                    $3, 'vector_collection', $4::UUID, $5::JSONB)
            """
            await self.db.execute(
                query,
                UUID(user_id),
                UUID(workspace_id),
                action,
                UUID('00000000-0000-0000-0000-000000000000'),  # Placeholder for collection ID
                metadata or {}
            )
        except Exception as e:
            logger.error(f"Error logging vector access: {e}")
    
    async def create_tenant_collection(
        self,
        workspace_id: str,
        collection_name: str,
        dimension: int,
        user_id: str,
        **kwargs
    ) -> bool:
        """Create collection with tenant isolation"""
        
        # Validate workspace access
        if not await self._validate_workspace_access(user_id, workspace_id, "write"):
            raise PermissionError(f"User {user_id} does not have write access to workspace {workspace_id}")
        
        # Generate tenant-isolated collection name
        tenant_collection = self._get_tenant_collection_name(workspace_id, collection_name)
        
        # Log access
        await self._log_vector_access(
            user_id, workspace_id, "create_collection", tenant_collection,
            {"original_name": collection_name, "dimension": dimension}
        )
        
        # Create collection with tenant isolation
        result = await self.vector_service.create_collection(
            name=tenant_collection,
            dimension=dimension,
            **kwargs
        )
        
        logger.info(f"Created tenant collection: {tenant_collection} for workspace: {workspace_id}")
        return result
    
    async def insert_points_with_tenant_metadata(
        self,
        workspace_id: str,
        collection_name: str,
        points: List[VectorPoint],
        user_id: str,
        **kwargs
    ) -> bool:
        """Insert vectors with tenant metadata"""
        
        # Validate workspace access
        if not await self._validate_workspace_access(user_id, workspace_id, "write"):
            raise PermissionError(f"User {user_id} does not have write access to workspace {workspace_id}")
        
        # Generate tenant-isolated collection name
        tenant_collection = self._get_tenant_collection_name(workspace_id, collection_name)
        
        # Add tenant metadata to all points
        for point in points:
            if not hasattr(point, 'metadata'):
                point.metadata = {}
            
            # Add mandatory tenant metadata
            point.metadata.update({
                "workspace_id": workspace_id,
                "tenant_id": f"ws_{workspace_id}",
                "created_at": datetime.utcnow().isoformat(),
                "created_by": user_id,
                "tenant_collection": tenant_collection
            })
        
        # Log access
        await self._log_vector_access(
            user_id, workspace_id, "insert_vectors", tenant_collection,
            {"points_count": len(points)}
        )
        
        # Insert points with tenant metadata
        result = await self.vector_service.insert_points(tenant_collection, points, **kwargs)
        
        logger.info(f"Inserted {len(points)} points to tenant collection: {tenant_collection}")
        return result
    
    async def search_with_tenant_isolation(
        self,
        workspace_id: str,
        collection_name: str,
        vector: List[float],
        user_id: str,
        limit: int = 10,
        include_private: bool = False,
        **kwargs
    ) -> List[SearchResult]:
        """Search with proper tenant and user isolation"""
        
        # Validate workspace access
        if not await self._validate_workspace_access(user_id, workspace_id, "read"):
            raise PermissionError(f"User {user_id} does not have read access to workspace {workspace_id}")
        
        # Generate tenant-isolated collection name
        tenant_collection = self._get_tenant_collection_name(workspace_id, collection_name)
        
        # Build metadata filters for tenant isolation
        metadata_filters = {
            "workspace_id": workspace_id,
            "tenant_id": f"ws_{workspace_id}"
        }
        
        # Add user-level filters if needed
        if not include_private:
            # Only include documents user has access to
            metadata_filters["$or"] = [
                {"visibility": "public"},
                {"visibility": "internal"},
                {"created_by": user_id}  # User's own documents
            ]
        
        # Merge with additional filters
        if "metadata_filters" in kwargs:
            metadata_filters.update(kwargs["metadata_filters"])
        kwargs["metadata_filters"] = metadata_filters
        
        # Log access
        await self._log_vector_access(
            user_id, workspace_id, "search_vectors", tenant_collection,
            {"limit": limit, "filters": metadata_filters}
        )
        
        # Perform search with tenant isolation
        results = await self.vector_service.search(
            collection_name=tenant_collection,
            vector=vector,
            limit=limit,
            **kwargs
        )
        
        logger.info(f"Search in tenant collection: {tenant_collection}, found {len(results)} results")
        return results
    
    async def delete_tenant_collection(
        self,
        workspace_id: str,
        collection_name: str,
        user_id: str
    ) -> bool:
        """Delete tenant collection with proper authorization"""
        
        # Validate workspace access (admin required for deletion)
        if not await self._validate_workspace_access(user_id, workspace_id, "delete"):
            raise PermissionError(f"User {user_id} does not have delete access to workspace {workspace_id}")
        
        # Generate tenant-isolated collection name
        tenant_collection = self._get_tenant_collection_name(workspace_id, collection_name)
        
        # Log access
        await self._log_vector_access(
            user_id, workspace_id, "delete_collection", tenant_collection,
            {"original_name": collection_name}
        )
        
        # Delete collection
        result = await self.vector_service.delete_collection(tenant_collection)
        
        logger.warning(f"Deleted tenant collection: {tenant_collection} by user: {user_id}")
        return result
    
    async def get_user_accessible_collections(
        self,
        user_id: str,
        workspace_id: str
    ) -> List[Dict[str, Any]]:
        """Get collections user can access in workspace"""
        
        # Validate workspace access
        if not await self._validate_workspace_access(user_id, workspace_id, "read"):
            return []
        
        try:
            query = """
                SELECT c.id, c.name, c.tenant_id, c.visibility, c.metadata
                FROM collections c
                WHERE c.workspace_id = $1::UUID
                AND (
                    c.visibility = 'public' OR
                    c.visibility = 'internal' OR
                    c.owner_id = $2::UUID
                )
                ORDER BY c.created_at DESC
            """
            
            rows = await self.db.fetch(query, UUID(workspace_id), UUID(user_id))
            
            collections = []
            for row in rows:
                collections.append({
                    "id": str(row["id"]),
                    "name": row["name"],
                    "tenant_id": row["tenant_id"],
                    "visibility": row["visibility"],
                    "metadata": row["metadata"] or {},
                    "tenant_collection_name": self._get_tenant_collection_name(
                        workspace_id, row["name"]
                    )
                })
            
            return collections
            
        except Exception as e:
            logger.error(f"Error getting accessible collections: {e}")
            return []
    
    async def migrate_existing_collection_to_tenant(
        self,
        old_collection_name: str,
        workspace_id: str,
        new_collection_name: str,
        user_id: str
    ) -> bool:
        """Migrate existing collection to tenant-isolated format"""
        
        # Validate workspace access
        if not await self._validate_workspace_access(user_id, workspace_id, "admin"):
            raise PermissionError(f"User {user_id} does not have admin access to workspace {workspace_id}")
        
        try:
            # Generate new tenant collection name
            tenant_collection = self._get_tenant_collection_name(workspace_id, new_collection_name)
            
            # Get all points from old collection
            old_points = await self.vector_service.get_all_points(old_collection_name)
            
            if old_points:
                # Add tenant metadata to existing points
                for point in old_points:
                    if not hasattr(point, 'metadata'):
                        point.metadata = {}
                    
                    point.metadata.update({
                        "workspace_id": workspace_id,
                        "tenant_id": f"ws_{workspace_id}",
                        "migrated_at": datetime.utcnow().isoformat(),
                        "migrated_by": user_id,
                        "original_collection": old_collection_name
                    })
                
                # Create new tenant collection
                await self.create_tenant_collection(
                    workspace_id, new_collection_name, 
                    len(old_points[0].vector), user_id
                )
                
                # Insert points to new collection
                await self.insert_points_with_tenant_metadata(
                    workspace_id, new_collection_name, old_points, user_id
                )
                
                logger.info(f"Migrated {len(old_points)} points from {old_collection_name} to {tenant_collection}")
            
            # Log migration
            await self._log_vector_access(
                user_id, workspace_id, "migrate_collection", tenant_collection,
                {
                    "old_collection": old_collection_name,
                    "points_migrated": len(old_points) if old_points else 0
                }
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error migrating collection: {e}")
            return False
