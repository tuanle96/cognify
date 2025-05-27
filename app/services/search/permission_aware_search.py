"""
Permission-Aware Search Service
CRITICAL: Security fix for multi-tenant search isolation
"""

from typing import List, Dict, Any, Optional, Union
from uuid import UUID
import logging
from datetime import datetime

from app.services.search.base import BaseSearchService, SearchResult, SearchRequest
from app.services.vector.tenant_isolated_service import TenantIsolatedVectorService
from app.core.database import get_database
from app.models.workspace import Workspace
from app.models.user import User

logger = logging.getLogger(__name__)

class PermissionAwareSearchService(BaseSearchService):
    """Search service with comprehensive permission validation"""
    
    def __init__(self, vector_service: TenantIsolatedVectorService):
        """Initialize with tenant-isolated vector service"""
        self.vector_service = vector_service
        self.db = None
    
    async def initialize(self):
        """Initialize the service"""
        await self.vector_service.initialize()
        self.db = await get_database()
    
    async def search_with_permissions(
        self,
        query: str,
        user_id: str,
        workspace_id: str,
        collection_id: Optional[str] = None,
        limit: int = 10,
        include_metadata: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Search with comprehensive permission checking"""
        
        try:
            # 1. Validate workspace access
            if not await self._validate_workspace_access(user_id, workspace_id):
                return {
                    "results": [],
                    "total": 0,
                    "error": "Access denied to workspace",
                    "workspace_id": workspace_id
                }
            
            # 2. Get user's accessible collections
            accessible_collections = await self._get_user_accessible_collections(
                user_id=user_id,
                workspace_id=workspace_id,
                collection_id=collection_id
            )
            
            if not accessible_collections:
                return {
                    "results": [],
                    "total": 0,
                    "message": "No accessible collections found",
                    "workspace_id": workspace_id
                }
            
            # 3. Generate query embedding
            query_vector = await self._get_query_embedding(query)
            if not query_vector:
                return {
                    "results": [],
                    "total": 0,
                    "error": "Failed to generate query embedding"
                }
            
            # 4. Perform search across accessible collections
            all_results = []
            search_metadata = {
                "collections_searched": [],
                "total_collections": len(accessible_collections),
                "search_timestamp": datetime.utcnow().isoformat()
            }
            
            for collection in accessible_collections:
                try:
                    # Search in tenant-isolated collection
                    collection_results = await self.vector_service.search_with_tenant_isolation(
                        workspace_id=workspace_id,
                        collection_name=collection["name"],
                        vector=query_vector,
                        user_id=user_id,
                        limit=limit,
                        include_private=self._user_can_see_private(collection, user_id),
                        **kwargs
                    )
                    
                    # Add collection context to results
                    for result in collection_results:
                        result.metadata = result.metadata or {}
                        result.metadata.update({
                            "collection_id": collection["id"],
                            "collection_name": collection["name"],
                            "workspace_id": workspace_id,
                            "search_query": query
                        })
                    
                    all_results.extend(collection_results)
                    search_metadata["collections_searched"].append({
                        "collection_id": collection["id"],
                        "collection_name": collection["name"],
                        "results_count": len(collection_results)
                    })
                    
                except Exception as e:
                    logger.error(f"Error searching collection {collection['name']}: {e}")
                    search_metadata["collections_searched"].append({
                        "collection_id": collection["id"],
                        "collection_name": collection["name"],
                        "error": str(e)
                    })
            
            # 5. Merge, deduplicate, and rank results
            final_results = await self._merge_and_rank_results(
                all_results, query, limit, **kwargs
            )
            
            # 6. Log search access
            await self._log_search_access(
                user_id, workspace_id, query, len(final_results), search_metadata
            )
            
            return {
                "results": final_results,
                "total": len(final_results),
                "query": query,
                "workspace_id": workspace_id,
                "metadata": search_metadata if include_metadata else None
            }
            
        except Exception as e:
            logger.error(f"Error in permission-aware search: {e}")
            return {
                "results": [],
                "total": 0,
                "error": str(e),
                "workspace_id": workspace_id
            }
    
    async def _validate_workspace_access(self, user_id: str, workspace_id: str) -> bool:
        """Validate user has access to workspace"""
        try:
            query = """
                SELECT validate_workspace_access($1::UUID, $2::UUID, 'read')
            """
            result = await self.db.fetchval(query, UUID(user_id), UUID(workspace_id))
            return result or False
        except Exception as e:
            logger.error(f"Error validating workspace access: {e}")
            return False
    
    async def _get_user_accessible_collections(
        self,
        user_id: str,
        workspace_id: str,
        collection_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get collections user can access"""
        
        try:
            # Base query for accessible collections
            base_query = """
                SELECT c.id, c.name, c.tenant_id, c.visibility, c.owner_id, c.metadata
                FROM collections c
                WHERE c.workspace_id = $1::UUID
                AND (
                    c.visibility = 'public' OR
                    c.visibility = 'internal' OR
                    c.owner_id = $2::UUID OR
                    EXISTS (
                        SELECT 1 FROM collection_members cm
                        WHERE cm.collection_id = c.id 
                        AND cm.user_id = $2::UUID
                        AND cm.status = 'active'
                    )
                )
            """
            
            params = [UUID(workspace_id), UUID(user_id)]
            
            # Filter by specific collection if provided
            if collection_id:
                base_query += " AND c.id = $3::UUID"
                params.append(UUID(collection_id))
            
            base_query += " ORDER BY c.created_at DESC"
            
            rows = await self.db.fetch(base_query, *params)
            
            collections = []
            for row in rows:
                collections.append({
                    "id": str(row["id"]),
                    "name": row["name"],
                    "tenant_id": row["tenant_id"],
                    "visibility": row["visibility"],
                    "owner_id": str(row["owner_id"]) if row["owner_id"] else None,
                    "metadata": row["metadata"] or {}
                })
            
            return collections
            
        except Exception as e:
            logger.error(f"Error getting accessible collections: {e}")
            return []
    
    def _user_can_see_private(self, collection: Dict[str, Any], user_id: str) -> bool:
        """Check if user can see private documents in collection"""
        return (
            collection["visibility"] == "public" or
            collection["owner_id"] == user_id
        )
    
    async def _get_query_embedding(self, query: str) -> Optional[List[float]]:
        """Generate embedding for search query"""
        try:
            # Use existing embedding service
            from app.services.llm.openai_service import OpenAIService
            
            openai_service = OpenAIService()
            await openai_service.initialize()
            
            embeddings = await openai_service.get_embeddings([query])
            return embeddings[0] if embeddings else None
            
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            return None
    
    async def _merge_and_rank_results(
        self,
        results: List[SearchResult],
        query: str,
        limit: int,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Merge and rank search results from multiple collections"""
        
        # Remove duplicates based on content hash
        seen_content = set()
        unique_results = []
        
        for result in results:
            content_hash = hash(result.content[:200])  # Hash first 200 chars
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_results.append(result)
        
        # Sort by relevance score (descending)
        unique_results.sort(key=lambda x: x.score, reverse=True)
        
        # Limit results
        limited_results = unique_results[:limit]
        
        # Convert to dict format
        formatted_results = []
        for i, result in enumerate(limited_results):
            formatted_results.append({
                "rank": i + 1,
                "content": result.content,
                "score": result.score,
                "metadata": result.metadata or {},
                "id": getattr(result, 'id', None)
            })
        
        return formatted_results
    
    async def _log_search_access(
        self,
        user_id: str,
        workspace_id: str,
        query: str,
        results_count: int,
        metadata: Dict[str, Any]
    ):
        """Log search access for audit"""
        try:
            query_log = """
                SELECT log_data_access($1::UUID, $2::UUID, 
                    (SELECT organization_id FROM workspaces WHERE id = $2::UUID),
                    'search', 'workspace', $2::UUID, $3::JSONB)
            """
            
            log_metadata = {
                "query": query,
                "results_count": results_count,
                "search_metadata": metadata
            }
            
            await self.db.execute(
                query_log,
                UUID(user_id),
                UUID(workspace_id),
                log_metadata
            )
            
        except Exception as e:
            logger.error(f"Error logging search access: {e}")
    
    async def search_across_workspaces(
        self,
        query: str,
        user_id: str,
        organization_id: Optional[str] = None,
        limit: int = 10,
        **kwargs
    ) -> Dict[str, Any]:
        """Search across all accessible workspaces for a user"""
        
        try:
            # Get user's accessible workspaces
            workspaces = await self._get_user_workspaces(user_id, organization_id)
            
            if not workspaces:
                return {
                    "results": [],
                    "total": 0,
                    "message": "No accessible workspaces found"
                }
            
            # Search in each workspace
            all_results = []
            workspace_metadata = []
            
            for workspace in workspaces:
                workspace_results = await self.search_with_permissions(
                    query=query,
                    user_id=user_id,
                    workspace_id=workspace["workspace_id"],
                    limit=limit,
                    include_metadata=False,
                    **kwargs
                )
                
                # Add workspace context to results
                for result in workspace_results.get("results", []):
                    result["metadata"]["workspace_name"] = workspace["workspace_name"]
                    result["metadata"]["organization_id"] = workspace["organization_id"]
                
                all_results.extend(workspace_results.get("results", []))
                workspace_metadata.append({
                    "workspace_id": workspace["workspace_id"],
                    "workspace_name": workspace["workspace_name"],
                    "results_count": len(workspace_results.get("results", []))
                })
            
            # Merge and rank results across workspaces
            final_results = await self._merge_and_rank_cross_workspace_results(
                all_results, query, limit
            )
            
            return {
                "results": final_results,
                "total": len(final_results),
                "query": query,
                "workspaces_searched": workspace_metadata
            }
            
        except Exception as e:
            logger.error(f"Error in cross-workspace search: {e}")
            return {
                "results": [],
                "total": 0,
                "error": str(e)
            }
    
    async def _get_user_workspaces(
        self, user_id: str, organization_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get user's accessible workspaces"""
        
        try:
            query = """
                SELECT workspace_id, workspace_name, organization_id, role
                FROM user_workspace_access
                WHERE user_id = $1::UUID
            """
            params = [UUID(user_id)]
            
            if organization_id:
                query += " AND organization_id = $2::UUID"
                params.append(UUID(organization_id))
            
            query += " ORDER BY workspace_name"
            
            rows = await self.db.fetch(query, *params)
            
            return [
                {
                    "workspace_id": str(row["workspace_id"]),
                    "workspace_name": row["workspace_name"],
                    "organization_id": str(row["organization_id"]),
                    "role": row["role"]
                }
                for row in rows
            ]
            
        except Exception as e:
            logger.error(f"Error getting user workspaces: {e}")
            return []
    
    async def _merge_and_rank_cross_workspace_results(
        self,
        results: List[Dict[str, Any]],
        query: str,
        limit: int
    ) -> List[Dict[str, Any]]:
        """Merge and rank results from multiple workspaces"""
        
        # Sort by score across all workspaces
        results.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        # Remove duplicates and limit
        seen_content = set()
        unique_results = []
        
        for result in results:
            content_key = (
                result.get("content", "")[:100],  # First 100 chars
                result.get("metadata", {}).get("collection_name", "")
            )
            
            if content_key not in seen_content and len(unique_results) < limit:
                seen_content.add(content_key)
                result["rank"] = len(unique_results) + 1
                unique_results.append(result)
        
        return unique_results
