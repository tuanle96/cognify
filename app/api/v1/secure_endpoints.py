"""
Secure API Endpoints with Multi-Tenant Isolation
CRITICAL: Security-enhanced endpoints with proper tenant isolation
"""

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from typing import List, Dict, Any, Optional
from uuid import UUID
import logging

from app.core.auth import get_current_user
from app.models.user import User
from app.models.workspace import Workspace
from app.services.search.permission_aware_search import PermissionAwareSearchService
from app.services.vector.tenant_isolated_service import TenantIsolatedVectorService
from app.core.database import get_database

logger = logging.getLogger(__name__)
router = APIRouter()

# Dependency to get current workspace with permission validation
async def get_current_workspace(
    workspace_id: str = Path(..., description="Workspace ID"),
    current_user: User = Depends(get_current_user),
    db = Depends(get_database)
) -> Workspace:
    """Get current workspace with permission validation"""
    
    try:
        # Validate workspace access
        query = """
            SELECT w.*, wm.role, wm.can_read, wm.can_write, wm.can_delete, wm.can_share, wm.can_admin
            FROM workspaces w
            JOIN workspace_members wm ON w.id = wm.workspace_id
            WHERE w.id = $1::UUID 
            AND wm.user_id = $2::UUID 
            AND wm.status = 'active'
            AND w.deleted_at IS NULL
        """
        
        row = await db.fetchrow(query, UUID(workspace_id), current_user.id)
        
        if not row:
            raise HTTPException(
                status_code=403, 
                detail=f"Access denied to workspace {workspace_id}"
            )
        
        # Create workspace object with permissions
        workspace = Workspace(
            id=row["id"],
            organization_id=row["organization_id"],
            name=row["name"],
            slug=row["slug"],
            description=row["description"],
            visibility=row["visibility"],
            settings=row["settings"] or {},
            created_at=row["created_at"],
            updated_at=row["updated_at"]
        )
        
        # Add user permissions to workspace object
        workspace.user_role = row["role"]
        workspace.user_permissions = {
            "can_read": row["can_read"],
            "can_write": row["can_write"],
            "can_delete": row["can_delete"],
            "can_share": row["can_share"],
            "can_admin": row["can_admin"]
        }
        
        return workspace
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting workspace {workspace_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Secure search endpoint with tenant isolation
@router.post("/workspaces/{workspace_id}/search")
async def secure_search_in_workspace(
    workspace_id: str,
    query: str = Query(..., description="Search query"),
    collection_id: Optional[str] = Query(None, description="Specific collection ID"),
    limit: int = Query(10, ge=1, le=100, description="Maximum results"),
    include_metadata: bool = Query(True, description="Include search metadata"),
    current_user: User = Depends(get_current_user),
    workspace: Workspace = Depends(get_current_workspace),
    search_service: PermissionAwareSearchService = Depends()
):
    """Secure search within workspace with proper tenant isolation"""
    
    try:
        # Validate read permission
        if not workspace.user_permissions.get("can_read", False):
            raise HTTPException(403, "Read permission required")
        
        # Perform permission-aware search
        results = await search_service.search_with_permissions(
            query=query,
            user_id=str(current_user.id),
            workspace_id=workspace_id,
            collection_id=collection_id,
            limit=limit,
            include_metadata=include_metadata
        )
        
        # Add audit information
        results["audit"] = {
            "user_id": str(current_user.id),
            "workspace_id": workspace_id,
            "user_role": workspace.user_role,
            "search_timestamp": results.get("metadata", {}).get("search_timestamp")
        }
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in secure search: {e}")
        raise HTTPException(status_code=500, detail="Search failed")

# Secure cross-workspace search
@router.post("/organizations/{organization_id}/search")
async def secure_cross_workspace_search(
    organization_id: str,
    query: str = Query(..., description="Search query"),
    limit: int = Query(10, ge=1, le=100, description="Maximum results"),
    current_user: User = Depends(get_current_user),
    search_service: PermissionAwareSearchService = Depends(),
    db = Depends(get_database)
):
    """Secure search across all accessible workspaces in organization"""
    
    try:
        # Validate organization membership
        org_member = await db.fetchrow("""
            SELECT role, status FROM organization_members 
            WHERE organization_id = $1::UUID AND user_id = $2::UUID
        """, UUID(organization_id), current_user.id)
        
        if not org_member or org_member["status"] != "active":
            raise HTTPException(403, "Access denied to organization")
        
        # Perform cross-workspace search
        results = await search_service.search_across_workspaces(
            query=query,
            user_id=str(current_user.id),
            organization_id=organization_id,
            limit=limit
        )
        
        # Add audit information
        results["audit"] = {
            "user_id": str(current_user.id),
            "organization_id": organization_id,
            "user_role": org_member["role"],
            "search_type": "cross_workspace"
        }
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in cross-workspace search: {e}")
        raise HTTPException(status_code=500, detail="Cross-workspace search failed")

# Secure document upload with tenant isolation
@router.post("/workspaces/{workspace_id}/documents/upload")
async def secure_upload_document(
    workspace_id: str,
    collection_id: str = Query(..., description="Collection ID"),
    current_user: User = Depends(get_current_user),
    workspace: Workspace = Depends(get_current_workspace),
    # file upload logic here
):
    """Secure document upload with tenant isolation"""
    
    try:
        # Validate write permission
        if not workspace.user_permissions.get("can_write", False):
            raise HTTPException(403, "Write permission required")
        
        # Validate collection belongs to workspace
        db = await get_database()
        collection = await db.fetchrow("""
            SELECT id, name, workspace_id FROM collections 
            WHERE id = $1::UUID AND workspace_id = $2::UUID
        """, UUID(collection_id), UUID(workspace_id))
        
        if not collection:
            raise HTTPException(404, "Collection not found in workspace")
        
        # TODO: Implement secure file upload with tenant metadata
        # This would include:
        # 1. File validation and scanning
        # 2. Tenant-aware storage
        # 3. Metadata tagging with workspace context
        # 4. Vector embedding with tenant isolation
        
        return {
            "message": "Document upload endpoint - implementation needed",
            "workspace_id": workspace_id,
            "collection_id": collection_id,
            "user_permissions": workspace.user_permissions
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in secure upload: {e}")
        raise HTTPException(status_code=500, detail="Upload failed")

# Get user's accessible workspaces
@router.get("/user/workspaces")
async def get_user_workspaces(
    organization_id: Optional[str] = Query(None, description="Filter by organization"),
    current_user: User = Depends(get_current_user),
    db = Depends(get_database)
):
    """Get user's accessible workspaces with permissions"""
    
    try:
        query = """
            SELECT 
                w.id, w.organization_id, w.name, w.slug, w.description, w.visibility,
                wm.role, wm.can_read, wm.can_write, wm.can_delete, wm.can_share, wm.can_admin,
                wm.last_access_at, wm.access_count,
                o.name as organization_name, o.slug as organization_slug
            FROM workspaces w
            JOIN workspace_members wm ON w.id = wm.workspace_id
            JOIN organizations o ON w.organization_id = o.id
            WHERE wm.user_id = $1::UUID 
            AND wm.status = 'active'
            AND w.deleted_at IS NULL
        """
        
        params = [current_user.id]
        
        if organization_id:
            query += " AND w.organization_id = $2::UUID"
            params.append(UUID(organization_id))
        
        query += " ORDER BY wm.last_access_at DESC NULLS LAST, w.name"
        
        rows = await db.fetch(query, *params)
        
        workspaces = []
        for row in rows:
            workspaces.append({
                "id": str(row["id"]),
                "organization_id": str(row["organization_id"]),
                "organization_name": row["organization_name"],
                "organization_slug": row["organization_slug"],
                "name": row["name"],
                "slug": row["slug"],
                "description": row["description"],
                "visibility": row["visibility"],
                "user_role": row["role"],
                "permissions": {
                    "can_read": row["can_read"],
                    "can_write": row["can_write"],
                    "can_delete": row["can_delete"],
                    "can_share": row["can_share"],
                    "can_admin": row["can_admin"]
                },
                "last_access_at": row["last_access_at"].isoformat() if row["last_access_at"] else None,
                "access_count": row["access_count"]
            })
        
        return {
            "workspaces": workspaces,
            "total": len(workspaces),
            "user_id": str(current_user.id)
        }
        
    except Exception as e:
        logger.error(f"Error getting user workspaces: {e}")
        raise HTTPException(status_code=500, detail="Failed to get workspaces")

# Get workspace collections with permissions
@router.get("/workspaces/{workspace_id}/collections")
async def get_workspace_collections(
    workspace_id: str,
    current_user: User = Depends(get_current_user),
    workspace: Workspace = Depends(get_current_workspace),
    db = Depends(get_database)
):
    """Get collections in workspace with user permissions"""
    
    try:
        # Validate read permission
        if not workspace.user_permissions.get("can_read", False):
            raise HTTPException(403, "Read permission required")
        
        query = """
            SELECT 
                c.id, c.name, c.description, c.visibility, c.tenant_id,
                c.owner_id, c.created_at, c.updated_at, c.metadata,
                u.email as owner_email,
                (c.owner_id = $2::UUID) as is_owner,
                CASE 
                    WHEN c.visibility = 'public' THEN true
                    WHEN c.visibility = 'internal' THEN true
                    WHEN c.owner_id = $2::UUID THEN true
                    WHEN EXISTS (
                        SELECT 1 FROM collection_members cm 
                        WHERE cm.collection_id = c.id AND cm.user_id = $2::UUID
                    ) THEN true
                    ELSE false
                END as has_access
            FROM collections c
            LEFT JOIN users u ON c.owner_id = u.id
            WHERE c.workspace_id = $1::UUID
            ORDER BY c.created_at DESC
        """
        
        rows = await db.fetch(query, UUID(workspace_id), current_user.id)
        
        collections = []
        for row in rows:
            if row["has_access"]:  # Only include accessible collections
                collections.append({
                    "id": str(row["id"]),
                    "name": row["name"],
                    "description": row["description"],
                    "visibility": row["visibility"],
                    "tenant_id": row["tenant_id"],
                    "owner_id": str(row["owner_id"]) if row["owner_id"] else None,
                    "owner_email": row["owner_email"],
                    "is_owner": row["is_owner"],
                    "created_at": row["created_at"].isoformat(),
                    "updated_at": row["updated_at"].isoformat(),
                    "metadata": row["metadata"] or {}
                })
        
        return {
            "collections": collections,
            "total": len(collections),
            "workspace_id": workspace_id,
            "user_permissions": workspace.user_permissions
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting workspace collections: {e}")
        raise HTTPException(status_code=500, detail="Failed to get collections")

# Security audit endpoint
@router.get("/workspaces/{workspace_id}/audit")
async def get_workspace_audit_log(
    workspace_id: str,
    limit: int = Query(100, ge=1, le=1000),
    action: Optional[str] = Query(None, description="Filter by action"),
    current_user: User = Depends(get_current_user),
    workspace: Workspace = Depends(get_current_workspace),
    db = Depends(get_database)
):
    """Get workspace audit log (admin only)"""
    
    try:
        # Validate admin permission
        if not workspace.user_permissions.get("can_admin", False):
            raise HTTPException(403, "Admin permission required")
        
        query = """
            SELECT 
                da.id, da.user_id, da.action, da.resource_type, da.resource_id,
                da.ip_address, da.user_agent, da.metadata, da.created_at,
                u.email as user_email
            FROM data_access_audit da
            LEFT JOIN users u ON da.user_id = u.id
            WHERE da.workspace_id = $1::UUID
        """
        
        params = [UUID(workspace_id)]
        
        if action:
            query += " AND da.action = $2"
            params.append(action)
        
        query += " ORDER BY da.created_at DESC LIMIT $" + str(len(params) + 1)
        params.append(limit)
        
        rows = await db.fetch(query, *params)
        
        audit_entries = []
        for row in rows:
            audit_entries.append({
                "id": str(row["id"]),
                "user_id": str(row["user_id"]),
                "user_email": row["user_email"],
                "action": row["action"],
                "resource_type": row["resource_type"],
                "resource_id": str(row["resource_id"]),
                "ip_address": str(row["ip_address"]) if row["ip_address"] else None,
                "user_agent": row["user_agent"],
                "metadata": row["metadata"] or {},
                "created_at": row["created_at"].isoformat()
            })
        
        return {
            "audit_entries": audit_entries,
            "total": len(audit_entries),
            "workspace_id": workspace_id,
            "requested_by": str(current_user.id)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting audit log: {e}")
        raise HTTPException(status_code=500, detail="Failed to get audit log")
