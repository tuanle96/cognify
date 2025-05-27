"""
Cognify API Keys Management
Allow users to create and manage API keys for authenticating with Cognify API
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import logging

from app.core.auth import get_current_user
from app.models.user import User
from app.services.auth.cognify_api_keys_service import (
    cognify_api_keys_service, APIKeyStatus
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Pydantic models
class CreateAPIKeyRequest(BaseModel):
    key_name: str = Field(..., min_length=1, max_length=255, description="Name for the API key")
    description: Optional[str] = Field(None, max_length=1000, description="Description of the API key")
    organization_id: Optional[str] = Field(None, description="Organization ID (optional)")
    workspace_id: Optional[str] = Field(None, description="Workspace ID (optional)")
    permissions: Optional[Dict[str, Any]] = Field(None, description="API permissions")
    rate_limits: Optional[Dict[str, Any]] = Field(None, description="Rate limiting configuration")
    expires_at: Optional[datetime] = Field(None, description="Expiration date (optional)")

class UpdateAPIKeyRequest(BaseModel):
    key_name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    permissions: Optional[Dict[str, Any]] = Field(None)
    rate_limits: Optional[Dict[str, Any]] = Field(None)
    status: Optional[APIKeyStatus] = Field(None)
    expires_at: Optional[datetime] = Field(None)

class APIKeyResponse(BaseModel):
    id: str
    key_name: str
    description: Optional[str]
    key_prefix: str
    permissions: Dict[str, Any]
    rate_limits: Dict[str, Any]
    status: str
    expires_at: Optional[str]
    last_used_at: Optional[str]
    usage_count: int
    created_at: str
    updated_at: str

class CreateAPIKeyResponse(BaseModel):
    id: str
    api_key: str  # Full key returned only once
    key_name: str
    key_prefix: str
    permissions: Dict[str, Any]
    rate_limits: Dict[str, Any]
    expires_at: Optional[str]
    created_at: str
    warning: str = "Save this API key securely. It will not be shown again."

class UsageStatisticsResponse(BaseModel):
    total_requests: int
    successful_requests: int
    failed_requests: int
    requests_today: int
    success_rate: float
    avg_response_time_ms: float
    last_request_at: Optional[str]
    top_endpoints: List[Dict[str, Any]]

# Create API key
@router.post("/", response_model=CreateAPIKeyResponse)
async def create_api_key(
    request: CreateAPIKeyRequest,
    http_request: Request,
    current_user: User = Depends(get_current_user)
):
    """Create a new Cognify API key"""
    
    try:
        await cognify_api_keys_service.initialize()
        
        # Get client IP
        client_ip = http_request.client.host if http_request.client else "127.0.0.1"
        
        result = await cognify_api_keys_service.create_api_key(
            user_id=str(current_user.id),
            key_name=request.key_name,
            description=request.description,
            organization_id=request.organization_id,
            workspace_id=request.workspace_id,
            permissions=request.permissions,
            rate_limits=request.rate_limits,
            expires_at=request.expires_at,
            created_from_ip=client_ip
        )
        
        return CreateAPIKeyResponse(**result)
        
    except Exception as e:
        logger.error(f"Error creating API key: {e}")
        raise HTTPException(status_code=500, detail="Failed to create API key")

# Get user's API keys
@router.get("/", response_model=List[APIKeyResponse])
async def get_api_keys(
    organization_id: Optional[str] = Query(None),
    workspace_id: Optional[str] = Query(None),
    include_inactive: bool = Query(False),
    current_user: User = Depends(get_current_user)
):
    """Get user's Cognify API keys"""
    
    try:
        await cognify_api_keys_service.initialize()
        
        keys = await cognify_api_keys_service.get_user_api_keys(
            user_id=str(current_user.id),
            organization_id=organization_id,
            workspace_id=workspace_id,
            include_inactive=include_inactive
        )
        
        return [APIKeyResponse(**key) for key in keys]
        
    except Exception as e:
        logger.error(f"Error getting API keys: {e}")
        raise HTTPException(status_code=500, detail="Failed to get API keys")

# Get specific API key
@router.get("/{key_id}", response_model=APIKeyResponse)
async def get_api_key(
    key_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get specific API key details"""
    
    try:
        await cognify_api_keys_service.initialize()
        
        keys = await cognify_api_keys_service.get_user_api_keys(
            user_id=str(current_user.id),
            include_inactive=True
        )
        
        key = next((k for k in keys if k["id"] == key_id), None)
        if not key:
            raise HTTPException(status_code=404, detail="API key not found")
        
        return APIKeyResponse(**key)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting API key: {e}")
        raise HTTPException(status_code=500, detail="Failed to get API key")

# Update API key
@router.put("/{key_id}")
async def update_api_key(
    key_id: str,
    request: UpdateAPIKeyRequest,
    current_user: User = Depends(get_current_user)
):
    """Update an API key"""
    
    try:
        await cognify_api_keys_service.initialize()
        
        success = await cognify_api_keys_service.update_api_key(
            key_id=key_id,
            user_id=str(current_user.id),
            key_name=request.key_name,
            description=request.description,
            permissions=request.permissions,
            rate_limits=request.rate_limits,
            status=request.status,
            expires_at=request.expires_at
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="API key not found or update failed")
        
        return {"message": "API key updated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating API key: {e}")
        raise HTTPException(status_code=500, detail="Failed to update API key")

# Delete API key
@router.delete("/{key_id}")
async def delete_api_key(
    key_id: str,
    current_user: User = Depends(get_current_user)
):
    """Delete an API key"""
    
    try:
        await cognify_api_keys_service.initialize()
        
        success = await cognify_api_keys_service.delete_api_key(
            key_id=key_id,
            user_id=str(current_user.id)
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="API key not found")
        
        return {"message": "API key deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting API key: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete API key")

# Get API key usage statistics
@router.get("/{key_id}/usage", response_model=UsageStatisticsResponse)
async def get_api_key_usage(
    key_id: str,
    days: int = Query(30, ge=1, le=365),
    current_user: User = Depends(get_current_user)
):
    """Get API key usage statistics"""
    
    try:
        await cognify_api_keys_service.initialize()
        
        stats = await cognify_api_keys_service.get_usage_statistics(
            api_key_id=key_id,
            user_id=str(current_user.id),
            days=days
        )
        
        return UsageStatisticsResponse(**stats)
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting API key usage: {e}")
        raise HTTPException(status_code=500, detail="Failed to get usage statistics")

# Revoke API key
@router.post("/{key_id}/revoke")
async def revoke_api_key(
    key_id: str,
    current_user: User = Depends(get_current_user)
):
    """Revoke an API key (set status to revoked)"""
    
    try:
        await cognify_api_keys_service.initialize()
        
        success = await cognify_api_keys_service.update_api_key(
            key_id=key_id,
            user_id=str(current_user.id),
            status=APIKeyStatus.REVOKED
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="API key not found")
        
        return {"message": "API key revoked successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error revoking API key: {e}")
        raise HTTPException(status_code=500, detail="Failed to revoke API key")

# Activate API key
@router.post("/{key_id}/activate")
async def activate_api_key(
    key_id: str,
    current_user: User = Depends(get_current_user)
):
    """Activate an API key (set status to active)"""
    
    try:
        await cognify_api_keys_service.initialize()
        
        success = await cognify_api_keys_service.update_api_key(
            key_id=key_id,
            user_id=str(current_user.id),
            status=APIKeyStatus.ACTIVE
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="API key not found")
        
        return {"message": "API key activated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error activating API key: {e}")
        raise HTTPException(status_code=500, detail="Failed to activate API key")

# Get API key permissions info
@router.get("/permissions/info")
async def get_permissions_info():
    """Get information about available API permissions"""
    
    return {
        "available_permissions": {
            "endpoints": {
                "description": "List of allowed API endpoints",
                "examples": ["*", "/api/v1/chat/*", "/api/v1/documents/*"],
                "default": ["*"]
            },
            "methods": {
                "description": "Allowed HTTP methods",
                "options": ["GET", "POST", "PUT", "DELETE", "PATCH"],
                "default": ["GET", "POST", "PUT", "DELETE"]
            },
            "scopes": {
                "description": "Permission scopes",
                "options": ["read", "write", "admin"],
                "default": ["read", "write"]
            }
        },
        "rate_limits": {
            "requests_per_minute": {
                "description": "Maximum requests per minute",
                "default": 60,
                "max": 1000
            },
            "requests_per_hour": {
                "description": "Maximum requests per hour",
                "default": 1000,
                "max": 10000
            },
            "requests_per_day": {
                "description": "Maximum requests per day",
                "default": 10000,
                "max": 100000
            }
        }
    }
