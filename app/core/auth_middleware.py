"""
Enhanced Authentication Middleware
Support both JWT tokens and Cognify API keys for authentication
"""

from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, Union
import logging
import asyncio
from datetime import datetime

from app.core.auth import get_current_user_from_token
from app.models.user import User
from app.services.auth.cognify_api_keys_service import cognify_api_keys_service

logger = logging.getLogger(__name__)

# Security scheme for Bearer tokens
security = HTTPBearer(auto_error=False)

class AuthenticationResult:
    """Result of authentication attempt"""

    def __init__(self,
                 user: Optional[User] = None,
                 api_key_id: Optional[str] = None,
                 auth_type: str = "none",
                 organization_id: Optional[str] = None,
                 workspace_id: Optional[str] = None,
                 permissions: Optional[dict] = None,
                 rate_limits: Optional[dict] = None):
        self.user = user
        self.api_key_id = api_key_id
        self.auth_type = auth_type  # "jwt", "api_key", or "none"
        self.organization_id = organization_id
        self.workspace_id = workspace_id
        self.permissions = permissions
        self.rate_limits = rate_limits

    @property
    def is_authenticated(self) -> bool:
        return self.user is not None

    @property
    def user_id(self) -> Optional[str]:
        return str(self.user.id) if self.user else None

async def authenticate_request(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> AuthenticationResult:
    """
    Authenticate request using either JWT token or API key

    Authentication methods (in order of priority):
    1. Bearer token (JWT)
    2. API key in Authorization header (Bearer api_key)
    3. API key in X-API-Key header
    4. API key in query parameter
    """

    try:
        # Method 1: Try JWT Bearer token first
        if credentials and credentials.scheme.lower() == "bearer":
            token = credentials.credentials

            # Check if it's a JWT token (starts with 'eyJ') or API key (starts with 'cog_')
            if token.startswith('eyJ'):
                # JWT token
                try:
                    user = await get_current_user_from_token(token)
                    if user:
                        return AuthenticationResult(
                            user=user,
                            auth_type="jwt"
                        )
                except Exception as e:
                    logger.debug(f"JWT authentication failed: {e}")

            elif token.startswith('cog_'):
                # Cognify API key
                auth_result = await authenticate_api_key(token, request)
                if auth_result.is_authenticated:
                    return auth_result

        # Method 2: Check X-API-Key header
        api_key = request.headers.get("X-API-Key")
        if api_key and api_key.startswith('cog_'):
            auth_result = await authenticate_api_key(api_key, request)
            if auth_result.is_authenticated:
                return auth_result

        # Method 3: Check query parameter
        api_key = request.query_params.get("api_key")
        if api_key and api_key.startswith('cog_'):
            auth_result = await authenticate_api_key(api_key, request)
            if auth_result.is_authenticated:
                return auth_result

        # No valid authentication found
        return AuthenticationResult()

    except Exception as e:
        logger.error(f"Authentication error: {e}")
        return AuthenticationResult()

async def authenticate_api_key(api_key: str, request: Request) -> AuthenticationResult:
    """Authenticate using Cognify API key"""

    try:
        await cognify_api_keys_service.initialize()

        # Authenticate the API key
        auth_data = await cognify_api_keys_service.authenticate_api_key(api_key)

        if not auth_data:
            return AuthenticationResult()

        # Check rate limits
        endpoint = str(request.url.path)
        rate_limit_ok = await cognify_api_keys_service.check_rate_limit(
            auth_data["api_key_id"],
            endpoint
        )

        if not rate_limit_ok:
            logger.warning(f"Rate limit exceeded for API key {auth_data['api_key_id']}")
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded"
            )

        # Get user object (you'll need to implement this)
        user = await get_user_by_id(auth_data["user_id"])

        if not user:
            return AuthenticationResult()

        # Track API key usage (async, don't wait)
        asyncio.create_task(track_api_key_usage(
            api_key_id=auth_data["api_key_id"],
            user_id=auth_data["user_id"],
            request=request,
            organization_id=auth_data.get("organization_id"),
            workspace_id=auth_data.get("workspace_id")
        ))

        return AuthenticationResult(
            user=user,
            api_key_id=auth_data["api_key_id"],
            auth_type="api_key",
            organization_id=auth_data.get("organization_id"),
            workspace_id=auth_data.get("workspace_id"),
            permissions=auth_data.get("permissions"),
            rate_limits=auth_data.get("rate_limits")
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"API key authentication error: {e}")
        return AuthenticationResult()

async def track_api_key_usage(
    api_key_id: str,
    user_id: str,
    request: Request,
    organization_id: Optional[str] = None,
    workspace_id: Optional[str] = None
):
    """Track API key usage (async background task)"""

    try:
        # Extract request details
        endpoint = str(request.url.path)
        method = request.method
        user_agent = request.headers.get("user-agent", "")
        ip_address = request.client.host if request.client else "127.0.0.1"

        # Track usage
        await cognify_api_keys_service.track_usage(
            api_key_id=api_key_id,
            user_id=user_id,
            endpoint=endpoint,
            method=method,
            status_code=200,  # Will be updated by response middleware
            user_agent=user_agent,
            ip_address=ip_address,
            organization_id=organization_id,
            workspace_id=workspace_id
        )

    except Exception as e:
        logger.error(f"Error tracking API key usage: {e}")

async def get_user_by_id(user_id: str) -> Optional[User]:
    """Get user by ID (implement this based on your user service)"""

    try:
        # TODO: Implement user lookup by ID
        # This should use your existing user service/repository
        from app.services.user.user_service import user_service
        return await user_service.get_user_by_id(user_id)
    except Exception as e:
        logger.error(f"Error getting user by ID: {e}")
        return None

# Dependency functions for FastAPI

async def get_current_user_or_api_key(
    auth_result: AuthenticationResult = Depends(authenticate_request)
) -> AuthenticationResult:
    """Get current user from either JWT or API key authentication"""

    if not auth_result.is_authenticated:
        raise HTTPException(
            status_code=401,
            detail="Authentication required"
        )

    return auth_result

async def get_current_user_enhanced(
    auth_result: AuthenticationResult = Depends(get_current_user_or_api_key)
) -> User:
    """Get current user (compatible with existing get_current_user)"""

    return auth_result.user

async def require_api_key_auth(
    auth_result: AuthenticationResult = Depends(authenticate_request)
) -> AuthenticationResult:
    """Require API key authentication specifically"""

    if not auth_result.is_authenticated:
        raise HTTPException(
            status_code=401,
            detail="API key authentication required"
        )

    if auth_result.auth_type != "api_key":
        raise HTTPException(
            status_code=401,
            detail="API key authentication required"
        )

    return auth_result

async def require_jwt_auth(
    auth_result: AuthenticationResult = Depends(authenticate_request)
) -> AuthenticationResult:
    """Require JWT authentication specifically"""

    if not auth_result.is_authenticated:
        raise HTTPException(
            status_code=401,
            detail="JWT authentication required"
        )

    if auth_result.auth_type != "jwt":
        raise HTTPException(
            status_code=401,
            detail="JWT authentication required"
        )

    return auth_result

# Middleware for response tracking
class APIKeyResponseMiddleware:
    """Middleware to track API key response details"""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Store original send
        original_send = send
        response_status = 200
        response_size = 0

        async def wrapped_send(message):
            nonlocal response_status, response_size

            if message["type"] == "http.response.start":
                response_status = message["status"]
            elif message["type"] == "http.response.body":
                if "body" in message:
                    response_size += len(message["body"])

            await original_send(message)

        # Process request
        await self.app(scope, receive, wrapped_send)

        # Update API key usage with response details if needed
        # This would require storing the API key ID in request state
        # and updating the usage record after response is sent
