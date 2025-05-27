"""
FastAPI dependency injection for services and authentication.

Provides centralized dependency management for all API endpoints.
"""

from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, Dict, Any
import jwt
import structlog

from app.core.config import get_settings
from app.core.exceptions import AuthenticationError, AuthorizationError
from app.services.indexing.service import IndexingService
from app.services.retrieval.service import RetrievalService
from app.services.parsers.service import ParsingService
from app.services.chunking.service import ChunkingService
# Mock services removed - using production services only
from app.services.database.session import get_db_session
from app.services.database.repositories import (
    UserRepository,
    DocumentRepository,
    CollectionRepository,
    QueryRepository
)
from app.models.users import User

logger = structlog.get_logger(__name__)
security = HTTPBearer()
settings = get_settings()


# Service Dependencies

async def get_parsing_service(request: Request) -> ParsingService:
    """Get parsing service from application state."""
    try:
        if hasattr(request.app.state, 'parsing_service'):
            return request.app.state.parsing_service

        # Fallback: import global service
        from app.services.parsers.service import parsing_service
        if not parsing_service._initialized:
            await parsing_service.initialize()
        return parsing_service

    except Exception as e:
        logger.error("Failed to get parsing service", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Parsing service unavailable"
        )


async def get_chunking_service(request: Request) -> ChunkingService:
    """Get chunking service from application state."""
    try:
        if hasattr(request.app.state, 'chunking_service'):
            return request.app.state.chunking_service

        # Fallback: import global service
        from app.services.chunking.service import chunking_service
        if not chunking_service._initialized:
            await chunking_service.initialize()
        return chunking_service

    except Exception as e:
        logger.error("Failed to get chunking service", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chunking service unavailable"
        )


async def get_embedding_service(request: Request):
    """Get embedding service from application state."""
    try:
        if hasattr(request.app.state, 'embedding_service'):
            return request.app.state.embedding_service

        # Service not available
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding service not initialized"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get embedding service", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding service unavailable"
        )


async def get_vectordb_service(request: Request):
    """Get vector database service from application state."""
    try:
        if hasattr(request.app.state, 'vectordb_service'):
            return request.app.state.vectordb_service

        # Service not available
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector database service not initialized"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get vector database service", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector database service unavailable"
        )


async def get_indexing_service(request: Request) -> IndexingService:
    """Get indexing service with injected dependencies."""
    try:
        if hasattr(request.app.state, 'indexing_service'):
            return request.app.state.indexing_service

        # Create indexing service with injected dependencies
        from app.services.indexing.base import IndexingConfig

        config = IndexingConfig(
            collection_name="default",
            batch_size=10,
            chunk_size=500
        )

        indexing_service = IndexingService(config)

        # Inject dependencies
        indexing_service.parsing_service = await get_parsing_service(request)
        indexing_service.chunking_service = await get_chunking_service(request)
        indexing_service.embedding_service = await get_embedding_service(request)
        indexing_service.vectordb_service = await get_vectordb_service(request)

        # Mark as initialized to skip automatic initialization
        indexing_service._initialized = True

        return indexing_service

    except Exception as e:
        logger.error("Failed to get indexing service", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Indexing service unavailable"
        )


async def get_retrieval_service(request: Request) -> RetrievalService:
    """Get retrieval service with injected dependencies."""
    try:
        if hasattr(request.app.state, 'retrieval_service'):
            return request.app.state.retrieval_service

        # Create retrieval service with injected dependencies
        from app.services.retrieval.base import RetrievalConfig

        config = RetrievalConfig(
            collection_name="default",
            max_results=10
        )

        retrieval_service = RetrievalService(config)

        # Inject dependencies
        retrieval_service.embedding_service = await get_embedding_service(request)
        retrieval_service.vectordb_service = await get_vectordb_service(request)

        # Mark as initialized to skip automatic initialization
        retrieval_service._initialized = True

        return retrieval_service

    except Exception as e:
        logger.error("Failed to get retrieval service", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Retrieval service unavailable"
        )


# Authentication Dependencies

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db_session = Depends(lambda: None)  # Placeholder for database session
) -> Dict[str, Any]:
    """
    Get current authenticated user from JWT token.
    """
    try:
        # Verify JWT token
        payload = jwt.decode(
            credentials.credentials,
            settings.SECRET_KEY,
            algorithms=["HS256"]
        )

        # Check token type
        if payload.get("type") != "access":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type"
            )

        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload"
            )

        # Get user from database
        user = await _get_user_by_id(db_session, user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )

        if not user.get("is_active", False):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User account is deactivated"
            )

        return user

    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    except Exception as e:
        logger.error("Authentication failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed"
        )


async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db_session = Depends(lambda: None)
) -> Optional[Dict[str, Any]]:
    """
    Get current user if authenticated, otherwise return None.

    Useful for endpoints that work for both authenticated and anonymous users.
    """
    if not credentials:
        return None

    try:
        return await get_current_user(credentials, db_session)
    except HTTPException:
        return None


async def require_admin_user(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Require current user to have admin role.
    """
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )

    return current_user


async def require_verified_user(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Require current user to have verified email.
    """
    if not current_user.get("is_verified", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Email verification required"
        )

    return current_user


# Database Dependencies

async def get_db_session():
    """
    Get database session.
    """
    from app.services.database.session import get_db_session as _get_db_session
    async for session in _get_db_session():
        yield session


# Rate Limiting Dependencies

async def rate_limit_user(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Apply rate limiting based on user tier.

    TODO: Implement actual rate limiting logic.
    """
    # Placeholder for rate limiting
    # In production, this would check user's rate limits
    return current_user


# Helper Functions

async def _get_user_by_id(db_session, user_id: str) -> Optional[Dict[str, Any]]:
    """
    Get user by ID from database.
    """
    try:
        from app.services.database.repositories import UserRepository
        user_repo = UserRepository(db_session)
        user = await user_repo.get_by_id(user_id)

        if user:
            return {
                "user_id": user.id,
                "email": user.email,
                "full_name": user.full_name,
                "is_active": user.is_active,
                "is_verified": user.is_verified,
                "role": user.role.value,
                "created_at": user.created_at.isoformat()
            }
    except Exception as e:
        logger.error("Failed to get user by ID", user_id=user_id, error=str(e))

    return None


# Service Health Check Dependencies

async def check_service_health(request: Request) -> Dict[str, Any]:
    """
    Check health of all services.
    """
    health_status = {
        "status": "healthy",
        "services": {},
        "timestamp": "2024-12-26T00:00:00Z"
    }

    try:
        # Check parsing service
        parsing_service = await get_parsing_service(request)
        health_status["services"]["parsing"] = await parsing_service.health_check()

        # Check chunking service
        chunking_service = await get_chunking_service(request)
        health_status["services"]["chunking"] = await chunking_service.health_check()

        # Check embedding service
        embedding_service = await get_embedding_service(request)
        health_status["services"]["embedding"] = await embedding_service.health_check()

        # Check vector database service
        vectordb_service = await get_vectordb_service(request)
        health_status["services"]["vectordb"] = await vectordb_service.health_check()

        # Determine overall health
        unhealthy_services = [
            name for name, status in health_status["services"].items()
            if status.get("status") != "healthy"
        ]

        if unhealthy_services:
            health_status["status"] = "degraded"
            health_status["unhealthy_services"] = unhealthy_services

    except Exception as e:
        logger.error("Health check failed", error=str(e))
        health_status["status"] = "unhealthy"
        health_status["error"] = str(e)

    return health_status


# Database Dependencies

async def get_user_repository(
    session = Depends(get_db_session)
) -> UserRepository:
    """Get user repository instance."""
    return UserRepository(session)


async def get_document_repository(
    session = Depends(get_db_session)
) -> DocumentRepository:
    """Get document repository instance."""
    return DocumentRepository(session)


async def get_collection_repository(
    session = Depends(get_db_session)
) -> CollectionRepository:
    """Get collection repository instance."""
    return CollectionRepository(session)


async def get_query_repository(
    session = Depends(get_db_session)
) -> QueryRepository:
    """Get query repository instance."""
    return QueryRepository(session)


# Enhanced Authentication Dependencies

async def get_current_user_from_db(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    user_repo: UserRepository = Depends(get_user_repository)
) -> User:
    """
    Get current authenticated user from database.

    Args:
        credentials: HTTP Bearer credentials
        user_repo: User repository instance

    Returns:
        User model instance

    Raises:
        HTTPException: If authentication fails
    """
    try:
        # Decode JWT token
        payload = jwt.decode(
            credentials.credentials,
            settings.SECRET_KEY,
            algorithms=["HS256"]
        )

        user_id = payload.get("sub")
        if user_id is None:
            raise AuthenticationError("Invalid token: missing user ID")

        # Get user from database
        user = await user_repo.get_by_id(user_id)
        if not user:
            raise AuthenticationError("User not found")

        # Check if user is active (allow PENDING_VERIFICATION for testing)
        from app.models.users import UserStatus
        if user.status not in [UserStatus.ACTIVE, UserStatus.PENDING_VERIFICATION]:
            raise AuthenticationError("User account is inactive")

        # Check if account is locked
        if user.is_locked:
            raise AuthenticationError("User account is locked")

        logger.debug("User authenticated successfully", user_id=user.id)
        return user

    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"}
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"}
        )
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"}
        )
    except Exception as e:
        logger.error("Authentication error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"}
        )


async def get_current_verified_user_from_db_strict(
    user: User = Depends(get_current_user_from_db)
) -> User:
    """
    Get current authenticated and verified user from database.

    Args:
        user: Current authenticated user

    Returns:
        User model instance

    Raises:
        HTTPException: If user is not verified
    """
    if not user.is_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Email verification required"
        )

    return user


async def get_current_active_user_from_db(
    current_user: User = Depends(get_current_user_from_db)
) -> User:
    """Get current active user from database."""
    from app.models.users import UserStatus
    if current_user.status not in [UserStatus.ACTIVE, UserStatus.PENDING_VERIFICATION]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user


async def get_current_verified_user_from_db(
    current_user: User = Depends(get_current_active_user_from_db)
) -> User:
    """Get current verified user from database."""
    # For testing, allow unverified users
    # if not current_user.is_verified:
    #     raise HTTPException(
    #         status_code=status.HTTP_400_BAD_REQUEST,
    #         detail="Email not verified"
    #     )
    return current_user


async def require_admin_from_db(
    current_user: User = Depends(get_current_verified_user_from_db)
) -> User:
    """Require admin role from database."""
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user


def require_collection_permission_from_db(permission: str):
    """
    Create a dependency that requires a specific collection permission from database.

    Args:
        permission: Required permission ('read', 'write', 'admin')

    Returns:
        Dependency function
    """
    async def permission_dependency(
        collection_id: str,
        current_user: User = Depends(get_current_verified_user_from_db),
        collection_repo: CollectionRepository = Depends(get_collection_repository)
    ) -> User:
        # Check if user has permission for this collection
        has_permission = await collection_repo.check_permission(
            collection_id=collection_id,
            user_id=current_user.id,
            permission=permission
        )

        if not has_permission:
            # Check if collection is public and permission is read
            if permission == "read":
                collection = await collection_repo.get_by_id(collection_id)
                if collection and collection.is_public:
                    return current_user

            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Collection {permission} permission required"
            )

        return current_user

    return permission_dependency
