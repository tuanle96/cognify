"""
Collection management API endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Optional
import structlog

from app.api.dependencies import (
    get_vectordb_service,
    get_current_verified_user_from_db,
    get_collection_repository,
    get_document_repository,
    get_query_repository
)
from app.api.models.collections import (
    CollectionCreateRequest,
    CollectionResponse,
    CollectionListResponse,
    CollectionStatsResponse,
    CollectionUpdateRequest
)
from app.services.database.repositories import CollectionRepository, DocumentRepository, QueryRepository
from app.models.users import User
from app.models.collections import CollectionStatus, CollectionVisibility
from app.core.exceptions import ValidationError, ConflictError, NotFoundError

logger = structlog.get_logger(__name__)
router = APIRouter()


@router.post("/", response_model=CollectionResponse)
async def create_collection(
    request: CollectionCreateRequest,
    vectordb_service = Depends(get_vectordb_service),
    current_user: User = Depends(get_current_verified_user_from_db),
    collection_repo: CollectionRepository = Depends(get_collection_repository)
):
    """Create a new collection."""
    try:
        # Create collection using repository
        collection = await collection_repo.create_collection(
            name=request.name,
            description=request.description,
            owner_id=current_user.id,
            visibility=getattr(request, 'visibility', CollectionVisibility.PRIVATE),
            settings={
                "embedding_dimension": getattr(request, 'embedding_dimension', 384),
                "distance_metric": getattr(request, 'distance_metric', 'cosine'),
                "metadata_schema": getattr(request, 'metadata_schema', {})
            }
        )

        # Create collection in vector database
        try:
            await vectordb_service.create_collection(
                request.name,
                getattr(request, 'embedding_dimension', 384)
            )
        except Exception as e:
            # If vector DB creation fails, delete the database record
            await collection_repo.delete(collection.id)
            raise e

        logger.info(
            "Collection created",
            collection_name=request.name,
            collection_id=collection.id,
            user_id=current_user.id
        )

        # Handle metadata properly - convert to dict if it's not already
        metadata = getattr(collection, 'metadata', {})
        if hasattr(metadata, '__dict__'):
            metadata = {}  # Use empty dict if metadata is not a proper dict

        return CollectionResponse(
            collection_id=collection.id,
            name=collection.name,
            description=collection.description,
            visibility=collection.visibility,
            tags=getattr(collection, 'tags', []) or [],
            metadata=metadata,
            document_count=collection.document_count or 0,
            total_chunks=getattr(collection, 'total_chunks', 0),
            storage_size=f"{collection.total_size or 0} bytes",
            embedding_dimension=getattr(collection, 'embedding_dimension', 384),
            distance_metric=getattr(collection, 'distance_metric', 'cosine'),
            created_at=collection.created_at,
            updated_at=collection.updated_at,
            owner_id=collection.created_by,
            is_public=(collection.visibility == CollectionVisibility.PUBLIC)
        )

    except ConflictError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to create collection", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create collection"
        )


@router.get("/", response_model=CollectionListResponse)
async def list_collections(
    limit: int = 50,
    offset: int = 0,
    current_user: User = Depends(get_current_verified_user_from_db),
    collection_repo: CollectionRepository = Depends(get_collection_repository)
):
    """List user's collections."""
    try:
        # Get user's collections
        collections = await collection_repo.get_user_collections(
            user_id=current_user.id,
            offset=offset,
            limit=limit
        )

        # Convert to response format
        collection_responses = []
        for collection in collections:
            # Handle metadata properly - convert to dict if it's not already
            metadata = getattr(collection, 'metadata', {})
            if hasattr(metadata, '__dict__'):
                metadata = {}  # Use empty dict if metadata is not a proper dict

            collection_responses.append(CollectionResponse(
                collection_id=collection.id,
                name=collection.name,
                description=collection.description,
                visibility=collection.visibility,
                tags=getattr(collection, 'tags', []) or [],
                metadata=metadata,
                document_count=collection.document_count or 0,
                total_chunks=getattr(collection, 'total_chunks', 0),
                storage_size=f"{collection.total_size or 0} bytes",
                embedding_dimension=getattr(collection, 'embedding_dimension', 384),
                distance_metric=getattr(collection, 'distance_metric', 'cosine'),
                created_at=collection.created_at,
                updated_at=collection.updated_at,
                owner_id=collection.created_by,
                is_public=(collection.visibility == CollectionVisibility.PUBLIC)
            ))

        # Get total count
        total_count = await collection_repo.count(filters={"owner_id": current_user.id})

        # Calculate pagination info
        page = (offset // limit) + 1
        has_next = (offset + limit) < total_count

        return CollectionListResponse(
            collections=collection_responses,
            total=total_count,
            page=page,
            per_page=limit,
            has_next=has_next
        )

    except Exception as e:
        logger.error("Failed to list collections", error=str(e), user_id=current_user.id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list collections"
        )


@router.get("/{collection_name}", response_model=CollectionResponse)
async def get_collection(
    collection_name: str,
    current_user: User = Depends(get_current_verified_user_from_db),
    collection_repo: CollectionRepository = Depends(get_collection_repository)
):
    """Get collection details."""
    try:
        # Get collection by name and owner
        collection = await collection_repo.get_by_name_and_owner(
            collection_name, current_user.id
        )

        if not collection:
            # Check if user has read access to the collection
            collections = await collection_repo.get_user_collections(current_user.id)
            collection = next((c for c in collections if c.name == collection_name), None)

            if not collection:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Collection not found"
                )

            # Check read permission
            has_read_permission = await collection_repo.check_permission(
                collection.id, current_user.id, "read"
            )
            if not has_read_permission:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied"
                )

        return CollectionResponse(
            collection_id=collection.id,
            name=collection.name,
            description=collection.description,
            visibility=collection.visibility.value,
            tags=[],
            metadata={},
            document_count=collection.document_count,
            total_chunks=collection.chunk_count,
            storage_size=f"{collection.total_size} bytes",
            embedding_dimension=collection.embedding_dimension,
            distance_metric=collection.distance_metric,
            created_at=collection.created_at,
            updated_at=collection.updated_at,
            owner_id=collection.created_by,
            is_public=(collection.visibility.value == "public")
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get collection", error=str(e), collection_name=collection_name)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get collection"
        )


@router.put("/{collection_name}", response_model=CollectionResponse)
async def update_collection(
    collection_name: str,
    request: CollectionUpdateRequest,
    current_user: User = Depends(get_current_verified_user_from_db),
    collection_repo: CollectionRepository = Depends(get_collection_repository)
):
    """Update collection metadata."""
    try:
        # Get collection and verify ownership
        collection = await collection_repo.get_by_name_and_owner(
            collection_name, current_user.id
        )

        if not collection:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Collection not found"
            )

        # Update collection using repository
        update_data = request.dict(exclude_unset=True)
        updated_collection = await collection_repo.update(collection.id, update_data)

        logger.info(
            "Collection updated",
            collection_name=collection_name,
            collection_id=collection.id,
            user_id=current_user.id
        )

        return CollectionResponse(
            collection_id=updated_collection.id,
            name=updated_collection.name,
            description=updated_collection.description,
            visibility=updated_collection.visibility.value,
            tags=[],
            metadata={},
            document_count=updated_collection.document_count,
            total_chunks=updated_collection.chunk_count,
            storage_size=f"{updated_collection.total_size} bytes",
            embedding_dimension=updated_collection.embedding_dimension,
            distance_metric=updated_collection.distance_metric,
            created_at=updated_collection.created_at,
            updated_at=updated_collection.updated_at,
            owner_id=updated_collection.created_by,
            is_public=(updated_collection.visibility.value == "public")
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update collection", error=str(e), collection_name=collection_name)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update collection"
        )


@router.delete("/{collection_name}")
async def delete_collection(
    collection_name: str,
    vectordb_service = Depends(get_vectordb_service),
    current_user: User = Depends(get_current_verified_user_from_db),
    collection_repo: CollectionRepository = Depends(get_collection_repository)
):
    """Delete a collection."""
    try:
        # Get collection and verify ownership
        collection = await collection_repo.get_by_name_and_owner(
            collection_name, current_user.id
        )

        if not collection:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Collection not found"
            )

        # Delete from vector database first
        try:
            await vectordb_service.delete_collection(collection_name)
        except Exception as e:
            logger.warning("Failed to delete from vector database", error=str(e))
            # Continue with database deletion even if vector DB fails

        # Delete from database (soft delete)
        success = await collection_repo.delete(collection.id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete collection"
            )

        logger.info(
            "Collection deleted",
            collection_name=collection_name,
            collection_id=collection.id,
            user_id=current_user.id
        )

        return {"message": f"Collection '{collection_name}' deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete collection", error=str(e), collection_name=collection_name)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete collection"
        )


@router.get("/{collection_name}/stats", response_model=CollectionStatsResponse)
async def get_collection_stats(
    collection_name: str,
    vectordb_service = Depends(get_vectordb_service),
    current_user: User = Depends(get_current_verified_user_from_db),
    collection_repo: CollectionRepository = Depends(get_collection_repository),
    document_repo: DocumentRepository = Depends(get_document_repository),
    query_repo: QueryRepository = Depends(get_query_repository)
):
    """Get collection statistics."""
    try:
        # Get collection and verify access
        collection = await collection_repo.get_by_name_and_owner(
            collection_name, current_user.id
        )

        if not collection:
            # Check if user has read access
            collections = await collection_repo.get_user_collections(current_user.id)
            collection = next((c for c in collections if c.name == collection_name), None)

            if not collection:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Collection not found"
                )

            # Check read permission
            has_read_permission = await collection_repo.check_permission(
                collection.id, current_user.id, "read"
            )
            if not has_read_permission:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied"
                )

        # Get basic stats (mock implementation for now)
        def format_size(size_bytes):
            """Format bytes to human readable string."""
            for unit in ['B', 'KB', 'MB', 'GB']:
                if size_bytes < 1024.0:
                    return f"{size_bytes:.1f} {unit}"
                size_bytes /= 1024.0
            return f"{size_bytes:.1f} TB"

        stats = CollectionStatsResponse(
            collection_id=collection.id,
            document_count=collection.document_count,
            total_chunks=collection.chunk_count or 0,
            total_tokens=None,
            storage_size_bytes=collection.total_size or 0,
            storage_size_human=format_size(collection.total_size or 0),
            avg_document_size=None,
            document_types={"text": collection.document_count},
            language_distribution=None,
            recent_activity=[],
            created_at=collection.created_at,
            last_updated=collection.updated_at
        )

        return stats

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get collection stats", error=str(e), collection_name=collection_name)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get collection stats"
        )


