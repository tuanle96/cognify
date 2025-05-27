"""
Collection repository for database operations.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any

from sqlalchemy import select, and_, or_, func, desc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload, joinedload
import structlog

from .base import BaseRepository
from app.models.collections import (
    Collection, CollectionMember, CollectionStats,
    CollectionStatus, CollectionVisibility, MemberRole
)
from app.models.documents import Document
from app.core.exceptions import ValidationError, ConflictError, NotFoundError, PermissionError
from app.core.config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()


class CollectionRepository(BaseRepository[Collection]):
    """Repository for collection-related database operations."""

    def __init__(self, session: AsyncSession):
        super().__init__(session, Collection)

    async def create_collection(
        self,
        name: str,
        owner_id: str,
        display_name: str = None,
        description: str = None,
        visibility: CollectionVisibility = CollectionVisibility.PRIVATE,
        embedding_dimension: int = None,
        distance_metric: str = "cosine",
        **kwargs
    ) -> Collection:
        """
        Create a new collection with owner membership.

        Args:
            name: Collection name (must be unique for the user)
            owner_id: ID of the user creating the collection
            display_name: Human-readable display name
            description: Collection description
            visibility: Collection visibility
            embedding_dimension: Vector embedding dimension
            distance_metric: Distance metric for vector similarity
            **kwargs: Additional collection fields

        Returns:
            Created collection instance
        """
        try:
            # Validate collection name
            if not self._is_valid_collection_name(name):
                raise ValidationError(
                    message="Collection name must be 3-50 characters, alphanumeric and underscores only",
                    field="name",
                    details={"value": name}
                )

            # Check if collection name already exists for this user
            existing_collection = await self.get_by_name_and_owner(name, owner_id)
            if existing_collection:
                raise ConflictError(
                    message="Collection with this name already exists",
                    details={"name": name, "owner_id": owner_id}
                )

            # Set defaults
            if not embedding_dimension:
                embedding_dimension = settings.VECTOR_DIMENSION

            if not display_name:
                display_name = name.replace('_', ' ').title()

            # Create collection
            collection_data = {
                "name": name.lower().strip(),
                "display_name": display_name,
                "description": description,
                "status": CollectionStatus.CREATING,
                "visibility": visibility,
                "embedding_dimension": embedding_dimension,
                "distance_metric": distance_metric,
                "created_by": owner_id,
                **kwargs
            }

            collection = await self.create(**collection_data)

            # Add owner as member
            await self._add_member(
                collection.id,
                owner_id,
                MemberRole.OWNER,
                invited_by=owner_id
            )

            # Create stats record
            await self._create_collection_stats(collection.id)

            # Update status to active
            await self.update(collection.id, status=CollectionStatus.ACTIVE)

            self.logger.info(
                "Collection created successfully",
                collection_id=collection.id,
                name=name,
                owner_id=owner_id
            )

            return collection

        except (ValidationError, ConflictError):
            raise
        except Exception as e:
            self.logger.error(
                "Failed to create collection",
                name=name,
                owner_id=owner_id,
                error=str(e)
            )
            raise

    async def get_by_name_and_owner(self, name: str, owner_id: str) -> Optional[Collection]:
        """Get collection by name and owner."""
        try:
            query = select(Collection).join(CollectionMember).where(
                and_(
                    Collection.name == name.lower().strip(),
                    CollectionMember.user_id == owner_id,
                    CollectionMember.role == MemberRole.OWNER,
                    Collection.deleted_at.is_(None)
                )
            ).options(selectinload(Collection.members))

            result = await self.session.execute(query)
            return result.scalar_one_or_none()

        except Exception as e:
            self.logger.error(
                "Failed to get collection by name and owner",
                name=name,
                owner_id=owner_id,
                error=str(e)
            )
            return None

    async def get_user_collections(
        self,
        user_id: str,
        role: MemberRole = None,
        offset: int = 0,
        limit: int = 100
    ) -> List[Collection]:
        """Get collections where user is a member."""
        try:
            query = select(Collection).join(CollectionMember).where(
                and_(
                    CollectionMember.user_id == user_id,
                    CollectionMember.is_active == True,
                    Collection.deleted_at.is_(None)
                )
            )

            if role:
                query = query.where(CollectionMember.role == role)

            query = query.order_by(desc(Collection.created_at)).offset(offset).limit(limit)
            query = query.options(
                selectinload(Collection.members),
                selectinload(Collection.stats)
            )

            result = await self.session.execute(query)
            return list(result.scalars().all())

        except Exception as e:
            self.logger.error(
                "Failed to get user collections",
                user_id=user_id,
                error=str(e)
            )
            return []

    async def get_public_collections(
        self,
        offset: int = 0,
        limit: int = 100,
        search_query: str = None
    ) -> List[Collection]:
        """Get public collections."""
        try:
            query = select(Collection).where(
                and_(
                    Collection.visibility == CollectionVisibility.PUBLIC,
                    Collection.status == CollectionStatus.ACTIVE,
                    Collection.deleted_at.is_(None)
                )
            )

            if search_query:
                search_term = f"%{search_query}%"
                query = query.where(
                    or_(
                        Collection.name.ilike(search_term),
                        Collection.display_name.ilike(search_term),
                        Collection.description.ilike(search_term)
                    )
                )

            query = query.order_by(desc(Collection.created_at)).offset(offset).limit(limit)
            query = query.options(selectinload(Collection.stats))

            result = await self.session.execute(query)
            return list(result.scalars().all())

        except Exception as e:
            self.logger.error(
                "Failed to get public collections",
                error=str(e)
            )
            return []

    async def add_member(
        self,
        collection_id: str,
        user_id: str,
        role: MemberRole,
        invited_by: str
    ) -> CollectionMember:
        """Add a member to a collection."""
        try:
            # Check if user is already a member
            existing_member = await self.get_member(collection_id, user_id)
            if existing_member and existing_member.is_active:
                raise ConflictError(
                    message="User is already a member of this collection",
                    details={"collection_id": collection_id, "user_id": user_id}
                )

            if existing_member:
                # Reactivate existing member
                existing_member.is_active = True
                existing_member.role = role
                existing_member.invited_by = invited_by
                existing_member.joined_at = datetime.utcnow()
                existing_member.update_permissions()

                await self.session.flush()
                await self.session.refresh(existing_member)

                self.logger.info(
                    "Collection member reactivated",
                    collection_id=collection_id,
                    user_id=user_id,
                    role=role.value
                )

                return existing_member
            else:
                # Add new member
                return await self._add_member(collection_id, user_id, role, invited_by)

        except ConflictError:
            raise
        except Exception as e:
            self.logger.error(
                "Failed to add collection member",
                collection_id=collection_id,
                user_id=user_id,
                error=str(e)
            )
            raise

    async def remove_member(self, collection_id: str, user_id: str) -> bool:
        """Remove a member from a collection."""
        try:
            member = await self.get_member(collection_id, user_id)
            if not member or not member.is_active:
                return False

            # Cannot remove the owner
            if member.role == MemberRole.OWNER:
                raise ValidationError(
                    message="Cannot remove collection owner",
                    field="role"
                )

            # Deactivate member
            member.is_active = False
            await self.session.flush()

            self.logger.info(
                "Collection member removed",
                collection_id=collection_id,
                user_id=user_id
            )

            return True

        except ValidationError:
            raise
        except Exception as e:
            self.logger.error(
                "Failed to remove collection member",
                collection_id=collection_id,
                user_id=user_id,
                error=str(e)
            )
            return False

    async def update_member_role(
        self,
        collection_id: str,
        user_id: str,
        new_role: MemberRole
    ) -> CollectionMember:
        """Update a member's role in a collection."""
        try:
            member = await self.get_member(collection_id, user_id)
            if not member or not member.is_active:
                raise NotFoundError(
                    message="Member not found in collection",
                    resource_type="CollectionMember"
                )

            # Cannot change owner role
            if member.role == MemberRole.OWNER:
                raise ValidationError(
                    message="Cannot change owner role",
                    field="role"
                )

            # Update role and permissions
            member.role = new_role
            member.update_permissions()

            await self.session.flush()
            await self.session.refresh(member)

            self.logger.info(
                "Collection member role updated",
                collection_id=collection_id,
                user_id=user_id,
                new_role=new_role.value
            )

            return member

        except (NotFoundError, ValidationError):
            raise
        except Exception as e:
            self.logger.error(
                "Failed to update member role",
                collection_id=collection_id,
                user_id=user_id,
                error=str(e)
            )
            raise

    async def get_member(self, collection_id: str, user_id: str) -> Optional[CollectionMember]:
        """Get a specific collection member."""
        try:
            query = select(CollectionMember).where(
                and_(
                    CollectionMember.collection_id == collection_id,
                    CollectionMember.user_id == user_id
                )
            )

            result = await self.session.execute(query)
            return result.scalar_one_or_none()

        except Exception as e:
            self.logger.error(
                "Failed to get collection member",
                collection_id=collection_id,
                user_id=user_id,
                error=str(e)
            )
            return None

    async def get_members(
        self,
        collection_id: str,
        active_only: bool = True
    ) -> List[CollectionMember]:
        """Get all members of a collection."""
        try:
            query = select(CollectionMember).where(
                CollectionMember.collection_id == collection_id
            )

            if active_only:
                query = query.where(CollectionMember.is_active == True)

            query = query.order_by(CollectionMember.role, CollectionMember.joined_at)

            result = await self.session.execute(query)
            return list(result.scalars().all())

        except Exception as e:
            self.logger.error(
                "Failed to get collection members",
                collection_id=collection_id,
                error=str(e)
            )
            return []

    async def check_permission(
        self,
        collection_id: str,
        user_id: str,
        permission: str
    ) -> bool:
        """Check if user has specific permission in collection."""
        try:
            member = await self.get_member(collection_id, user_id)
            if not member or not member.is_active:
                return False

            permission_map = {
                "read": member.can_read,
                "write": member.can_write,
                "admin": member.can_admin
            }

            return permission_map.get(permission, False)

        except Exception as e:
            self.logger.error(
                "Failed to check collection permission",
                collection_id=collection_id,
                user_id=user_id,
                permission=permission,
                error=str(e)
            )
            return False

    async def update_stats(
        self,
        collection_id: str,
        documents_delta: int = 0,
        chunks_delta: int = 0,
        vectors_delta: int = 0,
        size_delta: int = 0
    ) -> CollectionStats:
        """Update collection statistics."""
        try:
            # Get or create stats record
            query = select(CollectionStats).where(
                CollectionStats.collection_id == collection_id
            )
            result = await self.session.execute(query)
            stats = result.scalar_one_or_none()

            if not stats:
                stats = await self._create_collection_stats(collection_id)

            # Update stats
            stats.total_documents += documents_delta
            stats.total_chunks += chunks_delta
            stats.total_vectors += vectors_delta
            stats.total_storage_size += size_delta
            stats.last_stats_update = datetime.utcnow()

            # Update collection counters
            await self.update(
                collection_id,
                document_count=stats.total_documents,
                chunk_count=stats.total_chunks,
                vector_count=stats.total_vectors,
                total_size=stats.total_storage_size
            )

            await self.session.flush()
            await self.session.refresh(stats)

            return stats

        except Exception as e:
            self.logger.error(
                "Failed to update collection stats",
                collection_id=collection_id,
                error=str(e)
            )
            raise

    async def _add_member(
        self,
        collection_id: str,
        user_id: str,
        role: MemberRole,
        invited_by: str
    ) -> CollectionMember:
        """Internal method to add a new member."""
        member_data = {
            "collection_id": collection_id,
            "user_id": user_id,
            "role": role,
            "invited_by": invited_by,
            "invited_at": datetime.utcnow(),
            "joined_at": datetime.utcnow()
        }

        member = CollectionMember(**member_data)
        member.update_permissions()

        self.session.add(member)
        await self.session.flush()
        await self.session.refresh(member)

        self.logger.info(
            "Collection member added",
            collection_id=collection_id,
            user_id=user_id,
            role=role.value
        )

        return member

    async def _create_collection_stats(self, collection_id: str) -> CollectionStats:
        """Create initial stats record for collection."""
        stats = CollectionStats(
            collection_id=collection_id,
            last_stats_update=datetime.utcnow()
        )

        self.session.add(stats)
        await self.session.flush()
        await self.session.refresh(stats)

        return stats

    def _is_valid_collection_name(self, name: str) -> bool:
        """Validate collection name format."""
        import re
        # 3-50 characters, alphanumeric and underscores only
        pattern = r'^[a-zA-Z0-9_]{3,50}$'
        return re.match(pattern, name) is not None
