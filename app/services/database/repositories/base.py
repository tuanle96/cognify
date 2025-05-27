"""
Base repository pattern for database operations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, TypeVar, Generic, Union
from uuid import uuid4

from sqlalchemy import select, update, delete, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload, joinedload
from sqlalchemy.exc import IntegrityError, NoResultFound
import structlog

from app.models.base import BaseModel
from app.core.exceptions import (
    DatabaseError,
    NotFoundError,
    ValidationError,
    ConflictError
)

logger = structlog.get_logger(__name__)

# Type variable for model classes
ModelType = TypeVar("ModelType", bound=BaseModel)


class BaseRepository(Generic[ModelType], ABC):
    """
    Base repository class providing common database operations.
    
    This class implements the Repository pattern for database access,
    providing a clean interface for CRUD operations and common queries.
    """
    
    def __init__(self, session: AsyncSession, model_class: Type[ModelType]):
        self.session = session
        self.model_class = model_class
        self.logger = logger.bind(repository=self.__class__.__name__)
    
    async def create(self, **kwargs) -> ModelType:
        """
        Create a new record.
        
        Args:
            **kwargs: Field values for the new record
            
        Returns:
            Created model instance
            
        Raises:
            ValidationError: If validation fails
            ConflictError: If unique constraint is violated
            DatabaseError: If database operation fails
        """
        try:
            # Generate ID if not provided
            if 'id' not in kwargs:
                kwargs['id'] = str(uuid4())
            
            instance = self.model_class(**kwargs)
            self.session.add(instance)
            await self.session.flush()
            await self.session.refresh(instance)
            
            self.logger.info(
                "Record created",
                model=self.model_class.__name__,
                id=instance.id
            )
            
            return instance
            
        except IntegrityError as e:
            await self.session.rollback()
            self.logger.error(
                "Integrity constraint violation",
                model=self.model_class.__name__,
                error=str(e)
            )
            raise ConflictError(
                message=f"Record with these values already exists",
                details={"error": str(e)}
            )
        except Exception as e:
            await self.session.rollback()
            self.logger.error(
                "Failed to create record",
                model=self.model_class.__name__,
                error=str(e)
            )
            raise DatabaseError(
                message=f"Failed to create {self.model_class.__name__}",
                operation="create",
                details={"error": str(e)}
            )
    
    async def get_by_id(self, id: str, include_deleted: bool = False) -> Optional[ModelType]:
        """
        Get a record by ID.
        
        Args:
            id: Record ID
            include_deleted: Whether to include soft-deleted records
            
        Returns:
            Model instance or None if not found
        """
        try:
            query = select(self.model_class).where(self.model_class.id == id)
            
            # Filter out soft-deleted records unless explicitly requested
            if not include_deleted and hasattr(self.model_class, 'deleted_at'):
                query = query.where(self.model_class.deleted_at.is_(None))
            
            result = await self.session.execute(query)
            instance = result.scalar_one_or_none()
            
            if instance:
                self.logger.debug(
                    "Record found",
                    model=self.model_class.__name__,
                    id=id
                )
            
            return instance
            
        except Exception as e:
            self.logger.error(
                "Failed to get record by ID",
                model=self.model_class.__name__,
                id=id,
                error=str(e)
            )
            raise DatabaseError(
                message=f"Failed to get {self.model_class.__name__} by ID",
                operation="get_by_id",
                details={"id": id, "error": str(e)}
            )
    
    async def get_by_id_or_raise(self, id: str, include_deleted: bool = False) -> ModelType:
        """
        Get a record by ID or raise NotFoundError.
        
        Args:
            id: Record ID
            include_deleted: Whether to include soft-deleted records
            
        Returns:
            Model instance
            
        Raises:
            NotFoundError: If record not found
        """
        instance = await self.get_by_id(id, include_deleted)
        if not instance:
            raise NotFoundError(
                message=f"{self.model_class.__name__} not found",
                resource_type=self.model_class.__name__,
                resource_id=id
            )
        return instance
    
    async def update(self, id: str, **kwargs) -> ModelType:
        """
        Update a record.
        
        Args:
            id: Record ID
            **kwargs: Fields to update
            
        Returns:
            Updated model instance
            
        Raises:
            NotFoundError: If record not found
            ValidationError: If validation fails
            ConflictError: If unique constraint is violated
            DatabaseError: If database operation fails
        """
        try:
            instance = await self.get_by_id_or_raise(id)
            
            # Update fields
            for key, value in kwargs.items():
                if hasattr(instance, key):
                    setattr(instance, key, value)
            
            await self.session.flush()
            await self.session.refresh(instance)
            
            self.logger.info(
                "Record updated",
                model=self.model_class.__name__,
                id=id,
                fields=list(kwargs.keys())
            )
            
            return instance
            
        except (NotFoundError, ValidationError, ConflictError):
            raise
        except IntegrityError as e:
            await self.session.rollback()
            self.logger.error(
                "Integrity constraint violation during update",
                model=self.model_class.__name__,
                id=id,
                error=str(e)
            )
            raise ConflictError(
                message=f"Update would violate constraints",
                details={"error": str(e)}
            )
        except Exception as e:
            await self.session.rollback()
            self.logger.error(
                "Failed to update record",
                model=self.model_class.__name__,
                id=id,
                error=str(e)
            )
            raise DatabaseError(
                message=f"Failed to update {self.model_class.__name__}",
                operation="update",
                details={"id": id, "error": str(e)}
            )
    
    async def delete(self, id: str, soft_delete: bool = True) -> bool:
        """
        Delete a record.
        
        Args:
            id: Record ID
            soft_delete: Whether to soft delete (if supported) or hard delete
            
        Returns:
            True if deleted, False if not found
            
        Raises:
            DatabaseError: If database operation fails
        """
        try:
            instance = await self.get_by_id(id)
            if not instance:
                return False
            
            if soft_delete and hasattr(instance, 'soft_delete'):
                # Soft delete
                instance.soft_delete()
                await self.session.flush()
                self.logger.info(
                    "Record soft deleted",
                    model=self.model_class.__name__,
                    id=id
                )
            else:
                # Hard delete
                await self.session.delete(instance)
                await self.session.flush()
                self.logger.info(
                    "Record hard deleted",
                    model=self.model_class.__name__,
                    id=id
                )
            
            return True
            
        except Exception as e:
            await self.session.rollback()
            self.logger.error(
                "Failed to delete record",
                model=self.model_class.__name__,
                id=id,
                error=str(e)
            )
            raise DatabaseError(
                message=f"Failed to delete {self.model_class.__name__}",
                operation="delete",
                details={"id": id, "error": str(e)}
            )
    
    async def list(
        self,
        offset: int = 0,
        limit: int = 100,
        filters: Dict[str, Any] = None,
        order_by: str = None,
        include_deleted: bool = False
    ) -> List[ModelType]:
        """
        List records with pagination and filtering.
        
        Args:
            offset: Number of records to skip
            limit: Maximum number of records to return
            filters: Dictionary of field filters
            order_by: Field name to order by
            include_deleted: Whether to include soft-deleted records
            
        Returns:
            List of model instances
        """
        try:
            query = select(self.model_class)
            
            # Apply filters
            if filters:
                query = self._apply_filters(query, filters)
            
            # Filter out soft-deleted records unless explicitly requested
            if not include_deleted and hasattr(self.model_class, 'deleted_at'):
                query = query.where(self.model_class.deleted_at.is_(None))
            
            # Apply ordering
            if order_by:
                if hasattr(self.model_class, order_by):
                    order_field = getattr(self.model_class, order_by)
                    query = query.order_by(order_field)
            else:
                # Default ordering by created_at if available
                if hasattr(self.model_class, 'created_at'):
                    query = query.order_by(self.model_class.created_at.desc())
            
            # Apply pagination
            query = query.offset(offset).limit(limit)
            
            result = await self.session.execute(query)
            instances = result.scalars().all()
            
            self.logger.debug(
                "Records listed",
                model=self.model_class.__name__,
                count=len(instances),
                offset=offset,
                limit=limit
            )
            
            return list(instances)
            
        except Exception as e:
            self.logger.error(
                "Failed to list records",
                model=self.model_class.__name__,
                error=str(e)
            )
            raise DatabaseError(
                message=f"Failed to list {self.model_class.__name__}",
                operation="list",
                details={"error": str(e)}
            )
    
    async def count(
        self,
        filters: Dict[str, Any] = None,
        include_deleted: bool = False
    ) -> int:
        """
        Count records with optional filtering.
        
        Args:
            filters: Dictionary of field filters
            include_deleted: Whether to include soft-deleted records
            
        Returns:
            Number of matching records
        """
        try:
            query = select(func.count(self.model_class.id))
            
            # Apply filters
            if filters:
                query = self._apply_filters(query, filters)
            
            # Filter out soft-deleted records unless explicitly requested
            if not include_deleted and hasattr(self.model_class, 'deleted_at'):
                query = query.where(self.model_class.deleted_at.is_(None))
            
            result = await self.session.execute(query)
            count = result.scalar()
            
            self.logger.debug(
                "Records counted",
                model=self.model_class.__name__,
                count=count
            )
            
            return count or 0
            
        except Exception as e:
            self.logger.error(
                "Failed to count records",
                model=self.model_class.__name__,
                error=str(e)
            )
            raise DatabaseError(
                message=f"Failed to count {self.model_class.__name__}",
                operation="count",
                details={"error": str(e)}
            )
    
    def _apply_filters(self, query, filters: Dict[str, Any]):
        """Apply filters to a query."""
        for field, value in filters.items():
            if not hasattr(self.model_class, field):
                continue
            
            column = getattr(self.model_class, field)
            
            if isinstance(value, dict):
                # Handle complex filters
                for op, op_value in value.items():
                    if op == 'eq':
                        query = query.where(column == op_value)
                    elif op == 'ne':
                        query = query.where(column != op_value)
                    elif op == 'gt':
                        query = query.where(column > op_value)
                    elif op == 'gte':
                        query = query.where(column >= op_value)
                    elif op == 'lt':
                        query = query.where(column < op_value)
                    elif op == 'lte':
                        query = query.where(column <= op_value)
                    elif op == 'in':
                        query = query.where(column.in_(op_value))
                    elif op == 'not_in':
                        query = query.where(~column.in_(op_value))
                    elif op == 'like':
                        query = query.where(column.like(f"%{op_value}%"))
                    elif op == 'ilike':
                        query = query.where(column.ilike(f"%{op_value}%"))
            elif isinstance(value, list):
                # Handle IN filter
                query = query.where(column.in_(value))
            else:
                # Simple equality filter
                query = query.where(column == value)
        
        return query
