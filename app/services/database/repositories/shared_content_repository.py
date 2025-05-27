"""
Repository for shared content operations.

Handles content deduplication and shared content management.
"""

import hashlib
from datetime import datetime
from typing import Dict, Any, List, Optional
from sqlalchemy import select, and_, func, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.shared_content import SharedContent, SharedContentChunk
from app.services.database.repositories.base import BaseRepository
from app.core.exceptions import NotFoundError, ConflictError


class SharedContentRepository(BaseRepository[SharedContent]):
    """Repository for shared content operations."""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, SharedContent)
    
    async def get_by_hash(self, content_hash: str) -> Optional[SharedContent]:
        """Get shared content by hash."""
        try:
            query = select(SharedContent).where(
                and_(
                    SharedContent.content_hash == content_hash,
                    SharedContent.deleted_at.is_(None)
                )
            ).options(selectinload(SharedContent.chunks))
            
            result = await self.session.execute(query)
            return result.scalar_one_or_none()
            
        except Exception as e:
            self.logger.error(
                "Failed to get shared content by hash",
                hash=content_hash,
                error=str(e)
            )
            return None
    
    async def create_shared_content(
        self,
        content: str,
        content_type: str = None,
        language: str = None,
        encoding: str = "utf-8",
        processing_config: Dict[str, Any] = None
    ) -> SharedContent:
        """Create new shared content."""
        try:
            # Generate content hash
            content_hash = self._generate_content_hash(content)
            
            # Check if already exists
            existing = await self.get_by_hash(content_hash)
            if existing:
                # Increment reference count
                await self.increment_reference_count(existing.id)
                return existing
            
            # Create new shared content
            shared_content_data = {
                "content_hash": content_hash,
                "content": content,
                "content_size": len(content.encode('utf-8')),
                "content_type": content_type,
                "language": language,
                "encoding": encoding,
                "processing_config": processing_config or {},
                "reference_count": 1,
                "is_processed": False
            }
            
            shared_content = await self.create(**shared_content_data)
            
            self.logger.info(
                "Shared content created",
                shared_content_id=shared_content.id,
                content_hash=content_hash,
                content_size=shared_content.content_size
            )
            
            return shared_content
            
        except Exception as e:
            self.logger.error(
                "Failed to create shared content",
                error=str(e)
            )
            raise
    
    async def increment_reference_count(self, shared_content_id: str) -> None:
        """Increment reference count for shared content."""
        try:
            query = update(SharedContent).where(
                SharedContent.id == shared_content_id
            ).values(
                reference_count=SharedContent.reference_count + 1,
                last_accessed_at=func.now()
            )
            
            await self.session.execute(query)
            await self.session.commit()
            
            self.logger.debug(
                "Reference count incremented",
                shared_content_id=shared_content_id
            )
            
        except Exception as e:
            self.logger.error(
                "Failed to increment reference count",
                shared_content_id=shared_content_id,
                error=str(e)
            )
            raise
    
    async def decrement_reference_count(self, shared_content_id: str) -> None:
        """Decrement reference count for shared content."""
        try:
            # Get current shared content
            shared_content = await self.get_by_id(shared_content_id)
            if not shared_content:
                return
            
            new_count = max(0, shared_content.reference_count - 1)
            
            if new_count == 0:
                # No more references, can delete
                await self.delete(shared_content_id)
                self.logger.info(
                    "Shared content deleted (no references)",
                    shared_content_id=shared_content_id
                )
            else:
                # Update reference count
                query = update(SharedContent).where(
                    SharedContent.id == shared_content_id
                ).values(reference_count=new_count)
                
                await self.session.execute(query)
                await self.session.commit()
                
                self.logger.debug(
                    "Reference count decremented",
                    shared_content_id=shared_content_id,
                    new_count=new_count
                )
            
        except Exception as e:
            self.logger.error(
                "Failed to decrement reference count",
                shared_content_id=shared_content_id,
                error=str(e)
            )
            raise
    
    async def mark_as_processed(
        self,
        shared_content_id: str,
        chunk_count: int = 0,
        embedding_count: int = 0,
        quality_score: float = None,
        chunking_quality_score: float = None
    ) -> None:
        """Mark shared content as processed."""
        try:
            query = update(SharedContent).where(
                SharedContent.id == shared_content_id
            ).values(
                is_processed=True,
                processing_completed_at=func.now(),
                chunk_count=chunk_count,
                embedding_count=embedding_count,
                processing_quality_score=quality_score,
                chunking_quality_score=chunking_quality_score
            )
            
            await self.session.execute(query)
            await self.session.commit()
            
            self.logger.info(
                "Shared content marked as processed",
                shared_content_id=shared_content_id,
                chunk_count=chunk_count,
                embedding_count=embedding_count
            )
            
        except Exception as e:
            self.logger.error(
                "Failed to mark shared content as processed",
                shared_content_id=shared_content_id,
                error=str(e)
            )
            raise
    
    async def add_chunk(
        self,
        shared_content_id: str,
        chunk_index: int,
        content: str,
        chunk_type: str = None,
        start_position: int = None,
        end_position: int = None,
        quality_score: float = None,
        vector_id: str = None,
        embedding_model: str = None,
        metadata: Dict[str, Any] = None
    ) -> SharedContentChunk:
        """Add a chunk to shared content."""
        try:
            # Generate content hash
            content_hash = self._generate_content_hash(content)
            
            chunk_data = {
                "shared_content_id": shared_content_id,
                "chunk_index": chunk_index,
                "content": content,
                "content_hash": content_hash,
                "chunk_type": chunk_type,
                "start_position": start_position,
                "end_position": end_position,
                "quality_score": quality_score,
                "vector_id": vector_id,
                "embedding_model": embedding_model,
                "metadata": metadata or {}
            }
            
            # Create chunk using base repository
            chunk = SharedContentChunk(**chunk_data)
            self.session.add(chunk)
            await self.session.commit()
            await self.session.refresh(chunk)
            
            self.logger.debug(
                "Chunk added to shared content",
                shared_content_id=shared_content_id,
                chunk_index=chunk_index,
                chunk_id=chunk.id
            )
            
            return chunk
            
        except Exception as e:
            self.logger.error(
                "Failed to add chunk to shared content",
                shared_content_id=shared_content_id,
                chunk_index=chunk_index,
                error=str(e)
            )
            raise
    
    async def get_chunks(self, shared_content_id: str) -> List[SharedContentChunk]:
        """Get all chunks for shared content."""
        try:
            query = select(SharedContentChunk).where(
                and_(
                    SharedContentChunk.shared_content_id == shared_content_id,
                    SharedContentChunk.deleted_at.is_(None)
                )
            ).order_by(SharedContentChunk.chunk_index)
            
            result = await self.session.execute(query)
            return result.scalars().all()
            
        except Exception as e:
            self.logger.error(
                "Failed to get chunks for shared content",
                shared_content_id=shared_content_id,
                error=str(e)
            )
            return []
    
    def _generate_content_hash(self, content: str) -> str:
        """Generate SHA-256 hash of content."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
