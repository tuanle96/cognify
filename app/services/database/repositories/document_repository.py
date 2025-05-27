"""
Document repository for database operations.
"""

import hashlib
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

from sqlalchemy import select, and_, or_, func, desc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload, joinedload
import structlog

from .base import BaseRepository
from app.models.documents import (
    Document, DocumentMetadata, DocumentChunk,
    DocumentType, DocumentStatus, ProcessingStage
)
from app.core.exceptions import ValidationError, ConflictError, NotFoundError
from app.core.config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()


class DocumentRepository(BaseRepository[Document]):
    """Repository for document-related database operations."""

    def __init__(self, session: AsyncSession):
        super().__init__(session, Document)

    async def create_document(
        self,
        filename: str,
        content: str,
        user_id: str,
        collection_id: str = None,
        document_type: DocumentType = None,
        metadata: Dict[str, Any] = None,
        **kwargs
    ) -> Document:
        """
        Create a new document with content and metadata.

        Args:
            filename: Original filename
            content: Document content
            user_id: ID of user uploading the document
            collection_id: Optional collection ID
            document_type: Document type (auto-detected if not provided)
            metadata: Additional metadata
            **kwargs: Additional document fields

        Returns:
            Created document instance
        """
        try:
            # Auto-detect document type if not provided
            if not document_type:
                document_type = self._detect_document_type(filename)

            # Generate file hash
            file_hash = self._generate_content_hash(content)

            # Check for duplicate content and use shared content if available
            from app.services.database.repositories.shared_content_repository import SharedContentRepository
            shared_content_repo = SharedContentRepository(self.session)

            shared_content = await shared_content_repo.get_by_hash(file_hash)
            if shared_content:
                # Content already exists, create document reference
                self.logger.info(
                    "Using existing shared content for document",
                    filename=filename,
                    shared_content_id=shared_content.id,
                    content_hash=file_hash
                )

                # Increment reference count
                await shared_content_repo.increment_reference_count(shared_content.id)

                # Create document with reference to shared content
                document_data = {
                    "filename": self._sanitize_filename(filename),
                    "original_filename": filename,
                    "content": None,  # Content stored in shared_content
                    "content_preview": content[:1000] if content else None,
                    "document_type": document_type,
                    "file_size": len(content.encode('utf-8')) if content else 0,
                    "file_hash": file_hash,
                    "status": DocumentStatus.COMPLETED if shared_content.is_processed else DocumentStatus.PENDING,
                    "processing_stage": ProcessingStage.COMPLETED if shared_content.is_processed else ProcessingStage.UPLOADED,
                    "collection_id": collection_id,
                    "shared_content_id": shared_content.id,
                    "created_by": user_id,
                    **kwargs
                }
            else:
                # New content, create shared content first
                shared_content = await shared_content_repo.create_shared_content(
                    content=content,
                    content_type=document_type.value if document_type else None,
                    language=self._detect_language_and_encoding(content, filename)[0],
                    encoding=self._detect_language_and_encoding(content, filename)[1]
                )

                # Create document with reference to shared content
                document_data = {
                    "filename": self._sanitize_filename(filename),
                    "original_filename": filename,
                    "content": None,  # Content stored in shared_content
                    "content_preview": content[:1000] if content else None,
                    "document_type": document_type,
                    "file_size": len(content.encode('utf-8')) if content else 0,
                    "file_hash": file_hash,
                    "status": DocumentStatus.PENDING,
                    "processing_stage": ProcessingStage.UPLOADED,
                    "collection_id": collection_id,
                    "shared_content_id": shared_content.id,
                    "created_by": user_id,
                    **kwargs
                }

            # Detect language and encoding
            language, encoding = self._detect_language_and_encoding(content, filename)
            if language:
                document_data["language"] = language
            if encoding:
                document_data["encoding"] = encoding

            # Create document
            document = await self.create(**document_data)

            # Create metadata record
            if metadata or document_type == DocumentType.CODE:
                await self._create_document_metadata(document.id, content, filename, metadata)

            self.logger.info(
                "Document created successfully",
                document_id=document.id,
                filename=filename,
                type=document_type.value,
                size=document.file_size
            )

            return document

        except ConflictError:
            raise
        except Exception as e:
            self.logger.error(
                "Failed to create document",
                filename=filename,
                error=str(e)
            )
            raise

    async def get_by_hash(self, file_hash: str) -> Optional[Document]:
        """Get document by file hash."""
        try:
            query = select(Document).where(
                and_(
                    Document.file_hash == file_hash,
                    Document.deleted_at.is_(None)
                )
            ).options(selectinload(Document.metadata_record))

            result = await self.session.execute(query)
            return result.scalar_one_or_none()

        except Exception as e:
            self.logger.error(
                "Failed to get document by hash",
                hash=file_hash,
                error=str(e)
            )
            return None

    async def get_by_collection(
        self,
        collection_id: str,
        offset: int = 0,
        limit: int = 100,
        status: DocumentStatus = None
    ) -> List[Document]:
        """Get documents in a collection."""
        try:
            query = select(Document).where(
                and_(
                    Document.collection_id == collection_id,
                    Document.deleted_at.is_(None)
                )
            )

            if status:
                query = query.where(Document.status == status)

            query = query.order_by(desc(Document.created_at)).offset(offset).limit(limit)
            query = query.options(selectinload(Document.metadata_record))

            result = await self.session.execute(query)
            return list(result.scalars().all())

        except Exception as e:
            self.logger.error(
                "Failed to get documents by collection",
                collection_id=collection_id,
                error=str(e)
            )
            return []

    async def get_by_user(
        self,
        user_id: str,
        offset: int = 0,
        limit: int = 100,
        status: DocumentStatus = None
    ) -> List[Document]:
        """Get documents created by a user."""
        try:
            query = select(Document).where(
                and_(
                    Document.created_by == user_id,
                    Document.deleted_at.is_(None)
                )
            )

            if status:
                query = query.where(Document.status == status)

            query = query.order_by(desc(Document.created_at)).offset(offset).limit(limit)
            query = query.options(selectinload(Document.metadata_record))

            result = await self.session.execute(query)
            return list(result.scalars().all())

        except Exception as e:
            self.logger.error(
                "Failed to get documents by user",
                user_id=user_id,
                error=str(e)
            )
            return []

    async def update_processing_status(
        self,
        document_id: str,
        status: DocumentStatus,
        stage: ProcessingStage = None,
        error_message: str = None
    ) -> Document:
        """Update document processing status."""
        try:
            document = await self.get_by_id_or_raise(document_id)

            update_data = {"status": status}

            if stage:
                update_data["processing_stage"] = stage

            if status == DocumentStatus.PROCESSING and not document.processing_started_at:
                update_data["processing_started_at"] = datetime.utcnow()
            elif status in [DocumentStatus.COMPLETED, DocumentStatus.FAILED]:
                update_data["processing_completed_at"] = datetime.utcnow()

            if error_message:
                if document.processing_errors:
                    document.processing_errors.append(f"{datetime.utcnow().isoformat()}: {error_message}")
                else:
                    update_data["processing_errors"] = [f"{datetime.utcnow().isoformat()}: {error_message}"]

            updated_document = await self.update(document_id, **update_data)

            self.logger.info(
                "Document processing status updated",
                document_id=document_id,
                status=status.value,
                stage=stage.value if stage else None
            )

            return updated_document

        except Exception as e:
            self.logger.error(
                "Failed to update processing status",
                document_id=document_id,
                error=str(e)
            )
            raise

    async def add_chunk(
        self,
        document_id: str,
        chunk_index: int,
        content: str,
        chunk_type: str = None,
        start_position: int = None,
        end_position: int = None,
        quality_score: float = None,
        metadata: Dict[str, Any] = None
    ) -> DocumentChunk:
        """Add a chunk to a document."""
        try:
            # Generate content hash
            content_hash = self._generate_content_hash(content)

            chunk_data = {
                "document_id": document_id,
                "chunk_index": chunk_index,
                "content": content,
                "content_hash": content_hash,
                "chunk_type": chunk_type,
                "start_position": start_position,
                "end_position": end_position,
                "quality_score": quality_score,
                "metadata": metadata or {}
            }

            chunk = DocumentChunk(**chunk_data)
            self.session.add(chunk)
            await self.session.flush()
            await self.session.refresh(chunk)

            # Update document chunk count
            document = await self.get_by_id_or_raise(document_id)
            await self.update(document_id, chunk_count=document.chunk_count + 1)

            self.logger.debug(
                "Chunk added to document",
                document_id=document_id,
                chunk_index=chunk_index,
                chunk_id=chunk.id
            )

            return chunk

        except Exception as e:
            self.logger.error(
                "Failed to add chunk",
                document_id=document_id,
                chunk_index=chunk_index,
                error=str(e)
            )
            raise

    async def get_chunks(
        self,
        document_id: str,
        offset: int = 0,
        limit: int = 100
    ) -> List[DocumentChunk]:
        """Get chunks for a document."""
        try:
            query = select(DocumentChunk).where(
                DocumentChunk.document_id == document_id
            ).order_by(DocumentChunk.chunk_index).offset(offset).limit(limit)

            result = await self.session.execute(query)
            return list(result.scalars().all())

        except Exception as e:
            self.logger.error(
                "Failed to get chunks",
                document_id=document_id,
                error=str(e)
            )
            return []

    async def search_documents(
        self,
        query: str,
        user_id: str = None,
        collection_id: str = None,
        document_type: DocumentType = None,
        offset: int = 0,
        limit: int = 100
    ) -> List[Document]:
        """Search documents by content or metadata."""
        try:
            # Build base query
            sql_query = select(Document).where(Document.deleted_at.is_(None))

            # Add filters
            if user_id:
                sql_query = sql_query.where(Document.created_by == user_id)

            if collection_id:
                sql_query = sql_query.where(Document.collection_id == collection_id)

            if document_type:
                sql_query = sql_query.where(Document.document_type == document_type)

            # Add text search
            if query:
                search_term = f"%{query}%"
                sql_query = sql_query.where(
                    or_(
                        Document.filename.ilike(search_term),
                        Document.title.ilike(search_term),
                        Document.content.ilike(search_term),
                        Document.description.ilike(search_term)
                    )
                )

            # Order and paginate
            sql_query = sql_query.order_by(desc(Document.created_at)).offset(offset).limit(limit)
            sql_query = sql_query.options(selectinload(Document.metadata_record))

            result = await self.session.execute(sql_query)
            documents = list(result.scalars().all())

            self.logger.debug(
                "Document search completed",
                query=query,
                results_count=len(documents)
            )

            return documents

        except Exception as e:
            self.logger.error(
                "Document search failed",
                query=query,
                error=str(e)
            )
            return []

    async def get_processing_stats(self, user_id: str = None) -> Dict[str, Any]:
        """Get document processing statistics."""
        try:
            base_query = select(Document).where(Document.deleted_at.is_(None))

            if user_id:
                base_query = base_query.where(Document.created_by == user_id)

            # Count by status
            stats = {}
            for status in DocumentStatus:
                count_query = select(func.count(Document.id)).where(
                    and_(
                        Document.status == status,
                        Document.deleted_at.is_(None)
                    )
                )
                if user_id:
                    count_query = count_query.where(Document.created_by == user_id)

                result = await self.session.execute(count_query)
                stats[f"{status.value}_count"] = result.scalar() or 0

            # Count by type
            type_stats = {}
            for doc_type in DocumentType:
                count_query = select(func.count(Document.id)).where(
                    and_(
                        Document.document_type == doc_type,
                        Document.deleted_at.is_(None)
                    )
                )
                if user_id:
                    count_query = count_query.where(Document.created_by == user_id)

                result = await self.session.execute(count_query)
                type_stats[doc_type.value] = result.scalar() or 0

            stats["by_type"] = type_stats

            # Total size
            size_query = select(func.sum(Document.file_size)).where(Document.deleted_at.is_(None))
            if user_id:
                size_query = size_query.where(Document.created_by == user_id)

            result = await self.session.execute(size_query)
            stats["total_size"] = result.scalar() or 0

            return stats

        except Exception as e:
            self.logger.error(
                "Failed to get processing stats",
                user_id=user_id,
                error=str(e)
            )
            return {}

    async def _create_document_metadata(
        self,
        document_id: str,
        content: str,
        filename: str,
        additional_metadata: Dict[str, Any] = None
    ) -> DocumentMetadata:
        """Create metadata record for document."""
        try:
            metadata_data = {
                "document_id": document_id,
                "word_count": len(content.split()) if content else 0,
                "character_count": len(content) if content else 0,
                "line_count": content.count('\n') + 1 if content else 0
            }

            # Extract additional metadata based on file type
            if filename.endswith(('.py', '.js', '.ts', '.java', '.cpp', '.c', '.h')):
                # Code file metadata
                metadata_data.update(self._extract_code_metadata(content, filename))

            # Add any additional metadata
            if additional_metadata:
                metadata_data.update(additional_metadata)

            metadata = DocumentMetadata(**metadata_data)
            self.session.add(metadata)
            await self.session.flush()

            return metadata

        except Exception as e:
            self.logger.error(
                "Failed to create document metadata",
                document_id=document_id,
                error=str(e)
            )
            raise

    def _detect_document_type(self, filename: str) -> DocumentType:
        """Detect document type from filename."""
        extension = Path(filename).suffix.lower()

        type_mapping = {
            '.py': DocumentType.CODE,
            '.js': DocumentType.CODE,
            '.ts': DocumentType.CODE,
            '.java': DocumentType.CODE,
            '.cpp': DocumentType.CODE,
            '.c': DocumentType.CODE,
            '.h': DocumentType.CODE,
            '.go': DocumentType.CODE,
            '.rs': DocumentType.CODE,
            '.md': DocumentType.MARKDOWN,
            '.txt': DocumentType.TEXT,
            '.pdf': DocumentType.PDF,
            '.json': DocumentType.JSON,
            '.yaml': DocumentType.YAML,
            '.yml': DocumentType.YAML,
            '.xml': DocumentType.XML,
            '.csv': DocumentType.CSV,
            '.html': DocumentType.HTML,
            '.htm': DocumentType.HTML
        }

        return type_mapping.get(extension, DocumentType.UNKNOWN)

    def _generate_content_hash(self, content: str) -> str:
        """Generate SHA-256 hash of content."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for storage."""
        import re
        # Remove or replace invalid characters
        sanitized = re.sub(r'[^\w\-_\.]', '_', filename)
        return sanitized[:255]  # Limit length

    def _detect_language_and_encoding(self, content: str, filename: str) -> Tuple[Optional[str], Optional[str]]:
        """Detect programming language and encoding."""
        extension = Path(filename).suffix.lower()

        language_mapping = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.h': 'c',
            '.go': 'go',
            '.rs': 'rust'
        }

        language = language_mapping.get(extension)
        encoding = 'utf-8'  # Default encoding

        return language, encoding

    def _extract_code_metadata(self, content: str, filename: str) -> Dict[str, Any]:
        """Extract metadata from code files."""
        metadata = {
            "programming_language": self._detect_language_and_encoding(content, filename)[0],
            "functions_count": 0,
            "classes_count": 0,
            "imports_count": 0
        }

        if not content:
            return metadata

        lines = content.split('\n')

        # Simple pattern matching for different languages
        for line in lines:
            line = line.strip()

            # Function patterns
            if any(pattern in line for pattern in ['def ', 'function ', 'func ', 'public ', 'private ']):
                if any(keyword in line for keyword in ['def ', 'function', 'func']):
                    metadata["functions_count"] += 1

            # Class patterns
            if line.startswith('class ') or 'class ' in line:
                metadata["classes_count"] += 1

            # Import patterns
            if line.startswith(('import ', 'from ', '#include', 'using ', 'require(')):
                metadata["imports_count"] += 1

        return metadata
