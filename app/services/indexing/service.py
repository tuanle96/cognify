"""
Main indexing service orchestrating the complete RAG pipeline.
"""

import asyncio
import time
import uuid
from typing import List, Dict, Any, Optional, Set
from pathlib import Path
import logging

from .base import (
    IndexingRequest, IndexingResponse, IndexedDocument, IndexingProgress,
    IndexingStatus, IndexingConfig, IndexingError, IndexingTimeoutError,
    ProgressCallback
)
from .manager import index_manager
from ..parsers.service import parsing_service
from ..chunking.service import chunking_service
# from ..embedding.service import embedding_service  # Commented out to avoid early import
from ..vectordb.service import vectordb_service
from ..vectordb.base import VectorPoint

logger = logging.getLogger(__name__)

class IndexingService:
    """
    Main indexing service orchestrating the complete RAG pipeline.

    Pipeline: Parse → Chunk → Embed → Store

    Features:
    - Complete document processing pipeline
    - Batch processing with progress tracking
    - Incremental updates and change detection
    - Error handling and recovery
    - Performance monitoring
    """

    def __init__(self, config: Optional[IndexingConfig] = None):
        self.config = config or IndexingConfig()
        self._initialized = False
        self._active_jobs: Dict[str, IndexingProgress] = {}
        self._stats = {
            "total_jobs": 0,
            "successful_jobs": 0,
            "failed_jobs": 0,
            "total_documents_processed": 0,
            "total_chunks_created": 0,
            "total_embeddings_generated": 0,
            "total_processing_time": 0.0
        }

    async def initialize(self) -> None:
        """Initialize the indexing service and all dependencies."""
        try:
            # Import embedding service locally to avoid early import
            from ..embedding.service import embedding_service

            # Initialize all required services
            services_to_init = [
                ("parsing", parsing_service),
                ("chunking", chunking_service),
                ("embedding", embedding_service),
                ("vectordb", vectordb_service),
                ("index_manager", index_manager)
            ]

            for service_name, service in services_to_init:
                if not getattr(service, '_initialized', False):
                    logger.info(f"Initializing {service_name} service...")
                    await service.initialize()

            self._initialized = True
            logger.info("Indexing service initialized with complete pipeline")

        except Exception as e:
            raise IndexingError(f"Failed to initialize indexing service: {e}") from e

    async def index_document(
        self,
        request: IndexingRequest,
        progress_callback: Optional[ProgressCallback] = None
    ) -> IndexingResponse:
        """
        Index a single document through the complete pipeline.

        Args:
            request: Indexing request
            progress_callback: Optional callback for progress updates

        Returns:
            Indexing response with results
        """
        if not self._initialized:
            await self.initialize()

        # Create job
        job_id = str(uuid.uuid4())
        progress = IndexingProgress(
            job_id=job_id,
            status=IndexingStatus.PROCESSING,
            total_documents=1
        )

        self._active_jobs[job_id] = progress
        self._stats["total_jobs"] += 1

        try:
            # Use request config or service default
            config = request.config or self.config

            # Ensure collection exists
            await index_manager.ensure_collection(config)

            # Process single document
            indexed_doc = await self._process_document(request, config, progress, progress_callback)

            # Update progress
            progress.status = IndexingStatus.COMPLETED
            progress.end_time = time.time()
            progress.processed_documents = 1
            progress.successful_documents = 1

            # Update stats
            self._stats["successful_jobs"] += 1
            self._stats["total_documents_processed"] += 1
            self._stats["total_processing_time"] += progress.elapsed_time

            # Create response
            response = IndexingResponse(
                job_id=job_id,
                status=IndexingStatus.COMPLETED,
                progress=progress,
                indexed_documents=[indexed_doc],
                collection_name=config.collection_name,
                total_processing_time=progress.elapsed_time,
                total_chunks_created=len(indexed_doc.chunks),
                total_embeddings_generated=len(indexed_doc.embeddings),
                total_vectors_stored=len(indexed_doc.embeddings),
                config=config,
                success=True
            )

            logger.info(f"Successfully indexed document: {request.document_id}")
            return response

        except Exception as e:
            # Update progress on error
            progress.status = IndexingStatus.FAILED
            progress.end_time = time.time()
            progress.errors.append(str(e))

            # Update stats
            self._stats["failed_jobs"] += 1

            logger.error(f"Failed to index document {request.document_id}: {e}")

            # Create error response
            return IndexingResponse(
                job_id=job_id,
                status=IndexingStatus.FAILED,
                progress=progress,
                collection_name=self.config.collection_name,
                success=False,
                errors=[str(e)]
            )

        finally:
            # Cleanup job tracking
            self._active_jobs.pop(job_id, None)

    async def index_batch(
        self,
        requests: List[IndexingRequest],
        progress_callback: Optional[ProgressCallback] = None
    ) -> IndexingResponse:
        """
        Index multiple documents in batch with parallel processing.

        Args:
            requests: List of indexing requests
            progress_callback: Optional callback for progress updates

        Returns:
            Batch indexing response
        """
        if not self._initialized:
            await self.initialize()

        if not requests:
            raise IndexingError("No requests provided for batch indexing")

        # Create job
        job_id = str(uuid.uuid4())
        progress = IndexingProgress(
            job_id=job_id,
            status=IndexingStatus.PROCESSING,
            total_documents=len(requests)
        )

        self._active_jobs[job_id] = progress
        self._stats["total_jobs"] += 1

        try:
            # Use first request's config or service default
            config = requests[0].config or self.config

            # Ensure collection exists
            await index_manager.ensure_collection(config)

            # Process documents in batches
            indexed_documents = []
            semaphore = asyncio.Semaphore(config.max_concurrent)

            async def process_with_semaphore(req: IndexingRequest) -> Optional[IndexedDocument]:
                async with semaphore:
                    try:
                        return await self._process_document(req, config, progress, progress_callback)
                    except Exception as e:
                        progress.errors.append(f"Document {req.document_id}: {e}")
                        progress.failed_documents += 1
                        logger.error(f"Failed to process document {req.document_id}: {e}")
                        return None

            # Execute batch processing
            tasks = [process_with_semaphore(req) for req in requests]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Collect successful results
            for result in results:
                if isinstance(result, IndexedDocument):
                    indexed_documents.append(result)

            # Update final progress
            progress.status = IndexingStatus.COMPLETED
            progress.end_time = time.time()

            # Calculate totals
            total_chunks = sum(len(doc.chunks) for doc in indexed_documents)
            total_embeddings = sum(len(doc.embeddings) for doc in indexed_documents)

            # Update stats
            if progress.failed_documents == 0:
                self._stats["successful_jobs"] += 1
            else:
                self._stats["failed_jobs"] += 1

            self._stats["total_documents_processed"] += progress.successful_documents
            self._stats["total_chunks_created"] += total_chunks
            self._stats["total_embeddings_generated"] += total_embeddings
            self._stats["total_processing_time"] += progress.elapsed_time

            # Create response
            response = IndexingResponse(
                job_id=job_id,
                status=IndexingStatus.COMPLETED,
                progress=progress,
                indexed_documents=indexed_documents,
                collection_name=config.collection_name,
                total_processing_time=progress.elapsed_time,
                total_chunks_created=total_chunks,
                total_embeddings_generated=total_embeddings,
                total_vectors_stored=total_embeddings,
                config=config,
                success=progress.failed_documents == 0,
                errors=progress.errors
            )

            logger.info(f"Batch indexing completed: {progress.successful_documents}/{progress.total_documents} successful")
            return response

        except Exception as e:
            # Update progress on error
            progress.status = IndexingStatus.FAILED
            progress.end_time = time.time()
            progress.errors.append(str(e))

            # Update stats
            self._stats["failed_jobs"] += 1

            logger.error(f"Batch indexing failed: {e}")

            # Create error response
            return IndexingResponse(
                job_id=job_id,
                status=IndexingStatus.FAILED,
                progress=progress,
                collection_name=self.config.collection_name,
                success=False,
                errors=[str(e)]
            )

        finally:
            # Cleanup job tracking
            self._active_jobs.pop(job_id, None)

    async def index_directory(
        self,
        directory_path: str,
        recursive: bool = True,
        file_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        incremental: bool = True,
        progress_callback: Optional[ProgressCallback] = None
    ) -> IndexingResponse:
        """
        Index all supported files in a directory.

        Args:
            directory_path: Path to directory
            recursive: Whether to search subdirectories
            file_patterns: File patterns to include
            exclude_patterns: File patterns to exclude
            incremental: Whether to skip already indexed files
            progress_callback: Optional callback for progress updates

        Returns:
            Directory indexing response
        """
        directory = Path(directory_path)
        if not directory.exists() or not directory.is_dir():
            raise IndexingError(f"Directory not found: {directory_path}")

        # Find files to index
        files = await self._find_files(directory, recursive, file_patterns, exclude_patterns)

        # Filter for incremental indexing
        if incremental:
            files = await self._filter_changed_files(files)

        # Create indexing requests
        requests = []
        for file_path in files:
            request = IndexingRequest(
                file_path=str(file_path),
                document_id=str(file_path.relative_to(directory)),
                config=self.config
            )
            requests.append(request)

        logger.info(f"Found {len(requests)} files to index in {directory_path}")

        # Process as batch
        return await self.index_batch(requests, progress_callback)

    async def get_job_progress(self, job_id: str) -> Optional[IndexingProgress]:
        """Get progress information for a job."""
        return self._active_jobs.get(job_id)

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel an active indexing job."""
        if job_id in self._active_jobs:
            progress = self._active_jobs[job_id]
            progress.status = IndexingStatus.CANCELLED
            progress.end_time = time.time()
            logger.info(f"Cancelled indexing job: {job_id}")
            return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get indexing service statistics."""
        stats = self._stats.copy()

        # Add derived metrics
        if stats["total_jobs"] > 0:
            stats["job_success_rate"] = stats["successful_jobs"] / stats["total_jobs"]
            stats["avg_processing_time"] = stats["total_processing_time"] / stats["total_jobs"]
        else:
            stats["job_success_rate"] = 0.0
            stats["avg_processing_time"] = 0.0

        # Add active job info
        stats["active_jobs"] = len(self._active_jobs)
        stats["active_job_ids"] = list(self._active_jobs.keys())

        return stats

    async def health_check(self) -> Dict[str, Any]:
        """Check the health of the indexing service and all dependencies."""
        if not self._initialized:
            try:
                await self.initialize()
            except Exception as e:
                return {
                    "status": "unhealthy",
                    "error": f"Initialization failed: {e}",
                    "stats": self.get_stats()
                }

        health_status = {
            "status": "healthy",
            "services": {},
            "stats": self.get_stats(),
            "config": {
                "collection_name": self.config.collection_name,
                "batch_size": self.config.batch_size,
                "max_concurrent": self.config.max_concurrent
            }
        }

        # Import embedding service locally to avoid early import
        from ..embedding.service import embedding_service

        # Check all dependent services
        services = [
            ("parsing", parsing_service),
            ("chunking", chunking_service),
            ("embedding", embedding_service),
            ("vectordb", vectordb_service),
            ("index_manager", index_manager)
        ]

        for service_name, service in services:
            try:
                if hasattr(service, 'health_check'):
                    service_health = await service.health_check()
                    health_status["services"][service_name] = service_health
                else:
                    health_status["services"][service_name] = {
                        "status": "healthy" if getattr(service, '_initialized', False) else "unknown",
                        "initialized": getattr(service, '_initialized', False)
                    }
            except Exception as e:
                health_status["services"][service_name] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health_status["status"] = "degraded"

        return health_status

    async def cleanup(self) -> None:
        """Cleanup all resources."""
        # Cancel active jobs
        for job_id in list(self._active_jobs.keys()):
            await self.cancel_job(job_id)

        self._active_jobs.clear()
        self._initialized = False
        logger.info("Indexing service cleaned up")

    # Private methods

    async def _process_document(
        self,
        request: IndexingRequest,
        config: IndexingConfig,
        progress: IndexingProgress,
        progress_callback: Optional[ProgressCallback] = None
    ) -> IndexedDocument:
        """Process a single document through the complete pipeline."""
        start_time = time.time()

        # Update progress
        progress.current_document = request.document_id
        progress.current_operation = "parsing"
        if progress_callback:
            progress_callback(progress)

        # Step 1: Parse document
        parse_start = time.time()
        parse_response = await parsing_service.parse_document(
            content=request.content,
            file_path=request.file_path,
            file_data=request.file_data,
            extract_metadata=config.extract_metadata,
            extract_sections=config.extract_sections
        )

        if not parse_response.success:
            raise IndexingError(f"Parsing failed: {parse_response.errors}")

        parsing_time = time.time() - parse_start

        # Step 2: Chunk document
        progress.current_operation = "chunking"
        if progress_callback:
            progress_callback(progress)

        chunk_start = time.time()
        from app.services.chunking.base import ChunkingRequest

        chunking_request = ChunkingRequest(
            content=parse_response.document.content,
            language=parse_response.document.language or "unknown",
            file_path=request.file_path,
            purpose="indexing"
        )

        chunk_response = await chunking_service.chunk_content(chunking_request)

        if not chunk_response.chunks:
            raise IndexingError("Chunking failed: No chunks created")

        chunking_time = time.time() - chunk_start

        # Step 3: Generate embeddings
        progress.current_operation = "embedding"
        if progress_callback:
            progress_callback(progress)

        embed_start = time.time()

        # Extract text from chunks
        chunk_texts = [chunk.content for chunk in chunk_response.chunks]

        # Import embedding service locally to avoid early import
        from ..embedding.service import embedding_service

        # Generate embeddings for all chunks
        embeddings = await embedding_service.embed_batch(
            texts=chunk_texts,
            embedding_type=config.embedding_type
        )

        embedding_time = time.time() - embed_start

        # Step 4: Store in vector database
        progress.current_operation = "storing"
        if progress_callback:
            progress_callback(progress)

        store_start = time.time()

        # Create vector points
        vector_points = []
        for i, (chunk, embedding) in enumerate(zip(chunk_response.chunks, embeddings)):
            point_id = f"{request.document_id}_chunk_{i}"

            # Combine metadata
            chunk_metadata = chunk.metadata.__dict__ if hasattr(chunk, 'metadata') and chunk.metadata else {}
            point_metadata = {
                "document_id": request.document_id,
                "chunk_index": i,
                "chunk_type": chunk.chunk_type.value,
                "content": chunk.content,
                "language": chunk.language,
                **chunk_metadata,
                **request.metadata
            }

            point = VectorPoint(
                id=point_id,
                vector=embedding,
                metadata=point_metadata
            )
            vector_points.append(point)

        # Store vectors
        await vectordb_service.insert_points(
            collection_name=config.collection_name,
            points=vector_points,
            ensure_collection=False  # Already ensured
        )

        storage_time = time.time() - store_start
        total_time = time.time() - start_time

        # Update shared content as processed if using shared content
        from app.services.database.repositories.shared_content_repository import SharedContentRepository
        from app.services.database.repositories.document_repository import DocumentRepository
        from app.core.database import get_session

        async with get_session() as session:
            doc_repo = DocumentRepository(session)
            document = await doc_repo.get_by_id(request.document_id)

            if document and document.shared_content_id:
                # Update shared content as processed
                shared_content_repo = SharedContentRepository(session)
                await shared_content_repo.mark_as_processed(
                    shared_content_id=document.shared_content_id,
                    chunk_count=len(chunk_response.chunks),
                    embedding_count=len(embeddings),
                    quality_score=chunk_response.quality_score,
                    chunking_quality_score=chunk_response.quality_score
                )

                # Store chunks in shared content if not already stored
                existing_chunks = await shared_content_repo.get_chunks(document.shared_content_id)
                if not existing_chunks:
                    for i, (chunk, embedding) in enumerate(zip(chunk_response.chunks, embeddings)):
                        point_id = f"{document.shared_content_id}_chunk_{i}"
                        await shared_content_repo.add_chunk(
                            shared_content_id=document.shared_content_id,
                            chunk_index=i,
                            content=chunk.content,
                            chunk_type=chunk.chunk_type.value,
                            quality_score=chunk.quality_score if hasattr(chunk, 'quality_score') else None,
                            vector_id=point_id,
                            embedding_model="text-embedding-004",
                            metadata=chunk.metadata.__dict__ if hasattr(chunk, 'metadata') and chunk.metadata else {}
                        )

        # Register document with index manager
        await index_manager.register_document(
            document_id=request.document_id,
            collection_name=config.collection_name,
            metadata={
                **parse_response.document.metadata,
                **request.metadata,
                "parsing_time": parsing_time,
                "chunking_time": chunking_time,
                "embedding_time": embedding_time,
                "storage_time": storage_time,
                "total_time": total_time
            },
            chunk_count=len(chunk_response.chunks),
            file_path=request.file_path
        )

        # Update progress
        progress.processed_documents += 1
        progress.successful_documents += 1
        progress.total_chunks += len(chunk_response.chunks)
        progress.total_embeddings += len(embeddings)
        progress.total_processing_time += total_time
        progress.current_operation = "completed"

        if progress_callback:
            progress_callback(progress)

        # Create indexed document
        indexed_doc = IndexedDocument(
            document_id=request.document_id,
            content=parse_response.document.content,
            chunks=[{
                "content": chunk.content,
                "chunk_type": chunk.chunk_type.value,
                "metadata": chunk.metadata
            } for chunk in chunk_response.chunks],
            embeddings=embeddings,
            metadata={
                **parse_response.document.metadata,
                **request.metadata
            },
            parsing_time=parsing_time,
            chunking_time=chunking_time,
            embedding_time=embedding_time,
            storage_time=storage_time,
            total_time=total_time,
            status=IndexingStatus.COMPLETED
        )

        logger.debug(f"Processed document {request.document_id}: {len(chunk_response.chunks)} chunks, {total_time:.2f}s")
        return indexed_doc

    async def _find_files(
        self,
        directory: Path,
        recursive: bool,
        file_patterns: Optional[List[str]],
        exclude_patterns: Optional[List[str]]
    ) -> List[Path]:
        """Find files to index in directory."""
        files = []

        # Default patterns for supported file types
        if file_patterns is None:
            file_patterns = [
                '*.py', '*.js', '*.ts', '*.java', '*.cpp', '*.c', '*.h',
                '*.cs', '*.go', '*.rs', '*.php', '*.rb', '*.kt', '*.swift',
                '*.txt', '*.md', '*.markdown', '*.pdf', '*.json', '*.yaml',
                '*.yml', '*.xml', '*.csv', '*.html'
            ]

        # Search for files
        for pattern in file_patterns:
            if recursive:
                found_files = directory.rglob(pattern)
            else:
                found_files = directory.glob(pattern)

            for file_path in found_files:
                if file_path.is_file():
                    # Check exclude patterns
                    if exclude_patterns:
                        excluded = any(file_path.match(exclude_pattern) for exclude_pattern in exclude_patterns)
                        if excluded:
                            continue

                    files.append(file_path)

        return files

    async def _filter_changed_files(self, files: List[Path]) -> List[Path]:
        """Filter files to only include those that have changed or are new."""
        changed_files = []

        for file_path in files:
            document_id = str(file_path.name)

            # Check if document is already indexed and up-to-date
            is_indexed = await index_manager.is_document_indexed(
                document_id=document_id,
                file_path=str(file_path)
            )

            if not is_indexed:
                changed_files.append(file_path)
            else:
                logger.debug(f"Skipping unchanged file: {file_path}")

        logger.info(f"Incremental indexing: {len(changed_files)}/{len(files)} files need processing")
        return changed_files

# Global service instance
indexing_service = IndexingService()
