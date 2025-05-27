"""
Document management API endpoints.

Handles document upload, indexing, retrieval, and management operations.
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import uuid
import structlog

from app.api.dependencies import (
    get_indexing_service,
    get_parsing_service,
    get_current_verified_user_from_db,
    get_document_repository,
    get_collection_repository
)
from app.api.models.documents import (
    DocumentUploadRequest,
    DocumentResponse,
    DocumentListResponse,
    DocumentMetadata,
    IndexingJobResponse,
    BatchUploadResponse
)
from app.services.indexing.service import IndexingService
from app.services.indexing.base import IndexingRequest, IndexingConfig
from app.services.parsers.service import ParsingService
from app.services.parsers.base import DocumentType
from app.services.database.repositories import DocumentRepository, CollectionRepository
from app.models.users import User
from app.models.documents import DocumentType as DBDocumentType, DocumentStatus
from app.core.exceptions import ValidationError, ConflictError, NotFoundError

logger = structlog.get_logger(__name__)
router = APIRouter()


@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    collection_name: str = Form(...),
    document_type: Optional[str] = Form(None),
    extract_metadata: bool = Form(True),
    extract_sections: bool = Form(True),
    chunk_size: int = Form(500),
    chunk_overlap: int = Form(50),
    indexing_service: IndexingService = Depends(get_indexing_service),
    current_user: User = Depends(get_current_verified_user_from_db),
    document_repo: DocumentRepository = Depends(get_document_repository),
    collection_repo: CollectionRepository = Depends(get_collection_repository)
):
    """
    Upload and index a single document.

    Supports various file formats including:
    - Code files (.py, .js, .ts, .java, etc.)
    - Documents (.pdf, .md, .txt)
    - Structured data (.json, .yaml, .xml)
    """
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")

        # Read file content
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Empty file")

        # Decode content
        try:
            content_str = content.decode('utf-8')
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="File must be UTF-8 encoded text")

        # Determine document type
        if document_type:
            try:
                doc_type = DocumentType(document_type.lower())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Unsupported document type: {document_type}")
        else:
            # Auto-detect from file extension
            doc_type = _detect_document_type(file.filename)

        # Map to database document type
        db_doc_type = _map_to_db_document_type(doc_type)

        # Verify collection exists and user has write access
        collection = await collection_repo.get_by_name_and_owner(collection_name, current_user.id)
        if not collection:
            # Check if user has write access to existing collection
            collections = await collection_repo.get_user_collections(current_user.id)
            collection = next((c for c in collections if c.name == collection_name), None)
            if not collection:
                raise HTTPException(status_code=404, detail="Collection not found or access denied")

        # Check write permission
        has_write_permission = await collection_repo.check_permission(
            collection.id, current_user.id, "write"
        )
        if not has_write_permission:
            raise HTTPException(status_code=403, detail="Write permission required for this collection")

        # Create document in database
        document = await document_repo.create_document(
            filename=file.filename,
            content=content_str,
            user_id=current_user.id,
            collection_id=collection.id,
            document_type=db_doc_type
        )

        # Create indexing request
        indexing_config = IndexingConfig(
            collection_name=collection_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            extract_metadata=extract_metadata,
            extract_sections=extract_sections
        )

        indexing_request = IndexingRequest(
            content=content_str,
            document_id=document.id,
            file_path=file.filename,
            config=indexing_config,
            metadata={
                "uploaded_by": current_user.id,
                "upload_time": document.created_at.isoformat(),
                "original_filename": file.filename,
                "file_size": len(content),
                "document_type": doc_type.value
            }
        )

        # Start indexing in background
        background_tasks.add_task(
            _process_document_indexing,
            indexing_service,
            indexing_request,
            document_repo,
            document.id
        )

        logger.info(
            "Document upload initiated",
            document_id=document.id,
            filename=file.filename,
            user_id=current_user.id,
            collection=collection_name
        )

        return DocumentResponse(
            document_id=document.id,
            filename=file.filename,
            document_type=doc_type.value,
            status="processing",
            metadata={
                "file_size": document.file_size,
                "document_type": document.document_type.value,
                "created_at": document.created_at.isoformat()
            },
            upload_time=document.created_at,
            collection_name=collection_name
        )

    except Exception as e:
        logger.error("Document upload failed", error=str(e), filename=file.filename)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.post("/upload/batch", response_model=BatchUploadResponse)
async def upload_documents_batch(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    collection_name: str = Form(...),
    chunk_size: int = Form(500),
    chunk_overlap: int = Form(50),
    indexing_service: IndexingService = Depends(get_indexing_service),
    current_user: User = Depends(get_current_verified_user_from_db),
    document_repo: DocumentRepository = Depends(get_document_repository),
    collection_repo: CollectionRepository = Depends(get_collection_repository)
):
    """
    Upload and index multiple documents in batch.

    Processes files in parallel for better performance.
    """
    try:
        if len(files) == 0:
            raise HTTPException(status_code=400, detail="No files provided")

        if len(files) > 50:  # Limit batch size
            raise HTTPException(status_code=400, detail="Too many files. Maximum 50 files per batch.")

        # Verify collection exists and user has write access
        collection = await collection_repo.get_by_name_and_owner(collection_name, current_user.id)
        if not collection:
            # Check if user has write access to existing collection
            collections = await collection_repo.get_user_collections(current_user.id)
            collection = next((c for c in collections if c.name == collection_name), None)
            if not collection:
                raise HTTPException(status_code=404, detail="Collection not found or access denied")

        # Check write permission
        has_write_permission = await collection_repo.check_permission(
            collection.id, current_user.id, "write"
        )
        if not has_write_permission:
            raise HTTPException(status_code=403, detail="Write permission required for this collection")

        batch_id = str(uuid.uuid4())
        document_responses = []
        failed_files = []

        for file in files:
            if not file.filename:
                failed_files.append({"filename": "unnamed", "error": "No filename provided"})
                continue

            try:
                # Read file content
                content = await file.read()
                if len(content) == 0:
                    failed_files.append({"filename": file.filename, "error": "Empty file"})
                    continue

                # Decode content
                try:
                    content_str = content.decode('utf-8')
                except UnicodeDecodeError:
                    failed_files.append({"filename": file.filename, "error": "File must be UTF-8 encoded text"})
                    continue

                # Auto-detect document type
                doc_type = _detect_document_type(file.filename)
                db_doc_type = _map_to_db_document_type(doc_type)

                # Create document in database
                document = await document_repo.create_document(
                    filename=file.filename,
                    content=content_str,
                    user_id=current_user.id,
                    collection_id=collection.id,
                    document_type=db_doc_type
                )

                # Create indexing request
                indexing_config = IndexingConfig(
                    collection_name=collection_name,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )

                indexing_request = IndexingRequest(
                    content=content_str,
                    document_id=document.id,
                    file_path=file.filename,
                    config=indexing_config,
                    metadata={
                        "uploaded_by": current_user.id,
                        "upload_time": document.created_at.isoformat(),
                        "original_filename": file.filename,
                        "file_size": len(content),
                        "document_type": doc_type.value,
                        "batch_id": batch_id
                    }
                )

                # Start indexing in background
                background_tasks.add_task(
                    _process_document_indexing,
                    indexing_service,
                    indexing_request,
                    document_repo,
                    document.id
                )

                document_responses.append(DocumentResponse(
                    document_id=document.id,
                    filename=file.filename,
                    document_type=doc_type.value,
                    status="processing",
                    metadata={
                        "file_size": document.file_size,
                        "batch_id": batch_id,
                        "created_at": document.created_at.isoformat()
                    },
                    upload_time=document.created_at,
                    collection_name=collection_name
                ))

            except Exception as e:
                failed_files.append({"filename": file.filename, "error": str(e)})
                logger.error("Failed to process file in batch", filename=file.filename, error=str(e))

        logger.info(
            "Batch upload initiated",
            batch_id=batch_id,
            file_count=len(document_responses),
            failed_count=len(failed_files),
            user_id=current_user.id,
            collection=collection_name
        )

        return BatchUploadResponse(
            batch_id=batch_id,
            total_files=len(files),
            processed_files=len(document_responses),
            failed_files=len(failed_files),
            documents=document_responses,
            status="processing",
            errors=failed_files if failed_files else None
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Batch upload failed", error=str(e))
        raise HTTPException(status_code=500, detail="Batch upload failed")


@router.get("/", response_model=DocumentListResponse)
async def list_documents(
    collection_name: Optional[str] = None,
    document_type: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    current_user: User = Depends(get_current_verified_user_from_db),
    document_repo: DocumentRepository = Depends(get_document_repository)
):
    """
    List documents with optional filtering.

    Supports filtering by collection, document type, and status.
    """
    try:
        # Build filters
        status_filter = None
        if status:
            try:
                status_filter = DocumentStatus(status.lower())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid status: {status}")

        # Get documents by user
        documents = await document_repo.get_by_user(
            user_id=current_user.id,
            offset=offset,
            limit=limit,
            status=status_filter
        )

        # Convert to response format
        document_responses = []
        for doc in documents:
            document_responses.append(DocumentResponse(
                document_id=doc.id,
                filename=doc.filename,
                document_type=doc.document_type.value,
                status=doc.status.value,
                metadata={
                    "file_size": doc.file_size,
                    "chunk_count": doc.chunk_count,
                    "processing_stage": doc.processing_stage.value if doc.processing_stage else None
                },
                upload_time=doc.created_at,
                collection_name=collection_name  # TODO: Get actual collection name
            ))

        # Get total count
        total_count = await document_repo.count(filters={"created_by": current_user.id})

        logger.info(
            "Documents listed",
            user_id=current_user.id,
            count=len(documents),
            total=total_count
        )

        # Calculate pagination info
        page = (offset // limit) + 1
        has_next = (offset + limit) < total_count

        return DocumentListResponse(
            documents=document_responses,
            total=total_count,
            page=page,
            per_page=limit,
            has_next=has_next
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to list documents", error=str(e), user_id=current_user.id)
        raise HTTPException(status_code=500, detail="Failed to list documents")


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: str,
    current_user: User = Depends(get_current_verified_user_from_db),
    document_repo: DocumentRepository = Depends(get_document_repository)
):
    """
    Get detailed information about a specific document.
    """
    try:
        # Get document from database
        document = await document_repo.get_by_id(document_id)

        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        # Check if user owns the document or has access
        if document.created_by != current_user.id:
            # TODO: Check collection permissions
            raise HTTPException(status_code=403, detail="Access denied")

        return DocumentResponse(
            document_id=document.id,
            filename=document.filename,
            document_type=document.document_type.value,
            status=document.status.value,
            metadata={
                "file_size": document.file_size,
                "chunk_count": document.chunk_count,
                "processing_stage": document.processing_stage.value if document.processing_stage else None,
                "processing_time": document.processing_time,
                "created_at": document.created_at.isoformat()
            },
            upload_time=document.created_at,
            collection_name=None  # TODO: Get collection name
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get document", document_id=document_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get document")


@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    current_user: User = Depends(get_current_verified_user_from_db),
    document_repo: DocumentRepository = Depends(get_document_repository)
):
    """
    Delete a document and remove it from all collections.
    """
    try:
        # Get document and verify ownership
        document = await document_repo.get_by_id(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        # Check if user owns the document
        if document.created_by != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")

        # Delete document (soft delete)
        success = await document_repo.delete(document_id)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete document")

        # TODO: Remove from vector database

        logger.info(
            "Document deleted",
            document_id=document_id,
            user_id=current_user.id
        )

        return {"message": "Document deleted successfully", "document_id": document_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete document", document_id=document_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to delete document")


# Helper functions

def _detect_document_type(filename: str) -> DocumentType:
    """Detect document type from filename extension."""
    extension = filename.lower().split('.')[-1] if '.' in filename else ''

    type_mapping = {
        'py': DocumentType.CODE,
        'js': DocumentType.CODE,
        'ts': DocumentType.CODE,
        'java': DocumentType.CODE,
        'cpp': DocumentType.CODE,
        'c': DocumentType.CODE,
        'h': DocumentType.CODE,
        'md': DocumentType.MARKDOWN,
        'markdown': DocumentType.MARKDOWN,
        'txt': DocumentType.TEXT,
        'pdf': DocumentType.PDF,
        'json': DocumentType.JSON,
        'yaml': DocumentType.YAML,
        'yml': DocumentType.YAML,
        'xml': DocumentType.XML,
        'html': DocumentType.HTML,
        'csv': DocumentType.CSV
    }

    return type_mapping.get(extension, DocumentType.UNKNOWN)


async def _process_document_indexing(
    indexing_service: IndexingService,
    request: IndexingRequest,
    document_repo: DocumentRepository,
    document_id: str
):
    """Background task to process document indexing."""
    try:
        # Update status to processing
        await document_repo.update_processing_status(
            document_id,
            DocumentStatus.PROCESSING
        )

        # Process indexing
        response = await indexing_service.index_document(request)

        if response.success:
            # Update status to completed
            await document_repo.update_processing_status(
                document_id,
                DocumentStatus.COMPLETED,
                metadata={
                    "chunks_created": response.total_chunks_created,
                    "processing_time": response.total_processing_time,
                    "embeddings_generated": response.total_embeddings_generated
                }
            )
            logger.info("Document indexing completed", document_id=document_id)
        else:
            # Update status to failed
            await document_repo.update_processing_status(
                document_id,
                DocumentStatus.FAILED,
                error_message=f"Indexing failed: {'; '.join(response.errors)}"
            )
            logger.error("Document indexing failed", document_id=document_id, errors=response.errors)

    except Exception as e:
        # Update status to failed
        await document_repo.update_processing_status(
            document_id,
            DocumentStatus.FAILED,
            error_message=str(e)
        )
        logger.error("Document indexing crashed", document_id=document_id, error=str(e))


def _map_to_db_document_type(doc_type: DocumentType) -> DBDocumentType:
    """Map service document type to database document type."""
    mapping = {
        DocumentType.CODE: DBDocumentType.CODE,
        DocumentType.MARKDOWN: DBDocumentType.MARKDOWN,
        DocumentType.TEXT: DBDocumentType.TEXT,
        DocumentType.PDF: DBDocumentType.PDF,
        DocumentType.JSON: DBDocumentType.JSON,
        DocumentType.YAML: DBDocumentType.YAML,
        DocumentType.XML: DBDocumentType.XML,
        DocumentType.HTML: DBDocumentType.HTML,
        DocumentType.CSV: DBDocumentType.CSV,
        DocumentType.UNKNOWN: DBDocumentType.UNKNOWN
    }
    return mapping.get(doc_type, DBDocumentType.UNKNOWN)



