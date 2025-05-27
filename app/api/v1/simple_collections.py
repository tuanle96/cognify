
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict, Any
import uuid
from datetime import datetime

router = APIRouter()

# Simple in-memory storage
collections = {}
documents = {}

class SimpleCollection(BaseModel):
    name: str
    description: str = ""

class SimpleDocument(BaseModel):
    title: str
    content: str
    content_type: str = "text"
    collection_id: str
    metadata: Dict[str, Any] = {}

class CollectionResponse(BaseModel):
    id: str
    name: str
    description: str
    created_at: str

class DocumentResponse(BaseModel):
    id: str
    title: str
    content: str
    collection_id: str
    created_at: str

@router.post("/simple-collections", response_model=CollectionResponse)
async def create_simple_collection(collection: SimpleCollection):
    """Create collection without auth."""
    collection_id = str(uuid.uuid4())
    collections[collection_id] = {
        "id": collection_id,
        "name": collection.name,
        "description": collection.description,
        "created_at": datetime.utcnow().isoformat()
    }
    return CollectionResponse(**collections[collection_id])

@router.get("/simple-collections", response_model=List[CollectionResponse])
async def list_simple_collections():
    """List collections without auth."""
    return [CollectionResponse(**col) for col in collections.values()]

@router.post("/simple-documents", response_model=DocumentResponse)
async def create_simple_document(document: SimpleDocument):
    """Create document without auth."""
    document_id = str(uuid.uuid4())
    documents[document_id] = {
        "id": document_id,
        "title": document.title,
        "content": document.content,
        "collection_id": document.collection_id,
        "created_at": datetime.utcnow().isoformat()
    }
    return DocumentResponse(**documents[document_id])

@router.get("/simple-documents", response_model=List[DocumentResponse])
async def list_simple_documents(collection_id: str = None):
    """List documents without auth."""
    docs = documents.values()
    if collection_id:
        docs = [doc for doc in docs if doc["collection_id"] == collection_id]
    return [DocumentResponse(**doc) for doc in docs]
