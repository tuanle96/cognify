"""
Index manager for collection and metadata management.
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Set
from pathlib import Path
import logging
import json
import hashlib

from .base import IndexingConfig, IndexingError, IndexingResourceError
from ..vectordb.service import vectordb_service
from ..vectordb.base import CollectionInfo, VectorPoint

logger = logging.getLogger(__name__)

class IndexManager:
    """Manager for vector database collections and indexing metadata."""
    
    def __init__(self):
        self._initialized = False
        self._collections_cache: Dict[str, CollectionInfo] = {}
        self._document_registry: Dict[str, Dict[str, Any]] = {}
        self._stats = {
            "collections_managed": 0,
            "documents_tracked": 0,
            "total_vectors": 0,
            "last_update": None
        }
    
    async def initialize(self) -> None:
        """Initialize the index manager."""
        try:
            # Initialize vector database service
            if not vectordb_service._initialized:
                await vectordb_service.initialize()
            
            # Load existing collections
            await self._load_collections()
            
            # Load document registry
            await self._load_document_registry()
            
            self._initialized = True
            logger.info("Index manager initialized")
            
        except Exception as e:
            raise IndexingError(f"Failed to initialize index manager: {e}") from e
    
    async def create_collection(
        self,
        name: str,
        dimension: int,
        distance_metric: str = "cosine",
        overwrite: bool = False
    ) -> bool:
        """
        Create a new vector collection.
        
        Args:
            name: Collection name
            dimension: Vector dimension
            distance_metric: Distance metric for similarity
            overwrite: Whether to overwrite existing collection
        
        Returns:
            True if collection was created successfully
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Check if collection exists
            exists = await vectordb_service.client.collection_exists(name)
            
            if exists and not overwrite:
                logger.info(f"Collection {name} already exists")
                return True
            
            if exists and overwrite:
                logger.info(f"Deleting existing collection {name}")
                await vectordb_service.delete_collection(name)
            
            # Create new collection
            success = await vectordb_service.create_collection(
                name=name,
                dimension=dimension,
                distance_metric=distance_metric
            )
            
            if success:
                # Update cache
                collection_info = await vectordb_service.get_collection_info(name)
                self._collections_cache[name] = collection_info
                self._stats["collections_managed"] += 1
                
                logger.info(f"Created collection: {name} (dim={dimension})")
            
            return success
            
        except Exception as e:
            raise IndexingResourceError(f"Failed to create collection {name}: {e}", "collection")
    
    async def ensure_collection(
        self,
        config: IndexingConfig
    ) -> bool:
        """
        Ensure collection exists with the specified configuration.
        
        Args:
            config: Indexing configuration
        
        Returns:
            True if collection is ready
        """
        return await self.create_collection(
            name=config.collection_name,
            dimension=config.vector_dimension,
            distance_metric=config.distance_metric,
            overwrite=config.overwrite_existing
        )
    
    async def get_collection_info(self, name: str) -> Optional[CollectionInfo]:
        """Get information about a collection."""
        if not self._initialized:
            await self.initialize()
        
        # Check cache first
        if name in self._collections_cache:
            return self._collections_cache[name]
        
        try:
            # Get from vector database
            if await vectordb_service.client.collection_exists(name):
                info = await vectordb_service.get_collection_info(name)
                self._collections_cache[name] = info
                return info
        except Exception as e:
            logger.warning(f"Failed to get collection info for {name}: {e}")
        
        return None
    
    async def list_collections(self) -> List[str]:
        """List all managed collections."""
        if not self._initialized:
            await self.initialize()
        
        try:
            return await vectordb_service.list_collections()
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []
    
    async def delete_collection(self, name: str) -> bool:
        """Delete a collection and its metadata."""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Delete from vector database
            success = await vectordb_service.delete_collection(name)
            
            if success:
                # Remove from cache
                self._collections_cache.pop(name, None)
                
                # Remove documents from registry
                docs_to_remove = [
                    doc_id for doc_id, doc_info in self._document_registry.items()
                    if doc_info.get("collection") == name
                ]
                
                for doc_id in docs_to_remove:
                    del self._document_registry[doc_id]
                
                logger.info(f"Deleted collection: {name}")
            
            return success
            
        except Exception as e:
            raise IndexingResourceError(f"Failed to delete collection {name}: {e}", "collection")
    
    async def register_document(
        self,
        document_id: str,
        collection_name: str,
        metadata: Dict[str, Any],
        chunk_count: int,
        file_path: Optional[str] = None
    ) -> None:
        """
        Register a document in the index registry.
        
        Args:
            document_id: Unique document identifier
            collection_name: Collection where document is stored
            metadata: Document metadata
            chunk_count: Number of chunks created
            file_path: Source file path if applicable
        """
        if not self._initialized:
            await self.initialize()
        
        # Calculate content hash for change detection
        content_hash = None
        if file_path and Path(file_path).exists():
            content_hash = self._calculate_file_hash(file_path)
        
        # Register document
        self._document_registry[document_id] = {
            "collection": collection_name,
            "metadata": metadata,
            "chunk_count": chunk_count,
            "file_path": file_path,
            "content_hash": content_hash,
            "indexed_at": time.time(),
            "last_updated": time.time()
        }
        
        self._stats["documents_tracked"] += 1
        self._stats["total_vectors"] += chunk_count
        self._stats["last_update"] = time.time()
        
        logger.debug(f"Registered document: {document_id} in {collection_name}")
    
    async def is_document_indexed(
        self,
        document_id: str,
        file_path: Optional[str] = None
    ) -> bool:
        """
        Check if a document is already indexed and up-to-date.
        
        Args:
            document_id: Document identifier
            file_path: File path for change detection
        
        Returns:
            True if document is indexed and current
        """
        if not self._initialized:
            await self.initialize()
        
        if document_id not in self._document_registry:
            return False
        
        doc_info = self._document_registry[document_id]
        
        # Check if file has changed
        if file_path and Path(file_path).exists():
            current_hash = self._calculate_file_hash(file_path)
            stored_hash = doc_info.get("content_hash")
            
            if stored_hash and current_hash != stored_hash:
                logger.debug(f"Document {document_id} has changed (hash mismatch)")
                return False
        
        return True
    
    async def get_document_info(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a registered document."""
        if not self._initialized:
            await self.initialize()
        
        return self._document_registry.get(document_id)
    
    async def list_documents(
        self,
        collection_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List registered documents.
        
        Args:
            collection_name: Filter by collection (all if None)
        
        Returns:
            List of document information
        """
        if not self._initialized:
            await self.initialize()
        
        documents = []
        for doc_id, doc_info in self._document_registry.items():
            if collection_name is None or doc_info.get("collection") == collection_name:
                doc_data = doc_info.copy()
                doc_data["document_id"] = doc_id
                documents.append(doc_data)
        
        return documents
    
    async def remove_document(
        self,
        document_id: str,
        remove_vectors: bool = True
    ) -> bool:
        """
        Remove a document from the registry and optionally from vector storage.
        
        Args:
            document_id: Document identifier
            remove_vectors: Whether to remove vectors from collection
        
        Returns:
            True if document was removed
        """
        if not self._initialized:
            await self.initialize()
        
        if document_id not in self._document_registry:
            return False
        
        doc_info = self._document_registry[document_id]
        collection_name = doc_info.get("collection")
        
        try:
            # Remove vectors from collection if requested
            if remove_vectors and collection_name:
                # Find all vector IDs for this document
                # This is a simplified approach - in practice, you'd need to track vector IDs
                logger.warning(f"Vector removal not implemented for document {document_id}")
            
            # Remove from registry
            del self._document_registry[document_id]
            
            self._stats["documents_tracked"] -= 1
            self._stats["total_vectors"] -= doc_info.get("chunk_count", 0)
            self._stats["last_update"] = time.time()
            
            logger.info(f"Removed document: {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove document {document_id}: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get index manager statistics."""
        if not self._initialized:
            await self.initialize()
        
        stats = self._stats.copy()
        
        # Add collection details
        stats["collections"] = {}
        for name, info in self._collections_cache.items():
            stats["collections"][name] = {
                "vector_count": info.vector_count,
                "dimension": info.dimension,
                "distance_metric": info.distance_metric
            }
        
        return stats
    
    async def cleanup(self) -> None:
        """Cleanup index manager resources."""
        self._collections_cache.clear()
        self._document_registry.clear()
        self._initialized = False
        logger.info("Index manager cleaned up")
    
    # Private methods
    
    async def _load_collections(self) -> None:
        """Load existing collections into cache."""
        try:
            collections = await vectordb_service.list_collections()
            
            for collection_name in collections:
                try:
                    info = await vectordb_service.get_collection_info(collection_name)
                    self._collections_cache[collection_name] = info
                    self._stats["collections_managed"] += 1
                except Exception as e:
                    logger.warning(f"Failed to load collection {collection_name}: {e}")
                    
        except Exception as e:
            logger.warning(f"Failed to load collections: {e}")
    
    async def _load_document_registry(self) -> None:
        """Load document registry from persistent storage."""
        # In a real implementation, this would load from a database or file
        # For now, we start with an empty registry
        logger.debug("Document registry initialized (empty)")
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate hash of file content for change detection."""
        try:
            path = Path(file_path)
            if not path.exists():
                return ""
            
            # Use file size and modification time for quick hash
            stat = path.stat()
            hash_input = f"{stat.st_size}:{stat.st_mtime}:{path.name}"
            
            return hashlib.md5(hash_input.encode()).hexdigest()
            
        except Exception as e:
            logger.warning(f"Failed to calculate hash for {file_path}: {e}")
            return ""

# Global manager instance
index_manager = IndexManager()
