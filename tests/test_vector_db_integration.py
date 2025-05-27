"""
Integration tests for vector database operations with real Qdrant.
"""
import pytest
import asyncio
import random
from typing import List, Dict, Any

from app.services.vectordb.qdrant_service import QdrantService
from app.services.embedding.openai_client import OpenAIEmbeddingClient


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def qdrant_service():
    """Create Qdrant service for testing."""
    service = QdrantService(
        url="http://localhost:6333",
        api_key=None,
        timeout=30
    )

    # Wait for Qdrant to be ready
    await asyncio.sleep(2)

    yield service

    # Cleanup test collections
    try:
        collections = await service.list_collections()
        for collection in collections:
            if collection.startswith("test_"):
                await service.delete_collection(collection)
    except Exception:
        pass  # Ignore cleanup errors


@pytest.fixture
def embedding_service():
    """Create embedding service for testing."""
    return OpenAIEmbeddingClient(
        api_key="test-api-key-for-testing",
        base_url="https://ai.earnbase.io/v1"
    )


class TestQdrantConnection:
    """Test Qdrant connection and basic operations."""

    async def test_health_check(self, qdrant_service):
        """Test Qdrant health check."""
        health = await qdrant_service.health_check()

        assert health is not None
        assert health["status"] in ["healthy", "ok"]

    async def test_create_collection(self, qdrant_service):
        """Test creating a collection."""
        collection_name = "test_create_collection"
        dimension = 1536

        success = await qdrant_service.create_collection(
            name=collection_name,
            dimension=dimension
        )

        assert success is True

        # Verify collection exists
        collections = await qdrant_service.list_collections()
        assert collection_name in collections

        # Cleanup
        await qdrant_service.delete_collection(collection_name)

    async def test_delete_collection(self, qdrant_service):
        """Test deleting a collection."""
        collection_name = "test_delete_collection"

        # Create collection first
        await qdrant_service.create_collection(collection_name, 1536)

        # Delete collection
        success = await qdrant_service.delete_collection(collection_name)
        assert success is True

        # Verify collection is deleted
        collections = await qdrant_service.list_collections()
        assert collection_name not in collections


class TestVectorOperations:
    """Test vector operations with real embeddings."""

    @pytest.fixture
    async def test_collection(self, qdrant_service):
        """Create a test collection."""
        collection_name = "test_vectors"
        await qdrant_service.create_collection(collection_name, 1536)
        yield collection_name
        await qdrant_service.delete_collection(collection_name)

    async def test_insert_vectors(self, qdrant_service, embedding_service, test_collection):
        """Test inserting vectors."""
        # Generate test embeddings
        texts = [
            "Python is a programming language",
            "JavaScript is used for web development",
            "Machine learning algorithms learn patterns"
        ]

        embeddings = await embedding_service.embed_texts(texts)

        # Prepare vectors for insertion
        vectors = {}
        metadata = {}
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            vector_id = f"doc_{i}"
            vectors[vector_id] = embedding
            metadata[vector_id] = {
                "text": text,
                "type": "test_document",
                "index": i
            }

        # Insert vectors
        success = await qdrant_service.insert_vectors(
            collection=test_collection,
            vectors=vectors,
            metadata=metadata
        )

        assert success is True

        # Verify vectors were inserted
        info = await qdrant_service.get_collection_info(test_collection)
        assert info["vectors_count"] == 3

    async def test_search_vectors(self, qdrant_service, embedding_service, test_collection):
        """Test vector search."""
        # Insert test data first
        texts = [
            "Python programming language syntax",
            "JavaScript web development framework",
            "Machine learning neural networks",
            "Database query optimization",
            "API design best practices"
        ]

        embeddings = await embedding_service.embed_texts(texts)

        vectors = {}
        metadata = {}
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            vector_id = f"search_doc_{i}"
            vectors[vector_id] = embedding
            metadata[vector_id] = {
                "text": text,
                "category": "programming" if i < 3 else "general",
                "index": i
            }

        await qdrant_service.insert_vectors(test_collection, vectors, metadata)

        # Search for similar content
        query_text = "Python programming concepts"
        query_embedding = await embedding_service.embed_single(query_text)

        results = await qdrant_service.search_vectors(
            collection=test_collection,
            query_vector=query_embedding,
            limit=3
        )

        assert len(results) > 0
        assert len(results) <= 3

        # Results should be sorted by score (descending)
        scores = [result["score"] for result in results]
        assert scores == sorted(scores, reverse=True)

        # Top result should be Python-related
        top_result = results[0]
        assert "python" in top_result["metadata"]["text"].lower()

    async def test_search_with_filters(self, qdrant_service, embedding_service, test_collection):
        """Test vector search with metadata filters."""
        # Insert test data with categories
        texts = [
            "Python web framework Django",
            "JavaScript React components",
            "Python data science pandas",
            "JavaScript Node.js backend",
            "Python machine learning scikit-learn"
        ]

        embeddings = await embedding_service.embed_texts(texts)

        vectors = {}
        metadata = {}
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            vector_id = f"filter_doc_{i}"
            vectors[vector_id] = embedding
            metadata[vector_id] = {
                "text": text,
                "language": "python" if "Python" in text else "javascript",
                "domain": "web" if "web" in text.lower() or "React" in text or "Node" in text else "data",
                "index": i
            }

        await qdrant_service.insert_vectors(test_collection, vectors, metadata)

        # Search with language filter
        query_text = "web development framework"
        query_embedding = await embedding_service.embed_single(query_text)

        # Search only Python documents
        python_results = await qdrant_service.search_vectors(
            collection=test_collection,
            query_vector=query_embedding,
            limit=5,
            filters={"language": "python"}
        )

        assert len(python_results) > 0
        for result in python_results:
            assert result["metadata"]["language"] == "python"

        # Search only JavaScript documents
        js_results = await qdrant_service.search_vectors(
            collection=test_collection,
            query_vector=query_embedding,
            limit=5,
            filters={"language": "javascript"}
        )

        assert len(js_results) > 0
        for result in js_results:
            assert result["metadata"]["language"] == "javascript"

    async def test_update_vectors(self, qdrant_service, embedding_service, test_collection):
        """Test updating vectors."""
        # Insert initial vector
        text = "Original text content"
        embedding = await embedding_service.embed_single(text)

        vector_id = "update_test"
        vectors = {vector_id: embedding}
        metadata = {vector_id: {"text": text, "version": 1}}

        await qdrant_service.insert_vectors(test_collection, vectors, metadata)

        # Update with new content
        new_text = "Updated text content with new information"
        new_embedding = await embedding_service.embed_single(new_text)

        new_vectors = {vector_id: new_embedding}
        new_metadata = {vector_id: {"text": new_text, "version": 2}}

        success = await qdrant_service.insert_vectors(
            test_collection, new_vectors, new_metadata
        )
        assert success is True

        # Search to verify update
        results = await qdrant_service.search_vectors(
            collection=test_collection,
            query_vector=new_embedding,
            limit=1
        )

        assert len(results) == 1
        assert results[0]["id"] == vector_id
        assert results[0]["metadata"]["version"] == 2
        assert "Updated" in results[0]["metadata"]["text"]

    async def test_delete_vectors(self, qdrant_service, embedding_service, test_collection):
        """Test deleting specific vectors."""
        # Insert test vectors
        texts = ["Document 1", "Document 2", "Document 3"]
        embeddings = await embedding_service.embed_texts(texts)

        vectors = {}
        metadata = {}
        vector_ids = []
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            vector_id = f"delete_doc_{i}"
            vector_ids.append(vector_id)
            vectors[vector_id] = embedding
            metadata[vector_id] = {"text": text}

        await qdrant_service.insert_vectors(test_collection, vectors, metadata)

        # Delete specific vectors
        ids_to_delete = vector_ids[:2]  # Delete first two
        success = await qdrant_service.delete_vectors(test_collection, ids_to_delete)
        assert success is True

        # Verify deletion
        info = await qdrant_service.get_collection_info(test_collection)
        assert info["vectors_count"] == 1  # Only one should remain

        # Search should only return the remaining vector
        query_embedding = embeddings[2]  # Embedding of the remaining document
        results = await qdrant_service.search_vectors(
            collection=test_collection,
            query_vector=query_embedding,
            limit=5
        )

        assert len(results) == 1
        assert results[0]["id"] == vector_ids[2]


class TestPerformance:
    """Test vector database performance."""

    @pytest.fixture
    async def perf_collection(self, qdrant_service):
        """Create a performance test collection."""
        collection_name = "test_performance"
        await qdrant_service.create_collection(collection_name, 1536)
        yield collection_name
        await qdrant_service.delete_collection(collection_name)

    async def test_bulk_insert_performance(self, qdrant_service, perf_collection):
        """Test bulk insert performance."""
        import time

        # Generate test data
        num_vectors = 100
        dimension = 1536

        vectors = {}
        metadata = {}
        for i in range(num_vectors):
            vector_id = f"perf_doc_{i}"
            # Generate random vector
            vector = [random.random() for _ in range(dimension)]
            vectors[vector_id] = vector
            metadata[vector_id] = {
                "text": f"Performance test document {i}",
                "category": f"category_{i % 5}",
                "index": i
            }

        # Measure insert time
        start_time = time.time()
        success = await qdrant_service.insert_vectors(perf_collection, vectors, metadata)
        end_time = time.time()

        insert_time = end_time - start_time

        assert success is True
        assert insert_time < 10.0  # Should complete within 10 seconds

        # Verify all vectors were inserted
        info = await qdrant_service.get_collection_info(perf_collection)
        assert info["vectors_count"] == num_vectors

        print(f"Inserted {num_vectors} vectors in {insert_time:.2f} seconds")
        print(f"Rate: {num_vectors / insert_time:.2f} vectors/second")

    async def test_search_performance(self, qdrant_service, embedding_service, perf_collection):
        """Test search performance."""
        import time

        # Insert test data
        num_vectors = 50
        texts = [f"Test document {i} with content about topic {i % 10}" for i in range(num_vectors)]
        embeddings = await embedding_service.embed_texts(texts)

        vectors = {}
        metadata = {}
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            vector_id = f"search_perf_{i}"
            vectors[vector_id] = embedding
            metadata[vector_id] = {"text": text, "topic": i % 10}

        await qdrant_service.insert_vectors(perf_collection, vectors, metadata)

        # Perform multiple searches
        query_text = "test document content"
        query_embedding = await embedding_service.embed_single(query_text)

        num_searches = 10
        start_time = time.time()

        for _ in range(num_searches):
            results = await qdrant_service.search_vectors(
                collection=perf_collection,
                query_vector=query_embedding,
                limit=10
            )
            assert len(results) > 0

        end_time = time.time()
        search_time = end_time - start_time

        assert search_time < 5.0  # Should complete within 5 seconds

        print(f"Performed {num_searches} searches in {search_time:.2f} seconds")
        print(f"Rate: {num_searches / search_time:.2f} searches/second")

    async def test_concurrent_operations(self, qdrant_service, perf_collection):
        """Test concurrent vector operations."""
        import time

        async def insert_batch(batch_id, batch_size=10):
            vectors = {}
            metadata = {}
            for i in range(batch_size):
                vector_id = f"concurrent_{batch_id}_{i}"
                vector = [random.random() for _ in range(1536)]
                vectors[vector_id] = vector
                metadata[vector_id] = {"batch": batch_id, "index": i}

            return await qdrant_service.insert_vectors(perf_collection, vectors, metadata)

        # Run concurrent inserts
        num_batches = 5
        start_time = time.time()

        tasks = [insert_batch(i) for i in range(num_batches)]
        results = await asyncio.gather(*tasks)

        end_time = time.time()
        concurrent_time = end_time - start_time

        assert all(results)  # All inserts should succeed
        assert concurrent_time < 10.0  # Should complete within 10 seconds

        # Verify all vectors were inserted
        info = await qdrant_service.get_collection_info(perf_collection)
        expected_count = num_batches * 10
        assert info["vectors_count"] == expected_count

        print(f"Concurrent insert of {expected_count} vectors in {concurrent_time:.2f} seconds")
        print(f"Rate: {expected_count / concurrent_time:.2f} vectors/second")
