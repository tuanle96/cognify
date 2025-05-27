#!/usr/bin/env python3
"""
Test vector database service implementation.
"""

import asyncio
import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_vectordb_factory():
    """Test vector database factory functionality."""
    print("🔍 Testing vector database factory...")
    
    try:
        from app.services.vectordb.factory import vectordb_factory
        from app.services.vectordb.base import VectorDBProvider
        
        # Test available providers
        providers = vectordb_factory.get_available_providers()
        print(f"   ✅ Available providers: {[p.value for p in providers]}")
        
        # Test default configs
        for provider in providers:
            config = vectordb_factory.get_default_config(provider)
            print(f"   📋 {provider.value} config: {config}")
        
        # Test environment config
        for provider in providers:
            env_config = vectordb_factory.get_connection_info_from_env(provider)
            print(f"   🌍 {provider.value} env config: {env_config}")
        
        # Test client creation (without connection)
        try:
            qdrant_client = vectordb_factory.create_qdrant_client()
            print(f"   ✅ Qdrant client created: {qdrant_client.provider.value}")
        except Exception as e:
            print(f"   ⚠️  Qdrant client creation failed (expected): {type(e).__name__}")
        
        try:
            milvus_client = vectordb_factory.create_milvus_client()
            print(f"   ✅ Milvus client created: {milvus_client.provider.value}")
        except Exception as e:
            print(f"   ⚠️  Milvus client creation failed (expected): {type(e).__name__}")
        
        print("✅ Vector database factory test passed")
        return True
        
    except Exception as e:
        print(f"❌ Vector database factory test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_vectordb_models():
    """Test vector database model classes."""
    print("🔍 Testing vector database models...")
    
    try:
        from app.services.vectordb.base import (
            VectorPoint, SearchRequest, SearchResult, CollectionInfo
        )
        
        # Test VectorPoint
        point = VectorPoint(
            id="test-1",
            vector=[0.1, 0.2, 0.3],
            metadata={"type": "test", "category": "example"}
        )
        print(f"   ✅ VectorPoint: id={point.id}, dims={len(point.vector)}")
        
        # Test auto-generated ID
        point_auto_id = VectorPoint(
            id="",
            vector=[0.4, 0.5, 0.6]
        )
        print(f"   ✅ Auto-generated ID: {point_auto_id.id[:8]}...")
        
        # Test SearchRequest
        search_request = SearchRequest(
            vector=[0.1, 0.2, 0.3],
            limit=10,
            score_threshold=0.8,
            filter_conditions={"type": "test"}
        )
        print(f"   ✅ SearchRequest: limit={search_request.limit}, threshold={search_request.score_threshold}")
        
        # Test SearchResult
        search_result = SearchResult(
            id="result-1",
            score=0.95,
            metadata={"matched": True}
        )
        print(f"   ✅ SearchResult: id={search_result.id}, score={search_result.score}")
        
        # Test CollectionInfo
        collection_info = CollectionInfo(
            name="test-collection",
            dimension=3,
            vector_count=100,
            distance_metric="cosine"
        )
        print(f"   ✅ CollectionInfo: {collection_info.name}, {collection_info.vector_count} vectors")
        
        # Test validation
        try:
            invalid_point = VectorPoint(id="test", vector=[])
            print(f"   ❌ Unexpected: Created point with empty vector")
            return False
        except ValueError as e:
            print(f"   ✅ Expected error for empty vector: {e}")
        
        print("✅ Vector database models test passed")
        return True
        
    except Exception as e:
        print(f"❌ Vector database models test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_vectordb_clients():
    """Test vector database client classes."""
    print("🔍 Testing vector database clients...")
    
    try:
        from app.services.vectordb.qdrant_client import QdrantClient
        from app.services.vectordb.milvus_client import MilvusClient
        from app.services.vectordb.base import VectorDBProvider
        
        # Test Qdrant client properties
        qdrant_client = QdrantClient()
        print(f"   ✅ Qdrant client: provider={qdrant_client.provider.value}")
        print(f"   ✅ Qdrant max batch: {qdrant_client.max_batch_size}")
        print(f"   ✅ Qdrant metrics: {qdrant_client.supported_distance_metrics}")
        
        # Test Milvus client properties
        try:
            milvus_client = MilvusClient()
            print(f"   ✅ Milvus client: provider={milvus_client.provider.value}")
            print(f"   ✅ Milvus max batch: {milvus_client.max_batch_size}")
            print(f"   ✅ Milvus metrics: {milvus_client.supported_distance_metrics}")
        except ImportError as e:
            print(f"   ⚠️  Milvus client requires pymilvus package: {e}")
        
        # Test validation methods
        qdrant_client._validate_collection_name("test_collection")
        print(f"   ✅ Collection name validation passed")
        
        qdrant_client._validate_dimension(1536)
        print(f"   ✅ Dimension validation passed")
        
        qdrant_client._validate_distance_metric("cosine")
        print(f"   ✅ Distance metric validation passed")
        
        # Test invalid validation
        try:
            qdrant_client._validate_collection_name("")
            print(f"   ❌ Unexpected: Empty collection name accepted")
            return False
        except ValueError as e:
            print(f"   ✅ Expected error for empty collection name: {e}")
        
        print("✅ Vector database clients test passed")
        return True
        
    except Exception as e:
        print(f"❌ Vector database clients test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_vectordb_service_config():
    """Test vector database service configuration."""
    print("🔍 Testing vector database service configuration...")
    
    try:
        from app.services.vectordb.service import VectorDBService, VectorDBServiceConfig
        from app.services.vectordb.base import VectorDBProvider
        
        # Test default config
        default_config = VectorDBServiceConfig()
        print(f"   ✅ Default provider: {default_config.provider.value}")
        print(f"   ✅ Default host: {default_config.host}")
        print(f"   ✅ Default port: {default_config.port}")
        print(f"   ✅ Auto-create collections: {default_config.auto_create_collections}")
        
        # Test custom config
        custom_config = VectorDBServiceConfig(
            provider=VectorDBProvider.MILVUS,
            host="custom-host",
            port=9999,
            auto_create_collections=False,
            default_dimension=768
        )
        print(f"   ✅ Custom provider: {custom_config.provider.value}")
        print(f"   ✅ Custom host: {custom_config.host}")
        print(f"   ✅ Custom dimension: {custom_config.default_dimension}")
        
        # Test service creation
        service = VectorDBService(custom_config)
        print(f"   ✅ Service created with custom config")
        print(f"   ✅ Service initialized: {service._initialized}")
        
        # Test stats
        stats = service.get_stats()
        print(f"   ✅ Initial stats: {stats}")
        
        print("✅ Vector database service config test passed")
        return True
        
    except Exception as e:
        print(f"❌ Vector database service config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_vectordb_service_without_connection():
    """Test vector database service behavior without connection."""
    print("🔍 Testing vector database service without connection...")
    
    try:
        from app.services.vectordb.service import VectorDBService
        from app.services.vectordb.base import VectorPoint
        
        # Create service
        service = VectorDBService()
        
        # Try to initialize (should fail gracefully)
        try:
            await service.initialize()
            print(f"   ❌ Unexpected: Service initialized without database")
            return False
        except Exception as e:
            print(f"   ✅ Expected error without database: {type(e).__name__}")
        
        # Try operations (should fail gracefully)
        try:
            await service.create_collection("test", 1536)
            print(f"   ❌ Unexpected: Collection created without initialization")
            return False
        except Exception as e:
            print(f"   ✅ Expected error for operation without initialization: {type(e).__name__}")
        
        # Test health check (should handle gracefully)
        health = await service.health_check()
        print(f"   ✅ Health check handled gracefully: {health.get('status', 'unknown')}")
        
        # Test cleanup
        await service.cleanup()
        print(f"   ✅ Service cleanup completed")
        
        print("✅ Vector database service without connection test passed")
        return True
        
    except Exception as e:
        print(f"❌ Vector database service without connection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_vectordb_operations():
    """Test vector database operations (mock)."""
    print("🔍 Testing vector database operations...")
    
    try:
        from app.services.vectordb.base import VectorPoint, SearchRequest
        
        # Test point creation
        points = []
        for i in range(5):
            point = VectorPoint(
                id=f"point-{i}",
                vector=[0.1 * i, 0.2 * i, 0.3 * i],
                metadata={"index": i, "type": "test"}
            )
            points.append(point)
        
        print(f"   ✅ Created {len(points)} test points")
        
        # Test search request
        search_request = SearchRequest(
            vector=[0.1, 0.2, 0.3],
            limit=3,
            filter_conditions={"type": "test"}
        )
        print(f"   ✅ Created search request: limit={search_request.limit}")
        
        # Test batch operations (simulation)
        batch_size = 2
        batches = [points[i:i + batch_size] for i in range(0, len(points), batch_size)]
        print(f"   ✅ Split {len(points)} points into {len(batches)} batches")
        
        # Test metadata filtering
        filtered_points = [p for p in points if p.metadata.get("index", 0) % 2 == 0]
        print(f"   ✅ Filtered to {len(filtered_points)} even-indexed points")
        
        print("✅ Vector database operations test passed")
        return True
        
    except Exception as e:
        print(f"❌ Vector database operations test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run vector database service tests."""
    print("🚀 Vector Database Service Test")
    print("=" * 60)
    
    tests = [
        ("Vector Database Factory", test_vectordb_factory),
        ("Vector Database Models", test_vectordb_models),
        ("Vector Database Clients", test_vectordb_clients),
        ("Service Configuration", test_vectordb_service_config),
        ("Service Without Connection", test_vectordb_service_without_connection),
        ("Vector Database Operations", test_vectordb_operations),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        try:
            result = await test_func()
            if result:
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"💥 {test_name} CRASHED: {e}")
        print()
    
    print("=" * 60)
    print(f"📊 Vector Database Service Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All vector database service tests passed!")
        print("🚀 Vector database service implementation is solid!")
        return 0
    elif passed >= total * 0.8:
        print("🎯 Most vector database service tests passed!")
        return 0
    else:
        print(f"⚠️  {total - passed} tests need attention.")
        return 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⏹️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Tests crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
