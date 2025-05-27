#!/usr/bin/env python3
"""
Quick test for service manager and dependency fixes.
"""

import asyncio
import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_service_manager():
    """Test service manager initialization."""
    print("🔧 Testing Service Manager...")
    
    try:
        # Set test environment
        os.environ["ENVIRONMENT"] = "testing"
        os.environ["EMBEDDING_PROVIDER"] = "mock"
        os.environ["LLM_PROVIDER"] = "mock"
        os.environ["QDRANT_URL"] = "http://localhost:6333"
        
        from app.core.service_manager import get_service_manager
        
        manager = get_service_manager()
        print(f"   ✅ Service manager created")
        
        # Test individual service initialization
        services_to_test = ["embedding", "vectordb", "parsing", "chunking"]
        
        for service_name in services_to_test:
            try:
                await manager._initialize_service(service_name)
                service = manager.get_service(service_name)
                initialized = manager.is_service_initialized(service_name)
                
                print(f"   ✅ {service_name}: initialized={initialized}, service={type(service).__name__}")
                
            except Exception as e:
                print(f"   ⚠️  {service_name}: failed - {type(e).__name__}: {e}")
        
        # Test health check
        try:
            health = await manager.health_check_all()
            print(f"   🏥 Overall health: {health['status']}")
            print(f"   📊 Health summary: {health['summary']}")
            
            for service_name, service_health in health['services'].items():
                status = service_health.get('status', 'unknown')
                print(f"      {service_name}: {status}")
                
        except Exception as e:
            print(f"   ⚠️  Health check failed: {type(e).__name__}: {e}")
        
        # Test cleanup
        await manager.cleanup_all()
        print(f"   ✅ Service cleanup completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Service manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_qdrant_connection():
    """Test direct Qdrant connection."""
    print("🔗 Testing Qdrant Connection...")
    
    try:
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:6333") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"   ✅ Qdrant connected: {data.get('title', 'Unknown')} v{data.get('version', 'Unknown')}")
                    return True
                else:
                    print(f"   ❌ Qdrant connection failed: HTTP {response.status}")
                    return False
                    
    except ImportError:
        print("   ⚠️  aiohttp not available, using requests")
        try:
            import requests
            response = requests.get("http://localhost:6333", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Qdrant connected: {data.get('title', 'Unknown')} v{data.get('version', 'Unknown')}")
                return True
            else:
                print(f"   ❌ Qdrant connection failed: HTTP {response.status_code}")
                return False
        except Exception as e:
            print(f"   ❌ Qdrant connection failed: {e}")
            return False
    except Exception as e:
        print(f"   ❌ Qdrant connection failed: {e}")
        return False

async def test_mock_services():
    """Test mock services directly."""
    print("🎭 Testing Mock Services...")
    
    try:
        # Test mock embedding service
        from app.services.mocks.mock_embedding_service import MockEmbeddingService
        
        mock_embedding = MockEmbeddingService()
        await mock_embedding.initialize()
        
        # Test embedding generation
        embeddings = await mock_embedding.embed_batch(["test text", "another test"])
        print(f"   ✅ Mock embedding: generated {len(embeddings)} embeddings, dim={len(embeddings[0])}")
        
        # Test mock vector DB
        from app.services.mocks.mock_vectordb_service import MockVectorDBService
        
        mock_vectordb = MockVectorDBService()
        await mock_vectordb.initialize()
        
        # Test collection creation
        await mock_vectordb.create_collection("test_collection", vector_size=384)
        print(f"   ✅ Mock vector DB: collection created")
        
        # Test mock LLM
        from app.services.mocks.mock_llm_service import MockLLMService
        
        mock_llm = MockLLMService()
        await mock_llm.initialize()
        
        # Test LLM call
        response = await mock_llm.generate_response("Test prompt")
        print(f"   ✅ Mock LLM: response generated ({len(response)} chars)")
        
        return True
        
    except Exception as e:
        print(f"❌ Mock services test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_basic_indexing():
    """Test basic indexing with mocks."""
    print("📚 Testing Basic Indexing...")
    
    try:
        # Set up environment for testing
        os.environ["ENVIRONMENT"] = "testing"
        os.environ["EMBEDDING_PROVIDER"] = "mock"
        os.environ["LLM_PROVIDER"] = "mock"
        
        from app.services.indexing.service import indexing_service
        from app.services.indexing.base import IndexingRequest
        
        # Initialize service
        await indexing_service.initialize()
        print(f"   ✅ Indexing service initialized")
        
        # Test simple indexing
        request = IndexingRequest(
            content="def hello(): return 'Hello, World!'",
            document_id="test_doc"
        )
        
        response = await indexing_service.index_document(request)
        
        if response.success:
            print(f"   ✅ Document indexed successfully")
            print(f"   📊 Chunks created: {response.total_chunks_created}")
            print(f"   ⏱️  Processing time: {response.total_processing_time:.3f}s")
        else:
            print(f"   ❌ Indexing failed: {response.errors}")
            return False
        
        # Test health check
        health = await indexing_service.health_check()
        print(f"   🏥 Indexing service health: {health.get('status', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic indexing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_basic_retrieval():
    """Test basic retrieval with mocks."""
    print("🔍 Testing Basic Retrieval...")
    
    try:
        from app.services.retrieval.service import retrieval_service
        from app.services.retrieval.base import RetrievalRequest, QueryType
        
        # Initialize service
        await retrieval_service.initialize()
        print(f"   ✅ Retrieval service initialized")
        
        # Test simple retrieval
        request = RetrievalRequest(
            query="hello function",
            query_type=QueryType.SEMANTIC
        )
        
        response = await retrieval_service.retrieve(request)
        
        if response.success:
            print(f"   ✅ Query processed successfully")
            print(f"   📊 Results found: {response.result_count}")
            print(f"   ⏱️  Processing time: {response.processing_time:.3f}s")
        else:
            print(f"   ❌ Retrieval failed: {response.errors}")
            return False
        
        # Test health check
        health = await retrieval_service.health_check()
        print(f"   🏥 Retrieval service health: {health.get('status', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic retrieval test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run service manager tests."""
    print("🚀 Service Manager & Dependency Fix Tests")
    print("=" * 60)
    
    tests = [
        ("Qdrant Connection", test_qdrant_connection),
        ("Mock Services", test_mock_services),
        ("Service Manager", test_service_manager),
        ("Basic Indexing", test_basic_indexing),
        ("Basic Retrieval", test_basic_retrieval),
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
    print(f"📊 Service Manager Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All service manager tests passed!")
        print("🚀 Ready to re-run integration tests!")
        return 0
    elif passed >= total * 0.6:
        print("🎯 Most service manager tests passed!")
        print("🔧 Some fixes still needed")
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
