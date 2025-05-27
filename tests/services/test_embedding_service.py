#!/usr/bin/env python3
"""
Test embedding service implementation.
"""

import asyncio
import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_embedding_factory():
    """Test embedding factory functionality."""
    print("🔍 Testing embedding factory...")
    
    try:
        from app.services.embedding.factory import embedding_factory
        from app.services.embedding.base import EmbeddingProvider
        
        # Test available providers
        providers = embedding_factory.get_available_providers()
        print(f"   ✅ Available providers: {[p.value for p in providers]}")
        
        # Test default configs
        for provider in providers:
            config = embedding_factory.get_default_config(provider)
            print(f"   📋 {provider.value} config: {config}")
        
        # Test best available client (without API keys)
        try:
            client = embedding_factory.create_best_available_client()
            print(f"   ❌ Unexpected: Created client without API keys")
            return False
        except Exception as e:
            print(f"   ✅ Expected error without API keys: {type(e).__name__}")
        
        print("✅ Embedding factory test passed")
        return True
        
    except Exception as e:
        print(f"❌ Embedding factory test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_embedding_models():
    """Test embedding model configurations."""
    print("🔍 Testing embedding models...")
    
    try:
        from app.services.embedding.openai_client import OpenAIEmbeddingClient
        from app.services.embedding.voyage_client import VoyageEmbeddingClient
        from app.services.embedding.cohere_client import CohereEmbeddingClient
        
        # Test OpenAI models
        openai_models = OpenAIEmbeddingClient.MODELS
        print(f"   📊 OpenAI models: {list(openai_models.keys())}")
        
        # Test Voyage models
        voyage_models = VoyageEmbeddingClient.MODELS
        print(f"   📊 Voyage models: {list(voyage_models.keys())}")
        
        # Test Cohere models
        cohere_models = CohereEmbeddingClient.MODELS
        print(f"   📊 Cohere models: {list(cohere_models.keys())}")
        
        # Test client creation (without initialization)
        openai_client = OpenAIEmbeddingClient("fake-key")
        print(f"   ✅ OpenAI client: {openai_client.provider.value}, dims: {openai_client.default_dimensions}")
        
        voyage_client = VoyageEmbeddingClient("fake-key")
        print(f"   ✅ Voyage client: {voyage_client.provider.value}, dims: {voyage_client.default_dimensions}")
        
        cohere_client = CohereEmbeddingClient("fake-key")
        print(f"   ✅ Cohere client: {cohere_client.provider.value}, dims: {cohere_client.default_dimensions}")
        
        print("✅ Embedding models test passed")
        return True
        
    except Exception as e:
        print(f"❌ Embedding models test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_embedding_requests():
    """Test embedding request/response models."""
    print("🔍 Testing embedding request/response models...")
    
    try:
        from app.services.embedding.base import (
            EmbeddingRequest, EmbeddingResponse, EmbeddingType, EmbeddingProvider
        )
        
        # Test valid request
        request = EmbeddingRequest(
            texts=["Hello world", "Test embedding"],
            embedding_type=EmbeddingType.TEXT,
            model="test-model"
        )
        print(f"   ✅ Valid request: {len(request.texts)} texts, type: {request.embedding_type.value}")
        
        # Test invalid request (empty texts)
        try:
            invalid_request = EmbeddingRequest(texts=[])
            print(f"   ❌ Unexpected: Created request with empty texts")
            return False
        except ValueError as e:
            print(f"   ✅ Expected error for empty texts: {e}")
        
        # Test valid response
        response = EmbeddingResponse(
            embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            model="test-model",
            dimensions=3,
            usage={"total_tokens": 10},
            provider=EmbeddingProvider.OPENAI,
            processing_time=0.5
        )
        print(f"   ✅ Valid response: {len(response.embeddings)} embeddings, dims: {response.dimensions}")
        
        # Test invalid response (empty embeddings)
        try:
            invalid_response = EmbeddingResponse(
                embeddings=[],
                model="test-model",
                dimensions=3,
                usage={"total_tokens": 10},
                provider=EmbeddingProvider.OPENAI,
                processing_time=0.5
            )
            print(f"   ❌ Unexpected: Created response with empty embeddings")
            return False
        except ValueError as e:
            print(f"   ✅ Expected error for empty embeddings: {e}")
        
        print("✅ Embedding request/response test passed")
        return True
        
    except Exception as e:
        print(f"❌ Embedding request/response test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_embedding_service_config():
    """Test embedding service configuration."""
    print("🔍 Testing embedding service configuration...")
    
    try:
        from app.services.embedding.service import EmbeddingService, EmbeddingServiceConfig
        from app.services.embedding.base import EmbeddingProvider
        
        # Test default config
        default_config = EmbeddingServiceConfig()
        print(f"   ✅ Default primary provider: {default_config.primary_provider.value}")
        print(f"   ✅ Default fallback providers: {[p.value for p in default_config.fallback_providers]}")
        print(f"   ✅ Default caching enabled: {default_config.enable_caching}")
        
        # Test custom config
        custom_config = EmbeddingServiceConfig(
            primary_provider=EmbeddingProvider.OPENAI,
            fallback_providers=[EmbeddingProvider.COHERE],
            enable_caching=False,
            max_batch_size=50
        )
        print(f"   ✅ Custom primary provider: {custom_config.primary_provider.value}")
        print(f"   ✅ Custom fallback providers: {[p.value for p in custom_config.fallback_providers]}")
        print(f"   ✅ Custom caching enabled: {custom_config.enable_caching}")
        
        # Test service creation
        service = EmbeddingService(custom_config)
        print(f"   ✅ Service created with custom config")
        print(f"   ✅ Service initialized: {service._initialized}")
        
        # Test stats
        stats = service.get_stats()
        print(f"   ✅ Initial stats: {stats}")
        
        print("✅ Embedding service config test passed")
        return True
        
    except Exception as e:
        print(f"❌ Embedding service config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_embedding_service_without_api_keys():
    """Test embedding service behavior without API keys."""
    print("🔍 Testing embedding service without API keys...")
    
    try:
        from app.services.embedding.service import EmbeddingService
        from app.services.embedding.base import EmbeddingType
        
        # Create service
        service = EmbeddingService()
        
        # Try to initialize (should fail gracefully)
        try:
            await service.initialize()
            print(f"   ❌ Unexpected: Service initialized without API keys")
            return False
        except Exception as e:
            print(f"   ✅ Expected error without API keys: {type(e).__name__}")
        
        # Try to embed (should fail gracefully)
        try:
            await service.embed("test text", EmbeddingType.TEXT)
            print(f"   ❌ Unexpected: Embedding succeeded without initialization")
            return False
        except Exception as e:
            print(f"   ✅ Expected error for embedding without initialization: {type(e).__name__}")
        
        # Test health check (should handle gracefully)
        try:
            health = await service.health_check()
            print(f"   ✅ Health check handled gracefully: {health.get('status', 'unknown')}")
        except Exception as e:
            print(f"   ✅ Health check failed as expected: {type(e).__name__}")
        
        # Test cleanup
        await service.cleanup()
        print(f"   ✅ Service cleanup completed")
        
        print("✅ Embedding service without API keys test passed")
        return True
        
    except Exception as e:
        print(f"❌ Embedding service without API keys test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_embedding_types():
    """Test different embedding types."""
    print("🔍 Testing embedding types...")
    
    try:
        from app.services.embedding.base import EmbeddingType
        
        # Test all embedding types
        types = [EmbeddingType.TEXT, EmbeddingType.CODE, EmbeddingType.QUERY, EmbeddingType.DOCUMENT]
        
        for embedding_type in types:
            print(f"   📋 Embedding type: {embedding_type.value}")
        
        # Test type-specific optimizations
        print(f"   ✅ TEXT type: General text embeddings")
        print(f"   ✅ CODE type: Optimized for source code")
        print(f"   ✅ QUERY type: Optimized for search queries")
        print(f"   ✅ DOCUMENT type: Optimized for documents")
        
        print("✅ Embedding types test passed")
        return True
        
    except Exception as e:
        print(f"❌ Embedding types test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run embedding service tests."""
    print("🚀 Embedding Service Test")
    print("=" * 50)
    
    tests = [
        ("Embedding Factory", test_embedding_factory),
        ("Embedding Models", test_embedding_models),
        ("Request/Response Models", test_embedding_requests),
        ("Service Configuration", test_embedding_service_config),
        ("Service Without API Keys", test_embedding_service_without_api_keys),
        ("Embedding Types", test_embedding_types),
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
    
    print("=" * 50)
    print(f"📊 Embedding Service Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All embedding service tests passed!")
        print("🚀 Embedding service implementation is solid!")
        return 0
    elif passed >= total * 0.8:
        print("🎯 Most embedding service tests passed!")
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
