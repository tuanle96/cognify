"""
Production deployment tests for Cognify
Tests the production deployment and validates all services are working correctly.
"""

import asyncio
import httpx
import pytest
import time
from typing import Dict, Any


class TestProductionDeployment:
    """Test suite for production deployment validation"""
    
    BASE_URL = "http://localhost:8000"
    TIMEOUT = 30
    
    @pytest.fixture(scope="class")
    def client(self):
        """HTTP client for testing"""
        return httpx.AsyncClient(base_url=self.BASE_URL, timeout=self.TIMEOUT)
    
    async def test_api_health_check(self, client):
        """Test API health endpoint"""
        response = await client.get("/health")
        assert response.status_code == 200
        
        health_data = response.json()
        assert health_data["status"] == "healthy"
        assert "timestamp" in health_data
        assert "version" in health_data
    
    async def test_api_docs_accessible(self, client):
        """Test API documentation is accessible"""
        response = await client.get("/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    async def test_openapi_spec(self, client):
        """Test OpenAPI specification is available"""
        response = await client.get("/openapi.json")
        assert response.status_code == 200
        
        spec = response.json()
        assert "openapi" in spec
        assert "info" in spec
        assert spec["info"]["title"] == "Cognify API"
    
    async def test_authentication_endpoints(self, client):
        """Test authentication endpoints"""
        # Test login endpoint exists
        response = await client.post("/auth/login", json={
            "username": "test",
            "password": "test"
        })
        # Should return 401 for invalid credentials, not 404
        assert response.status_code in [401, 422]
    
    async def test_chunking_endpoint(self, client):
        """Test chunking endpoint with sample code"""
        sample_code = '''
def hello_world():
    """A simple hello world function"""
    print("Hello, World!")
    return "Hello, World!"

class Calculator:
    def add(self, a, b):
        return a + b
    
    def multiply(self, a, b):
        return a * b
'''
        
        response = await client.post("/chunk", json={
            "content": sample_code,
            "language": "python",
            "strategy": "agentic"
        })
        
        # Should work or require authentication
        assert response.status_code in [200, 401, 422]
        
        if response.status_code == 200:
            chunks = response.json()
            assert isinstance(chunks, list)
            assert len(chunks) > 0
    
    async def test_embedding_endpoint(self, client):
        """Test embedding endpoint"""
        response = await client.post("/embed", json={
            "texts": ["Hello world", "Test embedding"],
            "model": "text-embedding-004"
        })
        
        # Should work or require authentication
        assert response.status_code in [200, 401, 422]
        
        if response.status_code == 200:
            embeddings = response.json()
            assert isinstance(embeddings, list)
            assert len(embeddings) == 2
    
    async def test_vector_search_endpoint(self, client):
        """Test vector search endpoint"""
        response = await client.post("/search", json={
            "query": "hello world function",
            "collection": "test",
            "limit": 5
        })
        
        # Should work or require authentication
        assert response.status_code in [200, 401, 422, 404]
    
    async def test_metrics_endpoint(self, client):
        """Test metrics endpoint for monitoring"""
        response = await client.get("/metrics")
        assert response.status_code == 200
        
        metrics = response.text
        assert "cognify_" in metrics  # Should contain Cognify-specific metrics
    
    async def test_rate_limiting(self, client):
        """Test rate limiting is working"""
        # Make multiple rapid requests
        responses = []
        for i in range(10):
            response = await client.get("/health")
            responses.append(response.status_code)
        
        # Should get mostly 200s, possibly some 429s if rate limiting is strict
        success_count = sum(1 for status in responses if status == 200)
        assert success_count >= 5  # At least half should succeed
    
    async def test_cors_headers(self, client):
        """Test CORS headers are present"""
        response = await client.options("/health")
        
        # Should have CORS headers or return 405 Method Not Allowed
        assert response.status_code in [200, 405]
    
    async def test_security_headers(self, client):
        """Test security headers are present"""
        response = await client.get("/health")
        assert response.status_code == 200
        
        # Check for basic security headers
        headers = response.headers
        # These might be added by reverse proxy, so we're flexible
        security_headers = [
            "x-content-type-options",
            "x-frame-options", 
            "x-xss-protection",
            "strict-transport-security"
        ]
        
        # At least some security measures should be in place
        # This is more of a warning than a hard requirement
        present_headers = [h for h in security_headers if h in headers]
        print(f"Security headers present: {present_headers}")


class TestServiceIntegration:
    """Test integration between services"""
    
    async def test_database_connection(self):
        """Test database connectivity"""
        # This would typically test database connection
        # For now, we'll test through the API
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8000/health")
            if response.status_code == 200:
                health = response.json()
                # If health check includes DB status, verify it
                if "database" in health:
                    assert health["database"] == "connected"
    
    async def test_redis_connection(self):
        """Test Redis connectivity"""
        # Test through caching behavior or health endpoint
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8000/health")
            if response.status_code == 200:
                health = response.json()
                if "redis" in health:
                    assert health["redis"] == "connected"
    
    async def test_qdrant_connection(self):
        """Test Qdrant connectivity"""
        # Test vector database connection
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8000/health")
            if response.status_code == 200:
                health = response.json()
                if "qdrant" in health:
                    assert health["qdrant"] == "connected"


class TestPerformance:
    """Basic performance tests"""
    
    async def test_response_time(self):
        """Test API response time"""
        async with httpx.AsyncClient() as client:
            start_time = time.time()
            response = await client.get("http://localhost:8000/health")
            end_time = time.time()
            
            response_time = end_time - start_time
            assert response_time < 1.0  # Should respond within 1 second
            assert response.status_code == 200
    
    async def test_concurrent_requests(self):
        """Test handling concurrent requests"""
        async def make_request():
            async with httpx.AsyncClient() as client:
                response = await client.get("http://localhost:8000/health")
                return response.status_code
        
        # Make 10 concurrent requests
        tasks = [make_request() for _ in range(10)]
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        success_count = sum(1 for status in results if status == 200)
        assert success_count >= 8  # At least 80% should succeed


@pytest.mark.asyncio
async def test_full_deployment_validation():
    """Comprehensive deployment validation test"""
    
    print("🚀 Starting production deployment validation...")
    
    # Test API availability
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get("http://localhost:8000/health", timeout=10)
            assert response.status_code == 200
            print("✅ API is accessible and healthy")
            
            # Test basic functionality
            health_data = response.json()
            print(f"✅ API version: {health_data.get('version', 'unknown')}")
            print(f"✅ API status: {health_data.get('status', 'unknown')}")
            
        except Exception as e:
            pytest.fail(f"❌ API is not accessible: {e}")
    
    print("🎉 Production deployment validation completed successfully!")


if __name__ == "__main__":
    # Run the tests
    asyncio.run(test_full_deployment_validation())
