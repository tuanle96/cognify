#!/usr/bin/env python3
"""
Complete API test suite for Cognify Simple API.
"""

import requests
import json
import time
from typing import Dict, Any

API_BASE_URL = "http://localhost:8000"

def test_root_endpoint():
    """Test root endpoint."""
    print("🔧 Testing root endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/")
        response.raise_for_status()
        data = response.json()
        
        assert "name" in data
        assert "version" in data
        assert data["status"] == "running"
        
        print("✅ Root endpoint working")
        print(f"   App: {data['name']} v{data['version']}")
        return True
        
    except Exception as e:
        print(f"❌ Root endpoint failed: {e}")
        return False

def test_health_endpoint():
    """Test health check endpoint."""
    print("\n🔧 Testing health endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        response.raise_for_status()
        data = response.json()
        
        assert data["status"] == "healthy"
        assert "services" in data
        assert "chunking_service" in data["services"]
        
        print("✅ Health endpoint working")
        print(f"   Status: {data['status']}")
        print(f"   Services: {list(data['services'].keys())}")
        return True
        
    except Exception as e:
        print(f"❌ Health endpoint failed: {e}")
        return False

def test_supported_languages():
    """Test supported languages endpoint."""
    print("\n🔧 Testing supported languages...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/chunk/supported-languages")
        response.raise_for_status()
        data = response.json()
        
        assert "supported_languages" in data
        assert "coming_soon" in data
        assert len(data["supported_languages"]) > 0
        
        print("✅ Supported languages endpoint working")
        print(f"   Supported: {[lang['language'] for lang in data['supported_languages']]}")
        print(f"   Coming soon: {data['coming_soon'][:3]}...")
        return True
        
    except Exception as e:
        print(f"❌ Supported languages failed: {e}")
        return False

def test_chunking_test_endpoint():
    """Test chunking test endpoint."""
    print("\n🔧 Testing chunking test endpoint...")
    try:
        response = requests.post(f"{API_BASE_URL}/api/v1/chunk/test")
        response.raise_for_status()
        data = response.json()
        
        assert "chunks" in data
        assert "strategy_used" in data
        assert "quality_score" in data
        assert len(data["chunks"]) > 0
        
        print("✅ Chunking test endpoint working")
        print(f"   Chunks created: {data['chunk_count']}")
        print(f"   Strategy: {data['strategy_used']}")
        print(f"   Quality score: {data['quality_score']}")
        print(f"   Processing time: {data['processing_time']}s")
        return True
        
    except Exception as e:
        print(f"❌ Chunking test failed: {e}")
        return False

def test_custom_chunking():
    """Test custom chunking request."""
    print("\n🔧 Testing custom chunking...")
    try:
        custom_code = '''
import asyncio
from typing import List, Optional

class AsyncProcessor:
    """Asynchronous data processor."""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.processed_items = []
    
    async def process_item(self, item: str) -> Dict[str, Any]:
        """Process a single item asynchronously."""
        await asyncio.sleep(0.1)  # Simulate processing
        
        result = {
            "item": item,
            "length": len(item),
            "processed_at": time.time()
        }
        
        self.processed_items.append(result)
        return result
    
    async def process_batch(self, items: List[str]) -> List[Dict[str, Any]]:
        """Process multiple items concurrently."""
        tasks = [self.process_item(item) for item in items]
        results = await asyncio.gather(*tasks)
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            "total_processed": len(self.processed_items),
            "max_workers": self.max_workers
        }

async def main():
    """Main async function."""
    processor = AsyncProcessor(max_workers=8)
    items = ["item1", "item2", "item3"]
    results = await processor.process_batch(items)
    print(f"Processed {len(results)} items")

if __name__ == "__main__":
    asyncio.run(main())
'''
        
        request_data = {
            "content": custom_code,
            "language": "python",
            "file_path": "async_processor.py",
            "purpose": "performance_analysis",
            "quality_threshold": 0.9,
            "force_agentic": False
        }
        
        response = requests.post(
            f"{API_BASE_URL}/api/v1/chunk",
            json=request_data,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        data = response.json()
        
        assert "chunks" in data
        assert data["chunk_count"] > 0
        assert data["quality_score"] > 0
        
        print("✅ Custom chunking working")
        print(f"   File: {request_data['file_path']}")
        print(f"   Purpose: {request_data['purpose']}")
        print(f"   Chunks created: {data['chunk_count']}")
        print(f"   Average chunk size: {data['average_chunk_size']:.1f} lines")
        print(f"   Total lines: {data['total_lines']}")
        
        # Show first chunk details
        if data["chunks"]:
            first_chunk = data["chunks"][0]
            print(f"   First chunk: {first_chunk['name']} ({first_chunk['chunk_type']})")
            print(f"   Lines {first_chunk['start_line']}-{first_chunk['end_line']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Custom chunking failed: {e}")
        return False

def test_error_handling():
    """Test API error handling."""
    print("\n🔧 Testing error handling...")
    try:
        # Test with invalid JSON
        response = requests.post(
            f"{API_BASE_URL}/api/v1/chunk",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        # Should return 422 for validation error
        assert response.status_code == 422
        
        print("✅ Error handling working")
        print(f"   Invalid JSON returns: {response.status_code}")
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def test_performance():
    """Test API performance with multiple requests."""
    print("\n🔧 Testing performance...")
    try:
        start_time = time.time()
        successful_requests = 0
        total_requests = 5
        
        for i in range(total_requests):
            try:
                response = requests.post(f"{API_BASE_URL}/api/v1/chunk/test", timeout=10)
                if response.status_code == 200:
                    successful_requests += 1
            except Exception:
                pass
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / total_requests
        
        print("✅ Performance test completed")
        print(f"   Successful requests: {successful_requests}/{total_requests}")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Average time per request: {avg_time:.2f}s")
        
        return successful_requests >= total_requests * 0.8  # 80% success rate
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

def run_all_tests():
    """Run all API tests."""
    print("🚀 Starting Cognify API Test Suite")
    print("=" * 50)
    
    tests = [
        test_root_endpoint,
        test_health_endpoint,
        test_supported_languages,
        test_chunking_test_endpoint,
        test_custom_chunking,
        test_error_handling,
        test_performance
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! API is working correctly.")
    elif passed >= total * 0.8:
        print("✅ Most tests passed. API is mostly functional.")
    else:
        print("⚠️ Many tests failed. API needs attention.")
    
    return passed, total

if __name__ == "__main__":
    # Check if server is running
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        print("✅ Server is running")
    except Exception:
        print("❌ Server is not running. Please start the server first:")
        print("   python simple_main.py")
        exit(1)
    
    # Run tests
    passed, total = run_all_tests()
    
    # Exit with appropriate code
    exit(0 if passed == total else 1)
