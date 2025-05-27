#!/usr/bin/env python3
"""
Test Cognify API with existing token.
"""

import requests
import json
from typing import Dict, Any

# Use the existing token
# NOTE: Replace with actual token from login endpoint
TOKEN = "your-jwt-token-here"
BASE_URL = "http://localhost:8000"

def test_endpoint(method: str, url: str, headers: Dict[str, str] = None, data: Dict[str, Any] = None) -> Dict[str, Any]:
    """Test an API endpoint."""
    try:
        if method.upper() == "GET":
            response = requests.get(url, headers=headers, timeout=10)
        elif method.upper() == "POST":
            response = requests.post(url, headers=headers, json=data, timeout=10)
        else:
            return {"status": "error", "message": f"Unsupported method: {method}"}

        return {
            "status": "success" if response.status_code < 400 else "error",
            "status_code": response.status_code,
            "data": response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

def main():
    """Main test function."""
    print("🧠 COGNIFY API TEST WITH TOKEN")
    print("=" * 50)

    # Headers with authentication
    auth_headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json"
    }

    # Test endpoints
    tests = [
        # Health checks
        ("GET", f"{BASE_URL}/health", None, None, "Root Health"),
        ("GET", f"{BASE_URL}/api/v1/system/health", None, None, "System Health"),
        ("GET", f"{BASE_URL}/api/v1/chunking/health", None, None, "Chunking Health"),

        # Auth endpoints
        ("GET", f"{BASE_URL}/api/v1/auth/me", auth_headers, None, "Current User"),

        # Chunking endpoints
        ("POST", f"{BASE_URL}/api/v1/chunking/", None, {"content": "def hello():\n    print('Hello')", "language": "python"}, "Code Chunking"),
        ("GET", f"{BASE_URL}/api/v1/chunking/stats", None, None, "Chunking Stats"),
        ("GET", f"{BASE_URL}/api/v1/chunking/supported-languages", None, None, "Supported Languages"),

        # Documents endpoints
        ("GET", f"{BASE_URL}/api/v1/documents/", auth_headers, None, "Documents List"),

        # Collections endpoints
        ("GET", f"{BASE_URL}/api/v1/collections/", auth_headers, None, "Collections List"),
        ("POST", f"{BASE_URL}/api/v1/collections/", auth_headers, {"name": f"test_api_collection_{int(__import__('time').time())}", "description": "Test collection from API"}, "Collection Creation"),

        # Query endpoints
        ("POST", f"{BASE_URL}/api/v1/query/search", auth_headers, {"query": "test query", "collection_name": "test_api_collection"}, "Query Search"),

        # System endpoints
        ("GET", f"{BASE_URL}/api/v1/system/stats", auth_headers, None, "System Stats"),
        ("GET", f"{BASE_URL}/api/v1/system/metrics", auth_headers, None, "System Metrics"),
    ]

    results = {"passed": 0, "failed": 0, "total": len(tests)}

    for method, url, headers, data, name in tests:
        print(f"\n🔍 Testing {name}...")
        result = test_endpoint(method, url, headers, data)

        if result["status"] == "success":
            print(f"✅ {name}: PASSED (Status: {result['status_code']})")
            if isinstance(result["data"], dict):
                # Show some key info
                if "total" in result["data"]:
                    print(f"   📊 Total items: {result['data']['total']}")
                if "collection_id" in result["data"]:
                    print(f"   🆔 Collection ID: {result['data']['collection_id']}")
                if "chunks" in result["data"]:
                    print(f"   📦 Chunks created: {len(result['data']['chunks'])}")
                if "supported_languages" in result["data"]:
                    print(f"   🌐 Languages: {len(result['data']['supported_languages'])}")
            results["passed"] += 1
        else:
            print(f"❌ {name}: FAILED")
            print(f"   Error: {result.get('message', 'Unknown error')}")
            if "status_code" in result:
                print(f"   Status Code: {result['status_code']}")
            results["failed"] += 1

    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    print(f"✅ Passed: {results['passed']}")
    print(f"❌ Failed: {results['failed']}")
    print(f"📈 Success Rate: {(results['passed'] / results['total']) * 100:.1f}%")
    print("=" * 50)

if __name__ == "__main__":
    main()
