#!/usr/bin/env python3
"""
Complete test of ALL Cognify API endpoints.
"""

import requests
import json
import time
from typing import Dict, Any, Optional

# Use the existing token
# NOTE: Replace with actual token from login endpoint
TOKEN = "your-jwt-token-here"
BASE_URL = "http://localhost:8001"

def test_endpoint(method: str, url: str, headers: Dict[str, str] = None, data: Dict[str, Any] = None, files: Dict = None) -> Dict[str, Any]:
    """Test an API endpoint."""
    try:
        if method.upper() == "GET":
            response = requests.get(url, headers=headers, timeout=10)
        elif method.upper() == "POST":
            if files:
                response = requests.post(url, headers=headers, data=data, files=files, timeout=10)
            else:
                response = requests.post(url, headers=headers, json=data, timeout=10)
        elif method.upper() == "PUT":
            response = requests.put(url, headers=headers, json=data, timeout=10)
        elif method.upper() == "DELETE":
            response = requests.delete(url, headers=headers, timeout=10)
        else:
            return {"status": "error", "message": f"Unsupported method: {method}"}

        return {
            "status": "success" if response.status_code < 400 else "error",
            "status_code": response.status_code,
            "data": response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text[:200]
        }
    except Exception as e:
        return {"status": "error", "message": str(e)[:200]}

def main():
    """Main test function."""
    print("🧠 COMPLETE COGNIFY API TEST - ALL 32+ ENDPOINTS")
    print("=" * 60)

    # Headers with authentication
    auth_headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json"
    }

    # Test data
    timestamp = int(time.time())
    test_collection_name = f"test_collection_{timestamp}"

    # ALL ENDPOINTS TO TEST
    tests = [
        # Root & Health (3)
        ("GET", f"{BASE_URL}/", None, None, None, "Root Info"),
        ("GET", f"{BASE_URL}/health", None, None, None, "Root Health"),
        ("GET", f"{BASE_URL}/api/v1/system/health", None, None, None, "System Health"),

        # Auth endpoints (7)
        ("POST", f"{BASE_URL}/api/v1/auth/register", None, {"email": f"test{timestamp}@example.com", "password": "testpass123", "full_name": "Test User"}, None, "User Registration"),
        ("POST", f"{BASE_URL}/api/v1/auth/login", None, {"email": "test@example.com", "password": "testpass123"}, None, "User Login"),
        ("GET", f"{BASE_URL}/api/v1/auth/me", auth_headers, None, None, "Current User"),
        ("POST", f"{BASE_URL}/api/v1/auth/refresh", auth_headers, {"refresh_token": "dummy_token"}, None, "Token Refresh"),
        ("POST", f"{BASE_URL}/api/v1/auth/logout", auth_headers, None, None, "User Logout"),
        ("POST", f"{BASE_URL}/api/v1/auth/password/change", auth_headers, {"current_password": "testpass123", "new_password": "newpass123"}, None, "Password Change"),
        ("POST", f"{BASE_URL}/api/v1/auth/password/reset", None, {"email": "test@example.com"}, None, "Password Reset"),

        # Chunking endpoints (5)
        ("GET", f"{BASE_URL}/api/v1/chunking/health", None, None, None, "Chunking Health"),
        ("POST", f"{BASE_URL}/api/v1/chunking/", None, {"content": "def hello():\n    print('Hello')", "language": "python"}, None, "Code Chunking"),
        ("POST", f"{BASE_URL}/api/v1/chunking/test", None, {"content": "Test content", "language": "text"}, None, "Chunking Test"),
        ("GET", f"{BASE_URL}/api/v1/chunking/stats", None, None, None, "Chunking Stats"),
        ("GET", f"{BASE_URL}/api/v1/chunking/supported-languages", None, None, None, "Supported Languages"),

        # Documents endpoints (5)
        ("GET", f"{BASE_URL}/api/v1/documents/", auth_headers, None, None, "Documents List"),
        ("POST", f"{BASE_URL}/api/v1/documents/upload", {"Authorization": f"Bearer {TOKEN}"}, {"collection_name": test_collection_name}, {"file": ("test.txt", "Test content", "text/plain")}, "Document Upload"),
        ("POST", f"{BASE_URL}/api/v1/documents/upload/batch", auth_headers, {"collection_name": test_collection_name, "files": []}, None, "Batch Upload"),
        ("GET", f"{BASE_URL}/api/v1/documents/dummy-id", auth_headers, None, None, "Get Document"),
        ("DELETE", f"{BASE_URL}/api/v1/documents/dummy-id", auth_headers, None, None, "Delete Document"),

        # Collections endpoints (6)
        ("GET", f"{BASE_URL}/api/v1/collections/", auth_headers, None, None, "Collections List"),
        ("POST", f"{BASE_URL}/api/v1/collections/", auth_headers, {"name": test_collection_name, "description": "Test collection"}, None, "Collection Creation"),
        ("GET", f"{BASE_URL}/api/v1/collections/{test_collection_name}", auth_headers, None, None, "Get Collection"),
        ("PUT", f"{BASE_URL}/api/v1/collections/{test_collection_name}", auth_headers, {"description": "Updated description"}, None, "Update Collection"),
        ("GET", f"{BASE_URL}/api/v1/collections/{test_collection_name}/stats", auth_headers, None, None, "Collection Stats"),
        ("DELETE", f"{BASE_URL}/api/v1/collections/{test_collection_name}", auth_headers, None, None, "Delete Collection"),

        # Query endpoints (6)
        ("POST", f"{BASE_URL}/api/v1/query/search", auth_headers, {"query": "test query", "collection_name": test_collection_name}, None, "Query Search"),
        ("POST", f"{BASE_URL}/api/v1/query/", auth_headers, {"query": "test query", "collection_name": test_collection_name}, None, "Query Execute"),
        ("POST", f"{BASE_URL}/api/v1/query/batch", auth_headers, {"queries": [{"query": "test", "collection_name": test_collection_name}]}, None, "Batch Query"),
        ("GET", f"{BASE_URL}/api/v1/query/history/", auth_headers, None, None, "Query History"),
        ("GET", f"{BASE_URL}/api/v1/query/suggestions/{test_collection_name}", auth_headers, None, None, "Query Suggestions"),
        ("GET", f"{BASE_URL}/api/v1/query/dummy-id", auth_headers, None, None, "Get Query"),

        # System endpoints (6)
        ("GET", f"{BASE_URL}/api/v1/system/stats", auth_headers, None, None, "System Stats"),
        ("GET", f"{BASE_URL}/api/v1/system/metrics", auth_headers, None, None, "System Metrics"),
        ("GET", f"{BASE_URL}/api/v1/system/logs", auth_headers, None, None, "System Logs"),
        ("GET", f"{BASE_URL}/api/v1/system/alerts", auth_headers, None, None, "System Alerts"),
        ("POST", f"{BASE_URL}/api/v1/system/maintenance", auth_headers, None, None, "Start Maintenance"),
        ("DELETE", f"{BASE_URL}/api/v1/system/maintenance", auth_headers, None, None, "Stop Maintenance"),
    ]

    results = {"passed": 0, "failed": 0, "total": len(tests)}
    failed_tests = []

    print(f"🔍 Testing {len(tests)} endpoints...")
    print("-" * 60)

    for method, url, headers, data, files, name in tests:
        print(f"Testing {name}...", end=" ")
        result = test_endpoint(method, url, headers, data, files)

        if result["status"] == "success":
            print(f"✅ PASSED ({result['status_code']})")
            results["passed"] += 1
        else:
            print(f"❌ FAILED")
            if "status_code" in result:
                print(f"   Status: {result['status_code']}")
            print(f"   Error: {result.get('message', 'Unknown error')[:100]}")
            failed_tests.append(name)
            results["failed"] += 1

    # Summary
    print("\n" + "=" * 60)
    print("📊 COMPLETE TEST SUMMARY")
    print("=" * 60)
    print(f"✅ Passed: {results['passed']}")
    print(f"❌ Failed: {results['failed']}")
    print(f"📈 Success Rate: {(results['passed'] / results['total']) * 100:.1f}%")

    if failed_tests:
        print(f"\n❌ Failed endpoints ({len(failed_tests)}):")
        for i, test in enumerate(failed_tests, 1):
            print(f"   {i}. {test}")

    print("=" * 60)

if __name__ == "__main__":
    main()
