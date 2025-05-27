#!/usr/bin/env python3
"""
Detailed error analysis for failed Cognify API endpoints.
"""

import requests
import json
import time
from typing import Dict, Any, Optional

# NOTE: Replace with actual token from login endpoint
TOKEN = "your-jwt-token-here"
BASE_URL = "http://localhost:8001"

def test_endpoint_detailed(method: str, url: str, headers: Dict[str, str] = None, data: Dict[str, Any] = None, files: Dict = None) -> Dict[str, Any]:
    """Test an API endpoint with detailed error reporting."""
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

        # Try to get JSON response
        try:
            response_data = response.json()
        except:
            response_data = response.text[:500]

        return {
            "status": "success" if response.status_code < 400 else "error",
            "status_code": response.status_code,
            "data": response_data,
            "headers": dict(response.headers)
        }
    except Exception as e:
        return {"status": "error", "status_code": 0, "message": str(e), "data": {}}

def test_auth_endpoints():
    """Test authentication endpoints in detail."""
    print("🔐 TESTING AUTH ENDPOINTS")
    print("-" * 40)

    timestamp = int(time.time())

    # Test user registration
    print("1. Testing User Registration...")
    result = test_endpoint_detailed(
        "POST",
        f"{BASE_URL}/api/v1/auth/register",
        None,
        {
            "email": f"test{timestamp}@example.com",
            "password": "TestPass123!",
            "full_name": "Test User"
        }
    )
    print(f"   Status: {result.get('status_code', 'unknown')}")
    if result.get('status') == 'error':
        print(f"   Error: {json.dumps(result.get('data', {}), indent=2)}")

    # Test user login
    print("\n2. Testing User Login...")
    login_result = test_endpoint_detailed(
        "POST",
        f"{BASE_URL}/api/v1/auth/login",
        None,
        {
            "email": f"test{timestamp}@example.com",
            "password": "TestPass123!"
        }
    )
    print(f"   Status: {login_result.get('status_code', 'unknown')}")
    if login_result.get('status') == 'error':
        print(f"   Error: {json.dumps(login_result.get('data', {}), indent=2)}")

    # Extract new token for subsequent tests
    global TOKEN
    if login_result.get('status_code') == 200 and login_result.get('data', {}).get('access_token'):
        TOKEN = login_result['data']['access_token']
        print(f"   ✅ New token obtained for subsequent tests")
    else:
        print(f"   ⚠️ Could not get new token, using old token")

    # Test token refresh
    print("\n3. Testing Token Refresh...")
    refresh_token = login_result.get('data', {}).get('refresh_token', 'dummy_token')
    print(f"   Using refresh token: {refresh_token[:20]}...")
    result = test_endpoint_detailed(
        "POST",
        f"{BASE_URL}/api/v1/auth/refresh",
        {"Content-Type": "application/json"},
        {"refresh_token": refresh_token}
    )
    print(f"   Status: {result.get('status_code', 'unknown')}")
    if result.get('status') == 'error':
        print(f"   Error: {json.dumps(result.get('data', {}), indent=2)}")

def test_document_endpoints():
    """Test document endpoints in detail."""
    print("\n📄 TESTING DOCUMENT ENDPOINTS")
    print("-" * 40)

    auth_headers = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}
    timestamp = int(time.time())
    test_collection_name = f"test_collection_{timestamp}"

    # Create collection first
    print("0. Creating test collection for document upload...")
    result = test_endpoint_detailed(
        "POST",
        f"{BASE_URL}/api/v1/collections/",
        auth_headers,
        {"name": test_collection_name, "description": "Test collection for document upload"}
    )
    print(f"   Collection creation status: {result.get('status_code', 'unknown')}")

    # Test document upload
    print("1. Testing Document Upload...")
    unique_content = f"Test content for upload at {timestamp}"
    result = test_endpoint_detailed(
        "POST",
        f"{BASE_URL}/api/v1/documents/upload",
        {"Authorization": f"Bearer {TOKEN}"},
        {"collection_name": test_collection_name},
        {"file": (f"test_{timestamp}.txt", unique_content, "text/plain")}
    )
    print(f"   Status: {result.get('status_code', 'unknown')}")
    if result.get('status') == 'error':
        print(f"   Error: {json.dumps(result.get('data', {}), indent=2)}")

    # Test get document (use a real document ID if available)
    print("\n2. Testing Get Document...")
    # First get list of documents to find a real ID
    docs_result = test_endpoint_detailed("GET", f"{BASE_URL}/api/v1/documents/", auth_headers)

    # Use first document ID
    real_doc_id = "00000000-0000-0000-0000-000000000000"  # Default fallback
    if docs_result.get('status_code') == 200 and docs_result.get('data', {}).get('documents'):
        documents = docs_result['data']['documents']
        if documents and len(documents) > 0:
            # Try different possible field names for document ID
            doc = documents[0]
            real_doc_id = doc.get('id') or doc.get('document_id') or doc.get('doc_id') or real_doc_id
            print(f"   Using document ID: {real_doc_id}")

    result = test_endpoint_detailed(
        "GET",
        f"{BASE_URL}/api/v1/documents/{real_doc_id}",
        auth_headers
    )
    print(f"   Status: {result.get('status_code', 'unknown')}")
    if result.get('status') == 'error':
        print(f"   Error: {json.dumps(result.get('data', {}), indent=2)}")

def test_collection_endpoints():
    """Test collection endpoints in detail."""
    print("\n📚 TESTING COLLECTION ENDPOINTS")
    print("-" * 40)

    auth_headers = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}
    timestamp = int(time.time())
    test_collection_name = f"test_collection_{timestamp}"

    # Create collection first
    print("1. Creating test collection...")
    result = test_endpoint_detailed(
        "POST",
        f"{BASE_URL}/api/v1/collections/",
        auth_headers,
        {"name": test_collection_name, "description": "Test collection"}
    )
    print(f"   Status: {result.get('status_code', 'unknown')}")

    # Test get collection
    print("\n2. Testing Get Collection...")
    result = test_endpoint_detailed(
        "GET",
        f"{BASE_URL}/api/v1/collections/{test_collection_name}",
        auth_headers
    )
    print(f"   Status: {result.get('status_code', 'unknown')}")
    if result.get('status') == 'error':
        print(f"   Error: {json.dumps(result.get('data', {}), indent=2)}")

    # Test collection stats
    print("\n3. Testing Collection Stats...")
    result = test_endpoint_detailed(
        "GET",
        f"{BASE_URL}/api/v1/collections/{test_collection_name}/stats",
        auth_headers
    )
    print(f"   Status: {result.get('status_code', 'unknown')}")
    if result.get('status') == 'error':
        print(f"   Error: {json.dumps(result.get('data', {}), indent=2)}")

def test_query_endpoints():
    """Test query endpoints in detail."""
    print("\n🔍 TESTING QUERY ENDPOINTS")
    print("-" * 40)

    auth_headers = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}
    timestamp = int(time.time())
    test_collection_name = f"test_collection_{timestamp}"

    # Create collection first for query test
    print("0. Creating test collection for query...")
    result = test_endpoint_detailed(
        "POST",
        f"{BASE_URL}/api/v1/collections/",
        auth_headers,
        {"name": test_collection_name, "description": "Test collection for query"}
    )
    status_code = result.get('status_code', 'unknown')
    print(f"   Collection creation status: {status_code}")
    if status_code == 400:
        print(f"   ✅ Collection already exists (expected for duplicate name)")
    elif status_code == 200:
        print(f"   ✅ Collection created successfully")

    # Test query execute
    print("1. Testing Query Execute...")
    result = test_endpoint_detailed(
        "POST",
        f"{BASE_URL}/api/v1/query/",
        auth_headers,
        {"query": "test query", "collection_name": test_collection_name}
    )
    print(f"   Status: {result.get('status_code', 'unknown')}")
    if result.get('status') == 'error':
        print(f"   Error: {json.dumps(result.get('data', {}), indent=2)}")

    # Test query history
    print("\n2. Testing Query History...")
    result = test_endpoint_detailed(
        "GET",
        f"{BASE_URL}/api/v1/query/history/",
        auth_headers
    )
    print(f"   Status: {result.get('status_code', 'unknown')}")
    if result.get('status') == 'error':
        print(f"   Error: {json.dumps(result.get('data', {}), indent=2)}")

def test_system_endpoints():
    """Test system endpoints in detail."""
    print("\n🛠️ TESTING SYSTEM ENDPOINTS")
    print("-" * 40)

    auth_headers = {"Authorization": f"Bearer {TOKEN}"}

    # Test system logs
    print("1. Testing System Logs...")
    result = test_endpoint_detailed(
        "GET",
        f"{BASE_URL}/api/v1/system/logs",
        auth_headers
    )
    print(f"   Status: {result.get('status_code', 'unknown')}")
    if result.get('status') == 'error':
        print(f"   Error: {json.dumps(result.get('data', {}), indent=2)}")

    # Test system alerts
    print("\n2. Testing System Alerts...")
    result = test_endpoint_detailed(
        "GET",
        f"{BASE_URL}/api/v1/system/alerts",
        auth_headers
    )
    print(f"   Status: {result.get('status_code', 'unknown')}")
    if result.get('status') == 'error':
        print(f"   Error: {json.dumps(result.get('data', {}), indent=2)}")

def main():
    """Main test function."""
    print("🔍 DETAILED ERROR ANALYSIS - COGNIFY API")
    print("=" * 50)

    test_auth_endpoints()
    test_document_endpoints()
    test_collection_endpoints()
    test_query_endpoints()
    test_system_endpoints()

    print("\n" + "=" * 50)
    print("📋 ANALYSIS COMPLETE")
    print("=" * 50)

if __name__ == "__main__":
    main()
