#!/usr/bin/env python3
"""
Final comprehensive test of all Cognify API endpoints.
"""

import requests
import json
import time
from typing import Dict, Any, Optional

# Admin token from create_admin_user_db.py
# NOTE: Replace with actual admin token from login endpoint
ADMIN_TOKEN = "your-admin-jwt-token-here"
BASE_URL = "http://localhost:8001"

def test_endpoint_detailed(method: str, url: str, headers: Dict[str, str], data: Optional[Dict[str, Any]] = None, files: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Test an endpoint and return detailed results."""
    try:
        if method.upper() == "GET":
            response = requests.get(url, headers=headers)
        elif method.upper() == "POST":
            if files:
                # Remove Content-Type for file uploads
                upload_headers = {k: v for k, v in headers.items() if k.lower() != 'content-type'}
                response = requests.post(url, headers=upload_headers, data=data, files=files)
            else:
                response = requests.post(url, headers=headers, json=data)
        elif method.upper() == "PUT":
            response = requests.put(url, headers=headers, json=data)
        elif method.upper() == "DELETE":
            response = requests.delete(url, headers=headers)
        else:
            return {"status": "error", "data": {"error": f"Unsupported method: {method}"}}

        try:
            response_data = response.json()
        except:
            response_data = {"raw_response": response.text}

        return {
            "status": "success" if response.status_code < 400 else "error",
            "status_code": response.status_code,
            "data": response_data
        }
    except Exception as e:
        return {
            "status": "error",
            "data": {"error": str(e)}
        }

def test_auth_endpoints():
    """Test authentication endpoints."""
    print("\n🔐 TESTING AUTH ENDPOINTS")
    print("-" * 40)

    # Test user registration
    print("1. Testing User Registration...")
    timestamp = int(time.time())
    result = test_endpoint_detailed(
        "POST",
        f"{BASE_URL}/api/v1/auth/register",
        {"Content-Type": "application/json"},
        {
            "email": f"testuser_{timestamp}@example.com",
            "password": "TestPassword123!",
            "full_name": "Test User",
            "username": f"testuser_{timestamp}"
        }
    )
    print(f"   Status: {result.get('status_code', 'unknown')}")

    # Test user login
    print("\n2. Testing User Login...")
    login_result = test_endpoint_detailed(
        "POST",
        f"{BASE_URL}/api/v1/auth/login",
        {"Content-Type": "application/json"},
        {
            "email": f"testuser_{timestamp}@example.com",
            "password": "TestPassword123!"
        }
    )
    print(f"   Status: {login_result.get('status_code', 'unknown')}")

    # Get user token for subsequent tests
    user_token = None
    if login_result.get('status_code') == 200 and login_result.get('data', {}).get('access_token'):
        user_token = login_result['data']['access_token']
        print(f"   ✅ User token obtained")

    # Test token refresh
    print("\n3. Testing Token Refresh...")
    refresh_token = login_result.get('data', {}).get('refresh_token', 'dummy_token')
    result = test_endpoint_detailed(
        "POST",
        f"{BASE_URL}/api/v1/auth/refresh",
        {"Content-Type": "application/json"},
        {"refresh_token": refresh_token}
    )
    print(f"   Status: {result.get('status_code', 'unknown')}")

    return user_token

def test_document_endpoints(user_token):
    """Test document endpoints."""
    print("\n📄 TESTING DOCUMENT ENDPOINTS")
    print("-" * 40)

    auth_headers = {"Authorization": f"Bearer {user_token}"}
    timestamp = int(time.time())
    test_collection_name = f"test_collection_{timestamp}"

    # Create collection for document upload
    print("0. Creating test collection for document upload...")
    result = test_endpoint_detailed(
        "POST",
        f"{BASE_URL}/api/v1/collections/",
        {**auth_headers, "Content-Type": "application/json"},
        {"name": test_collection_name, "description": "Test collection for document upload"}
    )
    print(f"   Collection creation status: {result.get('status_code', 'unknown')}")

    # Test document upload
    print("\n1. Testing Document Upload...")
    unique_content = f"This is test content for document upload at {timestamp}"
    result = test_endpoint_detailed(
        "POST",
        f"{BASE_URL}/api/v1/documents/upload",
        auth_headers,
        {"collection_name": test_collection_name},
        {"file": (f"test_{timestamp}.txt", unique_content, "text/plain")}
    )
    print(f"   Status: {result.get('status_code', 'unknown')}")

    # Test get document list to get real document ID
    print("\n2. Testing Document List...")
    docs_result = test_endpoint_detailed("GET", f"{BASE_URL}/api/v1/documents/", auth_headers)

    # Use first document ID
    real_doc_id = "00000000-0000-0000-0000-000000000000"  # Default fallback
    if docs_result.get('status_code') == 200 and docs_result.get('data', {}).get('documents'):
        documents = docs_result['data']['documents']
        if documents and len(documents) > 0:
            doc = documents[0]
            real_doc_id = doc.get('id') or doc.get('document_id') or doc.get('doc_id') or real_doc_id
            print(f"   Using document ID: {real_doc_id}")

    # Test get specific document
    print("\n3. Testing Get Document...")
    result = test_endpoint_detailed(
        "GET",
        f"{BASE_URL}/api/v1/documents/{real_doc_id}",
        auth_headers
    )
    print(f"   Status: {result.get('status_code', 'unknown')}")

def test_collection_endpoints(user_token):
    """Test collection endpoints."""
    print("\n📚 TESTING COLLECTION ENDPOINTS")
    print("-" * 40)

    auth_headers = {"Authorization": f"Bearer {user_token}", "Content-Type": "application/json"}
    timestamp = int(time.time())
    test_collection_name = f"test_collection_{timestamp}"

    # Create collection
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

    # Test collection stats
    print("\n3. Testing Collection Stats...")
    result = test_endpoint_detailed(
        "GET",
        f"{BASE_URL}/api/v1/collections/{test_collection_name}/stats",
        auth_headers
    )
    print(f"   Status: {result.get('status_code', 'unknown')}")

def test_query_endpoints(user_token):
    """Test query endpoints."""
    print("\n🔍 TESTING QUERY ENDPOINTS")
    print("-" * 40)

    auth_headers = {"Authorization": f"Bearer {user_token}", "Content-Type": "application/json"}
    timestamp = int(time.time())
    test_collection_name = f"test_collection_{timestamp}"

    # Create collection for query test
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
    print("\n1. Testing Query Execute...")
    result = test_endpoint_detailed(
        "POST",
        f"{BASE_URL}/api/v1/query/",
        auth_headers,
        {"query": "test query", "collection_name": test_collection_name}
    )
    print(f"   Status: {result.get('status_code', 'unknown')}")

    # Test query history
    print("\n2. Testing Query History...")
    result = test_endpoint_detailed(
        "GET",
        f"{BASE_URL}/api/v1/query/history/",
        auth_headers
    )
    print(f"   Status: {result.get('status_code', 'unknown')}")

def test_system_endpoints():
    """Test system endpoints with admin token."""
    print("\n🛠️ TESTING SYSTEM ENDPOINTS (ADMIN)")
    print("-" * 40)

    admin_headers = {"Authorization": f"Bearer {ADMIN_TOKEN}"}

    # Test system logs
    print("1. Testing System Logs...")
    result = test_endpoint_detailed(
        "GET",
        f"{BASE_URL}/api/v1/system/logs",
        admin_headers
    )
    print(f"   Status: {result.get('status_code', 'unknown')}")

    # Test system alerts
    print("\n2. Testing System Alerts...")
    result = test_endpoint_detailed(
        "GET",
        f"{BASE_URL}/api/v1/system/alerts",
        admin_headers
    )
    print(f"   Status: {result.get('status_code', 'unknown')}")

def main():
    """Main test function."""
    print("🎯 FINAL COMPREHENSIVE TEST - COGNIFY API")
    print("=" * 50)

    # Test all endpoints
    user_token = test_auth_endpoints()

    if user_token:
        test_document_endpoints(user_token)
        test_collection_endpoints(user_token)
        test_query_endpoints(user_token)
    else:
        print("\n❌ Could not get user token, skipping user endpoints")

    test_system_endpoints()

    print("\n" + "=" * 50)
    print("🎉 COMPREHENSIVE TEST COMPLETE")
    print("=" * 50)

if __name__ == "__main__":
    main()
