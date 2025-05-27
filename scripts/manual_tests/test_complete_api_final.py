#!/usr/bin/env python3
"""
Complete comprehensive test of ALL Cognify API endpoints.
Final verification of 100% success rate.
"""

import requests
import json
import time
from typing import Dict, Any, Optional

# Admin token (update if needed)
# NOTE: Replace with actual admin token from login endpoint
ADMIN_TOKEN = "your-admin-jwt-token-here"
BASE_URL = "http://localhost:8001"

def test_endpoint(method: str, url: str, headers: Dict[str, str], data: Optional[Dict[str, Any]] = None, files: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Test an endpoint and return results."""
    try:
        if method.upper() == "GET":
            response = requests.get(url, headers=headers)
        elif method.upper() == "POST":
            if files:
                upload_headers = {k: v for k, v in headers.items() if k.lower() != 'content-type'}
                response = requests.post(url, headers=upload_headers, data=data, files=files)
            else:
                response = requests.post(url, headers=headers, json=data)
        elif method.upper() == "PUT":
            response = requests.put(url, headers=headers, json=data)
        elif method.upper() == "DELETE":
            response = requests.delete(url, headers=headers)
        else:
            return {"status_code": 0, "success": False, "error": f"Unsupported method: {method}"}

        try:
            response_data = response.json()
        except:
            response_data = {"raw_response": response.text}

        return {
            "status_code": response.status_code,
            "success": response.status_code < 400,
            "data": response_data
        }
    except Exception as e:
        return {
            "status_code": 0,
            "success": False,
            "error": str(e)
        }

def print_result(test_name: str, result: Dict[str, Any], expected_codes: list = [200]):
    """Print test result with formatting."""
    status_code = result.get('status_code', 0)
    success = status_code in expected_codes

    if success:
        print(f"   ✅ {test_name}: {status_code}")
    else:
        print(f"   ❌ {test_name}: {status_code}")
        if 'error' in result:
            print(f"      Error: {result['error']}")
        elif 'data' in result and isinstance(result['data'], dict) and 'detail' in result['data']:
            print(f"      Detail: {result['data']['detail']}")

    return success

def test_authentication():
    """Test all authentication endpoints."""
    print("\n🔐 AUTHENTICATION ENDPOINTS")
    print("-" * 50)

    results = []
    timestamp = int(time.time())

    # Test 1: User Registration
    print("1. User Registration")
    result = test_endpoint(
        "POST",
        f"{BASE_URL}/api/v1/auth/register",
        {"Content-Type": "application/json"},
        {
            "email": f"test_user_{timestamp}@example.com",
            "password": "TestPassword123!",
            "full_name": "Test User Final",
            "username": f"test_user_{timestamp}"
        }
    )
    success = print_result("Registration", result)
    results.append(success)

    # Test 2: User Login
    print("\n2. User Login")
    login_result = test_endpoint(
        "POST",
        f"{BASE_URL}/api/v1/auth/login",
        {"Content-Type": "application/json"},
        {
            "email": f"test_user_{timestamp}@example.com",
            "password": "TestPassword123!"
        }
    )
    success = print_result("Login", login_result)
    results.append(success)

    # Get tokens for subsequent tests
    user_token = None
    refresh_token = None
    if login_result.get('success') and login_result.get('data', {}).get('access_token'):
        user_token = login_result['data']['access_token']
        refresh_token = login_result['data']['refresh_token']
        print(f"   📝 User token obtained for subsequent tests")

    # Test 3: Token Refresh
    print("\n3. Token Refresh")
    if refresh_token:
        result = test_endpoint(
            "POST",
            f"{BASE_URL}/api/v1/auth/refresh",
            {"Content-Type": "application/json"},
            {"refresh_token": refresh_token}
        )
        success = print_result("Token Refresh", result)
        results.append(success)
    else:
        print("   ⚠️ Skipping token refresh - no refresh token available")
        results.append(False)

    return results, user_token

def test_documents(user_token: str):
    """Test all document endpoints."""
    print("\n📄 DOCUMENT ENDPOINTS")
    print("-" * 50)

    results = []
    auth_headers = {"Authorization": f"Bearer {user_token}"}
    timestamp = int(time.time())
    test_collection_name = f"doc_test_collection_{timestamp}"

    # Test 1: Create collection for document upload
    print("1. Create Collection for Documents")
    result = test_endpoint(
        "POST",
        f"{BASE_URL}/api/v1/collections/",
        {**auth_headers, "Content-Type": "application/json"},
        {"name": test_collection_name, "description": "Test collection for documents"}
    )
    success = print_result("Collection Creation", result)
    results.append(success)

    # Test 2: Document Upload
    print("\n2. Document Upload")
    unique_content = f"This is test document content uploaded at {timestamp}"
    result = test_endpoint(
        "POST",
        f"{BASE_URL}/api/v1/documents/upload",
        auth_headers,
        {"collection_name": test_collection_name},
        {"file": (f"test_doc_{timestamp}.txt", unique_content, "text/plain")}
    )
    success = print_result("Document Upload", result)
    results.append(success)

    # Test 3: Document List
    print("\n3. Document List")
    docs_result = test_endpoint("GET", f"{BASE_URL}/api/v1/documents/", auth_headers)
    success = print_result("Document List", docs_result)
    results.append(success)

    # Test 4: Get Specific Document
    print("\n4. Get Specific Document")
    real_doc_id = "00000000-0000-0000-0000-000000000000"  # Default fallback
    if docs_result.get('success') and docs_result.get('data', {}).get('documents'):
        documents = docs_result['data']['documents']
        if documents and len(documents) > 0:
            doc = documents[0]
            real_doc_id = doc.get('id') or doc.get('document_id') or doc.get('doc_id') or real_doc_id
            print(f"   📝 Using document ID: {real_doc_id}")

    result = test_endpoint(
        "GET",
        f"{BASE_URL}/api/v1/documents/{real_doc_id}",
        auth_headers
    )
    success = print_result("Get Document", result)
    results.append(success)

    return results

def test_collections(user_token: str):
    """Test all collection endpoints."""
    print("\n📚 COLLECTION ENDPOINTS")
    print("-" * 50)

    results = []
    auth_headers = {"Authorization": f"Bearer {user_token}", "Content-Type": "application/json"}
    timestamp = int(time.time())
    test_collection_name = f"collection_test_{timestamp}"

    # Test 1: Collection Creation
    print("1. Collection Creation")
    result = test_endpoint(
        "POST",
        f"{BASE_URL}/api/v1/collections/",
        auth_headers,
        {"name": test_collection_name, "description": "Test collection for endpoints"}
    )
    success = print_result("Collection Creation", result)
    results.append(success)

    # Test 2: Get Collection
    print("\n2. Get Collection")
    result = test_endpoint(
        "GET",
        f"{BASE_URL}/api/v1/collections/{test_collection_name}",
        auth_headers
    )
    success = print_result("Get Collection", result)
    results.append(success)

    # Test 3: Collection Stats
    print("\n3. Collection Stats")
    result = test_endpoint(
        "GET",
        f"{BASE_URL}/api/v1/collections/{test_collection_name}/stats",
        auth_headers
    )
    success = print_result("Collection Stats", result)
    results.append(success)

    # Test 4: List Collections
    print("\n4. List Collections")
    result = test_endpoint(
        "GET",
        f"{BASE_URL}/api/v1/collections/",
        auth_headers
    )
    success = print_result("List Collections", result)
    results.append(success)

    return results, test_collection_name

def test_queries(user_token: str, collection_name: str):
    """Test all query endpoints."""
    print("\n🔍 QUERY ENDPOINTS")
    print("-" * 50)

    results = []
    auth_headers = {"Authorization": f"Bearer {user_token}", "Content-Type": "application/json"}

    # Test 1: Query Execute
    print("1. Query Execute")
    result = test_endpoint(
        "POST",
        f"{BASE_URL}/api/v1/query/",
        auth_headers,
        {"query": "test query for final verification", "collection_name": collection_name}
    )
    success = print_result("Query Execute", result)
    results.append(success)

    # Test 2: Query History
    print("\n2. Query History")
    result = test_endpoint(
        "GET",
        f"{BASE_URL}/api/v1/query/history/",
        auth_headers
    )
    success = print_result("Query History", result)
    results.append(success)

    return results

def test_system_endpoints():
    """Test all system endpoints with admin token."""
    print("\n🛠️ SYSTEM ENDPOINTS (ADMIN)")
    print("-" * 50)

    results = []
    admin_headers = {"Authorization": f"Bearer {ADMIN_TOKEN}"}

    # Test 1: Health Check
    print("1. Health Check")
    result = test_endpoint("GET", f"{BASE_URL}/api/v1/system/health", {})
    success = print_result("Health Check", result)
    results.append(success)

    # Test 2: System Stats
    print("\n2. System Stats")
    result = test_endpoint("GET", f"{BASE_URL}/api/v1/system/stats", admin_headers)
    success = print_result("System Stats", result)
    results.append(success)

    # Test 3: System Metrics
    print("\n3. System Metrics")
    result = test_endpoint("GET", f"{BASE_URL}/api/v1/system/metrics", admin_headers)
    success = print_result("System Metrics", result)
    results.append(success)

    # Test 4: System Logs
    print("\n4. System Logs")
    result = test_endpoint("GET", f"{BASE_URL}/api/v1/system/logs", admin_headers)
    success = print_result("System Logs", result)
    results.append(success)

    # Test 5: System Alerts
    print("\n5. System Alerts")
    result = test_endpoint("GET", f"{BASE_URL}/api/v1/system/alerts", admin_headers)
    success = print_result("System Alerts", result)
    results.append(success)

    return results

def main():
    """Main test function."""
    print("🎯 COMPLETE API TEST - FINAL VERIFICATION")
    print("=" * 60)
    print(f"Testing against: {BASE_URL}")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    all_results = []

    # Test Authentication
    auth_results, user_token = test_authentication()
    all_results.extend(auth_results)

    if user_token:
        # Test Documents
        doc_results = test_documents(user_token)
        all_results.extend(doc_results)

        # Test Collections
        collection_results, collection_name = test_collections(user_token)
        all_results.extend(collection_results)

        # Test Queries
        query_results = test_queries(user_token, collection_name)
        all_results.extend(query_results)
    else:
        print("\n❌ Cannot test user endpoints - no user token available")
        # Add failed results for skipped tests
        all_results.extend([False] * 10)  # Approximate number of user endpoints

    # Test System Endpoints
    system_results = test_system_endpoints()
    all_results.extend(system_results)

    # Calculate final results
    total_tests = len(all_results)
    successful_tests = sum(all_results)
    success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0

    print("\n" + "=" * 60)
    print("📊 FINAL RESULTS")
    print("=" * 60)
    print(f"Total Tests: {total_tests}")
    print(f"Successful: {successful_tests}")
    print(f"Failed: {total_tests - successful_tests}")
    print(f"Success Rate: {success_rate:.1f}%")

    if success_rate == 100:
        print("\n🎉 PERFECT SCORE! ALL ENDPOINTS WORKING!")
        print("🚀 Cognify API is 100% operational and ready for production!")
    elif success_rate >= 90:
        print(f"\n✅ EXCELLENT! {success_rate:.1f}% success rate")
        print("🔧 Minor issues to address")
    elif success_rate >= 75:
        print(f"\n⚠️ GOOD: {success_rate:.1f}% success rate")
        print("🔧 Some issues need attention")
    else:
        print(f"\n❌ NEEDS WORK: {success_rate:.1f}% success rate")
        print("🔧 Significant issues to resolve")

    print("=" * 60)

if __name__ == "__main__":
    main()
