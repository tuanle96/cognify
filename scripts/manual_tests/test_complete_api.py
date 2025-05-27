#!/usr/bin/env python3
"""
Complete API Testing Script for Cognify
Tests all API endpoints with example user creation and authentication.
"""

import requests
import json
import time
from typing import Dict, Any, Optional

# Configuration
BASE_URL = "http://localhost:8000"
API_V1 = f"{BASE_URL}/api/v1"

# Test user data
TEST_USER = {
    "email": "test@example.com",
    "password": "TestPassword123",
    "full_name": "Test User",
    "username": "testuser"
}

ADMIN_USER = {
    "email": "admin@example.com",
    "password": "AdminPassword123",
    "full_name": "Admin User",
    "username": "admin"
}

class CognifyAPITester:
    def __init__(self):
        self.session = requests.Session()
        self.access_token = None
        self.admin_token = None
        self.user_id = None
        self.admin_id = None

    def log(self, message: str, status: str = "INFO"):
        """Log test results."""
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] {status}: {message}")

    def make_request(self, method: str, endpoint: str, data: Dict = None,
                    headers: Dict = None, use_auth: bool = False,
                    use_admin: bool = False) -> requests.Response:
        """Make HTTP request with optional authentication."""
        url = f"{API_V1}{endpoint}"

        # Prepare headers
        req_headers = {"Content-Type": "application/json"}
        if headers:
            req_headers.update(headers)

        # Add authentication if needed
        if use_auth and self.access_token:
            req_headers["Authorization"] = f"Bearer {self.access_token}"
        elif use_admin and self.admin_token:
            req_headers["Authorization"] = f"Bearer {self.admin_token}"

        # Make request
        try:
            response = self.session.request(
                method=method,
                url=url,
                json=data,
                headers=req_headers,
                timeout=30
            )
            return response
        except Exception as e:
            self.log(f"Request failed: {e}", "ERROR")
            return None

    def test_health_endpoints(self):
        """Test health check endpoints."""
        self.log("=== Testing Health Endpoints ===")

        # Root health check
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=10)
            if response.status_code == 200:
                self.log("✅ Root health check: PASSED")
            else:
                self.log(f"❌ Root health check: FAILED ({response.status_code})")
        except Exception as e:
            self.log(f"❌ Root health check: ERROR - {e}")

        # System health check
        response = self.make_request("GET", "/system/health")
        if response and response.status_code == 200:
            self.log("✅ System health check: PASSED")
            health_data = response.json()
            self.log(f"   Database status: {health_data['services']['database']['status']}")
        else:
            self.log(f"❌ System health check: FAILED ({response.status_code if response else 'No response'})")

        # Chunking health check
        response = self.make_request("GET", "/chunking/health")
        if response and response.status_code == 200:
            self.log("✅ Chunking health check: PASSED")
        else:
            self.log(f"❌ Chunking health check: FAILED ({response.status_code if response else 'No response'})")

    def test_user_registration(self):
        """Test user registration."""
        self.log("=== Testing User Registration ===")

        # Register test user
        response = self.make_request("POST", "/auth/register", TEST_USER)
        if response and response.status_code == 200:
            self.log("✅ User registration: PASSED")
            user_data = response.json()
            self.user_id = user_data["user_id"]
            self.log(f"   User ID: {self.user_id}")
        else:
            self.log(f"❌ User registration: FAILED ({response.status_code if response else 'No response'})")
            if response:
                self.log(f"   Error: {response.text}")

        # Try to register admin user
        response = self.make_request("POST", "/auth/register", ADMIN_USER)
        if response and response.status_code == 200:
            self.log("✅ Admin registration: PASSED")
            admin_data = response.json()
            self.admin_id = admin_data["user_id"]
        else:
            self.log(f"❌ Admin registration: FAILED ({response.status_code if response else 'No response'})")

    def test_user_login(self):
        """Test user login."""
        self.log("=== Testing User Login ===")

        # Login test user
        login_data = {
            "email": TEST_USER["email"],
            "password": TEST_USER["password"]
        }
        response = self.make_request("POST", "/auth/login", login_data)
        if response and response.status_code == 200:
            self.log("✅ User login: PASSED")
            token_data = response.json()
            self.access_token = token_data["access_token"]
            self.log(f"   Access token received: {self.access_token[:20]}...")
        else:
            self.log(f"❌ User login: FAILED ({response.status_code if response else 'No response'})")
            if response:
                self.log(f"   Error: {response.text}")

        # Login admin user
        admin_login = {
            "email": ADMIN_USER["email"],
            "password": ADMIN_USER["password"]
        }
        response = self.make_request("POST", "/auth/login", admin_login)
        if response and response.status_code == 200:
            self.log("✅ Admin login: PASSED")
            token_data = response.json()
            self.admin_token = token_data["access_token"]
        else:
            self.log(f"❌ Admin login: FAILED ({response.status_code if response else 'No response'})")

    def test_auth_endpoints(self):
        """Test authentication endpoints."""
        self.log("=== Testing Auth Endpoints ===")

        # Get current user info
        response = self.make_request("GET", "/auth/me", use_auth=True)
        if response and response.status_code == 200:
            self.log("✅ Get current user: PASSED")
            user_data = response.json()
            self.log(f"   User: {user_data['full_name']} ({user_data['email']})")
        else:
            self.log(f"❌ Get current user: FAILED ({response.status_code if response else 'No response'})")

    def test_chunking_endpoints(self):
        """Test chunking endpoints."""
        self.log("=== Testing Chunking Endpoints ===")

        # Test basic chunking
        chunk_data = {
            "content": "def hello_world():\n    print('Hello, World!')\n\nclass Calculator:\n    def add(self, a, b):\n        return a + b",
            "language": "python",
            "strategy": "hybrid"
        }
        response = self.make_request("POST", "/chunking/", chunk_data)
        if response and response.status_code == 200:
            self.log("✅ Code chunking: PASSED")
            result = response.json()
            self.log(f"   Chunks created: {result['total_chunks']}")
            self.log(f"   Strategy used: {result['strategy_used']}")
        else:
            self.log(f"❌ Code chunking: FAILED ({response.status_code if response else 'No response'})")

        # Test chunking stats
        response = self.make_request("GET", "/chunking/stats")
        if response and response.status_code == 200:
            self.log("✅ Chunking stats: PASSED")
        else:
            self.log(f"❌ Chunking stats: FAILED ({response.status_code if response else 'No response'})")

        # Test supported languages
        response = self.make_request("GET", "/chunking/supported-languages")
        if response and response.status_code == 200:
            self.log("✅ Supported languages: PASSED")
            langs = response.json()
            self.log(f"   Languages: {len(langs['supported_languages'])}")
        else:
            self.log(f"❌ Supported languages: FAILED ({response.status_code if response else 'No response'})")

        # Test chunking test endpoint
        response = self.make_request("POST", "/chunking/test")
        if response and response.status_code == 200:
            self.log("✅ Chunking test: PASSED")
        else:
            self.log(f"❌ Chunking test: FAILED ({response.status_code if response else 'No response'})")

    def test_documents_endpoints(self):
        """Test documents endpoints."""
        self.log("=== Testing Documents Endpoints ===")

        # Test list documents (should require auth)
        response = self.make_request("GET", "/documents/")
        if response and response.status_code == 401:
            self.log("✅ Documents list (no auth): PASSED (401 as expected)")
        else:
            self.log(f"❌ Documents list (no auth): FAILED ({response.status_code if response else 'No response'})")

        # Test list documents with auth
        response = self.make_request("GET", "/documents/", use_auth=True)
        if response and response.status_code in [200, 404]:
            self.log("✅ Documents list (with auth): PASSED")
            if response.status_code == 200:
                docs = response.json()
                self.log(f"   Documents found: {len(docs.get('documents', []))}")
        else:
            self.log(f"❌ Documents list (with auth): FAILED ({response.status_code if response else 'No response'})")

        # Test upload document
        doc_data = {
            "title": "Test Document",
            "content": "This is a test document for API testing.",
            "file_type": "text",
            "collection_id": None
        }
        response = self.make_request("POST", "/documents/", doc_data, use_auth=True)
        if response and response.status_code in [200, 201]:
            self.log("✅ Document upload: PASSED")
            doc_result = response.json()
            self.log(f"   Document ID: {doc_result.get('document_id', 'N/A')}")
        else:
            self.log(f"❌ Document upload: FAILED ({response.status_code if response else 'No response'})")

    def test_collections_endpoints(self):
        """Test collections endpoints."""
        self.log("=== Testing Collections Endpoints ===")

        # Test list collections (should require auth)
        response = self.make_request("GET", "/collections/")
        if response and response.status_code == 401:
            self.log("✅ Collections list (no auth): PASSED (401 as expected)")
        else:
            self.log(f"❌ Collections list (no auth): FAILED ({response.status_code if response else 'No response'})")

        # Test list collections with auth
        response = self.make_request("GET", "/collections/", use_auth=True)
        if response and response.status_code in [200, 404]:
            self.log("✅ Collections list (with auth): PASSED")
            if response.status_code == 200:
                collections = response.json()
                self.log(f"   Collections found: {len(collections.get('collections', []))}")
        else:
            self.log(f"❌ Collections list (with auth): FAILED ({response.status_code if response else 'No response'})")

        # Test create collection
        collection_data = {
            "name": "Test Collection",
            "description": "A test collection for API testing",
            "is_public": False
        }
        response = self.make_request("POST", "/collections/", collection_data, use_auth=True)
        if response and response.status_code in [200, 201]:
            self.log("✅ Collection creation: PASSED")
            collection_result = response.json()
            self.log(f"   Collection ID: {collection_result.get('collection_id', 'N/A')}")
        else:
            self.log(f"❌ Collection creation: FAILED ({response.status_code if response else 'No response'})")

    def test_query_endpoints(self):
        """Test query endpoints."""
        self.log("=== Testing Query Endpoints ===")

        # Test search (should require auth)
        search_data = {
            "query": "test document",
            "limit": 10,
            "collection_id": None
        }
        response = self.make_request("POST", "/query/search", search_data)
        if response and response.status_code == 401:
            self.log("✅ Query search (no auth): PASSED (401 as expected)")
        else:
            self.log(f"❌ Query search (no auth): FAILED ({response.status_code if response else 'No response'})")

        # Test search with auth
        response = self.make_request("POST", "/query/search", search_data, use_auth=True)
        if response and response.status_code in [200, 404]:
            self.log("✅ Query search (with auth): PASSED")
            if response.status_code == 200:
                results = response.json()
                self.log(f"   Search results: {len(results.get('results', []))}")
        else:
            self.log(f"❌ Query search (with auth): FAILED ({response.status_code if response else 'No response'})")

        # Test semantic search
        semantic_data = {
            "query": "hello world function",
            "limit": 5,
            "similarity_threshold": 0.7
        }
        response = self.make_request("POST", "/query/semantic", semantic_data, use_auth=True)
        if response and response.status_code in [200, 404]:
            self.log("✅ Semantic search: PASSED")
        else:
            self.log(f"❌ Semantic search: FAILED ({response.status_code if response else 'No response'})")

    def test_system_endpoints(self):
        """Test system endpoints."""
        self.log("=== Testing System Endpoints ===")

        # Test system stats (should require admin)
        response = self.make_request("GET", "/system/stats", use_auth=True)
        if response and response.status_code in [200, 403]:
            if response.status_code == 403:
                self.log("✅ System stats (user): PASSED (403 as expected)")
            else:
                self.log("✅ System stats (user): PASSED")
        else:
            self.log(f"❌ System stats (user): FAILED ({response.status_code if response else 'No response'})")

        # Test system metrics
        response = self.make_request("GET", "/system/metrics", use_auth=True)
        if response and response.status_code in [200, 403]:
            if response.status_code == 200:
                self.log("✅ System metrics: PASSED")
                metrics = response.json()
                self.log(f"   CPU usage: {metrics.get('cpu_usage', 'N/A')}%")
            else:
                self.log("✅ System metrics: PASSED (403 as expected)")
        else:
            self.log(f"❌ System metrics: FAILED ({response.status_code if response else 'No response'})")

    def run_all_tests(self):
        """Run all API tests."""
        self.log("🚀 Starting Complete Cognify API Test Suite")
        self.log(f"Base URL: {BASE_URL}")

        # Test sequence
        self.test_health_endpoints()
        self.test_user_registration()
        self.test_user_login()
        self.test_auth_endpoints()
        self.test_chunking_endpoints()
        self.test_documents_endpoints()
        self.test_collections_endpoints()
        self.test_query_endpoints()
        self.test_system_endpoints()

        self.log("✨ API Test Suite Completed!")

        # Summary
        self.log("=== Test Summary ===")
        if self.access_token:
            self.log(f"✅ User authentication: SUCCESS")
        else:
            self.log(f"❌ User authentication: FAILED")

        if self.admin_token:
            self.log(f"✅ Admin authentication: SUCCESS")
        else:
            self.log(f"❌ Admin authentication: FAILED")


def main():
    """Main test function."""
    print("=" * 60)
    print("🧠 COGNIFY API COMPLETE TEST SUITE")
    print("=" * 60)

    tester = CognifyAPITester()
    tester.run_all_tests()

    print("=" * 60)
    print("Test completed! Check the logs above for detailed results.")
    print("=" * 60)


if __name__ == "__main__":
    main()
