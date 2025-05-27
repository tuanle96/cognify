#!/usr/bin/env python3
"""
Comprehensive API Testing Script for Cognify
Tests all API endpoints with real data and scenarios
"""

import requests
import json
import time
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

class CognifyAPITester:
    """Comprehensive API tester for Cognify system"""

    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})

        # Test data
        self.test_user = {
            "email": "api_tester@cognify.test",
            "password": "ApiTest123!",
            "full_name": "API Comprehensive Tester",
            "username": "api_tester"
        }

        self.test_org = {
            "name": "Test Organization",
            "slug": "test-org",
            "description": "Organization for comprehensive API testing"
        }

        self.test_workspace = {
            "name": "Test Workspace",
            "slug": "test-workspace",
            "description": "Workspace for comprehensive API testing"
        }

        # Test results
        self.results = {
            "start_time": datetime.now().isoformat(),
            "tests": [],
            "summary": {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "errors": []
            }
        }

        # Test content
        self.test_content = self._load_test_content()

    def _load_test_content(self) -> str:
        """Load test content for document testing"""
        try:
            # Try to load from examples
            content_file = Path(__file__).parent.parent.parent / "examples" / "comprehensive_test_content.py"
            if content_file.exists():
                with open(content_file, 'r', encoding='utf-8') as f:
                    return f.read()
        except:
            pass

        # Fallback to sample content
        return '''
def sample_function():
    """Sample Python function for testing RAG search capabilities."""
    data = {
        "name": "Cognify API Test",
        "version": "1.0.0",
        "features": [
            "Document processing",
            "Semantic search",
            "Multi-tenant architecture",
            "API key authentication"
        ]
    }
    return data

class SampleClass:
    """Sample class for testing code analysis and chunking."""

    def __init__(self, name: str):
        self.name = name
        self.items = []

    def add_item(self, item: str) -> int:
        """Add item to the collection."""
        self.items.append(item)
        return len(self.items)

    def get_items(self) -> list:
        """Get all items in the collection."""
        return self.items.copy()

    def search_items(self, query: str) -> list:
        """Search for items containing the query."""
        return [item for item in self.items if query.lower() in item.lower()]

# Test data for various scenarios
TEST_SCENARIOS = [
    "authentication and authorization",
    "document upload and processing",
    "semantic search capabilities",
    "multi-tenant data isolation",
    "API key management",
    "chunking strategies",
    "embedding generation",
    "vector database operations"
]
'''

    def log_test(self, test_name: str, success: bool, details: Dict[str, Any] = None):
        """Log test result"""
        self.results["tests"].append({
            "name": test_name,
            "success": success,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        })

        self.results["summary"]["total"] += 1
        if success:
            self.results["summary"]["passed"] += 1
            print(f"✅ {test_name}")
        else:
            self.results["summary"]["failed"] += 1
            print(f"❌ {test_name}")
            if details and "error" in details:
                print(f"   Error: {details['error']}")
                self.results["summary"]["errors"].append(f"{test_name}: {details['error']}")

    def test_system_health(self) -> bool:
        """Test system health endpoints"""
        print("\n🔍 Testing System Health...")

        try:
            # Basic health check
            response = self.session.get(f"{self.base_url}/health", timeout=10)
            self.log_test("System Health Check", response.status_code == 200, {
                "status_code": response.status_code,
                "response": response.json() if response.status_code == 200 else response.text
            })

            # API health check
            response = self.session.get(f"{self.base_url}/api/v1/system/health", timeout=10)
            self.log_test("API Health Check", response.status_code == 200, {
                "status_code": response.status_code,
                "response": response.json() if response.status_code == 200 else response.text
            })

            # Root endpoint
            response = self.session.get(f"{self.base_url}/", timeout=10)
            self.log_test("Root Endpoint", response.status_code == 200, {
                "status_code": response.status_code,
                "response": response.json() if response.status_code == 200 else response.text
            })

            return True

        except Exception as e:
            self.log_test("System Health Tests", False, {"error": str(e)})
            return False

    def test_authentication(self) -> bool:
        """Test authentication endpoints"""
        print("\n🔐 Testing Authentication...")

        try:
            # Register user
            response = self.session.post(f"{self.base_url}/api/v1/auth/register", json=self.test_user)
            register_success = response.status_code in [200, 201, 409]  # 409 if user exists
            self.log_test("User Registration", register_success, {
                "status_code": response.status_code,
                "response": response.json() if response.status_code in [200, 201, 409] else response.text
            })

            # Login user
            login_data = {"email": self.test_user["email"], "password": self.test_user["password"]}
            response = self.session.post(f"{self.base_url}/api/v1/auth/login", json=login_data)

            if response.status_code == 200:
                data = response.json()
                token = data.get('access_token')
                if token:
                    self.session.headers.update({'Authorization': f'Bearer {token}'})
                    self.log_test("User Login", True, {"token_received": True})
                    return True
                else:
                    self.log_test("User Login", False, {"error": "No token in response"})
                    return False
            else:
                self.log_test("User Login", False, {
                    "status_code": response.status_code,
                    "response": response.text
                })
                return False

        except Exception as e:
            self.log_test("Authentication Tests", False, {"error": str(e)})
            return False

    def test_organizations(self) -> Optional[str]:
        """Test organization endpoints"""
        print("\n🏢 Testing Organizations...")

        try:
            # Create organization
            response = self.session.post(f"{self.base_url}/api/v1/organizations/", json=self.test_org)

            if response.status_code in [200, 201]:
                org_data = response.json()
                org_id = org_data.get('id')
                self.log_test("Create Organization", True, {"org_id": org_id})

                # Get organizations
                response = self.session.get(f"{self.base_url}/api/v1/organizations/")
                self.log_test("List Organizations", response.status_code == 200, {
                    "status_code": response.status_code,
                    "count": len(response.json()) if response.status_code == 200 else 0
                })

                # Get specific organization
                if org_id:
                    response = self.session.get(f"{self.base_url}/api/v1/organizations/{org_id}")
                    self.log_test("Get Organization", response.status_code == 200, {
                        "status_code": response.status_code
                    })

                return org_id

            elif response.status_code == 409:
                # Organization already exists, try to get it
                response = self.session.get(f"{self.base_url}/api/v1/organizations/")
                if response.status_code == 200:
                    orgs = response.json()
                    for org in orgs:
                        if org.get('slug') == self.test_org['slug']:
                            self.log_test("Create Organization", True, {"note": "Already exists", "org_id": org['id']})
                            return org['id']

                self.log_test("Create Organization", False, {"error": "Conflict but couldn't find existing org"})
                return None
            else:
                self.log_test("Create Organization", False, {
                    "status_code": response.status_code,
                    "response": response.text
                })
                return None

        except Exception as e:
            self.log_test("Organization Tests", False, {"error": str(e)})
            return None

    def test_workspaces(self, org_id: str) -> Optional[str]:
        """Test workspace endpoints"""
        print("\n🏗️ Testing Workspaces...")

        try:
            # Create workspace
            workspace_data = {**self.test_workspace, "organization_id": org_id}
            response = self.session.post(f"{self.base_url}/api/v1/workspaces/", json=workspace_data)

            if response.status_code in [200, 201]:
                workspace_data = response.json()
                workspace_id = workspace_data.get('id')
                self.log_test("Create Workspace", True, {"workspace_id": workspace_id})

                # Get workspaces
                response = self.session.get(f"{self.base_url}/api/v1/workspaces/")
                self.log_test("List Workspaces", response.status_code == 200, {
                    "status_code": response.status_code,
                    "count": len(response.json()) if response.status_code == 200 else 0
                })

                return workspace_id

            elif response.status_code == 409:
                # Workspace already exists, try to get it
                response = self.session.get(f"{self.base_url}/api/v1/workspaces/")
                if response.status_code == 200:
                    workspaces = response.json()
                    for ws in workspaces:
                        if ws.get('slug') == self.test_workspace['slug']:
                            self.log_test("Create Workspace", True, {"note": "Already exists", "workspace_id": ws['id']})
                            return ws['id']

                self.log_test("Create Workspace", False, {"error": "Conflict but couldn't find existing workspace"})
                return None
            else:
                self.log_test("Create Workspace", False, {
                    "status_code": response.status_code,
                    "response": response.text
                })
                return None

        except Exception as e:
            self.log_test("Workspace Tests", False, {"error": str(e)})
            return None

    def test_collections(self, workspace_id: str) -> Optional[str]:
        """Test collection endpoints"""
        print("\n📁 Testing Collections...")

        try:
            # Create collection
            collection_data = {
                "name": "Test Collection",
                "description": "Collection for comprehensive API testing",
                "workspace_id": workspace_id
            }
            response = self.session.post(f"{self.base_url}/api/v1/collections/", json=collection_data)

            if response.status_code in [200, 201]:
                collection = response.json()
                collection_id = collection.get('id')
                self.log_test("Create Collection", True, {"collection_id": collection_id})

                # Get collections
                response = self.session.get(f"{self.base_url}/api/v1/collections/")
                self.log_test("List Collections", response.status_code == 200, {
                    "status_code": response.status_code,
                    "count": len(response.json()) if response.status_code == 200 else 0
                })

                # Get specific collection
                if collection_id:
                    response = self.session.get(f"{self.base_url}/api/v1/collections/{collection_id}")
                    self.log_test("Get Collection", response.status_code == 200, {
                        "status_code": response.status_code
                    })

                return collection_id
            else:
                self.log_test("Create Collection", False, {
                    "status_code": response.status_code,
                    "response": response.text
                })
                return None

        except Exception as e:
            self.log_test("Collection Tests", False, {"error": str(e)})
            return None

    def test_documents(self, collection_id: str) -> Optional[str]:
        """Test document endpoints"""
        print("\n📄 Testing Documents...")

        try:
            # Upload document
            document_data = {
                "title": "Test Code Implementation",
                "content": self.test_content,
                "content_type": "python_code",
                "collection_id": collection_id,
                "metadata": {
                    "filename": "test_content.py",
                    "language": "python",
                    "lines_of_code": len(self.test_content.split('\n')),
                    "description": "Test content for comprehensive API testing"
                }
            }
            response = self.session.post(f"{self.base_url}/api/v1/documents/", json=document_data)

            if response.status_code in [200, 201]:
                document = response.json()
                document_id = document.get('id')
                self.log_test("Upload Document", True, {
                    "document_id": document_id,
                    "content_size": len(self.test_content)
                })

                # Get documents
                response = self.session.get(f"{self.base_url}/api/v1/documents/")
                self.log_test("List Documents", response.status_code == 200, {
                    "status_code": response.status_code,
                    "count": len(response.json()) if response.status_code == 200 else 0
                })

                # Get specific document
                if document_id:
                    response = self.session.get(f"{self.base_url}/api/v1/documents/{document_id}")
                    self.log_test("Get Document", response.status_code == 200, {
                        "status_code": response.status_code
                    })

                return document_id
            else:
                self.log_test("Upload Document", False, {
                    "status_code": response.status_code,
                    "response": response.text
                })
                return None

        except Exception as e:
            self.log_test("Document Tests", False, {"error": str(e)})
            return None

    def test_chunking_api(self) -> bool:
        """Test chunking API endpoints"""
        print("\n🔪 Testing Chunking API...")

        try:
            # Test chunking health
            response = self.session.get(f"{self.base_url}/api/v1/chunking/health")
            self.log_test("Chunking Health", response.status_code == 200, {
                "status_code": response.status_code,
                "response": response.json() if response.status_code == 200 else response.text
            })

            # Test supported languages
            response = self.session.get(f"{self.base_url}/api/v1/chunking/supported-languages")
            self.log_test("Supported Languages", response.status_code == 200, {
                "status_code": response.status_code,
                "count": len(response.json().get('supported_languages', [])) if response.status_code == 200 else 0
            })

            # Test chunking with sample content
            chunk_data = {
                "content": self.test_content,
                "content_type": "python",
                "strategy": "agentic",
                "chunk_size": 1000,
                "chunk_overlap": 200
            }
            response = self.session.post(f"{self.base_url}/api/v1/chunking/chunk", json=chunk_data)

            if response.status_code == 200:
                result = response.json()
                chunks = result.get('chunks', [])
                self.log_test("Content Chunking", True, {
                    "chunk_count": len(chunks),
                    "total_tokens": result.get('total_tokens', 0),
                    "quality_score": result.get('quality_score', 0)
                })
            else:
                self.log_test("Content Chunking", False, {
                    "status_code": response.status_code,
                    "response": response.text
                })

            return True

        except Exception as e:
            self.log_test("Chunking API Tests", False, {"error": str(e)})
            return False

    def test_query_search(self, collection_id: str) -> bool:
        """Test query and search endpoints"""
        print("\n🔍 Testing Query & Search...")

        # Wait for indexing
        print("   ⏳ Waiting for document indexing...")
        time.sleep(5)

        try:
            # Test search queries
            test_queries = [
                "How does the sample function work?",
                "What classes are defined in the code?",
                "How to add items to the collection?",
                "What are the main features?",
                "How does the search functionality work?"
            ]

            successful_queries = 0

            for i, query in enumerate(test_queries, 1):
                query_data = {
                    "query": query,
                    "collection_id": collection_id,
                    "top_k": 3,
                    "include_metadata": True
                }

                response = self.session.post(f"{self.base_url}/api/v1/query/search", json=query_data)

                if response.status_code == 200:
                    result = response.json()
                    search_results = result.get('results', [])
                    successful_queries += 1

                    self.log_test(f"Search Query {i}", True, {
                        "query": query,
                        "results_count": len(search_results),
                        "top_score": search_results[0].get('score', 0) if search_results else 0
                    })
                else:
                    self.log_test(f"Search Query {i}", False, {
                        "query": query,
                        "status_code": response.status_code,
                        "response": response.text
                    })

                time.sleep(0.5)  # Small delay between queries

            # Overall search success
            search_success_rate = successful_queries / len(test_queries)
            self.log_test("Overall Search Success", search_success_rate >= 0.7, {
                "success_rate": f"{search_success_rate:.2%}",
                "successful_queries": successful_queries,
                "total_queries": len(test_queries)
            })

            return search_success_rate >= 0.7

        except Exception as e:
            self.log_test("Query & Search Tests", False, {"error": str(e)})
            return False

    def test_api_keys(self) -> bool:
        """Test API key management"""
        print("\n🔑 Testing API Keys...")

        try:
            # Create API key
            api_key_data = {
                "name": "Test API Key",
                "description": "API key for comprehensive testing",
                "permissions": ["read", "write"]
            }
            response = self.session.post(f"{self.base_url}/api/v1/api-keys/", json=api_key_data)

            if response.status_code in [200, 201]:
                api_key_result = response.json()
                api_key = api_key_result.get('api_key')
                api_key_id = api_key_result.get('id')

                self.log_test("Create API Key", True, {
                    "api_key_id": api_key_id,
                    "has_key": bool(api_key)
                })

                # List API keys
                response = self.session.get(f"{self.base_url}/api/v1/api-keys/")
                self.log_test("List API Keys", response.status_code == 200, {
                    "status_code": response.status_code,
                    "count": len(response.json()) if response.status_code == 200 else 0
                })

                # Test API key authentication (if we have the key)
                if api_key:
                    # Create new session with API key
                    api_session = requests.Session()
                    api_session.headers.update({
                        'Content-Type': 'application/json',
                        'X-API-Key': api_key
                    })

                    # Test API key access
                    response = api_session.get(f"{self.base_url}/api/v1/system/health")
                    self.log_test("API Key Authentication", response.status_code == 200, {
                        "status_code": response.status_code
                    })

                return True
            else:
                self.log_test("Create API Key", False, {
                    "status_code": response.status_code,
                    "response": response.text
                })
                return False

        except Exception as e:
            self.log_test("API Key Tests", False, {"error": str(e)})
            return False

    def test_system_endpoints(self) -> bool:
        """Test system management endpoints"""
        print("\n⚙️ Testing System Endpoints...")

        try:
            # Test system status
            response = self.session.get(f"{self.base_url}/api/v1/system/status")
            self.log_test("System Status", response.status_code == 200, {
                "status_code": response.status_code,
                "response": response.json() if response.status_code == 200 else response.text
            })

            # Test system metrics
            response = self.session.get(f"{self.base_url}/api/v1/system/metrics")
            self.log_test("System Metrics", response.status_code == 200, {
                "status_code": response.status_code
            })

            # Test system settings
            response = self.session.get(f"{self.base_url}/api/v1/system/settings")
            self.log_test("System Settings", response.status_code == 200, {
                "status_code": response.status_code
            })

            return True

        except Exception as e:
            self.log_test("System Endpoint Tests", False, {"error": str(e)})
            return False

    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run all comprehensive tests"""
        print("🚀 Starting Comprehensive API Tests for Cognify")
        print("=" * 60)

        # Test system health first
        if not self.test_system_health():
            print("❌ System health check failed. Stopping tests.")
            return self.get_results()

        # Test authentication
        if not self.test_authentication():
            print("❌ Authentication failed. Stopping tests.")
            return self.get_results()

        # Test organizations
        org_id = self.test_organizations()
        if not org_id:
            print("❌ Organization tests failed. Stopping tests.")
            return self.get_results()

        # Test workspaces
        workspace_id = self.test_workspaces(org_id)
        if not workspace_id:
            print("❌ Workspace tests failed. Stopping tests.")
            return self.get_results()

        # Test collections
        collection_id = self.test_collections(workspace_id)
        if not collection_id:
            print("❌ Collection tests failed. Stopping tests.")
            return self.get_results()

        # Test documents
        document_id = self.test_documents(collection_id)
        if not document_id:
            print("❌ Document tests failed. Continuing with other tests.")

        # Test chunking API
        self.test_chunking_api()

        # Test query and search
        if collection_id:
            self.test_query_search(collection_id)

        # Test API keys
        self.test_api_keys()

        # Test system endpoints
        self.test_system_endpoints()

        return self.get_results()

    def get_results(self) -> Dict[str, Any]:
        """Get test results"""
        self.results["end_time"] = datetime.now().isoformat()
        self.results["duration"] = str(datetime.fromisoformat(self.results["end_time"]) -
                                     datetime.fromisoformat(self.results["start_time"]))
        return self.results

    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 60)
        print("📊 COMPREHENSIVE API TEST SUMMARY")
        print("=" * 60)

        summary = self.results["summary"]
        print(f"✅ Total Tests: {summary['total']}")
        print(f"✅ Passed: {summary['passed']}")
        print(f"❌ Failed: {summary['failed']}")
        print(f"📈 Success Rate: {(summary['passed'] / summary['total'] * 100):.1f}%" if summary['total'] > 0 else "0%")

        if summary['errors']:
            print(f"\n❌ Errors ({len(summary['errors'])}):")
            for error in summary['errors'][:5]:  # Show first 5 errors
                print(f"   - {error}")
            if len(summary['errors']) > 5:
                print(f"   ... and {len(summary['errors']) - 5} more errors")

        print(f"\n⏱️ Duration: {self.results.get('duration', 'Unknown')}")
        print("=" * 60)


def main():
    """Main function to run comprehensive tests"""
    import argparse

    parser = argparse.ArgumentParser(description='Comprehensive API Tests for Cognify')
    parser.add_argument('--url', default='http://localhost:8001', help='Base URL for API')
    parser.add_argument('--output', help='Output file for test results (JSON)')

    args = parser.parse_args()

    # Create tester
    tester = CognifyAPITester(args.url)

    # Run tests
    results = tester.run_comprehensive_tests()

    # Print summary
    tester.print_summary()

    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: {args.output}")

    # Exit with appropriate code
    success_rate = results["summary"]["passed"] / results["summary"]["total"] if results["summary"]["total"] > 0 else 0
    exit_code = 0 if success_rate >= 0.8 else 1

    print(f"\n🎯 Test Result: {'PASS' if exit_code == 0 else 'FAIL'}")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
