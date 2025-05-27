#!/usr/bin/env python3
"""
Detailed Chunking API Testing Script for Cognify
Tests all chunking strategies and features in detail
"""

import requests
import json
import time
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

class ChunkingAPITester:
    """Detailed chunking API tester"""

    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})

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

        # Load test content
        self.test_contents = self._load_test_contents()

    def _load_test_contents(self) -> Dict[str, str]:
        """Load various test contents for different scenarios"""

        # Python code sample
        python_code = '''
def fibonacci(n):
    """Calculate fibonacci number using dynamic programming."""
    if n <= 1:
        return n

    dp = [0] * (n + 1)
    dp[1] = 1

    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]

    return dp[n]

class Calculator:
    """Simple calculator class with basic operations."""

    def __init__(self):
        self.history = []

    def add(self, a, b):
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result

    def multiply(self, a, b):
        result = a * b
        self.history.append(f"{a} * {b} = {result}")
        return result

    def get_history(self):
        return self.history.copy()
'''

        # JavaScript code sample
        javascript_code = '''
function quickSort(arr) {
    if (arr.length <= 1) {
        return arr;
    }

    const pivot = arr[Math.floor(arr.length / 2)];
    const left = arr.filter(x => x < pivot);
    const middle = arr.filter(x => x === pivot);
    const right = arr.filter(x => x > pivot);

    return [...quickSort(left), ...middle, ...quickSort(right)];
}

class DataProcessor {
    constructor() {
        this.data = [];
    }

    addData(item) {
        this.data.push(item);
    }

    processData() {
        return this.data.map(item => item * 2);
    }

    sortData() {
        return quickSort(this.data);
    }
}
'''

        # Markdown documentation
        markdown_content = '''
# API Documentation

## Overview
This API provides comprehensive document processing and search capabilities.

## Authentication
The API supports two authentication methods:
- JWT tokens for user authentication
- API keys for programmatic access

### JWT Authentication
```bash
curl -X POST /api/v1/auth/login \\
  -H "Content-Type: application/json" \\
  -d '{"email": "user@example.com", "password": "password"}'
```

### API Key Authentication
```bash
curl -X GET /api/v1/documents \\
  -H "X-API-Key: your-api-key"
```

## Endpoints

### Documents
- `POST /api/v1/documents/` - Upload document
- `GET /api/v1/documents/` - List documents
- `GET /api/v1/documents/{id}` - Get document
- `DELETE /api/v1/documents/{id}` - Delete document

### Collections
- `POST /api/v1/collections/` - Create collection
- `GET /api/v1/collections/` - List collections
- `GET /api/v1/collections/{id}` - Get collection

### Search
- `POST /api/v1/query/search` - Search documents
- `GET /api/v1/query/history` - Get search history
'''

        # Plain text content
        plain_text = '''
The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet at least once. It is commonly used for testing fonts and keyboards.

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.

Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from and make predictions or decisions based on data. It involves training models on large datasets to identify patterns and relationships.

Natural language processing (NLP) is a field of AI that focuses on the interaction between computers and human language. It involves developing algorithms that can understand, interpret, and generate human language in a valuable way.
'''

        return {
            "python": python_code,
            "javascript": javascript_code,
            "markdown": markdown_content,
            "text": plain_text
        }

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

    def test_chunking_health(self) -> bool:
        """Test chunking service health"""
        print("\n🔍 Testing Chunking Service Health...")

        try:
            response = self.session.get(f"{self.base_url}/api/v1/chunking/health")

            if response.status_code == 200:
                data = response.json()
                self.log_test("Chunking Health Check", True, {
                    "status": data.get('status'),
                    "performance_stats": data.get('performance_stats', {})
                })
                return True
            else:
                self.log_test("Chunking Health Check", False, {
                    "status_code": response.status_code,
                    "response": response.text
                })
                return False

        except Exception as e:
            self.log_test("Chunking Health Check", False, {"error": str(e)})
            return False

    def test_supported_languages(self) -> bool:
        """Test supported languages endpoint"""
        print("\n🌐 Testing Supported Languages...")

        try:
            response = self.session.get(f"{self.base_url}/api/v1/chunking/supported-languages")

            if response.status_code == 200:
                data = response.json()
                languages = data.get('supported_languages', [])
                strategies = data.get('strategies', [])

                self.log_test("Supported Languages", True, {
                    "language_count": len(languages),
                    "languages": languages[:10],  # First 10 languages
                    "strategy_count": len(strategies),
                    "strategies": strategies
                })
                return True
            else:
                self.log_test("Supported Languages", False, {
                    "status_code": response.status_code,
                    "response": response.text
                })
                return False

        except Exception as e:
            self.log_test("Supported Languages", False, {"error": str(e)})
            return False

    def test_chunking_strategies(self) -> bool:
        """Test different chunking strategies"""
        print("\n🔪 Testing Chunking Strategies...")

        strategies = ["simple", "ast", "hybrid", "agentic", "semantic"]
        content = self.test_contents["python"]

        strategy_results = {}

        for strategy in strategies:
            try:
                chunk_data = {
                    "content": content,
                    "content_type": "python",
                    "strategy": strategy,
                    "chunk_size": 1000,
                    "chunk_overlap": 200
                }

                response = self.session.post(f"{self.base_url}/api/v1/chunking/chunk", json=chunk_data)

                if response.status_code == 200:
                    result = response.json()
                    chunks = result.get('chunks', [])

                    strategy_results[strategy] = {
                        "success": True,
                        "chunk_count": len(chunks),
                        "total_tokens": result.get('total_tokens', 0),
                        "quality_score": result.get('quality_score', 0),
                        "processing_time": result.get('processing_time', 0)
                    }

                    self.log_test(f"Chunking Strategy: {strategy}", True, strategy_results[strategy])
                else:
                    strategy_results[strategy] = {
                        "success": False,
                        "error": f"Status {response.status_code}: {response.text}"
                    }

                    self.log_test(f"Chunking Strategy: {strategy}", False, {
                        "status_code": response.status_code,
                        "response": response.text
                    })

                time.sleep(0.5)  # Small delay between requests

            except Exception as e:
                strategy_results[strategy] = {
                    "success": False,
                    "error": str(e)
                }
                self.log_test(f"Chunking Strategy: {strategy}", False, {"error": str(e)})

        # Summary of strategy performance
        successful_strategies = [s for s, r in strategy_results.items() if r.get('success')]
        self.log_test("Strategy Testing Summary", len(successful_strategies) >= 3, {
            "successful_strategies": successful_strategies,
            "total_strategies": len(strategies),
            "success_rate": f"{len(successful_strategies)/len(strategies):.1%}"
        })

        return len(successful_strategies) >= 3

    def test_content_types(self) -> bool:
        """Test chunking with different content types"""
        print("\n📝 Testing Different Content Types...")

        content_type_mapping = {
            "python": "python",
            "javascript": "javascript",
            "markdown": "markdown",
            "text": "text"
        }

        successful_types = 0

        for content_name, content_type in content_type_mapping.items():
            try:
                chunk_data = {
                    "content": self.test_contents[content_name],
                    "content_type": content_type,
                    "strategy": "hybrid",
                    "chunk_size": 800,
                    "chunk_overlap": 150
                }

                response = self.session.post(f"{self.base_url}/api/v1/chunking/chunk", json=chunk_data)

                if response.status_code == 200:
                    result = response.json()
                    chunks = result.get('chunks', [])
                    successful_types += 1

                    self.log_test(f"Content Type: {content_type}", True, {
                        "chunk_count": len(chunks),
                        "content_size": len(self.test_contents[content_name]),
                        "total_tokens": result.get('total_tokens', 0)
                    })
                else:
                    self.log_test(f"Content Type: {content_type}", False, {
                        "status_code": response.status_code,
                        "response": response.text
                    })

                time.sleep(0.3)

            except Exception as e:
                self.log_test(f"Content Type: {content_type}", False, {"error": str(e)})

        return successful_types >= 3

    def test_chunk_parameters(self) -> bool:
        """Test different chunk size and overlap parameters"""
        print("\n⚙️ Testing Chunk Parameters...")

        parameter_tests = [
            {"chunk_size": 500, "chunk_overlap": 100, "name": "Small chunks"},
            {"chunk_size": 1000, "chunk_overlap": 200, "name": "Medium chunks"},
            {"chunk_size": 2000, "chunk_overlap": 400, "name": "Large chunks"},
            {"chunk_size": 800, "chunk_overlap": 0, "name": "No overlap"},
            {"chunk_size": 1200, "chunk_overlap": 600, "name": "High overlap"}
        ]

        successful_tests = 0
        content = self.test_contents["python"]

        for test_params in parameter_tests:
            try:
                chunk_data = {
                    "content": content,
                    "content_type": "python",
                    "strategy": "hybrid",
                    "chunk_size": test_params["chunk_size"],
                    "chunk_overlap": test_params["chunk_overlap"]
                }

                response = self.session.post(f"{self.base_url}/api/v1/chunking/chunk", json=chunk_data)

                if response.status_code == 200:
                    result = response.json()
                    chunks = result.get('chunks', [])
                    successful_tests += 1

                    self.log_test(f"Parameters: {test_params['name']}", True, {
                        "chunk_size": test_params["chunk_size"],
                        "chunk_overlap": test_params["chunk_overlap"],
                        "result_chunks": len(chunks),
                        "total_tokens": result.get('total_tokens', 0)
                    })
                else:
                    self.log_test(f"Parameters: {test_params['name']}", False, {
                        "status_code": response.status_code,
                        "response": response.text
                    })

                time.sleep(0.2)

            except Exception as e:
                self.log_test(f"Parameters: {test_params['name']}", False, {"error": str(e)})

        return successful_tests >= 3

    def test_quality_assessment(self) -> bool:
        """Test quality assessment features"""
        print("\n🎯 Testing Quality Assessment...")

        try:
            # Test with good quality content (well-structured code)
            good_content = self.test_contents["python"]

            chunk_data = {
                "content": good_content,
                "content_type": "python",
                "strategy": "agentic",
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "assess_quality": True
            }

            response = self.session.post(f"{self.base_url}/api/v1/chunking/chunk", json=chunk_data)

            if response.status_code == 200:
                result = response.json()
                quality_score = result.get('quality_score', 0)
                quality_details = result.get('quality_details', {})

                self.log_test("Quality Assessment - Good Content", True, {
                    "quality_score": quality_score,
                    "quality_details": quality_details,
                    "chunk_count": len(result.get('chunks', []))
                })

                # Test with poor quality content (random text)
                poor_content = "random text without structure or meaning just words thrown together"

                chunk_data["content"] = poor_content
                chunk_data["content_type"] = "text"

                response = self.session.post(f"{self.base_url}/api/v1/chunking/chunk", json=chunk_data)

                if response.status_code == 200:
                    result = response.json()
                    poor_quality_score = result.get('quality_score', 0)

                    self.log_test("Quality Assessment - Poor Content", True, {
                        "quality_score": poor_quality_score,
                        "comparison": f"Good: {quality_score:.3f} vs Poor: {poor_quality_score:.3f}"
                    })

                    # Quality scores should be different
                    quality_difference = abs(quality_score - poor_quality_score)
                    self.log_test("Quality Score Differentiation", quality_difference > 0.1, {
                        "difference": quality_difference,
                        "good_score": quality_score,
                        "poor_score": poor_quality_score
                    })

                    return True
                else:
                    self.log_test("Quality Assessment - Poor Content", False, {
                        "status_code": response.status_code,
                        "response": response.text
                    })
                    return False
            else:
                self.log_test("Quality Assessment - Good Content", False, {
                    "status_code": response.status_code,
                    "response": response.text
                })
                return False

        except Exception as e:
            self.log_test("Quality Assessment Tests", False, {"error": str(e)})
            return False

    def test_performance_metrics(self) -> bool:
        """Test performance and timing metrics"""
        print("\n⏱️ Testing Performance Metrics...")

        try:
            # Test with different content sizes
            content_sizes = [
                ("Small", self.test_contents["text"][:500]),
                ("Medium", self.test_contents["python"]),
                ("Large", self.test_contents["markdown"] + self.test_contents["python"] + self.test_contents["javascript"])
            ]

            performance_results = {}

            for size_name, content in content_sizes:
                start_time = time.time()

                chunk_data = {
                    "content": content,
                    "content_type": "text",
                    "strategy": "hybrid",
                    "chunk_size": 1000,
                    "chunk_overlap": 200
                }

                response = self.session.post(f"{self.base_url}/api/v1/chunking/chunk", json=chunk_data)

                end_time = time.time()
                request_time = end_time - start_time

                if response.status_code == 200:
                    result = response.json()
                    processing_time = result.get('processing_time', 0)

                    performance_results[size_name] = {
                        "content_size": len(content),
                        "request_time": request_time,
                        "processing_time": processing_time,
                        "chunk_count": len(result.get('chunks', [])),
                        "tokens": result.get('total_tokens', 0)
                    }

                    self.log_test(f"Performance: {size_name} Content", True, performance_results[size_name])
                else:
                    self.log_test(f"Performance: {size_name} Content", False, {
                        "status_code": response.status_code,
                        "response": response.text
                    })

                time.sleep(0.5)

            # Performance summary
            if len(performance_results) >= 2:
                self.log_test("Performance Testing Summary", True, {
                    "tested_sizes": list(performance_results.keys()),
                    "avg_processing_time": sum(r["processing_time"] for r in performance_results.values()) / len(performance_results),
                    "total_content_processed": sum(r["content_size"] for r in performance_results.values())
                })
                return True
            else:
                return False

        except Exception as e:
            self.log_test("Performance Metrics Tests", False, {"error": str(e)})
            return False

    def run_comprehensive_chunking_tests(self) -> Dict[str, Any]:
        """Run all comprehensive chunking tests"""
        print("🔪 Starting Comprehensive Chunking API Tests")
        print("=" * 50)

        # Test chunking service health
        if not self.test_chunking_health():
            print("❌ Chunking service health check failed. Stopping tests.")
            return self.get_results()

        # Test supported languages
        self.test_supported_languages()

        # Test chunking strategies
        self.test_chunking_strategies()

        # Test content types
        self.test_content_types()

        # Test chunk parameters
        self.test_chunk_parameters()

        # Test quality assessment
        self.test_quality_assessment()

        # Test performance metrics
        self.test_performance_metrics()

        return self.get_results()

    def get_results(self) -> Dict[str, Any]:
        """Get test results"""
        self.results["end_time"] = datetime.now().isoformat()
        self.results["duration"] = str(datetime.fromisoformat(self.results["end_time"]) -
                                     datetime.fromisoformat(self.results["start_time"]))
        return self.results

    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 50)
        print("📊 CHUNKING API TEST SUMMARY")
        print("=" * 50)

        summary = self.results["summary"]
        print(f"✅ Total Tests: {summary['total']}")
        print(f"✅ Passed: {summary['passed']}")
        print(f"❌ Failed: {summary['failed']}")
        print(f"📈 Success Rate: {(summary['passed'] / summary['total'] * 100):.1f}%" if summary['total'] > 0 else "0%")

        if summary['errors']:
            print(f"\n❌ Errors ({len(summary['errors'])}):")
            for error in summary['errors'][:3]:  # Show first 3 errors
                print(f"   - {error}")
            if len(summary['errors']) > 3:
                print(f"   ... and {len(summary['errors']) - 3} more errors")

        print(f"\n⏱️ Duration: {self.results.get('duration', 'Unknown')}")
        print("=" * 50)


def main():
    """Main function to run chunking tests"""
    import argparse

    parser = argparse.ArgumentParser(description='Detailed Chunking API Tests for Cognify')
    parser.add_argument('--url', default='http://localhost:8001', help='Base URL for API')
    parser.add_argument('--output', help='Output file for test results (JSON)')

    args = parser.parse_args()

    # Create tester
    tester = ChunkingAPITester(args.url)

    # Run tests
    results = tester.run_comprehensive_chunking_tests()

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
