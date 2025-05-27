#!/usr/bin/env python3
"""
Master Test Runner for Cognify
Runs all test suites and provides comprehensive reporting
"""

import subprocess
import sys
import time
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

class CognifyTestRunner:
    """Master test runner for all Cognify test suites"""
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.results = {
            "start_time": datetime.now().isoformat(),
            "test_suites": [],
            "summary": {
                "total_suites": 0,
                "passed_suites": 0,
                "failed_suites": 0,
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0
            }
        }
        
        # Test suites to run
        self.test_suites = [
            {
                "name": "Comprehensive API Tests",
                "script": "scripts/test_runners/test_all_apis_comprehensive.py",
                "description": "Tests all API endpoints with real scenarios",
                "timeout": 300  # 5 minutes
            },
            {
                "name": "Detailed Chunking Tests", 
                "script": "scripts/test_runners/test_chunking_api_detailed.py",
                "description": "Tests chunking API with various strategies and content types",
                "timeout": 180  # 3 minutes
            },
            {
                "name": "Manual RAG Search Tests",
                "script": "tests/manual/test_rag_search.py", 
                "description": "Tests RAG search functionality end-to-end",
                "timeout": 120  # 2 minutes
            },
            {
                "name": "Database Health Check",
                "script": "tests/manual/check_database.py",
                "description": "Checks database connectivity and system status",
                "timeout": 60   # 1 minute
            }
        ]
    
    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are met"""
        print("🔍 Checking Prerequisites...")
        
        # Check if API is running
        try:
            import requests
            response = requests.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                print("✅ API is running and accessible")
                return True
            else:
                print(f"❌ API returned status {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Cannot connect to API: {e}")
            print(f"   Make sure the API is running at {self.base_url}")
            return False
    
    def run_test_suite(self, suite: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single test suite"""
        print(f"\n🧪 Running: {suite['name']}")
        print(f"   Description: {suite['description']}")
        print(f"   Script: {suite['script']}")
        
        start_time = time.time()
        
        try:
            # Prepare command
            script_path = Path(suite['script'])
            if not script_path.exists():
                return {
                    "name": suite['name'],
                    "success": False,
                    "error": f"Script not found: {suite['script']}",
                    "duration": 0,
                    "output": ""
                }
            
            # Run the test script
            cmd = [sys.executable, str(script_path), "--url", self.base_url]
            
            # Add output file for JSON results
            output_file = f"test_results_{suite['name'].lower().replace(' ', '_')}_{int(time.time())}.json"
            if suite['script'].endswith('.py') and 'test_runners' in suite['script']:
                cmd.extend(["--output", output_file])
            
            print(f"   Command: {' '.join(cmd)}")
            
            # Execute with timeout
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=suite['timeout'],
                cwd=Path(__file__).parent.parent  # Run from project root
            )
            
            duration = time.time() - start_time
            
            # Parse results if JSON output file exists
            test_details = {}
            if os.path.exists(output_file):
                try:
                    with open(output_file, 'r') as f:
                        test_details = json.load(f)
                    os.remove(output_file)  # Clean up
                except:
                    pass
            
            suite_result = {
                "name": suite['name'],
                "success": result.returncode == 0,
                "duration": duration,
                "return_code": result.returncode,
                "output": result.stdout,
                "error": result.stderr if result.stderr else None,
                "test_details": test_details
            }
            
            if result.returncode == 0:
                print(f"   ✅ PASSED ({duration:.1f}s)")
            else:
                print(f"   ❌ FAILED ({duration:.1f}s)")
                if result.stderr:
                    print(f"   Error: {result.stderr[:200]}...")
            
            return suite_result
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            print(f"   ⏰ TIMEOUT ({duration:.1f}s)")
            return {
                "name": suite['name'],
                "success": False,
                "error": f"Test timed out after {suite['timeout']} seconds",
                "duration": duration,
                "output": ""
            }
            
        except Exception as e:
            duration = time.time() - start_time
            print(f"   ❌ ERROR ({duration:.1f}s): {e}")
            return {
                "name": suite['name'],
                "success": False,
                "error": str(e),
                "duration": duration,
                "output": ""
            }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all test suites"""
        print("🚀 Starting Comprehensive Test Suite for Cognify")
        print("=" * 60)
        
        # Check prerequisites
        if not self.check_prerequisites():
            print("\n❌ Prerequisites not met. Please start the API first.")
            print("   Run: ./scripts/start-development.sh")
            return self.get_results()
        
        # Run each test suite
        for suite in self.test_suites:
            suite_result = self.run_test_suite(suite)
            self.results["test_suites"].append(suite_result)
            
            # Update summary
            self.results["summary"]["total_suites"] += 1
            if suite_result["success"]:
                self.results["summary"]["passed_suites"] += 1
            else:
                self.results["summary"]["failed_suites"] += 1
            
            # Add test details to summary if available
            if "test_details" in suite_result and suite_result["test_details"]:
                details = suite_result["test_details"]
                if "summary" in details:
                    self.results["summary"]["total_tests"] += details["summary"].get("total", 0)
                    self.results["summary"]["passed_tests"] += details["summary"].get("passed", 0)
                    self.results["summary"]["failed_tests"] += details["summary"].get("failed", 0)
            
            # Small delay between test suites
            time.sleep(2)
        
        return self.get_results()
    
    def get_results(self) -> Dict[str, Any]:
        """Get final results"""
        self.results["end_time"] = datetime.now().isoformat()
        self.results["duration"] = str(datetime.fromisoformat(self.results["end_time"]) - 
                                     datetime.fromisoformat(self.results["start_time"]))
        return self.results
    
    def print_summary(self):
        """Print comprehensive test summary"""
        print("\n" + "=" * 60)
        print("📊 COMPREHENSIVE TEST SUMMARY")
        print("=" * 60)
        
        summary = self.results["summary"]
        
        # Suite-level summary
        print(f"🧪 Test Suites:")
        print(f"   Total: {summary['total_suites']}")
        print(f"   ✅ Passed: {summary['passed_suites']}")
        print(f"   ❌ Failed: {summary['failed_suites']}")
        suite_success_rate = (summary['passed_suites'] / summary['total_suites'] * 100) if summary['total_suites'] > 0 else 0
        print(f"   📈 Success Rate: {suite_success_rate:.1f}%")
        
        # Individual test summary (if available)
        if summary['total_tests'] > 0:
            print(f"\n🔬 Individual Tests:")
            print(f"   Total: {summary['total_tests']}")
            print(f"   ✅ Passed: {summary['passed_tests']}")
            print(f"   ❌ Failed: {summary['failed_tests']}")
            test_success_rate = (summary['passed_tests'] / summary['total_tests'] * 100)
            print(f"   📈 Success Rate: {test_success_rate:.1f}%")
        
        # Suite details
        print(f"\n📋 Suite Details:")
        for suite in self.results["test_suites"]:
            status = "✅ PASS" if suite["success"] else "❌ FAIL"
            duration = suite.get("duration", 0)
            print(f"   {status} {suite['name']} ({duration:.1f}s)")
            
            if not suite["success"] and suite.get("error"):
                print(f"      Error: {suite['error']}")
        
        # Overall result
        print(f"\n⏱️ Total Duration: {self.results.get('duration', 'Unknown')}")
        
        overall_success = summary['failed_suites'] == 0
        print(f"\n🎯 Overall Result: {'✅ PASS' if overall_success else '❌ FAIL'}")
        
        if not overall_success:
            print("\n💡 Troubleshooting Tips:")
            print("   1. Check if all services are running: docker-compose ps")
            print("   2. Check API logs: docker-compose logs cognify-api")
            print("   3. Verify database connection: docker-compose logs postgres")
            print("   4. Check individual test outputs above for specific errors")
        
        print("=" * 60)


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run all Cognify test suites')
    parser.add_argument('--url', default='http://localhost:8001', help='Base URL for API')
    parser.add_argument('--output', help='Output file for complete results (JSON)')
    
    args = parser.parse_args()
    
    # Create test runner
    runner = CognifyTestRunner(args.url)
    
    # Run all tests
    results = runner.run_all_tests()
    
    # Print summary
    runner.print_summary()
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Complete results saved to: {args.output}")
    
    # Exit with appropriate code
    exit_code = 0 if results["summary"]["failed_suites"] == 0 else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
