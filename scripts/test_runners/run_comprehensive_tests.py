#!/usr/bin/env python3
"""
Comprehensive Test Runner for Cognify Coverage Improvement.

This script runs all test suites and generates detailed coverage reports
to track progress towards coverage targets:
- API Integration Tests: 0% → 60%+
- Database Tests: 0% → 70%+
- Core Service Tests: 0% → 50%+
"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Test configuration
TEST_CATEGORIES = {
    "api_integration": {
        "path": "tests/integration/test_*_api_comprehensive.py",
        "target_coverage": 60,
        "description": "API Integration Tests"
    },
    "database_models": {
        "path": "tests/unit/test_models/test_*_models_comprehensive.py",
        "target_coverage": 70,
        "description": "Database Model Tests"
    },
    "core_services": {
        "path": "tests/unit/test_services/test_*_service_comprehensive.py",
        "target_coverage": 50,
        "description": "Core Service Tests"
    },
    "existing_unit": {
        "path": "tests/unit/test_core/ tests/unit/test_services/test_chunking_basic.py tests/unit/test_services/test_litellm_service.py",
        "target_coverage": 80,
        "description": "Existing Unit Tests"
    }
}

COVERAGE_MODULES = {
    "api_integration": "app.api",
    "database_models": "app.models",
    "core_services": "app.services",
    "existing_unit": "app.core"
}


class TestRunner:
    """Comprehensive test runner with coverage tracking."""

    def __init__(self):
        self.project_root = project_root
        self.results = {}
        self.start_time = None
        self.end_time = None

    def log(self, message: str, level: str = "INFO"):
        """Log message with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")

    def run_command(self, command: str, capture_output: bool = True) -> Tuple[int, str, str]:
        """Run shell command and return result."""
        self.log(f"Running: {command}")

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.project_root,
                capture_output=capture_output,
                text=True,
                timeout=300  # 5 minute timeout
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            self.log("Command timed out", "ERROR")
            return 1, "", "Command timed out"
        except Exception as e:
            self.log(f"Command failed: {e}", "ERROR")
            return 1, "", str(e)

    def check_dependencies(self) -> bool:
        """Check if required dependencies are installed."""
        self.log("Checking dependencies...")

        required_packages = ["pytest", "pytest-cov", "pytest-asyncio"]
        missing_packages = []

        for package in required_packages:
            returncode, _, _ = self.run_command(f"python -c 'import {package.replace('-', '_')}'")
            if returncode != 0:
                missing_packages.append(package)

        if missing_packages:
            self.log(f"Missing packages: {missing_packages}", "ERROR")
            self.log("Install with: pip install " + " ".join(missing_packages))
            return False

        self.log("All dependencies available")
        return True

    def run_test_category(self, category: str, config: Dict) -> Dict:
        """Run tests for a specific category."""
        self.log(f"Running {config['description']}...")

        # Prepare pytest command with test environment
        test_path = config["path"]
        coverage_module = COVERAGE_MODULES.get(category, "app")

        # Set test environment variables
        env_vars = (
            "ENVIRONMENT=testing "
            "DATABASE_URL=sqlite+aiosqlite:///:memory: "
            "OPENAI_API_KEY=test-key-not-real "
            "SECRET_KEY=test-secret-key "
        )

        pytest_cmd = (
            f"{env_vars}python -m pytest {test_path} "
            f"--cov={coverage_module} "
            f"--cov-report=term-missing "
            f"--cov-report=html:htmlcov_{category} "
            f"--cov-report=json:coverage_{category}.json "
            f"-v --tb=short --disable-warnings"
        )

        # Run tests
        start_time = time.time()
        returncode, stdout, stderr = self.run_command(pytest_cmd)
        end_time = time.time()

        # Parse results
        result = {
            "category": category,
            "description": config["description"],
            "target_coverage": config["target_coverage"],
            "returncode": returncode,
            "duration": end_time - start_time,
            "stdout": stdout,
            "stderr": stderr
        }

        # Extract coverage information
        try:
            coverage_file = self.project_root / f"coverage_{category}.json"
            if coverage_file.exists():
                with open(coverage_file, 'r') as f:
                    coverage_data = json.load(f)
                    result["coverage_percent"] = coverage_data.get("totals", {}).get("percent_covered", 0)
                    result["statements_total"] = coverage_data.get("totals", {}).get("num_statements", 0)
                    result["statements_covered"] = coverage_data.get("totals", {}).get("covered_lines", 0)
        except Exception as e:
            self.log(f"Could not parse coverage for {category}: {e}", "WARNING")
            result["coverage_percent"] = 0

        # Extract test results from stdout
        if "failed" in stdout.lower() or "error" in stdout.lower():
            result["status"] = "FAILED"
        elif returncode == 0:
            result["status"] = "PASSED"
        else:
            result["status"] = "ERROR"

        # Extract test counts
        try:
            lines = stdout.split('\n')
            for line in lines:
                if "passed" in line and ("failed" in line or "error" in line):
                    # Parse line like "5 failed, 10 passed, 2 warnings in 30.45s"
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "passed":
                            result["tests_passed"] = int(parts[i-1])
                        elif part == "failed":
                            result["tests_failed"] = int(parts[i-1])
                        elif part == "error" or part == "errors":
                            result["tests_error"] = int(parts[i-1])
                    break
        except:
            pass

        return result

    def run_all_tests(self) -> Dict:
        """Run all test categories."""
        self.log("🚀 Starting Comprehensive Test Suite")
        self.log("=" * 60)

        self.start_time = time.time()

        # Check dependencies
        if not self.check_dependencies():
            return {"error": "Missing dependencies"}

        # Run each test category
        for category, config in TEST_CATEGORIES.items():
            self.log(f"\n📊 Category: {config['description']}")
            self.log(f"Target Coverage: {config['target_coverage']}%")

            result = self.run_test_category(category, config)
            self.results[category] = result

            # Log immediate results
            status_emoji = "✅" if result["status"] == "PASSED" else "❌"
            coverage = result.get("coverage_percent", 0)
            target = result["target_coverage"]

            self.log(f"{status_emoji} {result['description']}: {result['status']}")
            self.log(f"   Coverage: {coverage:.1f}% (Target: {target}%)")
            self.log(f"   Duration: {result['duration']:.1f}s")

            if coverage >= target:
                self.log(f"   🎯 TARGET ACHIEVED! ({coverage:.1f}% >= {target}%)")
            else:
                gap = target - coverage
                self.log(f"   📈 Gap to target: {gap:.1f}%")

        self.end_time = time.time()

        # Generate summary
        self.generate_summary()

        return self.results

    def generate_summary(self):
        """Generate comprehensive test summary."""
        self.log("\n" + "=" * 60)
        self.log("📊 COMPREHENSIVE TEST RESULTS SUMMARY")
        self.log("=" * 60)

        total_duration = self.end_time - self.start_time
        self.log(f"⏱️  Total Duration: {total_duration:.1f} seconds")

        # Overall statistics
        total_categories = len(self.results)
        passed_categories = sum(1 for r in self.results.values() if r["status"] == "PASSED")
        failed_categories = total_categories - passed_categories

        self.log(f"📈 Categories: {passed_categories}/{total_categories} passed")

        # Coverage summary
        self.log("\n🎯 COVERAGE TARGETS PROGRESS:")
        targets_met = 0

        for category, result in self.results.items():
            coverage = result.get("coverage_percent", 0)
            target = result["target_coverage"]
            status = "✅ MET" if coverage >= target else "❌ MISSED"
            gap = target - coverage

            self.log(f"   {result['description']}:")
            self.log(f"      Current: {coverage:.1f}% | Target: {target}% | {status}")

            if coverage >= target:
                targets_met += 1
                self.log(f"      🎉 Exceeded by {coverage - target:.1f}%")
            else:
                self.log(f"      📊 Need {gap:.1f}% more coverage")

        # Overall coverage calculation
        total_statements = sum(r.get("statements_total", 0) for r in self.results.values())
        total_covered = sum(r.get("statements_covered", 0) for r in self.results.values())
        overall_coverage = (total_covered / total_statements * 100) if total_statements > 0 else 0

        self.log(f"\n📊 OVERALL PROJECT COVERAGE: {overall_coverage:.1f}%")
        self.log(f"   Total Statements: {total_statements:,}")
        self.log(f"   Covered Statements: {total_covered:,}")
        self.log(f"   Uncovered Statements: {total_statements - total_covered:,}")

        # Success metrics
        success_rate = (targets_met / total_categories * 100) if total_categories > 0 else 0

        self.log(f"\n🏆 SUCCESS METRICS:")
        self.log(f"   Coverage Targets Met: {targets_met}/{total_categories} ({success_rate:.1f}%)")
        self.log(f"   Test Categories Passed: {passed_categories}/{total_categories}")

        # Recommendations
        self.log(f"\n💡 RECOMMENDATIONS:")

        if success_rate >= 75:
            self.log("   🎉 Excellent progress! Most targets achieved.")
        elif success_rate >= 50:
            self.log("   👍 Good progress! Focus on remaining gaps.")
        else:
            self.log("   🔧 Needs improvement. Review failed tests and add more coverage.")

        # Next steps
        failed_categories = [cat for cat, result in self.results.items()
                           if result.get("coverage_percent", 0) < result["target_coverage"]]

        if failed_categories:
            self.log(f"\n📋 PRIORITY ACTIONS:")
            for category in failed_categories:
                result = self.results[category]
                gap = result["target_coverage"] - result.get("coverage_percent", 0)
                self.log(f"   • {result['description']}: Add {gap:.1f}% more coverage")

        # File locations
        self.log(f"\n📁 DETAILED REPORTS:")
        for category in self.results.keys():
            self.log(f"   • {category}: htmlcov_{category}/index.html")

        self.log("\n✨ Test run completed!")

    def save_results(self, filename: str = "test_results.json"):
        """Save results to JSON file."""
        results_file = self.project_root / filename

        summary_data = {
            "timestamp": datetime.now().isoformat(),
            "duration": self.end_time - self.start_time if self.end_time and self.start_time else 0,
            "results": self.results,
            "summary": {
                "total_categories": len(self.results),
                "passed_categories": sum(1 for r in self.results.values() if r["status"] == "PASSED"),
                "targets_met": sum(1 for r in self.results.values()
                                 if r.get("coverage_percent", 0) >= r["target_coverage"]),
                "overall_coverage": sum(r.get("coverage_percent", 0) for r in self.results.values()) / len(self.results) if self.results else 0
            }
        }

        with open(results_file, 'w') as f:
            json.dump(summary_data, f, indent=2)

        self.log(f"Results saved to: {results_file}")


def main():
    """Main function to run comprehensive tests."""
    runner = TestRunner()

    try:
        results = runner.run_all_tests()
        runner.save_results()

        # Exit with appropriate code
        failed_categories = sum(1 for r in results.values() if r.get("status") != "PASSED")
        sys.exit(failed_categories)

    except KeyboardInterrupt:
        runner.log("Test run interrupted by user", "WARNING")
        sys.exit(1)
    except Exception as e:
        runner.log(f"Test run failed: {e}", "ERROR")
        sys.exit(1)


if __name__ == "__main__":
    main()
