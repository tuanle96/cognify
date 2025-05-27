#!/usr/bin/env python3
"""
Secret Scanner for Cognify
Scans codebase for hardcoded secrets and sensitive information.
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple


class SecretPattern:
    """Pattern for detecting secrets."""

    def __init__(self, name: str, pattern: str, description: str, severity: str = "HIGH"):
        self.name = name
        self.pattern = re.compile(pattern, re.IGNORECASE)
        self.description = description
        self.severity = severity


class SecretScanner:
    """Scanner for detecting hardcoded secrets."""

    def __init__(self):
        self.patterns = [
            # API Keys
            SecretPattern(
                "OpenAI API Key",
                r'sk-[a-zA-Z0-9]{48}',
                "OpenAI API key detected",
                "CRITICAL"
            ),
            SecretPattern(
                "GitHub Token",
                r'ghp_[a-zA-Z0-9]{36}',
                "GitHub personal access token detected",
                "HIGH"
            ),
            SecretPattern(
                "AWS Access Key",
                r'AKIA[0-9A-Z]{16}',
                "AWS access key detected",
                "CRITICAL"
            ),
            SecretPattern(
                "Slack Token",
                r'xoxb-[0-9]{11}-[0-9]{11}-[a-zA-Z0-9]{24}',
                "Slack bot token detected",
                "HIGH"
            ),

            # Generic patterns
            SecretPattern(
                "Hardcoded Password",
                r'password\s*=\s*["\'][^"\']{8,}["\']',
                "Hardcoded password detected",
                "HIGH"
            ),
            SecretPattern(
                "Hardcoded Secret",
                r'secret\s*=\s*["\'][^"\']{8,}["\']',
                "Hardcoded secret detected",
                "HIGH"
            ),
            SecretPattern(
                "Hardcoded Token",
                r'token\s*=\s*["\'][^"\']{8,}["\']',
                "Hardcoded token detected",
                "HIGH"
            ),
            SecretPattern(
                "API Key Assignment",
                r'api_key\s*=\s*["\'][^"\']{8,}["\']',
                "Hardcoded API key detected",
                "HIGH"
            ),

            # Database credentials
            SecretPattern(
                "Database URL with Password",
                r'postgresql://[^:]+:[^@]+@[^/]+',
                "Database URL with embedded password",
                "MEDIUM"
            ),
            SecretPattern(
                "MySQL URL with Password",
                r'mysql://[^:]+:[^@]+@[^/]+',
                "MySQL URL with embedded password",
                "MEDIUM"
            ),

            # Environment variable assignments
            SecretPattern(
                "Environment Secret Assignment",
                r'os\.environ\[["\'][^"\']*(?:KEY|SECRET|PASSWORD|TOKEN)[^"\']*["\']\]\s*=\s*["\'][^"\']{8,}["\']',
                "Environment variable with hardcoded secret",
                "HIGH"
            ),
        ]

        # Files to exclude from scanning
        self.exclude_patterns = [
            r'\.git/',
            r'__pycache__/',
            r'\.pyc$',
            r'\.env$',
            r'\.env\.development$',
            r'\.env\.production$',
            r'\.env\.example$',
            r'node_modules/',
            r'\.venv/',
            r'venv/',
            r'\.pytest_cache/',
            r'htmlcov/',
            r'\.coverage',
            r'coverage\.xml',
            r'\.log$',
            r'logs/',
        ]

        # Whitelist patterns (known safe values)
        self.whitelist_patterns = [
            r'your-.*-key.*here',
            r'your-.*-token.*here',
            r'test-key-not-real',
            r'test-api-key-for-testing',
            r'test-.*-key',
            r'mock.*key',
            r'fake.*key',
            r'example.*key',
            r'placeholder.*key',
            r'dev-.*-for-testing-only',
            r'test-secret-key-for-.*',
            r'test-jwt-secret-key-for-.*',
            r'hashed_password.*',
            r'.*_token.*',
            r'session_token.*',
            r'refresh_token.*',
            r'expired_token',
            r'valid_token',
            r'invalid_token',
            r'complete_session_token',
            r'test_token',
            r'invalid-key',
            r'test-litellm-key',
            r'PGPASSWORD=.*DB_PASSWORD',
            r'PASSWORD=.*DB_PASSWORD',
        ]

    def should_exclude_file(self, file_path: str) -> bool:
        """Check if file should be excluded from scanning."""
        for pattern in self.exclude_patterns:
            if re.search(pattern, file_path):
                return True
        return False

    def is_whitelisted(self, match: str) -> bool:
        """Check if match is in whitelist (known safe values)."""
        for pattern in self.whitelist_patterns:
            if re.search(pattern, match, re.IGNORECASE):
                return True
        return False

    def scan_file(self, file_path: Path) -> List[Dict]:
        """Scan a single file for secrets."""
        findings = []

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')

                for pattern in self.patterns:
                    for match in pattern.pattern.finditer(content):
                        matched_text = match.group()

                        # Skip whitelisted values
                        if self.is_whitelisted(matched_text):
                            continue

                        # Find line number
                        line_num = content[:match.start()].count('\n') + 1
                        line_content = lines[line_num - 1] if line_num <= len(lines) else ""

                        findings.append({
                            'file': str(file_path),
                            'line': line_num,
                            'pattern': pattern.name,
                            'description': pattern.description,
                            'severity': pattern.severity,
                            'match': matched_text,
                            'line_content': line_content.strip(),
                            'start': match.start(),
                            'end': match.end()
                        })

        except Exception as e:
            print(f"Error scanning {file_path}: {e}")

        return findings

    def scan_directory(self, directory: Path) -> List[Dict]:
        """Scan directory recursively for secrets."""
        all_findings = []

        for file_path in directory.rglob('*'):
            if file_path.is_file() and not self.should_exclude_file(str(file_path)):
                findings = self.scan_file(file_path)
                all_findings.extend(findings)

        return all_findings

    def print_report(self, findings: List[Dict]) -> None:
        """Print scan report."""
        if not findings:
            print("✅ No secrets detected!")
            return

        # Group by severity
        by_severity = {}
        for finding in findings:
            severity = finding['severity']
            if severity not in by_severity:
                by_severity[severity] = []
            by_severity[severity].append(finding)

        # Print summary
        print(f"\n🚨 SECURITY SCAN RESULTS")
        print("=" * 50)
        print(f"Total findings: {len(findings)}")

        for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
            if severity in by_severity:
                count = len(by_severity[severity])
                print(f"{severity}: {count}")

        # Print detailed findings
        for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
            if severity not in by_severity:
                continue

            print(f"\n{severity} SEVERITY FINDINGS:")
            print("-" * 30)

            for finding in by_severity[severity]:
                print(f"\n📁 File: {finding['file']}")
                print(f"📍 Line: {finding['line']}")
                print(f"🔍 Pattern: {finding['pattern']}")
                print(f"📝 Description: {finding['description']}")
                print(f"💡 Match: {finding['match']}")
                print(f"📄 Context: {finding['line_content']}")


def main():
    """Main scanner function."""
    print("🔍 Cognify Secret Scanner")
    print("=" * 50)

    # Get project root
    project_root = Path(__file__).parent.parent

    # Initialize scanner
    scanner = SecretScanner()

    # Scan project
    print(f"Scanning: {project_root}")
    findings = scanner.scan_directory(project_root)

    # Print report
    scanner.print_report(findings)

    # Exit with error code if critical/high findings
    critical_high = [f for f in findings if f['severity'] in ['CRITICAL', 'HIGH']]
    if critical_high:
        print(f"\n❌ Found {len(critical_high)} critical/high severity issues!")
        print("Please fix these before committing to version control.")
        sys.exit(1)
    else:
        print(f"\n✅ No critical security issues found!")
        sys.exit(0)


if __name__ == "__main__":
    main()
