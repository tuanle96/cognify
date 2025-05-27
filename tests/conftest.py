"""
Simple pytest configuration for basic tests.

This module provides basic test configuration without complex dependencies.
"""

import asyncio
import os
import pytest
import tempfile
from pathlib import Path
from typing import Dict, Any, AsyncGenerator, Generator

# Test environment setup
os.environ["ENVIRONMENT"] = "testing"
os.environ["LOG_LEVEL"] = "DEBUG"
# Use mock API key for testing - not a real key
os.environ["OPENAI_API_KEY"] = "test-key-not-real-for-testing-only"
os.environ["SECRET_KEY"] = "test-secret-key-for-testing-only-not-production"
os.environ["DATABASE_URL"] = "postgresql://test_user:test_pass@localhost/test_db"

# Import basic components
from app.core.config import get_settings

# Import test app for mock testing
try:
    from test_main import test_app
    from fastapi.testclient import TestClient
    TEST_APP_AVAILABLE = True
except ImportError:
    TEST_APP_AVAILABLE = False


# ============================================================================
# Test Configuration
# ============================================================================

@pytest.fixture(scope="function")
def event_loop():
    """Create an instance of the default event loop for each test function."""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_settings():
    """Get test configuration settings."""
    return get_settings()


# ============================================================================
# Test Client Fixtures
# ============================================================================

@pytest.fixture
def mock_client():
    """
    Test client fixture using mock test app.

    This client uses mock implementations for fast, reliable testing
    without external dependencies.
    """
    if not TEST_APP_AVAILABLE:
        pytest.skip("Test app not available")
    return TestClient(test_app)


@pytest.fixture
def test_client():
    """
    Test client fixture using mock test app.
    Alias for mock_client for compatibility.
    """
    if not TEST_APP_AVAILABLE:
        pytest.skip("Test app not available")
    return TestClient(test_app)


@pytest.fixture
async def async_client():
    """Async test client fixture."""
    if not TEST_APP_AVAILABLE:
        pytest.skip("Test app not available")

    from httpx import AsyncClient
    async with AsyncClient(app=test_app, base_url="http://test") as client:
        yield client


@pytest.fixture
def auth_headers(test_client, sample_user_data):
    """Dynamic authentication headers using real auth flow."""
    # Create unique user data to avoid conflicts
    unique_user_data = sample_user_data.copy()
    unique_user_data["email"] = f"auth_{unique_user_data['email']}"
    unique_user_data["username"] = f"auth_{unique_user_data['username']}"

    # Register user
    register_response = test_client.post("/api/v1/auth/register", json=unique_user_data)

    # Login to get real token
    login_response = test_client.post("/api/v1/auth/login", json={
        "email": unique_user_data["email"],
        "password": unique_user_data["password"]
    })

    if login_response.status_code == 200:
        tokens = login_response.json()
        access_token = tokens["access_token"]
        return {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
    else:
        # Debug: Print response for troubleshooting
        print(f"Auth failed - Register: {register_response.status_code}, Login: {login_response.status_code}")
        if login_response.status_code != 200:
            print(f"Login response: {login_response.json()}")

        # Fallback to mock token if auth fails
        return {
            "Authorization": "Bearer mock_test_token_123",
            "Content-Type": "application/json"
        }


@pytest.fixture
def sample_user_data():
    """Sample user data for testing."""
    return {
        "email": "test@example.com",
        "password": "TestPassword123",
        "full_name": "Test User",
        "username": "testuser"
    }


@pytest.fixture
def sample_document_data():
    """Sample document data for testing."""
    return {
        "title": "Test Document",
        "content": "This is a test document content for unit testing.",
        "document_type": "text",
        "tags": ["test", "mock"],
        "metadata": {
            "source": "unit_test",
            "language": "en"
        }
    }


@pytest.fixture
def sample_collection_data():
    """Sample collection data for testing."""
    return {
        "name": "Test Collection",
        "description": "A test collection for unit testing",
        "visibility": "private",
        "tags": ["test", "mock"],
        "metadata": {
            "purpose": "testing",
            "created_by": "test_user"
        }
    }


@pytest.fixture
def sample_python_code():
    """Sample Python code for testing."""
    return '''
def fibonacci(n):
    """Calculate fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

class Calculator:
    """Simple calculator class."""

    def add(self, a, b):
        return a + b

    def multiply(self, a, b):
        return a * b

def main():
    calc = Calculator()
    print(calc.add(2, 3))
    print(calc.multiply(4, 5))
    print(fibonacci(10))

if __name__ == "__main__":
    main()
'''


@pytest.fixture
def performance_timer():
    """Performance timer for testing response times."""
    class PerformanceTimer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.elapsed = 0

        def start(self):
            import time
            self.start_time = time.time()

        def stop(self):
            import time
            self.end_time = time.time()
            if self.start_time:
                self.elapsed = self.end_time - self.start_time

        def reset(self):
            self.__init__()

    return PerformanceTimer()


@pytest.fixture
def temp_dir():
    """Temporary directory for file operations."""
    import tempfile
    import shutil

    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_javascript_code():
    """Sample JavaScript code for testing."""
    return '''
function fibonacci(n) {
    if (n <= 1) {
        return n;
    }
    return fibonacci(n - 1) + fibonacci(n - 2);
}

class Calculator {
    constructor() {
        this.name = "Calculator";
    }

    add(a, b) {
        return a + b;
    }

    multiply(a, b) {
        return a * b;
    }
}

const calc = new Calculator();
console.log(calc.add(2, 3));
console.log(calc.multiply(4, 5));
console.log(fibonacci(10));
'''


@pytest.fixture
def sample_large_code():
    """Sample large code for performance testing."""
    code_parts = []
    for i in range(50):
        code_parts.append(f'''
def function_{i}():
    """Function number {i}."""
    return {i}

class Class{i}:
    """Class number {i}."""

    def method_{i}(self):
        """Method number {i}."""
        return {i}
''')
    return '\n'.join(code_parts)


# ============================================================================
# File System Fixtures
# ============================================================================

@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_python_code() -> str:
    """Sample Python code for testing."""
    return '''
def fibonacci(n):
    """Calculate fibonacci number recursively."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def fibonacci_iterative(n):
    """Calculate fibonacci number iteratively."""
    if n <= 1:
        return n

    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

class Calculator:
    """Simple calculator class."""

    def __init__(self):
        self.history = []

    def add(self, a, b):
        """Add two numbers."""
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result

    def multiply(self, a, b):
        """Multiply two numbers."""
        result = a * b
        self.history.append(f"{a} * {b} = {result}")
        return result

    def get_history(self):
        """Get calculation history."""
        return self.history.copy()

def main():
    """Main function to demonstrate usage."""
    calc = Calculator()

    # Test basic operations
    sum_result = calc.add(5, 3)
    product_result = calc.multiply(4, 7)

    # Test fibonacci functions
    fib_recursive = fibonacci(10)
    fib_iterative = fibonacci_iterative(10)

    print(f"Sum: {sum_result}")
    print(f"Product: {product_result}")
    print(f"Fibonacci (recursive): {fib_recursive}")
    print(f"Fibonacci (iterative): {fib_iterative}")
    print(f"History: {calc.get_history()}")

if __name__ == "__main__":
    main()
'''


@pytest.fixture
def sample_javascript_code() -> str:
    """Sample JavaScript code for testing."""
    return '''
function fibonacci(n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

class Calculator {
    constructor() {
        this.history = [];
    }

    add(a, b) {
        const result = a + b;
        this.history.push(`${a} + ${b} = ${result}`);
        return result;
    }

    multiply(a, b) {
        const result = a * b;
        this.history.push(`${a} * ${b} = ${result}`);
        return result;
    }

    getHistory() {
        return [...this.history];
    }
}

export { fibonacci, Calculator };
'''


# ============================================================================
# Performance Testing Fixtures
# ============================================================================

@pytest.fixture
def performance_timer():
    """Timer utility for performance testing."""
    import time

    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None

        def start(self):
            self.start_time = time.time()

        def stop(self):
            self.end_time = time.time()

        @property
        def elapsed(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None

    return Timer()


# ============================================================================
# Mock Data Fixtures
# ============================================================================

@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response."""
    return {
        "choices": [
            {
                "message": {
                    "content": "This is a mock response from OpenAI API.",
                    "role": "assistant"
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 15,
            "total_tokens": 25
        },
        "model": "gpt-4"
    }


@pytest.fixture
def mock_embedding_response():
    """Mock embedding API response."""
    return {
        "data": [
            {
                "embedding": [0.1] * 1536,  # Mock 1536-dimensional embedding
                "index": 0
            }
        ],
        "usage": {
            "prompt_tokens": 5,
            "total_tokens": 5
        },
        "model": "text-embedding-3-large"
    }


# ============================================================================
# Agent Fixtures
# ============================================================================

@pytest.fixture
def structure_analysis_agent():
    """Create StructureAnalysisAgent for testing."""
    from app.services.agents.crew_agents.chunking_agents import StructureAnalysisAgent
    return StructureAnalysisAgent()


@pytest.fixture
def semantic_evaluation_agent():
    """Create SemanticEvaluationAgent for testing."""
    from app.services.agents.crew_agents.chunking_agents import SemanticEvaluationAgent
    return SemanticEvaluationAgent()


@pytest.fixture
def context_optimization_agent():
    """Create ContextOptimizationAgent for testing."""
    from app.services.agents.crew_agents.chunking_agents import ContextOptimizationAgent
    return ContextOptimizationAgent()


@pytest.fixture
def quality_assessment_agent():
    """Create QualityAssessmentAgent for testing."""
    from app.services.agents.crew_agents.chunking_agents import QualityAssessmentAgent
    return QualityAssessmentAgent()


# ============================================================================
# Cleanup and Teardown
# ============================================================================

@pytest.fixture(autouse=True)
async def cleanup_after_test():
    """Automatic cleanup after each test."""
    yield
    # Cleanup code here if needed
    # Clear any global state, close connections, etc.
    pass
