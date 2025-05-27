"""
Pytest configuration and shared fixtures for Cognify tests.

This module provides:
- Test configuration and setup
- Shared fixtures for services and components
- Mock objects and test utilities
- Database and service initialization for tests
"""

import asyncio
import os
import pytest
import tempfile
from pathlib import Path
from typing import Dict, Any, AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock, patch

# Test environment setup
os.environ["ENVIRONMENT"] = "testing"
os.environ["LOG_LEVEL"] = "DEBUG"
os.environ["OPENAI_API_KEY"] = "test-key-not-real"

# Import application components
from app.core.config import get_settings
from app.services.chunking.service import ChunkingService
from app.services.chunking.base import ChunkingRequest, ChunkingResult, AgenticChunk
from app.services.llm.factory import create_llm_service
from app.services.llm.base import LLMMessage, LLMResponse
from app.services.agents.crew_agents.chunking_agents import (
    StructureAnalysisAgent,
    SemanticEvaluationAgent,
    ContextOptimizationAgent,
    QualityAssessmentAgent
)


# ============================================================================
# Test Configuration
# ============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_settings():
    """Get test configuration settings."""
    return get_settings()


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
# Service Fixtures
# ============================================================================

@pytest.fixture
async def chunking_service() -> AsyncGenerator[ChunkingService, None]:
    """Create and initialize a ChunkingService for testing."""
    service = ChunkingService()
    await service.initialize()
    yield service
    # Cleanup if needed
    if hasattr(service, 'cleanup'):
        await service.cleanup()


@pytest.fixture
async def mock_llm_service():
    """Create a mock LLM service for testing."""
    service = await create_llm_service(provider="mock")
    yield service
    if hasattr(service, 'cleanup'):
        await service.cleanup()


# ============================================================================
# Request/Response Fixtures
# ============================================================================

@pytest.fixture
def chunking_request(sample_python_code) -> ChunkingRequest:
    """Create a sample chunking request."""
    return ChunkingRequest(
        content=sample_python_code,
        language="python",
        file_path="test_sample.py",
        purpose="general"
    )


@pytest.fixture
def expected_chunking_result() -> Dict[str, Any]:
    """Expected structure for chunking result."""
    return {
        "chunks": [],
        "strategy_used": "agentic",
        "quality_score": 0.0,
        "processing_time": 0.0,
        "metadata": {
            "total_chunks": 0,
            "agent_decisions": []
        }
    }


# ============================================================================
# Agent Fixtures
# ============================================================================

@pytest.fixture
def structure_analysis_agent() -> StructureAnalysisAgent:
    """Create a StructureAnalysisAgent for testing."""
    return StructureAnalysisAgent()


@pytest.fixture
def semantic_evaluation_agent() -> SemanticEvaluationAgent:
    """Create a SemanticEvaluationAgent for testing."""
    return SemanticEvaluationAgent()


@pytest.fixture
def context_optimization_agent() -> ContextOptimizationAgent:
    """Create a ContextOptimizationAgent for testing."""
    return ContextOptimizationAgent()


@pytest.fixture
def quality_assessment_agent() -> QualityAssessmentAgent:
    """Create a QualityAssessmentAgent for testing."""
    return QualityAssessmentAgent()


# ============================================================================
# Mock Fixtures
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
# Test Utilities
# ============================================================================

@pytest.fixture
def assert_chunking_result():
    """Utility function to assert chunking result structure."""
    def _assert_result(result: ChunkingResult):
        assert hasattr(result, 'chunks')
        assert hasattr(result, 'metadata')
        assert hasattr(result, 'strategy_used')
        assert hasattr(result, 'quality_score')
        assert hasattr(result, 'processing_time')
        assert isinstance(result.chunks, list)
        assert isinstance(result.metadata, dict)
        assert result.chunk_count == len(result.chunks)

    return _assert_result


@pytest.fixture
def assert_llm_response():
    """Utility function to assert LLM response structure."""
    def _assert_response(response: LLMResponse):
        assert hasattr(response, 'content')
        assert hasattr(response, 'usage')
        assert isinstance(response.content, str)
        assert isinstance(response.usage, dict)
        assert len(response.content) > 0

    return _assert_response


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
# Cleanup and Teardown
# ============================================================================

@pytest.fixture(autouse=True)
async def cleanup_after_test():
    """Automatic cleanup after each test."""
    yield
    # Cleanup code here if needed
    # Clear any global state, close connections, etc.
    pass
