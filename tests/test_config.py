"""
Test configuration and utilities for Cognify tests.

This module provides test-specific configuration that avoids
database connection issues and provides proper test isolation.
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, Any
from unittest.mock import Mock, AsyncMock

# Set test environment variables
os.environ["ENVIRONMENT"] = "testing"
os.environ["LOG_LEVEL"] = "DEBUG"
os.environ["OPENAI_API_KEY"] = "test-key-not-real-for-testing-only"
os.environ["SECRET_KEY"] = "test-secret-key-for-testing-only-not-production"
# Use SQLite for unit tests to avoid PostgreSQL connection issues
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"


class TestSettings:
    """Test-specific settings that avoid external dependencies."""
    
    def __init__(self):
        self.ENVIRONMENT = "testing"
        self.LOG_LEVEL = "DEBUG"
        self.SECRET_KEY = "test-secret-key"
        self.DATABASE_URL = "sqlite+aiosqlite:///:memory:"
        self.DATABASE_ECHO = False
        self.DATABASE_POOL_SIZE = 1
        self.DATABASE_MAX_OVERFLOW = 0
        self.OPENAI_API_KEY = "test-key-not-real"
        self.EMBEDDING_MODEL = "text-embedding-3-small"
        self.EMBEDDING_DIMENSIONS = 1536
        self.EMBEDDING_BATCH_SIZE = 10
        self.VECTOR_DB_TYPE = "mock"
        self.QDRANT_URL = "http://localhost:6333"
        self.ALLOWED_ORIGINS = ["*"]
        self.ALLOWED_HOSTS = ["*"]
        self.is_development = True


class MockDatabaseSession:
    """Mock database session for testing without real database."""
    
    def __init__(self):
        self.engine = None
        self.session_factory = None
        self._initialized = False
    
    async def initialize(self):
        """Mock initialization."""
        self._initialized = True
        return True
    
    async def get_session(self):
        """Mock session context manager."""
        class MockSession:
            async def __aenter__(self):
                return self
            
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass
            
            async def execute(self, query):
                return Mock(fetchone=Mock(return_value=(1,)))
            
            async def commit(self):
                pass
            
            async def rollback(self):
                pass
            
            async def close(self):
                pass
        
        return MockSession()
    
    async def health_check(self):
        """Mock health check."""
        return {"status": "healthy", "database": "mock"}
    
    async def create_tables(self):
        """Mock table creation."""
        return True
    
    async def cleanup(self):
        """Mock cleanup."""
        return True


class MockEmbeddingService:
    """Mock embedding service for testing."""
    
    def __init__(self):
        self.model = "text-embedding-3-small"
        self.dimensions = 1536
    
    async def embed_text(self, text: str):
        """Mock single text embedding."""
        import random
        return [random.random() for _ in range(self.dimensions)]
    
    async def embed_texts(self, texts: list):
        """Mock batch text embedding."""
        import random
        return [[random.random() for _ in range(self.dimensions)] for _ in texts]
    
    async def health_check(self):
        """Mock health check."""
        return {"status": "healthy", "service": "embedding"}


class MockVectorDBService:
    """Mock vector database service for testing."""
    
    def __init__(self):
        self.collection_name = "test_collection"
        self.dimension = 1536
    
    async def create_collection(self, name: str, dimension: int):
        """Mock collection creation."""
        return {"status": "created", "name": name}
    
    async def insert_vectors(self, vectors: list, metadata: list = None):
        """Mock vector insertion."""
        return {"status": "inserted", "count": len(vectors)}
    
    async def search_vectors(self, query_vector: list, limit: int = 10):
        """Mock vector search."""
        import random
        return [
            {
                "id": f"mock_id_{i}",
                "score": random.random(),
                "metadata": {"text": f"mock_result_{i}"}
            }
            for i in range(min(limit, 5))
        ]
    
    async def health_check(self):
        """Mock health check."""
        return {"status": "healthy", "service": "vectordb"}


class MockChunkingService:
    """Mock chunking service for testing."""
    
    def __init__(self):
        self.supported_languages = ["python", "javascript", "java", "cpp", "go"]
    
    async def chunk_code(self, content: str, language: str, **kwargs):
        """Mock code chunking."""
        # Simple mock chunking - split by lines
        lines = content.split('\n')
        chunks = []
        
        chunk_size = kwargs.get('chunk_size', 10)
        for i in range(0, len(lines), chunk_size):
            chunk_lines = lines[i:i + chunk_size]
            chunks.append({
                "content": '\n'.join(chunk_lines),
                "chunk_type": "code_block",
                "start_line": i + 1,
                "end_line": min(i + chunk_size, len(lines)),
                "metadata": {
                    "language": language,
                    "chunk_index": len(chunks),
                    "total_lines": len(chunk_lines)
                }
            })
        
        return {
            "chunks": chunks,
            "metadata": {
                "language": language,
                "total_chunks": len(chunks),
                "total_lines": len(lines),
                "processing_time": 0.001
            }
        }
    
    def get_supported_languages(self):
        """Mock supported languages."""
        return [
            {"name": lang, "extensions": [f".{lang}"], "description": f"{lang.title()} language"}
            for lang in self.supported_languages
        ]
    
    async def health_check(self):
        """Mock health check."""
        return {"status": "healthy", "service": "chunking"}


class MockLLMService:
    """Mock LLM service for testing."""
    
    def __init__(self):
        self.model = "gpt-3.5-turbo"
    
    async def generate_response(self, prompt: str, **kwargs):
        """Mock LLM response generation."""
        return {
            "response": f"Mock response for: {prompt[:50]}...",
            "model": self.model,
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": 10,
                "total_tokens": len(prompt.split()) + 10
            }
        }
    
    async def health_check(self):
        """Mock health check."""
        return {"status": "healthy", "service": "llm"}


def get_test_settings():
    """Get test settings instance."""
    return TestSettings()


def get_mock_services():
    """Get all mock services for testing."""
    return {
        "database": MockDatabaseSession(),
        "embedding": MockEmbeddingService(),
        "vectordb": MockVectorDBService(),
        "chunking": MockChunkingService(),
        "llm": MockLLMService()
    }


def setup_test_environment():
    """Setup test environment with all necessary mocks."""
    # Set environment variables
    os.environ.update({
        "ENVIRONMENT": "testing",
        "LOG_LEVEL": "DEBUG",
        "DATABASE_URL": "sqlite+aiosqlite:///:memory:",
        "OPENAI_API_KEY": "test-key-not-real",
        "SECRET_KEY": "test-secret-key"
    })
    
    return get_test_settings(), get_mock_services()


# Test data fixtures
SAMPLE_PYTHON_CODE = '''
def fibonacci(n):
    """Calculate fibonacci number recursively."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

class Calculator:
    """Simple calculator class."""
    
    def __init__(self):
        self.history = []
    
    def add(self, a, b):
        """Add two numbers."""
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result

def main():
    """Main function."""
    calc = Calculator()
    print(f"5 + 3 = {calc.add(5, 3)}")
    print(f"Fibonacci(10) = {fibonacci(10)}")

if __name__ == "__main__":
    main()
'''

SAMPLE_JAVASCRIPT_CODE = '''
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
}

function main() {
    const calc = new Calculator();
    console.log(`5 + 3 = ${calc.add(5, 3)}`);
    console.log(`Fibonacci(10) = ${fibonacci(10)}`);
}

main();
'''

SAMPLE_USER_DATA = {
    "username": "testuser",
    "email": "test@example.com",
    "password": "TestPassword123!",
    "full_name": "Test User",
    "role": "user"
}

SAMPLE_ADMIN_DATA = {
    "username": "admin",
    "email": "admin@example.com",
    "password": "AdminPassword123!",
    "full_name": "Admin User",
    "role": "admin"
}
