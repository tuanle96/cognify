"""
Integration tests with real LLM services using provided OpenAI proxy.
"""
import pytest
import asyncio
import os
from typing import List

from app.services.llm.openai_service import OpenAIService
from app.services.llm.base import LLMMessage
from app.services.embedding.openai_client import OpenAIEmbeddingClient
from app.services.chunking.hybrid_chunker import HybridChunker
from app.services.agents.structure_analysis import StructureAnalysisAgent
from app.core.config import get_settings


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def llm_service():
    """Create OpenAI LLM service with real API."""
    return OpenAIService(
        api_key="test-api-key-for-testing",
        base_url="https://ai.earnbase.io/v1",
        model="gpt-3.5-turbo"
    )


@pytest.fixture
def embedding_service():
    """Create OpenAI embedding service with real API."""
    return OpenAIEmbeddingClient(
        api_key="test-api-key-for-testing",
        base_url="https://ai.earnbase.io/v1"
    )


class TestRealLLMService:
    """Test real LLM service integration."""

    async def test_basic_llm_call(self, llm_service):
        """Test basic LLM API call."""
        messages = [
            LLMMessage(role="user", content="What is 2 + 2?")
        ]

        response = await llm_service.generate(messages, max_tokens=50)

        assert response is not None
        assert response.content is not None
        assert len(response.content) > 0
        assert "4" in response.content
        assert response.usage is not None
        assert response.usage["total_tokens"] > 0

    async def test_code_analysis_prompt(self, llm_service):
        """Test LLM with code analysis prompt."""
        code_sample = '''
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def main():
    for i in range(10):
        print(f"fibonacci({i}) = {fibonacci(i)}")
'''

        messages = [
            LLMMessage(
                role="user",
                content=f"Analyze this Python code and identify the main functions:\n\n{code_sample}"
            )
        ]

        response = await llm_service.generate(messages, max_tokens=200)

        assert response is not None
        assert "fibonacci" in response.content.lower()
        assert "main" in response.content.lower()
        assert response.usage["total_tokens"] > 0

    async def test_chunking_analysis(self, llm_service):
        """Test LLM for chunking analysis."""
        code_sample = '''
class Calculator:
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
        return self.history
'''

        messages = [
            LLMMessage(
                role="user",
                content=f"""Analyze this code for chunking boundaries. Identify logical chunks and explain why they should be grouped together:

{code_sample}

Provide your analysis in a structured format."""
            )
        ]

        response = await llm_service.generate(messages, max_tokens=300)

        assert response is not None
        assert "calculator" in response.content.lower()
        assert "class" in response.content.lower() or "method" in response.content.lower()
        assert response.usage["total_tokens"] > 0

    async def test_streaming_response(self, llm_service):
        """Test streaming LLM response."""
        messages = [
            LLMMessage(
                role="user",
                content="Explain what Python is in 2-3 sentences."
            )
        ]

        chunks = []
        async for chunk in llm_service.stream_generate(messages, max_tokens=100):
            chunks.append(chunk)

        assert len(chunks) > 0
        full_content = "".join(chunk.get("content", "") for chunk in chunks)
        assert len(full_content) > 0
        assert "python" in full_content.lower()


class TestRealEmbeddingService:
    """Test real embedding service integration."""

    async def test_single_embedding(self, embedding_service):
        """Test generating single embedding."""
        text = "This is a test sentence for embedding generation."

        embedding = await embedding_service.embed_single(text)

        assert embedding is not None
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)

    async def test_batch_embeddings(self, embedding_service):
        """Test generating batch embeddings."""
        texts = [
            "Python is a programming language.",
            "JavaScript is used for web development.",
            "Machine learning uses algorithms to learn patterns."
        ]

        embeddings = await embedding_service.embed_texts(texts)

        assert embeddings is not None
        assert len(embeddings) == len(texts)
        assert all(isinstance(emb, list) for emb in embeddings)
        assert all(len(emb) > 0 for emb in embeddings)

    async def test_code_embeddings(self, embedding_service):
        """Test embeddings for code snippets."""
        code_snippets = [
            "def hello_world(): print('Hello, World!')",
            "function greet(name) { console.log(`Hello, ${name}!`); }",
            "public void sayHello() { System.out.println(\"Hello!\"); }"
        ]

        embeddings = await embedding_service.embed_texts(code_snippets)

        assert len(embeddings) == 3
        assert all(len(emb) > 0 for emb in embeddings)

        # Code snippets should have different embeddings
        assert embeddings[0] != embeddings[1]
        assert embeddings[1] != embeddings[2]

    async def test_semantic_similarity(self, embedding_service):
        """Test semantic similarity through embeddings."""
        similar_texts = [
            "The cat is sleeping on the mat.",
            "A cat is resting on a rug."
        ]

        different_texts = [
            "The cat is sleeping on the mat.",
            "Python is a programming language."
        ]

        similar_embeddings = await embedding_service.embed_texts(similar_texts)
        different_embeddings = await embedding_service.embed_texts(different_texts)

        # Calculate cosine similarity (simplified)
        def cosine_similarity(a, b):
            dot_product = sum(x * y for x, y in zip(a, b))
            magnitude_a = sum(x * x for x in a) ** 0.5
            magnitude_b = sum(x * x for x in b) ** 0.5
            return dot_product / (magnitude_a * magnitude_b)

        similar_score = cosine_similarity(similar_embeddings[0], similar_embeddings[1])
        different_score = cosine_similarity(different_embeddings[0], different_embeddings[1])

        # Similar texts should have higher similarity
        assert similar_score > different_score


class TestRealAgenticChunking:
    """Test real agentic chunking with LLM integration."""

    @pytest.fixture
    async def structure_agent(self, llm_service):
        """Create structure analysis agent with real LLM."""
        return StructureAnalysisAgent(llm_service=llm_service)

    async def test_structure_analysis_agent(self, structure_agent):
        """Test structure analysis agent with real LLM."""
        code_sample = '''
import os
import sys

def read_file(filename):
    """Read content from a file."""
    try:
        with open(filename, 'r') as f:
            return f.read()
    except FileNotFoundError:
        return None

class FileProcessor:
    def __init__(self, base_path):
        self.base_path = base_path
        self.processed_files = []

    def process_file(self, filename):
        """Process a single file."""
        full_path = os.path.join(self.base_path, filename)
        content = read_file(full_path)
        if content:
            self.processed_files.append(filename)
            return self.analyze_content(content)
        return None

    def analyze_content(self, content):
        """Analyze file content."""
        lines = content.split('\\n')
        return {
            'line_count': len(lines),
            'char_count': len(content),
            'has_imports': any(line.startswith('import') for line in lines)
        }

if __name__ == "__main__":
    processor = FileProcessor("/tmp")
    result = processor.process_file("test.txt")
    print(result)
'''

        analysis = await structure_agent.analyze_structure(
            code_sample,
            language="python",
            purpose="code_review"
        )

        assert analysis is not None
        assert "boundaries" in analysis
        assert len(analysis["boundaries"]) > 0

        # Should identify main structural elements
        boundary_types = [b.get("type", "").lower() for b in analysis["boundaries"]]
        assert any("function" in bt for bt in boundary_types)
        assert any("class" in bt for bt in boundary_types)

    async def test_hybrid_chunker_with_real_llm(self, llm_service):
        """Test hybrid chunker with real LLM integration."""
        chunker = HybridChunker(llm_service=llm_service)

        code_sample = '''
def factorial(n):
    """Calculate factorial of n."""
    if n <= 1:
        return 1
    return n * factorial(n - 1)

def fibonacci(n):
    """Calculate nth Fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def main():
    """Main function to demonstrate usage."""
    print("Factorial of 5:", factorial(5))
    print("Fibonacci of 10:", fibonacci(10))

if __name__ == "__main__":
    main()
'''

        chunks = await chunker.chunk_code(
            code_sample,
            language="python",
            purpose="documentation"
        )

        assert chunks is not None
        assert len(chunks) > 0

        # Should have meaningful chunks
        chunk_contents = [chunk.content for chunk in chunks]
        assert any("factorial" in content for content in chunk_contents)
        assert any("fibonacci" in content for content in chunk_contents)

        # Check quality scores
        for chunk in chunks:
            assert hasattr(chunk, 'quality_score')
            assert 0 <= chunk.quality_score <= 1


class TestRealPerformance:
    """Test performance with real API calls."""

    async def test_concurrent_llm_calls(self, llm_service):
        """Test concurrent LLM calls."""
        async def make_call(i):
            messages = [LLMMessage(role="user", content=f"What is {i} + {i}?")]
            return await llm_service.generate(messages, max_tokens=20)

        # Make 5 concurrent calls
        tasks = [make_call(i) for i in range(1, 6)]
        responses = await asyncio.gather(*tasks)

        assert len(responses) == 5
        assert all(r.content is not None for r in responses)
        assert all(r.usage["total_tokens"] > 0 for r in responses)

    async def test_concurrent_embeddings(self, embedding_service):
        """Test concurrent embedding generation."""
        texts = [f"This is test sentence number {i}." for i in range(10)]

        # Split into batches and process concurrently
        batch_size = 3
        batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]

        async def process_batch(batch):
            return await embedding_service.embed_texts(batch)

        tasks = [process_batch(batch) for batch in batches]
        results = await asyncio.gather(*tasks)

        # Flatten results
        all_embeddings = []
        for batch_embeddings in results:
            all_embeddings.extend(batch_embeddings)

        assert len(all_embeddings) == len(texts)
        assert all(len(emb) > 0 for emb in all_embeddings)

    async def test_api_rate_limiting(self, llm_service):
        """Test API rate limiting behavior."""
        import time

        start_time = time.time()

        # Make multiple quick calls
        tasks = []
        for i in range(3):
            messages = [LLMMessage(role="user", content=f"Count to {i+1}")]
            task = llm_service.generate(messages, max_tokens=30)
            tasks.append(task)

        responses = await asyncio.gather(*tasks)

        end_time = time.time()
        total_time = end_time - start_time

        assert len(responses) == 3
        assert all(r.content is not None for r in responses)

        # Should complete within reasonable time (allowing for rate limits)
        assert total_time < 30  # 30 seconds max
