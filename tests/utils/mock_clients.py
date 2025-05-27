"""Mock clients for testing Cognify services."""

import asyncio
import random
from typing import Dict, Any, List, Optional
from unittest.mock import AsyncMock, MagicMock

from app.services.llm.base import LLMMessage, LLMResponse


class MockLLMClient:
    """Mock LLM client for testing without API calls."""

    def __init__(self, provider: str = "mock"):
        self.provider = provider
        self.call_count = 0
        self.responses = []
        self.should_fail = False
        self.failure_message = "Mock LLM failure"

    async def generate(
        self,
        messages: List[LLMMessage],
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> LLMResponse:
        """Generate a mock LLM response."""
        self.call_count += 1

        if self.should_fail:
            raise Exception(self.failure_message)

        # Simulate processing delay
        await asyncio.sleep(0.01)

        # Generate mock response based on input
        content = self._generate_mock_content(messages, max_tokens)
        prompt_tokens = sum(len(msg.content.split()) for msg in messages)
        completion_tokens = len(content.split())

        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }

        response = LLMResponse(
            content=content,
            usage=usage,
            model=f"{self.provider}-mock",
            provider=self.provider
        )

        self.responses.append(response)
        return response

    def _generate_mock_content(self, messages: List[LLMMessage], max_tokens: int) -> str:
        """Generate mock content based on the input messages."""
        last_message = messages[-1].content.lower()

        # Structure analysis responses
        if "analyze" in last_message and "structure" in last_message:
            return '''
            Based on the code analysis, I've identified the following structural boundaries:

            1. Function Definition: fibonacci (lines 1-4)
               - Type: function
               - Reasoning: Self-contained recursive function with clear purpose

            2. Function Definition: fibonacci_iterative (lines 6-12)
               - Type: function
               - Reasoning: Alternative implementation with different algorithm

            3. Class Definition: Calculator (lines 14-35)
               - Type: class
               - Reasoning: Complete class with methods and state management

            4. Main Function: main (lines 37-50)
               - Type: function
               - Reasoning: Entry point function demonstrating usage
            '''

        # Semantic evaluation responses
        elif "semantic" in last_message or "relationship" in last_message:
            return '''
            Semantic analysis reveals the following relationships:

            - fibonacci and fibonacci_iterative are semantically related (same purpose, different implementation)
            - Calculator class methods are cohesive (all mathematical operations)
            - main function demonstrates usage of all other components
            - Strong semantic coherence within each component
            - Minimal coupling between components
            '''

        # Context optimization responses
        elif "optimize" in last_message or "purpose" in last_message:
            return '''
            For the specified purpose, I recommend the following optimizations:

            - Group related functions together (fibonacci variants)
            - Keep class definition as single unit for better understanding
            - Separate demonstration code (main function) for clarity
            - Maintain logical flow from simple to complex components
            '''

        # Quality assessment responses
        elif "quality" in last_message or "assess" in last_message:
            return '''
            Quality assessment results:

            - Code structure: Good (clear separation of concerns)
            - Documentation: Adequate (docstrings present)
            - Complexity: Low to Medium (simple algorithms)
            - Maintainability: High (readable and well-organized)
            - Overall quality score: 0.85
            '''

        # Default response
        else:
            return f"Mock response for: {last_message[:100]}..."

    async def stream_generate(
        self,
        messages: List[LLMMessage],
        max_tokens: int = 1000,
        **kwargs
    ):
        """Generate a streaming mock response."""
        content = self._generate_mock_content(messages, max_tokens)
        words = content.split()

        for i, word in enumerate(words):
            await asyncio.sleep(0.001)  # Simulate streaming delay
            yield {
                "content": word + " ",
                "finish_reason": "stop" if i == len(words) - 1 else None
            }

    async def health_check(self) -> Dict[str, Any]:
        """Mock health check."""
        return {
            "status": "healthy" if not self.should_fail else "unhealthy",
            "provider": self.provider,
            "call_count": self.call_count
        }

    def set_failure_mode(self, should_fail: bool, message: str = "Mock failure"):
        """Set the client to fail for testing error handling."""
        self.should_fail = should_fail
        self.failure_message = message

    def reset(self):
        """Reset the mock client state."""
        self.call_count = 0
        self.responses = []
        self.should_fail = False


class MockEmbeddingClient:
    """Mock embedding client for testing."""

    def __init__(self, dimension: int = 1536):
        self.dimension = dimension
        self.call_count = 0
        self.should_fail = False

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate mock embeddings."""
        self.call_count += 1

        if self.should_fail:
            raise Exception("Mock embedding failure")

        # Generate deterministic mock embeddings
        embeddings = []
        for text in texts:
            # Use hash of text to generate deterministic embedding
            text_hash = hash(text)
            embedding = [
                (text_hash + i) % 1000 / 1000.0
                for i in range(self.dimension)
            ]
            embeddings.append(embedding)

        return embeddings

    async def embed_single(self, text: str) -> List[float]:
        """Generate single mock embedding."""
        embeddings = await self.embed_texts([text])
        return embeddings[0]


class MockVectorDBClient:
    """Mock vector database client for testing."""

    def __init__(self):
        self.collections = {}
        self.call_count = 0
        self.should_fail = False

    async def create_collection(self, name: str, dimension: int) -> bool:
        """Create a mock collection."""
        self.call_count += 1

        if self.should_fail:
            raise Exception("Mock vector DB failure")

        self.collections[name] = {
            "dimension": dimension,
            "vectors": {},
            "metadata": {}
        }
        return True

    async def insert_vectors(
        self,
        collection: str,
        vectors: Dict[str, List[float]],
        metadata: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> bool:
        """Insert mock vectors."""
        self.call_count += 1

        if self.should_fail:
            raise Exception("Mock vector DB failure")

        if collection not in self.collections:
            await self.create_collection(collection, len(list(vectors.values())[0]))

        self.collections[collection]["vectors"].update(vectors)
        if metadata:
            self.collections[collection]["metadata"].update(metadata)

        return True

    async def search_vectors(
        self,
        collection: str,
        query_vector: List[float],
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search mock vectors."""
        self.call_count += 1

        if self.should_fail:
            raise Exception("Mock vector DB failure")

        if collection not in self.collections:
            return []

        # Generate mock search results
        vectors = self.collections[collection]["vectors"]
        metadata = self.collections[collection]["metadata"]

        results = []
        for vector_id, vector in list(vectors.items())[:limit]:
            # Mock similarity score
            score = random.uniform(0.7, 0.95)

            result = {
                "id": vector_id,
                "score": score,
                "vector": vector,
                "metadata": metadata.get(vector_id, {})
            }
            results.append(result)

        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    async def delete_vectors(self, collection: str, vector_ids: List[str]) -> bool:
        """Delete mock vectors."""
        self.call_count += 1

        if self.should_fail:
            raise Exception("Mock vector DB failure")

        if collection in self.collections:
            for vector_id in vector_ids:
                self.collections[collection]["vectors"].pop(vector_id, None)
                self.collections[collection]["metadata"].pop(vector_id, None)

        return True


class MockAgentTool:
    """Mock agent tool for testing."""

    def __init__(self, name: str):
        self.name = name
        self.call_count = 0
        self.should_fail = False

    async def execute(self, *args, **kwargs) -> Dict[str, Any]:
        """Execute mock tool."""
        self.call_count += 1

        if self.should_fail:
            raise Exception(f"Mock tool {self.name} failure")

        return {
            "tool": self.name,
            "result": f"Mock result from {self.name}",
            "args": args,
            "kwargs": kwargs
        }
