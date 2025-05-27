#!/usr/bin/env python3
"""
Test real LLM agentic chunking with forced LLM calls.
"""

import asyncio
import sys
import os
import time

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_forced_agentic_chunking():
    """Test chunking with forced agentic strategy using real LLM."""
    print("🔍 Testing forced agentic chunking with real LLM...")

    try:
        from app.services.chunking.service import ChunkingService
        from app.services.chunking.base import ChunkingRequest

        service = ChunkingService()
        await service.initialize()

        # Complex Python code that should trigger agentic chunking
        complex_code = """
import asyncio
import logging
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

@dataclass
class ProcessingConfig:
    '''Configuration for data processing operations.'''
    batch_size: int = 100
    timeout: float = 30.0
    retry_count: int = 3
    enable_caching: bool = True

class DataProcessor(ABC):
    '''Abstract base class for data processors.'''

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.processed_count = 0
        self.error_count = 0
        self.cache = {} if config.enable_caching else None

    @abstractmethod
    async def process_item(self, item: Dict) -> Dict:
        '''Process a single data item.'''
        pass

    async def process_batch(self, items: List[Dict]) -> List[Dict]:
        '''Process a batch of items with error handling.'''
        results = []

        for item in items:
            try:
                # Check cache first
                if self.cache and item.get('id') in self.cache:
                    result = self.cache[item['id']]
                    logger.debug(f"Cache hit for item {item['id']}")
                else:
                    result = await self.process_item(item)
                    if self.cache:
                        self.cache[item['id']] = result

                results.append(result)
                self.processed_count += 1

            except Exception as e:
                self.error_count += 1
                logger.error(f"Failed to process item {item.get('id', 'unknown')}: {e}")
                results.append({'error': str(e), 'item_id': item.get('id')})

        return results

    def get_statistics(self) -> Dict[str, Union[int, float]]:
        '''Get processing statistics.'''
        total = self.processed_count + self.error_count
        success_rate = (self.processed_count / total * 100) if total > 0 else 0

        return {
            'processed_count': self.processed_count,
            'error_count': self.error_count,
            'success_rate': success_rate,
            'cache_size': len(self.cache) if self.cache else 0
        }

class TextProcessor(DataProcessor):
    '''Concrete implementation for text processing.'''

    async def process_item(self, item: Dict) -> Dict:
        '''Process text data with validation and transformation.'''
        if not item.get('text'):
            raise ValueError("Text field is required")

        text = item['text'].strip()

        # Simulate async processing
        await asyncio.sleep(0.01)

        processed_text = text.upper()
        word_count = len(text.split())

        return {
            'id': item.get('id'),
            'original_text': text,
            'processed_text': processed_text,
            'word_count': word_count,
            'processed_at': time.time()
        }

async def main():
    '''Main processing function with error handling.'''
    config = ProcessingConfig(batch_size=50, timeout=60.0)
    processor = TextProcessor(config)

    test_items = [
        {'id': 1, 'text': 'Hello world'},
        {'id': 2, 'text': 'Python is awesome'},
        {'id': 3, 'text': ''},  # This will cause an error
        {'id': 4, 'text': 'Async processing rocks'}
    ]

    try:
        results = await processor.process_batch(test_items)
        stats = processor.get_statistics()

        print(f"Processed {len(results)} items")
        print(f"Statistics: {stats}")

        return results

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
"""

        # Force agentic chunking with high quality requirement
        request = ChunkingRequest(
            content=complex_code,
            language="python",
            file_path="complex_processor.py",
            purpose="code_review",
            force_agentic=True,  # Force agentic strategy
            quality_threshold=0.9,  # High quality requirement
            max_processing_time=120  # Allow more time for LLM calls
        )

        print("   🤖 Forcing agentic chunking with real LLM calls...")
        start_time = time.time()

        result = await service.chunk_content(request)

        processing_time = time.time() - start_time

        print(f"   ✅ Strategy used: {result.strategy_used.value}")
        print(f"   ✅ Chunks created: {len(result.chunks)}")
        print(f"   ✅ Quality score: {result.quality_score:.3f}")
        print(f"   ✅ Processing time: {processing_time:.3f}s")
        print(f"   ✅ LLM calls made: {processing_time > 1.0}")  # LLM calls should take longer

        # Show chunk details
        for i, chunk in enumerate(result.chunks[:3]):
            print(f"   📦 Chunk {i+1}: {chunk.name} ({chunk.chunk_type.value})")
            if hasattr(chunk, 'purpose_optimization') and chunk.purpose_optimization:
                print(f"      🎯 Purpose: {chunk.purpose_optimization.get('purpose', 'N/A')}")

        await service.cleanup()

        # Verify it actually used agentic strategy
        is_agentic = result.strategy_used.value == "agentic"
        has_quality_chunks = len(result.chunks) > 0 and result.quality_score > 0.8

        return is_agentic and has_quality_chunks

    except Exception as e:
        print(f"❌ Forced agentic chunking failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_llm_agent_calls():
    """Test individual LLM agent calls directly."""
    print("🔍 Testing individual LLM agent calls...")

    try:
        from agents.crew_agents.chunking_agents import (
            StructureAnalysisAgent, SemanticEvaluationAgent,
            ContextOptimizationAgent, QualityAssessmentAgent
        )

        # Test structure analysis agent
        structure_agent = StructureAnalysisAgent()

        test_code = """
def calculate_fibonacci(n):
    '''Calculate fibonacci number recursively.'''
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

class MathUtils:
    @staticmethod
    def factorial(n):
        if n <= 1:
            return 1
        return n * MathUtils.factorial(n-1)
"""

        print("   🔍 Testing Structure Analysis Agent...")
        start_time = time.time()

        boundaries = await structure_agent.analyze_structure(
            test_code, "python", "test.py"
        )

        structure_time = time.time() - start_time

        print(f"   ✅ Structure boundaries found: {len(boundaries)}")
        print(f"   ✅ Analysis time: {structure_time:.3f}s")
        print(f"   ✅ Real LLM call: {structure_time > 0.5}")  # LLM calls should take time

        if boundaries:
            print(f"   📍 First boundary: {boundaries[0].get('type', 'unknown')}")

        # Test semantic evaluation agent
        semantic_agent = SemanticEvaluationAgent()

        print("   🔍 Testing Semantic Evaluation Agent...")
        start_time = time.time()

        semantic_groups = await semantic_agent.evaluate_semantic_relationships(
            boundaries, test_code, {"file_path": "test.py", "language": "python"}
        )

        semantic_time = time.time() - start_time

        print(f"   ✅ Semantic groups: {len(semantic_groups)}")
        print(f"   ✅ Evaluation time: {semantic_time:.3f}s")
        print(f"   ✅ Real LLM call: {semantic_time > 0.5}")

        return len(boundaries) > 0 and len(semantic_groups) > 0

    except Exception as e:
        print(f"❌ LLM agent calls failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_llm_service_direct():
    """Test LLM service directly with chunking prompts."""
    print("🔍 Testing LLM service with chunking prompts...")

    try:
        from app.services.llm.factory import llm_factory

        # Initialize LLM factory
        await llm_factory.initialize()
        service = llm_factory.get_service()

        # Test chunking-specific prompt
        messages = [
            {
                "role": "user",
                "content": """Analyze this Python code and identify logical boundaries for chunking:

```python
def process_data(data):
    cleaned = clean_data(data)
    return transform_data(cleaned)

def clean_data(data):
    return [item for item in data if item is not None]

def transform_data(data):
    return [item.upper() for item in data]
```

Return JSON with boundaries:
[{"type": "function", "name": "process_data", "start_line": 1, "end_line": 3}]"""
            }
        ]

        print("   🤖 Making real LLM call for chunking analysis...")
        start_time = time.time()

        response = await service.complete(messages, max_tokens=500, temperature=0.1)

        llm_time = time.time() - start_time

        print(f"   ✅ LLM response received: {len(response)} characters")
        print(f"   ✅ Response time: {llm_time:.3f}s")
        print(f"   ✅ Contains JSON: {'[' in response and ']' in response}")
        print(f"   📝 Response preview: {response[:200]}...")

        return len(response) > 50 and llm_time > 0.3  # Real LLM call should take time

    except Exception as e:
        print(f"❌ Direct LLM service test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_agentic_vs_fallback_performance():
    """Compare agentic vs fallback performance with timing."""
    print("🔍 Testing agentic vs fallback performance...")

    try:
        from app.services.chunking.service import ChunkingService
        from app.services.chunking.base import ChunkingRequest

        service = ChunkingService()
        await service.initialize()

        test_code = """
import json
import requests
from typing import Dict, List

class APIClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({'Authorization': f'Bearer {api_key}'})

    def get(self, endpoint: str) -> Dict:
        response = self.session.get(f"{self.base_url}/{endpoint}")
        response.raise_for_status()
        return response.json()

    def post(self, endpoint: str, data: Dict) -> Dict:
        response = self.session.post(f"{self.base_url}/{endpoint}", json=data)
        response.raise_for_status()
        return response.json()
"""

        # Test fallback strategy
        print("   ⚡ Testing fallback strategy...")
        fallback_request = ChunkingRequest(
            content=test_code,
            language="python",
            file_path="api_client.py",
            purpose="general",
            force_agentic=False
        )

        start_time = time.time()
        fallback_result = await service.chunk_content(fallback_request)
        fallback_time = time.time() - start_time

        # Test agentic strategy
        print("   🤖 Testing agentic strategy...")
        agentic_request = ChunkingRequest(
            content=test_code,
            language="python",
            file_path="api_client.py",
            purpose="code_review",
            force_agentic=True,
            quality_threshold=0.9
        )

        start_time = time.time()
        agentic_result = await service.chunk_content(agentic_request)
        agentic_time = time.time() - start_time

        print(f"   📊 Fallback: {len(fallback_result.chunks)} chunks, {fallback_time:.3f}s, quality: {fallback_result.quality_score:.3f}")
        print(f"   🤖 Agentic: {len(agentic_result.chunks)} chunks, {agentic_time:.3f}s, quality: {agentic_result.quality_score:.3f}")
        print(f"   ⚡ Strategy comparison:")
        print(f"      Fallback: {fallback_result.strategy_used.value}")
        print(f"      Agentic: {agentic_result.strategy_used.value}")
        print(f"   🕐 Time difference: {agentic_time - fallback_time:.3f}s")

        await service.cleanup()

        # Verify agentic actually used LLM (should be slower)
        used_real_llm = (agentic_result.strategy_used.value == "agentic" and
                        agentic_time > fallback_time + 0.5)

        return used_real_llm

    except Exception as e:
        print(f"❌ Performance comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run real LLM agentic tests."""
    print("🚀 Real LLM Agentic Chunking Test")
    print("=" * 60)

    tests = [
        ("Direct LLM Service", test_llm_service_direct),
        ("LLM Agent Calls", test_llm_agent_calls),
        ("Forced Agentic Chunking", test_forced_agentic_chunking),
        ("Agentic vs Fallback Performance", test_agentic_vs_fallback_performance),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        try:
            result = await test_func()
            if result:
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"💥 {test_name} CRASHED: {e}")
        print()

    print("=" * 60)
    print(f"📊 Real LLM Test Results: {passed}/{total} tests passed")

    if passed >= total * 0.75:
        print("🎉 Real LLM integration working!")
        print("🤖 Agentic chunking with real LLM calls operational!")
        return 0
    else:
        print(f"⚠️  {total - passed} LLM tests need attention.")
        return 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⏹️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Tests crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
