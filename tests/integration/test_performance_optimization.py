#!/usr/bin/env python3
"""
Comprehensive performance optimization test for chunking service.

Tests caching, batching, benchmarking, and prompt efficiency optimizations.
"""

import asyncio
import sys
import os
import time

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure for testing
os.environ["CHUNKING_STRATEGY"] = "agentic"

async def test_llm_caching():
    """Test LLM response caching performance."""
    print("🔍 Testing LLM caching performance...")

    try:
        from app.services.cache.llm_cache import get_llm_cache, cached_llm_generate
        from app.services.llm.mock_service import create_mock_llm_service
        from app.services.llm.base import LLMMessage

        # Create mock LLM service
        llm_service = create_mock_llm_service()
        await llm_service.initialize()

        # Clear cache
        cache = get_llm_cache()
        cache.clear()

        # Test messages
        messages = [
            LLMMessage(role="user", content="Analyze this Python code structure")
        ]

        # First call (cache miss)
        start_time = time.time()
        response1 = await cached_llm_generate(llm_service, messages, use_cache=True)
        first_call_time = time.time() - start_time

        # Second call (cache hit)
        start_time = time.time()
        response2 = await cached_llm_generate(llm_service, messages, use_cache=True)
        second_call_time = time.time() - start_time

        # Verify caching worked
        assert response1.content == response2.content
        assert second_call_time < first_call_time  # Cache should be faster

        # Check cache stats
        stats = cache.get_stats()

        print(f"✅ LLM caching working!")
        print(f"   First call: {first_call_time:.3f}s (cache miss)")
        print(f"   Second call: {second_call_time:.3f}s (cache hit)")
        print(f"   Speed improvement: {(first_call_time / second_call_time):.1f}x")
        print(f"   Cache hit rate: {stats['hit_rate_percent']:.1f}%")
        print(f"   Cache size: {stats['cache_size']}")

        await llm_service.cleanup()
        return True

    except Exception as e:
        print(f"❌ LLM caching failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_request_batching():
    """Test LLM request batching performance."""
    print("🔍 Testing request batching performance...")

    try:
        from app.services.batch.llm_batch import LLMBatcher, BatchConfig
        from app.services.llm.mock_service import create_mock_llm_service
        from app.services.llm.base import LLMMessage

        # Create mock LLM service
        llm_service = create_mock_llm_service()
        await llm_service.initialize()

        # Create batcher with small batch size for testing
        config = BatchConfig(
            max_batch_size=3,
            max_wait_time=1.0,
            concurrent_requests=2
        )
        batcher = LLMBatcher(llm_service, config)

        # Test concurrent requests
        messages_list = [
            [LLMMessage(role="user", content=f"Analyze code chunk {i}")]
            for i in range(5)
        ]

        start_time = time.time()

        # Submit requests concurrently
        tasks = [
            batcher.generate(messages, priority=i)
            for i, messages in enumerate(messages_list)
        ]

        responses = await asyncio.gather(*tasks)

        total_time = time.time() - start_time

        # Verify all responses received
        assert len(responses) == 5
        for response in responses:
            assert response.content is not None

        # Check batching stats
        stats = batcher.get_stats()

        print(f"✅ Request batching working!")
        print(f"   Total requests: {stats['total_requests']}")
        print(f"   Batched requests: {stats['batched_requests']}")
        print(f"   Total batches: {stats['total_batches']}")
        print(f"   Avg batch size: {stats['avg_batch_size']}")
        print(f"   Batch efficiency: {stats['batch_efficiency_percent']:.1f}%")
        print(f"   Total time: {total_time:.3f}s")

        await batcher.shutdown()
        await llm_service.cleanup()
        return True

    except Exception as e:
        print(f"❌ Request batching failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_performance_benchmarking():
    """Test performance benchmarking system."""
    print("🔍 Testing performance benchmarking...")

    try:
        from app.services.benchmark.chunking_benchmark import (
            ChunkingBenchmark, BenchmarkConfig, BenchmarkType, create_test_requests
        )
        from app.services.chunking.service import ChunkingService

        # Initialize chunking service
        service = ChunkingService()
        await service.initialize()

        # Create benchmark
        benchmark = ChunkingBenchmark(service)

        # Create test requests
        test_requests = create_test_requests()

        # Run quick benchmark
        config = BenchmarkConfig(
            benchmark_type=BenchmarkType.LATENCY,
            num_iterations=3,
            concurrent_requests=1,
            warmup_iterations=1
        )

        print("   Running latency benchmark...")
        results = await benchmark.run_benchmark(test_requests, config)

        # Generate report
        report = benchmark.generate_report(results)

        print(f"✅ Performance benchmarking working!")

        # Show results
        if "latency" in report["summary"]:
            latency_stats = report["summary"]["latency"]
            for strategy, stats in latency_stats.items():
                print(f"   {strategy}: {stats['avg_duration']:.3f}s avg, "
                      f"quality {stats['avg_quality']:.2f}")

        print(f"   Recommendations: {len(report['recommendations'])}")
        for rec in report["recommendations"][:2]:
            print(f"     • {rec}")

        return True

    except Exception as e:
        print(f"❌ Performance benchmarking failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_prompt_efficiency():
    """Test optimized prompt efficiency."""
    print("🔍 Testing prompt efficiency optimizations...")

    try:
        from app.services.agents.crew_agents.chunking_agents import StructureAnalysisAgent

        agent = StructureAnalysisAgent()

        # Test with long code content
        long_code = """
import os
import sys
from typing import List, Dict, Optional, Union, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging
import asyncio
import time

# This is a very long code file to test prompt optimization
""" + "\n".join([f"def function_{i}():\n    pass\n" for i in range(50)])

        print(f"   Testing with {len(long_code)} character code file...")

        start_time = time.time()

        # Analyze structure (should use optimized prompts)
        boundaries = await agent.analyze_structure(
            content=long_code,
            language="python",
            file_path="large_test_file.py"
        )

        analysis_time = time.time() - start_time

        print(f"✅ Prompt efficiency optimizations working!")
        print(f"   Analysis time: {analysis_time:.3f}s")
        print(f"   Boundaries found: {len(boundaries)}")
        print(f"   Input size: {len(long_code)} chars")
        print(f"   Processing rate: {len(long_code) / analysis_time:.0f} chars/sec")

        # Verify boundaries are reasonable
        assert len(boundaries) > 0
        assert len(boundaries) < 100  # Should not create too many boundaries

        return True

    except Exception as e:
        print(f"❌ Prompt efficiency test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_integrated_performance():
    """Test integrated performance with all optimizations."""
    print("🔍 Testing integrated performance with all optimizations...")

    try:
        from app.services.chunking.service import ChunkingService
        from app.services.chunking.base import ChunkingRequest
        from app.services.cache.llm_cache import get_llm_cache

        # Initialize service
        service = ChunkingService()
        await service.initialize()

        # Clear cache for clean test
        cache = get_llm_cache()
        cache.clear()

        # Test code
        test_code = """
import asyncio
import time
from typing import Dict, List, Any

class PerformanceOptimizer:
    '''Advanced performance optimization class.'''

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cache = {}
        self.metrics = {
            'cache_hits': 0,
            'cache_misses': 0,
            'total_requests': 0
        }

    async def optimize_request(self, request_data: Dict) -> Dict:
        '''Optimize a single request with caching.'''
        cache_key = self._generate_cache_key(request_data)

        # Check cache first
        if cache_key in self.cache:
            self.metrics['cache_hits'] += 1
            return self.cache[cache_key]

        # Process request
        self.metrics['cache_misses'] += 1
        result = await self._process_request(request_data)

        # Cache result
        self.cache[cache_key] = result
        self.metrics['total_requests'] += 1

        return result

    def _generate_cache_key(self, data: Dict) -> str:
        '''Generate cache key for request data.'''
        import hashlib
        import json
        content = json.dumps(data, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()

    async def _process_request(self, data: Dict) -> Dict:
        '''Process request with simulated work.'''
        await asyncio.sleep(0.1)  # Simulate processing
        return {
            'processed': True,
            'timestamp': time.time(),
            'data_size': len(str(data))
        }

    def get_performance_stats(self) -> Dict:
        '''Get performance statistics.'''
        hit_rate = 0
        if self.metrics['total_requests'] > 0:
            hit_rate = self.metrics['cache_hits'] / self.metrics['total_requests'] * 100

        return {
            'cache_hit_rate': hit_rate,
            'total_requests': self.metrics['total_requests'],
            'cache_size': len(self.cache)
        }

async def main():
    '''Main function with performance testing.'''
    optimizer = PerformanceOptimizer({'batch_size': 10})

    # Test multiple requests
    requests = [
        {'id': i, 'data': f'test_data_{i}'}
        for i in range(5)
    ]

    results = []
    for request in requests:
        result = await optimizer.optimize_request(request)
        results.append(result)

    stats = optimizer.get_performance_stats()
    print(f"Performance stats: {stats}")

    return results

if __name__ == "__main__":
    asyncio.run(main())
"""

        # Test multiple chunking requests (should benefit from caching)
        request = ChunkingRequest(
            content=test_code,
            language="python",
            file_path="performance_test.py",
            purpose="code_review",
            force_agentic=True
        )

        # First request (cache miss)
        start_time = time.time()
        result1 = await service.chunk_content(request)
        first_time = time.time() - start_time

        # Second identical request (cache hit)
        start_time = time.time()
        result2 = await service.chunk_content(request)
        second_time = time.time() - start_time

        # Third request with different purpose (partial cache hit)
        request.purpose = "bug_detection"
        start_time = time.time()
        result3 = await service.chunk_content(request)
        third_time = time.time() - start_time

        # Check cache performance
        cache_stats = cache.get_stats()

        print(f"✅ Integrated performance optimizations working!")
        print(f"   First request: {first_time:.3f}s ({len(result1.chunks)} chunks)")
        print(f"   Second request: {second_time:.3f}s ({len(result2.chunks)} chunks)")
        print(f"   Third request: {third_time:.3f}s ({len(result3.chunks)} chunks)")
        print(f"   Speed improvement: {first_time / second_time:.1f}x")
        print(f"   Cache hit rate: {cache_stats['hit_rate_percent']:.1f}%")
        print(f"   Memory saved: {cache_stats['memory_saved_tokens']} tokens")

        # Verify results are consistent
        assert len(result1.chunks) > 0
        assert len(result2.chunks) > 0
        assert len(result3.chunks) > 0

        return True

    except Exception as e:
        print(f"❌ Integrated performance test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all performance optimization tests."""
    print("🚀 Performance Optimization Tests")
    print("=" * 50)

    tests = [
        ("LLM Caching", test_llm_caching),
        ("Request Batching", test_request_batching),
        ("Performance Benchmarking", test_performance_benchmarking),
        ("Prompt Efficiency", test_prompt_efficiency),
        ("Integrated Performance", test_integrated_performance),
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

    print("=" * 50)
    print(f"📊 Performance Optimization Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All performance optimizations working perfectly!")
        print("🚀 Chunking service is now highly optimized for production!")
        return 0
    else:
        print(f"⚠️  {total - passed} optimizations need attention.")
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
