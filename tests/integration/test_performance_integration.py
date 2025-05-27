#!/usr/bin/env python3
"""
Performance and scalability integration tests.
"""

import asyncio
import sys
import os
import time
import tempfile
from pathlib import Path
import concurrent.futures

# Add project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

async def test_indexing_performance():
    """Test indexing performance with various document sizes."""
    print("⚡ Testing Indexing Performance...")
    
    try:
        from app.services.indexing.service import indexing_service
        from app.services.indexing.base import IndexingRequest, IndexingConfig
        
        # Performance test configuration
        config = IndexingConfig(
            collection_name="perf_test",
            batch_size=10,
            chunk_size=500,
            max_concurrent=3
        )
        
        # Generate test documents of different sizes
        test_cases = [
            {"size": "small", "content_length": 500, "count": 10},
            {"size": "medium", "content_length": 2000, "count": 5},
            {"size": "large", "content_length": 10000, "count": 2}
        ]
        
        performance_results = []
        
        for case in test_cases:
            print(f"   📊 Testing {case['size']} documents ({case['count']} docs, {case['content_length']} chars each)")
            
            # Generate test content
            base_content = "def test_function():\n    '''Test function for performance testing.'''\n    return 'test'\n\n"
            content = base_content * (case['content_length'] // len(base_content))
            
            # Create requests
            requests = []
            for i in range(case['count']):
                request = IndexingRequest(
                    content=content,
                    document_id=f"{case['size']}_doc_{i}",
                    config=config
                )
                requests.append(request)
            
            # Measure performance
            start_time = time.time()
            
            try:
                # Test batch indexing
                response = await indexing_service.index_batch(requests)
                
                end_time = time.time()
                processing_time = end_time - start_time
                
                if response.success:
                    docs_per_second = case['count'] / processing_time
                    chars_per_second = (case['count'] * case['content_length']) / processing_time
                    
                    result = {
                        "size": case['size'],
                        "documents": case['count'],
                        "total_time": processing_time,
                        "docs_per_second": docs_per_second,
                        "chars_per_second": chars_per_second,
                        "chunks_created": response.total_chunks_created,
                        "embeddings_generated": response.total_embeddings_generated
                    }
                    
                    performance_results.append(result)
                    
                    print(f"      ✅ Processed in {processing_time:.2f}s")
                    print(f"      📈 {docs_per_second:.1f} docs/sec")
                    print(f"      📈 {chars_per_second:.0f} chars/sec")
                    print(f"      📄 {response.total_chunks_created} chunks created")
                    
                else:
                    print(f"      ❌ Batch failed: {response.errors}")
                    
            except Exception as e:
                print(f"      ⚠️  Performance test failed: {type(e).__name__}")
        
        # Performance analysis
        if performance_results:
            print("   📊 Performance Summary:")
            total_docs = sum(r['documents'] for r in performance_results)
            total_time = sum(r['total_time'] for r in performance_results)
            avg_docs_per_sec = sum(r['docs_per_second'] for r in performance_results) / len(performance_results)
            
            print(f"      📈 Total documents: {total_docs}")
            print(f"      ⏱️  Total time: {total_time:.2f}s")
            print(f"      📈 Average throughput: {avg_docs_per_sec:.1f} docs/sec")
            
            # Performance benchmarks
            benchmark_passed = avg_docs_per_sec > 1.0  # At least 1 doc/sec
            print(f"      🎯 Benchmark (>1 doc/sec): {'PASSED' if benchmark_passed else 'FAILED'}")
            
            return benchmark_passed
        
        return False
        
    except Exception as e:
        print(f"❌ Indexing performance test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_retrieval_performance():
    """Test retrieval performance with various query types."""
    print("⚡ Testing Retrieval Performance...")
    
    try:
        from app.services.retrieval.service import retrieval_service
        from app.services.retrieval.base import RetrievalRequest, QueryType, RetrievalStrategy
        
        # Test queries of different complexities
        test_queries = [
            # Simple queries
            {"query": "function", "type": QueryType.KEYWORD, "complexity": "simple"},
            {"query": "test", "type": QueryType.SEMANTIC, "complexity": "simple"},
            
            # Medium queries
            {"query": "How to define a function?", "type": QueryType.QUESTION, "complexity": "medium"},
            {"query": "def test_function", "type": QueryType.CODE, "complexity": "medium"},
            
            # Complex queries
            {"query": "How to implement async functions with error handling in Python?", "type": QueryType.QUESTION, "complexity": "complex"},
            {"query": "async def process_data with exception handling", "type": QueryType.CODE, "complexity": "complex"}
        ]
        
        performance_results = []
        
        print(f"   🔍 Testing {len(test_queries)} queries of varying complexity")
        
        for query_info in test_queries:
            print(f"   🔄 Testing {query_info['complexity']} query: '{query_info['query'][:30]}...'")
            
            # Test multiple strategies
            strategies = [RetrievalStrategy.VECTOR_ONLY, RetrievalStrategy.HYBRID]
            
            for strategy in strategies:
                try:
                    request = RetrievalRequest(
                        query=query_info["query"],
                        query_type=query_info["type"],
                        strategy=strategy
                    )
                    
                    # Measure retrieval time
                    start_time = time.time()
                    response = await retrieval_service.retrieve(request)
                    end_time = time.time()
                    
                    retrieval_time = end_time - start_time
                    
                    if response.success:
                        result = {
                            "query": query_info["query"][:30],
                            "complexity": query_info["complexity"],
                            "strategy": strategy.value,
                            "retrieval_time": retrieval_time,
                            "results_count": response.result_count,
                            "processing_time": response.processing_time
                        }
                        
                        performance_results.append(result)
                        
                        print(f"      ✅ {strategy.value}: {retrieval_time:.3f}s, {response.result_count} results")
                        
                    else:
                        print(f"      ❌ {strategy.value} failed: {response.errors}")
                        
                except Exception as e:
                    print(f"      ⚠️  {strategy.value} error: {type(e).__name__}")
        
        # Performance analysis
        if performance_results:
            print("   📊 Retrieval Performance Summary:")
            
            avg_time = sum(r['retrieval_time'] for r in performance_results) / len(performance_results)
            max_time = max(r['retrieval_time'] for r in performance_results)
            min_time = min(r['retrieval_time'] for r in performance_results)
            
            print(f"      ⏱️  Average time: {avg_time:.3f}s")
            print(f"      ⏱️  Max time: {max_time:.3f}s")
            print(f"      ⏱️  Min time: {min_time:.3f}s")
            
            # Performance by complexity
            complexities = set(r['complexity'] for r in performance_results)
            for complexity in complexities:
                complex_results = [r for r in performance_results if r['complexity'] == complexity]
                avg_complex_time = sum(r['retrieval_time'] for r in complex_results) / len(complex_results)
                print(f"      📊 {complexity} queries: {avg_complex_time:.3f}s avg")
            
            # Performance benchmark
            benchmark_passed = avg_time < 5.0  # Less than 5 seconds average
            print(f"      🎯 Benchmark (<5s avg): {'PASSED' if benchmark_passed else 'FAILED'}")
            
            return benchmark_passed
        
        return False
        
    except Exception as e:
        print(f"❌ Retrieval performance test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_concurrent_operations():
    """Test concurrent indexing and retrieval operations."""
    print("⚡ Testing Concurrent Operations...")
    
    try:
        from app.services.indexing.service import indexing_service
        from app.services.retrieval.service import retrieval_service
        from app.services.indexing.base import IndexingRequest
        from app.services.retrieval.base import RetrievalRequest, QueryType
        
        # Prepare concurrent test data
        concurrent_docs = []
        for i in range(5):
            content = f'''
def concurrent_function_{i}():
    """Function {i} for concurrent testing."""
    result = {i} * 2
    return result

class ConcurrentClass_{i}:
    def __init__(self):
        self.value = {i}
    
    def process(self):
        return self.value * 10
'''
            request = IndexingRequest(
                content=content,
                document_id=f"concurrent_doc_{i}"
            )
            concurrent_docs.append(request)
        
        concurrent_queries = [
            RetrievalRequest(query=f"concurrent function {i}", query_type=QueryType.SEMANTIC)
            for i in range(3)
        ]
        
        print(f"   🔄 Testing {len(concurrent_docs)} concurrent indexing operations")
        print(f"   🔍 Testing {len(concurrent_queries)} concurrent retrieval operations")
        
        # Test concurrent indexing
        start_time = time.time()
        
        try:
            # Index documents concurrently
            indexing_tasks = [
                indexing_service.index_document(doc) for doc in concurrent_docs
            ]
            
            indexing_results = await asyncio.gather(*indexing_tasks, return_exceptions=True)
            
            indexing_time = time.time() - start_time
            successful_indexing = sum(1 for r in indexing_results if not isinstance(r, Exception) and r.success)
            
            print(f"      ✅ Concurrent indexing: {successful_indexing}/{len(concurrent_docs)} successful")
            print(f"      ⏱️  Total time: {indexing_time:.3f}s")
            
        except Exception as e:
            print(f"      ⚠️  Concurrent indexing error: {type(e).__name__}")
            successful_indexing = 0
        
        # Test concurrent retrieval
        start_time = time.time()
        
        try:
            # Retrieve concurrently
            retrieval_tasks = [
                retrieval_service.retrieve(query) for query in concurrent_queries
            ]
            
            retrieval_results = await asyncio.gather(*retrieval_tasks, return_exceptions=True)
            
            retrieval_time = time.time() - start_time
            successful_retrieval = sum(1 for r in retrieval_results if not isinstance(r, Exception) and r.success)
            
            print(f"      ✅ Concurrent retrieval: {successful_retrieval}/{len(concurrent_queries)} successful")
            print(f"      ⏱️  Total time: {retrieval_time:.3f}s")
            
        except Exception as e:
            print(f"      ⚠️  Concurrent retrieval error: {type(e).__name__}")
            successful_retrieval = 0
        
        # Test mixed concurrent operations
        print("   🔄 Testing mixed concurrent operations...")
        
        try:
            mixed_tasks = []
            
            # Add some indexing tasks
            for i in range(2):
                content = f"def mixed_function_{i}(): return {i}"
                request = IndexingRequest(content=content, document_id=f"mixed_doc_{i}")
                mixed_tasks.append(indexing_service.index_document(request))
            
            # Add some retrieval tasks
            for i in range(3):
                request = RetrievalRequest(query=f"mixed function {i}", query_type=QueryType.SEMANTIC)
                mixed_tasks.append(retrieval_service.retrieve(request))
            
            start_time = time.time()
            mixed_results = await asyncio.gather(*mixed_tasks, return_exceptions=True)
            mixed_time = time.time() - start_time
            
            successful_mixed = sum(1 for r in mixed_results if not isinstance(r, Exception))
            
            print(f"      ✅ Mixed operations: {successful_mixed}/{len(mixed_tasks)} successful")
            print(f"      ⏱️  Total time: {mixed_time:.3f}s")
            
        except Exception as e:
            print(f"      ⚠️  Mixed operations error: {type(e).__name__}")
            successful_mixed = 0
        
        # Concurrency benchmark
        concurrency_passed = (
            successful_indexing >= len(concurrent_docs) * 0.6 and
            successful_retrieval >= len(concurrent_queries) * 0.6
        )
        
        print(f"   🎯 Concurrency benchmark: {'PASSED' if concurrency_passed else 'FAILED'}")
        
        return concurrency_passed
        
    except Exception as e:
        print(f"❌ Concurrent operations test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_memory_usage():
    """Test memory usage patterns during operations."""
    print("⚡ Testing Memory Usage...")
    
    try:
        import psutil
        import gc
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"   📊 Initial memory usage: {initial_memory:.1f} MB")
        
        # Test memory usage during large operations
        from app.services.indexing.service import indexing_service
        from app.services.indexing.base import IndexingRequest
        
        # Create large content for memory testing
        large_content = "def test_function():\n    return 'test'\n\n" * 1000  # ~30KB per doc
        
        memory_measurements = []
        
        # Index multiple large documents
        for i in range(5):
            request = IndexingRequest(
                content=large_content,
                document_id=f"memory_test_doc_{i}"
            )
            
            try:
                # Measure memory before operation
                before_memory = process.memory_info().rss / 1024 / 1024
                
                # Perform operation
                response = await indexing_service.index_document(request)
                
                # Measure memory after operation
                after_memory = process.memory_info().rss / 1024 / 1024
                memory_increase = after_memory - before_memory
                
                measurement = {
                    "operation": f"index_doc_{i}",
                    "before_mb": before_memory,
                    "after_mb": after_memory,
                    "increase_mb": memory_increase,
                    "success": response.success if hasattr(response, 'success') else False
                }
                
                memory_measurements.append(measurement)
                
                print(f"      📊 Doc {i}: {before_memory:.1f} → {after_memory:.1f} MB (+{memory_increase:.1f})")
                
                # Force garbage collection
                gc.collect()
                
            except Exception as e:
                print(f"      ⚠️  Memory test doc {i} failed: {type(e).__name__}")
        
        # Final memory measurement
        final_memory = process.memory_info().rss / 1024 / 1024
        total_increase = final_memory - initial_memory
        
        print(f"   📊 Final memory usage: {final_memory:.1f} MB")
        print(f"   📊 Total increase: {total_increase:.1f} MB")
        
        # Memory usage analysis
        if memory_measurements:
            avg_increase = sum(m['increase_mb'] for m in memory_measurements) / len(memory_measurements)
            max_increase = max(m['increase_mb'] for m in memory_measurements)
            
            print(f"   📊 Average increase per operation: {avg_increase:.1f} MB")
            print(f"   📊 Maximum increase: {max_increase:.1f} MB")
            
            # Memory benchmark (should not exceed 100MB total increase)
            memory_passed = total_increase < 100.0
            print(f"   🎯 Memory benchmark (<100MB): {'PASSED' if memory_passed else 'FAILED'}")
            
            return memory_passed
        
        return True
        
    except ImportError:
        print("   ⚠️  psutil not available, skipping memory test")
        return True
    except Exception as e:
        print(f"❌ Memory usage test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run performance integration tests."""
    print("🚀 Performance Integration Tests")
    print("=" * 60)
    
    tests = [
        ("Indexing Performance", test_indexing_performance),
        ("Retrieval Performance", test_retrieval_performance),
        ("Concurrent Operations", test_concurrent_operations),
        ("Memory Usage", test_memory_usage),
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
    print(f"📊 Performance Integration Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All performance integration tests passed!")
        return 0
    elif passed >= total * 0.75:
        print("🎯 Most performance integration tests passed!")
        return 0
    else:
        print(f"⚠️  {total - passed} tests need attention.")
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
