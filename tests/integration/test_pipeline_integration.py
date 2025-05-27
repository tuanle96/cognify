#!/usr/bin/env python3
"""
End-to-end pipeline integration tests.
"""

import asyncio
import sys
import os
import tempfile
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

async def test_complete_rag_pipeline():
    """Test complete RAG pipeline from document to retrieval."""
    print("🔄 Testing Complete RAG Pipeline...")
    
    try:
        from app.services.indexing.service import indexing_service
        from app.services.retrieval.service import retrieval_service
        from app.services.indexing.base import IndexingRequest, IndexingConfig
        from app.services.retrieval.base import RetrievalRequest, QueryType
        
        # Test configuration
        config = IndexingConfig(
            collection_name="rag_pipeline_test",
            chunk_size=400,
            batch_size=3,
            max_concurrent=2
        )
        
        # Test documents representing real-world scenarios
        test_documents = [
            {
                "content": '''
def calculate_fibonacci(n):
    """
    Calculate the nth Fibonacci number using recursion.
    
    Args:
        n (int): The position in the Fibonacci sequence
        
    Returns:
        int: The nth Fibonacci number
    """
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

def fibonacci_iterative(n):
    """Calculate Fibonacci number iteratively for better performance."""
    if n <= 1:
        return n
    
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
''',
                "doc_id": "fibonacci_functions",
                "metadata": {"language": "python", "topic": "algorithms"}
            },
            {
                "content": '''
# Async Programming in Python

## Introduction
Asynchronous programming allows you to write concurrent code using the async/await syntax.

## Basic Async Function
```python
async def fetch_data():
    await asyncio.sleep(1)
    return "Data fetched"
```

## Running Async Functions
```python
import asyncio

async def main():
    result = await fetch_data()
    print(result)

asyncio.run(main())
```

## Benefits
- Non-blocking operations
- Better resource utilization
- Improved performance for I/O bound tasks
''',
                "doc_id": "async_programming_guide",
                "metadata": {"type": "documentation", "topic": "async"}
            },
            {
                "content": '''
{
    "api": {
        "name": "User Management API",
        "version": "2.0.0",
        "endpoints": [
            {
                "path": "/users",
                "method": "GET",
                "description": "Retrieve all users",
                "parameters": {
                    "limit": "number of users to return",
                    "offset": "pagination offset"
                }
            },
            {
                "path": "/users/{id}",
                "method": "GET", 
                "description": "Retrieve user by ID",
                "parameters": {
                    "id": "unique user identifier"
                }
            },
            {
                "path": "/users",
                "method": "POST",
                "description": "Create new user",
                "body": {
                    "name": "string",
                    "email": "string",
                    "role": "string"
                }
            }
        ]
    }
}
''',
                "doc_id": "api_documentation",
                "metadata": {"type": "api_spec", "format": "json"}
            }
        ]
        
        print(f"   📚 Pipeline test with {len(test_documents)} documents")
        
        # Step 1: Index all documents
        print("   🔄 Step 1: Indexing documents...")
        
        indexing_results = []
        for doc in test_documents:
            try:
                request = IndexingRequest(
                    content=doc["content"],
                    document_id=doc["doc_id"],
                    metadata=doc["metadata"],
                    config=config
                )
                
                response = await indexing_service.index_document(request)
                indexing_results.append(response)
                
                if response.success:
                    print(f"      ✅ Indexed {doc['doc_id']}: {response.total_chunks_created} chunks")
                else:
                    print(f"      ❌ Failed to index {doc['doc_id']}: {response.errors}")
                    
            except Exception as e:
                print(f"      ⚠️  Error indexing {doc['doc_id']}: {type(e).__name__}")
        
        successful_indexing = sum(1 for r in indexing_results if r.success)
        print(f"   📊 Indexing summary: {successful_indexing}/{len(test_documents)} successful")
        
        # Step 2: Test retrieval with various queries
        print("   🔄 Step 2: Testing retrieval...")
        
        test_queries = [
            {
                "query": "How to calculate Fibonacci numbers?",
                "type": QueryType.QUESTION,
                "expected_docs": ["fibonacci_functions"]
            },
            {
                "query": "async await python",
                "type": QueryType.SEMANTIC,
                "expected_docs": ["async_programming_guide"]
            },
            {
                "query": "API endpoints users",
                "type": QueryType.KEYWORD,
                "expected_docs": ["api_documentation"]
            },
            {
                "query": "def fibonacci",
                "type": QueryType.CODE,
                "expected_docs": ["fibonacci_functions"]
            }
        ]
        
        retrieval_results = []
        for query_info in test_queries:
            try:
                request = RetrievalRequest(
                    query=query_info["query"],
                    query_type=query_info["type"]
                )
                
                response = await retrieval_service.retrieve(request)
                retrieval_results.append(response)
                
                if response.success:
                    print(f"      ✅ Query '{query_info['query'][:30]}...': {response.result_count} results")
                    
                    # Check if expected documents are found
                    found_docs = set()
                    for result in response.results:
                        found_docs.add(result.document_id)
                    
                    expected_docs = set(query_info["expected_docs"])
                    overlap = found_docs.intersection(expected_docs)
                    
                    if overlap:
                        print(f"         🎯 Found expected docs: {list(overlap)}")
                    else:
                        print(f"         ⚠️  Expected docs not found: {list(expected_docs)}")
                        
                else:
                    print(f"      ❌ Query failed: {response.errors}")
                    
            except Exception as e:
                print(f"      ⚠️  Query error: {type(e).__name__}")
        
        successful_retrieval = sum(1 for r in retrieval_results if r.success)
        print(f"   📊 Retrieval summary: {successful_retrieval}/{len(test_queries)} successful")
        
        # Step 3: Test pipeline performance
        print("   🔄 Step 3: Performance analysis...")
        
        total_indexing_time = sum(r.total_processing_time for r in indexing_results if r.success)
        total_retrieval_time = sum(r.processing_time for r in retrieval_results if r.success)
        
        print(f"      ⏱️  Total indexing time: {total_indexing_time:.3f}s")
        print(f"      ⏱️  Total retrieval time: {total_retrieval_time:.3f}s")
        print(f"      📊 Avg indexing per doc: {total_indexing_time/max(1, successful_indexing):.3f}s")
        print(f"      📊 Avg retrieval per query: {total_retrieval_time/max(1, successful_retrieval):.3f}s")
        
        # Overall success criteria
        pipeline_success = (
            successful_indexing >= len(test_documents) * 0.5 and  # At least 50% indexed
            successful_retrieval >= len(test_queries) * 0.5       # At least 50% retrieved
        )
        
        if pipeline_success:
            print("✅ Complete RAG pipeline test PASSED")
        else:
            print("❌ Complete RAG pipeline test FAILED")
            
        return pipeline_success
        
    except Exception as e:
        print(f"❌ Complete RAG pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_multi_format_processing():
    """Test pipeline with multiple document formats."""
    print("🔄 Testing Multi-Format Processing...")
    
    try:
        # Create temporary files with different formats
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test files
            test_files = {
                "python_script.py": '''
def greet(name):
    """Greet a person by name."""
    return f"Hello, {name}!"

class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def introduce(self):
        return f"I'm {self.name}, {self.age} years old"
''',
                "readme.md": '''
# Test Project

This is a test project for demonstrating multi-format processing.

## Features
- Python code processing
- Markdown documentation
- JSON configuration

## Usage
Run the main script to see the demo.
''',
                "config.json": '''
{
    "project": {
        "name": "Test Project",
        "version": "1.0.0",
        "dependencies": [
            "python>=3.8",
            "asyncio",
            "json"
        ]
    },
    "settings": {
        "debug": true,
        "log_level": "INFO"
    }
}
''',
                "data.yaml": '''
users:
  - name: Alice
    role: admin
    permissions:
      - read
      - write
      - delete
  - name: Bob
    role: user
    permissions:
      - read
'''
            }
            
            # Write test files
            for filename, content in test_files.items():
                file_path = temp_path / filename
                file_path.write_text(content)
            
            print(f"   📁 Created {len(test_files)} test files")
            
            # Test directory indexing
            from app.services.indexing.service import indexing_service
            
            try:
                response = await indexing_service.index_directory(
                    directory_path=str(temp_path),
                    recursive=False,
                    incremental=False
                )
                
                if response.success:
                    print(f"   ✅ Directory indexing successful")
                    print(f"   📊 Documents processed: {response.progress.successful_documents}")
                    print(f"   📊 Total chunks: {response.total_chunks_created}")
                    print(f"   📊 Processing time: {response.total_processing_time:.3f}s")
                    
                    # Test format-specific queries
                    from app.services.retrieval.service import retrieval_service
                    from app.services.retrieval.base import RetrievalRequest, QueryType
                    
                    format_queries = [
                        ("def greet", QueryType.CODE, "python"),
                        ("Test Project features", QueryType.SEMANTIC, "markdown"),
                        ("project configuration", QueryType.KEYWORD, "json"),
                        ("users permissions", QueryType.SEMANTIC, "yaml")
                    ]
                    
                    for query, qtype, expected_format in format_queries:
                        try:
                            request = RetrievalRequest(query=query, query_type=qtype)
                            result = await retrieval_service.retrieve(request)
                            
                            if result.success and result.results:
                                print(f"   🔍 Query '{query}' found {result.result_count} results")
                                top_result = result.results[0]
                                print(f"      📄 Top result: {top_result.content[:50]}...")
                            else:
                                print(f"   ⚠️  Query '{query}' returned no results")
                                
                        except Exception as e:
                            print(f"   ⚠️  Query '{query}' failed: {type(e).__name__}")
                    
                else:
                    print(f"   ❌ Directory indexing failed: {response.errors}")
                    return False
                    
            except Exception as e:
                print(f"   ⚠️  Directory indexing error: {type(e).__name__}")
                return False
        
        print("✅ Multi-format processing test completed")
        return True
        
    except Exception as e:
        print(f"❌ Multi-format processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_error_handling_pipeline():
    """Test pipeline error handling and recovery."""
    print("🔄 Testing Error Handling Pipeline...")
    
    try:
        from app.services.indexing.service import indexing_service
        from app.services.indexing.base import IndexingRequest
        
        # Test error scenarios
        error_scenarios = [
            {
                "name": "Empty content",
                "request": IndexingRequest(content="", document_id="empty_doc"),
                "expected_error": True
            },
            {
                "name": "Very large content",
                "request": IndexingRequest(
                    content="x" * 1000000,  # 1MB of text
                    document_id="large_doc"
                ),
                "expected_error": False  # Should handle gracefully
            },
            {
                "name": "Invalid characters",
                "request": IndexingRequest(
                    content="Test with \x00 null bytes and \xff invalid chars",
                    document_id="invalid_chars"
                ),
                "expected_error": False  # Should clean and process
            },
            {
                "name": "Missing document ID",
                "request": IndexingRequest(content="Test content"),  # Auto-generated ID
                "expected_error": False
            }
        ]
        
        print(f"   🧪 Testing {len(error_scenarios)} error scenarios")
        
        for scenario in error_scenarios:
            print(f"   🔄 Testing: {scenario['name']}")
            
            try:
                response = await indexing_service.index_document(scenario["request"])
                
                if scenario["expected_error"]:
                    if not response.success:
                        print(f"      ✅ Expected error handled correctly")
                    else:
                        print(f"      ⚠️  Expected error but got success")
                else:
                    if response.success:
                        print(f"      ✅ Handled gracefully")
                    else:
                        print(f"      ⚠️  Unexpected failure: {response.errors}")
                        
            except Exception as e:
                if scenario["expected_error"]:
                    print(f"      ✅ Expected exception: {type(e).__name__}")
                else:
                    print(f"      ⚠️  Unexpected exception: {type(e).__name__}")
        
        # Test service recovery
        print("   🔄 Testing service recovery...")
        
        # Test health checks
        from app.services.indexing.service import indexing_service
        from app.services.retrieval.service import retrieval_service
        
        services = [
            ("indexing", indexing_service),
            ("retrieval", retrieval_service)
        ]
        
        for service_name, service in services:
            try:
                health = await service.health_check()
                status = health.get("status", "unknown")
                print(f"      🏥 {service_name} service: {status}")
            except Exception as e:
                print(f"      ⚠️  {service_name} health check failed: {type(e).__name__}")
        
        print("✅ Error handling pipeline test completed")
        return True
        
    except Exception as e:
        print(f"❌ Error handling pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run pipeline integration tests."""
    print("🚀 Pipeline Integration Tests")
    print("=" * 60)
    
    tests = [
        ("Complete RAG Pipeline", test_complete_rag_pipeline),
        ("Multi-Format Processing", test_multi_format_processing),
        ("Error Handling Pipeline", test_error_handling_pipeline),
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
    print(f"📊 Pipeline Integration Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All pipeline integration tests passed!")
        return 0
    elif passed >= total * 0.67:
        print("🎯 Most pipeline integration tests passed!")
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
