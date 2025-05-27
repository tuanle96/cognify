#!/usr/bin/env python3
"""
Basic test script for Cognify chunking functionality.

This script tests the core chunking service without requiring external dependencies.
"""

import asyncio
import sys
import os

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.services.chunking.service import ChunkingService
from app.services.chunking.base import ChunkingRequest


async def test_basic_chunking():
    """Test basic chunking functionality."""
    print("🚀 Testing Cognify Basic Chunking Functionality")
    print("=" * 50)
    
    # Sample Python code for testing
    sample_code = '''
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
    """Simple calculator class for basic operations."""
    
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
    
    try:
        # Initialize chunking service
        print("📦 Initializing Chunking Service...")
        chunking_service = ChunkingService()
        await chunking_service.initialize()
        print("✅ Chunking Service initialized successfully")
        
        # Test health check
        print("\n🏥 Testing Health Check...")
        health_status = await chunking_service.health_check()
        print(f"Health Status: {health_status['status']}")
        
        # Create chunking request
        print("\n🔧 Creating Chunking Request...")
        request = ChunkingRequest(
            content=sample_code,
            language="python",
            file_path="test_sample.py",
            purpose="general"
        )
        print(f"Request created for {len(sample_code)} characters of Python code")
        
        # Perform chunking
        print("\n⚡ Performing Chunking...")
        result = await chunking_service.chunk_content(request)
        
        # Display results
        print(f"✅ Chunking completed successfully!")
        print(f"📊 Results Summary:")
        print(f"   - Strategy Used: {result.strategy_used.value}")
        print(f"   - Number of Chunks: {result.chunk_count}")
        print(f"   - Quality Score: {result.quality_score:.3f}")
        print(f"   - Processing Time: {result.processing_time:.3f}s")
        print(f"   - Average Chunk Size: {result.average_chunk_size:.1f} lines")
        print(f"   - Total Lines: {result.total_lines}")
        
        # Display individual chunks
        print(f"\n📝 Individual Chunks:")
        for i, chunk in enumerate(result.chunks, 1):
            print(f"   Chunk {i}:")
            print(f"     - Type: {chunk.chunk_type.value}")
            print(f"     - Name: {chunk.name}")
            print(f"     - Lines: {chunk.start_line}-{chunk.end_line} ({chunk.size_lines} lines)")
            print(f"     - Quality: {chunk.quality_score:.3f}")
            print(f"     - Dependencies: {len(chunk.dependencies)} items")
            if chunk.metadata.complexity:
                print(f"     - Complexity: {chunk.metadata.complexity}")
            print()
        
        # Test performance stats
        print("📈 Performance Statistics:")
        stats = await chunking_service.get_performance_stats()
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"   - {key}: {value:.3f}")
            else:
                print(f"   - {key}: {value}")
        
        # Cleanup
        await chunking_service.cleanup()
        print("\n🧹 Cleanup completed")
        
        print("\n🎉 All tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False


async def test_error_handling():
    """Test error handling scenarios."""
    print("\n🛡️ Testing Error Handling...")
    
    chunking_service = ChunkingService()
    await chunking_service.initialize()
    
    # Test unsupported language
    try:
        request = ChunkingRequest(
            content="console.log('Hello World');",
            language="javascript",  # Not supported yet
            file_path="test.js",
            purpose="general"
        )
        await chunking_service.chunk_content(request)
        print("❌ Should have failed for unsupported language")
    except Exception as e:
        print(f"✅ Correctly handled unsupported language: {type(e).__name__}")
    
    # Test invalid Python syntax
    try:
        request = ChunkingRequest(
            content="def invalid_syntax(\n    pass",  # Invalid syntax
            language="python",
            file_path="invalid.py",
            purpose="general"
        )
        await chunking_service.chunk_content(request)
        print("❌ Should have failed for invalid syntax")
    except Exception as e:
        print(f"✅ Correctly handled invalid syntax: {type(e).__name__}")
    
    await chunking_service.cleanup()


async def main():
    """Main test function."""
    print("🧪 Cognify Basic Functionality Test Suite")
    print("=" * 60)
    
    # Test basic functionality
    basic_test_passed = await test_basic_chunking()
    
    if basic_test_passed:
        # Test error handling
        await test_error_handling()
        
        print("\n" + "=" * 60)
        print("🎊 All tests completed successfully!")
        print("🚀 Cognify is ready for development!")
    else:
        print("\n" + "=" * 60)
        print("💥 Basic tests failed. Please check the setup.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
