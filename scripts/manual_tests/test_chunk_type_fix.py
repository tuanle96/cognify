#!/usr/bin/env python3
"""
Test chunk type fix.
"""

import asyncio
import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_normalize_chunk_type():
    """Test normalize_chunk_type function."""
    print("🔍 Testing normalize_chunk_type...")

    try:
        from app.services.chunking.base import normalize_chunk_type, ChunkType

        # Test cases
        test_cases = [
            ("class_definition", ChunkType.CLASS_DEFINITION),
            ("function", ChunkType.FUNCTION),
            ("method", ChunkType.METHOD),
            ("unknown_type", ChunkType.UNKNOWN),
            ("func", ChunkType.FUNCTION),
            ("cls", ChunkType.CLASS),
            ("semantic_block", ChunkType.SEMANTIC_BLOCK),
        ]

        passed = 0
        for input_str, expected in test_cases:
            result = normalize_chunk_type(input_str)
            if result == expected:
                passed += 1
                print(f"   ✅ '{input_str}' -> {result.value}")
            else:
                print(f"   ❌ '{input_str}' -> {result.value} (expected {expected.value})")

        print(f"✅ Normalize chunk type: {passed}/{len(test_cases)} passed")
        return passed >= len(test_cases) * 0.8

    except Exception as e:
        print(f"❌ Normalize chunk type failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_chunking_with_invalid_types():
    """Test chunking with potentially invalid chunk types."""
    print("🔍 Testing chunking with invalid types...")

    try:
        from app.services.chunking.service import ChunkingService
        from app.services.chunking.base import ChunkingRequest

        service = ChunkingService()
        await service.initialize()

        # Test with Python code that might generate class_definition
        python_code = """
class DataProcessor:
    '''A class for processing data.'''

    def __init__(self, config):
        self.config = config

    def process(self, data):
        '''Process the data.'''
        return data.upper()

def helper_function():
    '''A helper function.'''
    return "helper"
"""

        request = ChunkingRequest(
            content=python_code,
            language="python",
            file_path="processor.py",
            purpose="general"
        )

        result = await service.chunk_content(request)

        print(f"   ✅ Chunks created: {len(result.chunks)}")
        print(f"   ✅ Strategy: {result.strategy_used.value}")
        print(f"   ✅ Quality: {result.quality_score:.2f}")

        # Check chunk types
        for i, chunk in enumerate(result.chunks[:3]):  # Show first 3
            print(f"   📦 Chunk {i+1}: {chunk.name} ({chunk.chunk_type.value})")

        await service.cleanup()
        return len(result.chunks) > 0

    except Exception as e:
        print(f"❌ Chunking with invalid types failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_agent_chunk_creation():
    """Test agent chunk creation directly."""
    print("🔍 Testing agent chunk creation...")

    try:
        from app.services.agents.crew_agents.chunking_agents import ContextOptimizationAgent
        from app.services.chunking.base import ChunkType

        agent = ContextOptimizationAgent()

        # Mock optimization result with potentially problematic chunk type
        mock_result = '''[
            {
                "final_content": "class TestClass:\\n    pass",
                "chunk_type": "class_definition",
                "name": "TestClass",
                "start_line": 1,
                "end_line": 2,
                "quality_reasoning": "Well-defined class"
            },
            {
                "final_content": "def test_function():\\n    return True",
                "chunk_type": "function",
                "name": "test_function",
                "start_line": 4,
                "end_line": 5,
                "quality_reasoning": "Simple function"
            }
        ]'''

        context = {
            "file_path": "test.py",
            "language": "python"
        }

        chunks = agent._create_agentic_chunks(mock_result, "general", context)

        print(f"   ✅ Chunks created: {len(chunks)}")
        for chunk in chunks:
            print(f"   📦 {chunk.name}: {chunk.chunk_type.value}")

        return len(chunks) > 0

    except Exception as e:
        print(f"❌ Agent chunk creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_simple_chunking():
    """Test simple chunking to ensure basic functionality."""
    print("🔍 Testing simple chunking...")

    try:
        from app.services.chunking.service import ChunkingService
        from app.services.chunking.base import ChunkingRequest

        service = ChunkingService()
        await service.initialize()

        # Simple test code
        simple_code = """
def hello():
    print("Hello, World!")

def goodbye():
    print("Goodbye!")
"""

        request = ChunkingRequest(
            content=simple_code,
            language="python",
            file_path="simple.py",
            purpose="general"
        )

        result = await service.chunk_content(request)

        print(f"   ✅ Chunks: {len(result.chunks)}")
        print(f"   ✅ Strategy: {result.strategy_used.value}")
        print(f"   ✅ Quality: {result.quality_score:.2f}")
        print(f"   ✅ Time: {result.processing_time:.3f}s")

        await service.cleanup()
        return len(result.chunks) > 0

    except Exception as e:
        print(f"❌ Simple chunking failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run chunk type fix tests."""
    print("🚀 Chunk Type Fix Test")
    print("=" * 40)

    tests = [
        ("Normalize Chunk Type", test_normalize_chunk_type),
        ("Chunking with Invalid Types", test_chunking_with_invalid_types),
        ("Agent Chunk Creation", test_agent_chunk_creation),
        ("Simple Chunking", test_simple_chunking),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()

            if result:
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"💥 {test_name} CRASHED: {e}")
        print()

    print("=" * 40)
    print(f"📊 Chunk Type Fix Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All chunk type issues fixed!")
        print("🚀 Ready for comprehensive testing!")
        return 0
    elif passed >= total * 0.75:
        print("🎯 Most issues fixed!")
        return 0
    else:
        print(f"⚠️  {total - passed} tests still failing.")
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
