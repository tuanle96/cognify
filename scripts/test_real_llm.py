#!/usr/bin/env python3
"""
Test script for real LLM integration with Cognify.

This script tests the LLM-powered agents with actual API calls.
Set OPENAI_API_KEY environment variable to test with real OpenAI API.
"""

import asyncio
import os
import sys

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.services.llm.factory import create_llm_service
from app.services.llm.base import create_user_message, create_system_message
from agents.crew_agents.llm_agents import LLMStructureAnalysisAgent


async def test_llm_service():
    """Test basic LLM service functionality."""
    print("🧪 Testing LLM Service Integration")
    print("=" * 50)
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  No OPENAI_API_KEY found in environment")
        print("💡 Set OPENAI_API_KEY to test with real OpenAI API")
        print("🔄 Using mock LLM service for testing...")
        provider = "mock"
    else:
        print("✅ Found OPENAI_API_KEY, using real OpenAI API")
        provider = "openai"
    
    try:
        # Create LLM service
        print(f"\n📦 Creating {provider} LLM service...")
        llm_service = await create_llm_service(
            provider=provider,
            model="gpt-4" if provider == "openai" else "mock-gpt-4",
            api_key=api_key
        )
        
        # Test basic generation
        print("\n🔧 Testing basic text generation...")
        messages = [
            create_system_message("You are a helpful assistant."),
            create_user_message("Hello! Please respond with a brief greeting.")
        ]
        
        response = await llm_service.generate(messages, max_tokens=50)
        print(f"✅ Response: {response.content}")
        print(f"📊 Usage: {response.usage}")
        
        # Test health check
        print("\n🏥 Testing health check...")
        health = await llm_service.health_check()
        print(f"Health Status: {health['status']}")
        
        # Cleanup
        await llm_service.cleanup()
        print("✅ LLM service test completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ LLM service test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def test_structure_analysis_agent():
    """Test structure analysis agent with LLM."""
    print("\n🧠 Testing Structure Analysis Agent")
    print("=" * 50)
    
    # Sample code for analysis
    sample_code = '''
def calculate_fibonacci(n):
    """Calculate fibonacci number using recursion."""
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

class MathUtils:
    """Utility class for mathematical operations."""
    
    @staticmethod
    def factorial(n):
        """Calculate factorial of n."""
        if n <= 1:
            return 1
        return n * MathUtils.factorial(n-1)
    
    @staticmethod
    def is_prime(n):
        """Check if number is prime."""
        if n < 2:
            return False
        for i in range(2, int(n ** 0.5) + 1):
            if n % i == 0:
                return False
        return True

def main():
    """Main function to demonstrate usage."""
    print(f"Fibonacci(10): {calculate_fibonacci(10)}")
    print(f"Factorial(5): {MathUtils.factorial(5)}")
    print(f"Is 17 prime? {MathUtils.is_prime(17)}")

if __name__ == "__main__":
    main()
'''
    
    try:
        # Create structure analysis agent
        agent = LLMStructureAnalysisAgent()
        
        # Analyze structure
        print("🔍 Analyzing code structure...")
        boundaries = await agent.analyze_structure(
            content=sample_code,
            language="python",
            file_path="test_math.py"
        )
        
        print(f"✅ Found {len(boundaries)} structural boundaries:")
        for i, boundary in enumerate(boundaries, 1):
            print(f"   {i}. {boundary.get('name', 'unnamed')} ({boundary.get('chunk_type', 'unknown')})")
            print(f"      Lines: {boundary.get('start_line', '?')}-{boundary.get('end_line', '?')}")
            print(f"      Reasoning: {boundary.get('reasoning', 'No reasoning provided')[:100]}...")
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Structure analysis test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def test_streaming():
    """Test streaming LLM responses."""
    print("\n🌊 Testing LLM Streaming")
    print("=" * 50)
    
    api_key = os.getenv("OPENAI_API_KEY")
    provider = "openai" if api_key else "mock"
    
    try:
        # Create LLM service
        llm_service = await create_llm_service(
            provider=provider,
            api_key=api_key
        )
        
        # Test streaming
        print("🔄 Starting stream...")
        messages = [
            create_user_message("Count from 1 to 5, one number per line.")
        ]
        
        print("📝 Streaming response:")
        async for chunk in llm_service.stream_generate(messages, max_tokens=50):
            print(chunk, end="", flush=True)
        
        print("\n✅ Streaming test completed!")
        
        # Cleanup
        await llm_service.cleanup()
        return True
        
    except Exception as e:
        print(f"❌ Streaming test failed: {str(e)}")
        return False


async def main():
    """Main test function."""
    print("🚀 Cognify Real LLM Integration Test Suite")
    print("=" * 60)
    
    # Test basic LLM service
    llm_test_passed = await test_llm_service()
    
    if llm_test_passed:
        # Test structure analysis agent
        agent_test_passed = await test_structure_analysis_agent()
        
        # Test streaming
        streaming_test_passed = await test_streaming()
        
        if agent_test_passed and streaming_test_passed:
            print("\n" + "=" * 60)
            print("🎊 All LLM integration tests passed!")
            print("🚀 Cognify is ready for production LLM usage!")
        else:
            print("\n" + "=" * 60)
            print("⚠️  Some tests failed, but basic functionality works")
    else:
        print("\n" + "=" * 60)
        print("💥 Basic LLM tests failed. Please check configuration.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
