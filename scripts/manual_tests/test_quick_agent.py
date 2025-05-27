#!/usr/bin/env python3
"""
Quick test for agent functionality.
"""

def test_imports():
    """Test basic imports."""
    try:
        print("Testing LLM imports...")
        from app.services.llm.base import LLMMessage, LLMConfig, LLMProvider
        print("✅ LLM base imports OK")

        from app.services.llm.openai_service import OpenAIService
        print("✅ OpenAI service import OK")

        print("Testing agent imports...")
        from app.services.agents.structure_analysis import StructureAnalysisAgent
        print("✅ Agent import OK")

        print("Creating agent...")
        agent = StructureAnalysisAgent()
        print(f"✅ Agent created: {agent.llm_config.model}")

        return True

    except Exception as e:
        print(f"❌ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_llm_service():
    """Test LLM service directly."""
    try:
        from app.services.llm.base import LLMConfig, LLMProvider
        from app.services.llm.openai_service import OpenAIService

        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4o-mini",
            api_key="test-api-key-for-testing",
            base_url="https://ai.earnbase.io/v1"
        )

        service = OpenAIService(config)
        print("✅ LLM service created")
        return True

    except Exception as e:
        print(f"❌ LLM service error: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Quick Agent Test")
    print("=" * 50)

    success = True

    print("\n1. Testing imports...")
    if not test_imports():
        success = False

    print("\n2. Testing LLM service...")
    if not test_llm_service():
        success = False

    print("\n" + "=" * 50)
    if success:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
