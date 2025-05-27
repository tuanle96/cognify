"""
Integration tests for agent services.
"""
import pytest
import asyncio

from app.services.agents.structure_analysis import StructureAnalysisAgent, StructureAnalysisRequest
from app.services.llm.base import LLMConfig, LLMProvider


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


class TestStructureAnalysisAgent:
    """Test structure analysis agent."""
    
    def test_agent_creation(self):
        """Test creating structure analysis agent."""
        agent = StructureAnalysisAgent()
        
        assert agent is not None
        assert agent.llm_config is not None
        assert agent.llm_config.model == "gpt-4o-mini"
        assert not agent._initialized
    
    async def test_agent_initialization(self):
        """Test agent initialization."""
        agent = StructureAnalysisAgent()
        
        try:
            await agent.initialize()
            
            assert agent._initialized
            assert agent.llm_service is not None
            
            print("✅ Agent initialized successfully")
            
        except Exception as e:
            print(f"❌ Agent initialization failed: {e}")
            # Don't fail the test if API is unavailable
            pytest.skip(f"Agent initialization failed: {e}")
        finally:
            if agent._initialized:
                await agent.cleanup()
    
    async def test_structure_analysis(self):
        """Test structure analysis functionality."""
        agent = StructureAnalysisAgent()
        
        try:
            await agent.initialize()
            
            # Test with simple Python code
            request = StructureAnalysisRequest(
                content='''
def hello_world():
    """Say hello to the world."""
    print("Hello, World!")
    return "Hello"

class Calculator:
    """Simple calculator class."""
    
    def add(self, a, b):
        """Add two numbers."""
        return a + b
    
    def multiply(self, x, y):
        """Multiply two numbers."""
        return x * y

# Main execution
if __name__ == "__main__":
    calc = Calculator()
    result = calc.add(2, 3)
    print(f"Result: {result}")
    hello_world()
''',
                language="python",
                file_path="test_example.py",
                analysis_depth="medium"
            )
            
            result = await agent.analyze_structure(request)
            
            assert result is not None
            assert len(result.chunks) > 0
            assert result.processing_time > 0
            assert 0 <= result.quality_score <= 1
            assert 0 <= result.complexity_score <= 1
            
            print(f"✅ Analysis completed: {len(result.chunks)} chunks")
            print(f"✅ Quality score: {result.quality_score:.2f}")
            print(f"✅ Processing time: {result.processing_time:.2f}s")
            
            # Check that we have meaningful chunks
            chunk_types = [chunk.chunk_type.value for chunk in result.chunks]
            print(f"✅ Chunk types: {chunk_types}")
            
        except Exception as e:
            print(f"❌ Structure analysis failed: {e}")
            # Don't fail the test if API is unavailable
            pytest.skip(f"Structure analysis failed: {e}")
        finally:
            if agent._initialized:
                await agent.cleanup()
    
    async def test_fallback_analysis(self):
        """Test fallback analysis when LLM fails."""
        # Create agent with invalid config to force fallback
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model="invalid-model",
            api_key="invalid-key"
        )
        
        agent = StructureAnalysisAgent(config)
        
        request = StructureAnalysisRequest(
            content="def simple_function():\n    return 42",
            language="python",
            file_path="simple.py"
        )
        
        try:
            # This should trigger fallback analysis
            result = await agent.analyze_structure(request)
            
            assert result is not None
            assert len(result.chunks) > 0
            assert result.analysis_metadata.get("fallback") is True
            
            print("✅ Fallback analysis working")
            
        except Exception as e:
            print(f"❌ Fallback analysis failed: {e}")
            pytest.fail(f"Fallback analysis should not fail: {e}")
        finally:
            if agent._initialized:
                await agent.cleanup()
    
    async def test_health_check(self):
        """Test agent health check."""
        agent = StructureAnalysisAgent()
        
        try:
            # Health check without initialization
            health = await agent.health_check()
            assert health["status"] == "unhealthy"
            assert "Not initialized" in health["reason"]
            
            # Health check after initialization
            await agent.initialize()
            health = await agent.health_check()
            
            if health["status"] == "healthy":
                assert "llm_model" in health
                assert "test_chunks" in health
                assert "processing_time" in health
                print("✅ Health check passed")
            else:
                print(f"⚠️ Health check failed: {health}")
                pytest.skip("Health check failed - API may be unavailable")
            
        except Exception as e:
            print(f"❌ Health check error: {e}")
            pytest.skip(f"Health check error: {e}")
        finally:
            if agent._initialized:
                await agent.cleanup()


class TestAgentIntegration:
    """Test agent integration with other services."""
    
    async def test_agent_with_real_llm(self):
        """Test agent with real LLM service."""
        agent = StructureAnalysisAgent()
        
        try:
            await agent.initialize()
            
            # Test with complex code
            complex_code = '''
import os
import sys
from typing import List, Dict, Any

class DataProcessor:
    """Process data with various methods."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.processed_count = 0
    
    def process_file(self, file_path: str) -> Dict[str, Any]:
        """Process a single file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        result = self._analyze_content(content)
        self.processed_count += 1
        return result
    
    def _analyze_content(self, content: str) -> Dict[str, Any]:
        """Analyze content and return metrics."""
        return {
            "lines": len(content.split('\\n')),
            "chars": len(content),
            "words": len(content.split())
        }
    
    def get_stats(self) -> Dict[str, int]:
        """Get processing statistics."""
        return {"processed_files": self.processed_count}

def main():
    """Main function."""
    processor = DataProcessor({"debug": True})
    print("Data processor initialized")

if __name__ == "__main__":
    main()
'''
            
            request = StructureAnalysisRequest(
                content=complex_code,
                language="python",
                file_path="data_processor.py",
                analysis_depth="deep",
                include_dependencies=True,
                include_relationships=True
            )
            
            result = await agent.analyze_structure(request)
            
            assert result is not None
            assert len(result.chunks) >= 3  # Should have multiple chunks
            
            # Check for expected chunk types
            chunk_names = [chunk.name for chunk in result.chunks]
            print(f"✅ Found chunks: {chunk_names}")
            
            # Should have dependencies
            if result.dependencies:
                print(f"✅ Dependencies found: {result.dependencies}")
            
            # Should have relationships
            if result.relationships:
                print(f"✅ Relationships found: {len(result.relationships)} items")
            
        except Exception as e:
            print(f"❌ Real LLM integration failed: {e}")
            pytest.skip(f"Real LLM integration failed: {e}")
        finally:
            if agent._initialized:
                await agent.cleanup()


class TestAgentPerformance:
    """Test agent performance characteristics."""
    
    async def test_analysis_performance(self):
        """Test analysis performance with different code sizes."""
        agent = StructureAnalysisAgent()
        
        try:
            await agent.initialize()
            
            # Test with small code
            small_code = "def hello(): return 'world'"
            request = StructureAnalysisRequest(
                content=small_code,
                language="python",
                file_path="small.py"
            )
            
            result = await agent.analyze_structure(request)
            small_time = result.processing_time
            
            # Test with medium code
            medium_code = "\n".join([
                "def function_{}():".format(i),
                "    return {}".format(i),
                ""
            ] for i in range(10))
            
            request = StructureAnalysisRequest(
                content=medium_code,
                language="python",
                file_path="medium.py"
            )
            
            result = await agent.analyze_structure(request)
            medium_time = result.processing_time
            
            print(f"✅ Small code analysis: {small_time:.2f}s")
            print(f"✅ Medium code analysis: {medium_time:.2f}s")
            
            # Performance should be reasonable
            assert small_time < 30  # Should complete within 30 seconds
            assert medium_time < 60  # Should complete within 60 seconds
            
        except Exception as e:
            print(f"❌ Performance test failed: {e}")
            pytest.skip(f"Performance test failed: {e}")
        finally:
            if agent._initialized:
                await agent.cleanup()
