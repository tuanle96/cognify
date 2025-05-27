#!/usr/bin/env python3
"""
Test agentic chunking with mock LLM service.
This script tests the full agentic chunking pipeline using mock responses.
"""

import asyncio
import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Force mock LLM usage
os.environ["CHUNKING_STRATEGY"] = "agentic"

async def test_mock_llm_service():
    """Test mock LLM service functionality."""
    print("🔍 Testing mock LLM service...")

    try:
        from app.services.llm.mock_service import create_mock_llm_service
        from app.services.llm.base import LLMMessage

        service = create_mock_llm_service()
        await service.initialize()

        # Test health check
        health = await service.health_check()
        print(f"✅ Mock LLM health: {health['status']}")

        # Test structure analysis response
        messages = [
            LLMMessage(role="user", content="Analyze the structure of this Python code")
        ]

        response = await service.generate(messages)
        print(f"✅ Mock LLM response type: {response.metadata.get('response_type', 'unknown')}")
        print(f"   Response length: {len(response.content)} chars")

        await service.cleanup()
        return True

    except Exception as e:
        print(f"❌ Mock LLM service failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_chunking_agents_with_mock():
    """Test chunking agents with mock LLM."""
    print("🔍 Testing chunking agents with mock LLM...")

    try:
        from app.services.agents.crew_agents.chunking_agents import (
            StructureAnalysisAgent,
            SemanticEvaluationAgent,
            ContextOptimizationAgent,
            QualityAssessmentAgent
        )

        test_code = """
def fibonacci(n):
    '''Calculate fibonacci number recursively.'''
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

class Calculator:
    '''Simple calculator class.'''

    def __init__(self):
        self.history = []

    def add(self, a, b):
        '''Add two numbers.'''
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result

def main():
    calc = Calculator()
    print(calc.add(2, 3))
    print(fibonacci(5))
"""

        # Test Structure Analysis Agent
        print("   Testing StructureAnalysisAgent...")
        structure_agent = StructureAnalysisAgent()
        boundaries = await structure_agent.analyze_structure(
            content=test_code,
            language="python",
            file_path="test.py"
        )
        print(f"   ✅ Found {len(boundaries)} structural boundaries")

        # Test Semantic Evaluation Agent
        print("   Testing SemanticEvaluationAgent...")
        semantic_agent = SemanticEvaluationAgent()
        semantic_groups = await semantic_agent.evaluate_semantic_groups(
            initial_chunks=boundaries,
            content=test_code
        )
        print(f"   ✅ Created {len(semantic_groups)} semantic groups")

        # Test Context Optimization Agent
        print("   Testing ContextOptimizationAgent...")
        context_agent = ContextOptimizationAgent()
        optimized_chunks = await context_agent.optimize_for_purpose(
            semantic_groups=semantic_groups,
            purpose="code_review",
            context={"file_path": "test.py", "language": "python"}
        )
        print(f"   ✅ Optimized to {len(optimized_chunks)} chunks")

        # Test Quality Assessment Agent
        print("   Testing QualityAssessmentAgent...")
        quality_agent = QualityAssessmentAgent()
        quality_assessment = await quality_agent.assess_chunking_quality(
            chunks=optimized_chunks,
            original_content=test_code,
            purpose="code_review"
        )
        overall_score = quality_assessment.get('overall_quality_score', 0)
        print(f"   ✅ Quality assessment: {overall_score:.2f}")

        return True

    except Exception as e:
        print(f"❌ Chunking agents with mock failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_full_agentic_pipeline():
    """Test complete agentic chunking pipeline."""
    print("🔍 Testing full agentic chunking pipeline...")

    try:
        from app.services.chunking.service import ChunkingService
        from app.services.chunking.base import ChunkingRequest

        service = ChunkingService()
        await service.initialize()

        # Complex test code
        test_code = """
import os
import sys
from typing import List, Dict, Optional

class FileProcessor:
    '''Advanced file processing class.'''

    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.processed_files = []
        self.errors = []

    def validate_file(self, file_path: str) -> bool:
        '''Validate if file can be processed.'''
        if not os.path.exists(file_path):
            self.errors.append(f"File not found: {file_path}")
            return False

        if not os.access(file_path, os.R_OK):
            self.errors.append(f"File not readable: {file_path}")
            return False

        return True

    def process_file(self, file_path: str) -> Dict[str, any]:
        '''Process a single file and return metadata.'''
        if not self.validate_file(file_path):
            return None

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            metadata = {
                'path': file_path,
                'size': len(content),
                'lines': len(content.split('\\n')),
                'extension': os.path.splitext(file_path)[1],
                'encoding': 'utf-8'
            }

            self.processed_files.append(file_path)
            return metadata

        except Exception as e:
            self.errors.append(f"Error processing {file_path}: {str(e)}")
            return None

    def process_directory(self, directory: str) -> List[Dict]:
        '''Process all files in a directory recursively.'''
        results = []

        if not os.path.isdir(directory):
            self.errors.append(f"Directory not found: {directory}")
            return results

        for root, dirs, files in os.walk(directory):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.')]

            for file in files:
                if file.startswith('.'):
                    continue

                if file.endswith(('.py', '.js', '.ts', '.java', '.cpp')):
                    file_path = os.path.join(root, file)
                    result = self.process_file(file_path)
                    if result:
                        results.append(result)

        return results

    def get_statistics(self) -> Dict[str, int]:
        '''Get processing statistics.'''
        total_size = 0
        for file_path in self.processed_files:
            if os.path.exists(file_path):
                total_size += os.path.getsize(file_path)

        return {
            'total_files': len(self.processed_files),
            'total_size': total_size,
            'total_errors': len(self.errors)
        }

    def clear_errors(self):
        '''Clear error log.'''
        self.errors.clear()

def create_processor(base_dir: str = ".") -> FileProcessor:
    '''Factory function to create file processor.'''
    return FileProcessor(base_dir)

def main():
    '''Main execution function.'''
    processor = create_processor()
    results = processor.process_directory("./src")
    stats = processor.get_statistics()

    print(f"Processed {stats['total_files']} files")
    print(f"Total size: {stats['total_size']} bytes")

    if stats['total_errors'] > 0:
        print(f"Errors: {stats['total_errors']}")

if __name__ == "__main__":
    main()
"""

        # Test with force_agentic=True
        request = ChunkingRequest(
            content=test_code,
            language="python",
            file_path="file_processor.py",
            purpose="code_review",
            force_agentic=True,
            quality_threshold=0.7
        )

        result = await service.chunk_content(request)

        print(f"✅ Agentic chunking completed!")
        print(f"   Strategy used: {result.strategy_used.value}")
        print(f"   Chunks created: {len(result.chunks)}")
        print(f"   Quality score: {result.quality_score:.2f}")
        print(f"   Processing time: {result.processing_time:.3f}s")

        # Show chunk details
        for i, chunk in enumerate(result.chunks[:5]):  # Show first 5 chunks
            print(f"   Chunk {i+1}: {chunk.chunk_type.value} '{chunk.name}' "
                  f"({chunk.size_lines} lines, quality: {chunk.quality_score:.2f})")

        # Verify it's actually agentic
        if result.strategy_used.value == "agentic":
            print("🎉 Full agentic pipeline working with mock LLM!")
            return True
        else:
            print(f"⚠️  Expected agentic strategy, got {result.strategy_used.value}")
            return False

    except Exception as e:
        print(f"❌ Full agentic pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_chunking_service_health():
    """Test chunking service health with agentic mode."""
    print("🔍 Testing chunking service health...")

    try:
        from app.services.chunking.service import ChunkingService

        service = ChunkingService()
        await service.initialize()

        health = await service.health_check()
        print(f"✅ Service status: {health['status']}")
        print(f"   Available chunkers: {list(health.get('chunkers', {}).keys())}")

        # Check if hybrid chunker is available
        chunkers = health.get('chunkers', {})
        if 'hybrid' in chunkers:
            hybrid_status = chunkers['hybrid']['status']
            print(f"   Hybrid chunker: {hybrid_status}")

            if hybrid_status in ['healthy', 'degraded']:
                print("✅ Agentic chunking is available!")
                return True

        print("⚠️  Agentic chunking not fully available")
        return False

    except Exception as e:
        print(f"❌ Service health check failed: {e}")
        return False

async def main():
    """Run all agentic chunking tests."""
    print("🚀 Agentic Chunking Tests with Mock LLM")
    print("=" * 50)

    tests = [
        ("Mock LLM Service", test_mock_llm_service),
        ("Chunking Agents with Mock", test_chunking_agents_with_mock),
        ("Full Agentic Pipeline", test_full_agentic_pipeline),
        ("Service Health Check", test_chunking_service_health),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        try:
            result = await test_func()
            if result:
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")

    print("\n" + "=" * 50)
    print(f"📊 Agentic Chunking Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All agentic chunking tests passed! Mock LLM integration is working.")
        return 0
    else:
        print(f"⚠️  {total - passed} tests failed. Some agentic functionality may not be working.")
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
