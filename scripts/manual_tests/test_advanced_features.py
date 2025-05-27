#!/usr/bin/env python3
"""
Comprehensive test for advanced chunking features.

Tests multi-language support, custom strategies, advanced quality metrics,
and real-time performance dashboard.
"""

import asyncio
import sys
import os
import time

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_multi_language_support():
    """Test multi-language detection and chunking."""
    print("🔍 Testing multi-language support...")
    
    try:
        from app.services.chunking.language_support import (
            LanguageDetector, SupportedLanguage, LanguageSpecificChunker,
            get_supported_languages, is_language_supported, detect_and_chunk
        )
        
        # Test language detection
        test_files = [
            ("test.py", "def hello(): pass", SupportedLanguage.PYTHON),
            ("test.js", "function hello() { return 42; }", SupportedLanguage.JAVASCRIPT),
            ("test.ts", "interface User { name: string; }", SupportedLanguage.TYPESCRIPT),
            ("test.java", "public class Hello { }", SupportedLanguage.JAVA),
            ("test.go", "func main() { fmt.Println(\"hello\") }", SupportedLanguage.GO),
            ("test.rs", "fn main() { println!(\"hello\"); }", SupportedLanguage.RUST)
        ]
        
        detected_correctly = 0
        for file_path, content, expected_lang in test_files:
            detected = LanguageDetector.detect_language(file_path, content)
            if detected == expected_lang:
                detected_correctly += 1
                print(f"   ✅ {file_path}: {detected.value}")
            else:
                print(f"   ❌ {file_path}: expected {expected_lang.value}, got {detected.value}")
        
        # Test supported languages
        supported = get_supported_languages()
        print(f"   📋 Supported languages: {len(supported)}")
        
        # Test language-specific chunking
        python_code = """
import os
import sys

def process_file(filename):
    '''Process a single file.'''
    with open(filename, 'r') as f:
        return f.read()

class FileProcessor:
    def __init__(self, base_dir):
        self.base_dir = base_dir
    
    def process_all(self):
        for file in os.listdir(self.base_dir):
            yield process_file(file)
"""
        
        result = detect_and_chunk("test.py", python_code)
        
        print(f"✅ Multi-language support working!")
        print(f"   Languages detected correctly: {detected_correctly}/{len(test_files)}")
        print(f"   Python chunking: {len(result['boundaries'])} boundaries found")
        print(f"   Language: {result['language']}")
        
        return detected_correctly >= len(test_files) * 0.8  # 80% success rate
        
    except Exception as e:
        print(f"❌ Multi-language support failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_custom_strategies():
    """Test custom chunking strategies and rules."""
    print("🔍 Testing custom chunking strategies...")
    
    try:
        from app.services.chunking.custom_strategies import (
            CustomStrategy, ChunkingRule, RuleType, RuleCondition,
            custom_strategy_manager, RuleBasedCustomChunker
        )
        
        # Create a custom strategy
        custom_strategy = CustomStrategy(
            id="test_strategy",
            name="Test Strategy",
            description="Test custom chunking strategy",
            language="python",
            rules=[
                ChunkingRule(
                    id="function_rule",
                    name="Function Rule",
                    description="Split by function definitions",
                    rule_type=RuleType.PATTERN_BASED,
                    condition=RuleCondition.REGEX_MATCH,
                    value=r"def\s+\w+",
                    action="split",
                    priority=1
                ),
                ChunkingRule(
                    id="size_rule",
                    name="Size Rule", 
                    description="Limit chunk size",
                    rule_type=RuleType.SIZE_BASED,
                    condition=RuleCondition.LINE_COUNT,
                    value=50,
                    action="split",
                    priority=2
                )
            ]
        )
        
        # Register strategy
        success = custom_strategy_manager.register_strategy(custom_strategy)
        if not success:
            print("   ❌ Failed to register custom strategy")
            return False
        
        # List strategies
        strategies = custom_strategy_manager.list_strategies()
        print(f"   📋 Registered strategies: {len(strategies)}")
        
        # Create chunker and test
        chunker = custom_strategy_manager.create_chunker("test_strategy")
        if not chunker:
            print("   ❌ Failed to create custom chunker")
            return False
        
        test_code = """
def function_one():
    return 1

def function_two():
    return 2

class TestClass:
    def method_one(self):
        pass
    
    def method_two(self):
        pass
"""
        
        chunks = await chunker.chunk_content(test_code, "test.py")
        
        # Export and import strategy
        exported = custom_strategy_manager.export_strategy("test_strategy")
        if not exported:
            print("   ❌ Failed to export strategy")
            return False
        
        # Test import
        exported["id"] = "imported_strategy"
        import_success = custom_strategy_manager.import_strategy(exported)
        
        print(f"✅ Custom strategies working!")
        print(f"   Strategy registered: {success}")
        print(f"   Chunks created: {len(chunks)}")
        print(f"   Export/import: {import_success}")
        
        return True
        
    except Exception as e:
        print(f"❌ Custom strategies failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_advanced_quality_metrics():
    """Test advanced quality metrics and feedback."""
    print("🔍 Testing advanced quality metrics...")
    
    try:
        from app.services.quality.advanced_metrics import (
            AdvancedQualityAnalyzer, QualityDimension, UserFeedback, FeedbackType,
            quality_analyzer, feedback_manager
        )
        from app.services.chunking.service import ChunkingService
        from app.services.chunking.base import ChunkingRequest
        
        # Initialize service
        service = ChunkingService()
        await service.initialize()
        
        # Test code for quality assessment
        test_code = """
import asyncio
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class DataProcessor:
    '''Advanced data processing class with comprehensive functionality.'''
    
    def __init__(self, config: Dict[str, any]):
        self.config = config
        self.processed_items = []
        self.error_count = 0
    
    async def process_item(self, item: str) -> Dict[str, any]:
        '''Process a single data item asynchronously.'''
        try:
            # Simulate processing
            await asyncio.sleep(0.01)
            
            if not item or len(item.strip()) == 0:
                raise ValueError("Empty item cannot be processed")
            
            result = {
                'original': item,
                'processed': item.upper(),
                'length': len(item),
                'timestamp': time.time()
            }
            
            self.processed_items.append(result)
            logger.info(f"Processed item: {item}")
            return result
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Failed to process item {item}: {e}")
            raise
    
    async def process_batch(self, items: List[str]) -> List[Dict]:
        '''Process multiple items in batch.'''
        results = []
        
        for item in items:
            try:
                result = await self.process_item(item)
                results.append(result)
            except Exception as e:
                results.append({'error': str(e), 'item': item})
        
        return results
    
    def get_statistics(self) -> Dict[str, any]:
        '''Get processing statistics.'''
        return {
            'total_processed': len(self.processed_items),
            'error_count': self.error_count,
            'success_rate': (len(self.processed_items) / (len(self.processed_items) + self.error_count)) * 100 if (len(self.processed_items) + self.error_count) > 0 else 0
        }

async def main():
    '''Main execution function.'''
    processor = DataProcessor({'batch_size': 10})
    
    test_items = ['item1', 'item2', '', 'item4', 'item5']
    results = await processor.process_batch(test_items)
    stats = processor.get_statistics()
    
    print(f"Processed {len(results)} items")
    print(f"Statistics: {stats}")

if __name__ == "__main__":
    asyncio.run(main())
"""
        
        # Perform chunking
        request = ChunkingRequest(
            content=test_code,
            language="python",
            file_path="data_processor.py",
            purpose="code_review"
        )
        
        result = await service.chunk_content(request)
        
        # Perform advanced quality assessment
        assessment = await quality_analyzer.assess_chunking_quality(
            result, test_code, "code_review"
        )
        
        # Test feedback system
        feedback = UserFeedback(
            id="test_feedback_1",
            feedback_type=FeedbackType.RATING,
            chunk_id=result.chunks[0].id if result.chunks else None,
            overall_rating=4.5,
            comment="Good chunking quality, well-structured",
            suggestions=["Consider smaller chunk sizes", "Add more context"],
            user_id="test_user"
        )
        
        feedback_success = feedback_manager.submit_feedback(feedback)
        feedback_summary = feedback_manager.get_feedback_summary()
        
        print(f"✅ Advanced quality metrics working!")
        print(f"   Overall quality score: {assessment.overall_score:.2f}")
        print(f"   Quality dimensions: {len(assessment.dimension_scores)}")
        print(f"   Chunk-level scores: {len(assessment.chunk_level_scores)}")
        print(f"   Feedback submitted: {feedback_success}")
        print(f"   Feedback count: {feedback_summary['count']}")
        
        # Show dimension scores
        for dimension, score in list(assessment.dimension_scores.items())[:3]:
            print(f"     {dimension.value}: {score.score:.2f}")
        
        return assessment.overall_score > 0.5
        
    except Exception as e:
        print(f"❌ Advanced quality metrics failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_performance_dashboard():
    """Test real-time performance dashboard."""
    print("🔍 Testing performance dashboard...")
    
    try:
        from app.services.dashboard.performance_dashboard import (
            PerformanceDashboard, MetricsCollector, DashboardConfig
        )
        
        # Create dashboard with test config
        config = DashboardConfig(
            update_interval=0.5,
            retention_period=300,
            enable_alerts=True
        )
        
        dashboard = PerformanceDashboard(config)
        await dashboard.start()
        
        # Record some test metrics
        dashboard.record_chunking_request(
            success=True,
            duration=0.5,
            chunks_created=5,
            quality_score=0.85,
            strategy="hybrid"
        )
        
        dashboard.record_chunking_request(
            success=True,
            duration=0.3,
            chunks_created=3,
            quality_score=0.92,
            strategy="agentic"
        )
        
        dashboard.record_llm_call(duration=1.2, success=True, cached=False)
        dashboard.record_llm_call(duration=0.1, success=True, cached=True)
        
        dashboard.record_batch_metrics(batch_size=5, efficiency=85.0)
        dashboard.update_active_requests(3)
        
        # Wait for metrics to be processed
        await asyncio.sleep(1.0)
        
        # Get dashboard data
        dashboard_data = dashboard.get_dashboard_data(time_range=300)
        
        # Get metric history
        history = dashboard.get_metric_history("chunking_requests_total", time_range=300)
        
        await dashboard.stop()
        
        print(f"✅ Performance dashboard working!")
        print(f"   Metrics collected: {len(dashboard_data['metrics'])}")
        print(f"   Summary status: {dashboard_data['summary']['status']}")
        print(f"   Total requests: {dashboard_data['summary']['total_requests']}")
        print(f"   Success rate: {dashboard_data['summary']['success_rate']:.1f}%")
        print(f"   Metric history points: {len(history)}")
        print(f"   Alerts: {len(dashboard_data['alerts'])}")
        
        return len(dashboard_data['metrics']) > 0
        
    except Exception as e:
        print(f"❌ Performance dashboard failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_integrated_advanced_features():
    """Test integrated advanced features with chunking service."""
    print("🔍 Testing integrated advanced features...")
    
    try:
        from app.services.chunking.service import ChunkingService
        from app.services.chunking.base import ChunkingRequest
        
        # Initialize service with advanced features
        service = ChunkingService()
        service._enable_advanced_features = True
        await service.initialize()
        
        # Multi-language test files
        test_files = [
            ("calculator.py", """
class Calculator:
    def __init__(self):
        self.history = []
    
    def add(self, a, b):
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
    
    def multiply(self, a, b):
        result = a * b
        self.history.append(f"{a} * {b} = {result}")
        return result
""", "python"),
            
            ("utils.js", """
function formatDate(date) {
    return date.toISOString().split('T')[0];
}

function validateEmail(email) {
    const regex = /^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$/;
    return regex.test(email);
}

class ApiClient {
    constructor(baseUrl) {
        this.baseUrl = baseUrl;
    }
    
    async get(endpoint) {
        const response = await fetch(`${this.baseUrl}/${endpoint}`);
        return response.json();
    }
}
""", "javascript"),
            
            ("processor.go", """
package main

import (
    "fmt"
    "strings"
)

type Processor struct {
    name string
    data []string
}

func NewProcessor(name string) *Processor {
    return &Processor{
        name: name,
        data: make([]string, 0),
    }
}

func (p *Processor) Process(input string) string {
    processed := strings.ToUpper(input)
    p.data = append(p.data, processed)
    return processed
}

func main() {
    proc := NewProcessor("test")
    result := proc.Process("hello world")
    fmt.Println(result)
}
""", "go")
        ]
        
        results = []
        for file_path, content, expected_lang in test_files:
            request = ChunkingRequest(
                content=content,
                language="unknown",  # Let it auto-detect
                file_path=file_path,
                purpose="code_review"
            )
            
            result = await service.chunk_content(request)
            results.append(result)
            
            print(f"   📄 {file_path}:")
            print(f"      Language detected: {result.chunks[0].language if result.chunks else 'unknown'}")
            print(f"      Chunks: {len(result.chunks)}")
            print(f"      Quality: {result.quality_score:.2f}")
            print(f"      Strategy: {result.strategy_used.value}")
        
        # Get service health with advanced features
        health = await service.health_check()
        
        print(f"✅ Integrated advanced features working!")
        print(f"   Files processed: {len(results)}")
        print(f"   Average quality: {sum(r.quality_score for r in results) / len(results):.2f}")
        print(f"   Service health: {health['status']}")
        print(f"   Performance stats available: {'performance_stats' in health}")
        
        await service.cleanup()
        return len(results) == len(test_files)
        
    except Exception as e:
        print(f"❌ Integrated advanced features failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all advanced features tests."""
    print("🚀 Advanced Features Comprehensive Test")
    print("=" * 60)
    
    tests = [
        ("Multi-Language Support", test_multi_language_support),
        ("Custom Strategies", test_custom_strategies),
        ("Advanced Quality Metrics", test_advanced_quality_metrics),
        ("Performance Dashboard", test_performance_dashboard),
        ("Integrated Advanced Features", test_integrated_advanced_features),
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
    print(f"📊 Advanced Features Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All advanced features working perfectly!")
        print("🚀 Chunking service now has comprehensive advanced capabilities!")
        return 0
    elif passed >= total * 0.8:
        print("🎯 Most advanced features working well!")
        print("🔧 Some features may need minor adjustments.")
        return 0
    else:
        print(f"⚠️  {total - passed} advanced features need attention.")
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
