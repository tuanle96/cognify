"""
Comprehensive benchmarking system for chunking performance.

Provides detailed performance analysis and optimization recommendations.
"""

import asyncio
import time
import statistics
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import structlog

from app.services.chunking.base import ChunkingRequest, ChunkingResult, ChunkingStrategy
from app.services.chunking.service import ChunkingService

logger = structlog.get_logger(__name__)


class BenchmarkType(Enum):
    """Types of benchmarks."""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    QUALITY = "quality"
    MEMORY = "memory"
    COMPREHENSIVE = "comprehensive"


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking."""
    benchmark_type: BenchmarkType = BenchmarkType.COMPREHENSIVE
    num_iterations: int = 10
    concurrent_requests: int = 1
    warmup_iterations: int = 2
    timeout_seconds: float = 60.0
    include_memory_profiling: bool = True
    strategies_to_test: List[ChunkingStrategy] = field(default_factory=lambda: [
        ChunkingStrategy.AST_FALLBACK,
        ChunkingStrategy.HYBRID,
        ChunkingStrategy.AGENTIC
    ])


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    strategy: ChunkingStrategy
    duration_seconds: float
    quality_score: float
    chunk_count: int
    memory_usage_mb: Optional[float] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkSummary:
    """Summary of benchmark results."""
    config: BenchmarkConfig
    results: List[BenchmarkResult]
    start_time: datetime
    end_time: datetime
    total_duration: float
    
    def get_stats_by_strategy(self) -> Dict[str, Dict[str, float]]:
        """Get statistics grouped by strategy."""
        stats = {}
        
        for strategy in self.config.strategies_to_test:
            strategy_results = [r for r in self.results if r.strategy == strategy and r.error is None]
            
            if strategy_results:
                durations = [r.duration_seconds for r in strategy_results]
                qualities = [r.quality_score for r in strategy_results]
                chunk_counts = [r.chunk_count for r in strategy_results]
                
                stats[strategy.value] = {
                    "avg_duration": statistics.mean(durations),
                    "min_duration": min(durations),
                    "max_duration": max(durations),
                    "std_duration": statistics.stdev(durations) if len(durations) > 1 else 0,
                    "avg_quality": statistics.mean(qualities),
                    "min_quality": min(qualities),
                    "max_quality": max(qualities),
                    "avg_chunk_count": statistics.mean(chunk_counts),
                    "success_rate": len(strategy_results) / self.config.num_iterations * 100,
                    "total_runs": len(strategy_results)
                }
        
        return stats


class ChunkingBenchmark:
    """
    Comprehensive chunking performance benchmark suite.
    """
    
    def __init__(self, chunking_service: ChunkingService):
        self.chunking_service = chunking_service
        self._memory_profiler = None
        
        # Try to import memory profiler
        try:
            import psutil
            self._psutil = psutil
        except ImportError:
            self._psutil = None
            logger.warning("psutil not available, memory profiling disabled")
    
    def _get_memory_usage(self) -> Optional[float]:
        """Get current memory usage in MB."""
        if self._psutil:
            try:
                process = self._psutil.Process()
                return process.memory_info().rss / 1024 / 1024  # Convert to MB
            except Exception:
                pass
        return None
    
    async def _run_single_benchmark(
        self,
        request: ChunkingRequest,
        strategy: ChunkingStrategy
    ) -> BenchmarkResult:
        """Run a single benchmark iteration."""
        # Force specific strategy
        original_strategy = request.force_agentic
        if strategy == ChunkingStrategy.AGENTIC:
            request.force_agentic = True
        else:
            request.force_agentic = False
        
        start_memory = self._get_memory_usage()
        start_time = time.time()
        
        try:
            # Execute chunking
            result = await self.chunking_service.chunk_content(request)
            
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            # Calculate memory delta
            memory_delta = None
            if start_memory is not None and end_memory is not None:
                memory_delta = end_memory - start_memory
            
            return BenchmarkResult(
                strategy=strategy,
                duration_seconds=end_time - start_time,
                quality_score=result.quality_score,
                chunk_count=len(result.chunks),
                memory_usage_mb=memory_delta,
                metadata={
                    "actual_strategy": result.strategy_used.value,
                    "processing_time": result.processing_time,
                    "total_lines": result.total_lines,
                    "average_chunk_size": result.average_chunk_size
                }
            )
            
        except Exception as e:
            end_time = time.time()
            logger.error("Benchmark iteration failed", strategy=strategy.value, error=str(e))
            
            return BenchmarkResult(
                strategy=strategy,
                duration_seconds=end_time - start_time,
                quality_score=0.0,
                chunk_count=0,
                error=str(e)
            )
        
        finally:
            # Restore original strategy
            request.force_agentic = original_strategy
    
    async def run_latency_benchmark(
        self,
        test_requests: List[ChunkingRequest],
        config: BenchmarkConfig
    ) -> BenchmarkSummary:
        """Run latency-focused benchmark."""
        logger.info("Starting latency benchmark", config=config)
        
        start_time = datetime.now()
        all_results = []
        
        for strategy in config.strategies_to_test:
            logger.info("Testing strategy", strategy=strategy.value)
            
            # Warmup runs
            for i in range(config.warmup_iterations):
                request = test_requests[i % len(test_requests)]
                await self._run_single_benchmark(request, strategy)
            
            # Actual benchmark runs
            for i in range(config.num_iterations):
                request = test_requests[i % len(test_requests)]
                result = await self._run_single_benchmark(request, strategy)
                all_results.append(result)
        
        end_time = datetime.now()
        
        return BenchmarkSummary(
            config=config,
            results=all_results,
            start_time=start_time,
            end_time=end_time,
            total_duration=(end_time - start_time).total_seconds()
        )
    
    async def run_throughput_benchmark(
        self,
        test_requests: List[ChunkingRequest],
        config: BenchmarkConfig
    ) -> BenchmarkSummary:
        """Run throughput-focused benchmark with concurrent requests."""
        logger.info("Starting throughput benchmark", config=config)
        
        start_time = datetime.now()
        all_results = []
        
        for strategy in config.strategies_to_test:
            logger.info("Testing strategy throughput", strategy=strategy.value)
            
            # Create concurrent tasks
            tasks = []
            for i in range(config.num_iterations):
                request = test_requests[i % len(test_requests)]
                task = self._run_single_benchmark(request, strategy)
                tasks.append(task)
                
                # Limit concurrent requests
                if len(tasks) >= config.concurrent_requests:
                    batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    for result in batch_results:
                        if isinstance(result, BenchmarkResult):
                            all_results.append(result)
                        else:
                            # Handle exceptions
                            all_results.append(BenchmarkResult(
                                strategy=strategy,
                                duration_seconds=0.0,
                                quality_score=0.0,
                                chunk_count=0,
                                error=str(result)
                            ))
                    
                    tasks = []
            
            # Process remaining tasks
            if tasks:
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                for result in batch_results:
                    if isinstance(result, BenchmarkResult):
                        all_results.append(result)
        
        end_time = datetime.now()
        
        return BenchmarkSummary(
            config=config,
            results=all_results,
            start_time=start_time,
            end_time=end_time,
            total_duration=(end_time - start_time).total_seconds()
        )
    
    async def run_quality_benchmark(
        self,
        test_requests: List[ChunkingRequest],
        config: BenchmarkConfig
    ) -> BenchmarkSummary:
        """Run quality-focused benchmark."""
        logger.info("Starting quality benchmark", config=config)
        
        # Quality benchmark focuses on fewer iterations but more detailed analysis
        quality_config = BenchmarkConfig(
            benchmark_type=BenchmarkType.QUALITY,
            num_iterations=min(config.num_iterations, 5),
            concurrent_requests=1,
            strategies_to_test=config.strategies_to_test
        )
        
        return await self.run_latency_benchmark(test_requests, quality_config)
    
    async def run_comprehensive_benchmark(
        self,
        test_requests: List[ChunkingRequest],
        config: BenchmarkConfig
    ) -> Dict[str, BenchmarkSummary]:
        """Run comprehensive benchmark covering all aspects."""
        logger.info("Starting comprehensive benchmark", config=config)
        
        results = {}
        
        # Latency benchmark
        latency_config = BenchmarkConfig(
            benchmark_type=BenchmarkType.LATENCY,
            num_iterations=config.num_iterations,
            concurrent_requests=1,
            strategies_to_test=config.strategies_to_test
        )
        results["latency"] = await self.run_latency_benchmark(test_requests, latency_config)
        
        # Throughput benchmark
        throughput_config = BenchmarkConfig(
            benchmark_type=BenchmarkType.THROUGHPUT,
            num_iterations=config.num_iterations,
            concurrent_requests=config.concurrent_requests,
            strategies_to_test=config.strategies_to_test
        )
        results["throughput"] = await self.run_throughput_benchmark(test_requests, throughput_config)
        
        # Quality benchmark
        quality_config = BenchmarkConfig(
            benchmark_type=BenchmarkType.QUALITY,
            num_iterations=min(config.num_iterations, 5),
            strategies_to_test=config.strategies_to_test
        )
        results["quality"] = await self.run_quality_benchmark(test_requests, quality_config)
        
        return results
    
    async def run_benchmark(
        self,
        test_requests: List[ChunkingRequest],
        config: BenchmarkConfig
    ) -> Dict[str, Any]:
        """Run benchmark based on configuration."""
        if config.benchmark_type == BenchmarkType.LATENCY:
            summary = await self.run_latency_benchmark(test_requests, config)
            return {"latency": summary}
        
        elif config.benchmark_type == BenchmarkType.THROUGHPUT:
            summary = await self.run_throughput_benchmark(test_requests, config)
            return {"throughput": summary}
        
        elif config.benchmark_type == BenchmarkType.QUALITY:
            summary = await self.run_quality_benchmark(test_requests, config)
            return {"quality": summary}
        
        elif config.benchmark_type == BenchmarkType.COMPREHENSIVE:
            return await self.run_comprehensive_benchmark(test_requests, config)
        
        else:
            raise ValueError(f"Unsupported benchmark type: {config.benchmark_type}")
    
    def generate_report(self, benchmark_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive benchmark report."""
        report = {
            "summary": {},
            "detailed_results": benchmark_results,
            "recommendations": [],
            "performance_comparison": {}
        }
        
        # Generate summary for each benchmark type
        for benchmark_type, summary in benchmark_results.items():
            if isinstance(summary, BenchmarkSummary):
                stats = summary.get_stats_by_strategy()
                report["summary"][benchmark_type] = stats
                
                # Find best performing strategy
                if stats:
                    best_strategy = min(stats.keys(), key=lambda s: stats[s]["avg_duration"])
                    report["performance_comparison"][benchmark_type] = {
                        "best_strategy": best_strategy,
                        "performance_gain": self._calculate_performance_gain(stats, best_strategy)
                    }
        
        # Generate recommendations
        report["recommendations"] = self._generate_recommendations(report["summary"])
        
        return report
    
    def _calculate_performance_gain(self, stats: Dict[str, Dict], best_strategy: str) -> Dict[str, float]:
        """Calculate performance gain of best strategy vs others."""
        best_duration = stats[best_strategy]["avg_duration"]
        gains = {}
        
        for strategy, strategy_stats in stats.items():
            if strategy != best_strategy:
                gain = (strategy_stats["avg_duration"] - best_duration) / best_duration * 100
                gains[strategy] = round(gain, 2)
        
        return gains
    
    def _generate_recommendations(self, summary: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations based on results."""
        recommendations = []
        
        # Analyze latency results
        if "latency" in summary:
            latency_stats = summary["latency"]
            
            # Find fastest strategy
            fastest = min(latency_stats.keys(), key=lambda s: latency_stats[s]["avg_duration"])
            recommendations.append(f"For best latency, use {fastest} strategy")
            
            # Check for high variance
            for strategy, stats in latency_stats.items():
                if stats["std_duration"] > stats["avg_duration"] * 0.5:
                    recommendations.append(f"{strategy} shows high latency variance, consider optimization")
        
        # Analyze quality results
        if "quality" in summary:
            quality_stats = summary["quality"]
            
            # Find highest quality strategy
            highest_quality = max(quality_stats.keys(), key=lambda s: quality_stats[s]["avg_quality"])
            recommendations.append(f"For best quality, use {highest_quality} strategy")
        
        # General recommendations
        if not recommendations:
            recommendations.append("Performance is optimal across all strategies")
        
        return recommendations


# Utility functions for creating test data

def create_test_requests() -> List[ChunkingRequest]:
    """Create a set of test requests for benchmarking."""
    test_codes = [
        # Simple function
        """
def hello_world():
    print("Hello, World!")
    return "success"
""",
        
        # Class with methods
        """
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
""",
        
        # Complex module
        """
import os
import sys
from typing import List, Dict, Optional

class FileProcessor:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.processed_files = []
    
    def validate_file(self, file_path: str) -> bool:
        return os.path.exists(file_path) and os.path.isfile(file_path)
    
    def process_file(self, file_path: str) -> Dict:
        if not self.validate_file(file_path):
            return {"error": "Invalid file"}
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        return {
            "path": file_path,
            "size": len(content),
            "lines": len(content.split('\\n'))
        }
    
    def process_directory(self, directory: str) -> List[Dict]:
        results = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    result = self.process_file(file_path)
                    results.append(result)
        return results

def main():
    processor = FileProcessor("./src")
    results = processor.process_directory("./src")
    print(f"Processed {len(results)} files")

if __name__ == "__main__":
    main()
"""
    ]
    
    requests = []
    for i, code in enumerate(test_codes):
        request = ChunkingRequest(
            content=code,
            language="python",
            file_path=f"test_file_{i}.py",
            purpose="general"
        )
        requests.append(request)
    
    return requests
