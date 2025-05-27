"""
Real-time performance dashboard for chunking service.

Provides live monitoring, metrics visualization, and performance insights.
"""

import asyncio
import time
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import deque
from enum import Enum

import structlog

logger = structlog.get_logger(__name__)


class MetricType(Enum):
    """Types of metrics tracked."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class MetricPoint:
    """Single metric data point."""
    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class Metric:
    """Metric definition and data."""
    name: str
    metric_type: MetricType
    description: str
    unit: str = ""
    data_points: deque = field(default_factory=lambda: deque(maxlen=1000))
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class DashboardConfig:
    """Configuration for dashboard."""
    update_interval: float = 1.0  # seconds
    retention_period: int = 3600  # seconds
    max_data_points: int = 1000
    enable_alerts: bool = True
    alert_thresholds: Dict[str, float] = field(default_factory=dict)


class MetricsCollector:
    """Collects and stores performance metrics."""
    
    def __init__(self, config: DashboardConfig = None):
        self.config = config or DashboardConfig()
        self.metrics: Dict[str, Metric] = {}
        self.logger = structlog.get_logger("metrics_collector")
        self._running = False
        self._cleanup_task: Optional[asyncio.Task] = None
    
    def register_metric(self, metric: Metric) -> None:
        """Register a new metric."""
        self.metrics[metric.name] = metric
        self.logger.debug("Metric registered", name=metric.name, type=metric.metric_type.value)
    
    def record_counter(self, name: str, value: float = 1.0, labels: Dict[str, str] = None) -> None:
        """Record a counter metric."""
        if name not in self.metrics:
            self.register_metric(Metric(
                name=name,
                metric_type=MetricType.COUNTER,
                description=f"Counter metric: {name}"
            ))
        
        metric = self.metrics[name]
        point = MetricPoint(
            timestamp=datetime.now(),
            value=value,
            labels=labels or {}
        )
        metric.data_points.append(point)
    
    def record_gauge(self, name: str, value: float, labels: Dict[str, str] = None) -> None:
        """Record a gauge metric."""
        if name not in self.metrics:
            self.register_metric(Metric(
                name=name,
                metric_type=MetricType.GAUGE,
                description=f"Gauge metric: {name}"
            ))
        
        metric = self.metrics[name]
        point = MetricPoint(
            timestamp=datetime.now(),
            value=value,
            labels=labels or {}
        )
        metric.data_points.append(point)
    
    def record_histogram(self, name: str, value: float, labels: Dict[str, str] = None) -> None:
        """Record a histogram metric."""
        if name not in self.metrics:
            self.register_metric(Metric(
                name=name,
                metric_type=MetricType.HISTOGRAM,
                description=f"Histogram metric: {name}"
            ))
        
        metric = self.metrics[name]
        point = MetricPoint(
            timestamp=datetime.now(),
            value=value,
            labels=labels or {}
        )
        metric.data_points.append(point)
    
    def record_timer(self, name: str, duration: float, labels: Dict[str, str] = None) -> None:
        """Record a timer metric."""
        if name not in self.metrics:
            self.register_metric(Metric(
                name=name,
                metric_type=MetricType.TIMER,
                description=f"Timer metric: {name}",
                unit="seconds"
            ))
        
        metric = self.metrics[name]
        point = MetricPoint(
            timestamp=datetime.now(),
            value=duration,
            labels=labels or {}
        )
        metric.data_points.append(point)
    
    def get_metric_data(self, name: str, since: Optional[datetime] = None) -> List[MetricPoint]:
        """Get metric data points."""
        if name not in self.metrics:
            return []
        
        metric = self.metrics[name]
        if since is None:
            return list(metric.data_points)
        
        return [point for point in metric.data_points if point.timestamp >= since]
    
    def get_metric_summary(self, name: str, since: Optional[datetime] = None) -> Dict[str, Any]:
        """Get metric summary statistics."""
        data_points = self.get_metric_data(name, since)
        
        if not data_points:
            return {"count": 0}
        
        values = [point.value for point in data_points]
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "latest": values[-1],
            "first_timestamp": data_points[0].timestamp.isoformat(),
            "latest_timestamp": data_points[-1].timestamp.isoformat()
        }
    
    async def start_cleanup_task(self) -> None:
        """Start background cleanup task."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._running = True
            self._cleanup_task = asyncio.create_task(self._cleanup_old_data())
    
    async def stop_cleanup_task(self) -> None:
        """Stop background cleanup task."""
        self._running = False
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
    
    async def _cleanup_old_data(self) -> None:
        """Background task to cleanup old metric data."""
        while self._running:
            try:
                cutoff_time = datetime.now() - timedelta(seconds=self.config.retention_period)
                
                for metric in self.metrics.values():
                    # Remove old data points
                    while metric.data_points and metric.data_points[0].timestamp < cutoff_time:
                        metric.data_points.popleft()
                
                await asyncio.sleep(60)  # Cleanup every minute
                
            except Exception as e:
                self.logger.error("Error in cleanup task", error=str(e))
                await asyncio.sleep(60)


class PerformanceDashboard:
    """Real-time performance dashboard."""
    
    def __init__(self, config: DashboardConfig = None):
        self.config = config or DashboardConfig()
        self.metrics_collector = MetricsCollector(config)
        self.logger = structlog.get_logger("performance_dashboard")
        self._update_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Initialize core metrics
        self._initialize_core_metrics()
    
    def _initialize_core_metrics(self) -> None:
        """Initialize core performance metrics."""
        core_metrics = [
            Metric("chunking_requests_total", MetricType.COUNTER, "Total chunking requests"),
            Metric("chunking_requests_success", MetricType.COUNTER, "Successful chunking requests"),
            Metric("chunking_requests_failed", MetricType.COUNTER, "Failed chunking requests"),
            Metric("chunking_duration_seconds", MetricType.HISTOGRAM, "Chunking duration", "seconds"),
            Metric("chunks_created_total", MetricType.COUNTER, "Total chunks created"),
            Metric("quality_score", MetricType.GAUGE, "Quality score", "score"),
            Metric("cache_hit_rate", MetricType.GAUGE, "Cache hit rate", "percentage"),
            Metric("memory_usage_mb", MetricType.GAUGE, "Memory usage", "MB"),
            Metric("active_requests", MetricType.GAUGE, "Active requests", "count"),
            Metric("llm_api_calls", MetricType.COUNTER, "LLM API calls"),
            Metric("llm_response_time", MetricType.HISTOGRAM, "LLM response time", "seconds"),
            Metric("batch_efficiency", MetricType.GAUGE, "Batch efficiency", "percentage")
        ]
        
        for metric in core_metrics:
            self.metrics_collector.register_metric(metric)
    
    async def start(self) -> None:
        """Start the dashboard."""
        self._running = True
        await self.metrics_collector.start_cleanup_task()
        
        if self._update_task is None or self._update_task.done():
            self._update_task = asyncio.create_task(self._update_loop())
        
        self.logger.info("Performance dashboard started")
    
    async def stop(self) -> None:
        """Stop the dashboard."""
        self._running = False
        
        if self._update_task and not self._update_task.done():
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
        
        await self.metrics_collector.stop_cleanup_task()
        self.logger.info("Performance dashboard stopped")
    
    async def _update_loop(self) -> None:
        """Main update loop for dashboard."""
        while self._running:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(self.config.update_interval)
            except Exception as e:
                self.logger.error("Error in dashboard update loop", error=str(e))
                await asyncio.sleep(self.config.update_interval)
    
    async def _collect_system_metrics(self) -> None:
        """Collect system-level metrics."""
        try:
            # Memory usage
            try:
                import psutil
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                self.metrics_collector.record_gauge("memory_usage_mb", memory_mb)
            except ImportError:
                pass
            
            # Cache metrics (if available)
            try:
                from app.services.cache.llm_cache import get_llm_cache
                cache = get_llm_cache()
                stats = cache.get_stats()
                self.metrics_collector.record_gauge("cache_hit_rate", stats.get("hit_rate_percent", 0))
            except ImportError:
                pass
            
        except Exception as e:
            self.logger.error("Error collecting system metrics", error=str(e))
    
    # Metric recording methods for chunking service
    def record_chunking_request(self, success: bool, duration: float, chunks_created: int, 
                               quality_score: float, strategy: str) -> None:
        """Record chunking request metrics."""
        self.metrics_collector.record_counter("chunking_requests_total")
        
        if success:
            self.metrics_collector.record_counter("chunking_requests_success")
            self.metrics_collector.record_histogram("chunking_duration_seconds", duration)
            self.metrics_collector.record_counter("chunks_created_total", chunks_created)
            self.metrics_collector.record_gauge("quality_score", quality_score)
        else:
            self.metrics_collector.record_counter("chunking_requests_failed")
        
        # Record with strategy label
        self.metrics_collector.record_counter(
            "chunking_requests_by_strategy",
            labels={"strategy": strategy}
        )
    
    def record_llm_call(self, duration: float, success: bool, cached: bool = False) -> None:
        """Record LLM API call metrics."""
        self.metrics_collector.record_counter("llm_api_calls")
        
        if success:
            self.metrics_collector.record_histogram("llm_response_time", duration)
        
        labels = {"cached": str(cached).lower(), "success": str(success).lower()}
        self.metrics_collector.record_counter("llm_calls_detailed", labels=labels)
    
    def record_batch_metrics(self, batch_size: int, efficiency: float) -> None:
        """Record batching metrics."""
        self.metrics_collector.record_gauge("batch_efficiency", efficiency)
        self.metrics_collector.record_histogram("batch_size", batch_size)
    
    def update_active_requests(self, count: int) -> None:
        """Update active request count."""
        self.metrics_collector.record_gauge("active_requests", count)
    
    def get_dashboard_data(self, time_range: Optional[int] = None) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        since = None
        if time_range:
            since = datetime.now() - timedelta(seconds=time_range)
        
        dashboard_data = {
            "timestamp": datetime.now().isoformat(),
            "time_range_seconds": time_range,
            "metrics": {},
            "summary": {},
            "alerts": []
        }
        
        # Collect metric summaries
        for metric_name in self.metrics_collector.metrics:
            summary = self.metrics_collector.get_metric_summary(metric_name, since)
            dashboard_data["metrics"][metric_name] = summary
        
        # Generate overall summary
        dashboard_data["summary"] = self._generate_summary(dashboard_data["metrics"])
        
        # Check for alerts
        if self.config.enable_alerts:
            dashboard_data["alerts"] = self._check_alerts(dashboard_data["metrics"])
        
        return dashboard_data
    
    def _generate_summary(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall performance summary."""
        summary = {
            "status": "healthy",
            "total_requests": metrics.get("chunking_requests_total", {}).get("count", 0),
            "success_rate": 0.0,
            "avg_duration": 0.0,
            "avg_quality": 0.0,
            "cache_hit_rate": 0.0
        }
        
        # Calculate success rate
        total_requests = metrics.get("chunking_requests_total", {}).get("count", 0)
        successful_requests = metrics.get("chunking_requests_success", {}).get("count", 0)
        
        if total_requests > 0:
            summary["success_rate"] = (successful_requests / total_requests) * 100
        
        # Average duration
        duration_metrics = metrics.get("chunking_duration_seconds", {})
        if "avg" in duration_metrics:
            summary["avg_duration"] = duration_metrics["avg"]
        
        # Average quality
        quality_metrics = metrics.get("quality_score", {})
        if "avg" in quality_metrics:
            summary["avg_quality"] = quality_metrics["avg"]
        
        # Cache hit rate
        cache_metrics = metrics.get("cache_hit_rate", {})
        if "latest" in cache_metrics:
            summary["cache_hit_rate"] = cache_metrics["latest"]
        
        # Determine overall status
        if summary["success_rate"] < 95:
            summary["status"] = "degraded"
        elif summary["success_rate"] < 80:
            summary["status"] = "unhealthy"
        
        return summary
    
    def _check_alerts(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for alert conditions."""
        alerts = []
        
        # High error rate
        total_requests = metrics.get("chunking_requests_total", {}).get("count", 0)
        failed_requests = metrics.get("chunking_requests_failed", {}).get("count", 0)
        
        if total_requests > 10:  # Only alert if we have enough data
            error_rate = (failed_requests / total_requests) * 100
            if error_rate > 10:
                alerts.append({
                    "level": "warning" if error_rate < 20 else "critical",
                    "message": f"High error rate: {error_rate:.1f}%",
                    "metric": "error_rate",
                    "value": error_rate
                })
        
        # High response time
        duration_metrics = metrics.get("chunking_duration_seconds", {})
        if "avg" in duration_metrics and duration_metrics["avg"] > 10:
            alerts.append({
                "level": "warning",
                "message": f"High average response time: {duration_metrics['avg']:.2f}s",
                "metric": "response_time",
                "value": duration_metrics["avg"]
            })
        
        # Low cache hit rate
        cache_metrics = metrics.get("cache_hit_rate", {})
        if "latest" in cache_metrics and cache_metrics["latest"] < 30:
            alerts.append({
                "level": "info",
                "message": f"Low cache hit rate: {cache_metrics['latest']:.1f}%",
                "metric": "cache_hit_rate",
                "value": cache_metrics["latest"]
            })
        
        return alerts
    
    def get_metric_history(self, metric_name: str, time_range: int = 3600) -> List[Dict[str, Any]]:
        """Get metric history for visualization."""
        since = datetime.now() - timedelta(seconds=time_range)
        data_points = self.metrics_collector.get_metric_data(metric_name, since)
        
        return [
            {
                "timestamp": point.timestamp.isoformat(),
                "value": point.value,
                "labels": point.labels
            }
            for point in data_points
        ]


# Global dashboard instance
performance_dashboard = PerformanceDashboard()
