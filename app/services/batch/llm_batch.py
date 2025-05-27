"""
LLM request batching service for performance optimization.

Provides intelligent batching of LLM requests to reduce latency and improve throughput.
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import structlog

from app.services.llm.base import LLMMessage, LLMResponse

logger = structlog.get_logger(__name__)


class BatchStrategy(Enum):
    """Batching strategies."""
    TIME_BASED = "time_based"  # Batch by time window
    SIZE_BASED = "size_based"  # Batch by request count
    ADAPTIVE = "adaptive"      # Adaptive batching based on load


@dataclass
class BatchRequest:
    """Individual request in a batch."""
    id: str
    messages: List[LLMMessage]
    kwargs: Dict[str, Any]
    future: asyncio.Future
    created_at: datetime = field(default_factory=datetime.now)
    priority: int = 0  # Higher number = higher priority


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    max_batch_size: int = 10
    max_wait_time: float = 2.0  # seconds
    strategy: BatchStrategy = BatchStrategy.ADAPTIVE
    enable_priority: bool = True
    max_concurrent_batches: int = 3
    concurrent_requests: int = 1  # For compatibility


class LLMBatcher:
    """
    High-performance LLM request batcher with adaptive strategies.
    """

    def __init__(self, llm_service, config: BatchConfig = None):
        self.llm_service = llm_service
        self.config = config or BatchConfig()

        # Batch management
        self._pending_requests: List[BatchRequest] = []
        self._batch_lock = asyncio.Lock()
        self._batch_task: Optional[asyncio.Task] = None
        self._shutdown = False

        # Performance tracking
        self._stats = {
            "total_requests": 0,
            "batched_requests": 0,
            "individual_requests": 0,
            "total_batches": 0,
            "avg_batch_size": 0.0,
            "avg_wait_time": 0.0,
            "throughput_rps": 0.0
        }

        # Adaptive batching state
        self._recent_latencies: List[float] = []
        self._load_factor = 0.0

        logger.info("LLM batcher initialized", config=self.config)

        # Start batch processing task
        self._start_batch_processor()

    def _start_batch_processor(self):
        """Start the background batch processing task."""
        if self._batch_task is None or self._batch_task.done():
            self._batch_task = asyncio.create_task(self._batch_processor())

    async def _batch_processor(self):
        """Background task that processes batches."""
        logger.info("Batch processor started")

        while not self._shutdown:
            try:
                await self._process_pending_batches()
                await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
            except Exception as e:
                logger.error("Error in batch processor", error=str(e))
                await asyncio.sleep(1.0)  # Longer delay on error

    async def _process_pending_batches(self):
        """Process pending requests into batches."""
        async with self._batch_lock:
            if not self._pending_requests:
                return

            # Determine batch size based on strategy
            batch_size = self._calculate_optimal_batch_size()

            # Check if we should process a batch
            should_process = self._should_process_batch(batch_size)

            if should_process:
                # Extract batch requests
                batch_requests = self._extract_batch(batch_size)

                if batch_requests:
                    # Process batch in background
                    asyncio.create_task(self._execute_batch(batch_requests))

    def _calculate_optimal_batch_size(self) -> int:
        """Calculate optimal batch size based on strategy."""
        if self.config.strategy == BatchStrategy.SIZE_BASED:
            return min(len(self._pending_requests), self.config.max_batch_size)

        elif self.config.strategy == BatchStrategy.TIME_BASED:
            # Use max batch size for time-based
            return min(len(self._pending_requests), self.config.max_batch_size)

        elif self.config.strategy == BatchStrategy.ADAPTIVE:
            # Adaptive sizing based on load and latency
            base_size = min(len(self._pending_requests), self.config.max_batch_size)

            # Adjust based on recent performance
            if self._recent_latencies:
                avg_latency = sum(self._recent_latencies) / len(self._recent_latencies)

                # If latency is high, use smaller batches
                if avg_latency > 5.0:  # 5 seconds
                    return max(1, base_size // 2)
                # If latency is low, can use larger batches
                elif avg_latency < 1.0:  # 1 second
                    return base_size

            return base_size

        return 1

    def _should_process_batch(self, batch_size: int) -> bool:
        """Determine if we should process a batch now."""
        if not self._pending_requests:
            return False

        # Always process if we have max batch size
        if len(self._pending_requests) >= self.config.max_batch_size:
            return True

        # Check oldest request wait time
        oldest_request = min(self._pending_requests, key=lambda r: r.created_at)
        wait_time = (datetime.now() - oldest_request.created_at).total_seconds()

        if wait_time >= self.config.max_wait_time:
            return True

        # For adaptive strategy, consider load
        if self.config.strategy == BatchStrategy.ADAPTIVE:
            # Process smaller batches under high load
            if self._load_factor > 0.8 and len(self._pending_requests) >= 2:
                return True

        return False

    def _extract_batch(self, max_size: int) -> List[BatchRequest]:
        """Extract requests for batch processing."""
        if not self._pending_requests:
            return []

        # Sort by priority if enabled
        if self.config.enable_priority:
            self._pending_requests.sort(key=lambda r: (-r.priority, r.created_at))

        # Extract batch
        batch_size = min(max_size, len(self._pending_requests))
        batch_requests = self._pending_requests[:batch_size]
        self._pending_requests = self._pending_requests[batch_size:]

        return batch_requests

    async def _execute_batch(self, batch_requests: List[BatchRequest]):
        """Execute a batch of requests."""
        if not batch_requests:
            return

        batch_start = time.time()
        batch_id = f"batch_{int(batch_start)}"

        logger.debug(
            "Executing batch",
            batch_id=batch_id,
            batch_size=len(batch_requests),
            request_ids=[req.id for req in batch_requests]
        )

        try:
            # Check if we can actually batch these requests
            if self._can_batch_requests(batch_requests):
                # Execute as true batch
                await self._execute_true_batch(batch_requests, batch_id)
            else:
                # Execute individually but concurrently
                await self._execute_concurrent_individual(batch_requests, batch_id)

            # Update performance stats
            batch_duration = time.time() - batch_start
            self._update_batch_stats(len(batch_requests), batch_duration)

        except Exception as e:
            logger.error("Batch execution failed", batch_id=batch_id, error=str(e))

            # Fail all requests in batch
            for request in batch_requests:
                if not request.future.done():
                    request.future.set_exception(e)

    def _can_batch_requests(self, requests: List[BatchRequest]) -> bool:
        """Check if requests can be batched together."""
        if len(requests) <= 1:
            return False

        # For now, we'll execute individually but concurrently
        # True batching would require LLM service support for multiple requests
        return False

    async def _execute_true_batch(self, requests: List[BatchRequest], batch_id: str):
        """Execute requests as a true batch (if supported by LLM service)."""
        # This would be implemented when LLM service supports batching
        # For now, fall back to concurrent individual execution
        await self._execute_concurrent_individual(requests, batch_id)

    async def _execute_concurrent_individual(self, requests: List[BatchRequest], batch_id: str):
        """Execute requests individually but concurrently."""
        async def execute_single(request: BatchRequest):
            try:
                start_time = time.time()
                response = await self.llm_service.generate(request.messages, **request.kwargs)
                duration = time.time() - start_time

                # Track latency for adaptive batching
                self._recent_latencies.append(duration)
                if len(self._recent_latencies) > 10:
                    self._recent_latencies.pop(0)

                request.future.set_result(response)

                logger.debug(
                    "Request completed",
                    request_id=request.id,
                    batch_id=batch_id,
                    duration=duration
                )

            except Exception as e:
                logger.error("Request failed", request_id=request.id, error=str(e))
                request.future.set_exception(e)

        # Execute all requests concurrently
        await asyncio.gather(*[execute_single(req) for req in requests], return_exceptions=True)

    def _update_batch_stats(self, batch_size: int, duration: float):
        """Update batch processing statistics."""
        self._stats["total_batches"] += 1
        self._stats["batched_requests"] += batch_size

        # Update averages
        total_batches = self._stats["total_batches"]
        self._stats["avg_batch_size"] = (
            (self._stats["avg_batch_size"] * (total_batches - 1) + batch_size) / total_batches
        )
        self._stats["avg_wait_time"] = (
            (self._stats["avg_wait_time"] * (total_batches - 1) + duration) / total_batches
        )

        # Update load factor (simple moving average)
        current_load = len(self._pending_requests) / self.config.max_batch_size
        self._load_factor = 0.9 * self._load_factor + 0.1 * current_load

    async def generate(
        self,
        messages: List[LLMMessage],
        priority: int = 0,
        request_id: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate LLM response with batching.

        Args:
            messages: Messages to send
            priority: Request priority (higher = more important)
            request_id: Optional request identifier
            **kwargs: Additional generation parameters

        Returns:
            LLM response
        """
        if self._shutdown:
            raise RuntimeError("Batcher is shutting down")

        # Generate request ID if not provided
        if request_id is None:
            request_id = f"req_{int(time.time() * 1000)}"

        # Create future for response
        future = asyncio.Future()

        # Create batch request
        batch_request = BatchRequest(
            id=request_id,
            messages=messages,
            kwargs=kwargs,
            future=future,
            priority=priority
        )

        # Add to pending requests
        async with self._batch_lock:
            self._pending_requests.append(batch_request)
            self._stats["total_requests"] += 1

        logger.debug(
            "Request queued for batching",
            request_id=request_id,
            priority=priority,
            queue_size=len(self._pending_requests)
        )

        # Wait for response
        return await future

    async def generate_individual(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        """
        Generate LLM response without batching (bypass batch queue).

        Args:
            messages: Messages to send
            **kwargs: Additional generation parameters

        Returns:
            LLM response
        """
        self._stats["individual_requests"] += 1
        return await self.llm_service.generate(messages, **kwargs)

    def get_stats(self) -> Dict[str, Any]:
        """Get batching performance statistics."""
        total_requests = self._stats["total_requests"] + self._stats["individual_requests"]
        batch_efficiency = (
            self._stats["batched_requests"] / total_requests * 100
            if total_requests > 0 else 0
        )

        return {
            "total_requests": total_requests,
            "batched_requests": self._stats["batched_requests"],
            "individual_requests": self._stats["individual_requests"],
            "total_batches": self._stats["total_batches"],
            "avg_batch_size": round(self._stats["avg_batch_size"], 2),
            "avg_batch_duration": round(self._stats["avg_wait_time"], 3),
            "batch_efficiency_percent": round(batch_efficiency, 2),
            "current_queue_size": len(self._pending_requests),
            "load_factor": round(self._load_factor, 3),
            "recent_avg_latency": (
                round(sum(self._recent_latencies) / len(self._recent_latencies), 3)
                if self._recent_latencies else 0
            )
        }

    async def flush(self, timeout: float = 10.0) -> int:
        """
        Flush all pending requests.

        Args:
            timeout: Maximum time to wait for flush

        Returns:
            Number of requests flushed
        """
        start_time = time.time()
        initial_count = len(self._pending_requests)

        while self._pending_requests and (time.time() - start_time) < timeout:
            await asyncio.sleep(0.1)

        flushed_count = initial_count - len(self._pending_requests)

        if flushed_count > 0:
            logger.info("Batch flush completed", flushed_requests=flushed_count)

        return flushed_count

    async def shutdown(self, timeout: float = 30.0):
        """
        Shutdown the batcher gracefully.

        Args:
            timeout: Maximum time to wait for shutdown
        """
        logger.info("Shutting down LLM batcher")
        self._shutdown = True

        # Flush pending requests
        await self.flush(timeout)

        # Cancel batch processor
        if self._batch_task and not self._batch_task.done():
            self._batch_task.cancel()
            try:
                await asyncio.wait_for(self._batch_task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass

        # Fail any remaining requests
        async with self._batch_lock:
            for request in self._pending_requests:
                if not request.future.done():
                    request.future.set_exception(RuntimeError("Batcher shutdown"))
            self._pending_requests.clear()

        logger.info("LLM batcher shutdown completed")


# Global batcher instance
_llm_batcher: Optional[LLMBatcher] = None


def get_llm_batcher(llm_service, config: BatchConfig = None) -> LLMBatcher:
    """Get or create global LLM batcher instance."""
    global _llm_batcher
    if _llm_batcher is None:
        _llm_batcher = LLMBatcher(llm_service, config)
    return _llm_batcher


async def batched_llm_generate(
    llm_service,
    messages: List[LLMMessage],
    use_batching: bool = True,
    priority: int = 0,
    **kwargs
) -> LLMResponse:
    """
    Generate LLM response with optional batching.

    Args:
        llm_service: LLM service instance
        messages: Messages to send
        use_batching: Whether to use batching
        priority: Request priority
        **kwargs: Additional generation parameters

    Returns:
        LLM response
    """
    if not use_batching:
        return await llm_service.generate(messages, **kwargs)

    batcher = get_llm_batcher(llm_service)
    return await batcher.generate(messages, priority=priority, **kwargs)
