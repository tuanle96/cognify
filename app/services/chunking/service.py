"""
Main chunking service that orchestrates intelligent chunking operations.

Provides the primary interface for chunking requests with strategy selection,
quality evaluation, and performance monitoring.
"""

import asyncio
import time
from typing import Dict, Any, Optional

# import structlog

# Simple logger replacement
class SimpleLogger:
    def info(self, msg, **kwargs):
        print(f"INFO: {msg} {kwargs}")
    def debug(self, msg, **kwargs):
        print(f"DEBUG: {msg} {kwargs}")
    def warning(self, msg, **kwargs):
        print(f"WARNING: {msg} {kwargs}")
    def error(self, msg, **kwargs):
        print(f"ERROR: {msg} {kwargs}")

from app.core.config import get_settings
from app.core.exceptions import ChunkingError, ConfigurationError
from app.services.chunking.base import (
    BaseChunker,
    ChunkingRequest,
    ChunkingResult,
    ChunkingStrategy,
    calculate_content_hash,
)
from app.services.chunking.language_support import LanguageDetector, SupportedLanguage
from app.services.chunking.custom_strategies import custom_strategy_manager
from app.services.quality.advanced_metrics import quality_analyzer
from app.services.dashboard.performance_dashboard import performance_dashboard

logger = SimpleLogger()
settings = get_settings()


class ChunkingService:
    """
    Main chunking service that coordinates different chunking strategies.
    """

    def __init__(self):
        self._chunkers: Dict[ChunkingStrategy, BaseChunker] = {}
        self._performance_stats = {
            "total_requests": 0,
            "agentic_requests": 0,
            "ast_fallback_requests": 0,
            "cache_hits": 0,
            "average_processing_time": 0.0,
            "average_quality_score": 0.0,
        }
        self._initialized = False
        self._enable_advanced_features = True
        self._active_requests = 0

    async def initialize(self) -> None:
        """
        Initialize the chunking service and its components.
        """
        if self._initialized:
            logger.warning("Chunking service already initialized")
            return

        try:
            logger.info("Initializing chunking service")

            # Initialize chunkers based on configuration
            await self._initialize_chunkers()

            # Initialize advanced features
            if self._enable_advanced_features:
                await self._initialize_advanced_features()

            # Validate configuration
            self._validate_configuration()

            self._initialized = True
            logger.info("Chunking service initialized successfully")

        except Exception as e:
            logger.error("Failed to initialize chunking service", error=str(e))
            raise ChunkingError(
                message="Failed to initialize chunking service",
                details={"error": str(e)}
            )

    async def cleanup(self) -> None:
        """
        Cleanup chunking service resources.
        """
        if not self._initialized:
            return

        try:
            logger.info("Cleaning up chunking service")

            # Cleanup chunkers
            for chunker in self._chunkers.values():
                if hasattr(chunker, 'cleanup'):
                    await chunker.cleanup()

            self._chunkers.clear()
            self._initialized = False

            logger.info("Chunking service cleanup completed")

        except Exception as e:
            logger.error("Error during chunking service cleanup", error=str(e))

    async def chunk_content(self, request: ChunkingRequest) -> ChunkingResult:
        """
        Process a chunking request using the optimal strategy.

        Args:
            request: Chunking request

        Returns:
            ChunkingResult: Chunking results

        Raises:
            ChunkingError: If chunking fails
        """
        if not self._initialized:
            raise ChunkingError("Chunking service not initialized")

        start_time = time.time()
        content_hash = calculate_content_hash(request.content)

        # Track active requests
        self._active_requests += 1
        if self._enable_advanced_features:
            performance_dashboard.update_active_requests(self._active_requests)

        try:
            # Detect language if not specified
            if not request.language or request.language == "unknown":
                detected_language = LanguageDetector.detect_language(request.file_path, request.content)
                request.language = detected_language.value

            logger.info(
                "Processing chunking request",
                file_path=request.file_path,
                language=request.language,
                purpose=request.purpose,
                content_hash=content_hash[:12]
            )

            # Determine optimal chunking strategy
            strategy = await self._determine_strategy(request)

            # Get appropriate chunker
            chunker = self._chunkers.get(strategy)
            if not chunker:
                raise ChunkingError(
                    f"Chunker not available for strategy: {strategy.value}",
                    details={"strategy": strategy.value}
                )

            # Perform chunking
            result = await chunker.chunk(request)
            result.strategy_used = strategy

            # Advanced quality assessment
            if self._enable_advanced_features:
                try:
                    quality_assessment = await quality_analyzer.assess_chunking_quality(
                        result, request.content, request.purpose
                    )
                    result.quality_score = quality_assessment.overall_score
                    result.metadata = getattr(result, 'metadata', {})
                    result.metadata['quality_assessment'] = quality_assessment
                except Exception as e:
                    logger.warning("Advanced quality assessment failed", error=str(e))

            # Update performance statistics
            processing_time = time.time() - start_time
            result.processing_time = processing_time
            await self._update_performance_stats(result)

            # Record dashboard metrics
            if self._enable_advanced_features:
                performance_dashboard.record_chunking_request(
                    success=True,
                    duration=processing_time,
                    chunks_created=len(result.chunks),
                    quality_score=result.quality_score,
                    strategy=strategy.value
                )

            logger.info(
                "Chunking request completed",
                file_path=request.file_path,
                strategy=strategy.value,
                chunk_count=result.chunk_count,
                quality_score=result.quality_score,
                processing_time=processing_time
            )

            return result

        except Exception as e:
            processing_time = time.time() - start_time

            # Record failed request metrics
            if self._enable_advanced_features:
                performance_dashboard.record_chunking_request(
                    success=False,
                    duration=processing_time,
                    chunks_created=0,
                    quality_score=0.0,
                    strategy="failed"
                )

            logger.error(
                "Chunking request failed",
                file_path=request.file_path,
                error=str(e),
                processing_time=processing_time
            )

            if isinstance(e, ChunkingError):
                raise
            else:
                raise ChunkingError(
                    message=f"Chunking failed: {str(e)}",
                    file_path=request.file_path,
                    details={"original_error": type(e).__name__}
                )
        finally:
            # Update active request count
            self._active_requests = max(0, self._active_requests - 1)
            if self._enable_advanced_features:
                performance_dashboard.update_active_requests(self._active_requests)

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check for the chunking service.

        Returns:
            dict: Health check results
        """
        if not self._initialized:
            return {"status": "unhealthy", "reason": "Service not initialized"}

        try:
            health_results = {"status": "healthy", "chunkers": {}}

            # Check each chunker
            for strategy, chunker in self._chunkers.items():
                try:
                    chunker_health = await chunker.health_check()
                    health_results["chunkers"][strategy.value] = chunker_health
                except Exception as e:
                    health_results["chunkers"][strategy.value] = {
                        "status": "unhealthy",
                        "error": str(e)
                    }
                    health_results["status"] = "degraded"

            # Add performance statistics
            health_results["performance_stats"] = self._performance_stats.copy()

            return health_results

        except Exception as e:
            logger.error("Health check failed", error=str(e))
            return {"status": "unhealthy", "error": str(e)}

    async def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get current performance statistics.

        Returns:
            dict: Performance statistics
        """
        return self._performance_stats.copy()

    async def _initialize_chunkers(self) -> None:
        """
        Initialize available chunkers based on configuration.
        """
        from app.services.chunking.ast_chunker import ASTChunker
        from app.services.chunking.hybrid_chunker import HybridChunker

        # Initialize AST fallback chunker (always available)
        ast_chunker = ASTChunker()
        await ast_chunker.initialize()
        self._chunkers[ChunkingStrategy.AST_FALLBACK] = ast_chunker

        # Initialize hybrid chunker if agentic features are enabled
        if settings.CHUNKING_STRATEGY in ["hybrid", "agentic"]:
            try:
                hybrid_chunker = HybridChunker()
                await hybrid_chunker.initialize()
                self._chunkers[ChunkingStrategy.HYBRID] = hybrid_chunker
                self._chunkers[ChunkingStrategy.AGENTIC] = hybrid_chunker
                logger.info("Hybrid chunker initialized successfully")
            except Exception as e:
                logger.warning(
                    "Failed to initialize hybrid chunker, falling back to AST only",
                    error=str(e)
                )

        logger.info("Initialized chunkers", strategies=list(self._chunkers.keys()))

    async def _initialize_advanced_features(self) -> None:
        """Initialize advanced features like dashboard and quality metrics."""
        try:
            # Start performance dashboard
            await performance_dashboard.start()
            logger.info("Performance dashboard started")

            # Initialize custom strategy manager with default strategies
            await self._initialize_default_custom_strategies()

        except Exception as e:
            logger.warning("Failed to initialize some advanced features", error=str(e))
            # Don't fail the entire service if advanced features fail

    async def _initialize_default_custom_strategies(self) -> None:
        """Initialize default custom chunking strategies."""
        try:
            from app.services.chunking.custom_strategies import CustomStrategy, ChunkingRule, RuleType, RuleCondition

            # Create a default size-based strategy
            size_strategy = CustomStrategy(
                id="default_size_based",
                name="Default Size-Based Chunking",
                description="Chunks code based on optimal size ranges",
                rules=[
                    ChunkingRule(
                        id="max_size_rule",
                        name="Maximum Size Rule",
                        description="Split chunks larger than 200 lines",
                        rule_type=RuleType.SIZE_BASED,
                        condition=RuleCondition.LINE_COUNT,
                        value=200,
                        action="split",
                        priority=1
                    ),
                    ChunkingRule(
                        id="min_size_rule",
                        name="Minimum Size Rule",
                        description="Merge chunks smaller than 10 lines",
                        rule_type=RuleType.SIZE_BASED,
                        condition=RuleCondition.LINE_COUNT,
                        value=10,
                        action="merge",
                        priority=2
                    )
                ]
            )

            custom_strategy_manager.register_strategy(size_strategy)
            logger.info("Default custom strategies initialized")

        except Exception as e:
            logger.warning("Failed to initialize default custom strategies", error=str(e))

    async def _determine_strategy(self, request: ChunkingRequest) -> ChunkingStrategy:
        """
        Determine the optimal chunking strategy for the request.

        Args:
            request: Chunking request

        Returns:
            ChunkingStrategy: Selected strategy
        """
        # Force agentic if requested
        if request.force_agentic and ChunkingStrategy.AGENTIC in self._chunkers:
            return ChunkingStrategy.AGENTIC

        # Use configured default strategy
        configured_strategy = ChunkingStrategy(settings.CHUNKING_STRATEGY)

        # Check if configured strategy is available
        if configured_strategy in self._chunkers:
            return configured_strategy

        # Fallback to AST if configured strategy is not available
        logger.warning(
            "Configured chunking strategy not available, falling back to AST",
            configured_strategy=configured_strategy.value
        )
        return ChunkingStrategy.AST_FALLBACK

    def _validate_configuration(self) -> None:
        """
        Validate chunking service configuration.

        Raises:
            ConfigurationError: If configuration is invalid
        """
        if not self._chunkers:
            raise ConfigurationError(
                "No chunkers available",
                setting="CHUNKING_STRATEGY"
            )

    async def _update_performance_stats(self, result: ChunkingResult) -> None:
        """
        Update performance statistics with the latest result.

        Args:
            result: Chunking result
        """
        self._performance_stats["total_requests"] += 1

        if result.strategy_used == ChunkingStrategy.AGENTIC:
            self._performance_stats["agentic_requests"] += 1
        elif result.strategy_used == ChunkingStrategy.AST_FALLBACK:
            self._performance_stats["ast_fallback_requests"] += 1

        # Update running averages
        total_requests = self._performance_stats["total_requests"]

        # Update average processing time
        current_avg_time = self._performance_stats["average_processing_time"]
        new_avg_time = ((current_avg_time * (total_requests - 1)) + result.processing_time) / total_requests
        self._performance_stats["average_processing_time"] = new_avg_time

        # Update average quality score
        current_avg_quality = self._performance_stats["average_quality_score"]
        new_avg_quality = ((current_avg_quality * (total_requests - 1)) + result.quality_score) / total_requests
        self._performance_stats["average_quality_score"] = new_avg_quality


# Global service instance
chunking_service = ChunkingService()
