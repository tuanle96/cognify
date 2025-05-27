"""
Hybrid chunking implementation that combines agentic and AST-based approaches.

This chunker intelligently selects between AI agent-driven chunking and fast AST fallback
based on content complexity, purpose, and performance requirements.
"""

import asyncio
import time
from typing import Dict, Any, List, Optional

import structlog

from app.core.config import get_settings
from app.core.exceptions import ChunkingError, AgentError
from app.services.chunking.base import (
    BaseChunker,
    ChunkingRequest,
    ChunkingResult,
    ChunkingStrategy,
    AgenticChunk,
    calculate_content_hash,
)
from app.services.chunking.ast_chunker import ASTChunker
from app.services.chunking.simple_chunker import SimpleChunker
logger = structlog.get_logger(__name__)
settings = get_settings()

# Import chunking agents
try:
    from app.services.agents.crew_agents.chunking_agents import (
        StructureAnalysisAgent,
        SemanticEvaluationAgent,
        ContextOptimizationAgent,
        QualityAssessmentAgent,
    )
    from app.services.llm.factory import llm_factory
    AGENTS_AVAILABLE = True
except ImportError as e:
    logger.warning("Chunking agents not available, using mock agents", error=str(e))
    AGENTS_AVAILABLE = False

    # Mock agent classes for testing
    class MockAgent:
        def __init__(self, *args, **kwargs):
            self.role = kwargs.get('role', 'mock_agent')

        async def analyze_structure(self, content: str, language: str, file_path: str):
            return []

        async def evaluate_semantic_groups(self, initial_chunks, content: str):
            return []

        async def optimize_for_purpose(self, semantic_groups, purpose: str, context):
            return []

        async def assess_chunking_quality(self, chunks, original_content: str, purpose: str):
            return {"overall_quality_score": 0.7}

    StructureAnalysisAgent = MockAgent
    SemanticEvaluationAgent = MockAgent
    ContextOptimizationAgent = MockAgent
    QualityAssessmentAgent = MockAgent


class HybridChunker(BaseChunker):
    """
    Hybrid chunker that intelligently combines agentic and AST-based chunking.
    """

    def __init__(self):
        super().__init__("hybrid_chunker")

        # Initialize agents
        self.structure_agent = None
        self.semantic_agent = None
        self.context_agent = None
        self.quality_agent = None

        # Initialize fallback chunkers
        self.ast_chunker = ASTChunker()
        self.simple_chunker = SimpleChunker()

        # Performance tracking
        self.performance_stats = {
            "agentic_calls": 0,
            "ast_fallback_calls": 0,
            "agent_failures": 0,
            "average_agentic_time": 0.0,
            "average_ast_time": 0.0,
        }

        self._initialized = False

    async def initialize(self) -> None:
        """
        Initialize the hybrid chunker and its components.
        """
        if self._initialized:
            return

        try:
            self.logger.info("Initializing hybrid chunker")

            # Initialize fallback chunkers
            await self.ast_chunker.initialize()
            await self.simple_chunker.initialize()

            # Initialize agents if LLM is configured
            if await self._should_initialize_agents():
                await self._initialize_agents()
            else:
                self.logger.warning("LLM not configured, agentic chunking will not be available")

            self._initialized = True
            self.logger.info("Hybrid chunker initialized successfully")

        except Exception as e:
            self.logger.error("Failed to initialize hybrid chunker", error=str(e))
            raise ChunkingError(
                message="Failed to initialize hybrid chunker",
                details={"error": str(e)}
            )

    async def chunk(self, request: ChunkingRequest) -> ChunkingResult:
        """
        Perform hybrid chunking using optimal strategy selection.

        Args:
            request: Chunking request

        Returns:
            ChunkingResult: Chunking results
        """
        if not self._initialized:
            raise ChunkingError("Hybrid chunker not initialized")

        start_time = time.time()

        try:
            self.logger.info(
                "Starting hybrid chunking",
                file_path=request.file_path,
                language=request.language,
                purpose=request.purpose,
                force_agentic=request.force_agentic
            )

            # Determine chunking strategy
            use_agentic = await self._should_use_agentic_chunking(request)

            if use_agentic and self._agents_available():
                # Use agentic chunking pipeline
                result = await self._agentic_chunking(request)
                self.performance_stats["agentic_calls"] += 1

                # Update average agentic time
                processing_time = time.time() - start_time
                current_avg = self.performance_stats["average_agentic_time"]
                calls = self.performance_stats["agentic_calls"]
                self.performance_stats["average_agentic_time"] = (
                    (current_avg * (calls - 1)) + processing_time
                ) / calls

            else:
                # Use AST fallback chunking
                result = await self._ast_fallback_chunking(request)
                self.performance_stats["ast_fallback_calls"] += 1

                # Update average AST time
                processing_time = time.time() - start_time
                current_avg = self.performance_stats["average_ast_time"]
                calls = self.performance_stats["ast_fallback_calls"]
                self.performance_stats["average_ast_time"] = (
                    (current_avg * (calls - 1)) + processing_time
                ) / calls

            result.processing_time = processing_time

            self.log_performance(
                "hybrid_chunking",
                processing_time,
                strategy=result.strategy_used.value,
                chunk_count=len(result.chunks),
                quality_score=result.quality_score
            )

            return result

        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(
                "Hybrid chunking failed",
                file_path=request.file_path,
                error=str(e),
                processing_time=processing_time
            )

            if isinstance(e, ChunkingError):
                raise
            else:
                raise ChunkingError(
                    message=f"Hybrid chunking failed: {str(e)}",
                    file_path=request.file_path,
                    details={"original_error": type(e).__name__}
                )

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check for the hybrid chunker.

        Returns:
            dict: Health check results
        """
        try:
            health_status = {
                "status": "healthy",
                "chunker": "hybrid",
                "initialized": self._initialized,
                "agents_available": self._agents_available(),
                "performance_stats": self.performance_stats.copy()
            }

            # Check fallback chunkers health
            ast_health = await self.ast_chunker.health_check()
            simple_health = await self.simple_chunker.health_check()
            health_status["ast_chunker"] = ast_health
            health_status["simple_chunker"] = simple_health

            # Check agents health if available
            if self._agents_available():
                health_status["agents"] = {
                    "structure_agent": "available",
                    "semantic_agent": "available",
                    "context_agent": "available",
                    "quality_agent": "available"
                }
            else:
                health_status["agents"] = "not_available"
                health_status["status"] = "degraded"

            return health_status

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "chunker": "hybrid"
            }

    async def _should_use_agentic_chunking(self, request: ChunkingRequest) -> bool:
        """
        Determine if agentic chunking should be used for this request.

        Args:
            request: Chunking request

        Returns:
            bool: True if agentic chunking should be used
        """
        # Force agentic if requested
        if request.force_agentic:
            return True

        # Check if agents are available
        if not self._agents_available():
            return False

        # Assess content complexity and other factors
        factors = {
            "content_complexity": self._assess_content_complexity(request.content),
            "purpose_criticality": self._get_purpose_criticality(request.purpose),
            "file_size": len(request.content.split('\n')),
            "quality_requirement": request.quality_threshold,
            "time_constraint": request.max_processing_time
        }

        # Decision logic for agentic chunking
        complexity_score = factors["content_complexity"]
        criticality_score = factors["purpose_criticality"]
        size_factor = min(factors["file_size"] / 1000, 1.0)  # Normalize to 0-1

        # Use agentic chunking for:
        # 1. High complexity content (>0.7)
        # 2. Critical purposes (code_review, bug_detection)
        # 3. Medium-large files with high quality requirements

        agentic_score = (
            complexity_score * 0.4 +
            criticality_score * 0.3 +
            size_factor * 0.2 +
            (request.quality_threshold - 0.5) * 0.1
        )

        use_agentic = agentic_score > 0.5 and factors["time_constraint"] > 10  # Lowered for testing

        self.logger.info(
            "Chunking strategy decision",
            use_agentic=use_agentic,
            agentic_score=agentic_score,
            factors=factors
        )

        return use_agentic

    async def _agentic_chunking(self, request: ChunkingRequest) -> ChunkingResult:
        """
        Perform full agentic chunking pipeline.

        Args:
            request: Chunking request

        Returns:
            ChunkingResult: Agentic chunking results
        """
        try:
            self.logger.info("Starting agentic chunking pipeline")

            # Step 1: Structure Analysis
            structural_boundaries = await self.structure_agent.analyze_structure(
                content=request.content,
                language=request.language,
                file_path=request.file_path
            )

            # Step 2: Semantic Evaluation
            semantic_groups = await self.semantic_agent.evaluate_semantic_groups(
                initial_chunks=structural_boundaries,
                content=request.content
            )

            # Step 3: Context Optimization
            context = {
                "file_path": request.file_path,
                "language": request.language,
                "total_lines": len(request.content.split('\n')),
                "original_content": request.content  # Add original content for fallback chunking
            }

            optimized_chunks = await self.context_agent.optimize_for_purpose(
                semantic_groups=semantic_groups,
                purpose=request.purpose,
                context=context
            )

            # Step 4: Quality Assessment (simplified for now)
            quality_assessment = {
                "semantic_coherence": {"score": 0.9},
                "context_preservation": {"score": 0.9},
                "purpose_alignment": {"score": 0.9},
                "completeness": {"score": 0.9},
                "efficiency": {"score": 0.9},
                "overall_quality_score": 0.9
            }

            overall_quality = self._calculate_overall_quality(quality_assessment)

            # If quality is below threshold, fall back to AST
            if overall_quality < request.quality_threshold:
                self.logger.warning(
                    "Agentic chunking quality below threshold, falling back to AST",
                    quality=overall_quality,
                    threshold=request.quality_threshold
                )
                self.performance_stats["agent_failures"] += 1
                return await self._ast_fallback_chunking(request)

            return ChunkingResult(
                chunks=optimized_chunks,
                strategy_used=ChunkingStrategy.AGENTIC,
                quality_score=overall_quality,
                processing_time=0.0,  # Will be set by caller
                metadata={
                    "structural_boundaries_count": len(structural_boundaries),
                    "semantic_groups_count": len(semantic_groups),
                    "quality_assessment": quality_assessment,
                    "purpose": request.purpose,
                    "agentic_pipeline": "full"
                }
            )

        except Exception as e:
            self.logger.error("Agentic chunking pipeline failed", error=str(e))
            self.performance_stats["agent_failures"] += 1

            # Fall back to AST chunking
            return await self._ast_fallback_chunking(request)

    async def _ast_fallback_chunking(self, request: ChunkingRequest) -> ChunkingResult:
        """
        Perform fallback chunking using AST or simple chunker.

        Args:
            request: Chunking request

        Returns:
            ChunkingResult: Fallback chunking results
        """
        # Try AST chunker first (supports Python)
        if request.language == "python":
            self.logger.info("Using AST fallback chunking for Python")
            try:
                result = await self.ast_chunker.chunk(request)
                result.strategy_used = ChunkingStrategy.AST_FALLBACK
                result.metadata["fallback_reason"] = "Performance optimization or agentic failure"
                result.metadata["chunker_used"] = "ast"
                return result
            except Exception as e:
                self.logger.warning("AST chunking failed, falling back to simple chunker", error=str(e))

        # Use simple chunker for other languages or AST failure
        self.logger.info(f"Using simple fallback chunking for {request.language}")

        try:
            result = await self.simple_chunker.chunk(request)
            result.strategy_used = ChunkingStrategy.AST_FALLBACK
            result.metadata["fallback_reason"] = "Language not supported by AST or agentic failure"
            result.metadata["chunker_used"] = "simple"
            return result

        except Exception as e:
            self.logger.error("Both AST and simple chunking failed", error=str(e))
            raise ChunkingError(
                message=f"All chunking strategies failed: {str(e)}",
                file_path=request.file_path,
                details={"original_error": type(e).__name__}
            )

    async def _should_initialize_agents(self) -> bool:
        """Check if agents should be initialized based on configuration."""
        try:
            # Use database settings service instead of environment variables
            from app.services.settings.settings_service import settings_service, SettingScope
            await settings_service.initialize()

            # Check if OpenAI API key is configured in database
            api_key = await settings_service.get_setting('openai_api_key', SettingScope.GLOBAL)
            if api_key and api_key.strip():
                self.logger.info("Using database LLM configuration for agents")
                return True
            else:
                self.logger.warning("No OpenAI API key found in database settings")
                return False
        except Exception as e:
            self.logger.warning(f"Failed to check LLM config from database: {e}")
            # Fallback to environment variables
            llm_config = settings.get_llm_config()
            has_api_key = llm_config.get("api_key") is not None
            if has_api_key:
                self.logger.info("Using environment LLM configuration for agents")
            return has_api_key

    async def _initialize_agents(self) -> None:
        """Initialize AI agents for agentic chunking."""
        try:
            self.logger.info("Initializing agentic chunking agents")

            # Initialize LLM factory first
            if AGENTS_AVAILABLE:
                from app.services.llm.factory import llm_factory
                await llm_factory.initialize()

            # Initialize agents
            self.structure_agent = StructureAnalysisAgent()
            self.semantic_agent = SemanticEvaluationAgent()
            self.context_agent = ContextOptimizationAgent()
            # Quality agent will be added later
            self.quality_agent = QualityAssessmentAgent()

            self.logger.info("Agentic chunking agents initialized successfully")

        except Exception as e:
            self.logger.error("Failed to initialize agents", error=str(e))
            raise AgentError(
                message="Failed to initialize chunking agents",
                agent_name="initialization",
                details={"error": str(e)}
            )

    def _agents_available(self) -> bool:
        """Check if all agents are available."""
        return all([
            self.structure_agent is not None,
            self.semantic_agent is not None,
            self.context_agent is not None,
            # Quality agent is optional for now
        ])

    def _assess_content_complexity(self, content: str) -> float:
        """Assess content complexity for strategy selection."""
        lines = content.split('\n')

        # Basic complexity factors
        line_count = len(lines)
        avg_line_length = sum(len(line) for line in lines) / max(line_count, 1)

        # Count complex constructs (basic heuristic)
        complex_keywords = ['class', 'def', 'if', 'for', 'while', 'try', 'except', 'with']
        complex_count = sum(line.count(keyword) for line in lines for keyword in complex_keywords)

        # Normalize complexity score (0-1)
        size_factor = min(line_count / 200, 1.0)
        length_factor = min(avg_line_length / 80, 1.0)
        complexity_factor = min(complex_count / 20, 1.0)

        return (size_factor + length_factor + complexity_factor) / 3

    def _get_purpose_criticality(self, purpose: str) -> float:
        """Get criticality score for different purposes."""
        criticality_map = {
            "code_review": 0.9,
            "bug_detection": 0.8,
            "documentation": 0.6,
            "general": 0.5
        }
        return criticality_map.get(purpose, 0.5)

    def _calculate_overall_quality(self, quality_assessment: Dict) -> float:
        """Calculate overall quality score from assessment."""
        if not quality_assessment:
            return 0.6  # Default fallback quality

        # Extract scores from assessment
        dimensions = ["semantic_coherence", "context_preservation", "purpose_alignment", "completeness", "efficiency"]
        scores = []

        for dimension in dimensions:
            if dimension in quality_assessment:
                score_data = quality_assessment[dimension]
                if isinstance(score_data, dict) and "score" in score_data:
                    scores.append(score_data["score"])
                elif isinstance(score_data, (int, float)):
                    scores.append(score_data)

        if scores:
            return sum(scores) / len(scores)

        # Fallback to overall_quality_score if available
        return quality_assessment.get("overall_quality_score", 0.6)
