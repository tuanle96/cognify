"""
Advanced quality metrics and feedback integration for chunking.

Provides comprehensive quality assessment and user feedback integration.
"""

import time
import statistics
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

import structlog

from app.services.chunking.base import AgenticChunk, ChunkingResult

logger = structlog.get_logger(__name__)


class QualityDimension(Enum):
    """Quality assessment dimensions."""
    SEMANTIC_COHERENCE = "semantic_coherence"
    CONTEXT_PRESERVATION = "context_preservation"
    FUNCTIONAL_COMPLETENESS = "functional_completeness"
    OPTIMAL_GRANULARITY = "optimal_granularity"
    DEPENDENCY_CLARITY = "dependency_clarity"
    READABILITY = "readability"
    MAINTAINABILITY = "maintainability"
    TESTABILITY = "testability"
    REUSABILITY = "reusability"
    PERFORMANCE = "performance"


class FeedbackType(Enum):
    """Types of user feedback."""
    RATING = "rating"
    COMMENT = "comment"
    SUGGESTION = "suggestion"
    BUG_REPORT = "bug_report"
    IMPROVEMENT = "improvement"


@dataclass
class QualityScore:
    """Individual quality score for a dimension."""
    dimension: QualityDimension
    score: float  # 0.0 - 1.0
    confidence: float  # 0.0 - 1.0
    reasoning: str
    evidence: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


@dataclass
class QualityAssessment:
    """Comprehensive quality assessment."""
    overall_score: float
    dimension_scores: Dict[QualityDimension, QualityScore]
    chunk_level_scores: Dict[str, float]  # chunk_id -> score
    assessment_time: datetime = field(default_factory=datetime.now)
    assessor: str = "system"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserFeedback:
    """User feedback on chunking quality."""
    id: str
    feedback_type: FeedbackType
    chunk_id: Optional[str]
    overall_rating: Optional[float]  # 1.0 - 5.0
    dimension_ratings: Dict[QualityDimension, float] = field(default_factory=dict)
    comment: str = ""
    suggestions: List[str] = field(default_factory=list)
    user_id: str = "anonymous"
    timestamp: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)


class AdvancedQualityAnalyzer:
    """Advanced quality analysis with multiple assessment strategies."""

    def __init__(self):
        self.logger = structlog.get_logger("quality_analyzer")
        self._assessment_cache = {}

    async def assess_chunking_quality(
        self,
        result: ChunkingResult,
        original_content: str,
        purpose: str = "general"
    ) -> QualityAssessment:
        """Perform comprehensive quality assessment."""
        start_time = time.time()

        # Calculate dimension scores
        dimension_scores = {}

        # Semantic coherence
        dimension_scores[QualityDimension.SEMANTIC_COHERENCE] = await self._assess_semantic_coherence(
            result.chunks, original_content
        )

        # Context preservation
        dimension_scores[QualityDimension.CONTEXT_PRESERVATION] = await self._assess_context_preservation(
            result.chunks, original_content
        )

        # Functional completeness
        dimension_scores[QualityDimension.FUNCTIONAL_COMPLETENESS] = await self._assess_functional_completeness(
            result.chunks, original_content
        )

        # Optimal granularity
        dimension_scores[QualityDimension.OPTIMAL_GRANULARITY] = await self._assess_optimal_granularity(
            result.chunks, purpose
        )

        # Dependency clarity
        dimension_scores[QualityDimension.DEPENDENCY_CLARITY] = await self._assess_dependency_clarity(
            result.chunks
        )

        # Readability
        dimension_scores[QualityDimension.READABILITY] = await self._assess_readability(
            result.chunks
        )

        # Maintainability
        dimension_scores[QualityDimension.MAINTAINABILITY] = await self._assess_maintainability(
            result.chunks, original_content
        )

        # Performance
        dimension_scores[QualityDimension.PERFORMANCE] = await self._assess_performance(
            result, time.time() - start_time
        )

        # Calculate overall score (weighted average)
        weights = self._get_dimension_weights(purpose)
        overall_score = sum(
            score.score * weights.get(dim, 1.0)
            for dim, score in dimension_scores.items()
        ) / sum(weights.values())

        # Calculate chunk-level scores
        chunk_level_scores = {}
        for chunk in result.chunks:
            chunk_level_scores[chunk.id] = await self._assess_chunk_quality(chunk, original_content)

        assessment = QualityAssessment(
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            chunk_level_scores=chunk_level_scores,
            assessor="advanced_analyzer",
            metadata={
                "purpose": purpose,
                "assessment_duration": time.time() - start_time,
                "chunk_count": len(result.chunks),
                "strategy_used": result.strategy_used.value
            }
        )

        self.logger.info(
            "Quality assessment completed",
            overall_score=overall_score,
            chunk_count=len(result.chunks),
            duration=time.time() - start_time
        )

        return assessment

    async def _assess_semantic_coherence(self, chunks: List[AgenticChunk], content: str) -> QualityScore:
        """Assess semantic coherence of chunks."""
        scores = []
        evidence = []
        suggestions = []

        for chunk in chunks:
            # Check if chunk represents a coherent semantic unit
            coherence_score = 0.8  # Base score

            # Bonus for function/class boundaries
            if chunk.chunk_type.value in ["function", "class", "method"]:
                coherence_score += 0.1
                evidence.append(f"Chunk '{chunk.name}' respects {chunk.chunk_type.value} boundaries")

            # Penalty for incomplete constructs
            if self._has_incomplete_constructs(chunk.content):
                coherence_score -= 0.2
                suggestions.append(f"Chunk '{chunk.name}' contains incomplete constructs")

            # Bonus for related functionality
            if self._has_related_functionality(chunk.content):
                coherence_score += 0.1
                evidence.append(f"Chunk '{chunk.name}' contains related functionality")

            scores.append(max(0.0, min(1.0, coherence_score)))

        avg_score = statistics.mean(scores) if scores else 0.0

        return QualityScore(
            dimension=QualityDimension.SEMANTIC_COHERENCE,
            score=avg_score,
            confidence=0.8,
            reasoning=f"Average semantic coherence across {len(chunks)} chunks",
            evidence=evidence,
            suggestions=suggestions
        )

    async def _assess_context_preservation(self, chunks: List[AgenticChunk], content: str) -> QualityScore:
        """Assess context preservation across chunks."""
        scores = []
        evidence = []
        suggestions = []

        # Check for context dependencies between chunks
        for i, chunk in enumerate(chunks):
            context_score = 0.7  # Base score

            # Check if chunk can be understood independently
            if self._is_self_contained(chunk.content):
                context_score += 0.2
                evidence.append(f"Chunk '{chunk.name}' is self-contained")
            else:
                suggestions.append(f"Chunk '{chunk.name}' may need more context")

            # Check for proper import/dependency handling
            if chunk.dependencies:
                context_score += 0.1
                evidence.append(f"Chunk '{chunk.name}' has explicit dependencies")

            scores.append(max(0.0, min(1.0, context_score)))

        avg_score = statistics.mean(scores) if scores else 0.0

        return QualityScore(
            dimension=QualityDimension.CONTEXT_PRESERVATION,
            score=avg_score,
            confidence=0.7,
            reasoning=f"Context preservation analysis across {len(chunks)} chunks",
            evidence=evidence,
            suggestions=suggestions
        )

    async def _assess_functional_completeness(self, chunks: List[AgenticChunk], content: str) -> QualityScore:
        """Assess functional completeness of chunks."""
        evidence = []
        suggestions = []

        # Check if all important code elements are captured
        total_lines = len(content.split('\n'))
        covered_lines = sum(chunk.size_lines for chunk in chunks)
        coverage_ratio = covered_lines / total_lines if total_lines > 0 else 0.0

        completeness_score = coverage_ratio

        if coverage_ratio >= 0.95:
            evidence.append(f"Excellent coverage: {coverage_ratio:.1%} of code lines")
        elif coverage_ratio >= 0.8:
            evidence.append(f"Good coverage: {coverage_ratio:.1%} of code lines")
        else:
            suggestions.append(f"Low coverage: {coverage_ratio:.1%} of code lines - some code may be missing")

        # Check for missing important constructs
        important_patterns = ["class ", "def ", "function ", "import ", "from "]
        for pattern in important_patterns:
            if pattern in content:
                pattern_in_chunks = any(pattern in chunk.content for chunk in chunks)
                if not pattern_in_chunks:
                    suggestions.append(f"Pattern '{pattern.strip()}' found in original but missing in chunks")
                    completeness_score -= 0.1

        return QualityScore(
            dimension=QualityDimension.FUNCTIONAL_COMPLETENESS,
            score=max(0.0, min(1.0, completeness_score)),
            confidence=0.9,
            reasoning=f"Functional completeness based on coverage and construct analysis",
            evidence=evidence,
            suggestions=suggestions
        )

    async def _assess_optimal_granularity(self, chunks: List[AgenticChunk], purpose: str) -> QualityScore:
        """Assess if chunk granularity is optimal for the purpose."""
        evidence = []
        suggestions = []

        # Purpose-specific optimal sizes
        optimal_ranges = {
            "code_review": (50, 200),
            "documentation": (100, 300),
            "bug_detection": (30, 150),
            "general": (50, 150)
        }

        min_size, max_size = optimal_ranges.get(purpose, optimal_ranges["general"])

        size_scores = []
        for chunk in chunks:
            if min_size <= chunk.size_lines <= max_size:
                size_scores.append(1.0)
                evidence.append(f"Chunk '{chunk.name}' has optimal size ({chunk.size_lines} lines)")
            elif chunk.size_lines < min_size:
                size_scores.append(0.6)
                suggestions.append(f"Chunk '{chunk.name}' is too small ({chunk.size_lines} lines)")
            else:
                size_scores.append(0.7)
                suggestions.append(f"Chunk '{chunk.name}' is too large ({chunk.size_lines} lines)")

        # Also consider chunk count
        chunk_count = len(chunks)
        if 3 <= chunk_count <= 20:
            count_score = 1.0
            evidence.append(f"Good chunk count: {chunk_count}")
        elif chunk_count < 3:
            count_score = 0.6
            suggestions.append(f"Too few chunks: {chunk_count}")
        else:
            count_score = 0.7
            suggestions.append(f"Too many chunks: {chunk_count}")

        avg_size_score = statistics.mean(size_scores) if size_scores else 0.0
        granularity_score = (avg_size_score + count_score) / 2

        return QualityScore(
            dimension=QualityDimension.OPTIMAL_GRANULARITY,
            score=granularity_score,
            confidence=0.8,
            reasoning=f"Granularity assessment for {purpose} purpose",
            evidence=evidence,
            suggestions=suggestions
        )

    async def _assess_dependency_clarity(self, chunks: List[AgenticChunk]) -> QualityScore:
        """Assess clarity of dependencies between chunks."""
        evidence = []
        suggestions = []

        dependency_score = 0.8  # Base score

        # Check if dependencies are explicitly tracked
        chunks_with_deps = [c for c in chunks if c.dependencies]
        if chunks_with_deps:
            dependency_score += 0.1
            evidence.append(f"{len(chunks_with_deps)} chunks have explicit dependencies")

        # Check for circular dependencies
        if self._has_circular_dependencies(chunks):
            dependency_score -= 0.3
            suggestions.append("Circular dependencies detected between chunks")

        # Check for missing dependencies
        missing_deps = self._find_missing_dependencies(chunks)
        if missing_deps:
            dependency_score -= 0.1 * len(missing_deps)
            suggestions.extend([f"Missing dependency: {dep}" for dep in missing_deps[:3]])

        return QualityScore(
            dimension=QualityDimension.DEPENDENCY_CLARITY,
            score=max(0.0, min(1.0, dependency_score)),
            confidence=0.6,
            reasoning="Dependency analysis based on explicit tracking and circular dependency detection",
            evidence=evidence,
            suggestions=suggestions
        )

    async def _assess_readability(self, chunks: List[AgenticChunk]) -> QualityScore:
        """Assess readability of chunks."""
        evidence = []
        suggestions = []

        readability_scores = []
        for chunk in chunks:
            score = 0.7  # Base score

            # Check for descriptive names
            if chunk.name and len(chunk.name) > 3 and not chunk.name.startswith("chunk_"):
                score += 0.2
                evidence.append(f"Chunk '{chunk.name}' has descriptive name")

            # Check for reasonable complexity
            if hasattr(chunk.metadata, 'complexity') and chunk.metadata.complexity:
                if chunk.metadata.complexity <= 5:
                    score += 0.1
                    evidence.append(f"Chunk '{chunk.name}' has low complexity")
                elif chunk.metadata.complexity > 8:
                    score -= 0.1
                    suggestions.append(f"Chunk '{chunk.name}' has high complexity")

            readability_scores.append(max(0.0, min(1.0, score)))

        avg_score = statistics.mean(readability_scores) if readability_scores else 0.0

        return QualityScore(
            dimension=QualityDimension.READABILITY,
            score=avg_score,
            confidence=0.7,
            reasoning="Readability assessment based on naming and complexity",
            evidence=evidence,
            suggestions=suggestions
        )

    async def _assess_maintainability(self, chunks: List[AgenticChunk], content: str) -> QualityScore:
        """Assess maintainability of chunking structure."""
        evidence = []
        suggestions = []

        maintainability_score = 0.8  # Base score

        # Check for logical grouping
        if self._has_logical_grouping(chunks):
            maintainability_score += 0.1
            evidence.append("Chunks follow logical grouping patterns")

        # Check for consistent sizing
        sizes = [chunk.size_lines for chunk in chunks]
        if len(sizes) > 1:  # Need at least 2 data points for variance
            size_variance = statistics.variance(sizes)
            if size_variance < 100:  # Low variance
                maintainability_score += 0.1
                evidence.append("Consistent chunk sizes")
            elif size_variance > 500:  # High variance
                maintainability_score -= 0.1
                suggestions.append("Inconsistent chunk sizes may affect maintainability")
        elif len(sizes) == 1:
            # Single chunk - perfect consistency
            maintainability_score += 0.1
            evidence.append("Single chunk - perfect size consistency")

        return QualityScore(
            dimension=QualityDimension.MAINTAINABILITY,
            score=max(0.0, min(1.0, maintainability_score)),
            confidence=0.6,
            reasoning="Maintainability assessment based on structure and consistency",
            evidence=evidence,
            suggestions=suggestions
        )

    async def _assess_performance(self, result: ChunkingResult, assessment_time: float) -> QualityScore:
        """Assess performance aspects of chunking."""
        evidence = []
        suggestions = []

        performance_score = 0.8  # Base score

        # Processing time assessment
        if result.processing_time < 1.0:
            performance_score += 0.2
            evidence.append(f"Fast processing: {result.processing_time:.3f}s")
        elif result.processing_time > 10.0:
            performance_score -= 0.2
            suggestions.append(f"Slow processing: {result.processing_time:.3f}s")

        # Chunk count efficiency
        chunk_count = len(result.chunks)
        if 5 <= chunk_count <= 15:
            performance_score += 0.1
            evidence.append(f"Efficient chunk count: {chunk_count}")
        elif chunk_count > 50:
            performance_score -= 0.1
            suggestions.append(f"High chunk count may impact performance: {chunk_count}")

        return QualityScore(
            dimension=QualityDimension.PERFORMANCE,
            score=max(0.0, min(1.0, performance_score)),
            confidence=0.9,
            reasoning="Performance assessment based on processing time and efficiency",
            evidence=evidence,
            suggestions=suggestions
        )

    async def _assess_chunk_quality(self, chunk: AgenticChunk, original_content: str) -> float:
        """Assess quality of individual chunk."""
        score = 0.7  # Base score

        # Size appropriateness
        if 20 <= chunk.size_lines <= 150:
            score += 0.1

        # Type specificity
        if chunk.chunk_type.value != "unknown":
            score += 0.1

        # Name quality
        if chunk.name and not chunk.name.startswith("chunk_"):
            score += 0.1

        return max(0.0, min(1.0, score))

    def _get_dimension_weights(self, purpose: str) -> Dict[QualityDimension, float]:
        """Get dimension weights based on purpose."""
        weights = {
            "code_review": {
                QualityDimension.SEMANTIC_COHERENCE: 1.5,
                QualityDimension.READABILITY: 1.3,
                QualityDimension.MAINTAINABILITY: 1.2,
                QualityDimension.OPTIMAL_GRANULARITY: 1.0,
                QualityDimension.CONTEXT_PRESERVATION: 1.0,
                QualityDimension.FUNCTIONAL_COMPLETENESS: 0.8,
                QualityDimension.DEPENDENCY_CLARITY: 0.8,
                QualityDimension.PERFORMANCE: 0.5
            },
            "documentation": {
                QualityDimension.FUNCTIONAL_COMPLETENESS: 1.5,
                QualityDimension.CONTEXT_PRESERVATION: 1.3,
                QualityDimension.READABILITY: 1.2,
                QualityDimension.OPTIMAL_GRANULARITY: 1.0,
                QualityDimension.SEMANTIC_COHERENCE: 1.0,
                QualityDimension.DEPENDENCY_CLARITY: 0.8,
                QualityDimension.MAINTAINABILITY: 0.7,
                QualityDimension.PERFORMANCE: 0.5
            },
            "general": {
                QualityDimension.SEMANTIC_COHERENCE: 1.0,
                QualityDimension.CONTEXT_PRESERVATION: 1.0,
                QualityDimension.FUNCTIONAL_COMPLETENESS: 1.0,
                QualityDimension.OPTIMAL_GRANULARITY: 1.0,
                QualityDimension.DEPENDENCY_CLARITY: 1.0,
                QualityDimension.READABILITY: 1.0,
                QualityDimension.MAINTAINABILITY: 1.0,
                QualityDimension.PERFORMANCE: 1.0
            }
        }

        return weights.get(purpose, weights["general"])

    # Helper methods for quality assessment
    def _has_incomplete_constructs(self, content: str) -> bool:
        """Check if content has incomplete constructs."""
        # Simple heuristic: unmatched braces or incomplete function definitions
        open_braces = content.count('{')
        close_braces = content.count('}')
        return abs(open_braces - close_braces) > 0

    def _has_related_functionality(self, content: str) -> bool:
        """Check if content contains related functionality."""
        # Simple heuristic: similar function names or shared variables
        lines = content.split('\n')
        function_names = []
        for line in lines:
            if 'def ' in line or 'function ' in line:
                # Extract function name (simplified)
                parts = line.split()
                for i, part in enumerate(parts):
                    if part in ['def', 'function'] and i + 1 < len(parts):
                        func_name = parts[i + 1].split('(')[0]
                        function_names.append(func_name)

        # Check for similar naming patterns
        if len(function_names) >= 2:
            prefixes = set()
            for name in function_names:
                if '_' in name:
                    prefixes.add(name.split('_')[0])
            return len(prefixes) < len(function_names)  # Some shared prefixes

        return False

    def _is_self_contained(self, content: str) -> bool:
        """Check if content is self-contained."""
        # Simple heuristic: has imports or defines what it uses
        has_imports = any(keyword in content for keyword in ['import ', 'from ', 'require(', 'include '])
        has_definitions = any(keyword in content for keyword in ['def ', 'class ', 'function ', 'var ', 'let ', 'const '])

        return has_imports or has_definitions

    def _has_circular_dependencies(self, chunks: List[AgenticChunk]) -> bool:
        """Check for circular dependencies between chunks."""
        # Build dependency graph
        deps = {}
        for chunk in chunks:
            deps[chunk.id] = set(chunk.dependencies)

        # Simple cycle detection using DFS
        def has_cycle(node, visited, rec_stack):
            visited.add(node)
            rec_stack.add(node)

            for neighbor in deps.get(node, set()):
                if neighbor in deps:  # Only check chunks we know about
                    if neighbor not in visited:
                        if has_cycle(neighbor, visited, rec_stack):
                            return True
                    elif neighbor in rec_stack:
                        return True

            rec_stack.remove(node)
            return False

        visited = set()
        for chunk_id in deps:
            if chunk_id not in visited:
                if has_cycle(chunk_id, visited, set()):
                    return True

        return False

    def _find_missing_dependencies(self, chunks: List[AgenticChunk]) -> List[str]:
        """Find potentially missing dependencies."""
        # This is a simplified implementation
        # In practice, this would use more sophisticated analysis
        missing = []

        chunk_names = {chunk.name for chunk in chunks}

        for chunk in chunks:
            # Look for references to other chunks that aren't in dependencies
            for other_chunk in chunks:
                if other_chunk.id != chunk.id and other_chunk.name in chunk.content:
                    if other_chunk.name not in chunk.dependencies:
                        missing.append(f"{chunk.name} -> {other_chunk.name}")

        return missing[:5]  # Limit to first 5

    def _has_logical_grouping(self, chunks: List[AgenticChunk]) -> bool:
        """Check if chunks follow logical grouping."""
        # Simple heuristic: chunks of same type are grouped together
        types = [chunk.chunk_type for chunk in chunks]

        # Check if similar types are clustered
        type_changes = 0
        for i in range(1, len(types)):
            if types[i] != types[i-1]:
                type_changes += 1

        # If there are fewer type changes than chunks, there's some grouping
        return type_changes < len(chunks) - 1


class FeedbackManager:
    """Manager for user feedback and quality improvement."""

    def __init__(self):
        self.feedback_store: List[UserFeedback] = []
        self.logger = structlog.get_logger("feedback_manager")

    def submit_feedback(self, feedback: UserFeedback) -> bool:
        """Submit user feedback."""
        try:
            self.feedback_store.append(feedback)
            self.logger.info(
                "Feedback submitted",
                feedback_id=feedback.id,
                type=feedback.feedback_type.value,
                rating=feedback.overall_rating
            )
            return True
        except Exception as e:
            self.logger.error("Failed to submit feedback", error=str(e))
            return False

    def get_feedback_summary(self, chunk_id: Optional[str] = None) -> Dict[str, Any]:
        """Get summary of feedback."""
        relevant_feedback = self.feedback_store
        if chunk_id:
            relevant_feedback = [f for f in relevant_feedback if f.chunk_id == chunk_id]

        if not relevant_feedback:
            return {"count": 0, "average_rating": 0.0}

        ratings = [f.overall_rating for f in relevant_feedback if f.overall_rating is not None]

        return {
            "count": len(relevant_feedback),
            "average_rating": statistics.mean(ratings) if ratings else 0.0,
            "feedback_types": {
                ftype.value: len([f for f in relevant_feedback if f.feedback_type == ftype])
                for ftype in FeedbackType
            },
            "recent_comments": [
                f.comment for f in relevant_feedback[-5:] if f.comment
            ]
        }

    def get_improvement_suggestions(self) -> List[str]:
        """Get improvement suggestions based on feedback."""
        suggestions = []

        # Aggregate suggestions from feedback
        all_suggestions = []
        for feedback in self.feedback_store:
            all_suggestions.extend(feedback.suggestions)

        # Count frequency of suggestions
        suggestion_counts = {}
        for suggestion in all_suggestions:
            suggestion_counts[suggestion] = suggestion_counts.get(suggestion, 0) + 1

        # Return most common suggestions
        sorted_suggestions = sorted(
            suggestion_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [suggestion for suggestion, count in sorted_suggestions[:10]]


# Global instances
quality_analyzer = AdvancedQualityAnalyzer()
feedback_manager = FeedbackManager()
