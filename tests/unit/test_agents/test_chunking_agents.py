"""Unit tests for chunking agents."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from app.services.agents.crew_agents.chunking_agents import (
    StructureAnalysisAgent,
    SemanticEvaluationAgent,
    ContextOptimizationAgent,
    QualityAssessmentAgent
)
from tests.utils.assertions import (
    assert_agent_decision_valid,
    assert_performance_acceptable
)
from tests.utils.test_helpers import generate_test_code


class TestStructureAnalysisAgent:
    """Test cases for StructureAnalysisAgent."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_agent_initialization(self, structure_analysis_agent):
        """Test agent initialization."""
        assert structure_analysis_agent is not None
        assert hasattr(structure_analysis_agent, 'analyze_structure')
        assert structure_analysis_agent.role == "Code Structure Analysis Specialist"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_analyze_structure_basic(self, structure_analysis_agent, sample_python_code):
        """Test basic structure analysis."""
        boundaries = await structure_analysis_agent.analyze_structure(
            content=sample_python_code,
            language="python",
            file_path="test.py"
        )

        assert isinstance(boundaries, list)
        assert len(boundaries) > 0

        # Validate each boundary
        for boundary in boundaries:
            assert isinstance(boundary, dict)
            required_keys = ['start_line', 'end_line', 'chunk_type', 'reasoning']
            for key in required_keys:
                assert key in boundary, f"Boundary missing key: {key}"

            assert isinstance(boundary['start_line'], int)
            assert isinstance(boundary['end_line'], int)
            assert boundary['start_line'] <= boundary['end_line']
            assert isinstance(boundary['chunk_type'], str)
            assert isinstance(boundary['reasoning'], str)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_analyze_structure_different_languages(self, structure_analysis_agent):
        """Test structure analysis with different languages."""
        test_cases = [
            ("python", generate_test_code("python", "simple")),
            ("javascript", generate_test_code("javascript", "simple")),
        ]

        for language, code in test_cases:
            boundaries = await structure_analysis_agent.analyze_structure(
                content=code,
                language=language,
                file_path=f"test.{language}"
            )

            assert isinstance(boundaries, list)
            # Should find at least one boundary for non-empty code
            if code.strip():
                assert len(boundaries) > 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_analyze_structure_empty_content(self, structure_analysis_agent):
        """Test structure analysis with empty content."""
        boundaries = await structure_analysis_agent.analyze_structure(
            content="",
            language="python",
            file_path="empty.py"
        )

        assert isinstance(boundaries, list)
        assert len(boundaries) == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_analyze_structure_performance(self, structure_analysis_agent):
        """Test structure analysis performance."""
        complex_code = generate_test_code("python", "complex")

        import time
        start_time = time.time()

        boundaries = await structure_analysis_agent.analyze_structure(
            content=complex_code,
            language="python",
            file_path="complex.py"
        )

        end_time = time.time()

        assert isinstance(boundaries, list)
        assert_performance_acceptable(end_time - start_time, max_time=10.0)


class TestSemanticEvaluationAgent:
    """Test cases for SemanticEvaluationAgent."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_agent_initialization(self, semantic_evaluation_agent):
        """Test agent initialization."""
        assert semantic_evaluation_agent is not None
        assert hasattr(semantic_evaluation_agent, 'evaluate_semantic_groups')
        assert semantic_evaluation_agent.role == "Semantic Relationship Analyst"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_evaluate_relationships_basic(self, semantic_evaluation_agent, sample_python_code):
        """Test basic semantic evaluation."""
        evaluation = await semantic_evaluation_agent.evaluate_relationships(
            content=sample_python_code,
            language="python",
            file_path="test.py"
        )

        assert isinstance(evaluation, dict)
        required_keys = ['relationships', 'coherence_score', 'grouping_suggestions']
        for key in required_keys:
            assert key in evaluation, f"Evaluation missing key: {key}"

        assert isinstance(evaluation['relationships'], list)
        assert isinstance(evaluation['coherence_score'], (int, float))
        assert 0.0 <= evaluation['coherence_score'] <= 1.0
        assert isinstance(evaluation['grouping_suggestions'], list)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_evaluate_relationships_with_boundaries(self, semantic_evaluation_agent, sample_python_code):
        """Test semantic evaluation with predefined boundaries."""
        boundaries = [
            {'start_line': 1, 'end_line': 5, 'chunk_type': 'function'},
            {'start_line': 6, 'end_line': 15, 'chunk_type': 'function'},
            {'start_line': 16, 'end_line': 30, 'chunk_type': 'class'}
        ]

        evaluation = await semantic_evaluation_agent.evaluate_relationships(
            content=sample_python_code,
            language="python",
            file_path="test.py",
            boundaries=boundaries
        )

        assert isinstance(evaluation, dict)
        assert 'relationships' in evaluation
        assert 'coherence_score' in evaluation

        # Should have relationships between the boundaries
        if len(boundaries) > 1:
            assert len(evaluation['relationships']) > 0


class TestContextOptimizationAgent:
    """Test cases for ContextOptimizationAgent."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_agent_initialization(self, context_optimization_agent):
        """Test agent initialization."""
        assert context_optimization_agent is not None
        assert hasattr(context_optimization_agent, 'optimize_for_purpose')
        assert context_optimization_agent.name == "Context Optimization Agent"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_optimize_for_purpose_general(self, context_optimization_agent, sample_python_code):
        """Test optimization for general purpose."""
        optimization = await context_optimization_agent.optimize_for_purpose(
            content=sample_python_code,
            language="python",
            file_path="test.py",
            purpose="general"
        )

        assert isinstance(optimization, dict)
        required_keys = ['optimized_boundaries', 'optimization_reasoning', 'quality_improvements']
        for key in required_keys:
            assert key in optimization, f"Optimization missing key: {key}"

        assert isinstance(optimization['optimized_boundaries'], list)
        assert isinstance(optimization['optimization_reasoning'], str)
        assert isinstance(optimization['quality_improvements'], list)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_optimize_for_different_purposes(self, context_optimization_agent, sample_python_code):
        """Test optimization for different purposes."""
        purposes = ["general", "code_review", "bug_detection", "documentation"]

        for purpose in purposes:
            optimization = await context_optimization_agent.optimize_for_purpose(
                content=sample_python_code,
                language="python",
                file_path="test.py",
                purpose=purpose
            )

            assert isinstance(optimization, dict)
            assert 'optimized_boundaries' in optimization
            assert 'optimization_reasoning' in optimization

            # Reasoning should mention the purpose
            assert purpose in optimization['optimization_reasoning'].lower()


class TestQualityAssessmentAgent:
    """Test cases for QualityAssessmentAgent."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_agent_initialization(self, quality_assessment_agent):
        """Test agent initialization."""
        assert quality_assessment_agent is not None
        assert hasattr(quality_assessment_agent, 'assess_quality')
        assert quality_assessment_agent.name == "Quality Assessment Agent"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_assess_quality_basic(self, quality_assessment_agent, sample_python_code):
        """Test basic quality assessment."""
        chunks = [
            {
                'content': 'def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)',
                'start_line': 1,
                'end_line': 4,
                'chunk_type': 'function'
            },
            {
                'content': 'class Calculator:\n    def __init__(self):\n        self.history = []',
                'start_line': 5,
                'end_line': 7,
                'chunk_type': 'class'
            }
        ]

        assessment = await quality_assessment_agent.assess_quality(
            content=sample_python_code,
            language="python",
            file_path="test.py",
            chunks=chunks
        )

        assert isinstance(assessment, dict)
        required_keys = ['overall_score', 'chunk_scores', 'improvement_suggestions']
        for key in required_keys:
            assert key in assessment, f"Assessment missing key: {key}"

        assert isinstance(assessment['overall_score'], (int, float))
        assert 0.0 <= assessment['overall_score'] <= 1.0
        assert isinstance(assessment['chunk_scores'], list)
        assert len(assessment['chunk_scores']) == len(chunks)
        assert isinstance(assessment['improvement_suggestions'], list)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_assess_quality_empty_chunks(self, quality_assessment_agent, sample_python_code):
        """Test quality assessment with empty chunks."""
        assessment = await quality_assessment_agent.assess_quality(
            content=sample_python_code,
            language="python",
            file_path="test.py",
            chunks=[]
        )

        assert isinstance(assessment, dict)
        assert 'overall_score' in assessment
        assert assessment['overall_score'] == 0.0  # No chunks should result in 0 score

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_assess_quality_performance(self, quality_assessment_agent):
        """Test quality assessment performance."""
        complex_code = generate_test_code("python", "complex")
        chunks = [
            {
                'content': complex_code[:100],
                'start_line': 1,
                'end_line': 10,
                'chunk_type': 'function'
            }
        ]

        import time
        start_time = time.time()

        assessment = await quality_assessment_agent.assess_quality(
            content=complex_code,
            language="python",
            file_path="complex.py",
            chunks=chunks
        )

        end_time = time.time()

        assert isinstance(assessment, dict)
        assert_performance_acceptable(end_time - start_time, max_time=10.0)


class TestAgentIntegration:
    """Test cases for agent integration and workflows."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_multi_agent_workflow(
        self,
        structure_analysis_agent,
        semantic_evaluation_agent,
        context_optimization_agent,
        quality_assessment_agent,
        sample_python_code
    ):
        """Test complete multi-agent workflow."""
        # Step 1: Structure analysis
        boundaries = await structure_analysis_agent.analyze_structure(
            content=sample_python_code,
            language="python",
            file_path="test.py"
        )

        assert len(boundaries) > 0

        # Step 2: Semantic evaluation
        evaluation = await semantic_evaluation_agent.evaluate_relationships(
            content=sample_python_code,
            language="python",
            file_path="test.py",
            boundaries=boundaries
        )

        assert 'coherence_score' in evaluation

        # Step 3: Context optimization
        optimization = await context_optimization_agent.optimize_for_purpose(
            content=sample_python_code,
            language="python",
            file_path="test.py",
            purpose="general",
            initial_boundaries=boundaries
        )

        assert 'optimized_boundaries' in optimization

        # Step 4: Quality assessment
        chunks = [
            {
                'content': sample_python_code[boundary['start_line']:boundary['end_line']],
                'start_line': boundary['start_line'],
                'end_line': boundary['end_line'],
                'chunk_type': boundary['chunk_type']
            }
            for boundary in optimization['optimized_boundaries']
        ]

        assessment = await quality_assessment_agent.assess_quality(
            content=sample_python_code,
            language="python",
            file_path="test.py",
            chunks=chunks
        )

        assert 'overall_score' in assessment
        assert assessment['overall_score'] > 0.0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_agent_error_handling(self, structure_analysis_agent):
        """Test agent error handling with invalid input."""
        # Test with None content
        with pytest.raises((ValueError, TypeError)):
            await structure_analysis_agent.analyze_structure(
                content=None,
                language="python",
                file_path="test.py"
            )

        # Test with invalid language
        boundaries = await structure_analysis_agent.analyze_structure(
            content="print('hello')",
            language="invalid_language",
            file_path="test.invalid"
        )

        # Should handle gracefully and return empty or fallback result
        assert isinstance(boundaries, list)
