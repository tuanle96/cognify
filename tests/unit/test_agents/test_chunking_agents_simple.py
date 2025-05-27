"""Simple unit tests for chunking agents."""

import pytest
from unittest.mock import patch, AsyncMock

from app.services.agents.crew_agents.chunking_agents import (
    StructureAnalysisAgent,
    SemanticEvaluationAgent,
    ContextOptimizationAgent,
    QualityAssessmentAgent
)


class TestAgentInitialization:
    """Test cases for agent initialization."""

    @pytest.mark.unit
    def test_structure_analysis_agent_init(self):
        """Test StructureAnalysisAgent initialization."""
        agent = StructureAnalysisAgent()

        assert agent is not None
        assert agent.role == "Code Structure Analysis Specialist"
        assert agent.goal == "Analyze code structure and identify optimal initial boundaries for chunking"
        assert hasattr(agent, 'analyze_structure')
        assert not agent._initialized

    @pytest.mark.unit
    def test_semantic_evaluation_agent_init(self):
        """Test SemanticEvaluationAgent initialization."""
        agent = SemanticEvaluationAgent()

        assert agent is not None
        assert agent.role == "Semantic Relationship Analyst"
        assert agent.goal == "Evaluate semantic coherence and optimize chunk groupings for maximum meaning preservation"
        assert hasattr(agent, 'evaluate_semantic_groups')
        assert not agent._initialized

    @pytest.mark.unit
    def test_context_optimization_agent_init(self):
        """Test ContextOptimizationAgent initialization."""
        agent = ContextOptimizationAgent()

        assert agent is not None
        assert agent.role == "Context-Aware Optimization Specialist"
        assert agent.goal == "Optimize chunk boundaries and content based on specific purpose and contextual requirements"
        assert hasattr(agent, 'optimize_for_purpose')
        assert not agent._initialized

    @pytest.mark.unit
    def test_quality_assessment_agent_init(self):
        """Test QualityAssessmentAgent initialization."""
        agent = QualityAssessmentAgent()

        assert agent is not None
        assert agent.role == "Chunking Quality Evaluator"
        assert agent.goal == "Assess chunking quality and provide continuous improvement recommendations"
        assert hasattr(agent, 'assess_chunking_quality')
        assert not agent._initialized


class TestAgentBasicFunctionality:
    """Test cases for basic agent functionality."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_structure_analysis_agent_fallback(self):
        """Test StructureAnalysisAgent fallback behavior."""
        agent = StructureAnalysisAgent()

        # Test fallback when LLM service fails
        with patch.object(agent, 'execute_task', side_effect=Exception("LLM service failed")):
            result = await agent.analyze_structure(
                content="def hello():\n    print('hello')",
                language="python",
                file_path="test.py"
            )

            assert isinstance(result, list)
            assert len(result) == 1  # Should return fallback result
            assert result[0]['chunk_type'] == 'module'
            assert result[0]['name'] == 'entire_file'

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_semantic_evaluation_agent_fallback(self):
        """Test SemanticEvaluationAgent fallback behavior."""
        agent = SemanticEvaluationAgent()

        initial_chunks = [
            {'name': 'test_chunk', 'start_line': 1, 'end_line': 10}
        ]

        # Test fallback when LLM service fails
        with patch.object(agent, 'execute_task', side_effect=Exception("LLM service failed")):
            result = await agent.evaluate_semantic_groups(
                initial_chunks=initial_chunks,
                content="def hello():\n    print('hello')"
            )

            assert isinstance(result, list)
            assert len(result) == 1  # Should return fallback result
            assert result[0]['chunks_to_group'] == [0]
            assert result[0]['relationship_type'] == 'standalone'

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_context_optimization_agent_fallback(self):
        """Test ContextOptimizationAgent fallback behavior."""
        agent = ContextOptimizationAgent()

        semantic_groups = [
            {'suggested_name': 'test_group'}
        ]
        context = {'file_path': 'test.py', 'language': 'python'}

        # Test fallback when LLM service fails
        with patch.object(agent, 'execute_task', side_effect=Exception("LLM service failed")):
            result = await agent.optimize_for_purpose(
                semantic_groups=semantic_groups,
                purpose="general",
                context=context
            )

            assert isinstance(result, list)
            assert len(result) == 1  # Should return fallback result
            assert result[0].name == 'test_group'
            assert result[0].chunk_type.value == 'semantic_block'

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_quality_assessment_agent_fallback(self):
        """Test QualityAssessmentAgent fallback behavior."""
        agent = QualityAssessmentAgent()

        # Mock AgenticChunk
        mock_chunk = type('MockChunk', (), {
            'size_lines': 10,
            'id': 'test_id',
            'name': 'test_chunk',
            'chunk_type': type('MockType', (), {'value': 'function'})(),
            'start_line': 1,
            'end_line': 10,
            'content': 'def test(): pass'
        })()

        chunks = [mock_chunk]

        # Test fallback when LLM service fails
        with patch.object(agent, 'execute_task', side_effect=Exception("LLM service failed")):
            result = await agent.assess_chunking_quality(
                chunks=chunks,
                original_content="def test(): pass\n" * 10,
                purpose="general"
            )

            assert isinstance(result, dict)
            assert 'overall_quality_score' in result
            assert 'semantic_coherence' in result
            assert result['overall_quality_score'] == 0.7  # Fallback score


class TestAgentPerformance:
    """Test cases for agent performance."""

    @pytest.mark.unit
    def test_agent_performance_stats(self):
        """Test agent performance statistics."""
        agent = StructureAnalysisAgent()

        stats = agent.get_performance_stats()

        assert isinstance(stats, dict)
        assert 'role' in stats
        assert 'execution_count' in stats
        assert 'success_count' in stats
        assert 'failure_count' in stats
        assert 'success_rate' in stats
        assert 'initialized' in stats

        assert stats['role'] == "Code Structure Analysis Specialist"
        assert stats['execution_count'] == 0
        assert stats['success_count'] == 0
        assert stats['failure_count'] == 0
        assert stats['success_rate'] == 0.0
        assert stats['initialized'] is False

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_agent_cleanup(self):
        """Test agent cleanup."""
        agent = StructureAnalysisAgent()

        # Mock LLM service
        mock_llm_service = AsyncMock()
        agent._llm_service = mock_llm_service

        await agent.cleanup()

        # Should call cleanup on LLM service
        mock_llm_service.cleanup.assert_called_once()


class TestAgentErrorHandling:
    """Test cases for agent error handling."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_structure_analysis_with_none_content(self):
        """Test structure analysis with None content."""
        agent = StructureAnalysisAgent()

        # Should handle None content gracefully
        result = await agent.analyze_structure(
            content=None,
            language="python",
            file_path="test.py"
        )

        # Should return fallback result
        assert isinstance(result, list)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_semantic_evaluation_with_empty_chunks(self):
        """Test semantic evaluation with empty chunks."""
        agent = SemanticEvaluationAgent()

        result = await agent.evaluate_semantic_groups(
            initial_chunks=[],
            content="def hello(): pass"
        )

        # Should handle empty chunks gracefully
        assert isinstance(result, list)
        assert len(result) == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_context_optimization_with_empty_groups(self):
        """Test context optimization with empty semantic groups."""
        agent = ContextOptimizationAgent()

        result = await agent.optimize_for_purpose(
            semantic_groups=[],
            purpose="general",
            context={'file_path': 'test.py', 'language': 'python'}
        )

        # Should handle empty groups gracefully
        assert isinstance(result, list)
        assert len(result) == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_quality_assessment_with_empty_chunks(self):
        """Test quality assessment with empty chunks."""
        agent = QualityAssessmentAgent()

        result = await agent.assess_chunking_quality(
            chunks=[],
            original_content="def test(): pass",
            purpose="general"
        )

        # Should handle empty chunks gracefully
        assert isinstance(result, dict)
        assert 'overall_quality_score' in result
