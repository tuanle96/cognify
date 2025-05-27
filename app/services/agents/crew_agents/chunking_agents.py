"""
Agentic chunking agents using LLM services for intelligent decision making.

These agents collaborate to make intelligent chunking decisions based on
code structure, semantic relationships, and purpose optimization.
"""

import json
from typing import Dict, List
import structlog

from app.core.config import get_settings
from app.services.chunking.base import AgenticChunk, ChunkType, ChunkMetadata, generate_chunk_id, normalize_chunk_type
from app.services.agents.base import BaseAgent

logger = structlog.get_logger(__name__)
settings = get_settings()


class StructureAnalysisAgent(BaseAgent):
    """
    Agent specialized in analyzing code structure and identifying optimal boundaries.
    """

    def __init__(self):
        super().__init__(
            role="Code Structure Analysis Specialist",
            goal="Analyze code structure and identify optimal initial boundaries for chunking",
            backstory="""You are an expert in code structure analysis with deep understanding of programming
            languages, syntax patterns, and logical code organization. You excel at identifying natural
            boundaries in code that preserve semantic meaning and logical flow.""",
            max_iterations=3,
            temperature=0.1,
            verbose=True
        )

    async def analyze_structure(self, content: str, language: str, file_path: str) -> List[Dict]:
        """
        Analyze code structure and propose initial chunk boundaries.

        Args:
            content: Source code content
            language: Programming language
            file_path: File path for context

        Returns:
            List of proposed chunk boundaries with reasoning
        """
        # Optimize prompt for efficiency - truncate very long content
        content_preview = content if len(content) <= 1000 else content[:1000] + "\n# ... (truncated)"

        task_description = f"""Analyze {language} code for chunking boundaries:

```{language}
{content_preview}
```

File: {file_path} ({len(content.split('\n'))} lines)

Find boundaries for functions, classes, imports. Return JSON:
[{{"start_line": 1, "end_line": 10, "chunk_type": "function", "name": "func_name", "reasoning": "brief", "complexity_estimate": 3, "dependencies": []}}]"""

        try:
            result = await self.execute_task(task_description, {
                "language": language,
                "file_path": file_path,
                "content_length": len(content)
            })
            return self._parse_boundary_result(result)
        except Exception as e:
            self.logger.error("Structure analysis failed", error=str(e), file_path=file_path)
            return self._fallback_structure_analysis(content, language, file_path)

    def _parse_boundary_result(self, result: str) -> List[Dict]:
        """Parse the agent's boundary analysis result."""
        try:
            # Try to extract JSON from the result
            if isinstance(result, str):
                # Look for JSON array in the response
                start_idx = result.find('[')
                end_idx = result.rfind(']') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = result[start_idx:end_idx]
                    return json.loads(json_str)

            # If result is already parsed
            if isinstance(result, list):
                return result

            return []
        except Exception as e:
            logger.warning("Failed to parse boundary result", error=str(e))
            return []

    def _fallback_structure_analysis(self, content: str, language: str, file_path: str) -> List[Dict]:
        """Fallback structure analysis when agent fails - create intelligent chunks."""
        if not content:
            return []

        lines = content.split('\n')
        chunks = []

        # Simple intelligent chunking based on code structure
        current_chunk_start = 1
        current_chunk_lines = []

        for i, line in enumerate(lines, 1):
            current_chunk_lines.append(line)

            # Check for natural boundaries (functions, classes, etc.)
            stripped = line.strip()
            is_boundary = (
                stripped.startswith('def ') or
                stripped.startswith('class ') or
                stripped.startswith('async def ') or
                (stripped.startswith('#') and len(stripped) > 10) or  # Comments as separators
                (i > 0 and not stripped and len(current_chunk_lines) > 10)  # Empty lines after substantial content
            )

            # Create chunk if we hit a boundary or reach reasonable size
            if (is_boundary and len(current_chunk_lines) > 5) or len(current_chunk_lines) > 50:
                if len(current_chunk_lines) > 1:  # Don't create single-line chunks
                    chunk_content = '\n'.join(current_chunk_lines[:-1] if is_boundary else current_chunk_lines)
                    chunk_type = self._detect_chunk_type(chunk_content)

                    chunks.append({
                        "start_line": current_chunk_start,
                        "end_line": i - (1 if is_boundary else 0),
                        "chunk_type": chunk_type,
                        "name": self._generate_chunk_name(chunk_content, chunk_type),
                        "reasoning": f"Fallback: {chunk_type} chunk with {len(current_chunk_lines)} lines",
                        "complexity_estimate": min(len(current_chunk_lines) // 10 + 1, 10),
                        "dependencies": []
                    })

                    current_chunk_start = i if is_boundary else i + 1
                    current_chunk_lines = [line] if is_boundary else []

        # Add final chunk if there's remaining content
        if current_chunk_lines:
            chunk_content = '\n'.join(current_chunk_lines)
            chunk_type = self._detect_chunk_type(chunk_content)
            chunks.append({
                "start_line": current_chunk_start,
                "end_line": len(lines),
                "chunk_type": chunk_type,
                "name": self._generate_chunk_name(chunk_content, chunk_type),
                "reasoning": f"Fallback: final {chunk_type} chunk",
                "complexity_estimate": min(len(current_chunk_lines) // 10 + 1, 10),
                "dependencies": []
            })

        # If no chunks were created, create one for the entire file
        if not chunks:
            chunks.append({
                "start_line": 1,
                "end_line": len(lines),
                "chunk_type": "module",
                "name": "entire_file",
                "reasoning": "Fallback: entire file as single chunk",
                "complexity_estimate": 5,
                "dependencies": []
            })

        return chunks

    def _detect_chunk_type(self, content: str) -> str:
        """Detect chunk type based on content."""
        if 'class ' in content:
            return 'class'
        elif 'def ' in content or 'async def ' in content:
            return 'function'
        elif 'import ' in content or 'from ' in content:
            return 'import'
        else:
            return 'code_block'

    def _generate_chunk_name(self, content: str, chunk_type: str) -> str:
        """Generate a meaningful name for the chunk."""
        lines = content.strip().split('\n')

        for line in lines:
            stripped = line.strip()
            if stripped.startswith('class '):
                return stripped.split('(')[0].replace('class ', '').strip()
            elif stripped.startswith('def ') or stripped.startswith('async def '):
                func_name = stripped.split('(')[0].replace('def ', '').replace('async ', '').strip()
                return func_name

        # Fallback to generic names
        if chunk_type == 'import':
            return 'imports'
        elif chunk_type == 'class':
            return 'class_definition'
        elif chunk_type == 'function':
            return 'function_definition'
        else:
            return 'code_block'


class SemanticEvaluationAgent(BaseAgent):
    """
    Agent specialized in evaluating semantic relationships and optimizing groupings.
    """

    def __init__(self):
        super().__init__(
            role="Semantic Relationship Analyst",
            goal="Evaluate semantic coherence and optimize chunk groupings for maximum meaning preservation",
            backstory="""You are an expert in semantic analysis and code comprehension with a deep understanding
            of how different parts of code relate to each other. You can identify when code should be grouped
            together to preserve meaning, context, and logical flow. You understand design patterns, code
            relationships, and how to maintain semantic integrity.""",
            max_iterations=4,
            temperature=0.1,
            verbose=True
        )

    async def evaluate_semantic_groups(self, initial_chunks: List[Dict], content: str) -> List[Dict]:
        """
        Evaluate and optimize chunk groupings based on semantic relationships.

        Args:
            initial_chunks: Initial chunk boundaries from structure analysis
            content: Full source code content

        Returns:
            List of optimized chunk groups with semantic analysis
        """
        # Optimize prompt - summarize chunks and truncate content
        chunks_summary = [
            {"idx": i, "type": chunk.get("chunk_type", "unknown"), "name": chunk.get("name", f"chunk_{i}"),
             "lines": f"{chunk.get('start_line', 0)}-{chunk.get('end_line', 0)}"}
            for i, chunk in enumerate(initial_chunks[:10])  # Limit to first 10 chunks
        ]

        content_preview = content[:1000] + "..." if len(content) > 1000 else content

        task_description = f"""Optimize semantic groupings for {len(initial_chunks)} code chunks:

Chunks: {json.dumps(chunks_summary, separators=(',', ':'))}

Context preview:
```
{content_preview}
```

Group related chunks by:
• Functional relationships (calls, dependencies)
• Shared purpose/domain
• Architectural patterns

Return JSON:
[{{"chunks_to_group": [0,1], "semantic_reasoning": "brief reason", "relationship_type": "collaboration", "coherence_score": 0.9, "suggested_name": "group_name"}}]"""

        try:
            result = await self.execute_task(task_description, {
                "initial_chunks_count": len(initial_chunks),
                "content_length": len(content)
            })
            return self._parse_semantic_result(result)
        except Exception as e:
            self.logger.error("Semantic evaluation failed", error=str(e))
            return self._fallback_semantic_evaluation(initial_chunks)

    def _parse_semantic_result(self, result: str) -> List[Dict]:
        """Parse the agent's semantic evaluation result."""
        try:
            if isinstance(result, str):
                start_idx = result.find('[')
                end_idx = result.rfind(']') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = result[start_idx:end_idx]
                    return json.loads(json_str)

            if isinstance(result, list):
                return result

            return []
        except Exception as e:
            logger.warning("Failed to parse semantic result", error=str(e))
            return []

    def _fallback_semantic_evaluation(self, initial_chunks: List[Dict]) -> List[Dict]:
        """Fallback semantic evaluation when agent fails."""
        # Return chunks as-is with basic grouping
        return [{
            "chunks_to_group": [i],
            "semantic_reasoning": "Fallback: keeping original chunk boundaries",
            "relationship_type": "standalone",
            "coherence_score": 0.7,
            "suggested_name": chunk.get("name", f"chunk_{i}")
        } for i, chunk in enumerate(initial_chunks)]


class ContextOptimizationAgent(BaseAgent):
    """
    Agent specialized in optimizing chunks based on purpose and context.
    """

    def __init__(self):
        super().__init__(
            role="Context-Aware Optimization Specialist",
            goal="Optimize chunk boundaries and content based on specific purpose and contextual requirements",
            backstory="""You are an expert in context-aware optimization with deep understanding of how different
            use cases require different chunking strategies. You excel at adapting chunk boundaries and content to
            maximize effectiveness for specific purposes like code review, bug detection, or documentation generation.
            You understand user intent and can optimize accordingly.""",
            max_iterations=3,
            temperature=0.1,
            verbose=True
        )

    async def optimize_for_purpose(self, semantic_groups: List[Dict], purpose: str, context: Dict) -> List[AgenticChunk]:
        """
        Optimize chunks based on specific purpose and context.

        Args:
            semantic_groups: Semantic grouping recommendations
            purpose: Chunking purpose (code_review, bug_detection, documentation, general)
            context: Additional context information

        Returns:
            List of purpose-optimized AgenticChunk objects
        """
        # Optimize prompt - use purpose-specific guidelines and compact format
        purpose_guidelines = {
            "code_review": "Group related functions, 100-200 lines, focus readability",
            "bug_detection": "Isolate error patterns, 50-150 lines, emphasize edge cases",
            "documentation": "Group public APIs, 150-300 lines, focus interfaces",
            "general": "Balance comprehensiveness, 100-200 lines, preserve boundaries"
        }

        guideline = purpose_guidelines.get(purpose.lower(), purpose_guidelines["general"])
        groups_summary = [{"name": g.get("suggested_name", f"group_{i}")} for i, g in enumerate(semantic_groups[:5])]

        task_description = f"""Optimize {len(semantic_groups)} semantic groups for {purpose}:

Groups: {json.dumps(groups_summary, separators=(',', ':'))}
File: {context.get('file_path', 'unknown')}

Guideline: {guideline}

Valid chunk_types: function, class, method, module, import, comment, semantic_block, class_definition, function_definition, method_definition, imports, variable, constant, interface, struct, enum, namespace, package, configuration, documentation, test, utility, helper, main, initialization, unknown

Return JSON array:
[{{"final_content": "code here", "chunk_type": "function", "name": "chunk_name", "start_line": 1, "end_line": 10, "purpose_optimization": {{"purpose": "{purpose}"}}, "quality_reasoning": "brief reason"}}]"""

        try:
            result = await self.execute_task(task_description, {
                "purpose": purpose,
                "semantic_groups_count": len(semantic_groups),
                "context": context
            })
            return self._create_agentic_chunks(result, purpose, context)
        except Exception as e:
            self.logger.error("Context optimization failed", error=str(e), purpose=purpose)
            return self._fallback_context_optimization(semantic_groups, purpose, context)

    def _create_agentic_chunks(self, result: str, purpose: str, context: Dict) -> List[AgenticChunk]:
        """Create AgenticChunk objects from optimization result."""
        try:
            chunks_data = self._parse_optimization_result(result)
            agentic_chunks = []

            for i, chunk_data in enumerate(chunks_data):
                try:
                    # Safely get chunk type with fallback
                    chunk_type_str = chunk_data.get("chunk_type", "semantic_block")
                    chunk_type = normalize_chunk_type(chunk_type_str)

                    chunk = AgenticChunk(
                        id=generate_chunk_id(
                            context.get("file_path", "unknown"),
                            chunk_data.get("start_line", 1),
                            chunk_data.get("end_line", 1)
                        ),
                        content=chunk_data.get("final_content", ""),
                        language=context.get("language", "unknown"),
                        chunk_type=chunk_type,
                        name=chunk_data.get("name", f"chunk_{i}"),
                        start_line=chunk_data.get("start_line", 1),
                        end_line=chunk_data.get("end_line", 1),
                        file_path=context.get("file_path", "unknown"),
                        dependencies=chunk_data.get("dependencies", []),
                        semantic_relationships=chunk_data.get("semantic_relationships", []),
                        purpose_optimization={
                            "purpose": purpose,
                            "reasoning": chunk_data.get("quality_reasoning", ""),
                            "optimization_metadata": chunk_data.get("purpose_optimization", {})
                        },
                        quality_score=0.9,  # High quality for agentic chunks
                        metadata=ChunkMetadata(
                            complexity=chunk_data.get("complexity_estimate"),
                            dependencies=chunk_data.get("dependencies", [])
                        )
                    )
                    agentic_chunks.append(chunk)

                except Exception as chunk_error:
                    logger.warning(
                        "Failed to create individual chunk",
                        error=str(chunk_error),
                        chunk_index=i,
                        chunk_data=chunk_data
                    )
                    continue

            return agentic_chunks

        except Exception as e:
            logger.error("Failed to create agentic chunks", error=str(e))
            return []

    def _parse_optimization_result(self, result: str) -> List[Dict]:
        """Parse the optimization result."""
        try:
            if isinstance(result, str):
                start_idx = result.find('[')
                end_idx = result.rfind(']') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = result[start_idx:end_idx]
                    return json.loads(json_str)

            if isinstance(result, list):
                return result

            return []
        except Exception as e:
            logger.warning("Failed to parse optimization result", error=str(e))
            return []

    def _fallback_context_optimization(self, semantic_groups: List[Dict], purpose: str, context: Dict) -> List[AgenticChunk]:
        """Fallback optimization when agent fails - create real chunks from content."""
        chunks = []
        original_content = context.get("original_content", "")

        if not original_content:
            # If no original content, create minimal chunks
            for i, group in enumerate(semantic_groups):
                chunk = AgenticChunk(
                    id=generate_chunk_id(context.get("file_path", "unknown"), i * 10, (i + 1) * 10),
                    content=f"# Fallback chunk {i}\n# No original content available",
                    language=context.get("language", "unknown"),
                    chunk_type=ChunkType.SEMANTIC_BLOCK,
                    name=group.get("suggested_name", f"fallback_chunk_{i}"),
                    start_line=i * 10,
                    end_line=(i + 1) * 10,
                    file_path=context.get("file_path", "unknown"),
                    dependencies=[],
                    semantic_relationships=[],
                    purpose_optimization={"purpose": purpose, "reasoning": "Fallback optimization"},
                    quality_score=0.6,  # Lower quality for fallback
                    metadata=ChunkMetadata()
                )
                chunks.append(chunk)
        else:
            # Create chunks from original content using semantic groups
            lines = original_content.split('\n')

            for i, group in enumerate(semantic_groups):
                # Get chunks to group from the semantic group
                chunks_to_group = group.get("chunks_to_group", [i])

                # Calculate line ranges for this group
                start_line = 1 + (i * len(lines) // len(semantic_groups))
                end_line = min(start_line + (len(lines) // len(semantic_groups)), len(lines))

                # Extract content for this chunk
                chunk_lines = lines[start_line-1:end_line]
                chunk_content = '\n'.join(chunk_lines)

                # Determine chunk type
                chunk_type = self._determine_chunk_type_from_content(chunk_content)

                chunk = AgenticChunk(
                    id=generate_chunk_id(context.get("file_path", "unknown"), start_line, end_line),
                    content=chunk_content,
                    language=context.get("language", "unknown"),
                    chunk_type=chunk_type,
                    name=group.get("suggested_name", f"chunk_{i}"),
                    start_line=start_line,
                    end_line=end_line,
                    file_path=context.get("file_path", "unknown"),
                    dependencies=[],
                    semantic_relationships=[],
                    purpose_optimization={"purpose": purpose, "reasoning": "Fallback optimization with real content"},
                    quality_score=0.7,  # Better quality with real content
                    metadata=ChunkMetadata()
                )
                chunks.append(chunk)

        return chunks

    def _determine_chunk_type_from_content(self, content: str) -> ChunkType:
        """Determine chunk type from content analysis."""
        if 'class ' in content:
            return ChunkType.CLASS
        elif 'def ' in content or 'async def ' in content:
            return ChunkType.FUNCTION
        elif 'import ' in content or 'from ' in content:
            return ChunkType.IMPORT
        else:
            return ChunkType.SEMANTIC_BLOCK


class QualityAssessmentAgent(BaseAgent):
    """
    Agent specialized in assessing and improving chunking quality.
    """

    def __init__(self):
        super().__init__(
            role="Chunking Quality Evaluator",
            goal="Assess chunking quality and provide continuous improvement recommendations",
            backstory="""You are an expert in quality assessment with extensive experience in evaluating
            code chunking effectiveness. You can identify quality issues, measure semantic coherence,
            and provide actionable recommendations for improvement. You understand what makes code chunks
            effective for different purposes and can spot potential issues.""",
            max_iterations=4,
            temperature=0.1,
            verbose=True
        )

    async def assess_chunking_quality(self, chunks: List[AgenticChunk], original_content: str, purpose: str) -> Dict:
        """
        Assess the quality of chunking decisions across multiple dimensions.

        Args:
            chunks: List of chunks to evaluate
            original_content: Original content before chunking
            purpose: Chunking purpose

        Returns:
            Dictionary with quality scores and recommendations
        """
        # Prepare chunks summary for the agent
        chunks_summary = []
        for chunk in chunks:
            chunks_summary.append({
                "id": chunk.id,
                "name": chunk.name,
                "type": chunk.chunk_type.value,
                "lines": f"{chunk.start_line}-{chunk.end_line}",
                "size": chunk.size_lines,
                "content_preview": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
            })

        # Optimize prompt - compact format and focus on key metrics
        chunks_compact = [
            {"name": chunk["name"], "type": chunk["type"], "size": chunk["size"]}
            for chunk in chunks_summary[:5]  # Limit to first 5 chunks
        ]

        task_description = f"""Assess chunking quality for {purpose} ({len(chunks)} chunks, {len(original_content.split('\n'))} lines):

Chunks: {json.dumps(chunks_compact, separators=(',', ':'))}

Rate 0.0-1.0:
• Semantic coherence (logical grouping)
• Context preservation (understandable independently)
• Purpose alignment (fits {purpose} use case)
• Completeness (captures all important code)
• Efficiency (optimal sizes, minimal redundancy)

Return JSON:
{{"semantic_coherence": {{"score": 0.8, "reasoning": "brief reason", "specific_issues": [], "improvement_suggestions": []}}, "context_preservation": {{"score": 0.9, "reasoning": "brief", "specific_issues": [], "improvement_suggestions": []}}, "purpose_alignment": {{"score": 0.7, "reasoning": "brief", "specific_issues": [], "improvement_suggestions": []}}, "completeness": {{"score": 0.9, "reasoning": "brief", "specific_issues": [], "improvement_suggestions": []}}, "efficiency": {{"score": 0.8, "reasoning": "brief", "specific_issues": [], "improvement_suggestions": []}}, "overall_quality_score": 0.82, "critical_issues": [], "optimization_opportunities": []}}"""

        try:
            result = await self.execute_task(task_description, {
                "purpose": purpose,
                "chunks_count": len(chunks),
                "original_content_lines": len(original_content.split('\n'))
            })
            return self._parse_quality_assessment(result)
        except Exception as e:
            self.logger.error("Quality assessment failed", error=str(e), purpose=purpose)
            return self._fallback_quality_assessment(chunks, original_content, purpose)

    async def generate_improvement_suggestions(self, quality_assessment: Dict, feedback_history: List[Dict]) -> List[Dict]:
        """
        Generate specific improvement suggestions based on quality assessment and feedback.

        Args:
            quality_assessment: Current quality assessment results
            feedback_history: Historical feedback data

        Returns:
            List of prioritized improvement suggestions
        """
        task_description = f"""
        Based on this quality assessment and historical feedback, generate specific improvement suggestions:

        Current Quality Assessment: {json.dumps(quality_assessment, indent=2)}

        Historical Feedback: {json.dumps(feedback_history[-10:] if feedback_history else [], indent=2)}

        Generate actionable improvements for:
        1. Chunk boundary adjustments
        2. Semantic grouping optimizations
        3. Purpose-specific parameter tuning
        4. Context preservation enhancements
        5. Performance optimizations
        6. Quality threshold adjustments

        For each suggestion, provide:
        - category (boundary, semantic, purpose, context, performance, threshold)
        - priority (high, medium, low)
        - description (what to change)
        - rationale (why this will improve quality)
        - implementation_steps (how to implement)
        - expected_impact (predicted improvement)
        - effort_estimate (low, medium, high)

        Prioritize suggestions by impact and feasibility.
        Return as a JSON array of improvement suggestions.
        """

        try:
            result = await self.execute_task(task_description, {
                "quality_assessment": quality_assessment,
                "feedback_history_count": len(feedback_history) if feedback_history else 0
            })
            return self._parse_improvement_suggestions(result)
        except Exception as e:
            self.logger.error("Improvement suggestion generation failed", error=str(e))
            return self._fallback_improvement_suggestions(quality_assessment)

    def _parse_quality_assessment(self, result: str) -> Dict:
        """Parse the agent's quality assessment result."""
        try:
            if isinstance(result, str):
                # Look for JSON object in the response
                start_idx = result.find('{')
                end_idx = result.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = result[start_idx:end_idx]
                    return json.loads(json_str)

            if isinstance(result, dict):
                return result

            return {}
        except Exception as e:
            logger.warning("Failed to parse quality assessment", error=str(e))
            return {}

    def _parse_improvement_suggestions(self, result: str) -> List[Dict]:
        """Parse the agent's improvement suggestions."""
        try:
            if isinstance(result, str):
                start_idx = result.find('[')
                end_idx = result.rfind(']') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = result[start_idx:end_idx]
                    return json.loads(json_str)

            if isinstance(result, list):
                return result

            return []
        except Exception as e:
            logger.warning("Failed to parse improvement suggestions", error=str(e))
            return []

    def _fallback_quality_assessment(self, chunks: List[AgenticChunk], original_content: str, _purpose: str) -> Dict:
        """Fallback quality assessment when agent fails."""
        total_lines = len(original_content.split('\n'))
        covered_lines = sum(chunk.size_lines for chunk in chunks)
        coverage = min(covered_lines / total_lines, 1.0) if total_lines > 0 else 0.0

        return {
            "semantic_coherence": {"score": 0.7, "reasoning": "Fallback assessment"},
            "context_preservation": {"score": coverage, "reasoning": f"Coverage: {coverage:.2%}"},
            "purpose_alignment": {"score": 0.6, "reasoning": "Basic purpose alignment"},
            "completeness": {"score": coverage, "reasoning": f"Line coverage: {coverage:.2%}"},
            "efficiency": {"score": 0.7, "reasoning": "Standard efficiency"},
            "overall_quality_score": 0.7,
            "critical_issues": ["Agent assessment failed - using fallback"],
            "optimization_opportunities": ["Implement proper agent assessment"]
        }

    def _fallback_improvement_suggestions(self, _quality_assessment: Dict) -> List[Dict]:
        """Fallback improvement suggestions when agent fails."""
        return [{
            "category": "system",
            "priority": "high",
            "description": "Fix agent assessment system",
            "rationale": "Agent-based assessment failed, need to investigate",
            "implementation_steps": ["Check agent configuration", "Verify LLM connectivity"],
            "expected_impact": "Enable proper quality assessment",
            "effort_estimate": "medium"
        }]
