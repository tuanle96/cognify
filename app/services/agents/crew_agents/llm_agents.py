"""
LLM-powered chunking agents for intelligent code analysis.

These agents use LLM services to make intelligent chunking decisions based on
code structure, semantic relationships, and purpose optimization.
"""

import json
from typing import Dict, List, Any
import structlog

from app.services.chunking.base import AgenticChunk, ChunkType, ChunkMetadata, generate_chunk_id
from app.services.llm.base import create_system_message, create_user_message
from app.services.llm.factory import get_default_llm_service

logger = structlog.get_logger(__name__)


class LLMStructureAnalysisAgent:
    """
    LLM-powered agent for analyzing code structure and identifying optimal boundaries.
    """
    
    def __init__(self):
        self.role = "Code Structure Analysis Specialist"
        self.logger = structlog.get_logger(f"agent.{self.__class__.__name__}")
    
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
        try:
            self.logger.info("Starting structure analysis", file_path=file_path, language=language)
            
            # Get LLM service
            llm_service = await get_default_llm_service()
            
            # Create messages for LLM
            system_message = create_system_message(
                "You are an expert software engineer specializing in code structure analysis. "
                "Your task is to analyze code and identify optimal boundaries for chunking while "
                "preserving semantic meaning and logical relationships."
            )
            
            user_message = create_user_message(f"""
            Analyze the structure of this {language} code and identify optimal boundaries for chunking:
            
            File: {file_path}
            Language: {language}
            Content:
            ```{language}
            {content}
            ```
            
            Your analysis should consider:
            1. Function and class boundaries
            2. Logical code blocks and modules
            3. Import/dependency sections
            4. Comment blocks and documentation
            5. Related code that should stay together
            6. Code complexity and readability
            
            For each proposed boundary, provide:
            - start_line and end_line (1-based)
            - chunk_type (function, class, method, module, import, comment, semantic_block)
            - name (identifier or descriptive name)
            - reasoning (why this boundary makes sense)
            - complexity_estimate (1-10 scale)
            - dependencies (list of identifiers this chunk depends on)
            
            Return your analysis as a JSON array of boundary objects.
            """)
            
            # Generate response
            response = await llm_service.generate([system_message, user_message])
            
            # Parse and return results
            return self._parse_boundary_result(response.content)
            
        except Exception as e:
            self.logger.error("Structure analysis failed", error=str(e), file_path=file_path)
            return self._fallback_structure_analysis(content, language, file_path)
    
    def _parse_boundary_result(self, result: str) -> List[Dict]:
        """Parse the LLM's boundary analysis result."""
        try:
            # Try to extract JSON from the result
            if isinstance(result, str):
                # Look for JSON array in the response
                start_idx = result.find('[')
                end_idx = result.rfind(']') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = result[start_idx:end_idx]
                    return json.loads(json_str)
            
            return []
        except Exception as e:
            self.logger.warning("Failed to parse boundary result", error=str(e))
            return []
    
    def _fallback_structure_analysis(self, content: str, language: str, file_path: str) -> List[Dict]:
        """Fallback structure analysis when LLM fails."""
        lines = content.split('\n')
        return [{
            "start_line": 1,
            "end_line": len(lines),
            "chunk_type": "module",
            "name": "entire_file",
            "reasoning": "Fallback: entire file as single chunk due to analysis failure",
            "complexity_estimate": 5,
            "dependencies": []
        }]


class LLMSemanticEvaluationAgent:
    """
    LLM-powered agent for evaluating semantic relationships and optimizing groupings.
    """
    
    def __init__(self):
        self.role = "Semantic Relationship Analyst"
        self.logger = structlog.get_logger(f"agent.{self.__class__.__name__}")
    
    async def evaluate_semantic_groups(self, initial_chunks: List[Dict], content: str) -> List[Dict]:
        """
        Evaluate and optimize chunk groupings based on semantic relationships.
        
        Args:
            initial_chunks: Initial chunk boundaries from structure analysis
            content: Full source code content
            
        Returns:
            List of optimized chunk groups with semantic analysis
        """
        try:
            self.logger.info("Starting semantic evaluation", chunk_count=len(initial_chunks))
            
            # Get LLM service
            llm_service = await get_default_llm_service()
            
            # Create messages for LLM
            system_message = create_system_message(
                "You are an expert in semantic analysis and code comprehension. "
                "Your task is to evaluate semantic relationships between code chunks and "
                "optimize their groupings for maximum meaning preservation."
            )
            
            user_message = create_user_message(f"""
            Analyze these initial code chunks and optimize their groupings based on semantic relationships:
            
            Initial Chunks: {json.dumps(initial_chunks, indent=2)}
            
            Full Content Context:
            ```
            {content[:1500]}...
            ```
            
            Evaluate and optimize based on:
            1. Semantic coherence within each chunk
            2. Relationships between chunks that should be grouped together
            3. Dependencies and call relationships
            4. Shared context and purpose
            5. Design patterns and architectural relationships
            6. Data flow and control flow
            
            For each optimized group, provide:
            - chunks_to_group (list of chunk indices that should be combined)
            - semantic_reasoning (why these chunks belong together)
            - relationship_type (dependency, collaboration, composition, etc.)
            - coherence_score (0.0-1.0, how well the group holds together)
            - suggested_name (descriptive name for the grouped chunk)
            
            Return your analysis as a JSON array of semantic grouping recommendations.
            """)
            
            # Generate response
            response = await llm_service.generate([system_message, user_message])
            
            # Parse and return results
            return self._parse_semantic_result(response.content)
            
        except Exception as e:
            self.logger.error("Semantic evaluation failed", error=str(e))
            return self._fallback_semantic_evaluation(initial_chunks)
    
    def _parse_semantic_result(self, result: str) -> List[Dict]:
        """Parse the LLM's semantic evaluation result."""
        try:
            if isinstance(result, str):
                start_idx = result.find('[')
                end_idx = result.rfind(']') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = result[start_idx:end_idx]
                    return json.loads(json_str)
            
            return []
        except Exception as e:
            self.logger.warning("Failed to parse semantic result", error=str(e))
            return []
    
    def _fallback_semantic_evaluation(self, initial_chunks: List[Dict]) -> List[Dict]:
        """Fallback semantic evaluation when LLM fails."""
        # Return chunks as-is with basic grouping
        return [{
            "chunks_to_group": [i],
            "semantic_reasoning": "Fallback: keeping original chunk boundaries",
            "relationship_type": "standalone",
            "coherence_score": 0.7,
            "suggested_name": chunk.get("name", f"chunk_{i}")
        } for i, chunk in enumerate(initial_chunks)]


class LLMContextOptimizationAgent:
    """
    LLM-powered agent for optimizing chunks based on purpose and context.
    """
    
    def __init__(self):
        self.role = "Context-Aware Optimization Specialist"
        self.logger = structlog.get_logger(f"agent.{self.__class__.__name__}")
    
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
        try:
            self.logger.info("Starting context optimization", purpose=purpose)
            
            # Get LLM service
            llm_service = await get_default_llm_service()
            
            # Create messages for LLM
            system_message = create_system_message(
                "You are an expert in context-aware optimization with deep understanding of how "
                "different use cases require different chunking strategies. Your task is to optimize "
                "chunk boundaries and content for specific purposes."
            )
            
            user_message = create_user_message(f"""
            Optimize these semantic chunk groups for the specific purpose: {purpose}
            
            Semantic Groups: {json.dumps(semantic_groups, indent=2)}
            Context: {json.dumps(context, indent=2)}
            
            Purpose-specific optimization guidelines:
            
            For CODE_REVIEW:
            - Group related functions that should be reviewed together
            - Include sufficient context for understanding changes
            - Target chunk size: 100-200 lines, overlap: 10 lines
            
            For BUG_DETECTION:
            - Isolate error-prone patterns and exception handling
            - Include control flow context
            - Target chunk size: 50-150 lines, overlap: 20 lines
            
            For DOCUMENTATION:
            - Group by user-facing functionality
            - Include public interfaces and API boundaries
            - Target chunk size: 150-300 lines, overlap: 5 lines
            
            For GENERAL_ANALYSIS:
            - Balance between comprehensiveness and specificity
            - Target chunk size: 100-200 lines, overlap: 15 lines
            
            For each optimized chunk, provide:
            - final_content (the actual code content)
            - chunk_type (function, class, method, etc.)
            - name (descriptive identifier)
            - start_line and end_line
            - purpose_optimization (purpose-specific metadata)
            - quality_reasoning (why this chunking is optimal for the purpose)
            
            Return optimized chunks as a JSON array.
            """)
            
            # Generate response
            response = await llm_service.generate([system_message, user_message])
            
            # Parse and create AgenticChunk objects
            return self._create_agentic_chunks(response.content, purpose, context)
            
        except Exception as e:
            self.logger.error("Context optimization failed", error=str(e), purpose=purpose)
            return self._fallback_context_optimization(semantic_groups, purpose, context)
    
    def _create_agentic_chunks(self, result: str, purpose: str, context: Dict) -> List[AgenticChunk]:
        """Create AgenticChunk objects from optimization result."""
        try:
            chunks_data = self._parse_optimization_result(result)
            agentic_chunks = []
            
            for chunk_data in chunks_data:
                chunk = AgenticChunk(
                    id=generate_chunk_id(
                        context.get("file_path", "unknown"),
                        chunk_data.get("start_line", 1),
                        chunk_data.get("end_line", 1)
                    ),
                    content=chunk_data.get("final_content", ""),
                    language=context.get("language", "unknown"),
                    chunk_type=ChunkType(chunk_data.get("chunk_type", "unknown")),
                    name=chunk_data.get("name", "unnamed"),
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
                    quality_score=0.9,  # High quality for LLM chunks
                    metadata=ChunkMetadata(
                        complexity=chunk_data.get("complexity_estimate"),
                        dependencies=chunk_data.get("dependencies", [])
                    )
                )
                agentic_chunks.append(chunk)
            
            return agentic_chunks
            
        except Exception as e:
            self.logger.error("Failed to create agentic chunks", error=str(e))
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
            
            return []
        except Exception as e:
            self.logger.warning("Failed to parse optimization result", error=str(e))
            return []
    
    def _fallback_context_optimization(self, semantic_groups: List[Dict], purpose: str, context: Dict) -> List[AgenticChunk]:
        """Fallback optimization when LLM fails."""
        # Create basic chunks from semantic groups
        chunks = []
        for i, group in enumerate(semantic_groups):
            chunk = AgenticChunk(
                id=generate_chunk_id(context.get("file_path", "unknown"), i * 10, (i + 1) * 10),
                content=f"# Fallback chunk {i}",
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
        
        return chunks
