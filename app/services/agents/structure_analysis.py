"""
Structure Analysis Agent for intelligent code structure understanding.
"""

import asyncio
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from app.services.llm.base import LLMMessage, LLMConfig, LLMProvider
from app.services.llm.openai_service import OpenAIService

# Import ChunkType directly to avoid structlog issues
from enum import Enum

class ChunkType(Enum):
    """Types of code chunks."""
    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    MODULE = "module"
    IMPORT = "import"
    COMMENT = "comment"
    SEMANTIC_BLOCK = "semantic_block"
    FUNCTION_DEFINITION = "function_definition"
    CLASS_DEFINITION = "class_definition"
    METHOD_DEFINITION = "method_definition"
    UNKNOWN = "unknown"


@dataclass
class AgenticChunk:
    """Simplified chunk for agent use."""
    id: str
    content: str
    language: str
    chunk_type: ChunkType
    name: str
    start_line: int
    end_line: int
    file_path: str
    dependencies: List[str]
    semantic_relationships: List[str]
    purpose_optimization: Dict[str, Any]
    quality_score: float
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.semantic_relationships is None:
            self.semantic_relationships = []
        if self.purpose_optimization is None:
            self.purpose_optimization = {}
        if self.metadata is None:
            self.metadata = {}


@dataclass
class StructureAnalysisRequest:
    """Request for structure analysis."""
    content: str
    language: str
    file_path: str
    analysis_depth: str = "medium"  # shallow, medium, deep
    include_dependencies: bool = True
    include_relationships: bool = True


@dataclass
class StructureAnalysisResult:
    """Result of structure analysis."""
    chunks: List[AgenticChunk]
    dependencies: List[str]
    relationships: Dict[str, List[str]]
    complexity_score: float
    quality_score: float
    analysis_metadata: Dict[str, Any]
    processing_time: float


class StructureAnalysisAgent:
    """
    Agent for analyzing code structure and generating intelligent chunks.
    """

    def __init__(self, llm_config: Optional[LLMConfig] = None):
        """Initialize the structure analysis agent."""
        self.llm_config = llm_config or self._get_default_config()
        self.llm_service = None
        self._initialized = False

    def _get_default_config(self) -> LLMConfig:
        """Get default LLM configuration."""
        import os

        # Get API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is required for StructureAnalysisAgent. "
                "Please set it in your .env file or environment."
            )

        return LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4o-mini",
            api_key=api_key,
            base_url=os.getenv("OPENAI_BASE_URL", "https://ai.earnbase.io/v1"),
            temperature=0.1,
            max_tokens=2000
        )

    async def initialize(self) -> None:
        """Initialize the agent."""
        if not self._initialized:
            self.llm_service = OpenAIService(self.llm_config)
            await self.llm_service.initialize()
            self._initialized = True

    async def analyze_structure(self, request: StructureAnalysisRequest) -> StructureAnalysisResult:
        """
        Analyze code structure and generate intelligent chunks.
        """
        if not self._initialized:
            await self.initialize()

        start_time = time.time()

        try:
            # Generate analysis prompt
            analysis_prompt = self._create_analysis_prompt(request)

            # Get LLM analysis
            messages = [
                LLMMessage(role="system", content=self._get_system_prompt()),
                LLMMessage(role="user", content=analysis_prompt)
            ]

            response = await self.llm_service.generate(messages, max_tokens=2000)

            # Parse LLM response into structured result
            result = self._parse_analysis_response(response.content, request)

            processing_time = time.time() - start_time
            result.processing_time = processing_time

            return result

        except Exception as e:
            # Fallback to basic analysis
            return self._fallback_analysis(request, time.time() - start_time)

    def _get_system_prompt(self) -> str:
        """Get system prompt for structure analysis."""
        return """You are an expert code structure analyzer. Your task is to analyze code and break it down into meaningful, semantically coherent chunks.

For each chunk, identify:
1. Type (function, class, method, module, etc.)
2. Name and purpose
3. Dependencies on other code elements
4. Semantic relationships
5. Complexity and quality indicators

Respond in JSON format with the following structure:
{
    "chunks": [
        {
            "name": "chunk_name",
            "type": "function|class|method|module|import|comment|semantic_block",
            "start_line": 1,
            "end_line": 10,
            "content": "actual code content",
            "dependencies": ["dependency1", "dependency2"],
            "relationships": ["related_chunk1", "related_chunk2"],
            "complexity": 1-10,
            "quality": 0.0-1.0,
            "purpose": "description of what this chunk does"
        }
    ],
    "overall_dependencies": ["external_dep1", "external_dep2"],
    "complexity_score": 0.0-1.0,
    "quality_score": 0.0-1.0
}

Be precise and focus on semantic meaning rather than just syntactic boundaries."""

    def _create_analysis_prompt(self, request: StructureAnalysisRequest) -> str:
        """Create analysis prompt for the LLM."""
        return f"""Analyze the following {request.language} code from file: {request.file_path}

Analysis depth: {request.analysis_depth}
Include dependencies: {request.include_dependencies}
Include relationships: {request.include_relationships}

Code to analyze:
```{request.language}
{request.content}
```

Please provide a detailed structural analysis following the JSON format specified in the system prompt."""

    def _parse_analysis_response(self, response: str, request: StructureAnalysisRequest) -> StructureAnalysisResult:
        """Parse LLM response into structured result."""
        import json
        import re

        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                analysis_data = json.loads(json_match.group())
            else:
                raise ValueError("No JSON found in response")

            # Convert to AgenticChunk objects
            chunks = []
            for i, chunk_data in enumerate(analysis_data.get("chunks", [])):
                chunk = AgenticChunk(
                    id=f"chunk_{i}_{chunk_data.get('name', 'unknown')}",
                    content=chunk_data.get("content", ""),
                    language=request.language,
                    chunk_type=self._normalize_chunk_type(chunk_data.get("type", "unknown")),
                    name=chunk_data.get("name", f"chunk_{i}"),
                    start_line=chunk_data.get("start_line", 1),
                    end_line=chunk_data.get("end_line", 1),
                    file_path=request.file_path,
                    dependencies=chunk_data.get("dependencies", []),
                    semantic_relationships=chunk_data.get("relationships", []),
                    purpose_optimization={"purpose": chunk_data.get("purpose", "")},
                    quality_score=chunk_data.get("quality", 0.8),
                    metadata=None
                )
                chunks.append(chunk)

            return StructureAnalysisResult(
                chunks=chunks,
                dependencies=analysis_data.get("overall_dependencies", []),
                relationships=self._extract_relationships(chunks),
                complexity_score=analysis_data.get("complexity_score", 0.5),
                quality_score=analysis_data.get("quality_score", 0.8),
                analysis_metadata={
                    "llm_model": self.llm_config.model,
                    "analysis_depth": request.analysis_depth,
                    "chunk_count": len(chunks)
                },
                processing_time=0.0  # Will be set by caller
            )

        except Exception as e:
            # Fallback to basic parsing
            return self._fallback_analysis(request, 0.0)

    def _normalize_chunk_type(self, type_str: str) -> ChunkType:
        """Normalize chunk type string to ChunkType enum."""
        type_mapping = {
            "function": ChunkType.FUNCTION,
            "class": ChunkType.CLASS,
            "method": ChunkType.METHOD,
            "module": ChunkType.MODULE,
            "import": ChunkType.IMPORT,
            "comment": ChunkType.COMMENT,
            "semantic_block": ChunkType.SEMANTIC_BLOCK,
            "function_definition": ChunkType.FUNCTION_DEFINITION,
            "class_definition": ChunkType.CLASS_DEFINITION,
            "method_definition": ChunkType.METHOD_DEFINITION,
        }

        return type_mapping.get(type_str.lower(), ChunkType.UNKNOWN)

    def _extract_relationships(self, chunks: List[AgenticChunk]) -> Dict[str, List[str]]:
        """Extract relationships between chunks."""
        relationships = {}
        for chunk in chunks:
            relationships[chunk.id] = chunk.semantic_relationships
        return relationships

    def _fallback_analysis(self, request: StructureAnalysisRequest, processing_time: float) -> StructureAnalysisResult:
        """Fallback analysis when LLM fails."""
        # Simple line-based chunking as fallback
        lines = request.content.split('\n')
        chunk_size = 50  # lines per chunk

        chunks = []
        for i in range(0, len(lines), chunk_size):
            end_idx = min(i + chunk_size, len(lines))
            chunk_content = '\n'.join(lines[i:end_idx])

            chunk = AgenticChunk(
                id=f"fallback_chunk_{i//chunk_size}",
                content=chunk_content,
                language=request.language,
                chunk_type=ChunkType.SEMANTIC_BLOCK,
                name=f"Block {i//chunk_size + 1}",
                start_line=i + 1,
                end_line=end_idx,
                file_path=request.file_path,
                dependencies=[],
                semantic_relationships=[],
                purpose_optimization={"purpose": "Fallback chunk"},
                quality_score=0.6,
                metadata=None
            )
            chunks.append(chunk)

        return StructureAnalysisResult(
            chunks=chunks,
            dependencies=[],
            relationships={},
            complexity_score=0.5,
            quality_score=0.6,
            analysis_metadata={
                "fallback": True,
                "reason": "LLM analysis failed",
                "chunk_count": len(chunks)
            },
            processing_time=processing_time
        )

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        try:
            if not self._initialized:
                return {"status": "unhealthy", "reason": "Not initialized"}

            # Test with simple code
            test_request = StructureAnalysisRequest(
                content="def hello():\n    print('Hello, World!')",
                language="python",
                file_path="test.py"
            )

            result = await self.analyze_structure(test_request)

            return {
                "status": "healthy",
                "llm_model": self.llm_config.model,
                "test_chunks": len(result.chunks),
                "processing_time": result.processing_time
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    async def cleanup(self) -> None:
        """Cleanup resources."""
        if self.llm_service:
            await self.llm_service.cleanup()
        self._initialized = False
