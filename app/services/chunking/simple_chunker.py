"""
Simple text-based chunker for multi-language support.

Provides basic chunking functionality for languages not supported by AST parsing.
"""

import re
import time
from typing import Dict, List, Any

from app.core.exceptions import ChunkingError
from app.services.chunking.base import (
    BaseChunker, ChunkingRequest, ChunkingResult, ChunkingStrategy,
    AgenticChunk, ChunkType, ChunkMetadata, generate_chunk_id
)


class SimpleChunker(BaseChunker):
    """
    Simple text-based chunker for multi-language support.
    
    Uses pattern matching and heuristics to chunk code when AST parsing
    is not available.
    """
    
    def __init__(self):
        super().__init__("simple_chunker")
        self._supported_languages = {
            "javascript", "typescript", "java", "go", "rust", "cpp", "c", 
            "csharp", "php", "ruby", "kotlin", "swift", "scala", "python"
        }
        self._language_patterns = self._initialize_language_patterns()
        self._initialized = False
    
    def _initialize_language_patterns(self) -> Dict[str, Dict[str, str]]:
        """Initialize language-specific patterns for chunking."""
        return {
            "javascript": {
                "function": r"(?:function\s+\w+|const\s+\w+\s*=\s*(?:async\s+)?(?:function|\([^)]*\)\s*=>))",
                "class": r"class\s+\w+",
                "method": r"^\s*(?:async\s+)?\w+\s*\([^)]*\)\s*{",
            },
            "typescript": {
                "function": r"(?:function\s+\w+|const\s+\w+\s*=\s*(?:async\s+)?(?:function|\([^)]*\)\s*=>))",
                "class": r"(?:class|interface)\s+\w+",
                "method": r"^\s*(?:async\s+)?\w+\s*\([^)]*\)\s*:",
            },
            "java": {
                "function": r"(?:public|private|protected)?\s*(?:static\s+)?(?:\w+\s+)*\w+\s*\([^)]*\)\s*{",
                "class": r"(?:public\s+)?(?:class|interface)\s+\w+",
                "method": r"^\s*(?:public|private|protected)?\s*(?:static\s+)?\w+\s+\w+\s*\([^)]*\)\s*{",
            },
            "go": {
                "function": r"func\s+(?:\([^)]*\)\s+)?\w+\s*\([^)]*\)",
                "struct": r"type\s+\w+\s+struct",
                "method": r"func\s+\([^)]*\)\s+\w+\s*\([^)]*\)",
            },
            "rust": {
                "function": r"fn\s+\w+\s*\([^)]*\)",
                "struct": r"struct\s+\w+",
                "impl": r"impl\s+(?:\w+\s+for\s+)?\w+",
            },
            "python": {
                "function": r"def\s+\w+\s*\([^)]*\):",
                "class": r"class\s+\w+(?:\([^)]*\))?:",
                "method": r"^\s+def\s+\w+\s*\([^)]*\):",
            }
        }
    
    async def initialize(self) -> None:
        """Initialize the simple chunker."""
        if self._initialized:
            return
        
        try:
            self.logger.info("Initializing simple chunker")
            self._initialized = True
            self.logger.info("Simple chunker initialized successfully")
        except Exception as e:
            self.logger.error("Failed to initialize simple chunker", error=str(e))
            raise ChunkingError(f"Failed to initialize simple chunker: {e}")
    
    async def chunk(self, request: ChunkingRequest) -> ChunkingResult:
        """
        Perform simple text-based chunking.
        
        Args:
            request: Chunking request
        
        Returns:
            ChunkingResult: Chunking results
        """
        if not self._initialized:
            raise ChunkingError("Simple chunker not initialized")
        
        start_time = time.time()
        
        try:
            self.logger.info(
                "Starting simple chunking",
                file_path=request.file_path,
                language=request.language,
                content_length=len(request.content)
            )
            
            # Get language patterns
            patterns = self._language_patterns.get(request.language, {})
            
            # Chunk based on patterns
            chunks = await self._chunk_by_patterns(request, patterns)
            
            # If no chunks found, create line-based chunks
            if not chunks:
                chunks = await self._chunk_by_lines(request)
            
            # Calculate quality score
            quality_score = self._calculate_simple_quality_score(chunks, request.content)
            
            processing_time = time.time() - start_time
            
            result = ChunkingResult(
                chunks=chunks,
                strategy_used=ChunkingStrategy.AST_FALLBACK,  # Use AST_FALLBACK for simple chunking
                quality_score=quality_score,
                processing_time=processing_time,
                metadata={
                    "chunker": "simple",
                    "language": request.language,
                    "parsing_method": "pattern_based",
                    "patterns_used": len(patterns)
                }
            )
            
            self.log_performance(
                "simple_chunking",
                processing_time,
                chunk_count=len(chunks),
                quality_score=quality_score,
                language=request.language
            )
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(
                "Simple chunking failed",
                file_path=request.file_path,
                language=request.language,
                error=str(e),
                processing_time=processing_time
            )
            raise ChunkingError(f"Simple chunking failed: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check for the simple chunker."""
        return {
            "status": "healthy" if self._initialized else "not_initialized",
            "chunker": "simple",
            "supported_languages": list(self._supported_languages),
            "initialized": self._initialized,
            "patterns_loaded": len(self._language_patterns)
        }
    
    async def _chunk_by_patterns(self, request: ChunkingRequest, patterns: Dict[str, str]) -> List[AgenticChunk]:
        """Chunk content using language-specific patterns."""
        chunks = []
        lines = request.content.split('\n')
        
        if not patterns:
            return chunks
        
        # Find all pattern matches
        matches = []
        for pattern_type, pattern in patterns.items():
            for i, line in enumerate(lines):
                if re.search(pattern, line, re.IGNORECASE):
                    matches.append({
                        'line': i + 1,
                        'type': pattern_type,
                        'content': line.strip()
                    })
        
        # Sort matches by line number
        matches.sort(key=lambda x: x['line'])
        
        # Create chunks from matches
        for i, match in enumerate(matches):
            start_line = match['line']
            
            # Find end line (next match or end of file)
            if i + 1 < len(matches):
                end_line = matches[i + 1]['line'] - 1
            else:
                end_line = len(lines)
            
            # Extract chunk content
            chunk_lines = lines[start_line - 1:end_line]
            content = '\n'.join(chunk_lines)
            
            # Determine chunk type
            chunk_type = self._get_chunk_type(match['type'])
            
            # Extract name from content
            name = self._extract_name(match['content'], match['type'])
            
            chunk = AgenticChunk(
                id=generate_chunk_id(request.file_path, start_line, end_line),
                content=content,
                language=request.language,
                chunk_type=chunk_type,
                name=name,
                start_line=start_line,
                end_line=end_line,
                file_path=request.file_path,
                dependencies=[],
                semantic_relationships=[],
                purpose_optimization={},
                quality_score=0.75,  # Default quality for pattern-based chunks
                metadata=ChunkMetadata(complexity=self._estimate_complexity(content))
            )
            chunks.append(chunk)
        
        return chunks
    
    async def _chunk_by_lines(self, request: ChunkingRequest, max_lines: int = 50) -> List[AgenticChunk]:
        """Fallback chunking by line count."""
        chunks = []
        lines = request.content.split('\n')
        
        for i in range(0, len(lines), max_lines):
            start_line = i + 1
            end_line = min(i + max_lines, len(lines))
            
            chunk_lines = lines[i:end_line]
            content = '\n'.join(chunk_lines)
            
            chunk = AgenticChunk(
                id=generate_chunk_id(request.file_path, start_line, end_line),
                content=content,
                language=request.language,
                chunk_type=ChunkType.SEMANTIC_BLOCK,
                name=f"block_{i // max_lines + 1}",
                start_line=start_line,
                end_line=end_line,
                file_path=request.file_path,
                dependencies=[],
                semantic_relationships=[],
                purpose_optimization={},
                quality_score=0.6,  # Lower quality for line-based chunks
                metadata=ChunkMetadata(complexity=1)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _get_chunk_type(self, pattern_type: str) -> ChunkType:
        """Map pattern type to chunk type."""
        mapping = {
            "function": ChunkType.FUNCTION,
            "class": ChunkType.CLASS,
            "method": ChunkType.METHOD,
            "struct": ChunkType.CLASS,
            "impl": ChunkType.CLASS,
        }
        return mapping.get(pattern_type, ChunkType.SEMANTIC_BLOCK)
    
    def _extract_name(self, content: str, pattern_type: str) -> str:
        """Extract name from matched content."""
        # Simple name extraction
        words = content.split()
        for i, word in enumerate(words):
            if word in ["function", "class", "def", "func", "struct", "impl"]:
                if i + 1 < len(words):
                    name = words[i + 1]
                    # Clean up name (remove parentheses, etc.)
                    name = re.sub(r'[^\w].*', '', name)
                    return name
        
        return f"{pattern_type}_unknown"
    
    def _estimate_complexity(self, content: str) -> int:
        """Estimate complexity based on content."""
        complexity = 1
        
        # Count control structures
        control_patterns = [
            r'\bif\b', r'\bfor\b', r'\bwhile\b', r'\btry\b', r'\bcatch\b',
            r'\bswitch\b', r'\bmatch\b', r'\bloop\b'
        ]
        
        for pattern in control_patterns:
            complexity += len(re.findall(pattern, content, re.IGNORECASE))
        
        return min(complexity, 20)  # Cap at 20
    
    def _calculate_simple_quality_score(self, chunks: List[AgenticChunk], original_content: str) -> float:
        """Calculate quality score for simple chunking."""
        if not chunks:
            return 0.0
        
        total_lines = len(original_content.split('\n'))
        covered_lines = sum(chunk.size_lines for chunk in chunks)
        coverage = min(covered_lines / total_lines, 1.0) if total_lines > 0 else 0.0
        
        # Prefer more chunks (better granularity)
        chunk_count_score = min(len(chunks) / 10, 1.0)
        
        # Combine metrics
        quality_score = (coverage * 0.7) + (chunk_count_score * 0.3)
        
        return min(quality_score, 1.0)
