"""
AST-based chunking implementation for fallback scenarios.

Provides fast, reliable chunking using Abstract Syntax Tree parsing.
"""

import ast
import hashlib
import time
from typing import Dict, Any, List, Optional

import structlog

from app.core.exceptions import ParsingError, UnsupportedLanguageError
from app.services.chunking.base import (
    BaseChunker,
    ChunkingRequest,
    ChunkingResult,
    ChunkingStrategy,
    AgenticChunk,
    ChunkType,
    ChunkMetadata,
    generate_chunk_id,
)

logger = structlog.get_logger(__name__)


class ASTChunker(BaseChunker):
    """
    AST-based chunker for fast, reliable code chunking.
    """
    
    def __init__(self):
        super().__init__("ast_chunker")
        self._supported_languages = {"python"}  # Start with Python only
        self._initialized = False
    
    async def initialize(self) -> None:
        """
        Initialize the AST chunker.
        """
        if self._initialized:
            return
        
        try:
            self.logger.info("Initializing AST chunker")
            
            # Validate Python AST availability
            test_code = "def test(): pass"
            ast.parse(test_code)
            
            self._initialized = True
            self.logger.info("AST chunker initialized successfully")
            
        except Exception as e:
            self.logger.error("Failed to initialize AST chunker", error=str(e))
            raise ParsingError(
                message="Failed to initialize AST chunker",
                language="python",
                details={"error": str(e)}
            )
    
    async def chunk(self, request: ChunkingRequest) -> ChunkingResult:
        """
        Perform AST-based chunking.
        
        Args:
            request: Chunking request
        
        Returns:
            ChunkingResult: Chunking results
        """
        if not self._initialized:
            raise ParsingError("AST chunker not initialized", request.language)
        
        start_time = time.time()
        
        try:
            self.logger.info(
                "Starting AST chunking",
                file_path=request.file_path,
                language=request.language,
                content_length=len(request.content)
            )
            
            # Check language support
            if request.language not in self._supported_languages:
                raise UnsupportedLanguageError(
                    language=request.language,
                    file_path=request.file_path
                )
            
            # Parse and chunk based on language
            if request.language == "python":
                chunks = await self._chunk_python(request)
            else:
                # This shouldn't happen due to the check above, but just in case
                raise UnsupportedLanguageError(
                    language=request.language,
                    file_path=request.file_path
                )
            
            # Calculate quality score (basic for AST chunking)
            quality_score = self._calculate_ast_quality_score(chunks, request.content)
            
            processing_time = time.time() - start_time
            
            result = ChunkingResult(
                chunks=chunks,
                strategy_used=ChunkingStrategy.AST_FALLBACK,
                quality_score=quality_score,
                processing_time=processing_time,
                metadata={
                    "chunker": "ast",
                    "language": request.language,
                    "parsing_method": "python_ast" if request.language == "python" else "unknown"
                }
            )
            
            self.log_performance(
                "ast_chunking",
                processing_time,
                chunk_count=len(chunks),
                quality_score=quality_score,
                language=request.language
            )
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(
                "AST chunking failed",
                file_path=request.file_path,
                language=request.language,
                error=str(e),
                processing_time=processing_time
            )
            
            if isinstance(e, (ParsingError, UnsupportedLanguageError)):
                raise
            else:
                raise ParsingError(
                    message=f"AST chunking failed: {str(e)}",
                    language=request.language,
                    file_path=request.file_path,
                    details={"original_error": type(e).__name__}
                )
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check for the AST chunker.
        
        Returns:
            dict: Health check results
        """
        try:
            # Test basic AST parsing
            test_code = "def health_check_test(): return True"
            ast.parse(test_code)
            
            return {
                "status": "healthy",
                "chunker": "ast",
                "supported_languages": list(self._supported_languages),
                "initialized": self._initialized
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "chunker": "ast"
            }
    
    async def _chunk_python(self, request: ChunkingRequest) -> List[AgenticChunk]:
        """
        Chunk Python code using AST parsing.
        
        Args:
            request: Chunking request
        
        Returns:
            List[AgenticChunk]: List of chunks
        """
        try:
            # Parse the Python code
            tree = ast.parse(request.content)
            chunks = []
            lines = request.content.split('\n')
            
            # Extract top-level definitions
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    chunk = await self._create_chunk_from_node(
                        node, lines, request.file_path, request.language
                    )
                    if chunk:
                        chunks.append(chunk)
            
            # If no chunks found, create a single chunk for the entire content
            if not chunks:
                chunk = AgenticChunk(
                    id=generate_chunk_id(request.file_path, 1, len(lines)),
                    content=request.content,
                    language=request.language,
                    chunk_type=ChunkType.MODULE,
                    name="module",
                    start_line=1,
                    end_line=len(lines),
                    file_path=request.file_path,
                    dependencies=[],
                    semantic_relationships=[],
                    purpose_optimization={},
                    quality_score=0.7,  # Default quality for single module chunk
                    metadata=ChunkMetadata()
                )
                chunks.append(chunk)
            
            return chunks
            
        except SyntaxError as e:
            raise ParsingError(
                message=f"Python syntax error: {str(e)}",
                language="python",
                file_path=request.file_path,
                details={"line": e.lineno, "offset": e.offset}
            )
        except Exception as e:
            raise ParsingError(
                message=f"Python AST parsing failed: {str(e)}",
                language="python",
                file_path=request.file_path,
                details={"error": str(e)}
            )
    
    async def _create_chunk_from_node(
        self,
        node: ast.AST,
        lines: List[str],
        file_path: str,
        language: str
    ) -> Optional[AgenticChunk]:
        """
        Create a chunk from an AST node.
        
        Args:
            node: AST node
            lines: Source code lines
            file_path: File path
            language: Programming language
        
        Returns:
            AgenticChunk or None if chunk creation fails
        """
        try:
            # Get node boundaries
            start_line = node.lineno
            end_line = getattr(node, 'end_lineno', node.lineno)
            
            if end_line is None:
                end_line = start_line
            
            # Extract content
            chunk_lines = lines[start_line - 1:end_line]
            content = '\n'.join(chunk_lines)
            
            # Determine chunk type and name
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                chunk_type = ChunkType.FUNCTION
                name = node.name
            elif isinstance(node, ast.ClassDef):
                chunk_type = ChunkType.CLASS
                name = node.name
            else:
                chunk_type = ChunkType.UNKNOWN
                name = "unknown"
            
            # Calculate basic complexity
            complexity = self._calculate_node_complexity(node)
            
            # Extract dependencies (basic implementation)
            dependencies = self._extract_dependencies(node)
            
            chunk = AgenticChunk(
                id=generate_chunk_id(file_path, start_line, end_line),
                content=content,
                language=language,
                chunk_type=chunk_type,
                name=name,
                start_line=start_line,
                end_line=end_line,
                file_path=file_path,
                dependencies=dependencies,
                semantic_relationships=[],  # Not available in AST mode
                purpose_optimization={},    # Not available in AST mode
                quality_score=0.8,         # Default AST quality score
                metadata=ChunkMetadata(complexity=complexity)
            )
            
            return chunk
            
        except Exception as e:
            self.logger.warning(
                "Failed to create chunk from AST node",
                node_type=type(node).__name__,
                error=str(e)
            )
            return None
    
    def _calculate_node_complexity(self, node: ast.AST) -> int:
        """
        Calculate cyclomatic complexity of an AST node.
        
        Args:
            node: AST node
        
        Returns:
            int: Complexity score
        """
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (
                ast.If, ast.For, ast.While, ast.Try, ast.With,
                ast.ExceptHandler, ast.ListComp, ast.DictComp,
                ast.SetComp, ast.GeneratorExp
            )):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity
    
    def _extract_dependencies(self, node: ast.AST) -> List[str]:
        """
        Extract basic dependencies from an AST node.
        
        Args:
            node: AST node
        
        Returns:
            List[str]: List of dependencies
        """
        dependencies = []
        
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                dependencies.append(child.id)
            elif isinstance(child, ast.Attribute):
                # Handle attribute access like obj.method
                if isinstance(child.value, ast.Name):
                    dependencies.append(f"{child.value.id}.{child.attr}")
        
        # Remove duplicates and common built-ins
        built_ins = {"print", "len", "str", "int", "float", "list", "dict", "set", "tuple"}
        dependencies = list(set(dependencies) - built_ins)
        
        return dependencies[:10]  # Limit to top 10 dependencies
    
    def _calculate_ast_quality_score(self, chunks: List[AgenticChunk], original_content: str) -> float:
        """
        Calculate quality score for AST-based chunking.
        
        Args:
            chunks: List of chunks
            original_content: Original content
        
        Returns:
            float: Quality score between 0 and 1
        """
        if not chunks:
            return 0.0
        
        # Basic quality metrics for AST chunking
        total_lines = len(original_content.split('\n'))
        covered_lines = sum(chunk.size_lines for chunk in chunks)
        coverage = min(covered_lines / total_lines, 1.0) if total_lines > 0 else 0.0
        
        # Penalize very small or very large chunks
        avg_chunk_size = sum(chunk.size_lines for chunk in chunks) / len(chunks)
        size_score = 1.0
        if avg_chunk_size < 5:  # Too small
            size_score = 0.6
        elif avg_chunk_size > 200:  # Too large
            size_score = 0.7
        
        # Combine metrics
        quality_score = (coverage * 0.6) + (size_score * 0.4)
        
        return min(quality_score, 1.0)
