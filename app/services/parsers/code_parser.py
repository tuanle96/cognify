"""
Code file parser implementation.
"""

import asyncio
import time
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

from .base import (
    DocumentParser, ParsedDocument, ParsingRequest, ParsingResponse,
    DocumentType, ParsingError, ContentExtractionError
)

logger = logging.getLogger(__name__)

class CodeParser(DocumentParser):
    """Parser for source code files with language-specific processing."""
    
    # Language-specific comment patterns
    COMMENT_PATTERNS = {
        'python': [r'#.*$', r'"""[\s\S]*?"""', r"'''[\s\S]*?'''"],
        'javascript': [r'//.*$', r'/\*[\s\S]*?\*/'],
        'typescript': [r'//.*$', r'/\*[\s\S]*?\*/'],
        'java': [r'//.*$', r'/\*[\s\S]*?\*/'],
        'cpp': [r'//.*$', r'/\*[\s\S]*?\*/'],
        'c': [r'//.*$', r'/\*[\s\S]*?\*/'],
        'csharp': [r'//.*$', r'/\*[\s\S]*?\*/'],
        'go': [r'//.*$', r'/\*[\s\S]*?\*/'],
        'rust': [r'//.*$', r'/\*[\s\S]*?\*/'],
        'php': [r'//.*$', r'/\*[\s\S]*?\*/', r'#.*$'],
        'ruby': [r'#.*$', r'=begin[\s\S]*?=end'],
        'bash': [r'#.*$'],
        'sql': [r'--.*$', r'/\*[\s\S]*?\*/'],
    }
    
    # Function/class detection patterns
    STRUCTURE_PATTERNS = {
        'python': {
            'function': r'^\s*def\s+(\w+)\s*\(',
            'class': r'^\s*class\s+(\w+)\s*[:\(]',
            'import': r'^\s*(?:from\s+\S+\s+)?import\s+',
        },
        'javascript': {
            'function': r'^\s*(?:function\s+(\w+)|(\w+)\s*[:=]\s*(?:function|\([^)]*\)\s*=>))',
            'class': r'^\s*class\s+(\w+)',
            'import': r'^\s*(?:import|export)',
        },
        'java': {
            'function': r'^\s*(?:public|private|protected)?\s*(?:static\s+)?(?:\w+\s+)*(\w+)\s*\(',
            'class': r'^\s*(?:public\s+)?class\s+(\w+)',
            'import': r'^\s*import\s+',
        },
        'go': {
            'function': r'^\s*func\s+(?:\([^)]*\)\s+)?(\w+)\s*\(',
            'struct': r'^\s*type\s+(\w+)\s+struct',
            'import': r'^\s*import\s+',
        }
    }
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_file_size = kwargs.get('max_file_size', 10 * 1024 * 1024)  # 10MB
    
    @property
    def supported_types(self) -> List[DocumentType]:
        return [DocumentType.CODE]
    
    @property
    def parser_name(self) -> str:
        return "code_parser"
    
    async def initialize(self) -> None:
        """Initialize the code parser."""
        self._initialized = True
        logger.info("Code parser initialized")
    
    async def can_parse(self, request: ParsingRequest) -> bool:
        """Check if this parser can handle the request."""
        if request.document_type == DocumentType.CODE:
            return True
        
        # Check if file extension indicates code
        if request.file_path:
            path = Path(request.file_path)
            code_extensions = {
                '.py', '.js', '.ts', '.java', '.cpp', '.c', '.h',
                '.cs', '.go', '.rs', '.php', '.rb', '.kt', '.swift',
                '.scala', '.sh', '.sql', '.r', '.m'
            }
            return path.suffix.lower() in code_extensions
        
        return False
    
    async def parse(self, request: ParsingRequest) -> ParsingResponse:
        """Parse a code file."""
        if not self._initialized:
            await self.initialize()
        
        self._validate_request(request)
        
        start_time = time.time()
        errors = []
        warnings = []
        
        try:
            # Get content
            content = await self._get_content(request)
            
            # Validate file size
            if len(content.encode('utf-8')) > self.max_file_size:
                raise ContentExtractionError(
                    f"File too large: {len(content)} bytes > {self.max_file_size}",
                    self.parser_name
                )
            
            # Extract metadata
            metadata = self._extract_basic_metadata(request)
            
            if request.extract_metadata:
                code_metadata = await self._extract_code_metadata(content, request.language)
                metadata.update(code_metadata)
            
            # Extract sections if requested
            sections = []
            if request.extract_sections:
                sections = await self._extract_code_sections(content, request.language)
            
            # Create parsed document
            document = ParsedDocument(
                content=content,
                document_type=DocumentType.CODE,
                language=request.language,
                file_path=request.file_path,
                metadata=metadata,
                sections=sections
            )
            
            processing_time = time.time() - start_time
            
            return ParsingResponse(
                document=document,
                processing_time=processing_time,
                parser_used=self.parser_name,
                success=True,
                errors=errors,
                warnings=warnings
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Code parsing failed: {e}")
            
            # Return minimal document on error
            try:
                content = await self._get_content(request)
                document = ParsedDocument(
                    content=content,
                    document_type=DocumentType.CODE,
                    language=request.language,
                    file_path=request.file_path,
                    metadata=self._extract_basic_metadata(request)
                )
            except:
                document = ParsedDocument(
                    content="",
                    document_type=DocumentType.CODE,
                    language=request.language,
                    file_path=request.file_path
                )
            
            return ParsingResponse(
                document=document,
                processing_time=processing_time,
                parser_used=self.parser_name,
                success=False,
                errors=[str(e)]
            )
    
    async def _get_content(self, request: ParsingRequest) -> str:
        """Get content from request."""
        if request.content:
            return request.content
        
        if request.file_data:
            try:
                return request.file_data.decode(request.encoding)
            except UnicodeDecodeError as e:
                # Try common encodings
                for encoding in ['utf-8', 'latin-1', 'cp1252']:
                    try:
                        return request.file_data.decode(encoding)
                    except UnicodeDecodeError:
                        continue
                raise ContentExtractionError(f"Could not decode file data: {e}", self.parser_name)
        
        if request.file_path:
            path = Path(request.file_path)
            if not path.exists():
                raise ContentExtractionError(f"File not found: {request.file_path}", self.parser_name)
            
            try:
                return path.read_text(encoding=request.encoding)
            except UnicodeDecodeError as e:
                # Try common encodings
                for encoding in ['utf-8', 'latin-1', 'cp1252']:
                    try:
                        return path.read_text(encoding=encoding)
                    except UnicodeDecodeError:
                        continue
                raise ContentExtractionError(f"Could not decode file: {e}", self.parser_name)
        
        raise ContentExtractionError("No content source provided", self.parser_name)
    
    async def _extract_code_metadata(self, content: str, language: Optional[str]) -> Dict[str, Any]:
        """Extract metadata specific to code files."""
        metadata = {}
        
        lines = content.split('\n')
        metadata['line_count'] = len(lines)
        metadata['char_count'] = len(content)
        metadata['non_empty_lines'] = len([line for line in lines if line.strip()])
        
        if language:
            # Count comments
            comment_lines = self._count_comment_lines(content, language)
            metadata['comment_lines'] = comment_lines
            metadata['code_lines'] = metadata['non_empty_lines'] - comment_lines
            
            # Extract imports/includes
            imports = self._extract_imports(content, language)
            metadata['imports'] = imports
            metadata['import_count'] = len(imports)
            
            # Count functions/classes
            structures = self._count_structures(content, language)
            metadata.update(structures)
        
        # Calculate complexity metrics
        metadata['cyclomatic_complexity'] = self._calculate_complexity(content)
        
        return metadata
    
    async def _extract_code_sections(self, content: str, language: Optional[str]) -> List[Dict[str, Any]]:
        """Extract code sections (functions, classes, etc.)."""
        sections = []
        
        if not language or language not in self.STRUCTURE_PATTERNS:
            return sections
        
        lines = content.split('\n')
        patterns = self.STRUCTURE_PATTERNS[language]
        
        for i, line in enumerate(lines):
            for structure_type, pattern in patterns.items():
                match = re.search(pattern, line, re.MULTILINE)
                if match:
                    name = match.group(1) if match.groups() else "unnamed"
                    sections.append({
                        'type': structure_type,
                        'name': name,
                        'line_number': i + 1,
                        'content': line.strip()
                    })
        
        return sections
    
    def _count_comment_lines(self, content: str, language: str) -> int:
        """Count comment lines in the code."""
        if language not in self.COMMENT_PATTERNS:
            return 0
        
        comment_count = 0
        patterns = self.COMMENT_PATTERNS[language]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.MULTILINE)
            for match in matches:
                comment_count += match.count('\n') + 1
        
        return comment_count
    
    def _extract_imports(self, content: str, language: str) -> List[str]:
        """Extract import statements."""
        imports = []
        
        if language not in self.STRUCTURE_PATTERNS:
            return imports
        
        import_pattern = self.STRUCTURE_PATTERNS[language].get('import')
        if not import_pattern:
            return imports
        
        lines = content.split('\n')
        for line in lines:
            if re.match(import_pattern, line.strip()):
                imports.append(line.strip())
        
        return imports
    
    def _count_structures(self, content: str, language: str) -> Dict[str, int]:
        """Count code structures (functions, classes, etc.)."""
        counts = {}
        
        if language not in self.STRUCTURE_PATTERNS:
            return counts
        
        patterns = self.STRUCTURE_PATTERNS[language]
        lines = content.split('\n')
        
        for structure_type, pattern in patterns.items():
            if structure_type == 'import':
                continue
            
            count = 0
            for line in lines:
                if re.search(pattern, line):
                    count += 1
            
            counts[f'{structure_type}_count'] = count
        
        return counts
    
    def _calculate_complexity(self, content: str) -> int:
        """Calculate basic cyclomatic complexity."""
        # Simple complexity calculation based on control flow keywords
        complexity_keywords = [
            'if', 'elif', 'else', 'for', 'while', 'try', 'except',
            'catch', 'switch', 'case', 'default', '&&', '||', '?'
        ]
        
        complexity = 1  # Base complexity
        
        for keyword in complexity_keywords:
            complexity += content.count(keyword)
        
        return complexity
