"""
Text and markdown parser implementation.
"""

import asyncio
import time
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

try:
    import markdown
    from markdown.extensions import toc, tables, codehilite
except ImportError:
    markdown = None

from .base import (
    DocumentParser, ParsedDocument, ParsingRequest, ParsingResponse,
    DocumentType, ParsingError, ContentExtractionError
)

logger = logging.getLogger(__name__)

class TextParser(DocumentParser):
    """Parser for text and markdown files."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_file_size = kwargs.get('max_file_size', 50 * 1024 * 1024)  # 50MB
        self.extract_headings = kwargs.get('extract_headings', True)
        self.extract_links = kwargs.get('extract_links', True)
    
    @property
    def supported_types(self) -> List[DocumentType]:
        return [DocumentType.TEXT, DocumentType.MARKDOWN]
    
    @property
    def parser_name(self) -> str:
        return "text_parser"
    
    async def initialize(self) -> None:
        """Initialize the text parser."""
        self._initialized = True
        logger.info("Text parser initialized")
    
    async def can_parse(self, request: ParsingRequest) -> bool:
        """Check if this parser can handle the request."""
        if request.document_type in [DocumentType.TEXT, DocumentType.MARKDOWN]:
            return True
        
        # Check file extension
        if request.file_path:
            path = Path(request.file_path)
            text_extensions = {'.txt', '.md', '.markdown', '.rst', '.log'}
            return path.suffix.lower() in text_extensions
        
        return False
    
    async def parse(self, request: ParsingRequest) -> ParsingResponse:
        """Parse a text or markdown file."""
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
            
            # Determine document type
            doc_type = request.document_type or self._detect_type(content, request.file_path)
            
            # Extract metadata
            metadata = self._extract_basic_metadata(request)
            
            if request.extract_metadata:
                text_metadata = await self._extract_text_metadata(content, doc_type)
                metadata.update(text_metadata)
            
            # Extract sections if requested
            sections = []
            if request.extract_sections:
                sections = await self._extract_text_sections(content, doc_type)
            
            # Process markdown if applicable
            processed_content = content
            if doc_type == DocumentType.MARKDOWN and markdown:
                try:
                    processed_content = self._process_markdown(content)
                    metadata['markdown_processed'] = True
                except Exception as e:
                    warnings.append(f"Markdown processing failed: {e}")
                    metadata['markdown_processed'] = False
            
            # Create parsed document
            document = ParsedDocument(
                content=processed_content,
                document_type=doc_type,
                language=None,  # Text files don't have programming language
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
            logger.error(f"Text parsing failed: {e}")
            
            # Return minimal document on error
            try:
                content = await self._get_content(request)
                document = ParsedDocument(
                    content=content,
                    document_type=DocumentType.TEXT,
                    file_path=request.file_path,
                    metadata=self._extract_basic_metadata(request)
                )
            except:
                document = ParsedDocument(
                    content="",
                    document_type=DocumentType.TEXT,
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
    
    def _detect_type(self, content: str, file_path: Optional[str]) -> DocumentType:
        """Detect if content is markdown or plain text."""
        if file_path:
            path = Path(file_path)
            if path.suffix.lower() in ['.md', '.markdown']:
                return DocumentType.MARKDOWN
        
        # Check for markdown patterns
        markdown_patterns = [
            r'^#{1,6}\s+',  # Headers
            r'^\*\s+',      # Bullet lists
            r'^\d+\.\s+',   # Numbered lists
            r'\[.*\]\(.*\)', # Links
            r'```',         # Code blocks
            r'\*\*.*\*\*',  # Bold
            r'\*.*\*',      # Italic
        ]
        
        markdown_score = 0
        for pattern in markdown_patterns:
            if re.search(pattern, content, re.MULTILINE):
                markdown_score += 1
        
        # If multiple markdown patterns found, likely markdown
        return DocumentType.MARKDOWN if markdown_score >= 2 else DocumentType.TEXT
    
    async def _extract_text_metadata(self, content: str, doc_type: DocumentType) -> Dict[str, Any]:
        """Extract metadata specific to text files."""
        metadata = {}
        
        lines = content.split('\n')
        metadata['line_count'] = len(lines)
        metadata['char_count'] = len(content)
        metadata['word_count'] = len(content.split())
        metadata['paragraph_count'] = len([p for p in content.split('\n\n') if p.strip()])
        
        # Extract language-agnostic metrics
        metadata['avg_line_length'] = sum(len(line) for line in lines) / len(lines) if lines else 0
        metadata['avg_word_length'] = sum(len(word) for word in content.split()) / len(content.split()) if content.split() else 0
        
        if doc_type == DocumentType.MARKDOWN:
            # Markdown-specific metadata
            md_metadata = self._extract_markdown_metadata(content)
            metadata.update(md_metadata)
        
        return metadata
    
    def _extract_markdown_metadata(self, content: str) -> Dict[str, Any]:
        """Extract markdown-specific metadata."""
        metadata = {}
        
        # Count headers
        headers = re.findall(r'^(#{1,6})\s+(.+)$', content, re.MULTILINE)
        metadata['header_count'] = len(headers)
        
        # Count by header level
        for level in range(1, 7):
            level_headers = [h for h in headers if len(h[0]) == level]
            metadata[f'h{level}_count'] = len(level_headers)
        
        # Count links
        links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content)
        metadata['link_count'] = len(links)
        
        # Count images
        images = re.findall(r'!\[([^\]]*)\]\(([^)]+)\)', content)
        metadata['image_count'] = len(images)
        
        # Count code blocks
        code_blocks = re.findall(r'```[\s\S]*?```', content)
        metadata['code_block_count'] = len(code_blocks)
        
        # Count inline code
        inline_code = re.findall(r'`[^`]+`', content)
        metadata['inline_code_count'] = len(inline_code)
        
        # Count lists
        bullet_lists = re.findall(r'^\s*[\*\-\+]\s+', content, re.MULTILINE)
        numbered_lists = re.findall(r'^\s*\d+\.\s+', content, re.MULTILINE)
        metadata['bullet_list_items'] = len(bullet_lists)
        metadata['numbered_list_items'] = len(numbered_lists)
        
        return metadata
    
    async def _extract_text_sections(self, content: str, doc_type: DocumentType) -> List[Dict[str, Any]]:
        """Extract text sections."""
        sections = []
        
        if doc_type == DocumentType.MARKDOWN:
            # Extract markdown headers as sections
            lines = content.split('\n')
            for i, line in enumerate(lines):
                header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
                if header_match:
                    level = len(header_match.group(1))
                    title = header_match.group(2).strip()
                    sections.append({
                        'type': 'header',
                        'level': level,
                        'title': title,
                        'line_number': i + 1,
                        'content': line.strip()
                    })
        else:
            # For plain text, try to identify sections by empty lines
            paragraphs = content.split('\n\n')
            for i, paragraph in enumerate(paragraphs):
                if paragraph.strip():
                    sections.append({
                        'type': 'paragraph',
                        'index': i,
                        'content': paragraph.strip()[:100] + '...' if len(paragraph) > 100 else paragraph.strip()
                    })
        
        return sections
    
    def _process_markdown(self, content: str) -> str:
        """Process markdown content to HTML."""
        if not markdown:
            return content
        
        try:
            md = markdown.Markdown(
                extensions=['toc', 'tables', 'codehilite', 'fenced_code'],
                extension_configs={
                    'toc': {'title': 'Table of Contents'},
                    'codehilite': {'use_pygments': False}
                }
            )
            return md.convert(content)
        except Exception as e:
            logger.warning(f"Markdown processing failed: {e}")
            return content
