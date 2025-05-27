"""
PDF parser implementation.
"""

import asyncio
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

try:
    import PyPDF2
    from PyPDF2 import PdfReader
except ImportError:
    PyPDF2 = None
    PdfReader = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

from .base import (
    DocumentParser, ParsedDocument, ParsingRequest, ParsingResponse,
    DocumentType, ParsingError, ContentExtractionError
)

logger = logging.getLogger(__name__)

class PDFParser(DocumentParser):
    """Parser for PDF files with multiple extraction backends."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_file_size = kwargs.get('max_file_size', 100 * 1024 * 1024)  # 100MB
        self.preferred_backend = kwargs.get('preferred_backend', 'pdfplumber')  # or 'pypdf2'
        self.extract_tables = kwargs.get('extract_tables', True)
        self.extract_images = kwargs.get('extract_images', False)
    
    @property
    def supported_types(self) -> List[DocumentType]:
        return [DocumentType.PDF]
    
    @property
    def parser_name(self) -> str:
        return "pdf_parser"
    
    async def initialize(self) -> None:
        """Initialize the PDF parser."""
        if not PyPDF2 and not pdfplumber:
            raise ImportError("Either PyPDF2 or pdfplumber is required for PDF parsing")
        
        self._initialized = True
        logger.info(f"PDF parser initialized with backend: {self._get_available_backend()}")
    
    async def can_parse(self, request: ParsingRequest) -> bool:
        """Check if this parser can handle the request."""
        if request.document_type == DocumentType.PDF:
            return True
        
        # Check file extension
        if request.file_path:
            path = Path(request.file_path)
            return path.suffix.lower() == '.pdf'
        
        # Check file data magic bytes
        if request.file_data:
            return request.file_data.startswith(b'%PDF')
        
        return False
    
    async def parse(self, request: ParsingRequest) -> ParsingResponse:
        """Parse a PDF file."""
        if not self._initialized:
            await self.initialize()
        
        self._validate_request(request)
        
        start_time = time.time()
        errors = []
        warnings = []
        
        try:
            # Get file path or create temporary file
            file_path = await self._get_file_path(request)
            
            # Validate file size
            if file_path.stat().st_size > self.max_file_size:
                raise ContentExtractionError(
                    f"PDF file too large: {file_path.stat().st_size} bytes > {self.max_file_size}",
                    self.parser_name
                )
            
            # Extract content using available backend
            backend = self._get_available_backend()
            if backend == 'pdfplumber' and pdfplumber:
                content, pdf_metadata = await self._extract_with_pdfplumber(file_path)
            elif backend == 'pypdf2' and PyPDF2:
                content, pdf_metadata = await self._extract_with_pypdf2(file_path)
            else:
                raise ContentExtractionError("No PDF parsing backend available", self.parser_name)
            
            # Extract metadata
            metadata = self._extract_basic_metadata(request)
            metadata.update(pdf_metadata)
            
            # Extract sections if requested
            sections = []
            if request.extract_sections:
                sections = await self._extract_pdf_sections(content, pdf_metadata)
            
            # Create parsed document
            document = ParsedDocument(
                content=content,
                document_type=DocumentType.PDF,
                language=None,
                file_path=request.file_path,
                metadata=metadata,
                sections=sections
            )
            
            processing_time = time.time() - start_time
            
            return ParsingResponse(
                document=document,
                processing_time=processing_time,
                parser_used=f"{self.parser_name}_{backend}",
                success=True,
                errors=errors,
                warnings=warnings
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"PDF parsing failed: {e}")
            
            # Return minimal document on error
            document = ParsedDocument(
                content="",
                document_type=DocumentType.PDF,
                file_path=request.file_path,
                metadata=self._extract_basic_metadata(request)
            )
            
            return ParsingResponse(
                document=document,
                processing_time=processing_time,
                parser_used=self.parser_name,
                success=False,
                errors=[str(e)]
            )
    
    async def _get_file_path(self, request: ParsingRequest) -> Path:
        """Get file path, creating temporary file if needed."""
        if request.file_path:
            path = Path(request.file_path)
            if not path.exists():
                raise ContentExtractionError(f"PDF file not found: {request.file_path}", self.parser_name)
            return path
        
        if request.file_data:
            # Create temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                tmp_file.write(request.file_data)
                return Path(tmp_file.name)
        
        raise ContentExtractionError("No PDF file source provided", self.parser_name)
    
    async def _extract_with_pdfplumber(self, file_path: Path) -> tuple[str, Dict[str, Any]]:
        """Extract content using pdfplumber."""
        content_parts = []
        metadata = {'backend': 'pdfplumber'}
        
        with pdfplumber.open(file_path) as pdf:
            metadata['page_count'] = len(pdf.pages)
            metadata['pdf_metadata'] = pdf.metadata or {}
            
            tables = []
            for page_num, page in enumerate(pdf.pages):
                # Extract text
                page_text = page.extract_text()
                if page_text:
                    content_parts.append(f"--- Page {page_num + 1} ---\n{page_text}")
                
                # Extract tables if enabled
                if self.extract_tables:
                    page_tables = page.extract_tables()
                    for table_num, table in enumerate(page_tables):
                        if table:
                            tables.append({
                                'page': page_num + 1,
                                'table_index': table_num,
                                'rows': len(table),
                                'cols': len(table[0]) if table else 0,
                                'data': table
                            })
            
            metadata['table_count'] = len(tables)
            if tables:
                metadata['tables'] = tables
        
        content = '\n\n'.join(content_parts)
        return content, metadata
    
    async def _extract_with_pypdf2(self, file_path: Path) -> tuple[str, Dict[str, Any]]:
        """Extract content using PyPDF2."""
        content_parts = []
        metadata = {'backend': 'pypdf2'}
        
        with open(file_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            
            metadata['page_count'] = len(pdf_reader.pages)
            
            # Extract PDF metadata
            if pdf_reader.metadata:
                metadata['pdf_metadata'] = {
                    'title': pdf_reader.metadata.get('/Title', ''),
                    'author': pdf_reader.metadata.get('/Author', ''),
                    'subject': pdf_reader.metadata.get('/Subject', ''),
                    'creator': pdf_reader.metadata.get('/Creator', ''),
                    'producer': pdf_reader.metadata.get('/Producer', ''),
                    'creation_date': str(pdf_reader.metadata.get('/CreationDate', '')),
                    'modification_date': str(pdf_reader.metadata.get('/ModDate', ''))
                }
            
            # Extract text from each page
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        content_parts.append(f"--- Page {page_num + 1} ---\n{page_text}")
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num + 1}: {e}")
        
        content = '\n\n'.join(content_parts)
        return content, metadata
    
    async def _extract_pdf_sections(self, content: str, pdf_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract sections from PDF content."""
        sections = []
        
        # Split by page markers
        pages = content.split('--- Page ')
        for i, page_content in enumerate(pages[1:], 1):  # Skip first empty split
            if page_content.strip():
                sections.append({
                    'type': 'page',
                    'page_number': i,
                    'content': page_content[:200] + '...' if len(page_content) > 200 else page_content,
                    'char_count': len(page_content)
                })
        
        # Add table sections if available
        if 'tables' in pdf_metadata:
            for table in pdf_metadata['tables']:
                sections.append({
                    'type': 'table',
                    'page_number': table['page'],
                    'table_index': table['table_index'],
                    'rows': table['rows'],
                    'cols': table['cols']
                })
        
        return sections
    
    def _get_available_backend(self) -> str:
        """Get the best available PDF parsing backend."""
        if self.preferred_backend == 'pdfplumber' and pdfplumber:
            return 'pdfplumber'
        elif self.preferred_backend == 'pypdf2' and PyPDF2:
            return 'pypdf2'
        elif pdfplumber:
            return 'pdfplumber'
        elif PyPDF2:
            return 'pypdf2'
        else:
            raise ImportError("No PDF parsing backend available")
