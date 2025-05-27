"""
Base interfaces and models for document parsing services.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, BinaryIO, TextIO
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import mimetypes
import time

class DocumentType(Enum):
    """Types of documents that can be parsed."""
    CODE = "code"
    TEXT = "text"
    PDF = "pdf"
    MARKDOWN = "markdown"
    JSON = "json"
    YAML = "yaml"
    XML = "xml"
    HTML = "html"
    CSV = "csv"
    DOCX = "docx"
    UNKNOWN = "unknown"

@dataclass
class ParsedDocument:
    """A parsed document with extracted content and metadata."""
    content: str
    document_type: DocumentType
    language: Optional[str] = None
    file_path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    sections: Optional[List[Dict[str, Any]]] = None
    
    def __post_init__(self):
        if not self.content:
            raise ValueError("Document content cannot be empty")
        if self.metadata is None:
            self.metadata = {}
        if self.sections is None:
            self.sections = []

@dataclass
class ParsingRequest:
    """Request for document parsing."""
    content: Optional[str] = None
    file_path: Optional[str] = None
    file_data: Optional[bytes] = None
    document_type: Optional[DocumentType] = None
    language: Optional[str] = None
    extract_metadata: bool = True
    extract_sections: bool = False
    encoding: str = "utf-8"
    
    def __post_init__(self):
        # Must have either content, file_path, or file_data
        if not any([self.content, self.file_path, self.file_data]):
            raise ValueError("Must provide content, file_path, or file_data")
        
        # Auto-detect document type from file path if not provided
        if self.file_path and not self.document_type:
            self.document_type = self._detect_document_type(self.file_path)
    
    def _detect_document_type(self, file_path: str) -> DocumentType:
        """Detect document type from file extension."""
        path = Path(file_path)
        extension = path.suffix.lower()
        
        # Code file extensions
        code_extensions = {
            '.py': 'python', '.js': 'javascript', '.ts': 'typescript',
            '.java': 'java', '.cpp': 'cpp', '.c': 'c', '.h': 'c',
            '.cs': 'csharp', '.go': 'go', '.rs': 'rust', '.php': 'php',
            '.rb': 'ruby', '.kt': 'kotlin', '.swift': 'swift', '.scala': 'scala',
            '.sh': 'bash', '.sql': 'sql', '.r': 'r', '.m': 'matlab'
        }
        
        if extension in code_extensions:
            self.language = code_extensions[extension]
            return DocumentType.CODE
        
        # Document type mapping
        type_mapping = {
            '.txt': DocumentType.TEXT,
            '.md': DocumentType.MARKDOWN,
            '.markdown': DocumentType.MARKDOWN,
            '.pdf': DocumentType.PDF,
            '.json': DocumentType.JSON,
            '.yaml': DocumentType.YAML,
            '.yml': DocumentType.YAML,
            '.xml': DocumentType.XML,
            '.html': DocumentType.HTML,
            '.htm': DocumentType.HTML,
            '.csv': DocumentType.CSV,
            '.docx': DocumentType.DOCX,
        }
        
        return type_mapping.get(extension, DocumentType.UNKNOWN)

@dataclass
class ParsingResponse:
    """Response from document parsing."""
    document: ParsedDocument
    processing_time: float
    parser_used: str
    success: bool = True
    errors: Optional[List[str]] = None
    warnings: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []

class DocumentParser(ABC):
    """Abstract base class for document parsers."""
    
    def __init__(self, **kwargs):
        self.config = kwargs
        self._initialized = False
    
    @property
    @abstractmethod
    def supported_types(self) -> List[DocumentType]:
        """Get supported document types."""
        pass
    
    @property
    @abstractmethod
    def parser_name(self) -> str:
        """Get parser name."""
        pass
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the parser."""
        pass
    
    @abstractmethod
    async def parse(self, request: ParsingRequest) -> ParsingResponse:
        """Parse a document."""
        pass
    
    @abstractmethod
    async def can_parse(self, request: ParsingRequest) -> bool:
        """Check if this parser can handle the request."""
        pass
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        pass
    
    def _extract_basic_metadata(self, request: ParsingRequest) -> Dict[str, Any]:
        """Extract basic metadata from the request."""
        metadata = {}
        
        if request.file_path:
            path = Path(request.file_path)
            metadata.update({
                "file_name": path.name,
                "file_extension": path.suffix,
                "file_size": path.stat().st_size if path.exists() else None,
                "file_path": str(path)
            })
            
            # Try to get MIME type
            mime_type, _ = mimetypes.guess_type(str(path))
            if mime_type:
                metadata["mime_type"] = mime_type
        
        if request.language:
            metadata["language"] = request.language
        
        if request.document_type:
            metadata["document_type"] = request.document_type.value
        
        metadata["encoding"] = request.encoding
        metadata["parsed_at"] = time.time()
        
        return metadata
    
    def _validate_request(self, request: ParsingRequest) -> None:
        """Validate parsing request."""
        if not any([request.content, request.file_path, request.file_data]):
            raise ValueError("Must provide content, file_path, or file_data")
        
        if request.document_type and request.document_type not in self.supported_types:
            raise UnsupportedFormatError(
                f"Parser {self.parser_name} does not support {request.document_type.value}"
            )

class ParsingError(Exception):
    """Base exception for parsing operations."""
    pass

class UnsupportedFormatError(ParsingError):
    """Exception for unsupported document formats."""
    def __init__(self, message: str, document_type: Optional[DocumentType] = None):
        self.document_type = document_type
        super().__init__(message)

class ParsingTimeoutError(ParsingError):
    """Exception for parsing timeouts."""
    def __init__(self, message: str, timeout: float):
        self.timeout = timeout
        super().__init__(message)

class ContentExtractionError(ParsingError):
    """Exception for content extraction failures."""
    def __init__(self, message: str, parser_name: str):
        self.parser_name = parser_name
        super().__init__(message)
