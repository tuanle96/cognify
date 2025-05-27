"""
Factory for creating document parsers.
"""

from typing import Dict, List, Optional, Type
import logging

from .base import DocumentParser, DocumentType, ParsingRequest, UnsupportedFormatError
from .code_parser import CodeParser
from .text_parser import TextParser
from .pdf_parser import PDFParser
from .structured_parser import StructuredParser

logger = logging.getLogger(__name__)

class ParserFactory:
    """Factory for creating and managing document parsers."""
    
    # Registry of available parsers
    _parsers: Dict[str, Type[DocumentParser]] = {
        "code_parser": CodeParser,
        "text_parser": TextParser,
        "pdf_parser": PDFParser,
        "structured_parser": StructuredParser,
    }
    
    # Document type to parser mapping
    _type_mapping: Dict[DocumentType, List[str]] = {
        DocumentType.CODE: ["code_parser"],
        DocumentType.TEXT: ["text_parser"],
        DocumentType.MARKDOWN: ["text_parser"],
        DocumentType.PDF: ["pdf_parser"],
        DocumentType.JSON: ["structured_parser"],
        DocumentType.YAML: ["structured_parser"],
        DocumentType.XML: ["structured_parser"],
        DocumentType.CSV: ["structured_parser"],
        DocumentType.HTML: ["text_parser"],  # Can be handled as text
    }
    
    @classmethod
    def create_parser(cls, parser_name: str, **kwargs) -> DocumentParser:
        """
        Create a specific parser by name.
        
        Args:
            parser_name: Name of the parser to create
            **kwargs: Configuration options for the parser
        
        Returns:
            Configured parser instance
        
        Raises:
            ValueError: If parser name is not supported
        """
        if parser_name not in cls._parsers:
            raise ValueError(f"Unsupported parser: {parser_name}")
        
        parser_class = cls._parsers[parser_name]
        return parser_class(**kwargs)
    
    @classmethod
    def create_code_parser(cls, **kwargs) -> CodeParser:
        """Create a code parser."""
        return cls.create_parser("code_parser", **kwargs)
    
    @classmethod
    def create_text_parser(cls, **kwargs) -> TextParser:
        """Create a text parser."""
        return cls.create_parser("text_parser", **kwargs)
    
    @classmethod
    def create_pdf_parser(cls, **kwargs) -> PDFParser:
        """Create a PDF parser."""
        return cls.create_parser("pdf_parser", **kwargs)
    
    @classmethod
    def create_structured_parser(cls, **kwargs) -> StructuredParser:
        """Create a structured data parser."""
        return cls.create_parser("structured_parser", **kwargs)
    
    @classmethod
    def get_parsers_for_type(cls, document_type: DocumentType) -> List[DocumentParser]:
        """
        Get all parsers that can handle a specific document type.
        
        Args:
            document_type: The document type to find parsers for
        
        Returns:
            List of parser instances that can handle the document type
        """
        parsers = []
        parser_names = cls._type_mapping.get(document_type, [])
        
        for parser_name in parser_names:
            try:
                parser = cls.create_parser(parser_name)
                parsers.append(parser)
            except Exception as e:
                logger.warning(f"Failed to create parser {parser_name}: {e}")
        
        return parsers
    
    @classmethod
    async def get_best_parser(cls, request: ParsingRequest) -> Optional[DocumentParser]:
        """
        Get the best parser for a parsing request.
        
        Args:
            request: The parsing request
        
        Returns:
            Best parser for the request, or None if no parser can handle it
        """
        # If document type is specified, try parsers for that type first
        if request.document_type:
            parser_names = cls._type_mapping.get(request.document_type, [])
            for parser_name in parser_names:
                try:
                    parser = cls.create_parser(parser_name)
                    await parser.initialize()
                    if await parser.can_parse(request):
                        return parser
                except Exception as e:
                    logger.warning(f"Parser {parser_name} failed initialization: {e}")
        
        # Try all parsers to find one that can handle the request
        for parser_name, parser_class in cls._parsers.items():
            try:
                parser = parser_class()
                await parser.initialize()
                if await parser.can_parse(request):
                    return parser
            except Exception as e:
                logger.warning(f"Parser {parser_name} failed: {e}")
        
        return None
    
    @classmethod
    def get_available_parsers(cls) -> List[str]:
        """Get list of available parser names."""
        return list(cls._parsers.keys())
    
    @classmethod
    def get_supported_types(cls) -> List[DocumentType]:
        """Get list of supported document types."""
        return list(cls._type_mapping.keys())
    
    @classmethod
    def register_parser(
        cls,
        name: str,
        parser_class: Type[DocumentParser],
        supported_types: Optional[List[DocumentType]] = None
    ) -> None:
        """
        Register a new parser.
        
        Args:
            name: Name for the parser
            parser_class: Parser class
            supported_types: Document types the parser supports
        """
        cls._parsers[name] = parser_class
        
        if supported_types:
            for doc_type in supported_types:
                if doc_type not in cls._type_mapping:
                    cls._type_mapping[doc_type] = []
                if name not in cls._type_mapping[doc_type]:
                    cls._type_mapping[doc_type].append(name)
        
        logger.info(f"Registered parser: {name}")
    
    @classmethod
    def create_all_parsers(cls, **kwargs) -> Dict[str, DocumentParser]:
        """
        Create instances of all available parsers.
        
        Args:
            **kwargs: Configuration options for all parsers
        
        Returns:
            Dictionary mapping parser names to instances
        """
        parsers = {}
        
        for parser_name in cls._parsers:
            try:
                parser = cls.create_parser(parser_name, **kwargs)
                parsers[parser_name] = parser
            except Exception as e:
                logger.warning(f"Failed to create parser {parser_name}: {e}")
        
        return parsers
    
    @classmethod
    async def initialize_all_parsers(cls, parsers: Dict[str, DocumentParser]) -> Dict[str, DocumentParser]:
        """
        Initialize all parsers.
        
        Args:
            parsers: Dictionary of parser instances
        
        Returns:
            Dictionary of successfully initialized parsers
        """
        initialized_parsers = {}
        
        for name, parser in parsers.items():
            try:
                await parser.initialize()
                initialized_parsers[name] = parser
                logger.info(f"Initialized parser: {name}")
            except Exception as e:
                logger.warning(f"Failed to initialize parser {name}: {e}")
        
        return initialized_parsers
    
    @classmethod
    def get_parser_info(cls) -> Dict[str, Dict]:
        """Get information about all available parsers."""
        info = {}
        
        for parser_name, parser_class in cls._parsers.items():
            try:
                # Create temporary instance to get info
                parser = parser_class()
                info[parser_name] = {
                    "class": parser_class.__name__,
                    "supported_types": [t.value for t in parser.supported_types],
                    "parser_name": parser.parser_name
                }
            except Exception as e:
                info[parser_name] = {
                    "class": parser_class.__name__,
                    "error": str(e)
                }
        
        return info

# Global factory instance
parser_factory = ParserFactory()
