"""
Structured data parser implementation for JSON, YAML, XML, CSV.
"""

import asyncio
import time
import json
import csv
from typing import List, Dict, Any, Optional
from pathlib import Path
from io import StringIO
import logging

try:
    import yaml
except ImportError:
    yaml = None

try:
    import xml.etree.ElementTree as ET
    from xml.dom import minidom
except ImportError:
    ET = None
    minidom = None

from .base import (
    DocumentParser, ParsedDocument, ParsingRequest, ParsingResponse,
    DocumentType, ParsingError, ContentExtractionError
)

logger = logging.getLogger(__name__)

class StructuredParser(DocumentParser):
    """Parser for structured data formats: JSON, YAML, XML, CSV."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_file_size = kwargs.get('max_file_size', 20 * 1024 * 1024)  # 20MB
        self.pretty_format = kwargs.get('pretty_format', True)
        self.validate_structure = kwargs.get('validate_structure', True)
    
    @property
    def supported_types(self) -> List[DocumentType]:
        return [DocumentType.JSON, DocumentType.YAML, DocumentType.XML, DocumentType.CSV]
    
    @property
    def parser_name(self) -> str:
        return "structured_parser"
    
    async def initialize(self) -> None:
        """Initialize the structured parser."""
        self._initialized = True
        logger.info("Structured data parser initialized")
    
    async def can_parse(self, request: ParsingRequest) -> bool:
        """Check if this parser can handle the request."""
        if request.document_type in self.supported_types:
            return True
        
        # Check file extension
        if request.file_path:
            path = Path(request.file_path)
            structured_extensions = {'.json', '.yaml', '.yml', '.xml', '.csv'}
            return path.suffix.lower() in structured_extensions
        
        return False
    
    async def parse(self, request: ParsingRequest) -> ParsingResponse:
        """Parse a structured data file."""
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
            
            # Parse and validate structure
            parsed_data = None
            if self.validate_structure:
                parsed_data, parse_errors = await self._parse_structure(content, doc_type)
                errors.extend(parse_errors)
            
            # Format content if requested
            formatted_content = content
            if self.pretty_format and parsed_data is not None:
                formatted_content = await self._format_content(parsed_data, doc_type)
            
            # Extract metadata
            metadata = self._extract_basic_metadata(request)
            
            if request.extract_metadata:
                struct_metadata = await self._extract_structured_metadata(content, doc_type, parsed_data)
                metadata.update(struct_metadata)
            
            # Extract sections if requested
            sections = []
            if request.extract_sections:
                sections = await self._extract_structured_sections(parsed_data, doc_type)
            
            # Create parsed document
            document = ParsedDocument(
                content=formatted_content,
                document_type=doc_type,
                language=None,
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
            logger.error(f"Structured data parsing failed: {e}")
            
            # Return minimal document on error
            try:
                content = await self._get_content(request)
                document = ParsedDocument(
                    content=content,
                    document_type=request.document_type or DocumentType.JSON,
                    file_path=request.file_path,
                    metadata=self._extract_basic_metadata(request)
                )
            except:
                document = ParsedDocument(
                    content="",
                    document_type=request.document_type or DocumentType.JSON,
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
                raise ContentExtractionError(f"Could not decode file data: {e}", self.parser_name)
        
        if request.file_path:
            path = Path(request.file_path)
            if not path.exists():
                raise ContentExtractionError(f"File not found: {request.file_path}", self.parser_name)
            
            try:
                return path.read_text(encoding=request.encoding)
            except UnicodeDecodeError as e:
                raise ContentExtractionError(f"Could not decode file: {e}", self.parser_name)
        
        raise ContentExtractionError("No content source provided", self.parser_name)
    
    def _detect_type(self, content: str, file_path: Optional[str]) -> DocumentType:
        """Detect structured data type from content or file path."""
        if file_path:
            path = Path(file_path)
            extension_mapping = {
                '.json': DocumentType.JSON,
                '.yaml': DocumentType.YAML,
                '.yml': DocumentType.YAML,
                '.xml': DocumentType.XML,
                '.csv': DocumentType.CSV
            }
            if path.suffix.lower() in extension_mapping:
                return extension_mapping[path.suffix.lower()]
        
        # Try to detect from content
        content_stripped = content.strip()
        
        # JSON detection
        if (content_stripped.startswith('{') and content_stripped.endswith('}')) or \
           (content_stripped.startswith('[') and content_stripped.endswith(']')):
            try:
                json.loads(content)
                return DocumentType.JSON
            except json.JSONDecodeError:
                pass
        
        # XML detection
        if content_stripped.startswith('<') and content_stripped.endswith('>'):
            return DocumentType.XML
        
        # YAML detection (basic)
        if ':' in content and not content_stripped.startswith('<'):
            return DocumentType.YAML
        
        # CSV detection (basic)
        if ',' in content and '\n' in content:
            return DocumentType.CSV
        
        return DocumentType.JSON  # Default fallback
    
    async def _parse_structure(self, content: str, doc_type: DocumentType) -> tuple[Any, List[str]]:
        """Parse and validate structured data."""
        errors = []
        
        try:
            if doc_type == DocumentType.JSON:
                return json.loads(content), errors
            
            elif doc_type == DocumentType.YAML:
                if not yaml:
                    errors.append("YAML parsing requires PyYAML package")
                    return None, errors
                return yaml.safe_load(content), errors
            
            elif doc_type == DocumentType.XML:
                if not ET:
                    errors.append("XML parsing requires xml.etree.ElementTree")
                    return None, errors
                return ET.fromstring(content), errors
            
            elif doc_type == DocumentType.CSV:
                reader = csv.DictReader(StringIO(content))
                return list(reader), errors
            
            else:
                errors.append(f"Unsupported structured data type: {doc_type}")
                return None, errors
                
        except Exception as e:
            errors.append(f"Failed to parse {doc_type.value}: {e}")
            return None, errors
    
    async def _format_content(self, data: Any, doc_type: DocumentType) -> str:
        """Format structured data for better readability."""
        try:
            if doc_type == DocumentType.JSON:
                return json.dumps(data, indent=2, ensure_ascii=False)
            
            elif doc_type == DocumentType.YAML and yaml:
                return yaml.dump(data, default_flow_style=False, allow_unicode=True)
            
            elif doc_type == DocumentType.XML and ET and minidom:
                if isinstance(data, ET.Element):
                    rough_string = ET.tostring(data, encoding='unicode')
                    reparsed = minidom.parseString(rough_string)
                    return reparsed.toprettyxml(indent="  ")
            
            elif doc_type == DocumentType.CSV:
                if isinstance(data, list) and data:
                    output = StringIO()
                    if isinstance(data[0], dict):
                        fieldnames = data[0].keys()
                        writer = csv.DictWriter(output, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(data)
                    return output.getvalue()
            
        except Exception as e:
            logger.warning(f"Failed to format {doc_type.value}: {e}")
        
        # Return original content if formatting fails
        return str(data) if data is not None else ""
    
    async def _extract_structured_metadata(self, content: str, doc_type: DocumentType, parsed_data: Any) -> Dict[str, Any]:
        """Extract metadata specific to structured data."""
        metadata = {}
        
        lines = content.split('\n')
        metadata['line_count'] = len(lines)
        metadata['char_count'] = len(content)
        
        if parsed_data is not None:
            if doc_type == DocumentType.JSON:
                metadata.update(self._analyze_json(parsed_data))
            elif doc_type == DocumentType.YAML:
                metadata.update(self._analyze_yaml(parsed_data))
            elif doc_type == DocumentType.XML:
                metadata.update(self._analyze_xml(parsed_data))
            elif doc_type == DocumentType.CSV:
                metadata.update(self._analyze_csv(parsed_data))
        
        return metadata
    
    def _analyze_json(self, data: Any) -> Dict[str, Any]:
        """Analyze JSON structure."""
        metadata = {}
        
        if isinstance(data, dict):
            metadata['type'] = 'object'
            metadata['key_count'] = len(data)
            metadata['keys'] = list(data.keys())[:10]  # First 10 keys
            metadata['depth'] = self._calculate_depth(data)
        elif isinstance(data, list):
            metadata['type'] = 'array'
            metadata['item_count'] = len(data)
            metadata['depth'] = self._calculate_depth(data)
        else:
            metadata['type'] = type(data).__name__
        
        return metadata
    
    def _analyze_yaml(self, data: Any) -> Dict[str, Any]:
        """Analyze YAML structure."""
        # Similar to JSON analysis
        return self._analyze_json(data)
    
    def _analyze_xml(self, root: ET.Element) -> Dict[str, Any]:
        """Analyze XML structure."""
        metadata = {}
        
        metadata['root_tag'] = root.tag
        metadata['element_count'] = len(list(root.iter()))
        metadata['attribute_count'] = sum(len(elem.attrib) for elem in root.iter())
        
        # Get unique tags
        tags = set(elem.tag for elem in root.iter())
        metadata['unique_tags'] = list(tags)[:20]  # First 20 unique tags
        metadata['tag_count'] = len(tags)
        
        return metadata
    
    def _analyze_csv(self, data: List[Dict]) -> Dict[str, Any]:
        """Analyze CSV structure."""
        metadata = {}
        
        if data:
            metadata['row_count'] = len(data)
            metadata['column_count'] = len(data[0]) if data[0] else 0
            metadata['columns'] = list(data[0].keys()) if data[0] else []
        else:
            metadata['row_count'] = 0
            metadata['column_count'] = 0
            metadata['columns'] = []
        
        return metadata
    
    def _calculate_depth(self, obj: Any, current_depth: int = 0) -> int:
        """Calculate the maximum depth of nested structures."""
        if isinstance(obj, dict):
            if not obj:
                return current_depth
            return max(self._calculate_depth(v, current_depth + 1) for v in obj.values())
        elif isinstance(obj, list):
            if not obj:
                return current_depth
            return max(self._calculate_depth(item, current_depth + 1) for item in obj)
        else:
            return current_depth
    
    async def _extract_structured_sections(self, data: Any, doc_type: DocumentType) -> List[Dict[str, Any]]:
        """Extract sections from structured data."""
        sections = []
        
        if doc_type == DocumentType.JSON and isinstance(data, dict):
            for key, value in data.items():
                sections.append({
                    'type': 'json_key',
                    'key': key,
                    'value_type': type(value).__name__,
                    'content': str(value)[:100] + '...' if len(str(value)) > 100 else str(value)
                })
        
        elif doc_type == DocumentType.XML and ET:
            if isinstance(data, ET.Element):
                for child in data:
                    sections.append({
                        'type': 'xml_element',
                        'tag': child.tag,
                        'attributes': child.attrib,
                        'text': (child.text or '').strip()[:100]
                    })
        
        elif doc_type == DocumentType.CSV and isinstance(data, list):
            for i, row in enumerate(data[:10]):  # First 10 rows
                sections.append({
                    'type': 'csv_row',
                    'row_index': i,
                    'data': row
                })
        
        return sections
