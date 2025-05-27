"""
Main parsing service with multi-parser support and intelligent routing.
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Union
import logging
from dataclasses import dataclass
from pathlib import Path

from .base import (
    DocumentParser, ParsedDocument, ParsingRequest, ParsingResponse,
    DocumentType, ParsingError, UnsupportedFormatError
)
from .factory import parser_factory

logger = logging.getLogger(__name__)

@dataclass
class ParsingServiceConfig:
    """Configuration for the parsing service."""
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    timeout: float = 300.0  # 5 minutes
    enable_caching: bool = True
    cache_ttl: int = 3600  # 1 hour
    auto_detect_type: bool = True
    extract_metadata: bool = True
    extract_sections: bool = False
    parallel_parsing: bool = True
    max_concurrent_parsers: int = 5

class ParsingService:
    """
    Main parsing service with multi-parser support and intelligent routing.

    Features:
    - Multi-format support (code, text, PDF, structured data)
    - Intelligent parser selection
    - Batch processing with parallelization
    - Caching for performance
    - Error handling and fallback strategies
    """

    def __init__(self, config: Optional[ParsingServiceConfig] = None):
        self.config = config or ParsingServiceConfig()
        self.parsers: Dict[str, DocumentParser] = {}
        self._initialized = False
        self._cache: Dict[str, ParsingResponse] = {}
        self._stats = {
            "total_requests": 0,
            "successful_parses": 0,
            "failed_parses": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_processing_time": 0.0,
            "parsers_used": {}
        }

    async def initialize(self) -> None:
        """Initialize the parsing service."""
        try:
            # Create all available parsers
            self.parsers = parser_factory.create_all_parsers(
                max_file_size=self.config.max_file_size
            )

            # Initialize parsers
            self.parsers = await parser_factory.initialize_all_parsers(self.parsers)

            self._initialized = True
            logger.info(f"Parsing service initialized with {len(self.parsers)} parsers")

        except Exception as e:
            raise ParsingError(f"Failed to initialize parsing service: {e}") from e

    async def parse_document(
        self,
        content: Optional[str] = None,
        file_path: Optional[str] = None,
        file_data: Optional[bytes] = None,
        document_type: Optional[DocumentType] = None,
        **kwargs
    ) -> ParsingResponse:
        """
        Parse a single document.

        Args:
            content: Document content as string
            file_path: Path to the document file
            file_data: Document content as bytes
            document_type: Explicit document type (auto-detected if None)
            **kwargs: Additional parsing options

        Returns:
            Parsing response with extracted content and metadata
        """
        if not self._initialized:
            await self.initialize()

        # Create parsing request
        # Extract known parameters to avoid duplicates
        extract_metadata = kwargs.pop('extract_metadata', self.config.extract_metadata)
        extract_sections = kwargs.pop('extract_sections', self.config.extract_sections)

        request = ParsingRequest(
            content=content,
            file_path=file_path,
            file_data=file_data,
            document_type=document_type,
            extract_metadata=extract_metadata,
            extract_sections=extract_sections,
            **kwargs
        )

        return await self._parse_single(request)

    async def parse_batch(
        self,
        requests: List[ParsingRequest],
        max_concurrent: Optional[int] = None
    ) -> List[ParsingResponse]:
        """
        Parse multiple documents in parallel.

        Args:
            requests: List of parsing requests
            max_concurrent: Maximum concurrent parsers (uses config default if None)

        Returns:
            List of parsing responses
        """
        if not self._initialized:
            await self.initialize()

        if not requests:
            return []

        max_concurrent = max_concurrent or self.config.max_concurrent_parsers

        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)

        async def parse_with_semaphore(request: ParsingRequest) -> ParsingResponse:
            async with semaphore:
                return await self._parse_single(request)

        # Execute parsing tasks
        if self.config.parallel_parsing:
            tasks = [parse_with_semaphore(request) for request in requests]
            responses = await asyncio.gather(*tasks, return_exceptions=True)

            # Convert exceptions to error responses
            results = []
            for i, response in enumerate(responses):
                if isinstance(response, Exception):
                    error_response = ParsingResponse(
                        document=ParsedDocument(
                            content="",
                            document_type=requests[i].document_type or DocumentType.UNKNOWN,
                            file_path=requests[i].file_path
                        ),
                        processing_time=0.0,
                        parser_used="error",
                        success=False,
                        errors=[str(response)]
                    )
                    results.append(error_response)
                else:
                    results.append(response)

            return results
        else:
            # Sequential processing
            results = []
            for request in requests:
                response = await self._parse_single(request)
                results.append(response)
            return results

    async def parse_directory(
        self,
        directory_path: str,
        recursive: bool = True,
        file_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None
    ) -> List[ParsingResponse]:
        """
        Parse all supported files in a directory.

        Args:
            directory_path: Path to the directory
            recursive: Whether to search subdirectories
            file_patterns: File patterns to include (e.g., ['*.py', '*.js'])
            exclude_patterns: File patterns to exclude

        Returns:
            List of parsing responses for all processed files
        """
        directory = Path(directory_path)
        if not directory.exists() or not directory.is_dir():
            raise ParsingError(f"Directory not found: {directory_path}")

        # Find files to parse
        files = self._find_files(directory, recursive, file_patterns, exclude_patterns)

        # Create parsing requests
        requests = []
        for file_path in files:
            request = ParsingRequest(
                file_path=str(file_path),
                extract_metadata=self.config.extract_metadata,
                extract_sections=self.config.extract_sections
            )
            requests.append(request)

        logger.info(f"Found {len(requests)} files to parse in {directory_path}")

        # Parse files in batches
        return await self.parse_batch(requests)

    async def get_supported_formats(self) -> Dict[str, List[str]]:
        """Get information about supported file formats."""
        if not self._initialized:
            await self.initialize()

        formats = {}
        for parser_name, parser in self.parsers.items():
            formats[parser_name] = [doc_type.value for doc_type in parser.supported_types]

        return formats

    def get_stats(self) -> Dict[str, Any]:
        """Get parsing service statistics."""
        stats = self._stats.copy()

        # Calculate derived metrics
        if stats["total_requests"] > 0:
            stats["success_rate"] = stats["successful_parses"] / stats["total_requests"]
            stats["cache_hit_rate"] = stats["cache_hits"] / stats["total_requests"]
            stats["avg_processing_time"] = stats["total_processing_time"] / stats["total_requests"]
        else:
            stats["success_rate"] = 0.0
            stats["cache_hit_rate"] = 0.0
            stats["avg_processing_time"] = 0.0

        stats["cache_size"] = len(self._cache)
        stats["active_parsers"] = len(self.parsers)

        return stats

    async def health_check(self) -> Dict[str, Any]:
        """Check the health of the parsing service."""
        if not self._initialized:
            try:
                await self.initialize()
            except Exception as e:
                return {
                    "status": "unhealthy",
                    "error": f"Initialization failed: {e}",
                    "stats": self.get_stats()
                }

        health_status = {
            "status": "healthy",
            "parsers": {},
            "stats": self.get_stats(),
            "config": {
                "max_file_size": self.config.max_file_size,
                "timeout": self.config.timeout,
                "caching_enabled": self.config.enable_caching
            }
        }

        # Check each parser
        for parser_name, parser in self.parsers.items():
            try:
                # Simple health check - create a minimal request
                test_request = ParsingRequest(content="test")
                can_parse = await parser.can_parse(test_request)
                health_status["parsers"][parser_name] = {
                    "status": "healthy",
                    "can_parse_test": can_parse,
                    "supported_types": [t.value for t in parser.supported_types]
                }
            except Exception as e:
                health_status["parsers"][parser_name] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health_status["status"] = "degraded"

        return health_status

    async def cleanup(self) -> None:
        """Cleanup all resources."""
        for parser in self.parsers.values():
            try:
                await parser.cleanup()
            except Exception as e:
                logger.warning(f"Error cleaning up parser: {e}")

        self.parsers.clear()
        self._cache.clear()
        self._initialized = False
        logger.info("Parsing service cleaned up")

    # Private methods

    async def _parse_single(self, request: ParsingRequest) -> ParsingResponse:
        """Parse a single document with caching and error handling."""
        start_time = time.time()
        self._stats["total_requests"] += 1

        try:
            # Check cache
            if self.config.enable_caching:
                cache_key = self._generate_cache_key(request)
                if cache_key in self._cache:
                    self._stats["cache_hits"] += 1
                    cached_response = self._cache[cache_key]
                    logger.debug(f"Cache hit for {request.file_path or 'content'}")
                    return cached_response
                else:
                    self._stats["cache_misses"] += 1

            # Find best parser
            parser = await parser_factory.get_best_parser(request)
            if not parser:
                raise UnsupportedFormatError(
                    f"No parser available for document type: {request.document_type}"
                )

            # Parse document with timeout
            response = await asyncio.wait_for(
                parser.parse(request),
                timeout=self.config.timeout
            )

            # Update stats
            self._stats["successful_parses"] += 1
            self._stats["total_processing_time"] += response.processing_time

            parser_name = parser.parser_name
            if parser_name not in self._stats["parsers_used"]:
                self._stats["parsers_used"][parser_name] = 0
            self._stats["parsers_used"][parser_name] += 1

            # Cache successful response
            if self.config.enable_caching and response.success:
                cache_key = self._generate_cache_key(request)
                self._cache[cache_key] = response

            logger.debug(f"Successfully parsed {request.file_path or 'content'} using {parser_name}")
            return response

        except asyncio.TimeoutError:
            self._stats["failed_parses"] += 1
            error_msg = f"Parsing timeout after {self.config.timeout} seconds"
            logger.error(error_msg)

            return ParsingResponse(
                document=ParsedDocument(
                    content="",
                    document_type=request.document_type or DocumentType.UNKNOWN,
                    file_path=request.file_path
                ),
                processing_time=time.time() - start_time,
                parser_used="timeout",
                success=False,
                errors=[error_msg]
            )

        except Exception as e:
            self._stats["failed_parses"] += 1
            logger.error(f"Parsing failed: {e}")

            return ParsingResponse(
                document=ParsedDocument(
                    content="",
                    document_type=request.document_type or DocumentType.UNKNOWN,
                    file_path=request.file_path
                ),
                processing_time=time.time() - start_time,
                parser_used="error",
                success=False,
                errors=[str(e)]
            )

    def _generate_cache_key(self, request: ParsingRequest) -> str:
        """Generate cache key for the request."""
        import hashlib

        # Create a string representation of the request
        key_parts = [
            request.content or "",
            request.file_path or "",
            str(request.document_type.value if request.document_type else ""),
            str(request.extract_metadata),
            str(request.extract_sections)
        ]

        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _find_files(
        self,
        directory: Path,
        recursive: bool,
        file_patterns: Optional[List[str]],
        exclude_patterns: Optional[List[str]]
    ) -> List[Path]:
        """Find files to parse in directory."""
        files = []

        # Default patterns for supported file types
        if file_patterns is None:
            file_patterns = [
                '*.py', '*.js', '*.ts', '*.java', '*.cpp', '*.c', '*.h',
                '*.cs', '*.go', '*.rs', '*.php', '*.rb', '*.kt', '*.swift',
                '*.txt', '*.md', '*.markdown', '*.pdf', '*.json', '*.yaml',
                '*.yml', '*.xml', '*.csv', '*.html'
            ]

        # Search for files
        for pattern in file_patterns:
            if recursive:
                found_files = directory.rglob(pattern)
            else:
                found_files = directory.glob(pattern)

            for file_path in found_files:
                if file_path.is_file():
                    # Check exclude patterns
                    if exclude_patterns:
                        excluded = any(file_path.match(exclude_pattern) for exclude_pattern in exclude_patterns)
                        if excluded:
                            continue

                    files.append(file_path)

        return files

# Global service instance
parsing_service = ParsingService()
