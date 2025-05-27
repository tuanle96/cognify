"""Custom assertions for Cognify tests."""

from typing import Dict, Any, List
from app.services.chunking.base import ChunkingResult, AgenticChunk
from app.services.llm.base import LLMResponse


def assert_chunking_response_valid(result: ChunkingResult) -> None:
    """Assert that a chunking result has valid structure and content."""
    # Basic structure validation
    assert hasattr(result, 'chunks'), "Result must have 'chunks' attribute"
    assert hasattr(result, 'metadata'), "Result must have 'metadata' attribute"
    assert hasattr(result, 'strategy_used'), "Result must have 'strategy_used' attribute"
    assert hasattr(result, 'quality_score'), "Result must have 'quality_score' attribute"
    assert hasattr(result, 'processing_time'), "Result must have 'processing_time' attribute"
    assert isinstance(result.chunks, list), "Chunks must be a list"
    assert isinstance(result.metadata, dict), "Metadata must be a dictionary"

    # Consistency validation
    assert result.chunk_count == len(result.chunks), \
        "chunk_count must match actual chunk count"

    # Chunk validation
    for i, chunk in enumerate(result.chunks):
        assert_chunk_valid(chunk, f"Chunk {i}")


def assert_chunk_valid(chunk: AgenticChunk, context: str = "Chunk") -> None:
    """Assert that a chunk has valid structure and content."""
    assert hasattr(chunk, 'content'), f"{context} must have 'content' attribute"
    assert hasattr(chunk, 'metadata'), f"{context} must have 'metadata' attribute"
    assert hasattr(chunk, 'start_line'), f"{context} must have 'start_line' attribute"
    assert hasattr(chunk, 'end_line'), f"{context} must have 'end_line' attribute"
    assert hasattr(chunk, 'chunk_type'), f"{context} must have 'chunk_type' attribute"
    assert isinstance(chunk.content, str), f"{context} content must be a string"
    assert len(chunk.content.strip()) > 0, f"{context} content cannot be empty"

    # Line number validation
    assert isinstance(chunk.start_line, int), f"{context} start_line must be an integer"
    assert isinstance(chunk.end_line, int), f"{context} end_line must be an integer"
    assert chunk.start_line <= chunk.end_line, f"{context} start_line must be <= end_line"
    assert chunk.start_line >= 1, f"{context} start_line must be >= 1"


def assert_llm_response_valid(response: LLMResponse) -> None:
    """Assert that an LLM response has valid structure and content."""
    assert hasattr(response, 'content'), "LLM response must have 'content' attribute"
    assert hasattr(response, 'usage'), "LLM response must have 'usage' attribute"
    assert isinstance(response.content, str), "LLM response content must be a string"
    assert isinstance(response.usage, dict), "LLM response usage must be a dictionary"
    assert len(response.content.strip()) > 0, "LLM response content cannot be empty"

    # Usage validation
    if 'total_tokens' in response.usage:
        assert isinstance(response.usage['total_tokens'], int), \
            "total_tokens must be an integer"
        assert response.usage['total_tokens'] > 0, \
            "total_tokens must be positive"


def assert_agent_decision_valid(decision: Dict[str, Any]) -> None:
    """Assert that an agent decision has valid structure."""
    required_keys = ['agent_name', 'decision', 'reasoning']
    for key in required_keys:
        assert key in decision, f"Agent decision must contain '{key}'"

    assert isinstance(decision['agent_name'], str), "agent_name must be a string"
    assert isinstance(decision['reasoning'], str), "reasoning must be a string"
    assert len(decision['agent_name'].strip()) > 0, "agent_name cannot be empty"
    assert len(decision['reasoning'].strip()) > 0, "reasoning cannot be empty"


def assert_performance_acceptable(
    processing_time: float,
    max_time: float = 10.0,
    context: str = "Operation"
) -> None:
    """Assert that processing time is within acceptable limits."""
    assert isinstance(processing_time, (int, float)), \
        f"{context} processing time must be a number"
    assert processing_time >= 0, \
        f"{context} processing time cannot be negative"
    assert processing_time <= max_time, \
        f"{context} took {processing_time:.2f}s, exceeds limit of {max_time}s"


def assert_chunks_non_overlapping(chunks: List[AgenticChunk]) -> None:
    """Assert that chunks do not overlap in line numbers."""
    sorted_chunks = sorted(chunks, key=lambda c: c.start_line)

    for i in range(len(sorted_chunks) - 1):
        current_chunk = sorted_chunks[i]
        next_chunk = sorted_chunks[i + 1]

        current_end = current_chunk.end_line
        next_start = next_chunk.start_line

        assert current_end < next_start, \
            f"Chunks overlap: chunk ending at line {current_end} " \
            f"overlaps with chunk starting at line {next_start}"


def assert_chunks_cover_content(chunks: List[AgenticChunk], total_lines: int) -> None:
    """Assert that chunks adequately cover the source content."""
    if not chunks:
        return  # Empty chunks list is valid for empty content

    sorted_chunks = sorted(chunks, key=lambda c: c.start_line)

    # Check first chunk starts reasonably early
    first_start = sorted_chunks[0].start_line
    assert first_start <= 5, \
        f"First chunk starts too late at line {first_start}"

    # Check last chunk ends reasonably late
    last_end = sorted_chunks[-1].end_line
    assert last_end >= total_lines - 5, \
        f"Last chunk ends too early at line {last_end}, total lines: {total_lines}"


def assert_quality_score_valid(quality_score: float) -> None:
    """Assert that a quality score is within valid range."""
    assert isinstance(quality_score, (int, float)), \
        "Quality score must be a number"
    assert 0.0 <= quality_score <= 1.0, \
        f"Quality score {quality_score} must be between 0.0 and 1.0"


def assert_strategy_used_valid(strategy: str) -> None:
    """Assert that the strategy used is valid."""
    valid_strategies = ['agentic', 'ast', 'hybrid']
    assert strategy in valid_strategies, \
        f"Strategy '{strategy}' must be one of {valid_strategies}"


def assert_language_supported(language: str) -> None:
    """Assert that the language is supported."""
    supported_languages = ['python', 'javascript', 'typescript', 'go', 'java', 'rust']
    assert language in supported_languages, \
        f"Language '{language}' must be one of {supported_languages}"


def assert_purpose_valid(purpose: str) -> None:
    """Assert that the purpose is valid."""
    valid_purposes = ['general', 'code_review', 'bug_detection', 'documentation']
    assert purpose in valid_purposes, \
        f"Purpose '{purpose}' must be one of {valid_purposes}"


def assert_file_path_valid(file_path: str) -> None:
    """Assert that the file path is valid."""
    assert isinstance(file_path, str), "File path must be a string"
    assert len(file_path.strip()) > 0, "File path cannot be empty"
    assert not file_path.startswith('/'), "File path should be relative"


def assert_content_not_empty(content: str) -> None:
    """Assert that content is not empty."""
    assert isinstance(content, str), "Content must be a string"
    assert len(content.strip()) > 0, "Content cannot be empty"


def assert_metadata_contains_keys(metadata: Dict[str, Any], required_keys: List[str]) -> None:
    """Assert that metadata contains all required keys."""
    for key in required_keys:
        assert key in metadata, f"Metadata must contain key '{key}'"


def assert_response_time_reasonable(start_time: float, end_time: float, max_seconds: float = 30.0) -> None:
    """Assert that response time is reasonable."""
    elapsed = end_time - start_time
    assert elapsed <= max_seconds, \
        f"Response took {elapsed:.2f}s, exceeds limit of {max_seconds}s"
    assert elapsed >= 0, "Response time cannot be negative"
