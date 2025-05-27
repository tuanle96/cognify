"""
Custom chunking strategies and user-defined rules.

Allows users to define their own chunking logic and rules.
"""

import json
import re
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

import structlog

from app.services.chunking.base import AgenticChunk, ChunkType, ChunkMetadata
from app.services.chunking.language_support import SupportedLanguage, LanguageDetector

logger = structlog.get_logger(__name__)


class RuleType(Enum):
    """Types of chunking rules."""
    SIZE_BASED = "size_based"
    PATTERN_BASED = "pattern_based"
    SEMANTIC_BASED = "semantic_based"
    FUNCTION_BASED = "function_based"
    CLASS_BASED = "class_based"
    CUSTOM_LOGIC = "custom_logic"


class RuleCondition(Enum):
    """Rule condition operators."""
    EQUALS = "equals"
    CONTAINS = "contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    REGEX_MATCH = "regex_match"
    LINE_COUNT = "line_count"
    CHAR_COUNT = "char_count"
    COMPLEXITY = "complexity"


@dataclass
class ChunkingRule:
    """Definition of a custom chunking rule."""
    id: str
    name: str
    description: str
    rule_type: RuleType
    condition: RuleCondition
    value: Union[str, int, float]
    action: str  # "split", "merge", "mark", "skip"
    priority: int = 1  # Higher number = higher priority
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CustomStrategy:
    """Definition of a custom chunking strategy."""
    id: str
    name: str
    description: str
    language: Optional[str] = None
    rules: List[ChunkingRule] = field(default_factory=list)
    global_settings: Dict[str, Any] = field(default_factory=dict)
    created_by: str = "system"
    version: str = "1.0"
    enabled: bool = True


class BaseCustomChunker(ABC):
    """Base class for custom chunking implementations."""

    def __init__(self, strategy: CustomStrategy):
        self.strategy = strategy
        self.logger = structlog.get_logger(f"custom_chunker.{strategy.id}")

    @abstractmethod
    async def chunk_content(self, content: str, file_path: str) -> List[AgenticChunk]:
        """Chunk content according to custom strategy."""
        pass

    def _apply_rules(self, content: str, initial_chunks: List[Dict]) -> List[Dict]:
        """Apply custom rules to initial chunks."""
        chunks = initial_chunks.copy()

        # Sort rules by priority (highest first)
        sorted_rules = sorted(
            [rule for rule in self.strategy.rules if rule.enabled],
            key=lambda r: r.priority,
            reverse=True
        )

        for rule in sorted_rules:
            chunks = self._apply_single_rule(rule, chunks, content)

        return chunks

    def _apply_single_rule(self, rule: ChunkingRule, chunks: List[Dict], content: str) -> List[Dict]:
        """Apply a single rule to chunks."""
        try:
            if rule.rule_type == RuleType.SIZE_BASED:
                return self._apply_size_rule(rule, chunks)
            elif rule.rule_type == RuleType.PATTERN_BASED:
                return self._apply_pattern_rule(rule, chunks, content)
            elif rule.rule_type == RuleType.SEMANTIC_BASED:
                return self._apply_semantic_rule(rule, chunks, content)
            elif rule.rule_type == RuleType.FUNCTION_BASED:
                return self._apply_function_rule(rule, chunks, content)
            elif rule.rule_type == RuleType.CLASS_BASED:
                return self._apply_class_rule(rule, chunks, content)
            else:
                self.logger.warning("Unknown rule type", rule_type=rule.rule_type)
                return chunks
        except Exception as e:
            self.logger.error("Failed to apply rule", rule_id=rule.id, error=str(e))
            return chunks

    def _apply_size_rule(self, rule: ChunkingRule, chunks: List[Dict]) -> List[Dict]:
        """Apply size-based rules."""
        result = []

        for chunk in chunks:
            chunk_size = chunk.get("end_line", 0) - chunk.get("start_line", 0) + 1

            if rule.condition == RuleCondition.LINE_COUNT:
                if rule.action == "split" and chunk_size > rule.value:
                    # Split large chunks
                    result.extend(self._split_chunk(chunk, rule.value))
                elif rule.action == "merge" and chunk_size < rule.value:
                    # Mark for merging (will be handled later)
                    chunk["_merge_candidate"] = True
                    result.append(chunk)
                else:
                    result.append(chunk)
            else:
                result.append(chunk)

        # Handle merge candidates
        return self._merge_small_chunks(result)

    def _apply_pattern_rule(self, rule: ChunkingRule, chunks: List[Dict], content: str) -> List[Dict]:
        """Apply pattern-based rules."""
        lines = content.split('\n')
        result = []

        for chunk in chunks:
            start_idx = chunk.get("start_line", 1) - 1
            end_idx = chunk.get("end_line", len(lines)) - 1
            chunk_content = '\n'.join(lines[start_idx:end_idx + 1])

            if self._matches_condition(chunk_content, rule.condition, rule.value):
                if rule.action == "split":
                    result.extend(self._split_by_pattern(chunk, rule.value, lines))
                elif rule.action == "mark":
                    chunk["custom_type"] = rule.metadata.get("mark_type", "special")
                    result.append(chunk)
                elif rule.action == "skip":
                    continue  # Skip this chunk
                else:
                    result.append(chunk)
            else:
                result.append(chunk)

        return result

    def _apply_semantic_rule(self, rule: ChunkingRule, chunks: List[Dict], content: str) -> List[Dict]:
        """Apply semantic-based rules."""
        # This would integrate with semantic analysis
        # For now, implement basic semantic grouping
        return self._group_related_chunks(chunks, rule)

    def _apply_function_rule(self, rule: ChunkingRule, chunks: List[Dict], content: str) -> List[Dict]:
        """Apply function-based rules."""
        result = []

        for chunk in chunks:
            if chunk.get("type") == "function":
                if rule.action == "merge" and rule.condition == RuleCondition.CONTAINS:
                    # Merge functions that contain specific patterns
                    chunk["_merge_with_next"] = True
                result.append(chunk)
            else:
                result.append(chunk)

        return result

    def _apply_class_rule(self, rule: ChunkingRule, chunks: List[Dict], content: str) -> List[Dict]:
        """Apply class-based rules."""
        result = []

        for chunk in chunks:
            if chunk.get("type") == "class":
                if rule.action == "split" and rule.condition == RuleCondition.LINE_COUNT:
                    # Split large classes
                    if (chunk.get("end_line", 0) - chunk.get("start_line", 0)) > rule.value:
                        result.extend(self._split_class_by_methods(chunk, content))
                    else:
                        result.append(chunk)
                else:
                    result.append(chunk)
            else:
                result.append(chunk)

        return result

    def _matches_condition(self, content: str, condition: RuleCondition, value: Union[str, int, float]) -> bool:
        """Check if content matches the given condition."""
        if condition == RuleCondition.CONTAINS:
            return str(value) in content
        elif condition == RuleCondition.STARTS_WITH:
            return content.strip().startswith(str(value))
        elif condition == RuleCondition.ENDS_WITH:
            return content.strip().endswith(str(value))
        elif condition == RuleCondition.REGEX_MATCH:
            return bool(re.search(str(value), content))
        elif condition == RuleCondition.LINE_COUNT:
            return len(content.split('\n')) >= value
        elif condition == RuleCondition.CHAR_COUNT:
            return len(content) >= value
        else:
            return False

    def _split_chunk(self, chunk: Dict, max_size: int) -> List[Dict]:
        """Split a chunk into smaller pieces."""
        start_line = chunk.get("start_line", 1)
        end_line = chunk.get("end_line", start_line)
        chunk_size = end_line - start_line + 1

        if chunk_size <= max_size:
            return [chunk]

        # Split into roughly equal parts
        num_parts = (chunk_size + max_size - 1) // max_size
        part_size = chunk_size // num_parts

        parts = []
        current_start = start_line

        for i in range(num_parts):
            current_end = min(current_start + part_size - 1, end_line)
            if i == num_parts - 1:  # Last part gets remaining lines
                current_end = end_line

            part = chunk.copy()
            part["start_line"] = current_start
            part["end_line"] = current_end
            part["name"] = f"{chunk.get('name', 'chunk')}_{i+1}"
            parts.append(part)

            current_start = current_end + 1

        return parts

    def _split_by_pattern(self, chunk: Dict, pattern: str, lines: List[str]) -> List[Dict]:
        """Split chunk by pattern matches."""
        start_idx = chunk.get("start_line", 1) - 1
        end_idx = chunk.get("end_line", len(lines)) - 1

        split_points = []
        for i in range(start_idx, end_idx + 1):
            if re.search(pattern, lines[i]):
                split_points.append(i + 1)  # Convert to 1-based

        if not split_points:
            return [chunk]

        # Create chunks between split points
        parts = []
        current_start = chunk.get("start_line", 1)

        for split_point in split_points:
            if split_point > current_start:
                part = chunk.copy()
                part["start_line"] = current_start
                part["end_line"] = split_point - 1
                parts.append(part)
                current_start = split_point

        # Add final part
        if current_start <= chunk.get("end_line", current_start):
            part = chunk.copy()
            part["start_line"] = current_start
            part["end_line"] = chunk.get("end_line", current_start)
            parts.append(part)

        return parts

    def _merge_small_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Merge chunks marked for merging."""
        result = []
        i = 0

        while i < len(chunks):
            chunk = chunks[i]

            if chunk.get("_merge_candidate") and i + 1 < len(chunks):
                # Merge with next chunk
                next_chunk = chunks[i + 1]
                merged = {
                    "start_line": chunk.get("start_line", 1),
                    "end_line": next_chunk.get("end_line", chunk.get("end_line", 1)),
                    "type": "merged",
                    "name": f"{chunk.get('name', 'chunk')}+{next_chunk.get('name', 'chunk')}",
                    "complexity": max(chunk.get("complexity", 1), next_chunk.get("complexity", 1))
                }
                result.append(merged)
                i += 2  # Skip next chunk as it's been merged
            else:
                # Remove merge marker and add chunk
                chunk.pop("_merge_candidate", None)
                result.append(chunk)
                i += 1

        return result

    def _group_related_chunks(self, chunks: List[Dict], rule: ChunkingRule) -> List[Dict]:
        """Group semantically related chunks."""
        # Simple implementation - group by naming patterns
        groups = {}
        ungrouped = []

        for chunk in chunks:
            chunk_name = chunk.get("name", "")

            # Look for common prefixes or patterns
            group_key = None
            if rule.condition == RuleCondition.CONTAINS and str(rule.value) in chunk_name:
                group_key = str(rule.value)
            elif rule.condition == RuleCondition.STARTS_WITH and chunk_name.startswith(str(rule.value)):
                group_key = str(rule.value)

            if group_key:
                if group_key not in groups:
                    groups[group_key] = []
                groups[group_key].append(chunk)
            else:
                ungrouped.append(chunk)

        # Create grouped chunks
        result = ungrouped.copy()
        for group_key, group_chunks in groups.items():
            if len(group_chunks) > 1:
                merged = {
                    "start_line": min(c.get("start_line", 1) for c in group_chunks),
                    "end_line": max(c.get("end_line", 1) for c in group_chunks),
                    "type": "semantic_group",
                    "name": f"{group_key}_group",
                    "complexity": sum(c.get("complexity", 1) for c in group_chunks),
                    "grouped_chunks": group_chunks
                }
                result.append(merged)
            else:
                result.extend(group_chunks)

        return result

    def _split_class_by_methods(self, chunk: Dict, content: str) -> List[Dict]:
        """Split a class chunk by its methods."""
        lines = content.split('\n')
        start_idx = chunk.get("start_line", 1) - 1
        end_idx = chunk.get("end_line", len(lines)) - 1

        # Find method definitions within the class
        method_pattern = r"^\s*(def|function|public|private|protected)\s+\w+"
        method_starts = []

        for i in range(start_idx, end_idx + 1):
            if re.search(method_pattern, lines[i]):
                method_starts.append(i + 1)  # Convert to 1-based

        if len(method_starts) <= 1:
            return [chunk]  # No methods to split by

        # Create chunks for each method
        parts = []
        for i, method_start in enumerate(method_starts):
            method_end = method_starts[i + 1] - 1 if i + 1 < len(method_starts) else chunk.get("end_line", method_start)

            part = {
                "start_line": method_start,
                "end_line": method_end,
                "type": "method",
                "name": f"{chunk.get('name', 'class')}_method_{i+1}",
                "complexity": chunk.get("complexity", 1)
            }
            parts.append(part)

        return parts


class RuleBasedCustomChunker(BaseCustomChunker):
    """Rule-based custom chunker implementation."""

    async def chunk_content(self, content: str, file_path: str) -> List[AgenticChunk]:
        """Chunk content using rule-based approach."""
        # Start with language-specific chunking
        language = LanguageDetector.detect_language(file_path, content)

        if language.value != "unknown":
            from app.services.chunking.language_support import LanguageSpecificChunker
            lang_chunker = LanguageSpecificChunker(language)
            initial_chunks = lang_chunker.get_optimal_chunk_boundaries(content)
        else:
            # Fallback to simple line-based chunking
            lines = content.split('\n')
            initial_chunks = [{
                "start_line": 1,
                "end_line": len(lines),
                "type": "file",
                "name": "entire_file",
                "complexity": 1
            }]

        # Apply custom rules
        processed_chunks = self._apply_rules(content, initial_chunks)

        # Convert to Chunk objects
        chunks = []
        lines = content.split('\n')

        for i, chunk_data in enumerate(processed_chunks):
            start_line = chunk_data.get("start_line", 1)
            end_line = chunk_data.get("end_line", len(lines))

            # Extract content
            chunk_content = '\n'.join(lines[start_line-1:end_line])

            # Determine chunk type
            chunk_type = self._determine_chunk_type(chunk_data.get("type", "unknown"))

            chunk = AgenticChunk(
                id=f"custom_{self.strategy.id}_{i}",
                content=chunk_content,
                language=language.value,
                chunk_type=chunk_type,
                name=chunk_data.get("name", f"chunk_{i}"),
                start_line=start_line,
                end_line=end_line,
                file_path=file_path,
                dependencies=chunk_data.get("dependencies", []),
                semantic_relationships=[],
                purpose_optimization={},
                quality_score=0.8,
                metadata=ChunkMetadata(
                    complexity=chunk_data.get("complexity", 1)
                )
            )
            chunks.append(chunk)

        return chunks

    def _determine_chunk_type(self, type_str: str) -> ChunkType:
        """Convert string type to ChunkType enum."""
        type_mapping = {
            "function": ChunkType.FUNCTION,
            "class": ChunkType.CLASS,
            "method": ChunkType.METHOD,
            "import": ChunkType.IMPORT,
            "imports": ChunkType.IMPORT,
            "semantic_group": ChunkType.SEMANTIC_BLOCK,
            "merged": ChunkType.SEMANTIC_BLOCK,
            "file": ChunkType.MODULE
        }

        return type_mapping.get(type_str, ChunkType.UNKNOWN)


class CustomStrategyManager:
    """Manager for custom chunking strategies."""

    def __init__(self):
        self.strategies: Dict[str, CustomStrategy] = {}
        self.logger = structlog.get_logger("custom_strategy_manager")

    def register_strategy(self, strategy: CustomStrategy) -> bool:
        """Register a new custom strategy."""
        try:
            self.strategies[strategy.id] = strategy
            self.logger.info("Strategy registered", strategy_id=strategy.id, name=strategy.name)
            return True
        except Exception as e:
            self.logger.error("Failed to register strategy", strategy_id=strategy.id, error=str(e))
            return False

    def get_strategy(self, strategy_id: str) -> Optional[CustomStrategy]:
        """Get a strategy by ID."""
        return self.strategies.get(strategy_id)

    def list_strategies(self, language: Optional[str] = None) -> List[CustomStrategy]:
        """List all strategies, optionally filtered by language."""
        strategies = list(self.strategies.values())

        if language:
            strategies = [s for s in strategies if s.language is None or s.language == language]

        return [s for s in strategies if s.enabled]

    def create_chunker(self, strategy_id: str) -> Optional[BaseCustomChunker]:
        """Create a chunker for the given strategy."""
        strategy = self.get_strategy(strategy_id)
        if not strategy:
            return None

        return RuleBasedCustomChunker(strategy)

    def export_strategy(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """Export strategy to JSON-serializable format."""
        strategy = self.get_strategy(strategy_id)
        if not strategy:
            return None

        return {
            "id": strategy.id,
            "name": strategy.name,
            "description": strategy.description,
            "language": strategy.language,
            "rules": [
                {
                    "id": rule.id,
                    "name": rule.name,
                    "description": rule.description,
                    "rule_type": rule.rule_type.value,
                    "condition": rule.condition.value,
                    "value": rule.value,
                    "action": rule.action,
                    "priority": rule.priority,
                    "enabled": rule.enabled,
                    "metadata": rule.metadata
                }
                for rule in strategy.rules
            ],
            "global_settings": strategy.global_settings,
            "created_by": strategy.created_by,
            "version": strategy.version,
            "enabled": strategy.enabled
        }

    def import_strategy(self, strategy_data: Dict[str, Any]) -> bool:
        """Import strategy from JSON data."""
        try:
            rules = []
            for rule_data in strategy_data.get("rules", []):
                rule = ChunkingRule(
                    id=rule_data["id"],
                    name=rule_data["name"],
                    description=rule_data["description"],
                    rule_type=RuleType(rule_data["rule_type"]),
                    condition=RuleCondition(rule_data["condition"]),
                    value=rule_data["value"],
                    action=rule_data["action"],
                    priority=rule_data.get("priority", 1),
                    enabled=rule_data.get("enabled", True),
                    metadata=rule_data.get("metadata", {})
                )
                rules.append(rule)

            strategy = CustomStrategy(
                id=strategy_data["id"],
                name=strategy_data["name"],
                description=strategy_data["description"],
                language=strategy_data.get("language"),
                rules=rules,
                global_settings=strategy_data.get("global_settings", {}),
                created_by=strategy_data.get("created_by", "imported"),
                version=strategy_data.get("version", "1.0"),
                enabled=strategy_data.get("enabled", True)
            )

            return self.register_strategy(strategy)

        except Exception as e:
            self.logger.error("Failed to import strategy", error=str(e))
            return False


# Global strategy manager instance
custom_strategy_manager = CustomStrategyManager()
