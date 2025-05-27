"""
Multi-language support for code chunking.

Provides language-specific parsing, analysis, and chunking strategies.
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass

import structlog

logger = structlog.get_logger(__name__)


class SupportedLanguage(Enum):
    """Supported programming languages."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CSHARP = "csharp"
    CPP = "cpp"
    GO = "go"
    RUST = "rust"
    PHP = "php"
    RUBY = "ruby"
    KOTLIN = "kotlin"
    SWIFT = "swift"
    SCALA = "scala"
    UNKNOWN = "unknown"


@dataclass
class LanguageConfig:
    """Configuration for language-specific processing."""
    name: str
    extensions: List[str]
    comment_patterns: Dict[str, str]
    function_patterns: List[str]
    class_patterns: List[str]
    import_patterns: List[str]
    block_delimiters: Dict[str, str]
    string_delimiters: List[str]
    optimal_chunk_size: Tuple[int, int]  # (min, max) lines
    complexity_keywords: List[str]


class LanguageDetector:
    """Detects programming language from file content and extension."""
    
    LANGUAGE_CONFIGS = {
        SupportedLanguage.PYTHON: LanguageConfig(
            name="Python",
            extensions=[".py", ".pyw", ".pyi"],
            comment_patterns={"single": "#", "multi_start": '"""', "multi_end": '"""'},
            function_patterns=[r"def\s+(\w+)\s*\(", r"async\s+def\s+(\w+)\s*\("],
            class_patterns=[r"class\s+(\w+)(?:\([^)]*\))?:"],
            import_patterns=[r"import\s+(\w+)", r"from\s+(\w+)\s+import"],
            block_delimiters={"start": ":", "end": "indent"},
            string_delimiters=["'", '"', '"""', "'''"],
            optimal_chunk_size=(20, 150),
            complexity_keywords=["if", "elif", "else", "for", "while", "try", "except", "with", "async", "await"]
        ),
        
        SupportedLanguage.JAVASCRIPT: LanguageConfig(
            name="JavaScript",
            extensions=[".js", ".jsx", ".mjs"],
            comment_patterns={"single": "//", "multi_start": "/*", "multi_end": "*/"},
            function_patterns=[r"function\s+(\w+)\s*\(", r"(\w+)\s*=\s*function", r"(\w+)\s*=\s*\([^)]*\)\s*=>", r"async\s+function\s+(\w+)"],
            class_patterns=[r"class\s+(\w+)(?:\s+extends\s+\w+)?"],
            import_patterns=[r"import\s+.*\s+from\s+['\"]([^'\"]+)['\"]", r"require\s*\(\s*['\"]([^'\"]+)['\"]"],
            block_delimiters={"start": "{", "end": "}"},
            string_delimiters=["'", '"', "`"],
            optimal_chunk_size=(15, 120),
            complexity_keywords=["if", "else", "for", "while", "switch", "case", "try", "catch", "async", "await"]
        ),
        
        SupportedLanguage.TYPESCRIPT: LanguageConfig(
            name="TypeScript",
            extensions=[".ts", ".tsx"],
            comment_patterns={"single": "//", "multi_start": "/*", "multi_end": "*/"},
            function_patterns=[r"function\s+(\w+)\s*\(", r"(\w+)\s*=\s*\([^)]*\)\s*:\s*\w+\s*=>", r"async\s+function\s+(\w+)"],
            class_patterns=[r"class\s+(\w+)(?:\s+extends\s+\w+)?(?:\s+implements\s+[\w,\s]+)?"],
            import_patterns=[r"import\s+.*\s+from\s+['\"]([^'\"]+)['\"]", r"import\s+['\"]([^'\"]+)['\"]"],
            block_delimiters={"start": "{", "end": "}"},
            string_delimiters=["'", '"', "`"],
            optimal_chunk_size=(15, 120),
            complexity_keywords=["if", "else", "for", "while", "switch", "case", "try", "catch", "async", "await", "interface", "type"]
        ),
        
        SupportedLanguage.JAVA: LanguageConfig(
            name="Java",
            extensions=[".java"],
            comment_patterns={"single": "//", "multi_start": "/*", "multi_end": "*/"},
            function_patterns=[r"(?:public|private|protected|static|\s)+[\w<>\[\]]+\s+(\w+)\s*\("],
            class_patterns=[r"(?:public|private|protected)?\s*(?:abstract|final)?\s*class\s+(\w+)"],
            import_patterns=[r"import\s+(?:static\s+)?([^;]+);"],
            block_delimiters={"start": "{", "end": "}"},
            string_delimiters=['"'],
            optimal_chunk_size=(20, 200),
            complexity_keywords=["if", "else", "for", "while", "switch", "case", "try", "catch", "synchronized", "volatile"]
        ),
        
        SupportedLanguage.CSHARP: LanguageConfig(
            name="C#",
            extensions=[".cs"],
            comment_patterns={"single": "//", "multi_start": "/*", "multi_end": "*/"},
            function_patterns=[r"(?:public|private|protected|internal|static|\s)+[\w<>\[\]]+\s+(\w+)\s*\("],
            class_patterns=[r"(?:public|private|protected|internal)?\s*(?:abstract|sealed|static)?\s*class\s+(\w+)"],
            import_patterns=[r"using\s+([^;]+);"],
            block_delimiters={"start": "{", "end": "}"},
            string_delimiters=['"', "@\""],
            optimal_chunk_size=(20, 180),
            complexity_keywords=["if", "else", "for", "foreach", "while", "switch", "case", "try", "catch", "async", "await"]
        ),
        
        SupportedLanguage.GO: LanguageConfig(
            name="Go",
            extensions=[".go"],
            comment_patterns={"single": "//", "multi_start": "/*", "multi_end": "*/"},
            function_patterns=[r"func\s+(?:\([^)]*\)\s+)?(\w+)\s*\("],
            class_patterns=[r"type\s+(\w+)\s+struct"],
            import_patterns=[r"import\s+[\"']([^\"']+)[\"']", r"import\s+(\w+)\s+[\"']"],
            block_delimiters={"start": "{", "end": "}"},
            string_delimiters=['"', "`"],
            optimal_chunk_size=(15, 100),
            complexity_keywords=["if", "else", "for", "range", "switch", "case", "select", "go", "defer"]
        ),
        
        SupportedLanguage.RUST: LanguageConfig(
            name="Rust",
            extensions=[".rs"],
            comment_patterns={"single": "//", "multi_start": "/*", "multi_end": "*/"},
            function_patterns=[r"fn\s+(\w+)\s*\(", r"async\s+fn\s+(\w+)\s*\("],
            class_patterns=[r"struct\s+(\w+)", r"enum\s+(\w+)", r"trait\s+(\w+)"],
            import_patterns=[r"use\s+([^;]+);"],
            block_delimiters={"start": "{", "end": "}"},
            string_delimiters=['"', "r\"", "r#\""],
            optimal_chunk_size=(15, 120),
            complexity_keywords=["if", "else", "for", "while", "loop", "match", "async", "await", "unsafe"]
        )
    }
    
    @classmethod
    def detect_language(cls, file_path: str, content: str = "") -> SupportedLanguage:
        """
        Detect programming language from file path and content.
        
        Args:
            file_path: Path to the file
            content: File content (optional)
            
        Returns:
            Detected language
        """
        # First try by file extension
        for language, config in cls.LANGUAGE_CONFIGS.items():
            for ext in config.extensions:
                if file_path.lower().endswith(ext):
                    return language
        
        # If no extension match, try content-based detection
        if content:
            return cls._detect_by_content(content)
        
        return SupportedLanguage.UNKNOWN
    
    @classmethod
    def _detect_by_content(cls, content: str) -> SupportedLanguage:
        """Detect language by analyzing content patterns."""
        content_lower = content.lower()
        
        # Python indicators
        if any(pattern in content for pattern in ["def ", "import ", "from ", "class ", "if __name__"]):
            return SupportedLanguage.PYTHON
        
        # JavaScript/TypeScript indicators
        if any(pattern in content for pattern in ["function ", "const ", "let ", "var ", "=>"]):
            if "interface " in content or ": " in content:
                return SupportedLanguage.TYPESCRIPT
            return SupportedLanguage.JAVASCRIPT
        
        # Java indicators
        if any(pattern in content for pattern in ["public class", "private ", "import java"]):
            return SupportedLanguage.JAVA
        
        # C# indicators
        if any(pattern in content for pattern in ["using System", "namespace ", "public class"]):
            return SupportedLanguage.CSHARP
        
        # Go indicators
        if any(pattern in content for pattern in ["package ", "func ", "import ("]):
            return SupportedLanguage.GO
        
        # Rust indicators
        if any(pattern in content for pattern in ["fn ", "use ", "struct ", "impl "]):
            return SupportedLanguage.RUST
        
        return SupportedLanguage.UNKNOWN
    
    @classmethod
    def get_language_config(cls, language: SupportedLanguage) -> LanguageConfig:
        """Get configuration for a specific language."""
        return cls.LANGUAGE_CONFIGS.get(language, cls.LANGUAGE_CONFIGS[SupportedLanguage.PYTHON])


class LanguageSpecificChunker:
    """Language-specific chunking logic."""
    
    def __init__(self, language: SupportedLanguage):
        self.language = language
        self.config = LanguageDetector.get_language_config(language)
        self.logger = structlog.get_logger(f"chunker.{language.value}")
    
    def find_functions(self, content: str) -> List[Dict[str, Any]]:
        """Find all functions in the content."""
        functions = []
        lines = content.split('\n')
        
        for pattern in self.config.function_patterns:
            for i, line in enumerate(lines):
                match = re.search(pattern, line)
                if match:
                    function_name = match.group(1)
                    start_line = i + 1
                    end_line = self._find_block_end(lines, i)
                    
                    functions.append({
                        "name": function_name,
                        "start_line": start_line,
                        "end_line": end_line,
                        "type": "function",
                        "complexity": self._estimate_complexity(lines[i:end_line])
                    })
        
        return functions
    
    def find_classes(self, content: str) -> List[Dict[str, Any]]:
        """Find all classes in the content."""
        classes = []
        lines = content.split('\n')
        
        for pattern in self.config.class_patterns:
            for i, line in enumerate(lines):
                match = re.search(pattern, line)
                if match:
                    class_name = match.group(1)
                    start_line = i + 1
                    end_line = self._find_block_end(lines, i)
                    
                    classes.append({
                        "name": class_name,
                        "start_line": start_line,
                        "end_line": end_line,
                        "type": "class",
                        "complexity": self._estimate_complexity(lines[i:end_line])
                    })
        
        return classes
    
    def find_imports(self, content: str) -> List[Dict[str, Any]]:
        """Find all import statements."""
        imports = []
        lines = content.split('\n')
        
        for pattern in self.config.import_patterns:
            for i, line in enumerate(lines):
                match = re.search(pattern, line)
                if match:
                    import_name = match.group(1)
                    imports.append({
                        "name": import_name,
                        "start_line": i + 1,
                        "end_line": i + 1,
                        "type": "import",
                        "complexity": 1
                    })
        
        return imports
    
    def _find_block_end(self, lines: List[str], start_idx: int) -> int:
        """Find the end of a code block."""
        if self.config.block_delimiters["end"] == "indent":
            return self._find_python_block_end(lines, start_idx)
        else:
            return self._find_brace_block_end(lines, start_idx)
    
    def _find_python_block_end(self, lines: List[str], start_idx: int) -> int:
        """Find end of Python-style indented block."""
        if start_idx >= len(lines):
            return len(lines)
        
        start_indent = len(lines[start_idx]) - len(lines[start_idx].lstrip())
        
        for i in range(start_idx + 1, len(lines)):
            line = lines[i].strip()
            if not line:  # Skip empty lines
                continue
            
            current_indent = len(lines[i]) - len(lines[i].lstrip())
            if current_indent <= start_indent:
                return i
        
        return len(lines)
    
    def _find_brace_block_end(self, lines: List[str], start_idx: int) -> int:
        """Find end of brace-delimited block."""
        brace_count = 0
        start_char = self.config.block_delimiters["start"]
        end_char = self.config.block_delimiters["end"]
        
        for i in range(start_idx, len(lines)):
            line = lines[i]
            brace_count += line.count(start_char) - line.count(end_char)
            
            if i > start_idx and brace_count == 0:
                return i + 1
        
        return len(lines)
    
    def _estimate_complexity(self, lines: List[str]) -> int:
        """Estimate code complexity based on keywords."""
        complexity = 1
        content = '\n'.join(lines).lower()
        
        for keyword in self.config.complexity_keywords:
            complexity += content.count(keyword)
        
        return min(complexity, 10)  # Cap at 10
    
    def get_optimal_chunk_boundaries(self, content: str) -> List[Dict[str, Any]]:
        """Get optimal chunk boundaries for the language."""
        boundaries = []
        
        # Find all code elements
        functions = self.find_functions(content)
        classes = self.find_classes(content)
        imports = self.find_imports(content)
        
        # Combine and sort by line number
        all_elements = functions + classes + imports
        all_elements.sort(key=lambda x: x["start_line"])
        
        # Group imports together
        if imports:
            import_start = min(imp["start_line"] for imp in imports)
            import_end = max(imp["end_line"] for imp in imports)
            boundaries.append({
                "start_line": import_start,
                "end_line": import_end,
                "type": "imports",
                "name": "imports",
                "complexity": 1
            })
        
        # Add functions and classes
        for element in all_elements:
            if element["type"] in ["function", "class"]:
                boundaries.append(element)
        
        return boundaries


# Utility functions for multi-language support

def get_supported_languages() -> List[str]:
    """Get list of supported language names."""
    return [lang.value for lang in SupportedLanguage if lang != SupportedLanguage.UNKNOWN]


def is_language_supported(language: str) -> bool:
    """Check if a language is supported."""
    try:
        SupportedLanguage(language.lower())
        return True
    except ValueError:
        return False


def get_language_info(language: str) -> Dict[str, Any]:
    """Get detailed information about a language."""
    try:
        lang_enum = SupportedLanguage(language.lower())
        config = LanguageDetector.get_language_config(lang_enum)
        
        return {
            "name": config.name,
            "extensions": config.extensions,
            "optimal_chunk_size": config.optimal_chunk_size,
            "supported_features": {
                "functions": bool(config.function_patterns),
                "classes": bool(config.class_patterns),
                "imports": bool(config.import_patterns),
                "comments": bool(config.comment_patterns)
            }
        }
    except ValueError:
        return {"error": f"Language '{language}' not supported"}


def detect_and_chunk(file_path: str, content: str) -> Dict[str, Any]:
    """Detect language and perform language-specific chunking."""
    language = LanguageDetector.detect_language(file_path, content)
    
    if language == SupportedLanguage.UNKNOWN:
        return {
            "language": "unknown",
            "boundaries": [],
            "error": "Language not supported or could not be detected"
        }
    
    chunker = LanguageSpecificChunker(language)
    boundaries = chunker.get_optimal_chunk_boundaries(content)
    
    return {
        "language": language.value,
        "boundaries": boundaries,
        "config": {
            "optimal_chunk_size": chunker.config.optimal_chunk_size,
            "complexity_keywords": chunker.config.complexity_keywords
        }
    }
