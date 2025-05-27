"""
Query processing for intelligent query understanding and expansion.
"""

import asyncio
import re
from typing import List, Dict, Any, Optional, Set
import logging

from .base import QueryType, QueryProcessingError

logger = logging.getLogger(__name__)

class QueryProcessor:
    """Processor for query understanding, expansion, and optimization."""
    
    def __init__(self):
        self._initialized = False
        self._stop_words = self._load_stop_words()
        self._code_keywords = self._load_code_keywords()
        self._query_patterns = self._load_query_patterns()
    
    async def initialize(self) -> None:
        """Initialize the query processor."""
        self._initialized = True
        logger.info("Query processor initialized")
    
    async def process_query(
        self,
        query: str,
        query_type: QueryType,
        context: Optional[str] = None,
        expand_query: bool = True
    ) -> Dict[str, Any]:
        """
        Process and enhance a query for better retrieval.
        
        Args:
            query: Original query
            query_type: Type of query
            context: Additional context
            expand_query: Whether to expand the query
        
        Returns:
            Processed query information
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Clean and normalize query
            cleaned_query = self._clean_query(query)
            
            # Detect query intent and type
            detected_type = self._detect_query_type(cleaned_query)
            final_type = query_type if query_type != QueryType.SEMANTIC else detected_type
            
            # Extract keywords and entities
            keywords = self._extract_keywords(cleaned_query)
            entities = self._extract_entities(cleaned_query)
            
            # Generate query variations
            variations = []
            if expand_query:
                variations = await self._expand_query(cleaned_query, final_type)
            
            # Create search terms for different strategies
            vector_query = self._prepare_vector_query(cleaned_query, context)
            keyword_query = self._prepare_keyword_query(cleaned_query, keywords)
            
            return {
                "original_query": query,
                "cleaned_query": cleaned_query,
                "processed_query": vector_query,
                "keyword_query": keyword_query,
                "query_type": final_type,
                "keywords": keywords,
                "entities": entities,
                "variations": variations,
                "intent": self._analyze_intent(cleaned_query, final_type),
                "complexity": self._assess_complexity(cleaned_query),
                "language": self._detect_language(cleaned_query)
            }
            
        except Exception as e:
            raise QueryProcessingError(f"Failed to process query: {e}", query, e)
    
    def _clean_query(self, query: str) -> str:
        """Clean and normalize the query."""
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', query.strip())
        
        # Handle special characters for code queries
        if self._looks_like_code(cleaned):
            # Preserve code structure
            return cleaned
        
        # Remove special characters for text queries
        cleaned = re.sub(r'[^\w\s\-\.]', ' ', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned
    
    def _detect_query_type(self, query: str) -> QueryType:
        """Detect the type of query based on content and patterns."""
        query_lower = query.lower()
        
        # Check for question patterns
        question_patterns = [
            r'^(what|how|why|when|where|who|which|can|could|would|should|is|are|does|do|did)',
            r'\?$'
        ]
        
        if any(re.search(pattern, query_lower) for pattern in question_patterns):
            return QueryType.QUESTION
        
        # Check for code patterns
        if self._looks_like_code(query):
            return QueryType.CODE
        
        # Check for summary requests
        summary_keywords = ['summary', 'summarize', 'overview', 'explain', 'describe']
        if any(keyword in query_lower for keyword in summary_keywords):
            return QueryType.SUMMARY
        
        # Default to semantic search
        return QueryType.SEMANTIC
    
    def _looks_like_code(self, query: str) -> bool:
        """Check if query looks like code."""
        code_indicators = [
            r'def\s+\w+\s*\(',  # Python function
            r'function\s+\w+\s*\(',  # JavaScript function
            r'class\s+\w+',  # Class definition
            r'\w+\.\w+\(',  # Method call
            r'import\s+\w+',  # Import statement
            r'from\s+\w+\s+import',  # From import
            r'#include\s*<',  # C/C++ include
            r'public\s+class',  # Java class
            r'private\s+\w+',  # Private member
            r'async\s+def',  # Async function
            r'await\s+\w+',  # Await call
        ]
        
        return any(re.search(pattern, query, re.IGNORECASE) for pattern in code_indicators)
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from the query."""
        # Split into words
        words = re.findall(r'\b\w+\b', query.lower())
        
        # Remove stop words
        keywords = [word for word in words if word not in self._stop_words]
        
        # Add code-specific keywords if detected
        if self._looks_like_code(query):
            code_words = [word for word in words if word in self._code_keywords]
            keywords.extend(code_words)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for keyword in keywords:
            if keyword not in seen:
                seen.add(keyword)
                unique_keywords.append(keyword)
        
        return unique_keywords
    
    def _extract_entities(self, query: str) -> List[Dict[str, str]]:
        """Extract named entities from the query."""
        entities = []
        
        # Simple entity extraction patterns
        patterns = {
            'file_extension': r'\.\w{2,4}\b',
            'function_name': r'\b\w+\(\)',
            'class_name': r'\bclass\s+(\w+)',
            'variable_name': r'\b[a-z_]\w*\b',
            'constant': r'\b[A-Z_][A-Z0-9_]*\b',
            'number': r'\b\d+\.?\d*\b',
            'url': r'https?://[^\s]+',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        }
        
        for entity_type, pattern in patterns.items():
            matches = re.findall(pattern, query, re.IGNORECASE)
            for match in matches:
                entities.append({
                    'type': entity_type,
                    'value': match if isinstance(match, str) else match[0],
                    'original': match
                })
        
        return entities
    
    async def _expand_query(self, query: str, query_type: QueryType) -> List[str]:
        """Expand query with synonyms and related terms."""
        variations = []
        
        # Add basic variations
        words = query.split()
        
        # Add partial queries (useful for code search)
        if len(words) > 1:
            for i in range(len(words)):
                partial = ' '.join(words[:i+1])
                if partial != query and len(partial) > 3:
                    variations.append(partial)
        
        # Add keyword combinations
        if len(words) > 2:
            # Try different word combinations
            for i in range(len(words)):
                for j in range(i+2, len(words)+1):
                    combination = ' '.join(words[i:j])
                    if combination != query and len(combination) > 3:
                        variations.append(combination)
        
        # Add query type specific expansions
        if query_type == QueryType.CODE:
            variations.extend(self._expand_code_query(query))
        elif query_type == QueryType.QUESTION:
            variations.extend(self._expand_question_query(query))
        
        # Remove duplicates and return top variations
        unique_variations = list(dict.fromkeys(variations))
        return unique_variations[:5]  # Limit to top 5 variations
    
    def _expand_code_query(self, query: str) -> List[str]:
        """Expand code-specific queries."""
        variations = []
        
        # Add common code synonyms
        code_synonyms = {
            'function': ['method', 'def', 'func'],
            'class': ['object', 'type'],
            'variable': ['var', 'field', 'attribute'],
            'import': ['include', 'require', 'using'],
            'return': ['returns', 'output'],
            'parameter': ['param', 'argument', 'arg'],
            'error': ['exception', 'bug', 'issue'],
            'test': ['testing', 'unittest', 'spec']
        }
        
        query_lower = query.lower()
        for original, synonyms in code_synonyms.items():
            if original in query_lower:
                for synonym in synonyms:
                    variation = query_lower.replace(original, synonym)
                    if variation != query_lower:
                        variations.append(variation)
        
        return variations
    
    def _expand_question_query(self, query: str) -> List[str]:
        """Expand question queries."""
        variations = []
        
        # Convert questions to statements
        query_lower = query.lower()
        
        # Remove question words and punctuation
        statement = re.sub(r'^(what|how|why|when|where|who|which|can|could|would|should|is|are|does|do|did)\s+', '', query_lower)
        statement = re.sub(r'\?+$', '', statement).strip()
        
        if statement and statement != query_lower:
            variations.append(statement)
        
        # Add imperative forms
        if query_lower.startswith('how to'):
            imperative = query_lower.replace('how to', '').strip()
            if imperative:
                variations.append(imperative)
        
        return variations
    
    def _prepare_vector_query(self, query: str, context: Optional[str] = None) -> str:
        """Prepare query for vector search."""
        # Combine query with context if available
        if context:
            return f"{context} {query}"
        return query
    
    def _prepare_keyword_query(self, query: str, keywords: List[str]) -> str:
        """Prepare query for keyword search."""
        # Use extracted keywords for better keyword search
        if keywords:
            return ' '.join(keywords)
        return query
    
    def _analyze_intent(self, query: str, query_type: QueryType) -> Dict[str, Any]:
        """Analyze query intent."""
        intent = {
            'type': query_type.value,
            'specificity': 'general',
            'scope': 'broad',
            'action': 'search'
        }
        
        query_lower = query.lower()
        
        # Determine specificity
        if len(query.split()) > 5 or any(char in query for char in ['(', ')', '.', '->']):
            intent['specificity'] = 'specific'
        elif len(query.split()) <= 2:
            intent['specificity'] = 'broad'
        
        # Determine action intent
        action_keywords = {
            'find': ['find', 'search', 'locate', 'get'],
            'explain': ['explain', 'describe', 'what is', 'how does'],
            'compare': ['compare', 'difference', 'vs', 'versus'],
            'implement': ['implement', 'create', 'build', 'make'],
            'debug': ['debug', 'fix', 'error', 'problem', 'issue'],
            'optimize': ['optimize', 'improve', 'performance', 'faster']
        }
        
        for action, keywords in action_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                intent['action'] = action
                break
        
        return intent
    
    def _assess_complexity(self, query: str) -> str:
        """Assess query complexity."""
        word_count = len(query.split())
        
        if word_count <= 2:
            return 'simple'
        elif word_count <= 5:
            return 'medium'
        else:
            return 'complex'
    
    def _detect_language(self, query: str) -> str:
        """Detect query language (programming language for code queries)."""
        if not self._looks_like_code(query):
            return 'natural'
        
        # Simple language detection based on keywords
        language_patterns = {
            'python': [r'\bdef\b', r'\bimport\b', r'\bfrom\b', r'\bclass\b', r'\bself\b'],
            'javascript': [r'\bfunction\b', r'\bvar\b', r'\blet\b', r'\bconst\b', r'\bconsole\.log\b'],
            'java': [r'\bpublic\b', r'\bprivate\b', r'\bclass\b', r'\bstatic\b', r'\bvoid\b'],
            'cpp': [r'\b#include\b', r'\bstd::\b', r'\bnamespace\b', r'\btemplate\b'],
            'go': [r'\bfunc\b', r'\bpackage\b', r'\bimport\b', r'\bvar\b', r'\btype\b'],
            'rust': [r'\bfn\b', r'\blet\b', r'\bmut\b', r'\buse\b', r'\bstruct\b']
        }
        
        for language, patterns in language_patterns.items():
            if any(re.search(pattern, query, re.IGNORECASE) for pattern in patterns):
                return language
        
        return 'unknown'
    
    def _load_stop_words(self) -> Set[str]:
        """Load common stop words."""
        return {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have',
            'had', 'what', 'said', 'each', 'which', 'their', 'time', 'if'
        }
    
    def _load_code_keywords(self) -> Set[str]:
        """Load programming-related keywords."""
        return {
            'function', 'method', 'class', 'object', 'variable', 'parameter',
            'return', 'import', 'export', 'module', 'package', 'library',
            'api', 'interface', 'abstract', 'static', 'public', 'private',
            'protected', 'async', 'await', 'promise', 'callback', 'event',
            'error', 'exception', 'try', 'catch', 'finally', 'throw',
            'test', 'unittest', 'mock', 'stub', 'debug', 'log', 'trace'
        }
    
    def _load_query_patterns(self) -> Dict[str, List[str]]:
        """Load common query patterns."""
        return {
            'how_to': [
                r'how\s+to\s+\w+',
                r'how\s+do\s+i\s+\w+',
                r'how\s+can\s+i\s+\w+'
            ],
            'what_is': [
                r'what\s+is\s+\w+',
                r'what\s+are\s+\w+',
                r'what\s+does\s+\w+'
            ],
            'find_code': [
                r'find\s+\w+\s+function',
                r'search\s+for\s+\w+',
                r'locate\s+\w+\s+method'
            ]
        }

# Global processor instance
query_processor = QueryProcessor()
