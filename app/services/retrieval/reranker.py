"""
Re-ranking implementation for improving retrieval result quality.
"""

import asyncio
import math
from typing import List, Dict, Any, Optional
import logging
import re

from .base import RetrievalResult, QueryType, ReRankingError

logger = logging.getLogger(__name__)

class ReRanker:
    """Re-ranker for improving retrieval result quality and relevance."""
    
    def __init__(self):
        self._initialized = False
        self._feature_weights = {
            'semantic_similarity': 0.4,
            'keyword_match': 0.2,
            'content_quality': 0.15,
            'freshness': 0.1,
            'source_authority': 0.1,
            'user_preference': 0.05
        }
    
    async def initialize(self) -> None:
        """Initialize the re-ranker."""
        self._initialized = True
        logger.info("Re-ranker initialized")
    
    async def rerank_results(
        self,
        results: List[RetrievalResult],
        query: str,
        query_type: QueryType,
        top_k: Optional[int] = None
    ) -> List[RetrievalResult]:
        """
        Re-rank retrieval results for better relevance.
        
        Args:
            results: List of retrieval results
            query: Original query
            query_type: Type of query
            top_k: Number of top results to return
        
        Returns:
            Re-ranked list of results
        """
        if not self._initialized:
            await self.initialize()
        
        if not results:
            return results
        
        try:
            # Calculate re-ranking scores
            reranked_results = []
            
            for result in results:
                # Calculate multiple relevance features
                features = await self._extract_features(result, query, query_type)
                
                # Combine features into final score
                rerank_score = self._combine_features(features)
                
                # Create new result with re-ranking score
                reranked_result = RetrievalResult(
                    document_id=result.document_id,
                    chunk_id=result.chunk_id,
                    content=result.content,
                    score=rerank_score,  # Use re-ranking score as main score
                    vector_score=result.vector_score,
                    keyword_score=result.keyword_score,
                    rerank_score=rerank_score,
                    metadata=result.metadata,
                    source_file=result.source_file,
                    chunk_index=result.chunk_index,
                    language=result.language,
                    document_type=result.document_type,
                    highlighted_content=result.highlighted_content,
                    matched_keywords=result.matched_keywords
                )
                
                reranked_results.append(reranked_result)
            
            # Sort by re-ranking score
            reranked_results.sort(key=lambda x: x.rerank_score, reverse=True)
            
            # Apply top-k filtering
            if top_k:
                reranked_results = reranked_results[:top_k]
            
            logger.debug(f"Re-ranked {len(results)} results, returning top {len(reranked_results)}")
            return reranked_results
            
        except Exception as e:
            raise ReRankingError(f"Failed to re-rank results: {e}", "feature_based", e)
    
    async def _extract_features(
        self,
        result: RetrievalResult,
        query: str,
        query_type: QueryType
    ) -> Dict[str, float]:
        """Extract relevance features for a result."""
        features = {}
        
        # 1. Semantic similarity (from original vector score)
        features['semantic_similarity'] = result.vector_score or result.score
        
        # 2. Keyword match score
        features['keyword_match'] = self._calculate_keyword_match(result, query)
        
        # 3. Content quality score
        features['content_quality'] = self._assess_content_quality(result)
        
        # 4. Query type specific features
        if query_type == QueryType.CODE:
            features['code_relevance'] = self._assess_code_relevance(result, query)
        elif query_type == QueryType.QUESTION:
            features['answer_likelihood'] = self._assess_answer_likelihood(result, query)
        
        # 5. Freshness score (based on metadata if available)
        features['freshness'] = self._calculate_freshness(result)
        
        # 6. Source authority (based on file type, location, etc.)
        features['source_authority'] = self._assess_source_authority(result)
        
        # 7. Length appropriateness
        features['length_score'] = self._assess_length_appropriateness(result, query_type)
        
        return features
    
    def _calculate_keyword_match(self, result: RetrievalResult, query: str) -> float:
        """Calculate keyword match score."""
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        content_words = set(re.findall(r'\b\w+\b', result.content.lower()))
        
        if not query_words:
            return 0.0
        
        # Calculate exact matches
        exact_matches = len(query_words.intersection(content_words))
        exact_score = exact_matches / len(query_words)
        
        # Calculate partial matches (substring matches)
        partial_matches = 0
        for query_word in query_words:
            if any(query_word in content_word for content_word in content_words):
                partial_matches += 1
        
        partial_score = partial_matches / len(query_words) * 0.5
        
        # Combine scores
        return min(1.0, exact_score + partial_score)
    
    def _assess_content_quality(self, result: RetrievalResult) -> float:
        """Assess the quality of content."""
        content = result.content
        
        # Length factor (not too short, not too long)
        length_score = self._calculate_length_score(len(content))
        
        # Structure factor (presence of proper formatting)
        structure_score = self._calculate_structure_score(content)
        
        # Completeness factor (not truncated)
        completeness_score = 1.0 if not content.endswith('...') else 0.8
        
        # Language quality (basic checks)
        language_score = self._assess_language_quality(content)
        
        return (length_score + structure_score + completeness_score + language_score) / 4
    
    def _assess_code_relevance(self, result: RetrievalResult, query: str) -> float:
        """Assess relevance for code queries."""
        content = result.content
        
        # Check for code patterns
        code_patterns = [
            r'def\s+\w+',  # Function definitions
            r'class\s+\w+',  # Class definitions
            r'import\s+\w+',  # Imports
            r'\w+\.\w+\(',  # Method calls
            r'if\s+\w+',  # Conditionals
            r'for\s+\w+',  # Loops
            r'return\s+',  # Returns
        ]
        
        code_score = 0.0
        for pattern in code_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                code_score += 0.1
        
        # Check for query-specific code elements
        query_lower = query.lower()
        if 'function' in query_lower and re.search(r'def\s+\w+', content):
            code_score += 0.3
        if 'class' in query_lower and re.search(r'class\s+\w+', content):
            code_score += 0.3
        if 'import' in query_lower and re.search(r'import\s+\w+', content):
            code_score += 0.2
        
        return min(1.0, code_score)
    
    def _assess_answer_likelihood(self, result: RetrievalResult, query: str) -> float:
        """Assess likelihood that content answers the question."""
        content = result.content.lower()
        query_lower = query.lower()
        
        # Check for answer indicators
        answer_indicators = [
            'because', 'since', 'due to', 'as a result',
            'therefore', 'thus', 'hence', 'consequently',
            'the reason', 'this is', 'it works', 'you can',
            'to do this', 'the solution', 'the answer'
        ]
        
        indicator_score = 0.0
        for indicator in answer_indicators:
            if indicator in content:
                indicator_score += 0.1
        
        # Check for question word correspondence
        question_words = ['what', 'how', 'why', 'when', 'where', 'who']
        correspondence_score = 0.0
        
        for word in question_words:
            if word in query_lower:
                if word == 'how' and ('step' in content or 'method' in content):
                    correspondence_score += 0.2
                elif word == 'what' and ('definition' in content or 'means' in content):
                    correspondence_score += 0.2
                elif word == 'why' and ('reason' in content or 'because' in content):
                    correspondence_score += 0.2
        
        return min(1.0, indicator_score + correspondence_score)
    
    def _calculate_freshness(self, result: RetrievalResult) -> float:
        """Calculate freshness score based on metadata."""
        # Default freshness if no timestamp available
        if 'indexed_at' not in result.metadata and 'created_at' not in result.metadata:
            return 0.5
        
        # In a real implementation, you would calculate based on actual timestamps
        # For now, return a default score
        return 0.7
    
    def _assess_source_authority(self, result: RetrievalResult) -> float:
        """Assess source authority based on file type and location."""
        authority_score = 0.5  # Default
        
        # Check file type authority
        if result.source_file:
            file_lower = result.source_file.lower()
            
            # Documentation files have higher authority
            if any(doc_indicator in file_lower for doc_indicator in ['readme', 'doc', 'guide', 'tutorial']):
                authority_score += 0.3
            
            # Test files have lower authority for general queries
            if any(test_indicator in file_lower for test_indicator in ['test', 'spec', 'mock']):
                authority_score -= 0.2
            
            # Main/core files have higher authority
            if any(main_indicator in file_lower for main_indicator in ['main', 'core', 'index', 'app']):
                authority_score += 0.2
        
        # Check document type
        if result.document_type:
            if result.document_type in ['markdown', 'documentation']:
                authority_score += 0.2
            elif result.document_type == 'code':
                authority_score += 0.1
        
        return min(1.0, max(0.0, authority_score))
    
    def _assess_length_appropriateness(self, result: RetrievalResult, query_type: QueryType) -> float:
        """Assess if content length is appropriate for query type."""
        content_length = len(result.content)
        
        # Optimal length ranges for different query types
        optimal_ranges = {
            QueryType.SEMANTIC: (100, 1000),
            QueryType.CODE: (50, 500),
            QueryType.QUESTION: (100, 800),
            QueryType.SUMMARY: (200, 1500),
            QueryType.KEYWORD: (50, 800)
        }
        
        min_length, max_length = optimal_ranges.get(query_type, (100, 1000))
        
        if content_length < min_length:
            return content_length / min_length
        elif content_length > max_length:
            return max_length / content_length
        else:
            return 1.0
    
    def _calculate_length_score(self, length: int) -> float:
        """Calculate score based on content length."""
        # Optimal length around 200-800 characters
        if 200 <= length <= 800:
            return 1.0
        elif length < 200:
            return length / 200
        else:
            return 800 / length
    
    def _calculate_structure_score(self, content: str) -> float:
        """Calculate score based on content structure."""
        structure_score = 0.5  # Base score
        
        # Check for proper formatting
        if '\n' in content:  # Multi-line content
            structure_score += 0.2
        
        # Check for code structure
        if any(pattern in content for pattern in ['def ', 'class ', 'function ', 'import ']):
            structure_score += 0.2
        
        # Check for documentation structure
        if any(pattern in content for pattern in ['# ', '## ', '### ', '* ', '- ']):
            structure_score += 0.1
        
        return min(1.0, structure_score)
    
    def _assess_language_quality(self, content: str) -> float:
        """Assess basic language quality."""
        # Simple heuristics for language quality
        quality_score = 0.5
        
        # Check for proper capitalization
        sentences = content.split('. ')
        if sentences:
            capitalized = sum(1 for s in sentences if s and s[0].isupper())
            quality_score += (capitalized / len(sentences)) * 0.3
        
        # Check for reasonable word length
        words = content.split()
        if words:
            avg_word_length = sum(len(word) for word in words) / len(words)
            if 3 <= avg_word_length <= 8:  # Reasonable average
                quality_score += 0.2
        
        return min(1.0, quality_score)
    
    def _combine_features(self, features: Dict[str, float]) -> float:
        """Combine multiple features into a single score."""
        total_score = 0.0
        total_weight = 0.0
        
        for feature_name, feature_value in features.items():
            weight = self._feature_weights.get(feature_name, 0.1)
            total_score += feature_value * weight
            total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            return total_score / total_weight
        else:
            return 0.0

# Global re-ranker instance
reranker = ReRanker()
