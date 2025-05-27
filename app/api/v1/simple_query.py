
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict, Any
import re

router = APIRouter()

class SimpleQuery(BaseModel):
    query: str
    collection_id: str = None
    top_k: int = 5
    include_metadata: bool = True

class QueryResult(BaseModel):
    content: str
    score: float
    metadata: Dict[str, Any] = {}

class QueryResponse(BaseModel):
    results: List[QueryResult]
    query: str
    total_results: int

@router.post("/simple-search", response_model=QueryResponse)
async def simple_search(query: SimpleQuery):
    """Simple search without vector database."""
    from .simple_collections import documents
    
    results = []
    query_lower = query.query.lower()
    
    # Simple text matching
    for doc in documents.values():
        if query.collection_id and doc["collection_id"] != query.collection_id:
            continue
            
        content = doc["content"].lower()
        
        # Calculate simple score based on keyword matches
        score = 0.0
        query_words = query_lower.split()
        
        for word in query_words:
            if word in content:
                score += content.count(word) * 0.1
        
        if score > 0:
            # Normalize score
            score = min(score / len(query_words), 1.0)
            
            # Get relevant excerpt
            content_words = doc["content"].split()
            excerpt_start = 0
            excerpt_length = 200
            
            # Try to find excerpt containing query words
            for i, word in enumerate(content_words):
                if any(qw in word.lower() for qw in query_words):
                    excerpt_start = max(0, i - 50)
                    break
            
            excerpt = " ".join(content_words[excerpt_start:excerpt_start + excerpt_length])
            
            results.append(QueryResult(
                content=excerpt,
                score=score,
                metadata={
                    "document_id": doc["id"],
                    "title": doc["title"],
                    "collection_id": doc["collection_id"]
                }
            ))
    
    # Sort by score and limit results
    results.sort(key=lambda x: x.score, reverse=True)
    results = results[:query.top_k]
    
    return QueryResponse(
        results=results,
        query=query.query,
        total_results=len(results)
    )
