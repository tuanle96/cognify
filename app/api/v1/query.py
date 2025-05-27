"""
Query and retrieval API endpoints.

Handles RAG queries, search operations, and result processing.
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import uuid

# Simple logger replacement
class SimpleLogger:
    def info(self, msg, **kwargs):
        print(f"INFO: {msg} {kwargs}")
    def debug(self, msg, **kwargs):
        print(f"DEBUG: {msg} {kwargs}")
    def warning(self, msg, **kwargs):
        print(f"WARNING: {msg} {kwargs}")
    def error(self, msg, **kwargs):
        print(f"ERROR: {msg} {kwargs}")

logger = SimpleLogger()

from app.api.dependencies import (
    get_retrieval_service,
    get_current_verified_user_from_db,
    get_query_repository,
    get_collection_repository,
    get_current_user,
    get_db_session
)
from app.api.models.query import (
    QueryRequest,
    QueryResponse,
    SearchRequest,
    SearchResponse,
    QuerySuggestionResponse,
    QueryHistoryResponse,
    QueryHistoryItem
)
from app.services.retrieval.service import RetrievalService
from app.services.retrieval.base import RetrievalRequest, RetrievalConfig, QueryType
from app.services.database.repositories import QueryRepository, CollectionRepository
from app.models.users import User
from app.models.queries import QueryStatus, QueryType as DBQueryType
from app.core.exceptions import ValidationError, NotFoundError

# Use the logger defined above
router = APIRouter()


@router.post("/", response_model=QueryResponse)
async def submit_query(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
    current_user: User = Depends(get_current_verified_user_from_db),
    query_repo: QueryRepository = Depends(get_query_repository),
    collection_repo: CollectionRepository = Depends(get_collection_repository)
):
    """
    Submit a RAG query for processing.

    Processes natural language queries and returns relevant document chunks
    with intelligent ranking and context.
    """
    try:
        # Verify collection exists and user has access
        collection = await collection_repo.get_by_name_and_owner(
            request.collection_name, current_user.id
        )
        if not collection:
            # Check if user has read access to the collection
            collections = await collection_repo.get_user_collections(current_user.id)
            collection = next((c for c in collections if c.name == request.collection_name), None)

            if not collection:
                raise HTTPException(status_code=404, detail="Collection not found")

            # Check read permission
            has_read_permission = await collection_repo.check_permission(
                collection.id, current_user.id, "read"
            )
            if not has_read_permission:
                raise HTTPException(status_code=403, detail="Access denied")

        # Create query in database
        query = await query_repo.create_query(
            query_text=request.query,
            user_id=current_user.id,
            collection_id=collection.id,
            query_type=_map_query_type(request.query_type),
            max_results=request.max_results,
            min_score=request.min_score,
            rerank_results=request.rerank_results,
            include_metadata=request.include_metadata
        )

        # Create retrieval request
        retrieval_config = RetrievalConfig(
            collection_name=request.collection_name,
            max_results=request.max_results,
            min_score=request.min_score,
            enable_reranking=request.rerank_results,
            metadata_filters={"include_metadata": request.include_metadata} if request.include_metadata else None
        )

        # Map query type string to enum
        from app.services.retrieval.base import QueryType as RetrievalQueryType
        query_type_mapping = {
            "semantic": RetrievalQueryType.SEMANTIC,
            "keyword": RetrievalQueryType.KEYWORD,
            "hybrid": RetrievalQueryType.HYBRID,
            "question": RetrievalQueryType.QUESTION,
            "code": RetrievalQueryType.CODE,
            "summary": RetrievalQueryType.SUMMARY
        }

        retrieval_request = RetrievalRequest(
            query=request.query,
            query_type=query_type_mapping.get(request.query_type, RetrievalQueryType.SEMANTIC),
            config=retrieval_config,
            context={
                "user_id": current_user.id,
                "query_id": query.id,
                "timestamp": query.created_at.isoformat()
            }
        )

        # Process retrieval
        retrieval_response = await retrieval_service.retrieve(retrieval_request)

        if not retrieval_response.success:
            # Update query status to failed
            await query_repo.update_query_status(
                query.id,
                QueryStatus.FAILED,
                error_message=f"Query processing failed: {retrieval_response.errors}"
            )
            raise HTTPException(
                status_code=500,
                detail=f"Query processing failed: {retrieval_response.errors}"
            )

        # Update query with results in background
        background_tasks.add_task(
            _update_query_with_results,
            query_repo,
            query.id,
            retrieval_response
        )

        # Format response
        response = QueryResponse(
            query_id=query.id,
            query=request.query,
            collection_name=request.collection_name,
            results=retrieval_response.results,
            result_count=retrieval_response.result_count,
            processing_time=retrieval_response.processing_time,
            query_type=retrieval_response.query_type.value,
            metadata={
                "reranked": request.rerank_results,
                "min_score_applied": request.min_score,
                "total_candidates": len(retrieval_response.results),
                "query_expansion": {},
                "search_strategy": "hybrid",
                "vector_search_time": retrieval_response.vector_search_time,
                "keyword_search_time": retrieval_response.keyword_search_time,
                "rerank_time": retrieval_response.rerank_time
            },
            timestamp=query.created_at
        )

        logger.info(
            "Query processed successfully",
            query_id=query.id,
            user_id=current_user.id,
            collection=request.collection_name,
            result_count=retrieval_response.result_count,
            processing_time=retrieval_response.processing_time
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Query processing failed", error=str(e), query=request.query)
        raise HTTPException(status_code=500, detail="Query processing failed")


@router.get("/{query_id}", response_model=QueryResponse)
async def get_query_results(
    query_id: str,
    current_user: User = Depends(get_current_verified_user_from_db),
    query_repo: QueryRepository = Depends(get_query_repository)
):
    """
    Get results for a previously submitted query.
    """
    try:
        # Get query from database
        query = await query_repo.get_by_id(query_id)
        if not query:
            raise HTTPException(status_code=404, detail="Query not found")

        # Check if user owns the query
        if query.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")

        # Convert to response format
        response = QueryResponse(
            query_id=query.id,
            query=query.query_text,
            collection_name=query.collection.name if query.collection else "unknown",
            results=query.results or [],
            result_count=query.result_count or 0,
            processing_time=query.processing_time or 0.0,
            query_type=query.query_type.value,
            metadata=query.metadata or {},
            timestamp=query.created_at
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get query results", query_id=query_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get query results")


@router.post("/search", response_model=SearchResponse)
async def vector_search(
    request: SearchRequest,
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
    current_user: User = Depends(get_current_verified_user_from_db)
):
    """
    Perform direct vector similarity search.

    Lower-level search interface for advanced users who want direct
    vector search without query processing.
    """
    try:
        # Create retrieval request for vector search
        retrieval_config = RetrievalConfig(
            collection_name=request.collection_name,
            max_results=request.max_results or 10,
            min_score=request.min_score or 0.0
        )

        retrieval_request = RetrievalRequest(
            query=request.query,
            query_type=QueryType.SEMANTIC,  # Force semantic search
            config=retrieval_config
        )

        # Process search
        retrieval_response = await retrieval_service.retrieve(retrieval_request)

        if not retrieval_response.success:
            raise HTTPException(
                status_code=500,
                detail=f"Search failed: {retrieval_response.errors}"
            )

        response = SearchResponse(
            query=request.query,
            results=retrieval_response.results,
            total_results=retrieval_response.result_count,
            search_time=retrieval_response.processing_time,
            page=1,
            per_page=request.max_results or 10,
            facets={}
        )

        logger.info(
            "Vector search completed",
            user_id=current_user.id,
            collection=request.collection_name,
            result_count=retrieval_response.result_count
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Vector search failed", error=str(e), query=request.query)
        raise HTTPException(status_code=500, detail=f"Vector search failed: {str(e)}")


@router.get("/suggestions/{collection_name}", response_model=QuerySuggestionResponse)
async def get_query_suggestions(
    collection_name: str,
    query_prefix: str = "",
    limit: int = 10,
    current_user: dict = Depends(get_current_user),
    db_session = Depends(get_db_session)
):
    """
    Get query suggestions based on collection content and user history.
    """
    try:
        # Get suggestions from various sources
        suggestions = []

        # 1. Popular queries from this collection
        popular_queries = await _get_popular_queries(db_session, collection_name, limit // 3)
        suggestions.extend(popular_queries)

        # 2. User's recent queries
        user_queries = await _get_user_recent_queries(
            db_session, current_user["user_id"], collection_name, limit // 3
        )
        suggestions.extend(user_queries)

        # 3. Content-based suggestions
        content_suggestions = await _get_content_based_suggestions(
            collection_name, query_prefix, limit // 3
        )
        suggestions.extend(content_suggestions)

        # Filter by prefix if provided
        if query_prefix:
            suggestions = [s for s in suggestions if query_prefix.lower() in s.lower()]

        # Remove duplicates and limit
        unique_suggestions = list(dict.fromkeys(suggestions))[:limit]

        return QuerySuggestionResponse(
            collection_name=collection_name,
            query_prefix=query_prefix,
            suggestions=unique_suggestions,
            suggestion_count=len(unique_suggestions)
        )

    except Exception as e:
        logger.error("Failed to get query suggestions", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get suggestions: {str(e)}")


@router.get("/history/", response_model=QueryHistoryResponse)
async def get_query_history(
    collection_name: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    current_user: User = Depends(get_current_verified_user_from_db),
    query_repo: QueryRepository = Depends(get_query_repository)
):
    """
    Get user's query history with optional filtering.
    """
    try:
        # Get user's queries
        queries = await query_repo.get_user_queries(
            user_id=current_user.id,
            offset=offset,
            limit=limit
        )

        # Convert to response format
        query_items = []
        for query in queries:
            query_items.append(QueryHistoryItem(
                query_id=query.id,
                query=query.query_text,
                query_type=query.query_type,
                results_count=query.result_count or 0,
                search_time=query.processing_time or 0.0,
                executed_at=query.created_at,
                collection_id=query.collection_id
            ))

        # Get total count (mock for now)
        total_count = len(queries)

        # Calculate pagination
        page = (offset // limit) + 1
        has_next = (offset + limit) < total_count

        return QueryHistoryResponse(
            queries=query_items,
            total=total_count,
            page=page,
            per_page=limit,
            has_next=has_next
        )

    except Exception as e:
        logger.error("Failed to get query history", error=str(e), user_id=current_user.id)
        raise HTTPException(status_code=500, detail="Failed to get query history")


@router.post("/batch", response_model=List[QueryResponse])
async def submit_batch_queries(
    requests: List[QueryRequest],
    background_tasks: BackgroundTasks,
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
    current_user: dict = Depends(get_current_user),
    db_session = Depends(get_db_session)
):
    """
    Submit multiple queries for batch processing.

    Useful for processing multiple related queries efficiently.
    """
    try:
        if len(requests) > 20:  # Limit batch size
            raise HTTPException(status_code=400, detail="Too many queries. Maximum 20 queries per batch.")

        responses = []

        for request in requests:
            # Process each query (reuse single query logic)
            response = await submit_query(request, background_tasks, retrieval_service, current_user, db_session)
            responses.append(response)

        logger.info(
            "Batch queries processed",
            user_id=current_user["user_id"],
            query_count=len(requests)
        )

        return responses

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Batch query processing failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")


# Helper functions

def _map_query_type(query_type_str: str) -> DBQueryType:
    """Map API query type to database query type."""
    mapping = {
        "semantic": DBQueryType.SEMANTIC,
        "keyword": DBQueryType.KEYWORD,
        "hybrid": DBQueryType.HYBRID,
        "question": DBQueryType.QUESTION,
        "code": DBQueryType.CODE,
        "similarity": DBQueryType.SIMILARITY
    }
    return mapping.get(query_type_str.lower(), DBQueryType.SEMANTIC)

async def _update_query_with_results(
    query_repo: QueryRepository,
    query_id: str,
    retrieval_response
):
    """Update query with results in database."""
    try:
        # Update query status and results
        await query_repo.update_query_status(
            query_id,
            QueryStatus.COMPLETED,
            results=retrieval_response.results,
            result_count=retrieval_response.result_count,
            processing_time=retrieval_response.processing_time,
            metadata={
                "vector_search_time": retrieval_response.vector_search_time,
                "keyword_search_time": retrieval_response.keyword_search_time,
                "rerank_time": retrieval_response.rerank_time,
                "strategy_used": retrieval_response.strategy_used.value if hasattr(retrieval_response.strategy_used, 'value') else str(retrieval_response.strategy_used)
            }
        )
    except Exception as e:
        logger.error("Failed to update query results", query_id=query_id, error=str(e))
        # Update to failed status
        await query_repo.update_query_status(
            query_id,
            QueryStatus.FAILED,
            error_message=str(e)
        )


# Additional helper functions for query suggestions

async def _get_popular_queries(db_session, collection_name: str, limit: int) -> List[str]:
    """Get popular queries for a collection."""
    # TODO: Implement actual database query for popular queries
    return [
        f"How to use {collection_name}?",
        f"Best practices for {collection_name}",
        f"Examples of {collection_name}",
    ][:limit]


async def _get_user_recent_queries(db_session, user_id: str, collection_name: str, limit: int) -> List[str]:
    """Get user's recent queries for a collection."""
    # TODO: Implement actual database query for user's recent queries
    return [
        f"Recent query about {collection_name}",
        f"Previous search in {collection_name}",
    ][:limit]


async def _get_content_based_suggestions(collection_name: str, query_prefix: str, limit: int) -> List[str]:
    """Get content-based query suggestions."""
    # TODO: Implement actual content analysis for suggestions
    suggestions = [
        f"{query_prefix} implementation",
        f"{query_prefix} examples",
        f"{query_prefix} best practices",
        f"How to {query_prefix}",
        f"{query_prefix} tutorial",
    ]
    return [s for s in suggestions if s.strip()][:limit]

