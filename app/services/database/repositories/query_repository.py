"""
Query repository for database operations.
"""

import hashlib
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

from sqlalchemy import select, and_, or_, func, desc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload, joinedload
import structlog

from .base import BaseRepository
from app.models.queries import (
    Query, QueryResult, QueryFeedback,
    QueryType, QueryStatus, FeedbackType
)
from app.core.exceptions import ValidationError, NotFoundError
from app.core.config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()


class QueryRepository(BaseRepository[Query]):
    """Repository for query-related database operations."""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, Query)
    
    async def create_query(
        self,
        query_text: str,
        user_id: str,
        query_type: QueryType = QueryType.SEMANTIC,
        collection_id: str = None,
        max_results: int = 10,
        min_score: float = 0.0,
        session_id: str = None,
        ip_address: str = None,
        user_agent: str = None,
        **kwargs
    ) -> Query:
        """
        Create a new query record.
        
        Args:
            query_text: The query text
            user_id: ID of user making the query
            query_type: Type of query
            collection_id: Optional collection to search in
            max_results: Maximum number of results to return
            min_score: Minimum similarity score threshold
            session_id: User session ID
            ip_address: Client IP address
            user_agent: Client user agent
            **kwargs: Additional query fields
            
        Returns:
            Created query instance
        """
        try:
            # Validate query text
            if not query_text or len(query_text.strip()) < 3:
                raise ValidationError(
                    message="Query text must be at least 3 characters",
                    field="query_text",
                    value=query_text
                )
            
            # Generate query hash for deduplication
            query_hash = self._generate_query_hash(query_text, user_id, collection_id)
            
            # Prepare query data
            query_data = {
                "query_text": query_text.strip(),
                "query_hash": query_hash,
                "query_type": query_type,
                "collection_id": collection_id,
                "max_results": max_results,
                "min_score": min_score,
                "session_id": session_id,
                "ip_address": ip_address,
                "user_agent": user_agent,
                "created_by": user_id,
                **kwargs
            }
            
            query = await self.create(**query_data)
            
            self.logger.info(
                "Query created successfully",
                query_id=query.id,
                query_type=query_type.value,
                user_id=user_id
            )
            
            return query
            
        except ValidationError:
            raise
        except Exception as e:
            self.logger.error(
                "Failed to create query",
                query_text=query_text[:100],
                user_id=user_id,
                error=str(e)
            )
            raise
    
    async def start_processing(self, query_id: str) -> Query:
        """Mark query as processing started."""
        try:
            query = await self.get_by_id_or_raise(query_id)
            
            if query.status != QueryStatus.PENDING:
                raise ValidationError(
                    message=f"Query is not in pending status, current status: {query.status.value}",
                    field="status"
                )
            
            updated_query = await self.update(
                query_id,
                status=QueryStatus.PROCESSING,
                processing_started_at=datetime.utcnow()
            )
            
            self.logger.info("Query processing started", query_id=query_id)
            return updated_query
            
        except (NotFoundError, ValidationError):
            raise
        except Exception as e:
            self.logger.error(
                "Failed to start query processing",
                query_id=query_id,
                error=str(e)
            )
            raise
    
    async def complete_processing(
        self,
        query_id: str,
        result_count: int = 0,
        processing_time: float = None,
        embedding_time: float = None,
        search_time: float = None,
        rerank_time: float = None
    ) -> Query:
        """Mark query as processing completed."""
        try:
            query = await self.get_by_id_or_raise(query_id)
            
            update_data = {
                "status": QueryStatus.COMPLETED,
                "processing_completed_at": datetime.utcnow(),
                "result_count": result_count,
                "has_results": result_count > 0
            }
            
            if processing_time is not None:
                update_data["processing_time"] = processing_time
            
            if embedding_time is not None:
                update_data["embedding_time"] = embedding_time
            
            if search_time is not None:
                update_data["search_time"] = search_time
            
            if rerank_time is not None:
                update_data["rerank_time"] = rerank_time
            
            updated_query = await self.update(query_id, **update_data)
            
            self.logger.info(
                "Query processing completed",
                query_id=query_id,
                result_count=result_count,
                processing_time=processing_time
            )
            
            return updated_query
            
        except NotFoundError:
            raise
        except Exception as e:
            self.logger.error(
                "Failed to complete query processing",
                query_id=query_id,
                error=str(e)
            )
            raise
    
    async def fail_processing(
        self,
        query_id: str,
        error_message: str,
        error_details: Dict[str, Any] = None
    ) -> Query:
        """Mark query as processing failed."""
        try:
            updated_query = await self.update(
                query_id,
                status=QueryStatus.FAILED,
                processing_completed_at=datetime.utcnow(),
                error_message=error_message,
                error_details=error_details
            )
            
            self.logger.warning(
                "Query processing failed",
                query_id=query_id,
                error=error_message
            )
            
            return updated_query
            
        except Exception as e:
            self.logger.error(
                "Failed to mark query as failed",
                query_id=query_id,
                error=str(e)
            )
            raise
    
    async def add_result(
        self,
        query_id: str,
        rank: int,
        document_id: str,
        chunk_id: str,
        content: str,
        score: float,
        vector_score: float = None,
        rerank_score: float = None,
        highlights: List[str] = None,
        snippets: List[str] = None,
        metadata: Dict[str, Any] = None
    ) -> QueryResult:
        """Add a result to a query."""
        try:
            result_data = {
                "query_id": query_id,
                "rank": rank,
                "document_id": document_id,
                "chunk_id": chunk_id,
                "content": content,
                "content_preview": content[:500] if content else None,
                "score": score,
                "vector_score": vector_score,
                "rerank_score": rerank_score,
                "highlights": highlights,
                "snippets": snippets,
                "result_metadata": metadata or {}
            }
            
            result = QueryResult(**result_data)
            self.session.add(result)
            await self.session.flush()
            await self.session.refresh(result)
            
            self.logger.debug(
                "Query result added",
                query_id=query_id,
                rank=rank,
                score=score
            )
            
            return result
            
        except Exception as e:
            self.logger.error(
                "Failed to add query result",
                query_id=query_id,
                rank=rank,
                error=str(e)
            )
            raise
    
    async def get_results(
        self,
        query_id: str,
        offset: int = 0,
        limit: int = 100
    ) -> List[QueryResult]:
        """Get results for a query."""
        try:
            query = select(QueryResult).where(
                QueryResult.query_id == query_id
            ).order_by(QueryResult.rank).offset(offset).limit(limit)
            
            result = await self.session.execute(query)
            return list(result.scalars().all())
            
        except Exception as e:
            self.logger.error(
                "Failed to get query results",
                query_id=query_id,
                error=str(e)
            )
            return []
    
    async def record_result_click(self, result_id: str) -> bool:
        """Record that a result was clicked."""
        try:
            query = select(QueryResult).where(QueryResult.id == result_id)
            result = await self.session.execute(query)
            query_result = result.scalar_one_or_none()
            
            if query_result:
                query_result.record_click()
                await self.session.flush()
                
                self.logger.debug("Result click recorded", result_id=result_id)
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(
                "Failed to record result click",
                result_id=result_id,
                error=str(e)
            )
            return False
    
    async def add_feedback(
        self,
        query_id: str,
        user_id: str,
        feedback_type: FeedbackType,
        rating: int,
        comments: str = None,
        helpful_results: List[str] = None,
        unhelpful_results: List[str] = None,
        suggested_query: str = None,
        missing_information: str = None,
        context: Dict[str, Any] = None
    ) -> QueryFeedback:
        """Add feedback for a query."""
        try:
            # Validate rating
            if not 1 <= rating <= 5:
                raise ValidationError(
                    message="Rating must be between 1 and 5",
                    field="rating",
                    value=rating
                )
            
            feedback_data = {
                "query_id": query_id,
                "feedback_type": feedback_type,
                "rating": rating,
                "comments": comments,
                "helpful_results": helpful_results,
                "unhelpful_results": unhelpful_results,
                "suggested_query": suggested_query,
                "missing_information": missing_information,
                "feedback_context": context or {},
                "created_by": user_id
            }
            
            feedback = QueryFeedback(**feedback_data)
            self.session.add(feedback)
            await self.session.flush()
            await self.session.refresh(feedback)
            
            self.logger.info(
                "Query feedback added",
                query_id=query_id,
                feedback_type=feedback_type.value,
                rating=rating
            )
            
            return feedback
            
        except ValidationError:
            raise
        except Exception as e:
            self.logger.error(
                "Failed to add query feedback",
                query_id=query_id,
                error=str(e)
            )
            raise
    
    async def get_user_queries(
        self,
        user_id: str,
        status: QueryStatus = None,
        query_type: QueryType = None,
        collection_id: str = None,
        offset: int = 0,
        limit: int = 100
    ) -> List[Query]:
        """Get queries by user with optional filtering."""
        try:
            query = select(Query).where(Query.created_by == user_id)
            
            if status:
                query = query.where(Query.status == status)
            
            if query_type:
                query = query.where(Query.query_type == query_type)
            
            if collection_id:
                query = query.where(Query.collection_id == collection_id)
            
            query = query.order_by(desc(Query.created_at)).offset(offset).limit(limit)
            query = query.options(selectinload(Query.results))
            
            result = await self.session.execute(query)
            return list(result.scalars().all())
            
        except Exception as e:
            self.logger.error(
                "Failed to get user queries",
                user_id=user_id,
                error=str(e)
            )
            return []
    
    async def get_recent_queries(
        self,
        user_id: str,
        hours: int = 24,
        limit: int = 10
    ) -> List[Query]:
        """Get recent queries by user."""
        try:
            since = datetime.utcnow() - timedelta(hours=hours)
            
            query = select(Query).where(
                and_(
                    Query.created_by == user_id,
                    Query.created_at >= since
                )
            ).order_by(desc(Query.created_at)).limit(limit)
            
            result = await self.session.execute(query)
            return list(result.scalars().all())
            
        except Exception as e:
            self.logger.error(
                "Failed to get recent queries",
                user_id=user_id,
                error=str(e)
            )
            return []
    
    async def get_query_analytics(
        self,
        user_id: str = None,
        collection_id: str = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get query analytics and statistics."""
        try:
            since = datetime.utcnow() - timedelta(days=days)
            base_query = select(Query).where(Query.created_at >= since)
            
            if user_id:
                base_query = base_query.where(Query.created_by == user_id)
            
            if collection_id:
                base_query = base_query.where(Query.collection_id == collection_id)
            
            # Total queries
            total_query = select(func.count(Query.id)).where(Query.created_at >= since)
            if user_id:
                total_query = total_query.where(Query.created_by == user_id)
            if collection_id:
                total_query = total_query.where(Query.collection_id == collection_id)
            
            result = await self.session.execute(total_query)
            total_queries = result.scalar() or 0
            
            # Successful queries
            success_query = select(func.count(Query.id)).where(
                and_(
                    Query.created_at >= since,
                    Query.status == QueryStatus.COMPLETED,
                    Query.has_results == True
                )
            )
            if user_id:
                success_query = success_query.where(Query.created_by == user_id)
            if collection_id:
                success_query = success_query.where(Query.collection_id == collection_id)
            
            result = await self.session.execute(success_query)
            successful_queries = result.scalar() or 0
            
            # Average processing time
            avg_time_query = select(func.avg(Query.processing_time)).where(
                and_(
                    Query.created_at >= since,
                    Query.processing_time.is_not(None)
                )
            )
            if user_id:
                avg_time_query = avg_time_query.where(Query.created_by == user_id)
            if collection_id:
                avg_time_query = avg_time_query.where(Query.collection_id == collection_id)
            
            result = await self.session.execute(avg_time_query)
            avg_processing_time = result.scalar() or 0.0
            
            # Query types distribution
            type_query = select(
                Query.query_type,
                func.count(Query.id)
            ).where(Query.created_at >= since).group_by(Query.query_type)
            
            if user_id:
                type_query = type_query.where(Query.created_by == user_id)
            if collection_id:
                type_query = type_query.where(Query.collection_id == collection_id)
            
            result = await self.session.execute(type_query)
            type_distribution = {row[0].value: row[1] for row in result.fetchall()}
            
            analytics = {
                "total_queries": total_queries,
                "successful_queries": successful_queries,
                "success_rate": successful_queries / total_queries if total_queries > 0 else 0.0,
                "average_processing_time": float(avg_processing_time),
                "query_types": type_distribution,
                "period_days": days
            }
            
            return analytics
            
        except Exception as e:
            self.logger.error(
                "Failed to get query analytics",
                user_id=user_id,
                collection_id=collection_id,
                error=str(e)
            )
            return {}
    
    def _generate_query_hash(self, query_text: str, user_id: str, collection_id: str = None) -> str:
        """Generate hash for query deduplication."""
        content = f"{query_text.lower().strip()}:{user_id}"
        if collection_id:
            content += f":{collection_id}"
        
        return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]
