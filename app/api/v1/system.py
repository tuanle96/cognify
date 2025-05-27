"""
System management API endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Request
from typing import List, Optional
import structlog
import time
from datetime import datetime

from app.api.dependencies import (
    get_current_verified_user_from_db,
    get_user_repository,
    get_document_repository,
    get_collection_repository,
    get_query_repository
)
from app.services.database.repositories import (
    UserRepository,
    DocumentRepository,
    CollectionRepository,
    QueryRepository
)
from app.models.users import User, UserRole
from app.core.exceptions import AuthenticationError
from app.api.models.system import (
    HealthCheckResponse,
    SystemStatsResponse,
    SystemMetricsResponse,
    LogsResponse,
    SystemAlertsResponse
)

logger = structlog.get_logger(__name__)
router = APIRouter()


def require_admin_user(current_user: User = Depends(get_current_verified_user_from_db)):
    """Require admin user for system operations."""
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user


@router.get("/health", response_model=HealthCheckResponse)
async def health_check(
    request: Request,
    user_repo: UserRepository = Depends(get_user_repository),
    document_repo: DocumentRepository = Depends(get_document_repository),
    collection_repo: CollectionRepository = Depends(get_collection_repository),
    query_repo: QueryRepository = Depends(get_query_repository)
):
    """Comprehensive system health check."""
    try:
        # Check database connectivity
        services = {}

        current_time = datetime.utcnow()

        try:
            # Test database connection with a simple query
            user_count = await user_repo.count()
            services["database"] = {
                "status": "healthy",
                "last_check": current_time,
                "details": {"user_count": user_count, "message": f"Connected, {user_count} users"}
            }
        except Exception as e:
            services["database"] = {
                "status": "unhealthy",
                "last_check": current_time,
                "details": {"error": str(e)},
                "error_message": f"Database error: {str(e)}"
            }

        try:
            # Test document repository
            doc_count = await document_repo.count()
            services["document_service"] = {
                "status": "healthy",
                "last_check": current_time,
                "details": {"document_count": doc_count, "message": f"{doc_count} documents"}
            }
        except Exception as e:
            services["document_service"] = {
                "status": "unhealthy",
                "last_check": current_time,
                "details": {"error": str(e)},
                "error_message": f"Document service error: {str(e)}"
            }

        try:
            # Test collection repository
            collection_count = await collection_repo.count()
            services["collection_service"] = {
                "status": "healthy",
                "last_check": current_time,
                "details": {"collection_count": collection_count, "message": f"{collection_count} collections"}
            }
        except Exception as e:
            services["collection_service"] = {
                "status": "unhealthy",
                "last_check": current_time,
                "details": {"error": str(e)},
                "error_message": f"Collection service error: {str(e)}"
            }

        try:
            # Test query repository
            query_count = await query_repo.count()
            services["query_service"] = {
                "status": "healthy",
                "last_check": current_time,
                "details": {"query_count": query_count, "message": f"{query_count} queries"}
            }
        except Exception as e:
            services["query_service"] = {
                "status": "unhealthy",
                "last_check": current_time,
                "details": {"error": str(e)},
                "error_message": f"Query service error: {str(e)}"
            }

        # Determine overall status
        unhealthy_services = [name for name, service in services.items() if service["status"] == "unhealthy"]
        overall_status = "unhealthy" if unhealthy_services else "healthy"

        return HealthCheckResponse(
            status=overall_status,
            timestamp=datetime.utcnow(),
            services=services,
            uptime=time.time() - getattr(request.app.state, 'start_time', time.time()),
            version="2.0.0",
            environment="production"
        )

    except Exception as e:
        logger.error("Health check failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Health check failed"
        )


@router.get("/stats", response_model=SystemStatsResponse)
async def get_system_stats(
    current_user: User = Depends(get_current_verified_user_from_db),
    user_repo: UserRepository = Depends(get_user_repository),
    document_repo: DocumentRepository = Depends(get_document_repository),
    collection_repo: CollectionRepository = Depends(get_collection_repository),
    query_repo: QueryRepository = Depends(get_query_repository)
):
    """Get comprehensive system statistics."""
    try:
        # Collect system statistics
        system_info = await _get_system_info()
        performance_metrics = await _get_performance_metrics()
        resource_usage = await _get_resource_usage()
        service_metrics = await _get_service_metrics()

        # Get real database statistics
        database_stats = await _get_database_stats_real(
            user_repo, document_repo, collection_repo, query_repo
        )

        cache_stats = await _get_cache_stats()
        api_stats = await _get_api_stats()

        return SystemStatsResponse(
            active_users=database_stats.get("total_users", 0),  # Simplified: use total as active
            total_users=database_stats.get("total_users", 0),
            total_documents=database_stats.get("total_documents", 0),
            total_collections=database_stats.get("total_collections", 0),
            total_queries=database_stats.get("total_queries", 0),
            storage_used=f"{resource_usage.get('disk_used', 0) // (1024*1024)} MB",
            storage_available=f"{resource_usage.get('disk_total', 0) // (1024*1024)} MB",
            requests_per_minute=performance_metrics.get("requests_per_second", 0.0) * 60,
            avg_response_time=performance_metrics.get("average_response_time", 0.0),
            error_rate=performance_metrics.get("error_rate", 0.0),
            last_updated=datetime.utcnow()
        )

    except Exception as e:
        logger.error("Failed to get system stats", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get system stats"
        )


@router.get("/metrics", response_model=SystemMetricsResponse)
async def get_system_metrics(
    current_user: User = Depends(get_current_verified_user_from_db)
):
    """Get real-time system metrics."""
    try:
        import psutil

        # Get system metrics
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        network = psutil.net_io_counters()

        # Try to get network connections, but fallback if it fails
        try:
            database_connections = len(psutil.net_connections())
        except:
            database_connections = 1  # Fallback value

        return SystemMetricsResponse(
            cpu_usage=cpu_usage,
            memory_usage=memory.percent,
            disk_usage=(disk.used / disk.total) * 100,
            network_io={
                "bytes_sent": float(network.bytes_sent),
                "bytes_recv": float(network.bytes_recv),
                "packets_sent": float(network.packets_sent),
                "packets_recv": float(network.packets_recv)
            },
            database_connections=database_connections,
            cache_hit_rate=0.0,  # TODO: Implement cache hit rate tracking
            queue_size=0,        # TODO: Implement queue size tracking
            active_sessions=1,   # TODO: Implement active sessions tracking
            timestamp=datetime.utcnow()
        )

    except ImportError:
        # Fallback if psutil is not available
        return SystemMetricsResponse(
            cpu_usage=0.0,
            memory_usage=0.0,
            disk_usage=0.0,
            network_io={"bytes_sent": 0.0, "bytes_recv": 0.0, "packets_sent": 0.0, "packets_recv": 0.0},
            database_connections=0,
            cache_hit_rate=0.0,
            queue_size=0,
            active_sessions=0,
            timestamp=datetime.utcnow()
        )
    except Exception as e:
        logger.error("Failed to get system metrics", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get system metrics"
        )


@router.get("/logs", response_model=LogsResponse)
async def get_system_logs(
    level: Optional[str] = None,
    service: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    current_user: User = Depends(require_admin_user)
):
    """Get system logs with filtering."""
    try:
        # TODO: Implement log retrieval from logging system
        logs = []
        total_count = 0

        filters = {}
        if level:
            filters["level"] = level
        if service:
            filters["service"] = service

        return LogsResponse(
            logs=logs,
            total=total_count,
            page=(offset // limit) + 1 if limit > 0 else 1,
            per_page=limit,
            has_next=(offset + limit) < total_count,
            filters_applied=filters
        )

    except Exception as e:
        logger.error("Failed to get system logs", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get system logs: {str(e)}"
        )


@router.get("/alerts", response_model=SystemAlertsResponse)
async def get_system_alerts(
    level: Optional[str] = None,
    resolved: Optional[bool] = None,
    limit: int = 50,
    offset: int = 0,
    current_user: User = Depends(require_admin_user)
):
    """Get system alerts."""
    try:
        # TODO: Implement alert retrieval
        alerts = []
        total_count = 0
        unresolved_count = 0
        critical_count = 0

        return SystemAlertsResponse(
            alerts=alerts,
            total=total_count,
            active_alerts=unresolved_count,
            critical_alerts=critical_count,
            last_updated=datetime.utcnow()
        )

    except Exception as e:
        logger.error("Failed to get system alerts", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get system alerts: {str(e)}"
        )


@router.post("/maintenance")
async def enable_maintenance_mode(
    current_user: dict = Depends(require_admin_user)
):
    """Enable maintenance mode."""
    try:
        # TODO: Implement maintenance mode
        logger.info("Maintenance mode enabled", user_id=current_user["user_id"])

        return {"message": "Maintenance mode enabled", "status": "enabled"}

    except Exception as e:
        logger.error("Failed to enable maintenance mode", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to enable maintenance mode: {str(e)}"
        )


@router.delete("/maintenance")
async def disable_maintenance_mode(
    current_user: dict = Depends(require_admin_user)
):
    """Disable maintenance mode."""
    try:
        # TODO: Implement maintenance mode
        logger.info("Maintenance mode disabled", user_id=current_user["user_id"])

        return {"message": "Maintenance mode disabled", "status": "disabled"}

    except Exception as e:
        logger.error("Failed to disable maintenance mode", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to disable maintenance mode: {str(e)}"
        )


# Helper functions
async def _get_system_info():
    """Get system information."""
    import platform

    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "architecture": platform.architecture()[0],
        "processor": platform.processor(),
        "hostname": platform.node()
    }


async def _get_performance_metrics():
    """Get performance metrics."""
    return {
        "requests_per_second": 0.0,
        "average_response_time": 0.0,
        "error_rate": 0.0,
        "cache_hit_rate": 0.0
    }


async def _get_resource_usage():
    """Get resource usage statistics."""
    try:
        import psutil

        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": memory.percent,
            "memory_used": memory.used,
            "memory_total": memory.total,
            "disk_percent": (disk.used / disk.total) * 100,
            "disk_used": disk.used,
            "disk_total": disk.total
        }
    except ImportError:
        return {
            "cpu_percent": 0.0,
            "memory_percent": 0.0,
            "memory_used": 0,
            "memory_total": 0,
            "disk_percent": 0.0,
            "disk_used": 0,
            "disk_total": 0
        }


async def _get_service_metrics():
    """Get per-service metrics."""
    return {
        "parsing_service": {"status": "healthy", "requests": 0, "errors": 0},
        "chunking_service": {"status": "healthy", "requests": 0, "errors": 0},
        "embedding_service": {"status": "healthy", "requests": 0, "errors": 0},
        "vectordb_service": {"status": "healthy", "requests": 0, "errors": 0}
    }


async def _get_database_stats_real(
    user_repo: UserRepository,
    document_repo: DocumentRepository,
    collection_repo: CollectionRepository,
    query_repo: QueryRepository
):
    """Get real database statistics."""
    try:
        # Get counts from each repository
        user_count = await user_repo.count()
        document_count = await document_repo.count()
        collection_count = await collection_repo.count()
        query_count = await query_repo.count()

        # Get recent activity (last 24 hours)
        from datetime import datetime, timedelta
        yesterday = datetime.utcnow() - timedelta(days=1)

        # Count recent queries
        recent_queries = await query_repo.count(filters={"created_after": yesterday})

        return {
            "total_users": user_count,
            "total_documents": document_count,
            "total_collections": collection_count,
            "total_queries": query_count,
            "recent_queries_24h": recent_queries,
            "queries_per_second": recent_queries / (24 * 60 * 60) if recent_queries > 0 else 0.0,
            "database_size_mb": 0,  # TODO: Get actual database size
            "active_connections": 1  # TODO: Get actual connection count
        }
    except Exception as e:
        logger.error("Failed to get database stats", error=str(e))
        return {
            "total_users": 0,
            "total_documents": 0,
            "total_collections": 0,
            "total_queries": 0,
            "recent_queries_24h": 0,
            "queries_per_second": 0.0,
            "database_size_mb": 0,
            "active_connections": 0
        }

async def _get_database_stats():
    """Get database statistics."""
    return {
        "connections": 0,
        "queries_per_second": 0.0,
        "slow_queries": 0,
        "size": 0
    }


async def _get_cache_stats():
    """Get cache statistics."""
    return {
        "hit_rate": 0.0,
        "miss_rate": 0.0,
        "size": 0,
        "evictions": 0
    }


async def _get_api_stats():
    """Get API statistics."""
    return {
        "total_requests": 0,
        "successful_requests": 0,
        "failed_requests": 0,
        "average_response_time": 0.0
    }
