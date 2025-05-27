"""
System management API models.

Pydantic models for system monitoring, health checks, and administration.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from enum import Enum


class ServiceStatus(str, Enum):
    """Service status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class LogLevel(str, Enum):
    """Log level enumeration."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertSeverity(str, Enum):
    """Alert severity enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ServiceHealthInfo(BaseModel):
    """Individual service health information."""
    status: ServiceStatus = Field(..., description="Service status")
    response_time: Optional[float] = Field(None, description="Response time in seconds")
    last_check: datetime = Field(..., description="Last health check timestamp")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional health details")
    error_message: Optional[str] = Field(None, description="Error message if unhealthy")


class HealthCheckResponse(BaseModel):
    """System health check response model."""
    status: ServiceStatus = Field(..., description="Overall system status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    version: str = Field(..., description="System version")
    uptime: float = Field(..., description="System uptime in seconds")
    services: Dict[str, ServiceHealthInfo] = Field(..., description="Individual service health")
    environment: str = Field(..., description="Environment (dev/staging/prod)")
    build_info: Optional[Dict[str, str]] = Field(None, description="Build information")


class SystemStatsResponse(BaseModel):
    """System statistics response model."""
    active_users: int = Field(..., description="Number of active users")
    total_users: int = Field(..., description="Total number of users")
    total_documents: int = Field(..., description="Total number of documents")
    total_collections: int = Field(..., description="Total number of collections")
    total_queries: int = Field(..., description="Total number of queries")
    storage_used: str = Field(..., description="Storage space used")
    storage_available: str = Field(..., description="Storage space available")
    requests_per_minute: float = Field(..., description="Current requests per minute")
    avg_response_time: float = Field(..., description="Average response time")
    error_rate: float = Field(..., description="Current error rate")
    last_updated: datetime = Field(..., description="Last update timestamp")


class SystemMetricsResponse(BaseModel):
    """System metrics response model."""
    cpu_usage: float = Field(..., description="CPU usage percentage")
    memory_usage: float = Field(..., description="Memory usage percentage")
    disk_usage: float = Field(..., description="Disk usage percentage")
    network_io: Dict[str, float] = Field(..., description="Network I/O statistics")
    database_connections: int = Field(..., description="Active database connections")
    cache_hit_rate: float = Field(..., description="Cache hit rate percentage")
    queue_size: int = Field(..., description="Background job queue size")
    active_sessions: int = Field(..., description="Number of active user sessions")
    timestamp: datetime = Field(..., description="Metrics timestamp")


class LogEntry(BaseModel):
    """Log entry model."""
    timestamp: datetime = Field(..., description="Log timestamp")
    level: LogLevel = Field(..., description="Log level")
    message: str = Field(..., description="Log message")
    component: str = Field(..., description="Component that generated the log")
    user_id: Optional[str] = Field(None, description="User ID if applicable")
    request_id: Optional[str] = Field(None, description="Request ID if applicable")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional log metadata")


class LogsResponse(BaseModel):
    """System logs response model."""
    logs: List[LogEntry] = Field(..., description="Log entries")
    total: int = Field(..., description="Total number of log entries")
    page: int = Field(..., description="Current page number")
    per_page: int = Field(..., description="Log entries per page")
    has_next: bool = Field(..., description="Whether there are more logs")
    filters_applied: Optional[Dict[str, Any]] = Field(None, description="Applied filters")


class SystemAlert(BaseModel):
    """System alert model."""
    alert_id: str = Field(..., description="Alert ID")
    title: str = Field(..., description="Alert title")
    message: str = Field(..., description="Alert message")
    severity: AlertSeverity = Field(..., description="Alert severity")
    component: str = Field(..., description="Component that triggered the alert")
    created_at: datetime = Field(..., description="Alert creation timestamp")
    resolved_at: Optional[datetime] = Field(None, description="Alert resolution timestamp")
    is_resolved: bool = Field(..., description="Whether alert is resolved")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional alert metadata")


class SystemAlertsResponse(BaseModel):
    """System alerts response model."""
    alerts: List[SystemAlert] = Field(..., description="System alerts")
    total: int = Field(..., description="Total number of alerts")
    active_alerts: int = Field(..., description="Number of active alerts")
    critical_alerts: int = Field(..., description="Number of critical alerts")
    last_updated: datetime = Field(..., description="Last update timestamp")


class SystemConfigResponse(BaseModel):
    """System configuration response model."""
    environment: str = Field(..., description="Environment name")
    debug_mode: bool = Field(..., description="Whether debug mode is enabled")
    max_upload_size: int = Field(..., description="Maximum upload size in bytes")
    supported_file_types: List[str] = Field(..., description="Supported file types")
    rate_limits: Dict[str, int] = Field(..., description="Rate limit configurations")
    feature_flags: Dict[str, bool] = Field(..., description="Feature flag settings")
    api_version: str = Field(..., description="API version")
    last_updated: datetime = Field(..., description="Configuration last updated")


class BackupInfo(BaseModel):
    """Backup information model."""
    backup_id: str = Field(..., description="Backup ID")
    backup_type: str = Field(..., description="Backup type (full/incremental)")
    size: int = Field(..., description="Backup size in bytes")
    created_at: datetime = Field(..., description="Backup creation timestamp")
    status: str = Field(..., description="Backup status")
    file_path: Optional[str] = Field(None, description="Backup file path")


class SystemBackupsResponse(BaseModel):
    """System backups response model."""
    backups: List[BackupInfo] = Field(..., description="Available backups")
    total: int = Field(..., description="Total number of backups")
    total_size: int = Field(..., description="Total backup size in bytes")
    last_backup: Optional[datetime] = Field(None, description="Last backup timestamp")
    next_scheduled: Optional[datetime] = Field(None, description="Next scheduled backup")


class MaintenanceWindow(BaseModel):
    """Maintenance window model."""
    window_id: str = Field(..., description="Maintenance window ID")
    title: str = Field(..., description="Maintenance title")
    description: str = Field(..., description="Maintenance description")
    start_time: datetime = Field(..., description="Maintenance start time")
    end_time: datetime = Field(..., description="Maintenance end time")
    is_active: bool = Field(..., description="Whether maintenance is currently active")
    affected_services: List[str] = Field(..., description="Services affected by maintenance")
    created_by: str = Field(..., description="User who created the maintenance window")


class SystemMaintenanceResponse(BaseModel):
    """System maintenance response model."""
    current_maintenance: Optional[MaintenanceWindow] = Field(None, description="Current maintenance window")
    upcoming_maintenance: List[MaintenanceWindow] = Field(..., description="Upcoming maintenance windows")
    maintenance_history: List[MaintenanceWindow] = Field(..., description="Recent maintenance history")


class SystemUsageStats(BaseModel):
    """System usage statistics model."""
    daily_active_users: int = Field(..., description="Daily active users")
    weekly_active_users: int = Field(..., description="Weekly active users")
    monthly_active_users: int = Field(..., description="Monthly active users")
    documents_uploaded_today: int = Field(..., description="Documents uploaded today")
    queries_executed_today: int = Field(..., description="Queries executed today")
    storage_growth_rate: float = Field(..., description="Storage growth rate per day")
    user_growth_rate: float = Field(..., description="User growth rate per day")
    peak_concurrent_users: int = Field(..., description="Peak concurrent users")
    avg_session_duration: float = Field(..., description="Average session duration in minutes")
    timestamp: datetime = Field(..., description="Statistics timestamp")
