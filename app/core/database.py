"""
Database connection and session management for Cognify application.

Provides async PostgreSQL connections with connection pooling and health checks.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import NullPool, QueuePool

from app.core.config import get_settings
from app.core.exceptions import DatabaseError

logger = structlog.get_logger(__name__)
settings = get_settings()


class Base(DeclarativeBase):
    """Base class for SQLAlchemy models."""
    pass


class DatabaseManager:
    """
    Database connection manager with async support.
    """

    def __init__(self):
        self._engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[async_sessionmaker[AsyncSession]] = None
        self._initialized = False

    async def initialize(self) -> None:
        """
        Initialize database connections and session factory.
        """
        if self._initialized:
            logger.warning("Database manager already initialized")
            return

        try:
            logger.info("Initializing database connections")

            # Create async engine with proper configuration for asyncpg
            database_url = getattr(settings, 'database_url_async', settings.DATABASE_URL)

            engine_kwargs = {
                "echo": settings.DATABASE_ECHO,
                "future": True,
            }

            # Configure pool based on driver and environment
            if "asyncpg" in database_url:
                if settings.is_development or settings.ENVIRONMENT == "testing":
                    engine_kwargs["poolclass"] = NullPool
                else:
                    # For asyncpg in production, use minimal pool configuration
                    engine_kwargs["pool_pre_ping"] = True
                    engine_kwargs["pool_recycle"] = 3600
            else:
                # For other drivers, use traditional pool settings
                engine_kwargs.update({
                    "poolclass": QueuePool if not settings.is_development else NullPool,
                    "pool_size": settings.DATABASE_POOL_SIZE,
                    "max_overflow": settings.DATABASE_MAX_OVERFLOW,
                    "pool_pre_ping": True,
                    "pool_recycle": 3600,
                })

            self._engine = create_async_engine(database_url, **engine_kwargs)

            # Create session factory
            self._session_factory = async_sessionmaker(
                bind=self._engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=True,
                autocommit=False,
            )

            # Test connection
            await self.health_check()

            self._initialized = True
            logger.info("Database connections initialized successfully")

        except Exception as e:
            logger.error("Failed to initialize database connections", error=str(e))
            raise DatabaseError(
                message="Failed to initialize database connections",
                operation="initialize",
                details={"error": str(e)}
            )

    async def close(self) -> None:
        """
        Close database connections.
        """
        if not self._initialized:
            return

        try:
            logger.info("Closing database connections")

            if self._engine:
                await self._engine.dispose()
                self._engine = None

            self._session_factory = None
            self._initialized = False

            logger.info("Database connections closed successfully")

        except Exception as e:
            logger.error("Error closing database connections", error=str(e))
            raise DatabaseError(
                message="Error closing database connections",
                operation="close",
                details={"error": str(e)}
            )

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get an async database session with automatic cleanup.

        Yields:
            AsyncSession: Database session

        Raises:
            DatabaseError: If session creation fails
        """
        if not self._initialized:
            raise DatabaseError(
                message="Database manager not initialized",
                operation="get_session"
            )

        session = self._session_factory()
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error("Database session error", error=str(e))
            raise DatabaseError(
                message="Database session error",
                operation="session_transaction",
                details={"error": str(e)}
            )
        finally:
            await session.close()

    async def health_check(self) -> dict:
        """
        Perform database health check.

        Returns:
            dict: Health check results

        Raises:
            DatabaseError: If health check fails
        """
        if not self._engine:
            raise DatabaseError(
                message="Database engine not initialized",
                operation="health_check"
            )

        try:
            async with self._engine.begin() as conn:
                result = await conn.execute(text("SELECT 1 as health_check"))
                row = result.fetchone()

                if row and row.health_check == 1:
                    return {
                        "status": "healthy",
                        "database": "postgresql",
                        "connection_pool": {
                            "size": self._engine.pool.size(),
                            "checked_in": self._engine.pool.checkedin(),
                            "checked_out": self._engine.pool.checkedout(),
                        }
                    }
                else:
                    raise DatabaseError(
                        message="Health check query returned unexpected result",
                        operation="health_check"
                    )

        except Exception as e:
            logger.error("Database health check failed", error=str(e))
            raise DatabaseError(
                message="Database health check failed",
                operation="health_check",
                details={"error": str(e)}
            )

    async def execute_raw_query(self, query: str, parameters: dict = None) -> list:
        """
        Execute a raw SQL query.

        Args:
            query: SQL query string
            parameters: Query parameters

        Returns:
            list: Query results

        Raises:
            DatabaseError: If query execution fails
        """
        if not self._initialized:
            raise DatabaseError(
                message="Database manager not initialized",
                operation="execute_raw_query"
            )

        try:
            async with self.get_session() as session:
                result = await session.execute(text(query), parameters or {})
                return result.fetchall()

        except Exception as e:
            logger.error("Raw query execution failed", query=query, error=str(e))
            raise DatabaseError(
                message="Raw query execution failed",
                operation="execute_raw_query",
                details={"query": query, "error": str(e)}
            )

    @property
    def engine(self) -> AsyncEngine:
        """Get the database engine."""
        if not self._engine:
            raise DatabaseError(
                message="Database engine not initialized",
                operation="get_engine"
            )
        return self._engine

    @property
    def session_factory(self) -> async_sessionmaker[AsyncSession]:
        """Get the session factory."""
        if not self._session_factory:
            raise DatabaseError(
                message="Session factory not initialized",
                operation="get_session_factory"
            )
        return self._session_factory

    @property
    def is_initialized(self) -> bool:
        """Check if database manager is initialized."""
        return self._initialized


# Global database manager instance
database_manager = DatabaseManager()


# Dependency for FastAPI
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency for getting database sessions.

    Yields:
        AsyncSession: Database session
    """
    async with database_manager.get_session() as session:
        yield session


# Utility functions

async def create_tables() -> None:
    """
    Create all database tables.
    """
    if not database_manager.is_initialized:
        await database_manager.initialize()

    try:
        logger.info("Creating database tables")
        async with database_manager.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created successfully")

    except Exception as e:
        logger.error("Failed to create database tables", error=str(e))
        raise DatabaseError(
            message="Failed to create database tables",
            operation="create_tables",
            details={"error": str(e)}
        )


async def drop_tables() -> None:
    """
    Drop all database tables.
    """
    if not database_manager.is_initialized:
        await database_manager.initialize()

    try:
        logger.info("Dropping database tables")
        async with database_manager.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        logger.info("Database tables dropped successfully")

    except Exception as e:
        logger.error("Failed to drop database tables", error=str(e))
        raise DatabaseError(
            message="Failed to drop database tables",
            operation="drop_tables",
            details={"error": str(e)}
        )
