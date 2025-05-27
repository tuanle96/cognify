"""
Database session management for Cognify.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool
import structlog

from app.core.config import get_settings
from app.models.base import Base

logger = structlog.get_logger(__name__)
settings = get_settings()


class DatabaseSession:
    """
    Database session manager for async SQLAlchemy operations.
    """

    def __init__(self):
        self.engine = None
        self.session_factory = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize database engine and session factory."""
        if self._initialized:
            return

        try:
            # Create async engine
            database_url = self._get_database_url()

            # Configure engine parameters based on environment and driver
            engine_kwargs = {
                "echo": settings.DATABASE_ECHO if hasattr(settings, 'DATABASE_ECHO') else False,
                "future": True,
            }

            # For asyncpg driver, use different pool configuration
            if "asyncpg" in database_url:
                if settings.ENVIRONMENT == "testing":
                    # Use NullPool for testing to avoid connection issues
                    engine_kwargs["poolclass"] = NullPool
                else:
                    # For asyncpg, don't use pool_size and max_overflow
                    # Use connection pool parameters that work with asyncpg
                    engine_kwargs["pool_pre_ping"] = True
                    engine_kwargs["pool_recycle"] = 3600
            else:
                # For other drivers, use traditional pool settings
                engine_kwargs.update({
                    "pool_size": getattr(settings, 'DATABASE_POOL_SIZE', 10),
                    "max_overflow": getattr(settings, 'DATABASE_MAX_OVERFLOW', 20),
                    "pool_pre_ping": True,
                    "pool_recycle": 3600,
                })

            self.engine = create_async_engine(database_url, **engine_kwargs)

            # Create session factory
            self.session_factory = async_sessionmaker(
                bind=self.engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=True,
                autocommit=False,
            )

            self._initialized = True
            logger.info("Database session manager initialized", database_url=database_url.split('@')[0] + '@***')

        except Exception as e:
            logger.error("Failed to initialize database session", error=str(e))
            raise

    async def create_tables(self) -> None:
        """Create all database tables."""
        if not self._initialized:
            await self.initialize()

        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created successfully")

        except Exception as e:
            logger.error("Failed to create database tables", error=str(e))
            raise

    async def drop_tables(self) -> None:
        """Drop all database tables."""
        if not self._initialized:
            await self.initialize()

        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
            logger.info("Database tables dropped successfully")

        except Exception as e:
            logger.error("Failed to drop database tables", error=str(e))
            raise

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get database session with automatic cleanup."""
        if not self._initialized:
            await self.initialize()

        session = self.session_factory()
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error("Database session error", error=str(e))
            raise
        finally:
            await session.close()

    async def health_check(self) -> dict:
        """Check database health."""
        if not self._initialized:
            return {"status": "unhealthy", "error": "Not initialized"}

        try:
            async with self.get_session() as session:
                # Simple query to test connection
                from sqlalchemy import text
                result = await session.execute(text("SELECT 1"))
                result.fetchone()

            return {
                "status": "healthy",
                "database": "postgresql",
                "connection": "active"
            }

        except Exception as e:
            logger.error("Database health check failed", error=str(e))
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    async def cleanup(self) -> None:
        """Cleanup database resources."""
        if self.engine:
            await self.engine.dispose()
            logger.info("Database engine disposed")

        self._initialized = False

    def _get_database_url(self) -> str:
        """Get database URL from settings."""
        return settings.database_url_async


# Global database session manager
db_session = DatabaseSession()


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency function to get database session for FastAPI.
    """
    async with db_session.get_session() as session:
        yield session


async def init_database() -> None:
    """Initialize database for application startup."""
    await db_session.initialize()

    # Create tables if they don't exist
    if settings.ENVIRONMENT in ["development", "testing"]:
        await db_session.create_tables()


async def cleanup_database() -> None:
    """Cleanup database for application shutdown."""
    await db_session.cleanup()


# Utility functions for testing
async def reset_database() -> None:
    """Reset database (drop and recreate tables) - for testing only."""
    if settings.ENVIRONMENT != "testing":
        raise RuntimeError("Database reset is only allowed in testing environment")

    await db_session.drop_tables()
    await db_session.create_tables()
    logger.info("Database reset completed")


async def execute_raw_sql(sql: str, params: dict = None) -> list:
    """Execute raw SQL query - use with caution."""
    async with db_session.get_session() as session:
        result = await session.execute(sql, params or {})
        return result.fetchall()


# Transaction management utilities
@asynccontextmanager
async def transaction():
    """Context manager for explicit transaction control."""
    async with db_session.get_session() as session:
        async with session.begin():
            yield session


class DatabaseError(Exception):
    """Base exception for database operations."""
    pass


class ConnectionError(DatabaseError):
    """Database connection error."""
    pass


class IntegrityError(DatabaseError):
    """Database integrity constraint error."""
    pass


class NotFoundError(DatabaseError):
    """Record not found error."""
    pass
