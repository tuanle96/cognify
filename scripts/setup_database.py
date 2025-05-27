#!/usr/bin/env python3
"""
Database setup script for Cognify RAG System.

This script handles:
- Database initialization
- Running migrations
- Creating initial data
- Setting up vector collections
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import structlog
from sqlalchemy import text

from app.core.config import get_settings
from app.services.database.session import db_session, init_database, reset_database
from app.services.vectordb.service import VectorDBService
from app.models.users import User, UserProfile, UserRole, UserStatus
from app.models.collections import Collection, CollectionStatus, CollectionVisibility

logger = structlog.get_logger(__name__)
settings = get_settings()


async def setup_database():
    """Set up the complete database system."""
    logger.info("Starting database setup", environment=settings.ENVIRONMENT)
    
    try:
        # Initialize database connection
        await init_database()
        logger.info("Database connection initialized")
        
        # Run migrations if needed
        await run_migrations()
        
        # Create initial data
        await create_initial_data()
        
        # Set up vector database
        await setup_vector_database()
        
        logger.info("Database setup completed successfully")
        
    except Exception as e:
        logger.error("Database setup failed", error=str(e))
        raise


async def run_migrations():
    """Run database migrations using Alembic."""
    logger.info("Running database migrations")
    
    try:
        # Import alembic here to avoid circular imports
        from alembic.config import Config
        from alembic import command
        
        # Get alembic config
        alembic_cfg = Config(str(project_root / "alembic.ini"))
        
        # Run migrations
        command.upgrade(alembic_cfg, "head")
        logger.info("Database migrations completed")
        
    except ImportError:
        logger.warning("Alembic not available, skipping migrations")
    except Exception as e:
        logger.error("Migration failed", error=str(e))
        raise


async def create_initial_data():
    """Create initial data for the system."""
    logger.info("Creating initial data")
    
    try:
        async with db_session.get_session() as session:
            # Check if admin user exists
            result = await session.execute(
                text("SELECT id FROM users WHERE email = :email"),
                {"email": "admin@cognify.local"}
            )
            admin_exists = result.fetchone()
            
            if not admin_exists:
                await create_admin_user(session)
            
            # Create default collection
            result = await session.execute(
                text("SELECT id FROM collections WHERE name = :name"),
                {"name": "default"}
            )
            default_collection_exists = result.fetchone()
            
            if not default_collection_exists:
                await create_default_collection(session)
            
            await session.commit()
            logger.info("Initial data created successfully")
            
    except Exception as e:
        logger.error("Failed to create initial data", error=str(e))
        raise


async def create_admin_user(session):
    """Create the initial admin user."""
    from passlib.context import CryptContext
    
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    
    # Hash the default password
    password_hash = pwd_context.hash("admin123")
    
    # Create admin user
    admin_user = User(
        email="admin@cognify.local",
        username="admin",
        full_name="System Administrator",
        password_hash=password_hash,
        role=UserRole.ADMIN,
        status=UserStatus.ACTIVE,
        is_verified=True
    )
    
    session.add(admin_user)
    await session.flush()  # Get the user ID
    
    # Create admin profile
    admin_profile = UserProfile(
        user_id=admin_user.id,
        bio="System Administrator",
        company="Cognify",
        job_title="Administrator"
    )
    
    session.add(admin_profile)
    logger.info("Admin user created", email="admin@cognify.local")


async def create_default_collection(session):
    """Create the default collection."""
    default_collection = Collection(
        name="default",
        display_name="Default Collection",
        description="Default collection for documents",
        status=CollectionStatus.ACTIVE,
        visibility=CollectionVisibility.PUBLIC,
        embedding_dimension=settings.VECTOR_DIMENSION,
        distance_metric="cosine"
    )
    
    session.add(default_collection)
    logger.info("Default collection created")


async def setup_vector_database():
    """Set up vector database collections."""
    logger.info("Setting up vector database")
    
    try:
        # Initialize vector database service
        vector_service = VectorDBService()
        await vector_service.initialize()
        
        # Create default collection
        collection_name = settings.QDRANT_COLLECTION_NAME
        success = await vector_service.create_collection(
            collection_name=collection_name,
            vector_dimension=settings.VECTOR_DIMENSION,
            distance_metric="cosine"
        )
        
        if success:
            logger.info("Vector database collection created", collection=collection_name)
        else:
            logger.warning("Failed to create vector database collection", collection=collection_name)
        
        # Health check
        health = await vector_service.health_check()
        logger.info("Vector database health", status=health.get("status"))
        
        await vector_service.cleanup()
        
    except Exception as e:
        logger.error("Vector database setup failed", error=str(e))
        # Don't raise here as vector DB might not be available in all environments


async def reset_all_data():
    """Reset all data (for development/testing only)."""
    if settings.ENVIRONMENT not in ["development", "testing"]:
        raise RuntimeError("Data reset is only allowed in development/testing environments")
    
    logger.warning("Resetting all data", environment=settings.ENVIRONMENT)
    
    try:
        # Reset PostgreSQL database
        await reset_database()
        
        # Reset vector database
        vector_service = VectorDBService()
        await vector_service.initialize()
        
        collections = await vector_service.list_collections()
        for collection in collections:
            await vector_service.delete_collection(collection)
            logger.info("Deleted vector collection", collection=collection)
        
        await vector_service.cleanup()
        
        logger.info("All data reset completed")
        
    except Exception as e:
        logger.error("Data reset failed", error=str(e))
        raise


async def backup_database():
    """Create a database backup."""
    logger.info("Creating database backup")
    
    try:
        import subprocess
        from datetime import datetime
        
        # Create backup directory
        backup_dir = project_root / "backups"
        backup_dir.mkdir(exist_ok=True)
        
        # Generate backup filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = backup_dir / f"cognify_backup_{timestamp}.sql"
        
        # Extract database connection info
        db_url = settings.DATABASE_URL
        if "postgresql+asyncpg://" in db_url:
            db_url = db_url.replace("postgresql+asyncpg://", "postgresql://")
        
        # Run pg_dump
        cmd = [
            "pg_dump",
            db_url,
            "--no-password",
            "--verbose",
            "--clean",
            "--no-acl",
            "--no-owner",
            "-f", str(backup_file)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("Database backup created", file=str(backup_file))
            return str(backup_file)
        else:
            logger.error("Database backup failed", error=result.stderr)
            return None
            
    except Exception as e:
        logger.error("Database backup failed", error=str(e))
        return None


async def restore_database(backup_file: str):
    """Restore database from backup."""
    logger.info("Restoring database from backup", file=backup_file)
    
    try:
        import subprocess
        
        # Extract database connection info
        db_url = settings.DATABASE_URL
        if "postgresql+asyncpg://" in db_url:
            db_url = db_url.replace("postgresql+asyncpg://", "postgresql://")
        
        # Run psql to restore
        cmd = [
            "psql",
            db_url,
            "--no-password",
            "--verbose",
            "-f", backup_file
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("Database restored successfully")
            return True
        else:
            logger.error("Database restore failed", error=result.stderr)
            return False
            
    except Exception as e:
        logger.error("Database restore failed", error=str(e))
        return False


async def main():
    """Main function to handle command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Cognify Database Management")
    parser.add_argument("command", choices=[
        "setup", "reset", "backup", "restore", "migrate"
    ], help="Command to execute")
    parser.add_argument("--backup-file", help="Backup file for restore command")
    parser.add_argument("--force", action="store_true", help="Force operation without confirmation")
    
    args = parser.parse_args()
    
    if args.command == "setup":
        await setup_database()
    
    elif args.command == "reset":
        if not args.force:
            confirm = input("This will delete all data. Are you sure? (yes/no): ")
            if confirm.lower() != "yes":
                print("Operation cancelled")
                return
        await reset_all_data()
    
    elif args.command == "backup":
        backup_file = await backup_database()
        if backup_file:
            print(f"Backup created: {backup_file}")
        else:
            print("Backup failed")
            sys.exit(1)
    
    elif args.command == "restore":
        if not args.backup_file:
            print("--backup-file is required for restore command")
            sys.exit(1)
        
        if not args.force:
            confirm = input(f"This will restore from {args.backup_file}. Continue? (yes/no): ")
            if confirm.lower() != "yes":
                print("Operation cancelled")
                return
        
        success = await restore_database(args.backup_file)
        if not success:
            sys.exit(1)
    
    elif args.command == "migrate":
        await run_migrations()


if __name__ == "__main__":
    asyncio.run(main())
