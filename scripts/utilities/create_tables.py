#!/usr/bin/env python3
"""
Script to create database tables.
"""

import asyncio
import sys
import os

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from app.services.database.session import db_session
from app.models.base import Base
from app.models.users import User, UserProfile, UserSession

async def create_tables():
    """Create all database tables."""
    try:
        print("🔧 Initializing database connection...")
        await db_session.initialize()
        
        print("🔧 Creating tables...")
        async with db_session.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        print("✅ All tables created successfully!")
        
        # List created tables
        async with db_session.engine.begin() as conn:
            result = await conn.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
            tables = [row[0] for row in result.fetchall()]
            print(f"📋 Tables in database: {', '.join(tables)}")
        
    except Exception as e:
        print(f"❌ Error creating tables: {e}")
        raise

async def main():
    """Main function."""
    await create_tables()

if __name__ == "__main__":
    asyncio.run(main())
