#!/usr/bin/env python3
"""
Script to activate test users for API testing.
"""

import asyncio
import sys
import os

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from app.services.database.session import db_session
from app.services.database.repositories import UserRepository
from app.models.users import UserStatus

async def activate_user(email: str):
    """Activate a user by email."""
    try:
        async with db_session.get_session() as session:
            user_repo = UserRepository(session)
            
            # Get user by email
            user = await user_repo.get_by_email(email)
            if not user:
                print(f"❌ User not found: {email}")
                return False
                
            # Verify email and activate
            user.verify_email()
            await session.commit()
            
            print(f"✅ User activated: {email}")
            print(f"   Status: {user.status}")
            print(f"   Is Active: {user.is_active}")
            print(f"   Is Verified: {user.is_verified}")
            
            return True
            
    except Exception as e:
        print(f"❌ Error activating user {email}: {e}")
        return False

async def main():
    """Main function."""
    print("🔧 Activating test users...")
    
    # Initialize database
    await db_session.initialize()
    
    # Activate test users
    test_emails = [
        "test@example.com",
        "admin@example.com"
    ]
    
    for email in test_emails:
        await activate_user(email)
    
    print("✅ User activation completed!")

if __name__ == "__main__":
    asyncio.run(main())
