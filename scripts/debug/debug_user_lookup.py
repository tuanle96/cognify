#!/usr/bin/env python3
"""
Debug user lookup in database for token refresh.
"""

import asyncio
import sys
import os
import requests
import json
import jwt

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.core.config import get_settings
from app.services.database.session import get_db_session
from app.services.database.repositories.user_repository import UserRepository

BASE_URL = "http://localhost:8001"

async def debug_user_lookup():
    """Debug user lookup in database."""
    print("🔍 DEBUGGING USER LOOKUP FOR TOKEN REFRESH")
    print("=" * 60)
    
    # Step 1: Create a new user via API
    print("1. Creating new user via API...")
    timestamp = int(asyncio.get_event_loop().time())
    register_data = {
        "email": f"lookup_debug_{timestamp}@example.com",
        "password": "DebugPassword123!",
        "full_name": "Lookup Debug User",
        "username": f"lookup_debug_{timestamp}"
    }
    
    response = requests.post(
        f"{BASE_URL}/api/v1/auth/register",
        json=register_data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"   Registration Status: {response.status_code}")
    if response.status_code != 200:
        print(f"   Error: {response.text}")
        return
    
    # Step 2: Login to get tokens
    print("\n2. Logging in to get tokens...")
    login_data = {
        "email": register_data["email"],
        "password": register_data["password"]
    }
    
    response = requests.post(
        f"{BASE_URL}/api/v1/auth/login",
        json=login_data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"   Login Status: {response.status_code}")
    if response.status_code != 200:
        print(f"   Error: {response.text}")
        return
    
    login_result = response.json()
    refresh_token = login_result.get('refresh_token')
    
    # Step 3: Decode refresh token to get user ID
    print("\n3. Decoding refresh token to get user ID...")
    try:
        payload = jwt.decode(refresh_token, options={"verify_signature": False})
        user_id = payload.get("sub")
        print(f"   User ID from token: {user_id}")
    except Exception as e:
        print(f"   Error decoding token: {e}")
        return
    
    # Step 4: Check if user exists in database
    print("\n4. Checking user in database...")
    settings = get_settings()
    
    try:
        async for session in get_db_session():
            user_repo = UserRepository(session)
            
            # Try to get user by ID
            user = await user_repo.get_by_id(user_id)
            
            if user:
                print(f"   ✅ User found in database:")
                print(f"   📧 Email: {user.email}")
                print(f"   👤 Username: {user.username}")
                print(f"   🔑 Role: {user.role.value}")
                print(f"   ✅ Is Active: {user.is_active}")
                print(f"   ✅ Is Verified: {user.is_verified}")
                print(f"   📋 ID: {user.id}")
                
                # Check if user is active
                if user.is_active:
                    print(f"   ✅ User is active - should work for refresh")
                else:
                    print(f"   ❌ User is NOT active - this would cause refresh to fail")
                    
            else:
                print(f"   ❌ User NOT found in database with ID: {user_id}")
                
                # Try to find user by email
                user_by_email = await user_repo.get_by_email(register_data["email"])
                if user_by_email:
                    print(f"   ⚠️ But user found by email with different ID: {user_by_email.id}")
                else:
                    print(f"   ❌ User also not found by email")
            
            break  # Exit the async generator
            
    except Exception as e:
        print(f"   ❌ Error checking database: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Step 5: Test refresh token endpoint
    print("\n5. Testing refresh token endpoint...")
    refresh_request = {"refresh_token": refresh_token}
    
    response = requests.post(
        f"{BASE_URL}/api/v1/auth/refresh",
        json=refresh_request,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"   Refresh Status: {response.status_code}")
    print(f"   Response: {response.text}")

async def main():
    """Main function."""
    await debug_user_lookup()

if __name__ == "__main__":
    asyncio.run(main())
