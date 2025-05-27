#!/usr/bin/env python3
"""
Create admin user directly in database for testing system endpoints.
"""

import asyncio
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.core.config import get_settings
from app.services.database.session import get_db_session
from app.services.database.repositories.user_repository import UserRepository
from app.models.users import UserRole, UserStatus

async def create_admin_user():
    """Create admin user directly in database."""
    print("🔧 CREATING ADMIN USER IN DATABASE")
    print("=" * 40)

    settings = get_settings()

    # Admin user data
    admin_data = {
        "email": "admin@cognify.com",
        "password": "AdminPassword123!",
        "full_name": "System Administrator",
        "username": "sysadmin",
        "role": UserRole.ADMIN
    }

    try:
        # Get database session
        async for session in get_db_session():
            user_repo = UserRepository(session)

            # Check if admin user already exists
            existing_user = await user_repo.get_by_email(admin_data["email"])

            if existing_user:
                print(f"   ⚠️ Admin user already exists: {existing_user.email}")

                # Update role to admin if not already
                if existing_user.role != UserRole.ADMIN:
                    print("   🔄 Updating user role to admin...")
                    await user_repo.update(
                        existing_user.id,
                        role=UserRole.ADMIN,
                        status=UserStatus.ACTIVE
                    )
                    await session.commit()
                    print("   ✅ User role updated to admin")
                else:
                    print("   ✅ User is already admin")

                # Ensure user is active
                if existing_user.status != UserStatus.ACTIVE:
                    print("   🔄 Activating user account...")
                    await user_repo.update(
                        existing_user.id,
                        status=UserStatus.ACTIVE,
                        is_verified=True
                    )
                    await session.commit()
                    print("   ✅ User account activated")

                return existing_user
            else:
                print("   📝 Creating new admin user...")

                # Create admin user
                admin_user = await user_repo.create_user(
                    email=admin_data["email"],
                    password=admin_data["password"],
                    full_name=admin_data["full_name"],
                    username=admin_data["username"],
                    role=UserRole.ADMIN
                )

                # Activate the user immediately
                await user_repo.update(
                    admin_user.id,
                    status=UserStatus.ACTIVE,
                    is_verified=True
                )

                await session.commit()

                print(f"   ✅ Admin user created successfully")
                print(f"   📧 Email: {admin_user.email}")
                print(f"   👤 Username: {admin_user.username}")
                print(f"   🔑 Role: {admin_user.role.value}")
                print(f"   📋 ID: {admin_user.id}")

                return admin_user

    except Exception as e:
        print(f"   ❌ Error creating admin user: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

async def test_admin_login():
    """Test admin user login."""
    print("\n🔐 TESTING ADMIN LOGIN")
    print("=" * 30)

    import requests

    BASE_URL = "http://localhost:8001"

    login_data = {
        "email": "admin@cognify.com",
        "password": "AdminPassword123!"
    }

    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/auth/login",
            json=login_data,
            headers={"Content-Type": "application/json"}
        )

        if response.status_code == 200:
            print("   ✅ Admin login successful")
            data = response.json()
            admin_token = data.get('access_token')
            print(f"   🎫 Admin Token: {admin_token[:50]}...")

            # Test admin endpoints
            await test_admin_endpoints(admin_token)

            return admin_token
        else:
            print(f"   ❌ Login failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return None

    except Exception as e:
        print(f"   ❌ Error during login: {str(e)}")
        return None

async def test_admin_endpoints(admin_token):
    """Test admin-only endpoints."""
    print("\n🛠️ TESTING ADMIN ENDPOINTS")
    print("=" * 30)

    import requests

    BASE_URL = "http://localhost:8001"

    auth_headers = {
        "Authorization": f"Bearer {admin_token}",
        "Content-Type": "application/json"
    }

    # Test system logs
    print("1. Testing System Logs...")
    try:
        response = requests.get(
            f"{BASE_URL}/api/v1/system/logs",
            headers=auth_headers
        )
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print("   ✅ System logs accessible")
        else:
            print(f"   ❌ Error: {response.text}")
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")

    # Test system alerts
    print("\n2. Testing System Alerts...")
    try:
        response = requests.get(
            f"{BASE_URL}/api/v1/system/alerts",
            headers=auth_headers
        )
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print("   ✅ System alerts accessible")
        else:
            print(f"   ❌ Error: {response.text}")
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")

async def main():
    """Main function."""
    admin_user = await create_admin_user()

    if admin_user:
        admin_token = await test_admin_login()

        if admin_token:
            print(f"\n📋 ADMIN CREDENTIALS FOR TESTING:")
            print(f"   Email: admin@cognify.com")
            print(f"   Password: AdminPassword123!")
            print(f"   Token: {admin_token}")
            print("\n   Use these credentials to test admin endpoints!")
        else:
            print("\n❌ Failed to login admin user")
    else:
        print("\n❌ Failed to create admin user")

if __name__ == "__main__":
    asyncio.run(main())
