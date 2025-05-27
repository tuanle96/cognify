#!/usr/bin/env python3
"""
Create admin user for testing system endpoints.
"""

import requests
import json
import time

BASE_URL = "http://localhost:8001"

def create_admin_user():
    """Create admin user and return access token."""
    print("🔧 CREATING ADMIN USER")
    print("=" * 30)
    
    # Admin user data
    admin_data = {
        "email": "admin@cognify.com",
        "password": "AdminPassword123!",
        "full_name": "System Administrator",
        "username": "admin"
    }
    
    # Register admin user
    print("1. Registering admin user...")
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/auth/register",
            json=admin_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            print("   ✅ Admin user registered successfully")
            data = response.json()
            admin_token = data.get('access_token')
            print(f"   Admin Token: {admin_token[:50]}...")
            return admin_token
        else:
            print(f"   ❌ Registration failed: {response.status_code}")
            print(f"   Error: {response.text}")
            
            # Try to login if user already exists
            print("\n2. Trying to login existing admin user...")
            login_response = requests.post(
                f"{BASE_URL}/api/v1/auth/login",
                json={"email": admin_data["email"], "password": admin_data["password"]},
                headers={"Content-Type": "application/json"}
            )
            
            if login_response.status_code == 200:
                print("   ✅ Admin user logged in successfully")
                data = login_response.json()
                admin_token = data.get('access_token')
                print(f"   Admin Token: {admin_token[:50]}...")
                return admin_token
            else:
                print(f"   ❌ Login failed: {login_response.status_code}")
                print(f"   Error: {login_response.text}")
                return None
                
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        return None

def test_admin_endpoints(admin_token):
    """Test admin-only endpoints."""
    print("\n🛠️ TESTING ADMIN ENDPOINTS")
    print("=" * 30)
    
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

def main():
    """Main function."""
    admin_token = create_admin_user()
    
    if admin_token:
        test_admin_endpoints(admin_token)
        print(f"\n📋 ADMIN TOKEN FOR TESTING:")
        print(f"   {admin_token}")
        print("\n   Use this token to test admin endpoints!")
    else:
        print("\n❌ Failed to create/login admin user")

if __name__ == "__main__":
    main()
