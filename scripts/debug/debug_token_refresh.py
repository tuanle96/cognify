#!/usr/bin/env python3
"""
Debug token refresh issue in detail.
"""

import requests
import json
import jwt
from datetime import datetime

BASE_URL = "http://localhost:8001"

def decode_jwt_token(token):
    """Decode JWT token without verification for debugging."""
    try:
        # Decode without verification to see payload
        payload = jwt.decode(token, options={"verify_signature": False})
        return payload
    except Exception as e:
        return {"error": str(e)}

def test_token_refresh_debug():
    """Debug token refresh step by step."""
    print("🔍 DEBUGGING TOKEN REFRESH")
    print("=" * 50)
    
    # Step 1: Register new user
    print("1. Registering new user...")
    timestamp = int(datetime.now().timestamp())
    register_data = {
        "email": f"debug_user_{timestamp}@example.com",
        "password": "DebugPassword123!",
        "full_name": "Debug User",
        "username": f"debug_user_{timestamp}"
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
    access_token = login_result.get('access_token')
    refresh_token = login_result.get('refresh_token')
    
    print(f"   ✅ Tokens obtained")
    print(f"   Access Token: {access_token[:50]}...")
    print(f"   Refresh Token: {refresh_token[:50]}...")
    
    # Step 3: Decode tokens to see structure
    print("\n3. Analyzing token structure...")
    
    access_payload = decode_jwt_token(access_token)
    refresh_payload = decode_jwt_token(refresh_token)
    
    print("   Access Token Payload:")
    print(f"   {json.dumps(access_payload, indent=4, default=str)}")
    
    print("\n   Refresh Token Payload:")
    print(f"   {json.dumps(refresh_payload, indent=4, default=str)}")
    
    # Step 4: Test refresh token endpoint
    print("\n4. Testing refresh token endpoint...")
    
    refresh_request = {"refresh_token": refresh_token}
    
    response = requests.post(
        f"{BASE_URL}/api/v1/auth/refresh",
        json=refresh_request,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"   Refresh Status: {response.status_code}")
    print(f"   Response: {response.text}")
    
    if response.status_code == 200:
        print("   ✅ Token refresh successful!")
        refresh_result = response.json()
        new_access_token = refresh_result.get('access_token')
        new_refresh_token = refresh_result.get('refresh_token')
        
        print(f"   New Access Token: {new_access_token[:50]}...")
        print(f"   New Refresh Token: {new_refresh_token[:50]}...")
        
        # Decode new tokens
        new_access_payload = decode_jwt_token(new_access_token)
        new_refresh_payload = decode_jwt_token(new_refresh_token)
        
        print("\n   New Access Token Payload:")
        print(f"   {json.dumps(new_access_payload, indent=4, default=str)}")
        
        print("\n   New Refresh Token Payload:")
        print(f"   {json.dumps(new_refresh_payload, indent=4, default=str)}")
        
    else:
        print("   ❌ Token refresh failed")
        try:
            error_data = response.json()
            print(f"   Error Details: {json.dumps(error_data, indent=4)}")
        except:
            print(f"   Raw Error: {response.text}")
    
    # Step 5: Test with access token to verify it works
    print("\n5. Testing access token validity...")
    
    auth_headers = {"Authorization": f"Bearer {access_token}"}
    
    response = requests.get(
        f"{BASE_URL}/api/v1/documents/",
        headers=auth_headers
    )
    
    print(f"   Access Token Test Status: {response.status_code}")
    if response.status_code == 200:
        print("   ✅ Access token is valid")
    else:
        print("   ❌ Access token is invalid")
        print(f"   Error: {response.text}")

def test_manual_refresh_token():
    """Test with manually crafted refresh token."""
    print("\n🔧 TESTING MANUAL REFRESH TOKEN")
    print("=" * 50)
    
    # Create a manual refresh token with correct structure
    import jwt
    from datetime import datetime, timedelta
    
    # Use the same secret key as the app (you might need to check settings)
    SECRET_KEY = "your-secret-key-here"  # This should match app settings
    
    # Create manual refresh token
    payload = {
        "sub": "test-user-id",
        "type": "refresh",
        "exp": datetime.utcnow() + timedelta(days=7),
        "iat": datetime.utcnow()
    }
    
    manual_refresh_token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
    
    print(f"   Manual Refresh Token: {manual_refresh_token[:50]}...")
    
    # Test with manual token
    refresh_request = {"refresh_token": manual_refresh_token}
    
    response = requests.post(
        f"{BASE_URL}/api/v1/auth/refresh",
        json=refresh_request,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"   Manual Refresh Status: {response.status_code}")
    print(f"   Response: {response.text}")

def main():
    """Main debug function."""
    test_token_refresh_debug()
    # test_manual_refresh_token()  # Uncomment if needed

if __name__ == "__main__":
    main()
