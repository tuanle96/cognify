
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import jwt
from datetime import datetime, timedelta

router = APIRouter()

class SimpleUser(BaseModel):
    email: str
    password: str
    full_name: str = "Test User"

class SimpleToken(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: Dict[str, Any]

@router.post("/simple-register", response_model=SimpleToken)
async def simple_register(user: SimpleUser):
    """Simple registration without database."""
    # Generate a simple token
    payload = {
        "sub": "test-user-123",
        "email": user.email,
        "role": "user",
        "type": "access",
        "exp": datetime.utcnow() + timedelta(hours=24),
        "iat": datetime.utcnow()
    }
    
    token = jwt.encode(payload, "dev-secret-key-change-in-production", algorithm="HS256")
    
    return SimpleToken(
        access_token=token,
        user={
            "user_id": "test-user-123",
            "email": user.email,
            "full_name": user.full_name,
            "is_active": True,
            "is_verified": True,
            "role": "user"
        }
    )

@router.post("/simple-login", response_model=SimpleToken)
async def simple_login(user: SimpleUser):
    """Simple login without database."""
    return await simple_register(user)
