#!/usr/bin/env python3
"""
Test script for Cognify agentic chunking functionality.

This script tests the multi-agent chunking pipeline with different purposes
and compares results with AST-based chunking.
"""

import asyncio
import sys
import os

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.services.chunking.service import ChunkingService
from app.services.chunking.base import ChunkingRequest


async def test_agentic_vs_ast_chunking():
    """Test agentic chunking vs AST chunking comparison."""
    print("🧠 Testing Cognify Agentic Chunking Pipeline")
    print("=" * 60)
    
    # Complex Python code for testing
    complex_code = '''
import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

@dataclass
class UserProfile:
    """User profile data structure."""
    user_id: str
    username: str
    email: str
    preferences: Dict[str, Any]
    created_at: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary."""
        return {
            "user_id": self.user_id,
            "username": self.username,
            "email": self.email,
            "preferences": self.preferences,
            "created_at": self.created_at
        }

class DatabaseError(Exception):
    """Custom database exception."""
    def __init__(self, message: str, error_code: str = None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)

class BaseRepository(ABC):
    """Abstract base repository class."""
    
    @abstractmethod
    async def create(self, data: Dict[str, Any]) -> str:
        """Create new record."""
        pass
    
    @abstractmethod
    async def get_by_id(self, record_id: str) -> Optional[Dict[str, Any]]:
        """Get record by ID."""
        pass
    
    @abstractmethod
    async def update(self, record_id: str, data: Dict[str, Any]) -> bool:
        """Update existing record."""
        pass
    
    @abstractmethod
    async def delete(self, record_id: str) -> bool:
        """Delete record."""
        pass

class UserRepository(BaseRepository):
    """User repository implementation."""
    
    def __init__(self, db_connection):
        self.db = db_connection
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def create(self, data: Dict[str, Any]) -> str:
        """Create new user."""
        try:
            self.logger.info("Creating new user", username=data.get("username"))
            
            # Validate required fields
            required_fields = ["username", "email"]
            for field in required_fields:
                if field not in data:
                    raise ValueError(f"Missing required field: {field}")
            
            # Check if user already exists
            existing_user = await self.get_by_username(data["username"])
            if existing_user:
                raise DatabaseError("User already exists", "USER_EXISTS")
            
            # Insert user into database
            query = """
                INSERT INTO users (username, email, preferences, created_at)
                VALUES (%(username)s, %(email)s, %(preferences)s, NOW())
                RETURNING user_id
            """
            
            result = await self.db.execute(query, data)
            user_id = result.fetchone()["user_id"]
            
            self.logger.info("User created successfully", user_id=user_id)
            return user_id
            
        except Exception as e:
            self.logger.error("Failed to create user", error=str(e))
            raise DatabaseError(f"User creation failed: {str(e)}")
    
    async def get_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by ID."""
        try:
            query = "SELECT * FROM users WHERE user_id = %(user_id)s"
            result = await self.db.execute(query, {"user_id": user_id})
            row = result.fetchone()
            
            if row:
                return dict(row)
            return None
            
        except Exception as e:
            self.logger.error("Failed to get user", user_id=user_id, error=str(e))
            raise DatabaseError(f"User retrieval failed: {str(e)}")
    
    async def get_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user by username."""
        try:
            query = "SELECT * FROM users WHERE username = %(username)s"
            result = await self.db.execute(query, {"username": username})
            row = result.fetchone()
            
            if row:
                return dict(row)
            return None
            
        except Exception as e:
            self.logger.error("Failed to get user by username", username=username, error=str(e))
            raise DatabaseError(f"User retrieval failed: {str(e)}")
    
    async def update(self, user_id: str, data: Dict[str, Any]) -> bool:
        """Update user data."""
        try:
            # Build dynamic update query
            update_fields = []
            params = {"user_id": user_id}
            
            for field, value in data.items():
                if field != "user_id":  # Don't update ID
                    update_fields.append(f"{field} = %({field})s")
                    params[field] = value
            
            if not update_fields:
                return True  # Nothing to update
            
            query = f"""
                UPDATE users 
                SET {', '.join(update_fields)}, updated_at = NOW()
                WHERE user_id = %(user_id)s
            """
            
            result = await self.db.execute(query, params)
            return result.rowcount > 0
            
        except Exception as e:
            self.logger.error("Failed to update user", user_id=user_id, error=str(e))
            raise DatabaseError(f"User update failed: {str(e)}")
    
    async def delete(self, user_id: str) -> bool:
        """Delete user."""
        try:
            query = "DELETE FROM users WHERE user_id = %(user_id)s"
            result = await self.db.execute(query, {"user_id": user_id})
            return result.rowcount > 0
            
        except Exception as e:
            self.logger.error("Failed to delete user", user_id=user_id, error=str(e))
            raise DatabaseError(f"User deletion failed: {str(e)}")

class UserService:
    """User service with business logic."""
    
    def __init__(self, user_repository: UserRepository):
        self.user_repo = user_repository
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def create_user_profile(self, username: str, email: str, preferences: Dict[str, Any] = None) -> UserProfile:
        """Create a new user profile."""
        try:
            user_data = {
                "username": username,
                "email": email,
                "preferences": preferences or {}
            }
            
            user_id = await self.user_repo.create(user_data)
            
            # Get the created user
            user_dict = await self.user_repo.get_by_id(user_id)
            
            return UserProfile(
                user_id=user_dict["user_id"],
                username=user_dict["username"],
                email=user_dict["email"],
                preferences=user_dict["preferences"],
                created_at=user_dict["created_at"]
            )
            
        except Exception as e:
            self.logger.error("Failed to create user profile", error=str(e))
            raise

async def main():
    """Main function for user management."""
    # This would typically be in a separate main module
    pass

if __name__ == "__main__":
    asyncio.run(main())
'''
    
    try:
        # Initialize chunking service
        print("📦 Initializing Chunking Service...")
        chunking_service = ChunkingService()
        await chunking_service.initialize()
        print("✅ Chunking Service initialized successfully")
        
        # Test different purposes
        purposes = ["general", "code_review", "bug_detection", "documentation"]
        
        for purpose in purposes:
            print(f"\n🎯 Testing Purpose: {purpose.upper()}")
            print("-" * 40)
            
            # Test AST chunking
            print("🔧 AST Chunking:")
            ast_request = ChunkingRequest(
                content=complex_code,
                language="python",
                file_path="test_complex.py",
                purpose=purpose,
                force_agentic=False  # Force AST
            )
            
            ast_result = await chunking_service.chunk_content(ast_request)
            print(f"   Strategy: {ast_result.strategy_used.value}")
            print(f"   Chunks: {ast_result.chunk_count}")
            print(f"   Quality: {ast_result.quality_score:.3f}")
            print(f"   Time: {ast_result.processing_time:.3f}s")
            
            # Test Agentic chunking (if available)
            print("🧠 Agentic Chunking:")
            agentic_request = ChunkingRequest(
                content=complex_code,
                language="python",
                file_path="test_complex.py",
                purpose=purpose,
                force_agentic=True  # Force agentic
            )
            
            agentic_result = await chunking_service.chunk_content(agentic_request)
            print(f"   Strategy: {agentic_result.strategy_used.value}")
            print(f"   Chunks: {agentic_result.chunk_count}")
            print(f"   Quality: {agentic_result.quality_score:.3f}")
            print(f"   Time: {agentic_result.processing_time:.3f}s")
            
            # Compare results
            print("📊 Comparison:")
            chunk_diff = agentic_result.chunk_count - ast_result.chunk_count
            quality_diff = agentic_result.quality_score - ast_result.quality_score
            time_diff = agentic_result.processing_time - ast_result.processing_time
            
            print(f"   Chunk Count Difference: {chunk_diff:+d}")
            print(f"   Quality Difference: {quality_diff:+.3f}")
            print(f"   Time Difference: {time_diff:+.3f}s")
            
            if agentic_result.strategy_used.value == "agentic":
                print("   🎉 Agentic chunking successfully used!")
            else:
                print("   ⚠️  Agentic chunking fell back to AST")
        
        # Test performance stats
        print(f"\n📈 Performance Statistics:")
        stats = await chunking_service.get_performance_stats()
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.3f}")
            else:
                print(f"   {key}: {value}")
        
        # Cleanup
        await chunking_service.cleanup()
        print("\n🧹 Cleanup completed")
        
        print("\n🎉 Agentic chunking tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_agentic_vs_ast_chunking())
    if not success:
        sys.exit(1)
