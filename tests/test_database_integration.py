"""
Integration tests for database operations.
"""
import pytest
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text

from app.models.base import Base
from app.models.users import User
from app.models.documents import Document
from app.models.collections import Collection
from app.models.queries import Query
from app.core.config import get_settings


@pytest.fixture(scope="function")
async def test_engine():
    """Create test database engine."""
    settings = get_settings()
    # Use Docker PostgreSQL for integration tests
    test_db_url = "postgresql+asyncpg://cognify_test:test_password@localhost:5433/cognify_test"
    engine = create_async_engine(
        test_db_url,
        echo=settings.DATABASE_ECHO,
        future=True,
        pool_size=5,
        max_overflow=10
    )

    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    # Clean up
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


@pytest.fixture
async def db_session(test_engine):
    """Create test database session."""
    async_session = sessionmaker(
        test_engine, class_=AsyncSession, expire_on_commit=False
    )

    async with async_session() as session:
        try:
            yield session
        finally:
            await session.rollback()
            await session.close()


class TestDatabaseConnection:
    """Test database connection and basic operations."""

    async def test_database_connection(self, test_engine):
        """Test database connection."""
        async with test_engine.connect() as conn:
            result = await conn.execute(text("SELECT 1"))
            assert result.scalar() == 1

    async def test_create_tables(self, test_engine):
        """Test table creation."""
        async with test_engine.connect() as conn:
            # Check if tables exist
            result = await conn.execute(text("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public'
            """))
            tables = [row[0] for row in result.fetchall()]

            expected_tables = ['users', 'documents', 'collections', 'queries']
            for table in expected_tables:
                assert table in tables


class TestUserModel:
    """Test User model operations."""

    async def test_create_user(self, db_session):
        """Test creating a user."""
        user = User(
            username="testuser",
            email="test@example.com",
            hashed_password="hashed_password_123"
        )

        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)

        assert user.id is not None
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.is_active is True

    async def test_user_relationships(self, db_session):
        """Test user relationships."""
        # Create user
        user = User(
            username="testuser2",
            email="test2@example.com",
            hashed_password="hashed_password_123"
        )
        db_session.add(user)
        await db_session.flush()

        # Create document
        document = Document(
            title="Test Document",
            content="Test content",
            file_type="text",
            user_id=user.id
        )
        db_session.add(document)

        # Create collection
        collection = Collection(
            name="Test Collection",
            description="Test description",
            user_id=user.id
        )
        db_session.add(collection)

        await db_session.commit()
        await db_session.refresh(user)

        # Test relationships
        assert len(user.documents) == 1
        assert len(user.collections) == 1
        assert user.documents[0].title == "Test Document"
        assert user.collections[0].name == "Test Collection"


class TestDocumentModel:
    """Test Document model operations."""

    async def test_create_document(self, db_session):
        """Test creating a document."""
        # Create user first
        user = User(
            username="docuser",
            email="doc@example.com",
            hashed_password="hashed_password_123"
        )
        db_session.add(user)
        await db_session.flush()

        # Create document
        document = Document(
            title="Test Document",
            content="This is test content for the document.",
            file_type="text",
            file_size=100,
            user_id=user.id
        )

        db_session.add(document)
        await db_session.commit()
        await db_session.refresh(document)

        assert document.id is not None
        assert document.title == "Test Document"
        assert document.file_type == "text"
        assert document.file_size == 100
        assert document.user_id == user.id

    async def test_document_metadata(self, db_session):
        """Test document with metadata."""
        # Create user
        user = User(
            username="metauser",
            email="meta@example.com",
            hashed_password="hashed_password_123"
        )
        db_session.add(user)
        await db_session.flush()

        # Create document with metadata
        document = Document(
            title="Document with Metadata",
            content="Content with metadata",
            file_type="python",
            metadata={
                "language": "python",
                "functions": ["main", "helper"],
                "classes": ["MyClass"],
                "imports": ["os", "sys"]
            },
            user_id=user.id
        )

        db_session.add(document)
        await db_session.commit()
        await db_session.refresh(document)

        assert document.metadata["language"] == "python"
        assert "main" in document.metadata["functions"]
        assert "MyClass" in document.metadata["classes"]


class TestCollectionModel:
    """Test Collection model operations."""

    async def test_create_collection(self, db_session):
        """Test creating a collection."""
        # Create user
        user = User(
            username="colluser",
            email="coll@example.com",
            hashed_password="hashed_password_123"
        )
        db_session.add(user)
        await db_session.flush()

        # Create collection
        collection = Collection(
            name="My Collection",
            description="A test collection",
            user_id=user.id
        )

        db_session.add(collection)
        await db_session.commit()
        await db_session.refresh(collection)

        assert collection.id is not None
        assert collection.name == "My Collection"
        assert collection.description == "A test collection"
        assert collection.user_id == user.id
        assert collection.is_public is False


class TestQueryModel:
    """Test Query model operations."""

    async def test_create_query(self, db_session):
        """Test creating a query."""
        # Create user
        user = User(
            username="queryuser",
            email="query@example.com",
            hashed_password="hashed_password_123"
        )
        db_session.add(user)
        await db_session.flush()

        # Create query
        query = Query(
            query_text="How to implement async functions?",
            user_id=user.id,
            metadata={
                "search_type": "semantic",
                "filters": {"language": "python"}
            }
        )

        db_session.add(query)
        await db_session.commit()
        await db_session.refresh(query)

        assert query.id is not None
        assert query.query_text == "How to implement async functions?"
        assert query.user_id == user.id
        assert query.metadata["search_type"] == "semantic"


class TestDatabasePerformance:
    """Test database performance."""

    async def test_bulk_insert_users(self, db_session):
        """Test bulk inserting users."""
        users = []
        for i in range(100):
            user = User(
                username=f"user_{i}",
                email=f"user_{i}@example.com",
                hashed_password="hashed_password_123"
            )
            users.append(user)

        db_session.add_all(users)
        await db_session.commit()

        # Verify count
        result = await db_session.execute(text("SELECT COUNT(*) FROM users"))
        count = result.scalar()
        assert count >= 100

    async def test_query_performance(self, db_session):
        """Test query performance."""
        # Create test data
        user = User(
            username="perfuser",
            email="perf@example.com",
            hashed_password="hashed_password_123"
        )
        db_session.add(user)
        await db_session.flush()

        # Create multiple documents
        documents = []
        for i in range(50):
            doc = Document(
                title=f"Document {i}",
                content=f"Content for document {i}",
                file_type="text",
                user_id=user.id
            )
            documents.append(doc)

        db_session.add_all(documents)
        await db_session.commit()

        # Test query performance
        import time
        start_time = time.time()

        result = await db_session.execute(
            text("SELECT * FROM documents WHERE user_id = :user_id"),
            {"user_id": user.id}
        )
        docs = result.fetchall()

        end_time = time.time()
        query_time = end_time - start_time

        assert len(docs) == 50
        assert query_time < 1.0  # Should complete within 1 second
