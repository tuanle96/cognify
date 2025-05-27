"""
Simple database integration tests.
"""
import pytest
import asyncio
import asyncpg
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text


@pytest.mark.asyncio
async def test_direct_asyncpg_connection():
    """Test direct asyncpg connection to Docker PostgreSQL."""
    try:
        conn = await asyncpg.connect(
            'postgresql://cognify_test:test_password@localhost:5433/cognify_test'
        )
        result = await conn.fetchval('SELECT 1')
        assert result == 1
        print(f"✅ Direct asyncpg connection successful! Result: {result}")
        await conn.close()
    except Exception as e:
        pytest.fail(f"Direct asyncpg connection failed: {e}")


@pytest.mark.asyncio
async def test_sqlalchemy_async_connection():
    """Test SQLAlchemy async connection to Docker PostgreSQL."""
    try:
        engine = create_async_engine(
            "postgresql+asyncpg://cognify_test:test_password@localhost:5433/cognify_test",
            echo=False,
            future=True
        )
        
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT 1"))
            value = result.scalar()
            assert value == 1
            print(f"✅ SQLAlchemy async connection successful! Result: {value}")
        
        await engine.dispose()
    except Exception as e:
        pytest.fail(f"SQLAlchemy async connection failed: {e}")


@pytest.mark.asyncio
async def test_database_operations():
    """Test basic database operations."""
    try:
        engine = create_async_engine(
            "postgresql+asyncpg://cognify_test:test_password@localhost:5433/cognify_test",
            echo=False,
            future=True
        )
        
        async with engine.connect() as conn:
            # Test creating a simple table
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS test_table (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(100),
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """))
            
            # Test inserting data
            await conn.execute(text("""
                INSERT INTO test_table (name) VALUES ('test_user')
            """))
            
            # Test querying data
            result = await conn.execute(text("""
                SELECT name FROM test_table WHERE name = 'test_user'
            """))
            name = result.scalar()
            assert name == 'test_user'
            print(f"✅ Database operations successful! Retrieved: {name}")
            
            # Clean up
            await conn.execute(text("DROP TABLE IF EXISTS test_table"))
            await conn.commit()
        
        await engine.dispose()
    except Exception as e:
        pytest.fail(f"Database operations failed: {e}")


@pytest.mark.asyncio
async def test_database_extensions():
    """Test PostgreSQL extensions."""
    try:
        engine = create_async_engine(
            "postgresql+asyncpg://cognify_test:test_password@localhost:5433/cognify_test",
            echo=False,
            future=True
        )
        
        async with engine.connect() as conn:
            # Test UUID extension
            result = await conn.execute(text("SELECT uuid_generate_v4()"))
            uuid_val = result.scalar()
            assert uuid_val is not None
            print(f"✅ UUID extension working! Generated: {uuid_val}")
            
            # Test JSON operations
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS test_json (
                    id SERIAL PRIMARY KEY,
                    data JSONB
                )
            """))
            
            await conn.execute(text("""
                INSERT INTO test_json (data) VALUES ('{"test": "value", "number": 42}')
            """))
            
            result = await conn.execute(text("""
                SELECT data->>'test' FROM test_json WHERE data->>'number' = '42'
            """))
            json_value = result.scalar()
            assert json_value == 'value'
            print(f"✅ JSONB operations working! Retrieved: {json_value}")
            
            # Clean up
            await conn.execute(text("DROP TABLE IF EXISTS test_json"))
            await conn.commit()
        
        await engine.dispose()
    except Exception as e:
        pytest.fail(f"Database extensions test failed: {e}")
