-- Initialize test database for Cognify
-- This script sets up the test database with proper permissions and extensions

-- Create test database if it doesn't exist
SELECT 'CREATE DATABASE cognify_test_db'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'cognify_test_db')\gexec

-- Connect to the test database
\c cognify_test_db;

-- Create necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Create test user with proper permissions
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_user WHERE usename = 'cognify_test') THEN
        CREATE USER cognify_test WITH PASSWORD 'cognify_test_password';
    END IF;
END
$$;

-- Grant permissions to test user
GRANT ALL PRIVILEGES ON DATABASE cognify_test_db TO cognify_test;
GRANT ALL ON SCHEMA public TO cognify_test;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO cognify_test;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO cognify_test;

-- Set default privileges for future objects
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO cognify_test;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO cognify_test;

-- Create test-specific schemas if needed
CREATE SCHEMA IF NOT EXISTS test_data;
GRANT ALL ON SCHEMA test_data TO cognify_test;

-- Log successful initialization
INSERT INTO pg_catalog.pg_stat_statements_info (dealloc) VALUES (0) ON CONFLICT DO NOTHING;

-- Test connection
SELECT 'Test database initialized successfully' AS status;
