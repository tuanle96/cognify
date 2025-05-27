-- Database initialization script for Cognify RAG System
-- This script sets up the initial database structure and configurations

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";
CREATE EXTENSION IF NOT EXISTS "btree_gist";

-- Create additional databases for different environments
CREATE DATABASE cognify_dev;
CREATE DATABASE cognify_test;
CREATE DATABASE cognify_staging;

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE cognify_db TO cognify;
GRANT ALL PRIVILEGES ON DATABASE cognify_dev TO cognify;
GRANT ALL PRIVILEGES ON DATABASE cognify_test TO cognify;
GRANT ALL PRIVILEGES ON DATABASE cognify_staging TO cognify;

-- Connect to main database and set up schemas
\c cognify_db;

-- Create schemas for better organization
CREATE SCHEMA IF NOT EXISTS auth;
CREATE SCHEMA IF NOT EXISTS documents;
CREATE SCHEMA IF NOT EXISTS analytics;
CREATE SCHEMA IF NOT EXISTS system;

-- Grant schema permissions
GRANT ALL ON SCHEMA auth TO cognify;
GRANT ALL ON SCHEMA documents TO cognify;
GRANT ALL ON SCHEMA analytics TO cognify;
GRANT ALL ON SCHEMA system TO cognify;

-- Create custom types
CREATE TYPE user_role AS ENUM ('admin', 'user', 'viewer');
CREATE TYPE user_status AS ENUM ('active', 'inactive', 'suspended', 'pending_verification');
CREATE TYPE document_type AS ENUM ('code', 'markdown', 'text', 'pdf', 'json', 'yaml', 'xml', 'csv', 'html', 'unknown');
CREATE TYPE document_status AS ENUM ('pending', 'processing', 'completed', 'failed', 'archived');
CREATE TYPE processing_stage AS ENUM ('uploaded', 'parsing', 'chunking', 'embedding', 'indexing', 'completed', 'failed');
CREATE TYPE query_type AS ENUM ('semantic', 'keyword', 'hybrid', 'question', 'code', 'similarity');
CREATE TYPE query_status AS ENUM ('pending', 'processing', 'completed', 'failed', 'cancelled');

-- Create indexes for better performance
-- Note: Actual table indexes will be created by Alembic migrations

-- Create functions for common operations
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create function to generate short IDs
CREATE OR REPLACE FUNCTION generate_short_id(length INTEGER DEFAULT 8)
RETURNS TEXT AS $$
DECLARE
    chars TEXT := 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';
    result TEXT := '';
    i INTEGER := 0;
BEGIN
    FOR i IN 1..length LOOP
        result := result || substr(chars, floor(random() * length(chars) + 1)::INTEGER, 1);
    END LOOP;
    RETURN result;
END;
$$ LANGUAGE plpgsql;

-- Create function for full-text search
CREATE OR REPLACE FUNCTION create_search_vector(title TEXT, content TEXT, metadata JSONB DEFAULT '{}')
RETURNS tsvector AS $$
BEGIN
    RETURN setweight(to_tsvector('english', COALESCE(title, '')), 'A') ||
           setweight(to_tsvector('english', COALESCE(content, '')), 'B') ||
           setweight(to_tsvector('english', COALESCE(metadata->>'keywords', '')), 'C');
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Create function for similarity search
CREATE OR REPLACE FUNCTION cosine_similarity(vec1 FLOAT[], vec2 FLOAT[])
RETURNS FLOAT AS $$
DECLARE
    dot_product FLOAT := 0;
    norm1 FLOAT := 0;
    norm2 FLOAT := 0;
    i INTEGER;
BEGIN
    IF array_length(vec1, 1) != array_length(vec2, 1) THEN
        RETURN 0;
    END IF;
    
    FOR i IN 1..array_length(vec1, 1) LOOP
        dot_product := dot_product + (vec1[i] * vec2[i]);
        norm1 := norm1 + (vec1[i] * vec1[i]);
        norm2 := norm2 + (vec2[i] * vec2[i]);
    END LOOP;
    
    IF norm1 = 0 OR norm2 = 0 THEN
        RETURN 0;
    END IF;
    
    RETURN dot_product / (sqrt(norm1) * sqrt(norm2));
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Create materialized view for analytics
-- This will be created after tables are set up by migrations

-- Set up row-level security policies
-- These will be created after tables are set up by migrations

-- Create initial admin user (will be handled by application)
-- INSERT INTO users (email, username, full_name, password_hash, role, status, is_verified)
-- VALUES ('admin@cognify.local', 'admin', 'System Administrator', '$2b$12$...', 'admin', 'active', true);

-- Log successful initialization
INSERT INTO system.initialization_log (message, created_at) 
VALUES ('Database initialized successfully', CURRENT_TIMESTAMP)
ON CONFLICT DO NOTHING;
