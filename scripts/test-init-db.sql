-- Test Database Initialization Script for Cognify RAG System
-- This script tests the core schema without full complexity
-- Date: 2025-05-27

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ============================================================================
-- ENUM TYPES (SQLAlchemy compatible)
-- ============================================================================

-- User and authentication types
CREATE TYPE userrole AS ENUM ('ADMIN', 'USER', 'VIEWER');
CREATE TYPE userstatus AS ENUM ('ACTIVE', 'INACTIVE', 'SUSPENDED', 'PENDING_VERIFICATION');

-- Document and processing types
CREATE TYPE documenttype AS ENUM ('code', 'markdown', 'text', 'pdf', 'json', 'yaml', 'xml', 'csv', 'html', 'unknown');
CREATE TYPE documentstatus AS ENUM ('pending', 'processing', 'completed', 'failed', 'archived');
CREATE TYPE processingstage AS ENUM ('uploaded', 'parsing', 'chunking', 'embedding', 'indexing', 'completed', 'failed');

-- Query types
CREATE TYPE querytype AS ENUM ('semantic', 'keyword', 'hybrid', 'question', 'code', 'similarity');
CREATE TYPE querystatus AS ENUM ('pending', 'processing', 'completed', 'failed', 'cancelled');

-- Collection types
CREATE TYPE collectionstatus AS ENUM ('active', 'inactive', 'archived');
CREATE TYPE collectionvisibility AS ENUM ('private', 'internal', 'public');

-- ============================================================================
-- CORE TABLES
-- ============================================================================

-- Users table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(100) UNIQUE,  -- Made nullable
    full_name VARCHAR(255),
    password_hash VARCHAR(255) NOT NULL,
    role userrole DEFAULT 'USER',  -- Updated enum type and default
    status userstatus DEFAULT 'PENDING_VERIFICATION',  -- Updated enum type and default
    is_verified BOOLEAN DEFAULT false,
    email_verified_at TIMESTAMP WITH TIME ZONE,  -- Added missing column
    verification_token VARCHAR(255),
    reset_token VARCHAR(255),
    reset_token_expires_at TIMESTAMP WITH TIME ZONE,
    last_login_at TIMESTAMP WITH TIME ZONE,
    login_count INTEGER DEFAULT 0,
    failed_login_attempts INTEGER DEFAULT 0,  -- Added missing column
    locked_until TIMESTAMP WITH TIME ZONE,  -- Added missing column
    preferences JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    deleted_at TIMESTAMP WITH TIME ZONE
);

-- User sessions table for JWT token tracking
CREATE TABLE user_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_token VARCHAR(1000) NOT NULL UNIQUE,  -- Increased size for JWT tokens
    refresh_token VARCHAR(1000) UNIQUE,  -- Increased size for JWT tokens
    ip_address VARCHAR(45),  -- IPv6 max length
    user_agent TEXT,
    device_info JSONB,
    is_active BOOLEAN DEFAULT true,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    last_activity_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    logged_out_at TIMESTAMP WITH TIME ZONE,
    logout_reason VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    deleted_at TIMESTAMP WITH TIME ZONE
);

-- Collections table (simplified)
CREATE TABLE collections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    owner_id UUID NOT NULL REFERENCES users(id),
    status collectionstatus DEFAULT 'active',
    visibility collectionvisibility DEFAULT 'private',
    document_count INTEGER DEFAULT 0,
    total_size_bytes BIGINT DEFAULT 0,
    vector_collection_name VARCHAR(255),
    embedding_model VARCHAR(100),
    embedding_dimension INTEGER,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    deleted_at TIMESTAMP WITH TIME ZONE
);

-- Documents table (simplified)
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    collection_id UUID NOT NULL REFERENCES collections(id),
    owner_id UUID NOT NULL REFERENCES users(id),
    title VARCHAR(500) NOT NULL,
    content TEXT,
    file_path VARCHAR(1000),
    file_name VARCHAR(255),
    file_size BIGINT,
    file_hash VARCHAR(64),
    mime_type VARCHAR(100),
    document_type documenttype DEFAULT 'unknown',
    status documentstatus DEFAULT 'pending',
    processing_stage processingstage DEFAULT 'uploaded',
    chunk_count INTEGER DEFAULT 0,
    embedding_count INTEGER DEFAULT 0,
    processing_started_at TIMESTAMP WITH TIME ZONE,
    processing_completed_at TIMESTAMP WITH TIME ZONE,
    processing_error TEXT,
    quality_score FLOAT,
    language VARCHAR(10),
    metadata JSONB DEFAULT '{}',
    search_vector tsvector,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    deleted_at TIMESTAMP WITH TIME ZONE
);

-- Document chunks table (simplified)
CREATE TABLE document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id),
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    content_hash VARCHAR(64),
    start_position INTEGER,
    end_position INTEGER,
    chunk_type VARCHAR(50) DEFAULT 'text',
    token_count INTEGER,
    character_count INTEGER,
    language VARCHAR(10),
    quality_score FLOAT,
    coherence_score FLOAT,
    completeness_score FLOAT,
    vector_id VARCHAR(255) UNIQUE,
    embedding_model VARCHAR(100),
    embedding_dimension INTEGER,
    metadata JSONB DEFAULT '{}',
    search_vector tsvector,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    deleted_at TIMESTAMP WITH TIME ZONE
);

-- Queries table
CREATE TABLE queries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id),
    collection_id UUID REFERENCES collections(id),
    query_text TEXT NOT NULL,
    query_type querytype DEFAULT 'semantic',
    status querystatus DEFAULT 'pending',
    result_count INTEGER DEFAULT 0,
    processing_time_ms INTEGER,
    embedding_time_ms INTEGER,
    search_time_ms INTEGER,
    total_time_ms INTEGER,
    similarity_threshold FLOAT DEFAULT 0.7,
    max_results INTEGER DEFAULT 10,
    filters JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    deleted_at TIMESTAMP WITH TIME ZONE  -- Added missing column
);

-- Query results table
CREATE TABLE query_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_id UUID NOT NULL REFERENCES queries(id) ON DELETE CASCADE,
    document_id UUID REFERENCES documents(id),
    chunk_id UUID REFERENCES document_chunks(id),
    similarity_score FLOAT NOT NULL,
    rank_position INTEGER NOT NULL,
    content_snippet TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================================

-- Users indexes
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_status ON users(status);
CREATE INDEX idx_users_role ON users(role);
CREATE INDEX idx_users_created_at ON users(created_at);

-- User sessions indexes
CREATE INDEX idx_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX idx_user_sessions_session_token ON user_sessions(session_token);
CREATE INDEX idx_user_sessions_refresh_token ON user_sessions(refresh_token);
CREATE INDEX idx_user_sessions_expires_at ON user_sessions(expires_at);
CREATE INDEX idx_user_sessions_is_active ON user_sessions(is_active);
CREATE INDEX idx_user_sessions_last_activity ON user_sessions(last_activity_at);

-- Collections indexes
CREATE INDEX idx_collections_owner_id ON collections(owner_id);
CREATE INDEX idx_collections_name ON collections(name);
CREATE INDEX idx_collections_status ON collections(status);
CREATE INDEX idx_collections_created_at ON collections(created_at);

-- Documents indexes
CREATE INDEX idx_documents_collection_id ON documents(collection_id);
CREATE INDEX idx_documents_owner_id ON documents(owner_id);
CREATE INDEX idx_documents_status ON documents(status);
CREATE INDEX idx_documents_type ON documents(document_type);
CREATE INDEX idx_documents_file_hash ON documents(file_hash);
CREATE INDEX idx_documents_search_vector ON documents USING gin(search_vector);
CREATE INDEX idx_documents_created_at ON documents(created_at);

-- Document chunks indexes
CREATE INDEX idx_document_chunks_document_id ON document_chunks(document_id);
CREATE INDEX idx_document_chunks_vector_id ON document_chunks(vector_id);
CREATE INDEX idx_document_chunks_embedding_model ON document_chunks(embedding_model);
CREATE INDEX idx_document_chunks_quality_score ON document_chunks(quality_score);
CREATE INDEX idx_document_chunks_search_vector ON document_chunks USING gin(search_vector);
CREATE INDEX idx_document_chunks_created_at ON document_chunks(created_at);

-- Queries indexes
CREATE INDEX idx_queries_user_id ON queries(user_id);
CREATE INDEX idx_queries_collection_id ON queries(collection_id);
CREATE INDEX idx_queries_status ON queries(status);
CREATE INDEX idx_queries_type ON queries(query_type);
CREATE INDEX idx_queries_created_at ON queries(created_at);

-- Query results indexes
CREATE INDEX idx_query_results_query_id ON query_results(query_id);
CREATE INDEX idx_query_results_document_id ON query_results(document_id);
CREATE INDEX idx_query_results_chunk_id ON query_results(chunk_id);
CREATE INDEX idx_query_results_similarity_score ON query_results(similarity_score);
CREATE INDEX idx_query_results_rank_position ON query_results(rank_position);

-- ============================================================================
-- FUNCTIONS AND TRIGGERS
-- ============================================================================

-- Generic updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Function for full-text search
CREATE OR REPLACE FUNCTION create_search_vector(title TEXT, content TEXT, metadata JSONB DEFAULT '{}')
RETURNS tsvector AS $$
BEGIN
    RETURN setweight(to_tsvector('english', COALESCE(title, '')), 'A') ||
           setweight(to_tsvector('english', COALESCE(content, '')), 'B') ||
           setweight(to_tsvector('english', COALESCE(metadata->>'keywords', '')), 'C');
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- ============================================================================
-- TRIGGERS
-- ============================================================================

-- Updated_at triggers for all tables
CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_sessions_updated_at
    BEFORE UPDATE ON user_sessions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_collections_updated_at
    BEFORE UPDATE ON collections
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_documents_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_document_chunks_updated_at
    BEFORE UPDATE ON document_chunks
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_queries_updated_at
    BEFORE UPDATE ON queries
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Search vector triggers
CREATE OR REPLACE FUNCTION update_document_search_vector()
RETURNS TRIGGER AS $$
BEGIN
    NEW.search_vector := create_search_vector(NEW.title, NEW.content, NEW.metadata);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_documents_search_vector
    BEFORE INSERT OR UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION update_document_search_vector();

CREATE OR REPLACE FUNCTION update_chunk_search_vector()
RETURNS TRIGGER AS $$
BEGIN
    NEW.search_vector := to_tsvector('english', COALESCE(NEW.content, ''));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_document_chunks_search_vector
    BEFORE INSERT OR UPDATE ON document_chunks
    FOR EACH ROW EXECUTE FUNCTION update_chunk_search_vector();

-- Final success message
SELECT 'Cognify test database initialization completed successfully!' as status;
