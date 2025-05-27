-- Complete Database Initialization Script for Cognify RAG System
-- This script consolidates all migrations into a single initialization
-- Date: 2025-01-26

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";
CREATE EXTENSION IF NOT EXISTS "btree_gist";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Create database user if not exists
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'cognify_app') THEN
        CREATE ROLE cognify_app WITH LOGIN PASSWORD 'cognify_secure_password_2025';
    END IF;
END
$$;

-- Grant database permissions
GRANT ALL PRIVILEGES ON DATABASE cognify_production TO cognify_app;

-- Create schemas for better organization
CREATE SCHEMA IF NOT EXISTS auth;
CREATE SCHEMA IF NOT EXISTS documents;
CREATE SCHEMA IF NOT EXISTS analytics;
CREATE SCHEMA IF NOT EXISTS system;

-- Grant schema permissions
GRANT ALL ON SCHEMA auth TO cognify_app;
GRANT ALL ON SCHEMA documents TO cognify_app;
GRANT ALL ON SCHEMA analytics TO cognify_app;
GRANT ALL ON SCHEMA system TO cognify_app;
GRANT ALL ON SCHEMA public TO cognify_app;

-- ============================================================================
-- ENUM TYPES
-- ============================================================================

-- User and authentication types (using SQLAlchemy compatible names)
CREATE TYPE userrole AS ENUM ('ADMIN', 'USER', 'VIEWER');
CREATE TYPE userstatus AS ENUM ('ACTIVE', 'INACTIVE', 'SUSPENDED', 'PENDING_VERIFICATION');

-- Document and processing types
CREATE TYPE document_type AS ENUM ('code', 'markdown', 'text', 'pdf', 'json', 'yaml', 'xml', 'csv', 'html', 'unknown');
CREATE TYPE document_status AS ENUM ('pending', 'processing', 'completed', 'failed', 'archived');
CREATE TYPE processing_stage AS ENUM ('uploaded', 'parsing', 'chunking', 'embedding', 'indexing', 'completed', 'failed');

-- Query types
CREATE TYPE query_type AS ENUM ('semantic', 'keyword', 'hybrid', 'question', 'code', 'similarity');
CREATE TYPE query_status AS ENUM ('pending', 'processing', 'completed', 'failed', 'cancelled');

-- Multi-tenant types
CREATE TYPE organization_role AS ENUM ('owner', 'admin', 'member');
CREATE TYPE workspace_role AS ENUM ('admin', 'editor', 'viewer');
CREATE TYPE member_status AS ENUM ('active', 'inactive', 'pending', 'suspended');
CREATE TYPE visibility_type AS ENUM ('private', 'internal', 'public');

-- Settings system types
CREATE TYPE setting_category AS ENUM (
    'llm_chat', 'llm_completion', 'llm_embedding', 'llm_vision', 'llm_function_calling',
    'vector_database', 'security', 'performance', 'features', 'notifications', 'integrations'
);
CREATE TYPE setting_data_type AS ENUM ('string', 'integer', 'float', 'boolean', 'json', 'encrypted');
CREATE TYPE setting_scope AS ENUM ('global', 'organization', 'workspace', 'user');

-- API Keys types
CREATE TYPE cognify_api_key_status AS ENUM ('active', 'inactive', 'revoked', 'expired');

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

-- Organizations table (top-level tenants)
CREATE TABLE organizations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    slug VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    plan_type VARCHAR(50) NOT NULL DEFAULT 'free',
    status VARCHAR(50) NOT NULL DEFAULT 'active',
    settings JSONB DEFAULT '{}',
    billing_email VARCHAR(255),
    billing_address JSONB,
    max_workspaces INTEGER DEFAULT 1,
    max_users INTEGER DEFAULT 5,
    max_documents INTEGER DEFAULT 100,
    max_storage_gb INTEGER DEFAULT 1,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    deleted_at TIMESTAMP WITH TIME ZONE NULL,
    CONSTRAINT organizations_name_length CHECK (length(name) >= 2),
    CONSTRAINT organizations_slug_format CHECK (slug ~ '^[a-z0-9-]+$')
);

-- Workspaces table (team/project level)
CREATE TABLE workspaces (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    slug VARCHAR(100) NOT NULL,
    description TEXT,
    visibility visibility_type DEFAULT 'private',
    settings JSONB DEFAULT '{}',
    max_collections INTEGER DEFAULT 10,
    max_documents_per_collection INTEGER DEFAULT 1000,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    deleted_at TIMESTAMP WITH TIME ZONE NULL,
    CONSTRAINT workspaces_name_length CHECK (length(name) >= 2),
    CONSTRAINT workspaces_slug_format CHECK (slug ~ '^[a-z0-9-]+$'),
    UNIQUE(organization_id, slug)
);

-- Organization membership
CREATE TABLE organization_members (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role organization_role NOT NULL DEFAULT 'member',
    status member_status DEFAULT 'active',
    permissions JSONB DEFAULT '{}',
    invited_by UUID REFERENCES users(id),
    invited_at TIMESTAMP WITH TIME ZONE,
    joined_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_access_at TIMESTAMP WITH TIME ZONE,
    access_count INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(organization_id, user_id)
);

-- Workspace membership
CREATE TABLE workspace_members (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role workspace_role NOT NULL DEFAULT 'viewer',
    status member_status DEFAULT 'active',
    can_read BOOLEAN DEFAULT true,
    can_write BOOLEAN DEFAULT false,
    can_delete BOOLEAN DEFAULT false,
    can_share BOOLEAN DEFAULT false,
    can_admin BOOLEAN DEFAULT false,
    permissions JSONB DEFAULT '{}',
    invited_by UUID REFERENCES users(id),
    invited_at TIMESTAMP WITH TIME ZONE,
    joined_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_access_at TIMESTAMP WITH TIME ZONE,
    access_count INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(workspace_id, user_id)
);

-- Collections table
CREATE TABLE collections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL,
    organization_id UUID,
    tenant_id VARCHAR(255) NOT NULL DEFAULT 'default',
    visibility visibility_type DEFAULT 'private',
    name VARCHAR(255) NOT NULL,
    description TEXT,
    owner_id UUID NOT NULL REFERENCES users(id),
    is_public BOOLEAN DEFAULT false,
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

-- Shared contents table for deduplication
CREATE TABLE shared_contents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content_hash VARCHAR(64) NOT NULL UNIQUE,
    content TEXT NOT NULL,
    content_size INTEGER NOT NULL,
    content_type VARCHAR(100),
    language VARCHAR(10),
    encoding VARCHAR(50),
    is_processed BOOLEAN NOT NULL DEFAULT FALSE,
    processing_started_at TIMESTAMP WITH TIME ZONE,
    processing_completed_at TIMESTAMP WITH TIME ZONE,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    embedding_count INTEGER NOT NULL DEFAULT 0,
    processing_quality_score FLOAT,
    chunking_quality_score FLOAT,
    processing_config JSONB,
    reference_count INTEGER NOT NULL DEFAULT 0,
    last_accessed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    deleted_at TIMESTAMP WITH TIME ZONE,
    created_by UUID,
    updated_by UUID
);

-- Documents table
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL,
    organization_id UUID,
    visibility visibility_type DEFAULT 'private',
    shared_content_id UUID,
    collection_id UUID NOT NULL,
    owner_id UUID NOT NULL REFERENCES users(id),
    title VARCHAR(500) NOT NULL,
    content TEXT,
    file_path VARCHAR(1000),
    file_name VARCHAR(255),
    file_size BIGINT,
    file_hash VARCHAR(64),
    mime_type VARCHAR(100),
    document_type document_type DEFAULT 'unknown',
    status document_status DEFAULT 'pending',
    processing_stage processing_stage DEFAULT 'uploaded',
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

-- Shared content chunks table
CREATE TABLE shared_content_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    shared_content_id UUID NOT NULL,
    chunk_index INTEGER NOT NULL,
    content_hash VARCHAR(64) NOT NULL,
    content TEXT NOT NULL,
    start_position INTEGER,
    end_position INTEGER,
    chunk_type VARCHAR(50),
    language VARCHAR(10),
    quality_score FLOAT,
    coherence_score FLOAT,
    completeness_score FLOAT,
    vector_id VARCHAR(255) UNIQUE,
    embedding_model VARCHAR(100),
    embedding_dimension INTEGER,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    deleted_at TIMESTAMP WITH TIME ZONE,
    created_by UUID,
    updated_by UUID
);

-- Document chunks table
CREATE TABLE document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL,
    organization_id UUID,
    vector_metadata JSONB DEFAULT '{}',
    document_id UUID NOT NULL,
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
    query_type query_type DEFAULT 'semantic',
    status query_status DEFAULT 'pending',
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
-- SETTINGS SYSTEM TABLES
-- ============================================================================

-- Main settings table
CREATE TABLE system_settings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    key VARCHAR(255) NOT NULL,
    category setting_category NOT NULL,
    scope setting_scope NOT NULL DEFAULT 'global',
    name VARCHAR(255) NOT NULL,
    description TEXT,
    data_type setting_data_type NOT NULL DEFAULT 'string',
    value TEXT,
    default_value TEXT,
    validation_rules JSONB DEFAULT '{}',
    is_required BOOLEAN DEFAULT false,
    is_sensitive BOOLEAN DEFAULT false,
    organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    workspace_id UUID REFERENCES workspaces(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    created_by UUID REFERENCES users(id),
    updated_by UUID REFERENCES users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT unique_setting_scope UNIQUE (key, scope, organization_id, workspace_id, user_id),
    CONSTRAINT valid_scope_references CHECK (
        (scope = 'global' AND organization_id IS NULL AND workspace_id IS NULL AND user_id IS NULL) OR
        (scope = 'organization' AND organization_id IS NOT NULL AND workspace_id IS NULL AND user_id IS NULL) OR
        (scope = 'workspace' AND workspace_id IS NOT NULL AND user_id IS NULL) OR
        (scope = 'user' AND user_id IS NOT NULL)
    )
);

-- Settings history for audit trail
CREATE TABLE settings_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    setting_id UUID NOT NULL REFERENCES system_settings(id) ON DELETE CASCADE,
    old_value TEXT,
    new_value TEXT,
    change_reason TEXT,
    changed_by UUID REFERENCES users(id),
    changed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    ip_address INET,
    user_agent TEXT,
    metadata JSONB DEFAULT '{}'
);

-- ============================================================================
-- COGNIFY API KEYS TABLES
-- ============================================================================

-- Cognify API Keys table
CREATE TABLE cognify_api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    workspace_id UUID REFERENCES workspaces(id) ON DELETE CASCADE,
    key_name VARCHAR(255) NOT NULL,
    key_description TEXT,
    key_prefix VARCHAR(20) NOT NULL,
    key_hash VARCHAR(64) NOT NULL UNIQUE,
    permissions JSONB DEFAULT '{}',
    rate_limits JSONB DEFAULT '{}',
    status cognify_api_key_status NOT NULL DEFAULT 'active',
    last_used_at TIMESTAMP WITH TIME ZONE,
    usage_count INTEGER DEFAULT 0,
    expires_at TIMESTAMP WITH TIME ZONE,
    created_from_ip INET,
    last_used_from_ip INET,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT unique_user_key_name UNIQUE (user_id, key_name)
);

-- API key usage tracking
CREATE TABLE cognify_api_key_usage (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    api_key_id UUID NOT NULL REFERENCES cognify_api_keys(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    endpoint VARCHAR(255) NOT NULL,
    method VARCHAR(10) NOT NULL,
    status_code INTEGER NOT NULL,
    request_id VARCHAR(255),
    user_agent TEXT,
    ip_address INET,
    response_time_ms INTEGER,
    request_size_bytes INTEGER,
    response_size_bytes INTEGER,
    organization_id UUID REFERENCES organizations(id),
    workspace_id UUID REFERENCES workspaces(id),
    used_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- ============================================================================
-- AUDIT AND ANALYTICS TABLES
-- ============================================================================

-- Data access audit table
CREATE TABLE data_access_audit (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id),
    workspace_id UUID REFERENCES workspaces(id),
    organization_id UUID REFERENCES organizations(id),
    action VARCHAR(50) NOT NULL,
    resource_type VARCHAR(50) NOT NULL,
    resource_id UUID NOT NULL,
    ip_address INET,
    user_agent TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- System initialization log
CREATE TABLE IF NOT EXISTS system.initialization_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    message TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================================================
-- FOREIGN KEY CONSTRAINTS
-- ============================================================================

-- Add foreign key constraints for collections
ALTER TABLE collections ADD CONSTRAINT fk_collections_workspace
    FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE;
ALTER TABLE collections ADD CONSTRAINT fk_collections_organization
    FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE CASCADE;

-- Add foreign key constraints for documents
ALTER TABLE documents ADD CONSTRAINT fk_documents_workspace
    FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE;
ALTER TABLE documents ADD CONSTRAINT fk_documents_organization
    FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE CASCADE;
ALTER TABLE documents ADD CONSTRAINT fk_documents_shared_content
    FOREIGN KEY (shared_content_id) REFERENCES shared_contents(id) ON DELETE SET NULL;

-- Add foreign key constraints for document_chunks
ALTER TABLE document_chunks ADD CONSTRAINT fk_document_chunks_workspace
    FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE;
ALTER TABLE document_chunks ADD CONSTRAINT fk_document_chunks_organization
    FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE CASCADE;

-- Add foreign key constraints for shared_content_chunks
ALTER TABLE shared_content_chunks ADD CONSTRAINT fk_shared_content_chunks_shared_content
    FOREIGN KEY (shared_content_id) REFERENCES shared_contents(id) ON DELETE CASCADE;

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

-- Organizations indexes
CREATE INDEX idx_organizations_slug ON organizations(slug);
CREATE INDEX idx_organizations_status ON organizations(status) WHERE deleted_at IS NULL;
CREATE INDEX idx_organizations_plan_type ON organizations(plan_type);

-- Workspaces indexes
CREATE INDEX idx_workspaces_org_id ON workspaces(organization_id);
CREATE INDEX idx_workspaces_slug ON workspaces(organization_id, slug);
CREATE INDEX idx_workspaces_visibility ON workspaces(visibility) WHERE deleted_at IS NULL;

-- Organization members indexes
CREATE INDEX idx_org_members_org_id ON organization_members(organization_id);
CREATE INDEX idx_org_members_user_id ON organization_members(user_id);
CREATE INDEX idx_org_members_role ON organization_members(role);
CREATE INDEX idx_org_members_status ON organization_members(status);

-- Workspace members indexes
CREATE INDEX idx_workspace_members_workspace_id ON workspace_members(workspace_id);
CREATE INDEX idx_workspace_members_user_id ON workspace_members(user_id);
CREATE INDEX idx_workspace_members_role ON workspace_members(role);
CREATE INDEX idx_workspace_members_status ON workspace_members(status);

-- Collections indexes
CREATE INDEX idx_collections_workspace_id ON collections(workspace_id);
CREATE INDEX idx_collections_tenant_id ON collections(tenant_id);
CREATE INDEX idx_collections_visibility ON collections(visibility);
CREATE INDEX idx_collections_org_workspace ON collections(organization_id, workspace_id);
CREATE INDEX idx_collections_owner_id ON collections(owner_id);
CREATE INDEX idx_collections_name ON collections(name);
CREATE INDEX idx_collections_created_at ON collections(created_at);

-- Shared contents indexes
CREATE INDEX idx_shared_content_hash_processed ON shared_contents(content_hash, is_processed);
CREATE INDEX idx_shared_content_type_language ON shared_contents(content_type, language);
CREATE INDEX idx_shared_content_size_refs ON shared_contents(content_size, reference_count);
CREATE INDEX idx_shared_content_created_at ON shared_contents(created_at);
CREATE INDEX idx_shared_content_deleted_at ON shared_contents(deleted_at);

-- Documents indexes
CREATE INDEX idx_documents_workspace_id ON documents(workspace_id);
CREATE INDEX idx_documents_visibility ON documents(visibility);
CREATE INDEX idx_documents_org_workspace ON documents(organization_id, workspace_id);
CREATE INDEX idx_documents_collection_workspace ON documents(collection_id, workspace_id);
CREATE INDEX idx_documents_collection_id ON documents(collection_id);
CREATE INDEX idx_documents_owner_id ON documents(owner_id);
CREATE INDEX idx_documents_status ON documents(status);
CREATE INDEX idx_documents_type ON documents(document_type);
CREATE INDEX idx_documents_file_hash ON documents(file_hash);
CREATE INDEX idx_documents_shared_content_id ON documents(shared_content_id);
CREATE INDEX idx_documents_search_vector ON documents USING gin(search_vector);
CREATE INDEX idx_documents_created_at ON documents(created_at);

-- Shared content chunks indexes
CREATE INDEX idx_shared_chunk_content_index ON shared_content_chunks(shared_content_id, chunk_index);
CREATE INDEX idx_shared_chunk_vector ON shared_content_chunks(vector_id, embedding_model);
CREATE INDEX idx_shared_chunk_type_quality ON shared_content_chunks(chunk_type, quality_score);
CREATE INDEX idx_shared_chunk_hash ON shared_content_chunks(content_hash);
CREATE INDEX idx_shared_chunk_created_at ON shared_content_chunks(created_at);
CREATE INDEX idx_shared_chunk_deleted_at ON shared_content_chunks(deleted_at);

-- Document chunks indexes
CREATE INDEX idx_document_chunks_workspace_id ON document_chunks(workspace_id);
CREATE INDEX idx_document_chunks_org_workspace ON document_chunks(organization_id, workspace_id);
CREATE INDEX idx_document_chunks_document_id ON document_chunks(document_id);
CREATE INDEX idx_document_chunks_vector_id ON document_chunks(vector_id);
CREATE INDEX idx_document_chunks_embedding_model ON document_chunks(embedding_model);
CREATE INDEX idx_document_chunks_quality_score ON document_chunks(quality_score);
CREATE INDEX idx_document_chunks_vector_metadata ON document_chunks USING GIN(vector_metadata);
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

-- System settings indexes
CREATE INDEX idx_system_settings_key ON system_settings(key);
CREATE INDEX idx_system_settings_category ON system_settings(category);
CREATE INDEX idx_system_settings_scope ON system_settings(scope);
CREATE INDEX idx_system_settings_org ON system_settings(organization_id) WHERE organization_id IS NOT NULL;
CREATE INDEX idx_system_settings_workspace ON system_settings(workspace_id) WHERE workspace_id IS NOT NULL;
CREATE INDEX idx_system_settings_user ON system_settings(user_id) WHERE user_id IS NOT NULL;
CREATE INDEX idx_system_settings_sensitive ON system_settings(is_sensitive) WHERE is_sensitive = true;

-- Settings history indexes
CREATE INDEX idx_settings_history_setting ON settings_history(setting_id);
CREATE INDEX idx_settings_history_changed_at ON settings_history(changed_at);
CREATE INDEX idx_settings_history_changed_by ON settings_history(changed_by);

-- Cognify API keys indexes
CREATE INDEX idx_cognify_api_keys_user_id ON cognify_api_keys(user_id);
CREATE INDEX idx_cognify_api_keys_org_id ON cognify_api_keys(organization_id) WHERE organization_id IS NOT NULL;
CREATE INDEX idx_cognify_api_keys_workspace_id ON cognify_api_keys(workspace_id) WHERE workspace_id IS NOT NULL;
CREATE INDEX idx_cognify_api_keys_status ON cognify_api_keys(status);
CREATE INDEX idx_cognify_api_keys_hash ON cognify_api_keys(key_hash);
CREATE INDEX idx_cognify_api_keys_expires ON cognify_api_keys(expires_at) WHERE expires_at IS NOT NULL;
CREATE INDEX idx_cognify_api_keys_last_used ON cognify_api_keys(last_used_at) WHERE last_used_at IS NOT NULL;

-- Cognify API key usage indexes
CREATE INDEX idx_cognify_api_key_usage_api_key_id ON cognify_api_key_usage(api_key_id);
CREATE INDEX idx_cognify_api_key_usage_user_id ON cognify_api_key_usage(user_id);
CREATE INDEX idx_cognify_api_key_usage_used_at ON cognify_api_key_usage(used_at);
CREATE INDEX idx_cognify_api_key_usage_endpoint ON cognify_api_key_usage(endpoint);
CREATE INDEX idx_cognify_api_key_usage_status ON cognify_api_key_usage(status_code);

-- Audit indexes
CREATE INDEX idx_audit_user_id ON data_access_audit(user_id);
CREATE INDEX idx_audit_workspace_id ON data_access_audit(workspace_id);
CREATE INDEX idx_audit_action ON data_access_audit(action);
CREATE INDEX idx_audit_resource ON data_access_audit(resource_type, resource_id);
CREATE INDEX idx_audit_created_at ON data_access_audit(created_at);

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

-- Function to generate short IDs
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

-- Function for full-text search
CREATE OR REPLACE FUNCTION create_search_vector(title TEXT, content TEXT, metadata JSONB DEFAULT '{}')
RETURNS tsvector AS $$
BEGIN
    RETURN setweight(to_tsvector('english', COALESCE(title, '')), 'A') ||
           setweight(to_tsvector('english', COALESCE(content, '')), 'B') ||
           setweight(to_tsvector('english', COALESCE(metadata->>'keywords', '')), 'C');
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Function for similarity search
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

-- Function to generate Cognify API key
CREATE OR REPLACE FUNCTION generate_cognify_api_key()
RETURNS TEXT AS $$
DECLARE
    key_value TEXT;
    random_part TEXT;
BEGIN
    -- Generate random part (32 characters)
    random_part := encode(gen_random_bytes(24), 'base64');
    -- Remove padding and make URL-safe
    random_part := replace(replace(random_part, '+', ''), '/', '');
    random_part := replace(random_part, '=', '');

    -- Create key with prefix
    key_value := 'cog_' || random_part;

    RETURN key_value;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to hash Cognify API key
CREATE OR REPLACE FUNCTION hash_cognify_api_key(p_key TEXT)
RETURNS TEXT AS $$
BEGIN
    RETURN encode(digest(p_key, 'sha256'), 'hex');
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to get API key prefix for display
CREATE OR REPLACE FUNCTION get_api_key_prefix(p_key TEXT)
RETURNS TEXT AS $$
BEGIN
    -- Return first 12 characters + "..."
    RETURN substring(p_key from 1 for 12) || '...';
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to authenticate Cognify API key
CREATE OR REPLACE FUNCTION authenticate_cognify_api_key(p_key TEXT)
RETURNS TABLE (
    api_key_id UUID,
    user_id UUID,
    organization_id UUID,
    workspace_id UUID,
    permissions JSONB,
    rate_limits JSONB
) AS $$
DECLARE
    key_hash TEXT;
BEGIN
    -- Hash the provided key
    key_hash := hash_cognify_api_key(p_key);

    -- Find active API key
    RETURN QUERY
    SELECT
        cak.id,
        cak.user_id,
        cak.organization_id,
        cak.workspace_id,
        cak.permissions,
        cak.rate_limits
    FROM cognify_api_keys cak
    WHERE cak.key_hash = authenticate_cognify_api_key.key_hash
    AND cak.status = 'active'
    AND (cak.expires_at IS NULL OR cak.expires_at > NOW());
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to track Cognify API key usage
CREATE OR REPLACE FUNCTION track_cognify_api_key_usage(
    p_api_key_id UUID,
    p_user_id UUID,
    p_endpoint VARCHAR(255),
    p_method VARCHAR(10),
    p_status_code INTEGER,
    p_request_id VARCHAR(255) DEFAULT NULL,
    p_user_agent TEXT DEFAULT NULL,
    p_ip_address INET DEFAULT NULL,
    p_response_time_ms INTEGER DEFAULT NULL,
    p_request_size_bytes INTEGER DEFAULT NULL,
    p_response_size_bytes INTEGER DEFAULT NULL,
    p_organization_id UUID DEFAULT NULL,
    p_workspace_id UUID DEFAULT NULL,
    p_metadata JSONB DEFAULT '{}'
) RETURNS UUID AS $$
DECLARE
    usage_id UUID;
BEGIN
    -- Insert usage record
    INSERT INTO cognify_api_key_usage (
        api_key_id, user_id, endpoint, method, status_code,
        request_id, user_agent, ip_address, response_time_ms,
        request_size_bytes, response_size_bytes,
        organization_id, workspace_id, metadata
    ) VALUES (
        p_api_key_id, p_user_id, p_endpoint, p_method, p_status_code,
        p_request_id, p_user_agent, p_ip_address, p_response_time_ms,
        p_request_size_bytes, p_response_size_bytes,
        p_organization_id, p_workspace_id, p_metadata
    ) RETURNING id INTO usage_id;

    -- Update API key usage count and last used timestamp
    UPDATE cognify_api_keys
    SET
        usage_count = usage_count + 1,
        last_used_at = NOW(),
        last_used_from_ip = p_ip_address,
        updated_at = NOW()
    WHERE id = p_api_key_id;

    RETURN usage_id;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to check Cognify API key rate limits
CREATE OR REPLACE FUNCTION check_cognify_api_key_rate_limit(
    p_api_key_id UUID,
    p_endpoint VARCHAR(255)
) RETURNS BOOLEAN AS $$
DECLARE
    rate_limits JSONB;
    requests_per_minute INTEGER;
    requests_per_hour INTEGER;
    requests_per_day INTEGER;
    current_minute_count INTEGER;
    current_hour_count INTEGER;
    current_day_count INTEGER;
BEGIN
    -- Get rate limits for this API key
    SELECT cak.rate_limits INTO rate_limits
    FROM cognify_api_keys cak
    WHERE cak.id = p_api_key_id;

    IF rate_limits IS NULL THEN
        -- No rate limits set, allow request
        RETURN TRUE;
    END IF;

    -- Extract rate limits
    requests_per_minute := COALESCE((rate_limits->>'requests_per_minute')::INTEGER, 60);
    requests_per_hour := COALESCE((rate_limits->>'requests_per_hour')::INTEGER, 1000);
    requests_per_day := COALESCE((rate_limits->>'requests_per_day')::INTEGER, 10000);

    -- Check current usage
    SELECT COUNT(*) INTO current_minute_count
    FROM cognify_api_key_usage
    WHERE api_key_id = p_api_key_id
    AND used_at >= NOW() - INTERVAL '1 minute';

    SELECT COUNT(*) INTO current_hour_count
    FROM cognify_api_key_usage
    WHERE api_key_id = p_api_key_id
    AND used_at >= NOW() - INTERVAL '1 hour';

    SELECT COUNT(*) INTO current_day_count
    FROM cognify_api_key_usage
    WHERE api_key_id = p_api_key_id
    AND used_at >= NOW() - INTERVAL '1 day';

    -- Check limits
    IF current_minute_count >= requests_per_minute THEN
        RETURN FALSE;
    END IF;

    IF current_hour_count >= requests_per_hour THEN
        RETURN FALSE;
    END IF;

    IF current_day_count >= requests_per_day THEN
        RETURN FALSE;
    END IF;

    -- All checks passed
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

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

CREATE TRIGGER update_organizations_updated_at
    BEFORE UPDATE ON organizations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_workspaces_updated_at
    BEFORE UPDATE ON workspaces
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_organization_members_updated_at
    BEFORE UPDATE ON organization_members
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_workspace_members_updated_at
    BEFORE UPDATE ON workspace_members
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_collections_updated_at
    BEFORE UPDATE ON collections
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_shared_contents_updated_at
    BEFORE UPDATE ON shared_contents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_documents_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_shared_content_chunks_updated_at
    BEFORE UPDATE ON shared_content_chunks
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_document_chunks_updated_at
    BEFORE UPDATE ON document_chunks
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_queries_updated_at
    BEFORE UPDATE ON queries
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_system_settings_updated_at
    BEFORE UPDATE ON system_settings
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_cognify_api_keys_updated_at
    BEFORE UPDATE ON cognify_api_keys
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

-- ============================================================================
-- PERMISSIONS
-- ============================================================================

-- Grant permissions to cognify_app role
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO cognify_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA auth TO cognify_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA documents TO cognify_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA analytics TO cognify_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA system TO cognify_app;

-- Grant sequence permissions
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO cognify_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA auth TO cognify_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA documents TO cognify_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA analytics TO cognify_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA system TO cognify_app;

-- Grant function execution permissions
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO cognify_app;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA auth TO cognify_app;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA documents TO cognify_app;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA analytics TO cognify_app;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA system TO cognify_app;

-- Specific function permissions
GRANT EXECUTE ON FUNCTION generate_cognify_api_key() TO cognify_app;
GRANT EXECUTE ON FUNCTION hash_cognify_api_key(TEXT) TO cognify_app;
GRANT EXECUTE ON FUNCTION get_api_key_prefix(TEXT) TO cognify_app;
GRANT EXECUTE ON FUNCTION authenticate_cognify_api_key(TEXT) TO cognify_app;
GRANT EXECUTE ON FUNCTION track_cognify_api_key_usage(UUID, UUID, VARCHAR, VARCHAR, INTEGER, VARCHAR, TEXT, INET, INTEGER, INTEGER, INTEGER, UUID, UUID, JSONB) TO cognify_app;
GRANT EXECUTE ON FUNCTION check_cognify_api_key_rate_limit(UUID, VARCHAR) TO cognify_app;

-- ============================================================================
-- INITIAL DATA AND SETTINGS
-- ============================================================================

-- Insert default LLM settings
INSERT INTO system_settings (key, category, scope, name, description, data_type, default_value, is_required) VALUES
-- Chat/Completion Models
('llm_chat_provider', 'llm_chat', 'global', 'Chat Provider', 'Default LLM provider for chat completions', 'string', 'openai', true),
('llm_chat_model', 'llm_chat', 'global', 'Chat Model', 'Default model for chat completions', 'string', 'gpt-4o', true),
('llm_chat_temperature', 'llm_chat', 'global', 'Chat Temperature', 'Temperature for chat completions', 'float', '0.7', false),
('llm_chat_max_tokens', 'llm_chat', 'global', 'Chat Max Tokens', 'Maximum tokens for chat completions', 'integer', '4000', false),

-- Embedding Models
('llm_embedding_provider', 'llm_embedding', 'global', 'Embedding Provider', 'Default provider for embeddings', 'string', 'openai', true),
('llm_embedding_model', 'llm_embedding', 'global', 'Embedding Model', 'Default model for embeddings', 'string', 'text-embedding-004', true),
('llm_embedding_dimension', 'llm_embedding', 'global', 'Embedding Dimension', 'Dimension of embedding vectors', 'integer', '768', true),

-- LiteLLM Configuration
('litellm_base_url', 'llm_chat', 'global', 'LiteLLM Base URL', 'Base URL for LiteLLM proxy', 'string', '', false),
('litellm_api_key', 'llm_chat', 'global', 'LiteLLM API Key', 'API key for LiteLLM proxy', 'encrypted', '', false),

-- OpenAI Configuration
('openai_api_key', 'llm_chat', 'global', 'OpenAI API Key', 'OpenAI API key', 'encrypted', '', false),
('openai_organization', 'llm_chat', 'global', 'OpenAI Organization', 'OpenAI organization ID', 'string', '', false),
('openai_base_url', 'llm_chat', 'global', 'OpenAI Base URL', 'Custom OpenAI base URL', 'string', 'https://ai.earnbase.io/v1', false),

-- Anthropic Configuration
('anthropic_api_key', 'llm_chat', 'global', 'Anthropic API Key', 'Anthropic API key', 'encrypted', '', false),
('anthropic_base_url', 'llm_chat', 'global', 'Anthropic Base URL', 'Custom Anthropic base URL', 'string', 'https://api.anthropic.com', false),

-- Google Configuration
('google_api_key', 'llm_chat', 'global', 'Google API Key', 'Google Gemini API key', 'encrypted', '', false),

-- Vector Database Configuration
('vector_db_provider', 'vector_database', 'global', 'Vector Database Provider', 'Vector database provider', 'string', 'qdrant', true),
('vector_db_url', 'vector_database', 'global', 'Vector Database URL', 'Vector database connection URL', 'string', 'http://localhost:6333', true),
('vector_db_api_key', 'vector_database', 'global', 'Vector Database API Key', 'Vector database API key', 'encrypted', '', false),

-- Performance Settings
('chunk_size', 'performance', 'global', 'Chunk Size', 'Default chunk size for document processing', 'integer', '1000', false),
('chunk_overlap', 'performance', 'global', 'Chunk Overlap', 'Overlap between chunks', 'integer', '200', false),
('max_concurrent_requests', 'performance', 'global', 'Max Concurrent Requests', 'Maximum concurrent API requests', 'integer', '10', false),

-- Security Settings
('api_rate_limit_per_minute', 'security', 'global', 'API Rate Limit Per Minute', 'Default API rate limit per minute', 'integer', '60', false),
('api_rate_limit_per_hour', 'security', 'global', 'API Rate Limit Per Hour', 'Default API rate limit per hour', 'integer', '1000', false),
('max_file_size_mb', 'security', 'global', 'Max File Size (MB)', 'Maximum file size for uploads', 'integer', '50', false),

-- Feature Flags
('enable_shared_content', 'features', 'global', 'Enable Shared Content', 'Enable content deduplication', 'boolean', 'true', false),
('enable_analytics', 'features', 'global', 'Enable Analytics', 'Enable usage analytics', 'boolean', 'true', false),
('enable_audit_logging', 'features', 'global', 'Enable Audit Logging', 'Enable audit logging', 'boolean', 'true', false)

ON CONFLICT (key, scope, organization_id, workspace_id, user_id) DO NOTHING;

-- Log successful initialization
INSERT INTO system.initialization_log (message)
VALUES ('Complete database structure initialized successfully with all tables, indexes, functions, and triggers');

-- Final success message
SELECT 'Cognify database initialization completed successfully!' as status;
