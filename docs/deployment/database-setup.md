# 🗄️ Database Setup & Management Guide

**Complete guide for Cognify database initialization, reset, and management**

---

## 🎯 **Quick Start**

### **Option 1: Full Reset (Recommended)**
```bash
# Make scripts executable
chmod +x scripts/*.sh

# Run complete reset (cleans migrations + resets DB)
./scripts/full-reset.sh
```

### **Option 2: Database Reset Only**
```bash
# Reset database only (keep migration files)
./scripts/reset-database.sh
```

### **Option 3: Manual SQL Execution**
```bash
# Run SQL script directly
psql -h localhost -U postgres -d cognify_db -f scripts/init-complete-db.sql
```

---

## 🗄️ **Complete Database Structure**

### **Core Tables (15 tables):**
```sql
-- Authentication & Users
users, organizations, workspaces
organization_members, workspace_members

-- Document Management
collections, documents, document_chunks
shared_contents, shared_content_chunks

-- Query System
queries, query_results

-- Settings & Configuration
system_settings, settings_history

-- API Keys & Security
cognify_api_keys, cognify_api_key_usage

-- Audit & Analytics
data_access_audit
```

### **Functions (8 functions):**
```sql
-- Utility functions
update_updated_at_column()
generate_short_id()
create_search_vector()
cosine_similarity()

-- API key functions
generate_cognify_api_key()
hash_cognify_api_key()
authenticate_cognify_api_key()
track_cognify_api_key_usage()
check_cognify_api_key_rate_limit()
```

### **Indexes (50+ indexes):**
- Performance indexes for all major queries
- Full-text search indexes
- Multi-column indexes for complex queries
- Partial indexes for filtered queries

---

## 🔧 **Environment Configuration**

### **Database Connection Settings:**
```bash
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=cognify_db
export DB_USER=postgres
export DB_PASSWORD=postgres
export APP_USER=cognify_app
```

### **Default Settings Included:**
```sql
-- LLM Providers
llm_chat_provider = 'openai'
llm_chat_model = 'gpt-4o'
llm_embedding_provider = 'openai'
llm_embedding_model = 'text-embedding-3-small'

-- Vector Database
vector_db_provider = 'qdrant'
vector_db_url = 'http://localhost:6333'

-- Performance
chunk_size = 1000
chunk_overlap = 200
max_concurrent_requests = 10

-- Security
api_rate_limit_per_minute = 60
api_rate_limit_per_hour = 1000
max_file_size_mb = 50

-- Features
enable_shared_content = true
enable_analytics = true
enable_audit_logging = true
```

---

## 🔍 **Verification Process**

### **Automatic Verification:**
The reset scripts automatically verify:
- ✅ Database connection
- ✅ Table creation (15+ tables)
- ✅ Function creation (8 functions)
- ✅ Application user creation
- ✅ Permissions setup

### **Manual Verification:**
```sql
-- Check table count
SELECT COUNT(*) FROM information_schema.tables 
WHERE table_schema = 'public' AND table_type = 'BASE TABLE';

-- Check function count
SELECT COUNT(*) FROM information_schema.routines 
WHERE routine_schema = 'public';

-- Check settings
SELECT COUNT(*) FROM system_settings;

-- Test API key generation
SELECT generate_cognify_api_key();
```

---

## 🛡️ **Security Features**

### **Database Security:**
- ✅ **Dedicated App User** - `cognify_app` with limited permissions
- ✅ **Encrypted Settings** - Sensitive data encryption
- ✅ **Audit Logging** - Complete operation tracking
- ✅ **Rate Limiting** - Built-in rate limit functions

### **API Key Security:**
- ✅ **SHA-256 Hashing** - Secure key storage
- ✅ **Rate Limiting** - Per-key rate limits
- ✅ **Usage Tracking** - Complete usage analytics
- ✅ **Permission Control** - Granular permissions

---

## 📈 **Performance Optimizations**

### **Indexing Strategy:**
- **Primary Keys** - UUID with gen_random_uuid()
- **Foreign Keys** - All relationships indexed
- **Search Indexes** - Full-text search with tsvector
- **Composite Indexes** - Multi-column for complex queries
- **Partial Indexes** - Filtered for specific conditions

### **Query Optimizations:**
- **Multi-tenant Queries** - Optimized for workspace/org filtering
- **Document Search** - Full-text + vector search ready
- **Analytics Queries** - Pre-indexed for reporting
- **API Key Lookups** - Hash-based for fast authentication

---

## 🎯 **Key Features Included**

### **1. Multi-Tenant Architecture:**
- Organizations (top-level tenants)
- Workspaces (team/project level)
- User membership management
- Data isolation and permissions

### **2. Document Management:**
- Document upload and processing
- Content deduplication (shared_contents)
- Chunking and embedding support
- Full-text search capabilities

### **3. Settings System:**
- Hierarchical settings (global → org → workspace → user)
- LLM provider configuration
- Feature flags and performance tuning
- Encrypted sensitive data

### **4. API Authentication:**
- JWT token support (existing)
- Cognify API keys (new)
- Usage tracking and analytics
- Rate limiting and permissions

### **5. Analytics & Audit:**
- Complete usage tracking
- Performance monitoring
- Security audit trail
- Data access logging

---

## 🚨 **Troubleshooting**

### **Common Issues:**

#### **Permission Denied**
```bash
chmod +x scripts/*.sh
```

#### **Database Connection Failed**
- Check PostgreSQL is running
- Verify connection settings
- Ensure user has proper permissions

#### **Script Not Found**
- Run from cognify root directory
- Ensure scripts exist in `scripts/` folder

### **Getting Help:**
- Check script output for specific error messages
- Verify environment variables are set correctly
- Review logs for detailed error information

---

## 🔄 **Migration History**

### **Consolidated Migrations:**
The current init script consolidates these previous migrations:

1. **`001_create_multi_tenant_schema.sql`** → Multi-tenant architecture
2. **`002_update_existing_tables_for_tenancy.sql`** → Tenancy updates  
3. **`003_create_settings_system.sql`** → Settings system
4. **`004_create_cognify_api_keys.sql`** → API keys system
5. **`add_shared_content.sql`** → Content deduplication

### **Benefits of Consolidation:**
- ✅ **Faster Setup** - Single script execution
- ✅ **No Dependencies** - No migration order issues
- ✅ **Consistent State** - Same result every time
- ✅ **Easier Debugging** - All structure in one place
- ✅ **Better Maintenance** - Single file to update

---

## 📋 **Next Steps**

### **After Database Setup:**
1. **Update Application** - Point to new database structure
2. **Test Connection** - Verify application connectivity
3. **Create Admin User** - Set up initial admin account
4. **Configure LLM Settings** - Set up OpenAI/Anthropic keys
5. **Create Organizations** - Set up initial org/workspace

### **For Production:**
1. **Security Hardening** - Configure SSL, firewall
2. **Backup Strategy** - Set up automated backups
3. **Monitoring** - Configure alerts and dashboards
4. **Performance Tuning** - Optimize for your workload

---

**Database setup complete! Ready for Cognify deployment.** 🚀
