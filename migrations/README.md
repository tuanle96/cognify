# Migrations Directory

This directory previously contained individual migration files, but they have been consolidated into a single initialization script for better maintainability and consistency.

## Current Setup

- **Single Init Script**: `scripts/init-complete-db.sql`
- **Reset Script**: `scripts/reset-database.sh`
- **Cleanup Script**: `scripts/cleanup-migrations.sh`
- **Full Reset Script**: `scripts/full-reset.sh`

## Usage

### Quick Start (Recommended)
```bash
# Make scripts executable
chmod +x scripts/*.sh

# Run complete reset (cleans migrations + resets DB)
./scripts/full-reset.sh
```

### Individual Scripts
```bash
# Reset database only
./scripts/reset-database.sh

# Clean up migration files only
./scripts/cleanup-migrations.sh

# Manual SQL execution
psql -h localhost -U postgres -d cognify_db -f scripts/init-complete-db.sql
```

## Migration History

The following migrations were consolidated into `scripts/init-complete-db.sql`:

### 1. `001_create_multi_tenant_schema.sql`
- **Purpose**: Multi-tenant architecture foundation
- **Features**: Organizations, workspaces, user membership
- **Tables**: `organizations`, `workspaces`, `organization_members`, `workspace_members`

### 2. `002_update_existing_tables_for_tenancy.sql`
- **Purpose**: Update existing tables for multi-tenancy
- **Features**: Added workspace/organization references
- **Updates**: `collections`, `documents`, `document_chunks` with tenant support

### 3. `003_create_settings_system.sql`
- **Purpose**: Comprehensive settings management
- **Features**: Hierarchical settings, LLM configuration
- **Tables**: `system_settings`, `settings_history`

### 4. `004_create_cognify_api_keys.sql`
- **Purpose**: API key authentication system
- **Features**: User API keys, usage tracking, rate limiting
- **Tables**: `cognify_api_keys`, `cognify_api_key_usage`

### 5. `add_shared_content.sql`
- **Purpose**: Content deduplication system
- **Features**: Shared content storage, reference counting
- **Tables**: `shared_contents`, `shared_content_chunks`

## Benefits of Consolidation

### ✅ **Advantages**
- **Faster Setup**: Single script execution
- **No Dependencies**: No migration order issues
- **Consistent State**: Same result every time
- **Easier Debugging**: All structure in one place
- **Better Maintenance**: Single file to update
- **Simplified Deployment**: One script for all environments

### 📊 **Complete Database Structure**
The consolidated script includes:
- **15+ Tables**: Complete data model
- **50+ Indexes**: Performance optimizations
- **8 Functions**: Utility and business logic
- **11 Enums**: Type safety
- **Triggers**: Automatic updates
- **Permissions**: Security setup
- **Initial Data**: Default settings

## Environment Configuration

```bash
# Database connection settings
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=cognify_db
export DB_USER=postgres
export DB_PASSWORD=postgres
export APP_USER=cognify_app
```

## Verification

After running the init script, verify the setup:

```sql
-- Check table count
SELECT COUNT(*) FROM information_schema.tables 
WHERE table_schema = 'public' AND table_type = 'BASE TABLE';

-- Check function count
SELECT COUNT(*) FROM information_schema.routines 
WHERE routine_schema = 'public';

-- Check default settings
SELECT COUNT(*) FROM system_settings;

-- Test API key generation
SELECT generate_cognify_api_key();
```

## Troubleshooting

### Common Issues

1. **Permission Denied**
   ```bash
   chmod +x scripts/*.sh
   ```

2. **Database Connection Failed**
   - Check PostgreSQL is running
   - Verify connection settings
   - Ensure user has proper permissions

3. **Script Not Found**
   - Run from cognify root directory
   - Ensure scripts exist in `scripts/` folder

### Getting Help

- Check `DATABASE_RESET_SUMMARY.md` for detailed documentation
- Review script output for specific error messages
- Verify environment variables are set correctly

## Future Migrations

If you need to add new database changes:

1. **Small Changes**: Update `scripts/init-complete-db.sql` directly
2. **Major Changes**: Consider creating a new version of the init script
3. **Production**: Use proper migration tools for live databases

## Related Files

- `scripts/init-complete-db.sql` - Complete database initialization
- `scripts/reset-database.sh` - Database reset script
- `scripts/cleanup-migrations.sh` - Migration cleanup script
- `scripts/full-reset.sh` - Complete reset process
- `DATABASE_RESET_SUMMARY.md` - Detailed documentation
