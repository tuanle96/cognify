#!/bin/bash

# Cleanup Migration Files Script for Cognify
# This script removes all migration files since we now use a single init script
# Date: 2025-01-26

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🧹 Cognify Migration Cleanup Script${NC}"
echo -e "${BLUE}===================================${NC}"
echo ""

# Check if we're in the right directory
if [ ! -d "migrations" ]; then
    echo -e "${RED}❌ migrations directory not found. Please run this script from the cognify root directory.${NC}"
    exit 1
fi

# List migration files to be removed
echo -e "${YELLOW}📋 Migration files to be removed:${NC}"
migration_files=(
    "migrations/001_create_multi_tenant_schema.sql"
    "migrations/002_update_existing_tables_for_tenancy.sql"
    "migrations/003_create_settings_system.sql"
    "migrations/004_create_cognify_api_keys.sql"
    "migrations/add_shared_content.sql"
)

files_found=0
for file in "${migration_files[@]}"; do
    if [ -f "$file" ]; then
        echo -e "  🗑️  $file"
        ((files_found++))
    fi
done

if [ $files_found -eq 0 ]; then
    echo -e "${GREEN}✅ No migration files found to remove${NC}"
    exit 0
fi

echo ""
echo -e "${YELLOW}📄 Total files to remove: $files_found${NC}"
echo ""

# Confirm before proceeding
echo -e "${RED}⚠️  WARNING: This will permanently delete all migration files!${NC}"
echo -e "${YELLOW}These files will be removed because we now use a single init script.${NC}"
echo ""
read -p "Are you sure you want to continue? (type 'yes' to confirm): " confirm

if [ "$confirm" != "yes" ]; then
    echo -e "${YELLOW}❌ Operation cancelled${NC}"
    exit 0
fi

echo ""
echo -e "${BLUE}🚀 Starting migration cleanup...${NC}"
echo ""

# Remove migration files
removed_count=0
for file in "${migration_files[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${YELLOW}⏳ Removing $file...${NC}"
        if rm "$file"; then
            echo -e "${GREEN}✅ Removed $file${NC}"
            ((removed_count++))
        else
            echo -e "${RED}❌ Failed to remove $file${NC}"
        fi
    fi
done

# Check if migrations directory is empty (except for README or .gitkeep)
remaining_files=$(find migrations -name "*.sql" -type f | wc -l)

if [ $remaining_files -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ All migration files removed successfully${NC}"
    
    # Create a README in migrations directory
    cat > migrations/README.md << 'EOF'
# Migrations Directory

This directory previously contained individual migration files, but they have been consolidated into a single initialization script.

## Current Setup

- **Single Init Script**: `scripts/init-complete-db.sql`
- **Reset Script**: `scripts/reset-database.sh`

## Usage

To initialize or reset the database:

```bash
# Make script executable
chmod +x scripts/reset-database.sh

# Run the reset script
./scripts/reset-database.sh
```

## Migration History

The following migrations were consolidated:
- `001_create_multi_tenant_schema.sql` - Multi-tenant architecture
- `002_update_existing_tables_for_tenancy.sql` - Tenancy updates
- `003_create_settings_system.sql` - Settings system
- `004_create_cognify_api_keys.sql` - API keys system
- `add_shared_content.sql` - Content deduplication

All functionality from these migrations is now included in the single init script.
EOF

    echo -e "${GREEN}✅ Created README.md in migrations directory${NC}"
else
    echo -e "${YELLOW}⚠️  $remaining_files SQL files still remain in migrations directory${NC}"
fi

# Summary
echo ""
echo -e "${BLUE}📊 Cleanup Summary${NC}"
echo -e "${BLUE}==================${NC}"
echo -e "${GREEN}✅ Files removed: $removed_count${NC}"
echo -e "${YELLOW}📁 Remaining SQL files: $remaining_files${NC}"
echo ""

if [ $removed_count -gt 0 ]; then
    echo -e "${GREEN}🎉 Migration cleanup completed successfully!${NC}"
    echo ""
    echo -e "${YELLOW}📝 What's Next:${NC}"
    echo -e "  1. Use ${BLUE}scripts/init-complete-db.sql${NC} for database initialization"
    echo -e "  2. Use ${BLUE}scripts/reset-database.sh${NC} to reset the database"
    echo -e "  3. All previous migration functionality is preserved in the init script"
    echo ""
    echo -e "${BLUE}💡 Benefits of Single Init Script:${NC}"
    echo -e "  ✅ Faster database setup"
    echo -e "  ✅ No migration order dependencies"
    echo -e "  ✅ Consistent database state"
    echo -e "  ✅ Easier maintenance"
else
    echo -e "${YELLOW}ℹ️  No files were removed${NC}"
fi

echo ""
echo -e "${GREEN}✨ Cleanup complete!${NC}"
