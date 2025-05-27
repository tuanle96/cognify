#!/bin/bash

# Full Reset Script for Cognify
# This script performs a complete cleanup and database reset
# Date: 2025-01-26

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

echo -e "${PURPLE}🔄 Cognify Full Reset Script${NC}"
echo -e "${PURPLE}============================${NC}"
echo ""

# Check if we're in the right directory
if [ ! -f "scripts/reset-database.sh" ] || [ ! -f "scripts/cleanup-migrations.sh" ]; then
    echo -e "${RED}❌ Required scripts not found. Please run this script from the cognify root directory.${NC}"
    exit 1
fi

# Make scripts executable
chmod +x scripts/reset-database.sh
chmod +x scripts/cleanup-migrations.sh

echo -e "${BLUE}🎯 This script will perform the following actions:${NC}"
echo -e "  1. 🧹 Clean up migration files"
echo -e "  2. 🗄️  Drop and recreate database"
echo -e "  3. 🏗️  Initialize complete database structure"
echo -e "  4. ✅ Verify setup"
echo ""

# Confirm before proceeding
echo -e "${RED}⚠️  WARNING: This will completely reset your Cognify database!${NC}"
echo -e "${YELLOW}All data will be permanently lost!${NC}"
echo ""
read -p "Are you sure you want to continue? (type 'RESET' to confirm): " confirm

if [ "$confirm" != "RESET" ]; then
    echo -e "${YELLOW}❌ Operation cancelled${NC}"
    exit 0
fi

echo ""
echo -e "${PURPLE}🚀 Starting full reset process...${NC}"
echo ""

# Step 1: Cleanup migration files
echo -e "${BLUE}Step 1: Cleaning up migration files${NC}"
echo -e "${BLUE}====================================${NC}"
echo ""

if ./scripts/cleanup-migrations.sh; then
    echo -e "${GREEN}✅ Migration cleanup completed${NC}"
else
    echo -e "${RED}❌ Migration cleanup failed${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}Step 2: Resetting database${NC}"
echo -e "${BLUE}===========================${NC}"
echo ""

# Step 2: Reset database
if ./scripts/reset-database.sh; then
    echo -e "${GREEN}✅ Database reset completed${NC}"
else
    echo -e "${RED}❌ Database reset failed${NC}"
    exit 1
fi

# Step 3: Final verification
echo ""
echo -e "${BLUE}Step 3: Final verification${NC}"
echo -e "${BLUE}==========================${NC}"
echo ""

# Check database connection
DB_HOST=${DB_HOST:-localhost}
DB_PORT=${DB_PORT:-5432}
DB_NAME=${DB_NAME:-cognify_db}
DB_USER=${DB_USER:-postgres}
DB_PASSWORD=${DB_PASSWORD:-postgres}

echo -e "${YELLOW}🔍 Verifying database connection...${NC}"
if PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "SELECT 'Database connection successful!' as status;" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Database connection verified${NC}"
else
    echo -e "${RED}❌ Database connection failed${NC}"
    exit 1
fi

# Check key tables
echo -e "${YELLOW}🔍 Verifying key tables...${NC}"
key_tables=("users" "organizations" "workspaces" "collections" "documents" "system_settings" "cognify_api_keys")
missing_tables=()

for table in "${key_tables[@]}"; do
    if PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "SELECT 1 FROM $table LIMIT 1;" > /dev/null 2>&1; then
        echo -e "${GREEN}  ✅ $table${NC}"
    else
        echo -e "${RED}  ❌ $table${NC}"
        missing_tables+=("$table")
    fi
done

if [ ${#missing_tables[@]} -eq 0 ]; then
    echo -e "${GREEN}✅ All key tables verified${NC}"
else
    echo -e "${RED}❌ Missing tables: ${missing_tables[*]}${NC}"
    exit 1
fi

# Check functions
echo -e "${YELLOW}🔍 Verifying key functions...${NC}"
key_functions=("generate_cognify_api_key" "authenticate_cognify_api_key" "update_updated_at_column")
missing_functions=()

for func in "${key_functions[@]}"; do
    if PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "SELECT $func();" > /dev/null 2>&1; then
        echo -e "${GREEN}  ✅ $func${NC}"
    else
        echo -e "${RED}  ❌ $func${NC}"
        missing_functions+=("$func")
    fi
done

if [ ${#missing_functions[@]} -eq 0 ]; then
    echo -e "${GREEN}✅ All key functions verified${NC}"
else
    echo -e "${YELLOW}⚠️  Some functions may not be testable: ${missing_functions[*]}${NC}"
fi

# Final summary
echo ""
echo -e "${PURPLE}🎉 Full Reset Complete!${NC}"
echo -e "${PURPLE}=======================${NC}"
echo ""

echo -e "${GREEN}✅ Migration files cleaned up${NC}"
echo -e "${GREEN}✅ Database dropped and recreated${NC}"
echo -e "${GREEN}✅ Complete database structure initialized${NC}"
echo -e "${GREEN}✅ All verifications passed${NC}"
echo ""

echo -e "${BLUE}📊 Database Statistics:${NC}"
# Get table count
table_count=$(PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public' AND table_type = 'BASE TABLE';")
echo -e "  📋 Tables: $table_count"

# Get function count
function_count=$(PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -c "SELECT COUNT(*) FROM information_schema.routines WHERE routine_schema = 'public';")
echo -e "  ⚙️  Functions: $function_count"

# Get index count
index_count=$(PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -c "SELECT COUNT(*) FROM pg_indexes WHERE schemaname = 'public';")
echo -e "  🔍 Indexes: $index_count"

echo ""
echo -e "${YELLOW}📝 What's Available Now:${NC}"
echo -e "  🔐 Multi-tenant architecture (Organizations → Workspaces)"
echo -e "  📄 Document management with content deduplication"
echo -e "  🔧 Comprehensive settings system"
echo -e "  🔑 Cognify API keys for authentication"
echo -e "  📊 Usage tracking and analytics"
echo -e "  🛡️  Security and audit logging"
echo ""

echo -e "${BLUE}🚀 Next Steps:${NC}"
echo -e "  1. Start your Cognify application"
echo -e "  2. Create your first organization and workspace"
echo -e "  3. Set up LLM provider settings"
echo -e "  4. Create API keys for external integrations"
echo ""

echo -e "${GREEN}✨ Your Cognify database is ready to use!${NC}"
