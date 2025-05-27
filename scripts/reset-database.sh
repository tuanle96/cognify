#!/bin/bash

# Reset Database Script for Cognify
# This script drops the existing database and recreates it with the complete structure
# Date: 2025-01-26

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Database configuration
DB_HOST=${DB_HOST:-localhost}
DB_PORT=${DB_PORT:-5432}
DB_NAME=${DB_NAME:-cognify_db}
DB_USER=${DB_USER:-postgres}
DB_PASSWORD=${DB_PASSWORD:-postgres}
APP_USER=${APP_USER:-cognify_app}

echo -e "${BLUE}🔄 Cognify Database Reset Script${NC}"
echo -e "${BLUE}=================================${NC}"
echo ""

# Function to run SQL command
run_sql() {
    local sql="$1"
    local description="$2"
    
    echo -e "${YELLOW}⏳ $description...${NC}"
    
    if PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d postgres -c "$sql" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ $description completed${NC}"
    else
        echo -e "${RED}❌ $description failed${NC}"
        return 1
    fi
}

# Function to run SQL file
run_sql_file() {
    local file="$1"
    local description="$2"
    
    echo -e "${YELLOW}⏳ $description...${NC}"
    
    if PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -f "$file" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ $description completed${NC}"
    else
        echo -e "${RED}❌ $description failed${NC}"
        return 1
    fi
}

# Check if PostgreSQL is running
echo -e "${YELLOW}🔍 Checking PostgreSQL connection...${NC}"
if ! PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d postgres -c "SELECT 1;" > /dev/null 2>&1; then
    echo -e "${RED}❌ Cannot connect to PostgreSQL. Please check your connection settings.${NC}"
    echo -e "${YELLOW}Current settings:${NC}"
    echo -e "  Host: $DB_HOST"
    echo -e "  Port: $DB_PORT"
    echo -e "  User: $DB_USER"
    echo -e "  Database: $DB_NAME"
    exit 1
fi
echo -e "${GREEN}✅ PostgreSQL connection successful${NC}"

# Confirm before proceeding
echo ""
echo -e "${RED}⚠️  WARNING: This will completely destroy the existing database!${NC}"
echo -e "${YELLOW}Database to be reset: $DB_NAME${NC}"
echo -e "${YELLOW}All data will be permanently lost!${NC}"
echo ""
read -p "Are you sure you want to continue? (type 'yes' to confirm): " confirm

if [ "$confirm" != "yes" ]; then
    echo -e "${YELLOW}❌ Operation cancelled${NC}"
    exit 0
fi

echo ""
echo -e "${BLUE}🚀 Starting database reset...${NC}"
echo ""

# Step 1: Terminate existing connections
echo -e "${YELLOW}⏳ Terminating existing connections to $DB_NAME...${NC}"
PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d postgres -c "
SELECT pg_terminate_backend(pid) 
FROM pg_stat_activity 
WHERE datname = '$DB_NAME' AND pid <> pg_backend_pid();" > /dev/null 2>&1
echo -e "${GREEN}✅ Connections terminated${NC}"

# Step 2: Drop the database
run_sql "DROP DATABASE IF EXISTS $DB_NAME;" "Dropping database $DB_NAME"

# Step 3: Drop the application user if exists
run_sql "DROP ROLE IF EXISTS $APP_USER;" "Dropping application user $APP_USER"

# Step 4: Create the database
run_sql "CREATE DATABASE $DB_NAME;" "Creating database $DB_NAME"

# Step 5: Run the complete initialization script
echo -e "${YELLOW}⏳ Running complete database initialization...${NC}"
if [ -f "scripts/init-complete-db.sql" ]; then
    if PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -f "scripts/init-complete-db.sql"; then
        echo -e "${GREEN}✅ Database initialization completed${NC}"
    else
        echo -e "${RED}❌ Database initialization failed${NC}"
        exit 1
    fi
else
    echo -e "${RED}❌ Initialization script not found: scripts/init-complete-db.sql${NC}"
    exit 1
fi

# Step 6: Verify the setup
echo ""
echo -e "${YELLOW}🔍 Verifying database setup...${NC}"

# Check if tables were created
table_count=$(PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -c "
SELECT COUNT(*) 
FROM information_schema.tables 
WHERE table_schema = 'public' AND table_type = 'BASE TABLE';")

if [ "$table_count" -gt 0 ]; then
    echo -e "${GREEN}✅ Tables created: $table_count tables${NC}"
else
    echo -e "${RED}❌ No tables found${NC}"
    exit 1
fi

# Check if functions were created
function_count=$(PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -c "
SELECT COUNT(*) 
FROM information_schema.routines 
WHERE routine_schema = 'public';")

if [ "$function_count" -gt 0 ]; then
    echo -e "${GREEN}✅ Functions created: $function_count functions${NC}"
else
    echo -e "${YELLOW}⚠️  No functions found${NC}"
fi

# Check if application user was created
user_exists=$(PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d postgres -t -c "
SELECT EXISTS(SELECT 1 FROM pg_roles WHERE rolname = '$APP_USER');")

if [ "$user_exists" = " t" ]; then
    echo -e "${GREEN}✅ Application user created: $APP_USER${NC}"
else
    echo -e "${RED}❌ Application user not found: $APP_USER${NC}"
fi

# Step 7: Show summary
echo ""
echo -e "${BLUE}📊 Database Reset Summary${NC}"
echo -e "${BLUE}=========================${NC}"

# Get table list
echo -e "${YELLOW}📋 Created Tables:${NC}"
PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "
SELECT 
    schemaname as schema,
    tablename as table,
    CASE 
        WHEN schemaname = 'public' THEN '🔹'
        WHEN schemaname = 'system' THEN '⚙️'
        ELSE '📁'
    END as icon
FROM pg_tables 
WHERE schemaname IN ('public', 'auth', 'documents', 'analytics', 'system')
ORDER BY schemaname, tablename;" 2>/dev/null || echo "Could not retrieve table list"

echo ""
echo -e "${GREEN}🎉 Database reset completed successfully!${NC}"
echo ""
echo -e "${YELLOW}📝 Next Steps:${NC}"
echo -e "  1. Update your application configuration"
echo -e "  2. Run your application to test the connection"
echo -e "  3. Create initial admin user if needed"
echo ""
echo -e "${BLUE}Database Connection Info:${NC}"
echo -e "  Host: $DB_HOST"
echo -e "  Port: $DB_PORT"
echo -e "  Database: $DB_NAME"
echo -e "  App User: $APP_USER"
echo ""
echo -e "${GREEN}✨ Ready to use!${NC}"
