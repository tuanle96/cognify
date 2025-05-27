#!/usr/bin/env python3
"""
Production Database Initialization Script
Initialize database structure and create admin user for production deployment
"""

import asyncio
import asyncpg
import os
import sys
import hashlib
import secrets
import logging
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProductionInitializer:
    """Initialize production database and admin user"""
    
    def __init__(self):
        # Production database configuration
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', 5432)),
            'database': os.getenv('DB_NAME', 'cognify'),
            'user': os.getenv('DB_USER', 'cognify'),
            'password': os.getenv('DB_PASSWORD', 'cognify_prod_password')
        }
        self.connection = None
        
        # Admin user configuration
        self.admin_email = os.getenv('ADMIN_EMAIL', 'admin@cognify.ai')
        self.admin_password = os.getenv('ADMIN_PASSWORD', self._generate_secure_password())
        self.admin_name = os.getenv('ADMIN_NAME', 'Cognify Administrator')
        
        # Organization configuration
        self.org_name = os.getenv('ORG_NAME', 'Cognify')
        self.org_slug = os.getenv('ORG_SLUG', 'cognify')
        self.org_description = os.getenv('ORG_DESCRIPTION', 'Cognify AI Platform')
    
    def _generate_secure_password(self):
        """Generate secure random password"""
        return secrets.token_urlsafe(16)
    
    def _hash_password(self, password: str) -> str:
        """Hash password for storage"""
        salt = secrets.token_hex(16)
        password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return f"{salt}:{password_hash.hex()}"
    
    async def connect(self):
        """Connect to production database"""
        try:
            self.connection = await asyncpg.connect(**self.db_config)
            logger.info("✅ Connected to production database")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to database: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from database"""
        if self.connection:
            await self.connection.close()
            logger.info("✅ Disconnected from database")
    
    async def check_existing_schema(self):
        """Check if schema already exists"""
        try:
            # Check if organizations table exists
            exists = await self.connection.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'organizations'
                );
            """)
            
            if exists:
                logger.warning("⚠️  Database schema already exists")
                return True
            else:
                logger.info("📋 Database schema not found - will create")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error checking schema: {e}")
            return False
    
    async def run_migrations(self):
        """Run all production migrations"""
        logger.info("🔄 Running production migrations...")
        
        try:
            # Read and execute migration files
            migration_files = [
                Path(__file__).parent.parent / "migrations" / "001_create_multi_tenant_schema.sql",
                Path(__file__).parent.parent / "migrations" / "002_update_existing_tables_for_tenancy.sql"
            ]
            
            for migration_file in migration_files:
                if migration_file.exists():
                    logger.info(f"📄 Executing {migration_file.name}")
                    
                    with open(migration_file, 'r') as f:
                        migration_sql = f.read()
                    
                    # Execute migration in transaction
                    async with self.connection.transaction():
                        await self.connection.execute(migration_sql)
                    
                    logger.info(f"✅ Completed {migration_file.name}")
                else:
                    logger.error(f"❌ Migration file not found: {migration_file}")
                    return False
            
            # Add missing functions if needed
            await self._ensure_functions_exist()
            
            logger.info("✅ All migrations completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Migration failed: {e}")
            return False
    
    async def _ensure_functions_exist(self):
        """Ensure all required functions exist"""
        functions_sql = """
        -- Function to get tenant-isolated collection name
        CREATE OR REPLACE FUNCTION get_tenant_collection_name(
            p_workspace_id UUID,
            p_collection_name TEXT
        ) RETURNS TEXT AS $$
        BEGIN
            RETURN 'ws_' || p_workspace_id::text || '_' || p_collection_name;
        END;
        $$ LANGUAGE plpgsql IMMUTABLE;

        -- Function to validate workspace access
        CREATE OR REPLACE FUNCTION validate_workspace_access(
            p_user_id UUID,
            p_workspace_id UUID,
            p_permission TEXT DEFAULT 'read'
        ) RETURNS BOOLEAN AS $$
        BEGIN
            RETURN EXISTS (
                SELECT 1 FROM workspace_members wm
                JOIN workspaces w ON w.id = wm.workspace_id
                WHERE wm.user_id = p_user_id 
                AND wm.workspace_id = p_workspace_id
                AND wm.status = 'active'
                AND w.deleted_at IS NULL
                AND (
                    wm.role = 'admin' OR
                    (p_permission = 'read' AND wm.can_read = true) OR
                    (p_permission = 'write' AND wm.can_write = true) OR
                    (p_permission = 'delete' AND wm.can_delete = true) OR
                    (p_permission = 'share' AND wm.can_share = true)
                )
            );
        END;
        $$ LANGUAGE plpgsql SECURITY DEFINER;

        -- Function to log data access
        CREATE OR REPLACE FUNCTION log_data_access(
            p_user_id UUID,
            p_workspace_id UUID,
            p_organization_id UUID,
            p_action VARCHAR(50),
            p_resource_type VARCHAR(50),
            p_resource_id UUID,
            p_metadata JSONB DEFAULT '{}'
        ) RETURNS VOID AS $$
        BEGIN
            CREATE TABLE IF NOT EXISTS data_access_audit (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID NOT NULL,
                workspace_id UUID,
                organization_id UUID,
                action VARCHAR(50) NOT NULL,
                resource_type VARCHAR(50) NOT NULL,
                resource_id UUID NOT NULL,
                ip_address INET,
                user_agent TEXT,
                metadata JSONB DEFAULT '{}',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
            
            INSERT INTO data_access_audit (
                user_id, workspace_id, organization_id, action, 
                resource_type, resource_id, metadata
            ) VALUES (
                p_user_id, p_workspace_id, p_organization_id, p_action,
                p_resource_type, p_resource_id, p_metadata
            );
        END;
        $$ LANGUAGE plpgsql SECURITY DEFINER;
        """
        
        await self.connection.execute(functions_sql)
        logger.info("✅ Security functions ensured")
    
    async def create_production_organization(self):
        """Create production organization"""
        logger.info("🏢 Creating production organization...")
        
        try:
            # Create organization
            org_id = await self.connection.fetchval("""
                INSERT INTO organizations (name, slug, description, plan_type, status)
                VALUES ($1, $2, $3, 'enterprise', 'active')
                ON CONFLICT (slug) DO UPDATE SET 
                    name = EXCLUDED.name,
                    description = EXCLUDED.description,
                    plan_type = EXCLUDED.plan_type
                RETURNING id
            """, self.org_name, self.org_slug, self.org_description)
            
            logger.info(f"✅ Organization created: {self.org_name} ({org_id})")
            return org_id
            
        except Exception as e:
            logger.error(f"❌ Failed to create organization: {e}")
            return None
    
    async def create_admin_workspace(self, org_id):
        """Create admin workspace"""
        logger.info("📁 Creating admin workspace...")
        
        try:
            workspace_id = await self.connection.fetchval("""
                INSERT INTO workspaces (organization_id, name, slug, description, visibility)
                VALUES ($1, 'Admin Workspace', 'admin', 'Administrative workspace', 'private')
                ON CONFLICT (organization_id, slug) DO UPDATE SET 
                    name = EXCLUDED.name,
                    description = EXCLUDED.description
                RETURNING id
            """, org_id)
            
            logger.info(f"✅ Admin workspace created: {workspace_id}")
            return workspace_id
            
        except Exception as e:
            logger.error(f"❌ Failed to create admin workspace: {e}")
            return None
    
    async def create_admin_user(self, org_id, workspace_id):
        """Create admin user"""
        logger.info("👤 Creating admin user...")
        
        try:
            # Hash password
            password_hash = self._hash_password(self.admin_password)
            
            # Create admin user
            admin_id = await self.connection.fetchval("""
                INSERT INTO users (email, password_hash, full_name, is_active, is_verified, role)
                VALUES ($1, $2, $3, true, true, 'admin')
                ON CONFLICT (email) DO UPDATE SET 
                    password_hash = EXCLUDED.password_hash,
                    full_name = EXCLUDED.full_name,
                    role = EXCLUDED.role,
                    is_active = true,
                    is_verified = true
                RETURNING id
            """, self.admin_email, password_hash, self.admin_name)
            
            # Add to organization as owner
            await self.connection.execute("""
                INSERT INTO organization_members (organization_id, user_id, role, status)
                VALUES ($1, $2, 'owner', 'active')
                ON CONFLICT (organization_id, user_id) DO UPDATE SET 
                    role = EXCLUDED.role,
                    status = EXCLUDED.status
            """, org_id, admin_id)
            
            # Add to admin workspace with full permissions
            await self.connection.execute("""
                INSERT INTO workspace_members (workspace_id, user_id, role, status, can_read, can_write, can_delete, can_share, can_admin)
                VALUES ($1, $2, 'admin', 'active', true, true, true, true, true)
                ON CONFLICT (workspace_id, user_id) DO UPDATE SET 
                    role = EXCLUDED.role,
                    status = EXCLUDED.status,
                    can_read = EXCLUDED.can_read,
                    can_write = EXCLUDED.can_write,
                    can_delete = EXCLUDED.can_delete,
                    can_share = EXCLUDED.can_share,
                    can_admin = EXCLUDED.can_admin
            """, workspace_id, admin_id)
            
            logger.info(f"✅ Admin user created: {self.admin_email} ({admin_id})")
            return admin_id
            
        except Exception as e:
            logger.error(f"❌ Failed to create admin user: {e}")
            return None
    
    async def create_indexes_and_constraints(self):
        """Create production indexes and constraints"""
        logger.info("📊 Creating production indexes...")
        
        try:
            indexes_sql = """
            -- Performance indexes
            CREATE INDEX IF NOT EXISTS idx_users_email_active ON users(email) WHERE is_active = true;
            CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);
            CREATE INDEX IF NOT EXISTS idx_organizations_slug_active ON organizations(slug) WHERE deleted_at IS NULL;
            CREATE INDEX IF NOT EXISTS idx_workspaces_org_slug ON workspaces(organization_id, slug) WHERE deleted_at IS NULL;
            CREATE INDEX IF NOT EXISTS idx_collections_workspace_tenant ON collections(workspace_id, tenant_id);
            CREATE INDEX IF NOT EXISTS idx_documents_collection_workspace ON documents(collection_id, workspace_id);
            CREATE INDEX IF NOT EXISTS idx_document_chunks_workspace_metadata ON document_chunks(workspace_id) INCLUDE (vector_metadata);
            CREATE INDEX IF NOT EXISTS idx_workspace_members_user_active ON workspace_members(user_id) WHERE status = 'active';
            CREATE INDEX IF NOT EXISTS idx_organization_members_user_active ON organization_members(user_id) WHERE status = 'active';
            
            -- Audit indexes
            CREATE INDEX IF NOT EXISTS idx_data_access_audit_user_time ON data_access_audit(user_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_data_access_audit_workspace_time ON data_access_audit(workspace_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_data_access_audit_action ON data_access_audit(action);
            """
            
            await self.connection.execute(indexes_sql)
            logger.info("✅ Production indexes created")
            
        except Exception as e:
            logger.error(f"❌ Failed to create indexes: {e}")
    
    async def validate_production_setup(self):
        """Validate production setup"""
        logger.info("🔍 Validating production setup...")
        
        try:
            # Check admin user
            admin_exists = await self.connection.fetchval(
                "SELECT EXISTS(SELECT 1 FROM users WHERE email = $1 AND role = 'admin')",
                self.admin_email
            )
            
            # Check organization
            org_exists = await self.connection.fetchval(
                "SELECT EXISTS(SELECT 1 FROM organizations WHERE slug = $1)",
                self.org_slug
            )
            
            # Check functions
            functions_count = await self.connection.fetchval("""
                SELECT COUNT(*) FROM pg_proc p
                JOIN pg_namespace n ON n.oid = p.pronamespace
                WHERE n.nspname = 'public'
                AND p.proname IN ('validate_workspace_access', 'get_tenant_collection_name', 'log_data_access')
            """)
            
            # Check RLS
            rls_count = await self.connection.fetchval("""
                SELECT COUNT(*) FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE n.nspname = 'public' 
                AND c.relname IN ('collections', 'documents', 'document_chunks')
                AND c.relrowsecurity = true
            """)
            
            logger.info(f"✅ Admin user exists: {admin_exists}")
            logger.info(f"✅ Organization exists: {org_exists}")
            logger.info(f"✅ Security functions: {functions_count}/3")
            logger.info(f"✅ RLS enabled: {rls_count}/3")
            
            if admin_exists and org_exists and functions_count == 3:
                logger.info("🎉 Production setup validation PASSED")
                return True
            else:
                logger.error("❌ Production setup validation FAILED")
                return False
                
        except Exception as e:
            logger.error(f"❌ Validation error: {e}")
            return False
    
    async def initialize_production(self):
        """Initialize complete production environment"""
        logger.info("🚀 INITIALIZING PRODUCTION ENVIRONMENT")
        logger.info("=" * 60)
        
        if not await self.connect():
            return False
        
        try:
            # Check if already initialized
            schema_exists = await self.check_existing_schema()
            
            if not schema_exists:
                # Run migrations
                if not await self.run_migrations():
                    return False
            
            # Create organization
            org_id = await self.create_production_organization()
            if not org_id:
                return False
            
            # Create admin workspace
            workspace_id = await self.create_admin_workspace(org_id)
            if not workspace_id:
                return False
            
            # Create admin user
            admin_id = await self.create_admin_user(org_id, workspace_id)
            if not admin_id:
                return False
            
            # Create indexes
            await self.create_indexes_and_constraints()
            
            # Validate setup
            if not await self.validate_production_setup():
                return False
            
            # Success summary
            logger.info("=" * 60)
            logger.info("🎉 PRODUCTION INITIALIZATION COMPLETED!")
            logger.info("=" * 60)
            logger.info(f"🏢 Organization: {self.org_name} ({self.org_slug})")
            logger.info(f"👤 Admin Email: {self.admin_email}")
            logger.info(f"🔑 Admin Password: {self.admin_password}")
            logger.info(f"🔒 Database: {self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}")
            logger.info("=" * 60)
            logger.info("⚠️  IMPORTANT: Save admin credentials securely!")
            logger.info("✅ Production environment is ready for deployment")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Production initialization failed: {e}")
            return False
        
        finally:
            await self.disconnect()

async def main():
    """Main function"""
    print("🚀 COGNIFY PRODUCTION INITIALIZATION")
    print("=" * 60)
    print("This will initialize the production database and create admin user.")
    print()
    print("Environment variables:")
    print(f"  DB_HOST: {os.getenv('DB_HOST', 'localhost')}")
    print(f"  DB_PORT: {os.getenv('DB_PORT', '5432')}")
    print(f"  DB_NAME: {os.getenv('DB_NAME', 'cognify')}")
    print(f"  DB_USER: {os.getenv('DB_USER', 'cognify')}")
    print(f"  ADMIN_EMAIL: {os.getenv('ADMIN_EMAIL', 'admin@cognify.ai')}")
    print(f"  ORG_NAME: {os.getenv('ORG_NAME', 'Cognify')}")
    print()
    
    response = input("Proceed with production initialization? (y/N): ")
    if response.lower() != 'y':
        print("Initialization cancelled.")
        return
    
    initializer = ProductionInitializer()
    success = await initializer.initialize_production()
    
    if success:
        print("\n🎊 Production initialization completed successfully!")
        print("Your Cognify production environment is ready!")
    else:
        print("\n❌ Production initialization failed.")
        print("Please check logs and fix issues before retrying.")

if __name__ == "__main__":
    asyncio.run(main())
