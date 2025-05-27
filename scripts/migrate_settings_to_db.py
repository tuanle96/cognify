#!/usr/bin/env python3
"""
Migrate Settings to Database
Move hardcoded settings from environment variables to database
"""

import asyncio
import asyncpg
import os
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SettingsMigrator:
    """Migrate settings from environment to database"""
    
    def __init__(self):
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', 5433)),
            'database': os.getenv('DB_NAME', 'cognify_test'),
            'user': os.getenv('DB_USER', 'cognify_test'),
            'password': os.getenv('DB_PASSWORD', 'test_password')
        }
        self.connection = None
    
    async def connect(self):
        """Connect to database"""
        try:
            self.connection = await asyncpg.connect(**self.db_config)
            logger.info("✅ Connected to database")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from database"""
        if self.connection:
            await self.connection.close()
            logger.info("✅ Disconnected from database")
    
    async def run_settings_migration(self):
        """Run the settings table migration"""
        logger.info("🔄 Running settings migration...")
        
        try:
            # Read migration file
            migration_file = "migrations/003_create_settings_system.sql"
            
            if not os.path.exists(migration_file):
                logger.error(f"❌ Migration file not found: {migration_file}")
                return False
            
            with open(migration_file, 'r') as f:
                migration_sql = f.read()
            
            # Execute migration
            async with self.connection.transaction():
                await self.connection.execute(migration_sql)
            
            logger.info("✅ Settings migration completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Migration failed: {e}")
            return False
    
    async def migrate_environment_settings(self):
        """Migrate settings from environment variables"""
        logger.info("🔄 Migrating environment settings to database...")
        
        try:
            # Define environment variable mappings
            env_mappings = [
                # OpenAI Settings
                ('OPENAI_API_KEY', 'openai_api_key', 'encrypted'),
                ('OPENAI_BASE_URL', 'openai_base_url', 'string'),
                ('OPENAI_MODEL', 'openai_model', 'string'),
                ('OPENAI_EMBEDDING_MODEL', 'openai_embedding_model', 'string'),
                ('OPENAI_MAX_TOKENS', 'openai_max_tokens', 'integer'),
                ('OPENAI_TEMPERATURE', 'openai_temperature', 'float'),
                
                # Anthropic Settings
                ('ANTHROPIC_API_KEY', 'anthropic_api_key', 'encrypted'),
                ('ANTHROPIC_MODEL', 'anthropic_model', 'string'),
                
                # Vector Database Settings
                ('QDRANT_URL', 'qdrant_url', 'string'),
                ('QDRANT_API_KEY', 'qdrant_api_key', 'encrypted'),
                ('VECTOR_DIMENSION', 'vector_dimension', 'integer'),
                
                # Security Settings
                ('JWT_EXPIRATION_HOURS', 'jwt_expiration_hours', 'integer'),
                ('MAX_LOGIN_ATTEMPTS', 'max_login_attempts', 'integer'),
                ('PASSWORD_MIN_LENGTH', 'password_min_length', 'integer'),
                ('RATE_LIMIT_REQUESTS_PER_MINUTE', 'rate_limit_requests_per_minute', 'integer'),
                
                # Performance Settings
                ('CHUNKING_MAX_SIZE', 'chunking_max_size', 'integer'),
                ('CACHE_TTL_SECONDS', 'cache_ttl_seconds', 'integer'),
                
                # Feature Flags
                ('ENABLE_MULTIMODAL', 'enable_multimodal', 'boolean'),
                ('ENABLE_REAL_TIME_SYNC', 'enable_real_time_sync', 'boolean'),
                ('ENABLE_ADVANCED_SEARCH', 'enable_advanced_search', 'boolean'),
                
                # Notification Settings
                ('SMTP_HOST', 'smtp_host', 'string'),
                ('SMTP_PORT', 'smtp_port', 'integer'),
                ('SMTP_USERNAME', 'smtp_username', 'string'),
                ('SMTP_PASSWORD', 'smtp_password', 'encrypted'),
            ]
            
            migrated_count = 0
            
            for env_var, setting_key, data_type in env_mappings:
                env_value = os.getenv(env_var)
                
                if env_value:
                    # Convert value based on data type
                    converted_value = self._convert_value(env_value, data_type)
                    
                    # Update setting in database
                    success = await self._update_setting_value(setting_key, converted_value)
                    
                    if success:
                        migrated_count += 1
                        logger.info(f"✅ Migrated {env_var} → {setting_key}")
                    else:
                        logger.warning(f"⚠️  Failed to migrate {env_var}")
                else:
                    logger.debug(f"🔍 Environment variable {env_var} not set")
            
            logger.info(f"✅ Migrated {migrated_count} settings from environment")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error migrating environment settings: {e}")
            return False
    
    def _convert_value(self, value: str, data_type: str):
        """Convert string value to appropriate type"""
        try:
            if data_type == 'integer':
                return int(value)
            elif data_type == 'float':
                return float(value)
            elif data_type == 'boolean':
                return value.lower() in ('true', '1', 'yes', 'on')
            else:
                return value
        except (ValueError, TypeError):
            return value
    
    async def _update_setting_value(self, key: str, value) -> bool:
        """Update setting value in database"""
        try:
            # Check if setting exists
            setting_exists = await self.connection.fetchval("""
                SELECT EXISTS(SELECT 1 FROM system_settings WHERE key = $1)
            """, key)
            
            if setting_exists:
                # Update existing setting
                await self.connection.execute("""
                    UPDATE system_settings 
                    SET value = $2, updated_at = NOW()
                    WHERE key = $1
                """, key, str(value))
            else:
                logger.warning(f"⚠️  Setting {key} not found in database schema")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error updating setting {key}: {e}")
            return False
    
    async def validate_migration(self):
        """Validate that migration was successful"""
        logger.info("🔍 Validating settings migration...")
        
        try:
            # Check if settings table exists
            table_exists = await self.connection.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'system_settings'
                );
            """)
            
            if not table_exists:
                logger.error("❌ Settings table does not exist")
                return False
            
            # Count settings
            settings_count = await self.connection.fetchval("""
                SELECT COUNT(*) FROM system_settings
            """)
            
            # Count settings with values
            valued_settings = await self.connection.fetchval("""
                SELECT COUNT(*) FROM system_settings WHERE value IS NOT NULL AND value != ''
            """)
            
            # Check critical settings
            critical_settings = [
                'openai_api_key',
                'default_llm_provider',
                'vector_db_type'
            ]
            
            critical_count = 0
            for setting in critical_settings:
                value = await self.connection.fetchval("""
                    SELECT value FROM system_settings WHERE key = $1
                """, setting)
                
                if value:
                    critical_count += 1
                    logger.info(f"✅ Critical setting {setting}: configured")
                else:
                    logger.warning(f"⚠️  Critical setting {setting}: not configured")
            
            logger.info(f"📊 Validation Results:")
            logger.info(f"   Total settings: {settings_count}")
            logger.info(f"   Configured settings: {valued_settings}")
            logger.info(f"   Critical settings: {critical_count}/{len(critical_settings)}")
            
            if settings_count > 0 and critical_count >= 2:
                logger.info("✅ Migration validation PASSED")
                return True
            else:
                logger.error("❌ Migration validation FAILED")
                return False
            
        except Exception as e:
            logger.error(f"❌ Validation error: {e}")
            return False
    
    async def show_current_settings(self):
        """Show current settings in database"""
        logger.info("📋 Current Settings in Database:")
        
        try:
            settings = await self.connection.fetch("""
                SELECT category, key, name, value, is_sensitive, data_type
                FROM system_settings 
                ORDER BY category, name
            """)
            
            current_category = None
            for setting in settings:
                if setting['category'] != current_category:
                    current_category = setting['category']
                    logger.info(f"\n📂 {current_category.upper()}:")
                
                value_display = setting['value']
                if setting['is_sensitive'] and value_display:
                    value_display = '***' + value_display[-4:] if len(value_display) > 4 else '***'
                elif not value_display:
                    value_display = '(not set)'
                
                logger.info(f"   {setting['name']}: {value_display}")
            
        except Exception as e:
            logger.error(f"❌ Error showing settings: {e}")
    
    async def run_complete_migration(self):
        """Run complete settings migration process"""
        logger.info("🚀 STARTING SETTINGS MIGRATION TO DATABASE")
        logger.info("=" * 60)
        
        if not await self.connect():
            return False
        
        try:
            # Step 1: Run settings table migration
            if not await self.run_settings_migration():
                return False
            
            # Step 2: Migrate environment settings
            if not await self.migrate_environment_settings():
                return False
            
            # Step 3: Validate migration
            if not await self.validate_migration():
                return False
            
            # Step 4: Show current settings
            await self.show_current_settings()
            
            logger.info("\n" + "=" * 60)
            logger.info("🎉 SETTINGS MIGRATION COMPLETED SUCCESSFULLY!")
            logger.info("=" * 60)
            logger.info("✅ Settings are now stored in database")
            logger.info("✅ Admin can manage settings via API")
            logger.info("✅ Dynamic configuration is active")
            logger.info("\n📋 Next Steps:")
            logger.info("1. Update application to use DynamicLLMService")
            logger.info("2. Configure settings via admin API")
            logger.info("3. Test LLM provider connections")
            logger.info("4. Remove hardcoded environment variables")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Migration failed: {e}")
            return False
        
        finally:
            await self.disconnect()

async def main():
    """Main function"""
    print("🔧 COGNIFY SETTINGS MIGRATION")
    print("=" * 50)
    print("This will migrate settings from environment variables to database.")
    print("Settings will be dynamically configurable by admin users.")
    print()
    
    response = input("Proceed with settings migration? (y/N): ")
    if response.lower() != 'y':
        print("Migration cancelled.")
        return
    
    migrator = SettingsMigrator()
    success = await migrator.run_complete_migration()
    
    if success:
        print("\n🎊 Settings migration completed successfully!")
        print("Your Cognify instance now supports dynamic configuration!")
    else:
        print("\n❌ Settings migration failed.")
        print("Please check logs and fix issues before retrying.")

if __name__ == "__main__":
    asyncio.run(main())
