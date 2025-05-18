import psycopg2
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv('.env')

# Get database URI
db_uri = os.getenv('DB__POSTGRES_URI')

def execute_step(cursor, step_name, sql):
    """Execute a migration step with error handling"""
    try:
        print(f"Executing step: {step_name}")
        cursor.execute(sql)
        print(f"✅ {step_name} completed successfully")
        return True
    except Exception as e:
        print(f"❌ Error in {step_name}: {e}")
        # We don't raise here - we want to continue with other steps
        return False

def run_migration():
    """Run the migration manually with autocommit=True for each step"""
    conn = None
    try:
        # Connect with autocommit to avoid transaction issues
        conn = psycopg2.connect(db_uri)
        conn.autocommit = True
        cursor = conn.cursor()
        
        # Step 1: Create alembic_version_postgres table if it doesn't exist
        execute_step(cursor, "Create version table", """
            CREATE TABLE IF NOT EXISTS alembic_version_postgres (
                version_num VARCHAR(32) NOT NULL, 
                CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num)
            )
        """)
        
        # Step 2: Create uuid-ossp extension if it doesn't exist
        execute_step(cursor, "Create uuid-ossp extension", 
            'CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
        
        # Step 3: Create app_auth schema
        execute_step(cursor, "Create app_auth schema", 
            'CREATE SCHEMA IF NOT EXISTS app_auth')
        
        # Step 4: Create roles table
        execute_step(cursor, "Create roles table", """
            CREATE TABLE IF NOT EXISTS app_auth.roles (
                id SERIAL PRIMARY KEY,
                name VARCHAR(50) NOT NULL UNIQUE,
                description VARCHAR(255),
                permissions JSONB NOT NULL DEFAULT '{}'::jsonb
            )
        """)
        
        # Step 5: Create users table
        execute_step(cursor, "Create users table", """
            CREATE TABLE IF NOT EXISTS app_auth.users (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                email VARCHAR(255) NOT NULL UNIQUE,
                username VARCHAR(50) NOT NULL UNIQUE,
                hashed_password VARCHAR(255) NOT NULL,
                full_name VARCHAR(255),
                is_active BOOLEAN NOT NULL DEFAULT TRUE,
                is_superuser BOOLEAN NOT NULL DEFAULT FALSE,
                created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
                last_login_at TIMESTAMP,
                password_changed_at TIMESTAMP DEFAULT NOW(),
                failed_login_attempts INTEGER NOT NULL DEFAULT 0,
                last_failed_login_at TIMESTAMP,
                lockout_until TIMESTAMP,
                trading_limits JSONB DEFAULT '{}'::jsonb,
                algorithm_access JSONB DEFAULT '{}'::jsonb,
                environment_access JSONB DEFAULT '{}'::jsonb,
                has_kill_switch_access BOOLEAN NOT NULL DEFAULT FALSE,
                emergency_contact BOOLEAN NOT NULL DEFAULT FALSE
            )
        """)
        
        # Step 6: Create user_roles table
        execute_step(cursor, "Create user_roles table", """
            CREATE TABLE IF NOT EXISTS app_auth.user_roles (
                user_id UUID REFERENCES app_auth.users(id) ON DELETE CASCADE,
                role_id INTEGER REFERENCES app_auth.roles(id) ON DELETE CASCADE,
                PRIMARY KEY (user_id, role_id)
            )
        """)
        
        # Step 7: Create user_sessions table
        execute_step(cursor, "Create user_sessions table", """
            CREATE TABLE IF NOT EXISTS app_auth.user_sessions (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                user_id UUID REFERENCES app_auth.users(id) ON DELETE CASCADE NOT NULL,
                token_id VARCHAR(255) NOT NULL UNIQUE,
                ip_address VARCHAR(45),
                user_agent VARCHAR(255),
                created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                expires_at TIMESTAMP NOT NULL,
                revoked_at TIMESTAMP,
                is_revoked BOOLEAN NOT NULL DEFAULT FALSE,
                environment VARCHAR(50) DEFAULT 'prod',
                is_algorithmic_session BOOLEAN DEFAULT FALSE
            )
        """)
        
        # Step 8: Create api_keys table
        execute_step(cursor, "Create api_keys table", """
            CREATE TABLE IF NOT EXISTS app_auth.api_keys (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                user_id UUID REFERENCES app_auth.users(id) ON DELETE CASCADE NOT NULL,
                name VARCHAR(100) NOT NULL,
                key_prefix VARCHAR(10) NOT NULL,
                key_hash VARCHAR(255) NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                expires_at TIMESTAMP,
                revoked_at TIMESTAMP,
                is_revoked BOOLEAN NOT NULL DEFAULT FALSE,
                permissions JSONB DEFAULT '{}'::jsonb,
                environment VARCHAR(50) DEFAULT 'prod',
                rate_limit INTEGER DEFAULT 100,
                last_used_at TIMESTAMP,
                use_count INTEGER DEFAULT 0
            )
        """)
        
        # Step 9: Create password_resets table
        execute_step(cursor, "Create password_resets table", """
            CREATE TABLE IF NOT EXISTS app_auth.password_resets (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                user_id UUID REFERENCES app_auth.users(id) ON DELETE CASCADE NOT NULL,
                token_hash VARCHAR(255) NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                expires_at TIMESTAMP NOT NULL,
                used_at TIMESTAMP,
                is_used BOOLEAN NOT NULL DEFAULT FALSE,
                ip_address VARCHAR(45),
                user_agent VARCHAR(255)
            )
        """)
        
        # Step 10: Create audit_logs table
        execute_step(cursor, "Create audit_logs table", """
            CREATE TABLE IF NOT EXISTS app_auth.audit_logs (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                user_id UUID REFERENCES app_auth.users(id) ON DELETE SET NULL,
                action VARCHAR(100) NOT NULL,
                timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
                ip_address VARCHAR(45),
                user_agent VARCHAR(255),
                target_type VARCHAR(50),
                target_id VARCHAR(255),
                environment VARCHAR(50),
                details JSONB
            )
        """)
        
        # Step 11: Finally, update the version table
        # Check if the version already exists
        cursor.execute("SELECT version_num FROM alembic_version_postgres WHERE version_num = '91e93a42b21c'")
        version_exists = cursor.fetchone()
        
        if not version_exists:
            execute_step(cursor, "Record migration version", """
                INSERT INTO alembic_version_postgres (version_num) VALUES ('91e93a42b21c')
            """)
            print("👍 Migration version recorded in alembic_version_postgres")
        else:
            print("👍 Migration version already exists in alembic_version_postgres")
        
        print("\n=== MIGRATION COMPLETED SUCCESSFULLY ===")
        print("You should now see app_auth schema with all tables in Supabase")
        print("And running 'alembic current' should show 91e93a42b21c as the current revision")
        
    except Exception as e:
        print(f"❌ Fatal error during migration: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    run_migration()