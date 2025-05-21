"""create_auth_schema

Revision ID: 91e93a42b21c
Revises: 
Create Date: 2025-04-28 15:57:04.218998+00:00
Database: postgres

This migration creates the auth schema and related tables.
Schema creation will be handled specially by the env.py file.
"""
import logging
from alembic import op
import sqlalchemy as sa
from sqlalchemy import text

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("alembic.migration")

# revision identifiers, used by Alembic.
revision = '91e93a42b21c'
down_revision = None
branch_labels = None
depends_on = None
database = 'postgres'


def upgrade():
    # Get the connection
    connection = op.get_bind()
    
    # Use autocommit for schema and extension creation
    logger.info("Creating extension and schema with autocommit")
    autocommit_conn = connection.execution_options(isolation_level="AUTOCOMMIT")
    
    try:
        # Create uuid-ossp extension
        autocommit_conn.execute(text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"'))
        logger.info("Extension created successfully")
        
        # Create app_auth schema
        autocommit_conn.execute(text('CREATE SCHEMA IF NOT EXISTS app_auth'))
        logger.info("Schema created successfully")
    except Exception as e:
        logger.error(f"Error creating extension or schema: {e}")
        raise
    
    # Now create tables using explicit SQL to avoid any issues with Alembic's table creation
    try:
        logger.info("Creating tables")
        
        # Create roles table
        logger.info("Creating roles table")
        connection.execute(text("""
            CREATE TABLE app_auth.roles (
                id INTEGER PRIMARY KEY,
                name VARCHAR(50) NOT NULL UNIQUE,
                description VARCHAR(255),
                permissions JSON NOT NULL DEFAULT '{}'
            )
        """))
        
        # Create users table
        logger.info("Creating users table")
        connection.execute(text("""
            CREATE TABLE app_auth.users (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                email VARCHAR(255) NOT NULL UNIQUE,
                username VARCHAR(50) NOT NULL UNIQUE,
                hashed_password VARCHAR(255) NOT NULL,
                full_name VARCHAR(255),
                is_active BOOLEAN NOT NULL DEFAULT true,
                is_superuser BOOLEAN NOT NULL DEFAULT false,
                created_at TIMESTAMP NOT NULL DEFAULT now(),
                updated_at TIMESTAMP NOT NULL DEFAULT now(),
                last_login_at TIMESTAMP,
                password_changed_at TIMESTAMP DEFAULT now(),
                failed_login_attempts INTEGER NOT NULL DEFAULT 0,
                last_failed_login_at TIMESTAMP,
                lockout_until TIMESTAMP,
                trading_limits JSON DEFAULT '{}',
                algorithm_access JSON DEFAULT '{}',
                environment_access JSON DEFAULT '{}',
                has_kill_switch_access BOOLEAN NOT NULL DEFAULT false,
                emergency_contact BOOLEAN NOT NULL DEFAULT false
            )
        """))
        
        # Create user_roles association table
        logger.info("Creating user_roles table")
        connection.execute(text("""
            CREATE TABLE app_auth.user_roles (
                user_id UUID REFERENCES app_auth.users(id),
                role_id INTEGER REFERENCES app_auth.roles(id),
                PRIMARY KEY (user_id, role_id)
            )
        """))
        
        # Create user_sessions table
        logger.info("Creating user_sessions table")
        connection.execute(text("""
            CREATE TABLE app_auth.user_sessions (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                user_id UUID REFERENCES app_auth.users(id) NOT NULL,
                token_id VARCHAR(255) NOT NULL UNIQUE,
                ip_address VARCHAR(45),
                user_agent VARCHAR(255),
                created_at TIMESTAMP NOT NULL DEFAULT now(),
                expires_at TIMESTAMP NOT NULL,
                revoked_at TIMESTAMP,
                is_revoked BOOLEAN NOT NULL DEFAULT false,
                environment VARCHAR(50) DEFAULT 'prod',
                is_algorithmic_session BOOLEAN DEFAULT false
            )
        """))
        
        # Create api_keys table
        logger.info("Creating api_keys table")
        connection.execute(text("""
            CREATE TABLE app_auth.api_keys (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                user_id UUID REFERENCES app_auth.users(id) NOT NULL,
                name VARCHAR(100) NOT NULL,
                key_prefix VARCHAR(10) NOT NULL,
                key_hash VARCHAR(255) NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT now(),
                expires_at TIMESTAMP,
                revoked_at TIMESTAMP,
                is_revoked BOOLEAN NOT NULL DEFAULT false,
                permissions JSON DEFAULT '{}',
                environment VARCHAR(50) DEFAULT 'prod',
                rate_limit INTEGER DEFAULT 100,
                last_used_at TIMESTAMP,
                use_count INTEGER DEFAULT 0
            )
        """))
        
        # Create password_resets table
        logger.info("Creating password_resets table")
        connection.execute(text("""
            CREATE TABLE app_auth.password_resets (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                user_id UUID REFERENCES app_auth.users(id) NOT NULL,
                token_hash VARCHAR(255) NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT now(),
                expires_at TIMESTAMP NOT NULL,
                used_at TIMESTAMP,
                is_used BOOLEAN NOT NULL DEFAULT false,
                ip_address VARCHAR(45),
                user_agent VARCHAR(255)
            )
        """))
        
        # Create audit_logs table
        logger.info("Creating audit_logs table")
        connection.execute(text("""
            CREATE TABLE app_auth.audit_logs (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                user_id UUID REFERENCES app_auth.users(id),
                action VARCHAR(100) NOT NULL,
                timestamp TIMESTAMP NOT NULL DEFAULT now(),
                ip_address VARCHAR(45),
                user_agent VARCHAR(255),
                target_type VARCHAR(50),
                target_id VARCHAR(255),
                environment VARCHAR(50),
                details JSON
            )
        """))
        
        logger.info("All tables created successfully")
        
    except Exception as e:
        logger.error(f"Error creating tables: {e}")
        raise


def downgrade():
    """Drop all tables and schema in reverse order."""
    connection = op.get_bind()
    
    # Drop all tables first
    logger.info("Dropping tables")
    for table in ['audit_logs', 'password_resets', 'api_keys', 'user_sessions', 
                  'user_roles', 'users', 'roles']:
        try:
            connection.execute(text(f"DROP TABLE IF EXISTS app_auth.{table} CASCADE"))
            logger.info(f"Dropped table {table}")
        except Exception as e:
            logger.error(f"Error dropping table {table}: {e}")
    
    # Drop schema with autocommit
    logger.info("Dropping schema")
    try:
        autocommit_conn = connection.execution_options(isolation_level="AUTOCOMMIT")
        autocommit_conn.execute(text("DROP SCHEMA IF EXISTS app_auth CASCADE"))
        logger.info("Dropped schema app_auth")
    except Exception as e:
        logger.error(f"Error dropping schema: {e}")