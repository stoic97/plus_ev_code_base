"""create_auth_schema

Revision ID: 91e93a42b21c
Revises: 
Create Date: 2025-04-28 15:57:04.218998+00:00
Database: postgres

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '91e93a42b21c'
down_revision = None
branch_labels = None
depends_on = None
database = 'postgres'


def upgrade():
    # Create uuid-ossp extension for UUID generation if not exists
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
    
    # Create schema for authentication
    op.execute('CREATE SCHEMA IF NOT EXISTS auth')
    
    # Create roles table
    op.create_table('roles',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('name', sa.String(50), nullable=False, unique=True, index=True),
        sa.Column('description', sa.String(255)),
        sa.Column('permissions', sa.JSON(), nullable=False, server_default='{}'),
        schema='auth'
    )
    
    # Create users table
    op.create_table('users',
        sa.Column('id', sa.UUID(), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column('email', sa.String(255), nullable=False, unique=True, index=True),
        sa.Column('username', sa.String(50), nullable=False, unique=True, index=True),
        sa.Column('hashed_password', sa.String(255), nullable=False),
        sa.Column('full_name', sa.String(255)),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column('is_superuser', sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.Column('last_login_at', sa.DateTime()),
        sa.Column('password_changed_at', sa.DateTime(), server_default=sa.text("now()")),
        sa.Column('failed_login_attempts', sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column('last_failed_login_at', sa.DateTime()),
        sa.Column('lockout_until', sa.DateTime()),
        sa.Column('trading_limits', sa.JSON(), server_default='{}'),
        sa.Column('algorithm_access', sa.JSON(), server_default='{}'),
        sa.Column('environment_access', sa.JSON(), server_default='{}'),
        sa.Column('has_kill_switch_access', sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column('emergency_contact', sa.Boolean(), nullable=False, server_default=sa.text("false")),
        schema='auth'
    )
    
    # Create user_roles association table for many-to-many relationship
    op.create_table('user_roles',
        sa.Column('user_id', sa.UUID(), sa.ForeignKey('auth.users.id'), primary_key=True),
        sa.Column('role_id', sa.Integer(), sa.ForeignKey('auth.roles.id'), primary_key=True),
        schema='auth'
    )
    
    # Create user_sessions table
    op.create_table('user_sessions',
        sa.Column('id', sa.UUID(), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column('user_id', sa.UUID(), sa.ForeignKey('auth.users.id'), nullable=False),
        sa.Column('token_id', sa.String(255), nullable=False, unique=True),
        sa.Column('ip_address', sa.String(45)),
        sa.Column('user_agent', sa.String(255)),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.Column('expires_at', sa.DateTime(), nullable=False),
        sa.Column('revoked_at', sa.DateTime()),
        sa.Column('is_revoked', sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column('environment', sa.String(50), server_default="prod"),
        sa.Column('is_algorithmic_session', sa.Boolean(), server_default=sa.text("false")),
        schema='auth'
    )
    
    # Create api_keys table
    op.create_table('api_keys',
        sa.Column('id', sa.UUID(), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column('user_id', sa.UUID(), sa.ForeignKey('auth.users.id'), nullable=False),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('key_prefix', sa.String(10), nullable=False),
        sa.Column('key_hash', sa.String(255), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.Column('expires_at', sa.DateTime()),
        sa.Column('revoked_at', sa.DateTime()),
        sa.Column('is_revoked', sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column('permissions', sa.JSON(), server_default='{}'),
        sa.Column('environment', sa.String(50), server_default="prod"),
        sa.Column('rate_limit', sa.Integer(), server_default=sa.text("100")),
        sa.Column('last_used_at', sa.DateTime()),
        sa.Column('use_count', sa.Integer(), server_default=sa.text("0")),
        schema='auth'
    )
    
    # Create password_resets table
    op.create_table('password_resets',
        sa.Column('id', sa.UUID(), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column('user_id', sa.UUID(), sa.ForeignKey('auth.users.id'), nullable=False),
        sa.Column('token_hash', sa.String(255), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.Column('expires_at', sa.DateTime(), nullable=False),
        sa.Column('used_at', sa.DateTime()),
        sa.Column('is_used', sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column('ip_address', sa.String(45)),
        sa.Column('user_agent', sa.String(255)),
        schema='auth'
    )
    
    # Create audit_logs table
    op.create_table('audit_logs',
        sa.Column('id', sa.UUID(), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column('user_id', sa.UUID(), sa.ForeignKey('auth.users.id')),
        sa.Column('action', sa.String(100), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.Column('ip_address', sa.String(45)),
        sa.Column('user_agent', sa.String(255)),
        sa.Column('target_type', sa.String(50)),
        sa.Column('target_id', sa.String(255)),
        sa.Column('environment', sa.String(50)),
        sa.Column('details', sa.JSON()),
        schema='auth'
    )


def downgrade():
    # Drop all tables in reverse order to respect foreign key constraints
    op.drop_table('audit_logs', schema='auth')
    op.drop_table('password_resets', schema='auth')
    op.drop_table('api_keys', schema='auth')
    op.drop_table('user_sessions', schema='auth')
    op.drop_table('user_roles', schema='auth')
    op.drop_table('users', schema='auth')
    op.drop_table('roles', schema='auth')
    
    # Drop schema
    op.execute('DROP SCHEMA IF EXISTS auth CASCADE')