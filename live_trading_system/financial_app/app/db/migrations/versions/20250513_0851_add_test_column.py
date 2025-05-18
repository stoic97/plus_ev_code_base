"""add_test_column

Revision ID: 823ae8172aef
Revises: 91e93a42b21c
Create Date: 2025-05-13 08:51:45.893848+00:00
Database: postgres

This migration adds a test column to verify the migration framework is working correctly.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import text

# revision identifiers, used by Alembic.
revision = '823ae8172aef'
down_revision = '91e93a42b21c'
branch_labels = None
depends_on = None
database = 'postgres'


def upgrade():
    """Add a test column to the users table in app_auth schema."""
    try:
        # Log start of migration
        op.execute(text("SELECT 1"))  # Dummy query to check if connection works
        
        # Add the test column
        op.add_column('users', 
                     sa.Column('test_column', sa.String(50), nullable=True),
                     schema='app_auth')
        
        # Log successful completion
        print("Successfully added test_column to app_auth.users table")
    except Exception as e:
        print(f"Error adding test column: {e}")
        raise


def downgrade():
    """Remove the test column from the users table."""
    try:
        # Drop the test column
        op.drop_column('users', 'test_column', schema='app_auth')
        
        print("Successfully removed test_column from app_auth.users table")
    except Exception as e:
        print(f"Error removing test column: {e}")
        raise