"""test_initial_setup

Revision ID: db2c786f49fd
Revises: 
Create Date: 2025-06-07 08:23:28.528658+00:00

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'db2c786f49fd'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """PostgreSQL-specific upgrade operations."""
    pass


def downgrade() -> None:
    """PostgreSQL-specific downgrade operations."""
    pass