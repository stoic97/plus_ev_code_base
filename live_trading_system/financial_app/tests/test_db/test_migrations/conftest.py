"""
Test fixtures specific to migration testing.
These fixtures provide common functionality needed across migration test files.
"""

import os
import sys
import pytest
import tempfile
import subprocess
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
import alembic
from alembic.config import Config
from alembic.script import ScriptDirectory
from alembic.runtime.migration import MigrationContext

# Add the project root directory to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def mock_alembic_context():
    """Create a mocked Alembic context object."""
    mock_context = MagicMock(spec=alembic.context)
    
    # Set up context attributes
    mock_context.config = MagicMock()
    mock_context.script = MagicMock()
    mock_context.get_current_revision = MagicMock(return_value=None)
    
    # Transaction context manager
    tx_context = MagicMock()
    tx_context.__enter__ = MagicMock(return_value=None)
    tx_context.__exit__ = MagicMock(return_value=None)
    mock_context.begin_transaction = MagicMock(return_value=tx_context)
    
    return mock_context


@pytest.fixture
def temp_alembic_structure():
    """
    Create a temporary directory with a complete Alembic structure for testing.
    """
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    
    # Create the Alembic directory structure
    migrations_dir = os.path.join(temp_dir, "migrations")
    versions_dir = os.path.join(migrations_dir, "versions")
    os.makedirs(versions_dir)
    
    # Create alembic.ini file
    alembic_ini = os.path.join(temp_dir, "alembic.ini")
    with open(alembic_ini, "w") as f:
        f.write("[alembic]\n")
        f.write(f"script_location = {migrations_dir}\n")
        f.write("sqlalchemy.url = postgresql://postgres:postgres@localhost:5432/test_db\n")
        f.write("prepend_sys_path = .\n")
    
    # Create env.py file
    env_py = os.path.join(migrations_dir, "env.py")
    with open(env_py, "w") as f:
        f.write("""
from alembic import context
from sqlalchemy import engine_from_config, pool

config = context.config

def run_migrations_offline():
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=None,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    
    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online():
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=None
        )
        
        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
""")
    
    # Create script.py.mako template file
    template_dir = os.path.join(migrations_dir, "script.py.mako")
    with open(template_dir, "w") as f:
        f.write("""
\"\"\"${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
Create Date: ${create_date}

\"\"\"
from alembic import op
import sqlalchemy as sa
${imports if imports else ""}

# revision identifiers, used by Alembic.
revision = ${repr(up_revision)}
down_revision = ${repr(down_revision)}
branch_labels = ${repr(branch_labels)}
depends_on = ${repr(depends_on)}


def upgrade():
    ${upgrades if upgrades else "pass"}


def downgrade():
    ${downgrades if downgrades else "pass"}
""")
    
    yield temp_dir
    
    # Clean up
    shutil.rmtree(temp_dir)


@pytest.fixture
def alembic_config(temp_alembic_structure):
    """Create an Alembic config using the temporary structure."""
    alembic_ini = os.path.join(temp_alembic_structure, "alembic.ini")
    config = Config(alembic_ini)
    return config


@pytest.fixture
def alembic_script_directory(alembic_config):
    """Create an Alembic script directory from the config."""
    script_dir = ScriptDirectory.from_config(alembic_config)
    return script_dir


@pytest.fixture
def sample_migrations(temp_alembic_structure):
    """Create sample migration files for testing."""
    versions_dir = os.path.join(temp_alembic_structure, "migrations", "versions")
    
    # Create a base migration
    base_file = os.path.join(versions_dir, "a1b2c3d4e5f6_create_users_table.py")
    with open(base_file, "w") as f:
        f.write("""
\"\"\"create users table

Revision ID: a1b2c3d4e5f6
Revises: 
Create Date: 2023-01-01 00:00:00.000000

\"\"\"
from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = 'a1b2c3d4e5f6'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    op.create_table(
        'users',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('username', sa.String(50), nullable=False),
        sa.Column('email', sa.String(120), nullable=False)
    )

def downgrade():
    op.drop_table('users')
""")
    
    # Create a second migration that depends on the first
    second_file = os.path.join(versions_dir, "b2c3d4e5f6a1_add_user_roles.py")
    with open(second_file, "w") as f:
        f.write("""
\"\"\"add user roles

Revision ID: b2c3d4e5f6a1
Revises: a1b2c3d4e5f6
Create Date: 2023-01-02 00:00:00.000000

\"\"\"
from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = 'b2c3d4e5f6a1'
down_revision = 'a1b2c3d4e5f6'
branch_labels = None
depends_on = None

def upgrade():
    op.create_table(
        'roles',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('name', sa.String(50), nullable=False)
    )
    op.create_table(
        'user_roles',
        sa.Column('user_id', sa.Integer, sa.ForeignKey('users.id'), primary_key=True),
        sa.Column('role_id', sa.Integer, sa.ForeignKey('roles.id'), primary_key=True)
    )

def downgrade():
    op.drop_table('user_roles')
    op.drop_table('roles')
""")
    
    # Create a third migration that adds a TimescaleDB hypertable
    third_file = os.path.join(versions_dir, "c3d4e5f6a1b2_add_metrics_table.py")
    with open(third_file, "w") as f:
        f.write("""
\"\"\"add metrics table

Revision ID: c3d4e5f6a1b2
Revises: b2c3d4e5f6a1
Create Date: 2023-01-03 00:00:00.000000

\"\"\"
from alembic import op
import sqlalchemy as sa
from app.db.migrations.helpers.timescale import create_hypertable

# revision identifiers
revision = 'c3d4e5f6a1b2'
down_revision = 'b2c3d4e5f6a1'
branch_labels = None
depends_on = None

def upgrade():
    op.create_table(
        'metrics',
        sa.Column('time', sa.TIMESTAMP, primary_key=True),
        sa.Column('user_id', sa.Integer, sa.ForeignKey('users.id'), primary_key=True),
        sa.Column('metric_name', sa.String(50), primary_key=True),
        sa.Column('value', sa.Float, nullable=False)
    )
    
    # Convert to TimescaleDB hypertable
    create_hypertable(op, 'metrics', 'time')

def downgrade():
    op.drop_table('metrics')
""")
    
    # Return the migration files
    return [base_file, second_file, third_file]


@pytest.fixture
def migration_files_content(sample_migrations):
    """Return the content of the sample migration files."""
    contents = {}
    for file_path in sample_migrations:
        with open(file_path, 'r') as f:
            contents[os.path.basename(file_path)] = f.read()
    return contents


@pytest.fixture
def mock_op():
    """Create a mock Alembic operations object."""
    mock = MagicMock(spec=alembic.operations.Operations)
    mock.execute = MagicMock()
    mock.create_table = MagicMock()
    mock.drop_table = MagicMock()
    mock.add_column = MagicMock()
    mock.drop_column = MagicMock()
    return mock


@pytest.fixture
def migration_diff_file():
    """Generate a sample migration diff file for testing."""
    diff = """
[added] Table 'users'
    id INTEGER
    username VARCHAR(50) NOT NULL
    email VARCHAR(120) NOT NULL
    
[added] Table 'roles'
    id INTEGER
    name VARCHAR(50) NOT NULL
    
[added] Table 'user_roles'
    user_id INTEGER REFERENCES users(id)
    role_id INTEGER REFERENCES roles(id)
"""
    
    # Create a temporary file with the diff content
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    with open(temp_file.name, 'w') as f:
        f.write(diff)
    
    yield temp_file.name
    
    # Clean up
    os.unlink(temp_file.name)


@pytest.fixture
def migration_execution_plan():
    """Generate a sample migration execution plan for testing."""
    plan = """
Operations to perform:
  Target revision: c3d4e5f6a1b2 (head)
  Migration scripts:
    -> a1b2c3d4e5f6 (create users table)
    -> b2c3d4e5f6a1 (add user roles)
    -> c3d4e5f6a1b2 (add metrics table)
"""
    
    # Create a temporary file with the plan content
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    with open(temp_file.name, 'w') as f:
        f.write(plan)
    
    yield temp_file.name
    
    # Clean up
    os.unlink(temp_file.name)


@pytest.fixture
def mock_migration_helpers():
    """Mock all migration helper functions."""
    with patch("app.db.migrations.helpers.timescale.create_hypertable") as mock_create_hypertable, \
         patch("app.db.migrations.helpers.timescale.add_compression_policy") as mock_add_compression, \
         patch("app.db.migrations.helpers.db_init.ensure_schema_exists") as mock_ensure_schema, \
         patch("app.db.migrations.helpers.db_init.create_extension") as mock_create_extension, \
         patch("app.db.migrations.helpers.alembic_utils.get_alembic_config") as mock_get_config:
        
        # Configure default returns
        mock_get_config.return_value = MagicMock(spec=Config)
        
        # Return all mocks as a dictionary
        yield {
            "create_hypertable": mock_create_hypertable,
            "add_compression_policy": mock_add_compression,
            "ensure_schema_exists": mock_ensure_schema,
            "create_extension": mock_create_extension,
            "get_alembic_config": mock_get_config
        }