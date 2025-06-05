"""
Tests for the core migration module of the financial application.
"""

import pytest
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import tempfile
import shutil
import alembic
from alembic.config import Config
from datetime import datetime

# Import the module to test
from financial_app.app.core.migrations import (
    MigrationManager,
    DatabaseMigrationError,
    init_migrations,
    run_migrations_cli,
    get_migration_status
)


class TestCoreDbMigration:
    """Test suite for core database migration functionality."""

    @pytest.fixture
    def mock_alembic_config(self):
        """Create a mock Alembic configuration."""
        config = MagicMock(spec=Config)
        config.get_main_option.return_value = "app/db/migrations"
        config.get_section_option.return_value = None
        config.config_file_name = "alembic.ini"
        return config

    @pytest.fixture
    def mock_migration_manager(self, mock_alembic_config):
        """Create a mocked MigrationManager instance."""
        with patch("app.core.migration.get_alembic_config") as mock_get_config:
            mock_get_config.return_value = mock_alembic_config
            
            manager = MigrationManager()
            yield manager

    def test_migration_manager_init(self):
        """Test initializing the MigrationManager."""
        with patch("app.core.migration.get_alembic_config") as mock_get_config:
            mock_config = MagicMock()
            mock_get_config.return_value = mock_config
            
            # Initialize MigrationManager
            manager = MigrationManager()
            
            # Assertions
            assert manager.config == mock_config
            assert manager.alembic_location is not None

    def test_migration_manager_upgrade(self, mock_migration_manager):
        """Test the upgrade method of MigrationManager."""
        with patch("app.core.migration.run_migrations") as mock_run:
            # Call the method to test
            mock_migration_manager.upgrade("head")
            
            # Verify run_migrations was called correctly
            mock_run.assert_called_once_with(
                mock_migration_manager.config, 
                "upgrade", 
                "head"
            )

    def test_migration_manager_downgrade(self, mock_migration_manager):
        """Test the downgrade method of MigrationManager."""
        with patch("app.core.migration.run_migrations") as mock_run:
            # Call the method to test
            mock_migration_manager.downgrade("base")
            
            # Verify run_migrations was called correctly
            mock_run.assert_called_once_with(
                mock_migration_manager.config, 
                "downgrade", 
                "base"
            )

    def test_migration_manager_generate(self, mock_migration_manager):
        """Test generating a migration script."""
        with patch("app.core.migration.generate_migration_script") as mock_generate:
            # Call the method to test
            mock_migration_manager.generate("new_migration", autogenerate=True)
            
            # Verify generate_migration_script was called correctly
            mock_generate.assert_called_once_with(
                mock_migration_manager.config,
                "new_migration",
                autogenerate=True
            )

    def test_migration_manager_current(self, mock_migration_manager):
        """Test getting the current migration version."""
        with patch("app.core.migration.get_current_revision") as mock_get_current:
            # Set the return value
            mock_get_current.return_value = "abc123"
            
            # Call the method to test
            result = mock_migration_manager.current()
            
            # Assertions
            assert result == "abc123"
            mock_get_current.assert_called_once_with(mock_migration_manager.config)

    def test_migration_manager_status(self, mock_migration_manager):
        """Test getting migration status."""
        with patch("app.core.migration.check_migration_history") as mock_check:
            # Set the return value
            mock_check.return_value = {
                "current": "abc123",
                "latest": "def456",
                "status": "behind",
                "pending": ["def456"]
            }
            
            # Call the method to test
            result = mock_migration_manager.status()
            
            # Assertions
            assert result == mock_check.return_value
            mock_check.assert_called_once_with(mock_migration_manager.config)

    def test_migration_manager_verify(self, mock_migration_manager):
        """Test verifying the migration chain."""
        with patch("app.core.migration.verify_migration_chain") as mock_verify:
            # Set the return value
            mock_verify.return_value = {
                "valid": True,
                "revisions": ["abc123", "def456"],
                "errors": []
            }
            
            # Call the method to test
            result = mock_migration_manager.verify()
            
            # Assertions
            assert result == mock_verify.return_value
            mock_verify.assert_called_once_with(mock_migration_manager.config)

    def test_migration_manager_stamp(self, mock_migration_manager):
        """Test stamping a revision."""
        with patch("app.core.migration.stamp_revision") as mock_stamp:
            # Call the method to test
            mock_migration_manager.stamp("abc123")
            
            # Verify stamp_revision was called correctly
            mock_stamp.assert_called_once_with(
                mock_migration_manager.config, 
                "abc123"
            )

    def test_init_migrations(self):
        """Test initializing the migration environment."""
        with patch("app.core.migration.MigrationManager") as mock_manager_class:
            # Set up the mock
            mock_manager = MagicMock()
            mock_manager_class.return_value = mock_manager
            
            # Call the function to test
            result = init_migrations()
            
            # Assertions
            assert result == mock_manager
            mock_manager_class.assert_called_once()

    def test_get_migration_status(self):
        """Test getting the migration status."""
        with patch("app.core.migration.MigrationManager") as mock_manager_class:
            # Set up the mock
            mock_manager = MagicMock()
            mock_manager.status.return_value = {
                "current": "abc123",
                "latest": "def456",
                "status": "behind",
                "pending": ["def456"]
            }
            mock_manager_class.return_value = mock_manager
            
            # Call the function to test
            result = get_migration_status()
            
            # Assertions
            assert result == mock_manager.status.return_value
            mock_manager.status.assert_called_once()

    def test_run_migrations_cli_upgrade(self):
        """Test running migrations via CLI - upgrade command."""
        with patch("app.core.migration.MigrationManager") as mock_manager_class:
            # Set up the mock
            mock_manager = MagicMock()
            mock_manager_class.return_value = mock_manager
            
            # Call the function with upgrade command
            run_migrations_cli(["upgrade", "head"])
            
            # Verify the correct method was called
            mock_manager.upgrade.assert_called_once_with("head")

    def test_run_migrations_cli_downgrade(self):
        """Test running migrations via CLI - downgrade command."""
        with patch("app.core.migration.MigrationManager") as mock_manager_class:
            # Set up the mock
            mock_manager = MagicMock()
            mock_manager_class.return_value = mock_manager
            
            # Call the function with downgrade command
            run_migrations_cli(["downgrade", "base"])
            
            # Verify the correct method was called
            mock_manager.downgrade.assert_called_once_with("base")

    def test_run_migrations_cli_generate(self):
        """Test running migrations via CLI - generate command."""
        with patch("app.core.migration.MigrationManager") as mock_manager_class:
            # Set up the mock
            mock_manager = MagicMock()
            mock_manager_class.return_value = mock_manager
            
            # Call the function with generate command
            run_migrations_cli(["generate", "new_migration", "--autogenerate"])
            
            # Verify the correct method was called
            mock_manager.generate.assert_called_once_with("new_migration", autogenerate=True)

    def test_run_migrations_cli_status(self):
        """Test running migrations via CLI - status command."""
        with patch("app.core.migration.MigrationManager") as mock_manager_class:
            # Set up the mock
            mock_manager = MagicMock()
            mock_manager.status.return_value = {
                "current": "abc123",
                "latest": "def456",
                "status": "behind",
                "pending": ["def456"]
            }
            mock_manager_class.return_value = mock_manager
            
            # Call the function with status command
            run_migrations_cli(["status"])
            
            # Verify the correct method was called
            mock_manager.status.assert_called_once()

    def test_run_migrations_cli_invalid_command(self):
        """Test running migrations with an invalid command."""
        with patch("app.core.migration.MigrationManager") as mock_manager_class:
            # Set up the mock
            mock_manager = MagicMock()
            mock_manager_class.return_value = mock_manager
            
            # Call the function with an invalid command
            with pytest.raises(ValueError, match="Invalid command: invalid_command"):
                run_migrations_cli(["invalid_command"])

    def test_error_handling_during_migration(self):
        """Test error handling during migrations."""
        with patch("app.core.migration.MigrationManager") as mock_manager_class:
            # Set up the mock to raise an exception
            mock_manager = MagicMock()
            mock_manager.upgrade.side_effect = Exception("Migration failed")
            mock_manager_class.return_value = mock_manager
            
            # Check that the exception is converted to DatabaseMigrationError
            with pytest.raises(DatabaseMigrationError):
                run_migrations_cli(["upgrade", "head"])

    @pytest.mark.integration
    @pytest.mark.usefixtures("docker_postgres")
    def test_integration_migration_workflow(self):
        """
        Integration test for a full migration workflow.
        Requires the docker_postgres fixture.
        """
        # Create a temporary directory for the test
        with tempfile.TemporaryDirectory() as temp_dir:
            # Set up a minimal migration environment
            migrations_dir = os.path.join(temp_dir, "migrations")
            versions_dir = os.path.join(migrations_dir, "versions")
            os.makedirs(versions_dir)
            
            # Create a simple alembic.ini file
            alembic_ini = os.path.join(temp_dir, "alembic.ini")
            with open(alembic_ini, "w") as f:
                f.write("[alembic]\n")
                f.write(f"script_location = {migrations_dir}\n")
                port = os.environ.get("POSTGRESQL_PORT", "5433")
                f.write(f"sqlalchemy.url = postgresql://postgres:postgres@localhost:{port}/test_db\n")
                f.write("prepend_sys_path = .\n")
            
            # Create a simple env.py file (minimal working version)
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
"""
                )
            
            # Create a sample migration script
            migration_file = os.path.join(versions_dir, "a1b2c3d4e5f6_create_test_table.py")
            with open(migration_file, "w") as f:
                f.write('''
"""
create test table

Revision ID: a1b2c3d4e5f6
Revises: 
Create Date: 2023-01-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = 'a1b2c3d4e5f6'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    op.create_table(
        'test_table',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('name', sa.String(50), nullable=False)
    )

def downgrade():
    op.drop_table('test_table')
'''
                )
            
            # Patch the get_alembic_config function to use our test config
            with patch("app.core.migration.get_alembic_config") as mock_get_config:
                mock_get_config.return_value = Config(alembic_ini)
                
                # Initialize MigrationManager
                manager = MigrationManager()
                
                try:
                    # Test the upgrade method
                    manager.upgrade("head")
                    
                    # Test getting status
                    status = manager.status()
                    assert status["current"] == "a1b2c3d4e5f6"
                    assert status["status"] == "current"
                    
                    # Verify in database that the table was created
                    from sqlalchemy import create_engine, inspect
                    port = os.environ.get("POSTGRESQL_PORT", "5433")
                    engine = create_engine(f"postgresql://postgres:postgres@localhost:{port}/test_db")
                    inspector = inspect(engine)
                    
                    # Check that our table exists
                    assert 'test_table' in inspector.get_table_names()
                    
                    # Test the downgrade method
                    manager.downgrade("base")
                    
                    # Verify table was removed
                    inspector = inspect(engine)
                    assert 'test_table' not in inspector.get_table_names()
                    
                except Exception as e:
                    pytest.fail(f"Integration test failed: {str(e)}")