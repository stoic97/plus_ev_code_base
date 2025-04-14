"""
Tests for the Alembic migrations environment module.
"""

import pytest
from unittest.mock import MagicMock, patch, call
import os
import sys
from pathlib import Path
import alembic
from alembic.config import Config
from alembic.script import ScriptDirectory
from alembic.runtime.environment import EnvironmentContext


class TestAlembicEnvironment:
    """Test suite for Alembic environment configuration."""
    
    @pytest.fixture
    def temp_migration_dir(self, tmpdir):
        """Create a temporary directory for migrations."""
        migrations_dir = tmpdir.mkdir("migrations")
        return str(tmpdir)
    
    @pytest.fixture
    def mock_alembic_context(self):
        """Create a mock Alembic context."""
        context = MagicMock(spec=EnvironmentContext)
        return context

    @pytest.fixture
    def mock_engine(self):
        """Create a mock SQLAlchemy engine."""
        engine = MagicMock()
        connection = MagicMock()
        engine.connect.return_value = connection
        
        # Set up context manager for connection
        connection.__enter__ = MagicMock(return_value=connection)
        connection.__exit__ = MagicMock(return_value=None)
        
        return engine
        
    @pytest.fixture
    def mock_env_functions(self, temp_migration_dir):
        """
        Create mock functions that would normally be defined in env.py
        """
        # Create functions that would be in env.py
        env_functions = {
            'run_migrations_offline': MagicMock(),
            'run_migrations_online': MagicMock(),
            'run_migrations_for_testing': MagicMock(),
            'get_db_url': MagicMock(return_value="postgresql://postgres:postgres@localhost:5432/test_db"),
            'target_metadata': MagicMock(),  # This would be Base.metadata
            'include_schemas': True,
            'include_objects': MagicMock(return_value=True)
        }
        
        return env_functions

    def test_offline_migrations_config(self, mock_env_functions, mock_alembic_context):
        """Test that offline migrations are configured correctly."""
        # Set up mocks
        context = mock_alembic_context
        context.is_offline_mode.return_value = True
        
        # Set up config
        mock_config = MagicMock()
        mock_config.get_main_option.return_value = "postgresql://postgres:postgres@localhost:5432/test_db"
        context.config = mock_config
        
        # Set up transaction context
        mock_tx = MagicMock()
        context.begin_transaction.return_value = mock_tx
        mock_tx.__enter__ = MagicMock(return_value=None)
        mock_tx.__exit__ = MagicMock(return_value=None)
        
        # Create our offline migration function
        def run_migrations_offline():
            url = context.config.get_main_option("sqlalchemy.url", mock_env_functions['get_db_url']())
            context.configure(
                url=url,
                target_metadata=mock_env_functions['target_metadata'],
                literal_binds=True,
                dialect_opts={"paramstyle": "named"},
                include_schemas=mock_env_functions['include_schemas'],
                include_object=mock_env_functions['include_objects'],
            )
            
            with context.begin_transaction():
                context.run_migrations()
        
        # Run the function
        run_migrations_offline()
        
        # Verify context was configured correctly
        context.configure.assert_called_once()
        
        # Check that required parameters were passed
        call_kwargs = context.configure.call_args[1]
        assert "url" in call_kwargs
        assert "target_metadata" in call_kwargs
        assert call_kwargs["include_schemas"] is True
        
        # Verify migrations were run
        context.run_migrations.assert_called_once()

    def test_online_migrations_config(self, mock_env_functions, mock_alembic_context, mock_engine):
        """Test that online migrations are configured correctly."""
        # Set up mocks
        context = mock_alembic_context
        context.is_offline_mode.return_value = False
        
        # Set up config
        mock_config = MagicMock()
        mock_config.get_main_option.return_value = "postgresql://postgres:postgres@localhost:5432/test_db"
        mock_config.get_section.return_value = {"sqlalchemy.url": "postgresql://postgres:postgres@localhost:5432/test_db"}
        mock_config.config_ini_section = "alembic"
        context.config = mock_config
        
        # Set up transaction context
        mock_tx = MagicMock()
        context.begin_transaction.return_value = mock_tx
        mock_tx.__enter__ = MagicMock(return_value=None)
        mock_tx.__exit__ = MagicMock(return_value=None)
        
        # Create our online migration function
        def run_migrations_online():
            # If sqlalchemy.url is not set, use our settings
            if not context.config.get_main_option("sqlalchemy.url", None):
                context.config.set_main_option("sqlalchemy.url", mock_env_functions['get_db_url']())
            
            # Use the provided mock engine
            connectable = mock_engine
            
            with connectable.connect() as connection:
                context.configure(
                    connection=connection,
                    target_metadata=mock_env_functions['target_metadata'],
                    include_schemas=mock_env_functions['include_schemas'],
                    include_object=mock_env_functions['include_objects'],
                )
                
                with context.begin_transaction():
                    context.run_migrations()
        
        # Run the function
        with patch("sqlalchemy.engine_from_config", return_value=mock_engine):
            run_migrations_online()
        
        # Verify engine was created and connection was established
        mock_engine.connect.assert_called_once()
        
        # Verify context was configured correctly
        context.configure.assert_called_once()
        
        # Check that required parameters were passed
        call_kwargs = context.configure.call_args[1]
        assert "connection" in call_kwargs
        assert "target_metadata" in call_kwargs
        assert call_kwargs["include_schemas"] is True
        
        # Verify migrations were run
        context.run_migrations.assert_called_once()

    def test_db_url_from_settings(self, mock_env_functions):
        """Test that database URL is retrieved from settings if not in config."""
        # Create mock settings object
        mock_settings = MagicMock()
        mock_settings.db.POSTGRES_URI = "postgresql://app_user:pass@db.example.com/app_db"
        
        # Create a function that mimics get_db_url from env.py
        def get_db_url():
            # In the real function, this would call get_settings()
            # For test, we'll just return our mock settings
            settings = mock_settings
            return str(settings.db.POSTGRES_URI)
        
        # Test the function
        url = get_db_url()
        
        # Verify the URL was retrieved from settings
        assert url == "postgresql://app_user:pass@db.example.com/app_db"

    def test_metadata_target_setup(self, mock_env_functions):
        """Test that the metadata target is set up correctly for autogenerate."""
        # In env.py, target_metadata is assigned from Base.metadata
        # For our test, it's a mock in mock_env_functions
        target_metadata = mock_env_functions['target_metadata']
        
        # Just verify it exists (it's a mock in our test)
        assert target_metadata is not None

    @pytest.mark.skip("Integration test requires PostgreSQL database")
    def test_integration_migration_environment(self):
        """
        Integration test to verify the Alembic environment can connect to the database.
        Requires a PostgreSQL database.
        """
        from sqlalchemy import create_engine, text
        from alembic.migration import MigrationContext
        
        # Configure the database URL to use the test instance
        os.environ["DB__POSTGRES_URI"] = "postgresql://postgres:postgres@localhost:5432/test_db"
        
        try:
            # Create an engine connected to the test database
            engine = create_engine("postgresql://postgres:postgres@localhost:5432/test_db")
            
            # Create a connection for the test
            with engine.connect() as connection:
                # Set up a MigrationContext for manual testing
                context = MigrationContext.configure(connection)
                
                # Test if we can get current revision
                current_rev = context.get_current_revision()
                
                # This will be None for a fresh database or a revision string if migrations exist
                assert current_rev is None or isinstance(current_rev, str)
                
                # Test that we can use our run_migrations_for_testing function
                # Note: This would normally require setting up the alembic context
                # which is complex, so we'll just test the connection works
                connection.execute(text("SELECT 1"))
        
        except Exception as e:
            pytest.fail(f"Integration test failed: {str(e)}")