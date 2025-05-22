"""
Tests for database initialization helper functions used in migrations.
"""

import pytest
from unittest.mock import MagicMock, patch, call
import sqlalchemy as sa
from sqlalchemy.schema import CreateSchema
from alembic.operations import Operations
from alembic.runtime.migration import MigrationContext

# Import the module to test
from app.db.migrations.helpers.db_init import (
    ensure_schema_exists,
    create_extension,
    create_database_role,
    grant_schema_privileges,
    revoke_schema_privileges,
    initialize_timescaledb,
)


class TestDatabaseInit:
    """Test suite for database initialization helper functions."""

    @pytest.fixture
    def mock_op(self):
        """Create a mock Alembic Operations object."""
        # Create a mock connection
        conn = MagicMock()
        
        # Create a properly configured mock dialect
        mock_dialect = MagicMock()
        mock_dialect.name = "postgresql"
        conn.dialect = mock_dialect
        
        # Configure the context
        context = MagicMock()
        context.dialect = mock_dialect
        
        # Create operations object with our mocked context
        op = Operations(context)
        
        return op

    @pytest.fixture
    def mock_execute(self, mock_op):
        """Mock the execute method of the operations object."""
        with patch.object(mock_op, "execute") as mock_ex:
            yield mock_ex

    def test_ensure_schema_exists_basic(self, mock_op, mock_execute):
        """Test basic schema creation."""
        # Test function with default parameters
        ensure_schema_exists(mock_op, 'market_data')
        
        # Verify the operations were called correctly
        mock_execute.assert_called_once()
        # Check that it's a CREATE SCHEMA IF NOT EXISTS statement
        sql = mock_execute.call_args[0][0]
        assert "CREATE SCHEMA IF NOT EXISTS" in str(sql)
        assert "market_data" in str(sql)

    def test_ensure_schema_exists_with_owner(self, mock_op, mock_execute):
        """Test schema creation with owner specified."""
        ensure_schema_exists(mock_op, 'market_data', 'trading_app')
        
        mock_execute.assert_called_once()
        sql = mock_execute.call_args[0][0]
        
        assert "CREATE SCHEMA IF NOT EXISTS" in str(sql)
        assert "market_data" in str(sql)
        assert "AUTHORIZATION trading_app" in str(sql)

    def test_create_extension(self, mock_op, mock_execute):
        """Test database extension creation."""
        create_extension(mock_op, 'timescaledb')
        
        mock_execute.assert_called_once()
        sql = mock_execute.call_args[0][0]
        
        assert "CREATE EXTENSION IF NOT EXISTS" in sql
        assert "timescaledb" in sql

    def test_create_extension_with_schema(self, mock_op, mock_execute):
        """Test database extension creation with schema specified."""
        create_extension(mock_op, 'timescaledb', 'extensions')
        
        mock_execute.assert_called_once()
        sql = mock_execute.call_args[0][0]
        
        assert "CREATE EXTENSION IF NOT EXISTS" in sql
        assert "timescaledb" in sql
        assert "SCHEMA extensions" in sql

    def test_create_database_role(self, mock_op, mock_execute):
        """Test creation of a database role."""
        create_database_role(
            mock_op, 
            'app_user', 
            password='secure_password', 
            login=True, 
            superuser=False, 
            createdb=False,
            createrole=False
        )
        
        mock_execute.assert_called_once()
        sql = mock_execute.call_args[0][0]
        
        assert "CREATE ROLE app_user" in sql
        assert "WITH PASSWORD" in sql
        assert "LOGIN" in sql
        assert "NOSUPERUSER" in sql
        assert "NOCREATEDB" in sql
        assert "NOCREATEROLE" in sql

    def test_grant_schema_privileges(self, mock_op, mock_execute):
        """Test granting privileges on a schema."""
        grant_schema_privileges(
            mock_op,
            'market_data',
            'app_user',
            ['USAGE', 'CREATE']
        )
        
        mock_execute.assert_called_once()
        sql = mock_execute.call_args[0][0]
        
        assert "GRANT USAGE, CREATE ON SCHEMA market_data TO app_user" in sql

    def test_revoke_schema_privileges(self, mock_op, mock_execute):
        """Test revoking privileges from a schema."""
        revoke_schema_privileges(
            mock_op,
            'market_data',
            'app_user',
            ['USAGE', 'CREATE']
        )
        
        mock_execute.assert_called_once()
        sql = mock_execute.call_args[0][0]
        
        assert "REVOKE USAGE, CREATE ON SCHEMA market_data FROM app_user" in sql

    def test_initialize_timescaledb(self, mock_op, mock_execute):
        """Test complete TimescaleDB initialization."""
        initialize_timescaledb(
            mock_op, 
            schema_name='market_data', 
            role_name='app_user',
            password='secure_password'
        )
        
        # Should have at least 3 calls: create extension, create schema, create role
        assert mock_execute.call_count >= 3
        
        # Check the first call creates the TimescaleDB extension
        first_call = mock_execute.call_args_list[0][0][0]
        assert "CREATE EXTENSION IF NOT EXISTS timescaledb" in str(first_call)
        
        # Verify other expected SQL statements were executed
        all_sql = ' '.join(str(call_args[0][0]) for call_args in mock_execute.call_args_list)
        assert "CREATE SCHEMA" in all_sql
        assert "market_data" in all_sql
        assert "CREATE ROLE" in all_sql or "CREATE USER" in all_sql
        assert "app_user" in all_sql

    @pytest.mark.parametrize(
        "function,args,expected_error",
        [
            (ensure_schema_exists, ('',), "Schema name cannot be empty"),
            (create_extension, ('',), "Extension name cannot be empty"),
            (create_database_role, ('',), "Role name cannot be empty"),
            # For the privilege functions, add empty list to avoid TypeError
            (grant_schema_privileges, ('schema', '', []), "Role name cannot be empty"),
            (grant_schema_privileges, ('', 'role', []), "Schema name cannot be empty"),
            (grant_schema_privileges, ('schema', 'role', []), "At least one privilege must be specified"),
            (revoke_schema_privileges, ('schema', '', []), "Role name cannot be empty"),
            (revoke_schema_privileges, ('', 'role', []), "Schema name cannot be empty"),
            (revoke_schema_privileges, ('schema', 'role', []), "At least one privilege must be specified"),
        ]
    )
    def test_input_validation_errors(self, mock_op, function, args, expected_error):
        """Test that appropriate errors are raised for invalid inputs."""
        with pytest.raises(ValueError, match=expected_error):
            full_args = (mock_op,) + args
            function(*full_args)

    # Integration tests - skip these if no Docker or PostgreSQL is available
    @pytest.mark.skip(reason="Integration test requiring Docker PostgreSQL")
    @pytest.mark.integration
    def test_integration_ensure_schema_exists(self):
        """
        Integration test for creating a schema in a real PostgreSQL database.
        Requires the docker_postgres fixture.
        """
        from sqlalchemy import create_engine, inspect
        
        # Connect to the PostgreSQL instance
        engine = create_engine("postgresql://postgres:postgres@localhost:5433/test_db")
        
        # Create a migration context and operations object
        with engine.connect() as connection:
            context = MigrationContext.configure(connection)
            op = Operations(context)
            
            # Ensure the test schema doesn't exist before we start
            connection.execute("DROP SCHEMA IF EXISTS test_schema CASCADE")
            
            # Create the schema
            ensure_schema_exists(op, 'test_schema')
            
            # Verify the schema was created
            inspector = inspect(engine)
            schemas = inspector.get_schema_names()
            
            assert 'test_schema' in schemas, "Schema was not created"
            
            # Clean up
            connection.execute("DROP SCHEMA test_schema CASCADE")

    @pytest.mark.skip(reason="Integration test requiring Docker PostgreSQL")
    @pytest.mark.integration
    def test_integration_create_extension(self):
        """
        Integration test for creating an extension in a real PostgreSQL database.
        Requires the docker_postgres fixture.
        """
        from sqlalchemy import create_engine
        
        # Connect to the PostgreSQL instance
        engine = create_engine("postgresql://postgres:postgres@localhost:5433/test_db")
        
        # Create a migration context and operations object
        with engine.connect() as connection:
            context = MigrationContext.configure(connection)
            op = Operations(context)
            
            # Create a standard PostgreSQL extension
            create_extension(op, 'pg_stat_statements')
            
            # Verify the extension was created
            result = connection.execute(
                "SELECT * FROM pg_extension WHERE extname = 'pg_stat_statements'"
            ).fetchone()