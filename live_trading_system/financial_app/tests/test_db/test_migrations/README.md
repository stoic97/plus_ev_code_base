# Alembic Database Migration Testing Framework

This directory contains a comprehensive testing framework for the Alembic database migration system in our financial trading platform. The tests ensure database migrations work correctly, validate migration behavior, and maintain database integrity throughout schema changes.

## Table of Contents

- [Overview](#overview)
- [Test Structure](#test-structure)
- [Running Tests](#running-tests)
- [Test Types](#test-types)
- [Fixtures](#fixtures)
- [Writing New Tests](#writing-new-tests)
- [Troubleshooting](#troubleshooting)

## Overview

The migration testing framework tests both the functionality and correctness of our Alembic migrations. It includes:

- **Unit tests**: Test individual helper functions and utility modules
- **Integration tests**: Test actual database operations with containerized databases
- **Migration tests**: Test migration idempotency, upgrade/downgrade paths, and chain verification

## Test Structure

The test directory structure mirrors the application structure:

```
tests/
├── test_db/
│   ├── test_migrations/
│   │   ├── test_helpers/
│   │   │   ├── test_timescale.py     # Tests for TimescaleDB-specific functions
│   │   │   ├── test_db_init.py       # Tests for database initialization helpers
│   │   │   └── test_alembic_utils.py # Tests for Alembic utility functions
│   │   ├── test_scripts/
│   │   │   └── test_deploy_migrations.sh # Tests for shell scripts
│   │   ├── test_env.py               # Tests for Alembic environment
│   │   └── conftest.py               # Test fixtures specific to migrations
│   └── test_commands.py              # Tests for CLI commands
├── test_core/
│   └── test_migration.py             # Tests for core migration module
└── conftest.py                       # Global test fixtures
```

## Running Tests

### Prerequisites

- Python 3.8+
- Docker (for integration tests)
- PostgreSQL client utilities

### Basic Test Execution

Run all tests:

```bash
pytest tests/test_db/test_migrations/
```

Run specific test modules:

```bash
# Run TimescaleDB helper tests
pytest tests/test_db/test_migrations/test_helpers/test_timescale.py

# Run migration script tests
pytest tests/test_db/test_migrations/test_scripts/test_deploy_migrations.sh
```

Run with coverage report:

```bash
pytest --cov=app.db.migrations tests/test_db/test_migrations/
```

### Running Integration Tests

To run all tests including integration tests with real databases:

```bash
pytest tests/test_db/test_migrations/ --runintegration
```

> **Note**: Integration tests require Docker to be running. The tests will automatically create and destroy Docker containers as needed.

## Test Types

### Unit Tests

Unit tests focus on testing individual functions in isolation by mocking dependencies. They should be fast and not require external resources.

### Integration Tests

Integration tests verify that migrations work correctly with real databases. They are marked with `@pytest.mark.integration` and require Docker to run PostgreSQL and TimescaleDB containers.

### Migration Chain Tests

Tests that verify the entire migration chain is valid, with no gaps or conflicts in revision history.

### Migration Idempotency Tests

Tests that ensure running migrations multiple times produces the same result as running them once.

## Fixtures

The framework provides several fixtures to simplify test writing:

### Global Fixtures (in conftest.py)

- `mock_settings`: Provides mock application settings
- `mock_alembic_config`: Creates mock Alembic configuration
- `temp_migration_dir`: Creates a temporary directory with migration structure
- `docker_postgres`: Starts a PostgreSQL container for integration tests
- `docker_timescale`: Starts a TimescaleDB container for integration tests

### Migration-Specific Fixtures (in test_migrations/conftest.py)

- `mock_alembic_context`: Mocks the Alembic context object
- `temp_alembic_structure`: Creates a complete temporary Alembic directory structure
- `alembic_config`: Creates a real Alembic config for testing
- `sample_migrations`: Creates sample migration files for testing
- `mock_op`: Mocks Alembic operations object
- `mock_migration_helpers`: Mocks all migration helper functions

## Writing New Tests

### Testing New Helper Functions

1. Identify the appropriate test file based on the helper function's purpose
2. Create a new test method in the relevant test class
3. Use the provided fixtures to mock dependencies
4. Test both success and failure paths

Example:

```python
def test_new_helper_function(self, mock_op, mock_execute):
    """Test a new helper function."""
    # Call the function with test parameters
    new_helper_function(mock_op, 'table_name', 'column_name')
    
    # Verify the operations that should have occurred
    mock_execute.assert_called_once()
    sql = mock_execute.call_args[0][0]
    
    # Check the SQL contains expected components
    assert "EXPECTED SQL FRAGMENT" in sql
```

### Testing New Migrations

When adding tests for specific migrations:

1. Create appropriate fixtures representing the database state
2. Test both upgrade and downgrade paths
3. Verify the expected schema changes occurred
4. Test edge cases (e.g., running migrations on non-empty tables)

Example:

```python
@pytest.mark.integration
@pytest.mark.usefixtures("docker_postgres")
def test_specific_migration(alembic_config, sample_migrations):
    """Test a specific migration file."""
    from sqlalchemy import create_engine, inspect
    
    # Initialize engine
    engine = create_engine("postgresql://postgres:postgres@localhost:5433/test_db")
    
    # Run migration up to specific revision
    run_migrations(alembic_config, "upgrade", "target_revision_id")
    
    # Verify schema changes
    inspector = inspect(engine)
    assert "expected_table" in inspector.get_table_names()
    
    # Verify column details
    columns = inspector.get_columns("expected_table")
    assert any(col["name"] == "expected_column" for col in columns)
    
    # Test downgrade
    run_migrations(alembic_config, "downgrade", "previous_revision_id")
    
    # Verify schema reverted
    inspector = inspect(engine)
    assert "expected_table" not in inspector.get_table_names()
```

## Troubleshooting

### Common Issues

#### Integration Tests Failing to Connect to Database

- Check Docker is running
- Verify ports 5433 and 5434 are available (used for test PostgreSQL and TimescaleDB)
- Increase wait time for database startup if needed

#### Tests Pass Individually But Fail When Run Together

- Check for fixture isolation issues
- Look for test methods modifying shared state
- Ensure proper cleanup in teardown

#### Shell Script Tests Failing

- Check execution permissions on scripts
- Verify line endings (use LF not CRLF)
- Check the bash version being used