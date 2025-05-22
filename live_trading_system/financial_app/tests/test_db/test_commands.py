"""
Tests for database command-line interface functions.
"""

import pytest
from unittest.mock import MagicMock, patch, call
import sys
import os
from datetime import datetime

# Import the module to test
from app.db.commands import (
    db_upgrade,
    db_downgrade,
    db_revision,
    db_current,
    db_history,
    db_migrate,
    db_init,
    db_stamp,
    db_check,
    main
)


class TestDbCommands:
    """Test suite for database command-line interface functions."""

    @pytest.fixture
    def mock_migration_manager(self):
        """Create a mocked MigrationManager instance."""
        with patch("app.db.commands.MigrationManager") as mock_cls:
            mock_manager = MagicMock()
            mock_cls.return_value = mock_manager
            yield mock_manager

    def test_db_upgrade(self, mock_migration_manager):
        """Test the database upgrade command."""
        # Call the function
        db_upgrade(["--revision", "head"])
        
        # Verify the upgrade method was called correctly
        mock_migration_manager.upgrade.assert_called_once_with("head")

    def test_db_downgrade(self, mock_migration_manager):
        """Test the database downgrade command."""
        # Call the function
        db_downgrade(["--revision", "base"])
        
        # Verify the downgrade method was called correctly
        mock_migration_manager.downgrade.assert_called_once_with("base")

    def test_db_revision(self, mock_migration_manager):
        """Test the database revision command."""
        # Call the function
        db_revision(["--message", "new_migration", "--autogenerate"])
        
        # Verify the generate method was called correctly
        mock_migration_manager.generate.assert_called_once_with(
            "new_migration", 
            autogenerate=True
        )

    def test_db_current(self, mock_migration_manager):
        """Test the database current command."""
        # Set up the mock to return a specific revision
        mock_migration_manager.current.return_value = "abc123"
        
        # Call the function
        with patch("app.db.commands.print") as mock_print:
            db_current([])
            
            # Verify current method was called and the output was printed
            mock_migration_manager.current.assert_called_once()
            mock_print.assert_called_with("Current revision: abc123")

    def test_db_history(self, mock_migration_manager):
        """Test the database history command."""
        # Set up the mock to return a specific history
        mock_migration_manager.history.return_value = [
            {"revision": "abc123", "description": "first migration"},
            {"revision": "def456", "description": "second migration"}
        ]
        
        # Call the function
        with patch("app.db.commands.print") as mock_print:
            db_history([])
            
            # Verify history method was called and the output was printed
            mock_migration_manager.history.assert_called_once()
            assert mock_print.call_count >= 2  # At least 2 lines should be printed

    def test_db_migrate(self, mock_migration_manager):
        """Test the database migrate command (for upgrading to latest)."""
        # Call the function
        db_migrate([])
        
        # Verify the upgrade to head was called
        mock_migration_manager.upgrade.assert_called_once_with("head")

    def test_db_init(self, mock_migration_manager):
        """Test the database init command."""
        # Mock the initialization methods
        mock_migration_manager.init_schemas.return_value = True
        
        # Call the function
        with patch("app.db.commands.print") as mock_print:
            db_init([])
            
            # Verify the initialization methods were called
            mock_migration_manager.init_schemas.assert_called_once()
            mock_print.assert_called_with("Database initialized successfully")

    def test_db_stamp(self, mock_migration_manager):
        """Test the database stamp command."""
        # Call the function
        db_stamp(["--revision", "abc123"])
        
        # Verify the stamp method was called correctly
        mock_migration_manager.stamp.assert_called_once_with("abc123")

    def test_db_check(self, mock_migration_manager):
        """Test the database check command."""
        # Set up the mock to return a specific status
        mock_migration_manager.status.return_value = {
            "current": "abc123",
            "latest": "def456",
            "status": "behind",
            "pending": ["def456"]
        }
        
        # Call the function
        with patch("app.db.commands.print") as mock_print:
            db_check([])
            
            # Verify the status method was called and the output was printed
            mock_migration_manager.status.assert_called_once()
            assert mock_print.call_count >= 3  # At least 3 lines should be printed

    def test_main_parser(self):
        """Test the main function's argument parser."""
        # Set up the mock to prevent actual execution
        with patch("app.db.commands.argparse.ArgumentParser.parse_args") as mock_parse:
            mock_parse.return_value = MagicMock(command="upgrade", args=["--revision", "head"])
            
            # Also patch the command functions to prevent execution
            with patch("app.db.commands.db_upgrade") as mock_upgrade:
                # Call the function
                with patch.object(sys, 'argv', ['db_command.py', 'upgrade', '--revision', 'head']):
                    main()
                
                # Verify the correct command function was called
                mock_upgrade.assert_called_once()

    @pytest.mark.parametrize(
        "command,args,expected_function",
        [
            ("upgrade", ["--revision", "head"], "db_upgrade"),
            ("downgrade", ["--revision", "base"], "db_downgrade"),
            ("revision", ["--message", "new_migration"], "db_revision"),
            ("current", [], "db_current"),
            ("history", [], "db_history"),
            ("migrate", [], "db_migrate"),
            ("init", [], "db_init"),
            ("stamp", ["--revision", "abc123"], "db_stamp"),
            ("check", [], "db_check")
        ]
    )
    def test_main_command_routing(self, command, args, expected_function):
        """Test main function routes to correct command handler."""
        # Create full command line args
        cmd_args = [command] + args
        
        # Patch the expected function
        with patch(f"app.db.commands.{expected_function}") as mock_func:
            # Patch sys.argv
            with patch.object(sys, 'argv', ['db_command.py'] + cmd_args):
                # Call the main function
                main()
                
                # Verify the correct function was called
                mock_func.assert_called_once()

    def test_error_handling(self, mock_migration_manager):
        """Test error handling in command functions."""
        # Make the upgrade method raise an exception
        mock_migration_manager.upgrade.side_effect = Exception("Migration failed")
        
        # Call the function and check for graceful error handling
        with patch("app.db.commands.print") as mock_print:
            with pytest.raises(SystemExit):
                db_upgrade(["--revision", "head"])
            
            # Verify error was printed
            error_calls = [call for call in mock_print.call_args_list if "Error" in str(call)]
            assert len(error_calls) > 0

    @pytest.mark.integration
    @pytest.mark.usefixtures("docker_postgres")
    def test_integration_commands(self):
        """
        Integration test for database commands.
        Requires the docker_postgres fixture.
        """
        # This test requires a proper migration setup which would be complex to create here
        # A minimal version might look like this:
        
        # Set up minimal environment variables for the test
        old_env = os.environ.copy()
        try:
            os.environ["DB__POSTGRES_URI"] = "postgresql://postgres:postgres@localhost:5433/test_db"
            
            # Patch sys.argv for the init command
            with patch.object(sys, 'argv', ['db_command.py', 'init']):
                # Patch print to avoid output during tests
                with patch("app.db.commands.print"):
                    try:
                        # This is a simplified test just to check the command runs without errors
                        # A real test would setup proper migrations and verify they ran
                        main()
                    except Exception as e:
                        if "already exists" not in str(e):  # Ignore if already initialized
                            pytest.fail(f"Integration test failed: {str(e)}")
        
        finally:
            # Restore environment
            os.environ.clear()
            os.environ.update(old_env)