"""
Unit tests for TimescaleDB hypertable migration.

These tests verify that the migration script correctly converts regular
tables to hypertables, sets up compression policies, creates continuous
aggregates, and configures retention policies.
"""

import os
import pytest
import sqlalchemy as sa
from unittest.mock import patch, MagicMock, call
from sqlalchemy import text
from datetime import datetime

# Import the migration script to test
import app.db.migrations.timescale.versions._20250428_setup_market_data_hypertables as migration

# Import helper modules
from app.db.migrations.helpers.timescale_support import (
    create_hypertable,
    add_hypertable_compression,
    create_continuous_aggregate,
    create_retention_policy,
    create_time_bucket_index,
    is_timescaledb_available
)

# Import test utilities
from app.core.database import DatabaseType, get_db_instance, TimescaleDB


class TestTimescaleHypertableMigration:
    """Tests for TimescaleDB hypertable migration script."""

    @pytest.fixture
    def alembic_op(self):
        """Mock the alembic operations object."""
        mock_op = MagicMock()
        mock_conn = MagicMock()
        mock_op.get_bind.return_value = mock_conn
        
        # Set up the execute method to return appropriate results
        def execute_side_effect(query, *args, **kwargs):
            query_str = str(query)
            if "pg_extension WHERE extname = 'timescaledb'" in query_str:
                # Mock response for checking TimescaleDB extension
                mock_result = MagicMock()
                mock_result.fetchone.return_value = None  # Extension not found initially
                return mock_result
            elif "hypertable_schema" in query_str:
                # Mock response for hypertable check
                mock_result = MagicMock()
                mock_result.fetchall.return_value = [('public', 'ohlcv', '1 day')]
                return mock_result
            else:
                # Default mock result
                mock_result = MagicMock()
                mock_result.fetchall.return_value = []
                return mock_result
                
        mock_conn.execute.side_effect = execute_side_effect
        return mock_op

    @pytest.fixture
    def mock_helper_functions(self):
        """Mock all the helper functions used in the migration."""
        with patch('app.db.migrations.timescale.versions._20250428_setup_market_data_hypertables.create_hypertable') as mock_hypertable, \
             patch('app.db.migrations.timescale.versions._20250428_setup_market_data_hypertables.add_hypertable_compression') as mock_compression, \
             patch('app.db.migrations.timescale.versions._20250428_setup_market_data_hypertables.create_continuous_aggregate') as mock_aggregate, \
             patch('app.db.migrations.timescale.versions._20250428_setup_market_data_hypertables.create_retention_policy') as mock_retention, \
             patch('app.db.migrations.timescale.versions._20250428_setup_market_data_hypertables.create_time_bucket_index') as mock_index, \
             patch('app.db.migrations.timescale.versions._20250428_setup_market_data_hypertables.is_timescaledb_available') as mock_is_available, \
             patch('app.db.migrations.timescale.versions._20250428_setup_market_data_hypertables.create_extension') as mock_extension:
            
            # Configure mock behaviors
            mock_is_available.return_value = False  # Make it install the extension
            
            yield {
                'create_hypertable': mock_hypertable,
                'add_hypertable_compression': mock_compression,
                'create_continuous_aggregate': mock_aggregate,
                'create_retention_policy': mock_retention,
                'create_time_bucket_index': mock_index,
                'is_timescaledb_available': mock_is_available,
                'create_extension': mock_extension
            }

    def test_upgrade_extension_check(self, alembic_op, mock_helper_functions):
        """Test that the upgrade function checks for TimescaleDB extension."""
        # Set up the mock
        mock_helper_functions['is_timescaledb_available'].return_value = False
        
        # Run the upgrade
        migration.upgrade.__globals__['op'] = alembic_op
        migration.upgrade()
        
        # Verify extension check and installation
        mock_helper_functions['is_timescaledb_available'].assert_called_once_with(alembic_op)
        mock_helper_functions['create_extension'].assert_called_once_with(alembic_op, "timescaledb")
        
        # Test the case where extension is already available
        mock_helper_functions['is_timescaledb_available'].reset_mock()
        mock_helper_functions['create_extension'].reset_mock()
        mock_helper_functions['is_timescaledb_available'].return_value = True
        
        migration.upgrade()
        
        mock_helper_functions['is_timescaledb_available'].assert_called_once_with(alembic_op)
        mock_helper_functions['create_extension'].assert_not_called()

    def test_convert_ohlcv_to_hypertable(self, alembic_op, mock_helper_functions):
        """Test OHLCV table conversion to hypertable."""
        # Set up the global op
        migration.convert_ohlcv_to_hypertable.__globals__['op'] = alembic_op
        
        # Run the function
        migration.convert_ohlcv_to_hypertable()
        
        # Verify hypertable creation
        mock_helper_functions['create_hypertable'].assert_called_once_with(
            alembic_op,
            table_name="ohlcv",
            time_column_name="timestamp",
            chunk_time_interval="1 day",
            if_not_exists=True,
            migrate_data=True
        )
        
        # Verify compression setup
        mock_helper_functions['add_hypertable_compression'].assert_called_once_with(
            alembic_op,
            table_name="ohlcv",
            compress_after="30 days",
            compress_segmentby=["instrument_id", "interval"],
            compress_orderby="timestamp"
        )
        
        # Verify time bucket index creation
        mock_helper_functions['create_time_bucket_index'].assert_called_once_with(
            alembic_op,
            table_name="ohlcv",
            time_column="timestamp",
            bucket_interval="1 hour",
            include_columns=["instrument_id", "interval"]
        )
        
        # Verify SQL execution for additional index
        assert alembic_op.execute.call_count >= 1
        create_index_call = False
        
        for call_args in alembic_op.execute.call_args_list:
            query = str(call_args[0][0])
            if "CREATE INDEX IF NOT EXISTS ix_ohlcv_instrument_interval_time" in query:
                create_index_call = True
                break
                
        assert create_index_call, "Missing CREATE INDEX call for OHLCV"

    def test_convert_tick_to_hypertable(self, alembic_op, mock_helper_functions):
        """Test Tick table conversion to hypertable."""
        # Set up the global op
        migration.convert_tick_to_hypertable.__globals__['op'] = alembic_op
        
        # Run the function
        migration.convert_tick_to_hypertable()
        
        # Verify hypertable creation with appropriate chunk interval
        mock_helper_functions['create_hypertable'].assert_called_once_with(
            alembic_op,
            table_name="tick",
            time_column_name="timestamp",
            chunk_time_interval="4 hours",
            if_not_exists=True,
            migrate_data=True
        )
        
        # Verify compression setup with appropriate params
        mock_helper_functions['add_hypertable_compression'].assert_called_once_with(
            alembic_op,
            table_name="tick",
            compress_after="7 days",
            compress_segmentby=["instrument_id"],
            compress_orderby="timestamp"
        )
        
        # Verify SQL execution for indexes
        assert alembic_op.execute.call_count >= 2
        
        # Check for the instrument_time index
        instrument_time_index_call = False
        # Check for the trade_id index
        trade_id_index_call = False
        
        for call_args in alembic_op.execute.call_args_list:
            query = str(call_args[0][0])
            if "CREATE INDEX IF NOT EXISTS ix_tick_instrument_time" in query:
                instrument_time_index_call = True
            elif "CREATE INDEX IF NOT EXISTS ix_tick_trade_id_time" in query:
                trade_id_index_call = True
                
        assert instrument_time_index_call, "Missing CREATE INDEX call for tick instrument_time"
        assert trade_id_index_call, "Missing CREATE INDEX call for tick trade_id_time"

    def test_convert_orderbook_to_hypertable(self, alembic_op, mock_helper_functions):
        """Test OrderBookSnapshot table conversion to hypertable."""
        # Set up the global op
        migration.convert_orderbook_to_hypertable.__globals__['op'] = alembic_op
        
        # Run the function
        migration.convert_orderbook_to_hypertable()
        
        # Verify hypertable creation with appropriate chunk interval
        mock_helper_functions['create_hypertable'].assert_called_once_with(
            alembic_op,
            table_name="order_book_snapshot",
            time_column_name="timestamp",
            chunk_time_interval="6 hours",
            if_not_exists=True,
            migrate_data=True
        )
        
        # Verify compression setup
        mock_helper_functions['add_hypertable_compression'].assert_called_once_with(
            alembic_op,
            table_name="order_book_snapshot",
            compress_after="3 days",
            compress_segmentby=["instrument_id"],
            compress_orderby="timestamp"
        )
        
        # Verify SQL execution for indexes
        assert alembic_op.execute.call_count >= 2
        
        # Check for the instrument_time index
        instrument_time_index_call = False
        # Check for the spread index
        spread_index_call = False
        
        for call_args in alembic_op.execute.call_args_list:
            query = str(call_args[0][0])
            if "CREATE INDEX IF NOT EXISTS ix_orderbook_instrument_time" in query:
                instrument_time_index_call = True
            elif "CREATE INDEX IF NOT EXISTS ix_orderbook_spread" in query:
                spread_index_call = True
                
        assert instrument_time_index_call, "Missing CREATE INDEX call for orderbook instrument_time"
        assert spread_index_call, "Missing CREATE INDEX call for orderbook spread"

    def test_create_ohlcv_aggregates(self, alembic_op, mock_helper_functions):
        """Test creation of continuous aggregates for OHLCV data."""
        # Set up the global op
        migration.create_ohlcv_aggregates.__globals__['op'] = alembic_op
        
        # Run the function
        migration.create_ohlcv_aggregates()
        
        # Verify continuous aggregate creation
        assert mock_helper_functions['create_continuous_aggregate'].call_count == 2
        
        # Extract the calls for hourly and daily aggregates
        hourly_call = None
        daily_call = None
        
        for call in mock_helper_functions['create_continuous_aggregate'].call_args_list:
            # Extract view_name from either args or kwargs
            view_name = None
            if len(call.args) > 1:
                view_name = call.args[1]
            elif 'view_name' in call.kwargs:
                view_name = call.kwargs['view_name']
            
            if view_name == "ohlcv_hourly":
                hourly_call = call
            elif view_name == "ohlcv_daily":
                daily_call = call
        
        assert hourly_call is not None, "Missing call to create hourly aggregate"
        assert daily_call is not None, "Missing call to create daily aggregate"
        
        # Verify hourly aggregate parameters
        if len(hourly_call.args) > 0:
            # Access via positional args if available
            assert hourly_call.args[0] == alembic_op  # op
            if len(hourly_call.args) > 1:
                assert hourly_call.args[1] == "ohlcv_hourly"  # view_name
            if len(hourly_call.args) > 2:
                assert hourly_call.args[2] == "ohlcv"  # hypertable_name
            if len(hourly_call.args) > 3:
                assert "time_bucket('1 hour', timestamp)" in hourly_call.args[3]  # query
            if len(hourly_call.args) > 4:
                assert hourly_call.args[4] == "30 minutes"  # refresh_interval
            if len(hourly_call.args) > 5:
                assert hourly_call.args[5] == "1 hour"  # refresh_lag
        else:
            # Access via keyword args
            assert hourly_call.kwargs.get('view_name') == "ohlcv_hourly"
            assert hourly_call.kwargs.get('hypertable_name') == "ohlcv"
            assert "time_bucket('1 hour', timestamp)" in hourly_call.kwargs.get('query', '')
            assert hourly_call.kwargs.get('refresh_interval') == "30 minutes"
            assert hourly_call.kwargs.get('refresh_lag') == "1 hour"
        
        # Verify daily aggregate parameters
        if len(daily_call.args) > 0:
            # Access via positional args if available
            assert daily_call.args[0] == alembic_op  # op
            if len(daily_call.args) > 1:
                assert daily_call.args[1] == "ohlcv_daily"  # view_name
            if len(daily_call.args) > 2:
                assert daily_call.args[2] == "ohlcv"  # hypertable_name
            if len(daily_call.args) > 3:
                assert "time_bucket('1 day', timestamp)" in daily_call.args[3]  # query
            if len(daily_call.args) > 4:
                assert daily_call.args[4] == "1 hour"  # refresh_interval
            if len(daily_call.args) > 5:
                assert daily_call.args[5] == "3 hours"  # refresh_lag
        else:
            # Access via keyword args
            assert daily_call.kwargs.get('view_name') == "ohlcv_daily"
            assert daily_call.kwargs.get('hypertable_name') == "ohlcv"
            assert "time_bucket('1 day', timestamp)" in daily_call.kwargs.get('query', '')
            assert daily_call.kwargs.get('refresh_interval') == "1 hour"
            assert daily_call.kwargs.get('refresh_lag') == "3 hours"
        
        # Verify index creation on continuous aggregates
        assert alembic_op.execute.call_count >= 2
        
        # Check for indices on continuous aggregates
        hourly_index_call = False
        daily_index_call = False
        
        for call_args in alembic_op.execute.call_args_list:
            query = str(call_args[0][0])
            if "CREATE INDEX IF NOT EXISTS ix_ohlcv_hourly_instrument_bucket" in query:
                hourly_index_call = True
            elif "CREATE INDEX IF NOT EXISTS ix_ohlcv_daily_instrument_bucket" in query:
                daily_index_call = True
                
        assert hourly_index_call, "Missing CREATE INDEX call for hourly aggregate"
        assert daily_index_call, "Missing CREATE INDEX call for daily aggregate"

    def test_setup_retention_policies(self, alembic_op, mock_helper_functions):
        """Test setup of data retention policies."""
        # Set up the global op
        migration.setup_retention_policies.__globals__['op'] = alembic_op
        
        # Run the function
        migration.setup_retention_policies()
        
        # Verify retention policy creation
        assert mock_helper_functions['create_retention_policy'].call_count == 2
        
        # Check retention policies for tick and orderbook tables
        tick_retention_call = None
        orderbook_retention_call = None
        
        for call in mock_helper_functions['create_retention_policy'].call_args_list:
            # Extract table_name from either args or kwargs
            table_name = None
            if len(call.args) > 1:
                table_name = call.args[1]
            elif 'table_name' in call.kwargs:
                table_name = call.kwargs['table_name']
                
            if table_name == "tick":
                tick_retention_call = call
            elif table_name == "order_book_snapshot":
                orderbook_retention_call = call
                
        assert tick_retention_call is not None, "Missing retention policy for tick table"
        assert orderbook_retention_call is not None, "Missing retention policy for order_book_snapshot table"
        
        # Verify retention periods
        if len(tick_retention_call.args) > 0:
            # Access via positional args if available
            assert tick_retention_call.args[0] == alembic_op  # op
            if len(tick_retention_call.args) > 1:
                assert tick_retention_call.args[1] == "tick"  # table_name
            if len(tick_retention_call.args) > 2:
                assert tick_retention_call.args[2] == "90 days"  # retention_period
        else:
            # Access via keyword args
            assert tick_retention_call.kwargs.get('table_name') == "tick"
            assert tick_retention_call.kwargs.get('retention_period') == "90 days"
            
        if len(orderbook_retention_call.args) > 0:
            # Access via positional args if available
            assert orderbook_retention_call.args[0] == alembic_op  # op
            if len(orderbook_retention_call.args) > 1:
                assert orderbook_retention_call.args[1] == "order_book_snapshot"  # table_name
            if len(orderbook_retention_call.args) > 2:
                assert orderbook_retention_call.args[2] == "30 days"  # retention_period
        else:
            # Access via keyword args
            assert orderbook_retention_call.kwargs.get('table_name') == "order_book_snapshot"
            assert orderbook_retention_call.kwargs.get('retention_period') == "30 days"
        
        # Verify logging job creation
        log_job_call = False
        
        for call_args in alembic_op.execute.call_args_list:
            query = str(call_args[0][0])
            if "SELECT add_job('log_retention_policy_status'" in query:
                log_job_call = True
                
        assert log_job_call, "Missing call to create retention policy logging job"

    def test_exception_handling(self, alembic_op, mock_helper_functions):
        """Test that exceptions are properly caught and handled."""
        # Set up the global op
        migration.convert_ohlcv_to_hypertable.__globals__['op'] = alembic_op
        
        # Make create_hypertable raise an exception
        mock_helper_functions['create_hypertable'].side_effect = Exception("Test error")
        
        # Verify that the exception is caught and reraised with details
        with pytest.raises(Exception) as excinfo:
            migration.convert_ohlcv_to_hypertable()
            
        assert "Failed to convert OHLCV table to hypertable" in str(excinfo.value)
        
        # Verify rollback was called
        rollback_call = False
        for call_args in alembic_op.execute.call_args_list:
            query = str(call_args[0][0])
            if "ROLLBACK" in query:
                rollback_call = True
                
        assert rollback_call, "Missing ROLLBACK call on exception"


@pytest.mark.integration
class TestTimescaleHypertableIntegration:
    """
    Integration tests for TimescaleDB hypertable migration.
    
    These tests require an actual TimescaleDB instance and will be skipped
    if the environment is not properly configured.
    """
    
    @pytest.fixture
    def timescale_db(self):
        """
        Get a TimescaleDB connection if available,
        or skip the test if not available.
        """
        try:
            # Try to get a TimescaleDB instance
            db = get_db_instance(DatabaseType.TIMESCALEDB)
            
            # Check if TimescaleDB extension is available
            with db.session() as session:
                result = session.execute(text(
                    "SELECT extname FROM pg_extension WHERE extname = 'timescaledb'"
                ))
                if not result.fetchone():
                    pytest.skip("TimescaleDB extension not available")
                    
            return db
        except Exception as e:
            pytest.skip(f"Could not connect to TimescaleDB: {e}")
    
    @pytest.mark.skipif(
        os.environ.get("RUN_INTEGRATION_TESTS") != "1", 
        reason="Integration tests disabled"
    )
    def test_hypertable_creation(self, timescale_db):
        """
        Test actual hypertable creation in a real database.
        
        This test will only run if RUN_INTEGRATION_TESTS=1 is set in environment.
        """
        # Create a test table
        with timescale_db.session() as session:
            # Create a simple test table
            session.execute(text("""
                CREATE TABLE IF NOT EXISTS test_hypertable (
                    id SERIAL PRIMARY KEY,
                    time TIMESTAMPTZ NOT NULL,
                    value DOUBLE PRECISION NOT NULL
                )
            """))
            session.commit()
            
            # Convert to hypertable
            session.execute(text("""
                SELECT create_hypertable('test_hypertable', 'time', if_not_exists => TRUE)
            """))
            session.commit()
            
            # Insert some test data
            session.execute(text("""
                INSERT INTO test_hypertable (time, value)
                SELECT 
                    NOW() - (i || ' hours')::INTERVAL,
                    random() * 100
                FROM generate_series(0, 24) i
            """))
            session.commit()
            
            # Verify it's a hypertable
            result = session.execute(text("""
                SELECT hypertable_name
                FROM timescaledb_information.hypertables
                WHERE hypertable_name = 'test_hypertable'
            """))
            
            assert result.fetchone() is not None, "Table was not converted to hypertable"
            
            # Clean up
            session.execute(text("DROP TABLE test_hypertable CASCADE"))
            session.commit()