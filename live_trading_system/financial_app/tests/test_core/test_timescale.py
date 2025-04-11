"""
Unit tests for timescale.py module.

This test suite validates the TimescaleDB-specific functionality including
hypertable creation, time-bucket queries, and compression policies.
"""

import pytest
import uuid
from datetime import datetime, timedelta
from unittest.mock import MagicMock, call

from sqlalchemy import text, DDL
from sqlalchemy.engine import Connection
from sqlalchemy.sql.elements import TextClause

from app.core.timescale import (
    create_ohlcv_hypertable,
    create_tick_hypertable,
    create_orderbook_hypertable,
    execute_time_bucket_query,
    get_ohlcv_from_ticks,
    configure_continuous_aggregates,
    compress_chunks,
    decompress_chunks,
    get_chunk_statistics,
    get_compression_statistics
)
from app.models.market_data import OHLCV, Tick, OrderBookSnapshot


def normalize_sql(sql: str) -> str:
    """Helper function to remove extra whitespace and newlines for SQL comparisons."""
    return " ".join(sql.split())


@pytest.fixture
def mock_connection():
    """Create a mock SQLAlchemy connection object"""
    mock_conn = MagicMock(spec=Connection)

    # Create a mock result object
    mock_result = MagicMock()
    mock_result.rowcount = 5

    # For queries that iterate over the result (like fetchall or iteration),
    # simulate at least one row with attribute access.
    row_time = datetime.utcnow()
    mock_row = MagicMock()
    mock_row.bucket = row_time
    mock_row.value = 150.0

    # Set the iterator of the mock result to yield the mock row.
    mock_result.__iter__.return_value = [mock_row]

    # For queries that use fetchone (like in compression statistics), return a plain dictionary.
    mock_result.fetchone.return_value = {
        "hypertable_name": "ohlcv",
        "total_uncompressed": "1 GB",
        "total_compressed": "100 MB",
        "compression_ratio": 0.1
    }

    mock_conn.execute.return_value = mock_result
    yield mock_conn


class TestHypertableCreation:
    """Test suite for hypertable creation functions"""

    def test_create_ohlcv_hypertable(self, mock_connection):
        """Test creation of OHLCV hypertable"""
        create_ohlcv_hypertable(OHLCV.__table__, mock_connection)

        # Verify the correct SQL commands were executed
        assert mock_connection.execute.call_count >= 3

        # Get all SQL commands executed
        args_list = [call_args[0][0] for call_args in mock_connection.execute.call_args_list]

        # Check for the presence of critical SQL commands
        assert any('create_hypertable' in str(arg) and 'ohlcv' in str(arg) for arg in args_list)
        assert any('timescaledb.compress' in str(arg) and 'ohlcv' in str(arg) for arg in args_list)
        assert any('add_compression_policy' in str(arg) and 'ohlcv' in str(arg) for arg in args_list)

    def test_create_tick_hypertable(self, mock_connection):
        """Test creation of Tick hypertable"""
        create_tick_hypertable(Tick.__table__, mock_connection)

        # Verify the correct SQL commands were executed
        assert mock_connection.execute.call_count >= 3

        # Get all SQL commands executed
        args_list = [call_args[0][0] for call_args in mock_connection.execute.call_args_list]

        # Check for the presence of critical SQL commands
        assert any('create_hypertable' in str(arg) and 'tick' in str(arg) for arg in args_list)
        assert any('timescaledb.compress' in str(arg) and 'tick' in str(arg) for arg in args_list)
        assert any('add_compression_policy' in str(arg) and 'tick' in str(arg) for arg in args_list)

    def test_create_orderbook_hypertable(self, mock_connection):
        """Test creation of OrderBookSnapshot hypertable"""
        create_orderbook_hypertable(OrderBookSnapshot.__table__, mock_connection)

        # Verify the correct SQL commands were executed
        assert mock_connection.execute.call_count >= 3

        # Get all SQL commands executed
        args_list = [call_args[0][0] for call_args in mock_connection.execute.call_args_list]

        # Check for the presence of critical SQL commands
        assert any('create_hypertable' in str(arg) and 'order_book_snapshot' in str(arg) for arg in args_list)
        assert any('timescaledb.compress' in str(arg) and 'order_book_snapshot' in str(arg) for arg in args_list)
        assert any('add_compression_policy' in str(arg) and 'order_book_snapshot' in str(arg) for arg in args_list)


class TestTimeseriesQueries:
    """Test suite for TimescaleDB query functions"""

    def test_execute_time_bucket_query(self, mock_connection):
        """Test time_bucket query execution"""
        start_time = datetime.utcnow() - timedelta(days=1)
        end_time = datetime.utcnow()

        result = execute_time_bucket_query(
            connection=mock_connection,
            table="ohlcv",
            time_column="timestamp",
            interval="1 hour",
            agg_func="AVG",
            value_column="close",
            start_time=start_time,
            end_time=end_time,
            filters={"interval": "1h"}
        )

        # Verify query was executed
        assert mock_connection.execute.called

        # Check that the first argument is a SQL text object (TextClause)
        args, _ = mock_connection.execute.call_args
        assert isinstance(args[0], TextClause)

        # Normalize the SQL query text for easier substring matching
        query_text = normalize_sql(str(args[0]))

        # Check that the query contains the expected elements
        assert "time_bucket('1 hour', timestamp) AS bucket" in query_text
        assert "AVG(close) AS value" in query_text
        assert "FROM ohlcv" in query_text
        # Allow for small spacing differences in the GROUP BY clause.
        assert "GROUP BY bucket" in query_text or "GROUP BY  bucket" in query_text

        # The parameters are expected as the second positional argument.
        params = args[1]
        assert params["start_time"] == start_time
        assert params["end_time"] == end_time
        assert params["param_0"] == "1h"

        # Check the result structure (should contain one row)
        assert isinstance(result, list)
        assert len(result) > 0
        # Verify the returned row has a non-null bucket and the expected value.
        assert result[0]["bucket"] is not None
        assert result[0]["value"] == 150.0

    def test_get_ohlcv_from_ticks(self, mock_connection):
        """Test generating OHLCV from tick data"""
        start_time = datetime.utcnow() - timedelta(days=1)
        end_time = datetime.utcnow()
        instrument_id = str(uuid.uuid4())

        result = get_ohlcv_from_ticks(
            connection=mock_connection,
            instrument_id=instrument_id,
            interval="1 hour",
            start_time=start_time,
            end_time=end_time
        )

        # Verify query was executed
        assert mock_connection.execute.called

        # Check that the first argument is a SQL text object (TextClause)
        args, _ = mock_connection.execute.call_args
        assert isinstance(args[0], TextClause)

        # Normalize the SQL query text for easier substring matching
        query_text = normalize_sql(str(args[0]))

        # Check that the query contains the expected elements
        assert "time_bucket('1 hour', timestamp) AS time" in query_text
        assert "instrument_id" in query_text
        assert "FIRST(price, timestamp) AS open" in query_text
        assert "MAX(price) AS high" in query_text
        assert "MIN(price) AS low" in query_text
        assert "LAST(price, timestamp) AS close" in query_text
        assert "FROM tick" in query_text
        # Check for GROUP BY clause.
        assert "GROUP BY time, instrument_id" in query_text

        # Check that the parameters contain our time range and instrument.
        params = args[1]
        assert params["start_time"] == start_time
        assert params["end_time"] == end_time
        assert params["instrument_id"] == instrument_id


class TestContinuousAggregates:
    """Test suite for continuous aggregates functions"""

    def test_configure_continuous_aggregates_ohlcv(self, mock_connection):
        """Test configuring continuous aggregates for OHLCV"""
        configure_continuous_aggregates(
            connection=mock_connection,
            table_name="ohlcv",
            interval="1 day"
        )

        # Verify SQL commands were executed
        assert mock_connection.execute.call_count >= 2

        # Get all executed SQL commands
        args_list = [call_args[0][0] for call_args in mock_connection.execute.call_args_list]

        # Check for the presence of critical SQL commands
        assert any('CREATE MATERIALIZED VIEW' in str(arg) for arg in args_list)
        assert any('WITH (timescaledb.continuous)' in str(arg) for arg in args_list)
        assert any('add_continuous_aggregate_policy' in str(arg) for arg in args_list)

        # Verify OHLCV-specific functionality in the view query
        view_query = normalize_sql(str(args_list[0]))
        assert "FIRST(open, timestamp)" in view_query
        assert "MAX(high)" in view_query
        assert "MIN(low)" in view_query
        assert "LAST(close, timestamp)" in view_query

    def test_configure_continuous_aggregates_tick(self, mock_connection):
        """Test configuring continuous aggregates for tick data"""
        configure_continuous_aggregates(
            connection=mock_connection,
            table_name="tick",
            interval="1 hour"
        )

        # Verify SQL commands were executed
        assert mock_connection.execute.call_count >= 2

        # Get all executed SQL commands
        args_list = [call_args[0][0] for call_args in mock_connection.execute.call_args_list]

        # Check for tick-specific aggregate functionality in the view query
        view_query = normalize_sql(str(args_list[0]))
        assert "AVG(price) AS avg_price" in view_query
        assert "SUM(volume) AS volume" in view_query
        assert "COUNT(*) AS trade_count" in view_query

    def test_configure_continuous_aggregates_invalid_table(self, mock_connection):
        """Test configuring continuous aggregates for an unsupported table"""
        with pytest.raises(ValueError, match="Unsupported table"):
            configure_continuous_aggregates(
                connection=mock_connection,
                table_name="unsupported_table",
                interval="1 hour"
            )


class TestCompressionManagement:
    """Test suite for compression management functions"""

    def test_compress_chunks(self, mock_connection):
        """Test manual chunk compression"""
        compress_chunks(
            connection=mock_connection,
            table_name="ohlcv",
            older_than="7 days"
        )

        # Verify SQL was executed
        assert mock_connection.execute.called

        # Check that the query contains the expected elements
        args, _ = mock_connection.execute.call_args
        query_text = normalize_sql(str(args[0]))
        assert "compress_chunk" in query_text
        assert "timescaledb_information.chunks" in query_text
        assert "hypertable_name = 'ohlcv'" in query_text
        assert "chunk_status = 'Uncompressed'" in query_text
        assert "range_end < NOW() - INTERVAL '7 days'" in query_text

    def test_decompress_chunks(self, mock_connection):
        """Test chunk decompression for a time range"""
        start_time = datetime(2023, 1, 1)
        end_time = datetime(2023, 1, 7)

        decompress_chunks(
            connection=mock_connection,
            table_name="tick",
            time_range=(start_time, end_time)
        )

        # Verify SQL was executed
        assert mock_connection.execute.called

        # Check that the query contains the expected elements
        args, _ = mock_connection.execute.call_args
        query_text = normalize_sql(str(args[0]))
        assert "decompress_chunk" in query_text
        assert "timescaledb_information.chunks" in query_text
        assert "hypertable_name = 'tick'" in query_text
        assert "chunk_status = 'Compressed'" in query_text
        assert "range_start" in query_text
        assert "range_end" in query_text
        assert "2023-01-01" in query_text
        assert "2023-01-07" in query_text


class TestMonitoringFunctions:
    """Test suite for TimescaleDB monitoring functions"""

    def test_get_chunk_statistics(self, mock_connection):
        """Test retrieving chunk statistics"""
        result = get_chunk_statistics(
            connection=mock_connection,
            table_name="ohlcv"
        )

        # Verify SQL was executed
        assert mock_connection.execute.called

        # Check that the query contains the expected elements
        args, _ = mock_connection.execute.call_args
        query_text = normalize_sql(str(args[0]))
        assert "timescaledb_information.chunks" in query_text
        assert "hypertable_name = 'ohlcv'" in query_text
        assert "pg_size_pretty" in query_text
        assert "ORDER BY range_start DESC" in query_text

        # Check the result structure is a list of dictionaries
        assert isinstance(result, list)

    def test_get_compression_statistics(self, mock_connection):
        """Test retrieving compression statistics"""
        result = get_compression_statistics(
            connection=mock_connection,
            table_name="ohlcv"
        )

        # Verify SQL was executed
        assert mock_connection.execute.called

        # Check that the query contains the expected elements
        args, _ = mock_connection.execute.call_args
        query_text = normalize_sql(str(args[0]))
        assert "timescaledb_information.chunks" in query_text
        assert "hypertable_name = 'ohlcv'" in query_text
        assert "pg_size_pretty" in query_text
        assert "compression_ratio" in query_text
        assert "GROUP BY hypertable_name" in query_text

        # Check that the result structure is a dictionary
        assert isinstance(result, dict)
        assert "hypertable_name" in result
        assert "total_uncompressed" in result
        assert "total_compressed" in result
        assert "compression_ratio" in result
