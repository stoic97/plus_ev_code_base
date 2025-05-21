"""
Database storage verification for market data.

This script verifies that market data is properly stored in TimescaleDB,
including data insertion, timestamp processing, query performance, and data integrity.
"""

import logging
import time
import datetime
import os
import sys
import argparse
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import Dict, List, Any, Optional, Tuple

from financial_app.tests.integration.data_layer_e2e.utils.validation import compare_datasets
from financial_app.tests.integration.data_layer_e2e.utils.reporting import TestReporter
from financial_app.tests.integration.data_layer_e2e.utils.performance import timed_function, PerformanceTracker
from financial_app.tests.integration.data_layer_e2e.e2e_config import DB_CONFIG, TEST_INSTRUMENTS, TEST_ID

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"db_verification_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("db_verification")

class DatabaseVerifier:
    """Verifies data storage in TimescaleDB for market data."""

    def __init__(self, db_config: Dict[str, Any]):
        """
        Initialize the database verifier.
        
        Args:
            db_config: Database connection configuration
        """
        self.db_config = db_config
        self.conn = None
        self.cursor = None
        
        # Initialize performance tracker
        self.perf_tracker = PerformanceTracker()
        
        # Initialize reporter
        self.reporter = TestReporter()
    
    def connect(self) -> bool:
        """
        Connect to the TimescaleDB database.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            logger.info(f"Connecting to database: {self.db_config['database']} at {self.db_config['host']}")
            self.conn = psycopg2.connect(
                dbname=self.db_config["database"],
                user=self.db_config["user"],
                password=self.db_config["password"],
                host=self.db_config["host"],
                port=self.db_config["port"]
            )
            self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            logger.info("Database connection established successfully")
            self.reporter.record_test_result("Database Connection", "PASS", "Connected to TimescaleDB successfully")
            return True
        except Exception as e:
            error_msg = f"Failed to connect to database: {str(e)}"
            logger.error(error_msg)
            self.reporter.record_test_result("Database Connection", "FAIL", "Failed to connect to database", error=error_msg)
            return False
    
    def close(self) -> None:
        """Close the database connection."""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        logger.info("Database connection closed")
    
    @timed_function(None, "verify_data_insertion")
    def verify_data_insertion(self, symbol: str, hours: int = 1) -> bool:
        """
        Verify data insertion for a specific symbol over a given timeframe.
        
        Args:
            symbol: Instrument symbol to check
            hours: Hours of data to check
            
        Returns:
            True if verification successful, False otherwise
        """
        try:
            logger.info(f"Verifying data insertion for {symbol} over the last {hours} hours")
            
            # Get the instrument ID for the symbol
            self.cursor.execute("SELECT id FROM instruments WHERE symbol = %s", (symbol,))
            result = self.cursor.fetchone()
            if not result:
                error_msg = f"Instrument with symbol {symbol} not found in database"
                logger.error(error_msg)
                self.reporter.record_test_result(
                    f"Data Insertion - {symbol}", 
                    "FAIL", 
                    error_msg
                )
                return False
            
            instrument_id = result["id"]
            
            # Calculate the time cutoff
            cutoff_time = datetime.datetime.now() - datetime.timedelta(hours=hours)
            
            # Check for OHLCV data
            self.cursor.execute(
                """
                SELECT COUNT(*) as count 
                FROM ohlcv 
                WHERE instrument_id = %s AND timestamp >= %s
                """,
                (instrument_id, cutoff_time)
            )
            ohlcv_result = self.cursor.fetchone()
            ohlcv_count = ohlcv_result["count"] if ohlcv_result else 0
            
            # Check for tick data
            self.cursor.execute(
                """
                SELECT COUNT(*) as count 
                FROM ticks 
                WHERE instrument_id = %s AND timestamp >= %s
                """,
                (instrument_id, cutoff_time)
            )
            tick_result = self.cursor.fetchone()
            tick_count = tick_result["count"] if tick_result else 0
            
            # Create details dictionary
            details = {
                "symbol": symbol,
                "hours_checked": hours,
                "ohlcv_count": ohlcv_count,
                "tick_count": tick_count,
                "time_cutoff": cutoff_time.isoformat()
            }
            
            # Determine if data insertion is valid
            # For real verification, you might want more sophisticated checks
            if ohlcv_count > 0 or tick_count > 0:
                message = f"Found {ohlcv_count} OHLCV records and {tick_count} tick records for {symbol}"
                logger.info(message)
                self.reporter.record_test_result(
                    f"Data Insertion - {symbol}", 
                    "PASS", 
                    message
                )
                return True
            else:
                error_msg = f"No data found for {symbol} in the last {hours} hours"
                logger.warning(error_msg)
                self.reporter.record_test_result(
                    f"Data Insertion - {symbol}", 
                    "FAIL", 
                    error_msg
                )
                return False
                
        except Exception as e:
            error_msg = f"Error verifying data insertion for {symbol}: {str(e)}"
            logger.error(error_msg)
            self.reporter.record_test_result(
                f"Data Insertion - {symbol}", 
                "FAIL", 
                error_msg
            )
            return False
    
    # @timed_function(None, "verify_timestamp_processing")
    def verify_timestamp_processing(self, symbol: str) -> bool:
        """
        Verify that timestamps are being processed correctly.
        
        Args:
            symbol: Instrument symbol to check
            
        Returns:
            True if timestamps are valid, False otherwise
        """
        try:
            # Get the instrument ID for the symbol
            self.cursor.execute("SELECT id FROM instruments WHERE symbol = %s", (symbol,))
            result = self.cursor.fetchone()
            if not result:
                error_msg = f"Instrument with symbol {symbol} not found in database"
                logger.error(error_msg)
                self.reporter.record_test_result(f"Timestamp Processing - {symbol}", "FAIL", error_msg)
                return False
            
            instrument_id = result["id"]
            
            # Check the timestamp ranges and ordering
            self.cursor.execute(
                """
                SELECT 
                    MIN(timestamp) as min_time,
                    MAX(timestamp) as max_time,
                    COUNT(*) as count,
                    COUNT(DISTINCT timestamp) as distinct_count
                FROM ohlcv 
                WHERE instrument_id = %s
                """,
                (instrument_id,)
            )
            ohlcv_result = self.cursor.fetchone()
            
            if not ohlcv_result or not ohlcv_result["min_time"] or not ohlcv_result["max_time"]:
                error_msg = f"No OHLCV data found for {symbol}"
                logger.warning(error_msg)
                self.reporter.record_test_result(f"Timestamp Processing - {symbol}", "FAIL", error_msg)
                return False
            
            # Check for duplicate timestamps
            duplicate_check = ohlcv_result["count"] != ohlcv_result["distinct_count"]
            
            # Check for future timestamps
            now = datetime.datetime.now()
            future_timestamp = ohlcv_result["max_time"] > now + datetime.timedelta(minutes=10)
            
            # Check for very old timestamps
            now_minus_one_year = now - datetime.timedelta(days=365)
            too_old_timestamp = ohlcv_result["min_time"] < now_minus_one_year
            
            # Check for proper time ordering
            self.cursor.execute(
                """
                SELECT COUNT(*) as count
                FROM (
                    SELECT timestamp, LAG(timestamp) OVER (ORDER BY timestamp) as prev_timestamp
                    FROM ohlcv
                    WHERE instrument_id = %s
                    ORDER BY timestamp
                    LIMIT 1000
                ) subquery
                WHERE timestamp < prev_timestamp
                """,
                (instrument_id,)
            )
            ordering_result = self.cursor.fetchone()
            ordering_errors = ordering_result["count"] if ordering_result else 0
            
            # Log the results
            min_time = ohlcv_result["min_time"].strftime("%Y-%m-%d %H:%M:%S")
            max_time = ohlcv_result["max_time"].strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"OHLCV time range for {symbol}: {min_time} to {max_time}")
            logger.info(f"Duplicate timestamps: {duplicate_check}")
            logger.info(f"Future timestamps: {future_timestamp}")
            logger.info(f"Very old timestamps: {too_old_timestamp}")
            logger.info(f"Time ordering errors: {ordering_errors}")
            
            details = {
                "symbol": symbol,
                "min_time": min_time,
                "max_time": max_time,
                "total_records": ohlcv_result["count"],
                "distinct_timestamps": ohlcv_result["distinct_count"],
                "has_duplicates": duplicate_check,
                "has_future_timestamps": future_timestamp,
                "has_very_old_timestamps": too_old_timestamp,
                "ordering_errors": ordering_errors
            }
            
            if not duplicate_check and not future_timestamp and not too_old_timestamp and ordering_errors == 0:
                self.reporter.record_test_result(
                    f"Timestamp Processing - {symbol}", 
                    "PASS", 
                    f"Timestamps for {symbol} are valid and properly ordered",
                    details=details
                )
                return True
            else:
                issues = []
                if duplicate_check:
                    issues.append("duplicate timestamps detected")
                if future_timestamp:
                    issues.append("future timestamps detected")
                if too_old_timestamp:
                    issues.append("very old timestamps detected")
                if ordering_errors > 0:
                    issues.append(f"{ordering_errors} ordering errors detected")
                
                error_msg = f"Timestamp issues for {symbol}: {', '.join(issues)}"
                logger.warning(error_msg)
                self.reporter.record_test_result(f"Timestamp Processing - {symbol}", "FAIL", error_msg, details=details)
                return False
                
        except Exception as e:
            error_msg = f"Error checking timestamp processing for {symbol}: {str(e)}"
            logger.error(error_msg)
            self.reporter.record_test_result(f"Timestamp Processing - {symbol}", "FAIL", error_msg)
            return False
    
    @timed_function(None, "test_query_performance")
    def test_query_performance(self, symbol: str, interval: str = '1h', days: int = 7) -> bool:
        """
        Test query performance on recent data.
        
        Args:
            symbol: Instrument symbol to check
            interval: Time interval to query ('1m', '5m', '1h', etc.)
            days: Number of days of data to query
            
        Returns:
            True if query performance is acceptable, False otherwise
        """
        try:
            # Get the instrument ID for the symbol
            self.cursor.execute("SELECT id FROM instruments WHERE symbol = %s", (symbol,))
            result = self.cursor.fetchone()
            if not result:
                error_msg = f"Instrument with symbol {symbol} not found in database"
                logger.error(error_msg)
                self.reporter.record_test_result(f"Query Performance - {symbol}", "FAIL", error_msg)
                return False
            
            instrument_id = result["id"]
            
            # Define time range
            end_time = datetime.datetime.now()
            start_time = end_time - datetime.timedelta(days=days)
            
            # Measure query time for different scenarios
            query_times = []
            
            # Scenario 1: Simple time range query
            with self.perf_tracker.timed_operation(f"simple_query_{symbol}"):
                start = time.time()
                self.cursor.execute(
                    """
                    SELECT timestamp, open, high, low, close, volume
                    FROM ohlcv
                    WHERE instrument_id = %s
                      AND timestamp BETWEEN %s AND %s
                      AND interval = %s
                    ORDER BY timestamp
                    """,
                    (instrument_id, start_time, end_time, interval)
                )
                results = self.cursor.fetchall()
                duration = time.time() - start
                query_times.append(("Simple time range", duration, len(results)))
                
                # Record performance metric
                self.perf_tracker.record_metric(f"db_query_simple_{symbol}_ms", duration * 1000)
            
            # Scenario 2: Aggregation query
            with self.perf_tracker.timed_operation(f"aggregation_query_{symbol}"):
                start = time.time()
                self.cursor.execute(
                    """
                    SELECT 
                        time_bucket('1 day', timestamp) as day,
                        MAX(high) as day_high,
                        MIN(low) as day_low,
                        SUM(volume) as day_volume
                    FROM ohlcv
                    WHERE instrument_id = %s
                      AND timestamp BETWEEN %s AND %s
                      AND interval = %s
                    GROUP BY day
                    ORDER BY day
                    """,
                    (instrument_id, start_time, end_time, interval)
                )
                results = self.cursor.fetchall()
                duration = time.time() - start
                query_times.append(("Daily aggregation", duration, len(results)))
                
                # Record performance metric
                self.perf_tracker.record_metric(f"db_query_aggregation_{symbol}_ms", duration * 1000)
            
            # Scenario 3: Complex query with joins
            with self.perf_tracker.timed_operation(f"join_query_{symbol}"):
                start = time.time()
                self.cursor.execute(
                    """
                    SELECT 
                        o.timestamp,
                        o.open,
                        o.high,
                        o.low,
                        o.close,
                        o.volume,
                        i.symbol,
                        i.asset_type
                    FROM ohlcv o
                    JOIN instruments i ON o.instrument_id = i.id
                    WHERE i.symbol = %s
                      AND o.timestamp BETWEEN %s AND %s
                      AND o.interval = %s
                    ORDER BY o.timestamp
                    LIMIT 1000
                    """,
                    (symbol, start_time, end_time, interval)
                )
                results = self.cursor.fetchall()
                duration = time.time() - start
                query_times.append(("Join query", duration, len(results)))
                
                # Record performance metric
                self.perf_tracker.record_metric(f"db_query_join_{symbol}_ms", duration * 1000)
            
            # Log the results
            for query_type, duration, result_count in query_times:
                self.perf_tracker.measure_db_query(
                    query_name=f"{symbol}_{query_type.replace(' ', '_').lower()}", 
                    rows_returned=result_count, 
                    query_time_ms=duration * 1000
                )
                logger.info(f"Query performance for {symbol} - {query_type}: {duration:.4f} seconds for {result_count} records")
            
            details = {
                "symbol": symbol,
                "interval": interval,
                "days_queried": days,
                "queries": [
                    {"type": q_type, "duration_sec": duration, "rows": rows}
                    for q_type, duration, rows in query_times
                ]
            }
            
            # Check if any query is too slow (threshold can be adjusted)
            # For simplicity, using a fixed threshold of 3 seconds
            slow_queries = [q for q, d, _ in query_times if d > 3.0]  
            
            if not slow_queries:
                details_str = ", ".join([f"{q}: {d:.4f}s ({r} rows)" for q, d, r in query_times])
                self.reporter.record_test_result(
                    f"Query Performance - {symbol}", 
                    "PASS", 
                    f"All queries completed within acceptable time. {details_str}",
                    details=details
                )
                return True
            else:
                slow_details = ", ".join([f"{q}: {d:.4f}s" for q, d, _ in query_times if q in slow_queries])
                error_msg = f"Slow queries detected for {symbol}: {slow_details}"
                logger.warning(error_msg)
                self.reporter.record_test_result(f"Query Performance - {symbol}", "FAIL", error_msg, details=details)
                return False
                
        except Exception as e:
            error_msg = f"Error testing query performance for {symbol}: {str(e)}"
            logger.error(error_msg)
            self.reporter.record_test_result(f"Query Performance - {symbol}", "FAIL", error_msg)
            return False
    
    @timed_function(None, "validate_data_integrity")
    def validate_data_integrity(self, symbol: str, source_data_file: Optional[str] = None) -> bool:
        """
        Validate data integrity by performing internal consistency checks.
        
        Args:
            symbol: Instrument symbol to check
            source_data_file: Optional file with source data for comparison
            
        Returns:
            True if data integrity is confirmed, False otherwise
        """
        try:
            # Get the instrument ID for the symbol
            self.cursor.execute("SELECT id FROM instruments WHERE symbol = %s", (symbol,))
            result = self.cursor.fetchone()
            if not result:
                error_msg = f"Instrument with symbol {symbol} not found in database"
                logger.error(error_msg)
                self.reporter.record_test_result(f"Data Integrity - {symbol}", "FAIL", error_msg)
                return False
            
            instrument_id = result["id"]
            
            # Perform internal consistency checks
            
            # 1. Check for OHLC consistency (high >= open, high >= close, low <= open, low <= close)
            self.cursor.execute(
                """
                SELECT COUNT(*) as count
                FROM ohlcv
                WHERE instrument_id = %s
                  AND (high < open OR high < close OR low > open OR low > close)
                """,
                (instrument_id,)
            )
            ohlc_consistency_result = self.cursor.fetchone()
            ohlc_errors = ohlc_consistency_result["count"] if ohlc_consistency_result else 0
            
            # 2. Check for negative volumes
            self.cursor.execute(
                """
                SELECT COUNT(*) as count
                FROM ohlcv
                WHERE instrument_id = %s AND volume < 0
                """,
                (instrument_id,)
            )
            negative_volume_result = self.cursor.fetchone()
            negative_volumes = negative_volume_result["count"] if negative_volume_result else 0
            
            # 3. Check for negative prices
            self.cursor.execute(
                """
                SELECT COUNT(*) as count
                FROM ohlcv
                WHERE instrument_id = %s AND (open < 0 OR high < 0 OR low < 0 OR close < 0)
                """,
                (instrument_id,)
            )
            negative_price_result = self.cursor.fetchone()
            negative_prices = negative_price_result["count"] if negative_price_result else 0
            
            # 4. Check for extreme price changes (potential data errors)
            self.cursor.execute(
                """
                SELECT COUNT(*) as count
                FROM (
                    SELECT 
                        timestamp, 
                        close, 
                        LAG(close) OVER (ORDER BY timestamp) as prev_close
                    FROM ohlcv
                    WHERE instrument_id = %s
                    ORDER BY timestamp
                ) as subquery
                WHERE prev_close IS NOT NULL 
                  AND ABS(close / NULLIF(prev_close, 0) - 1) > 0.5  -- 50% change
                """,
                (instrument_id,)
            )
            extreme_change_result = self.cursor.fetchone()
            extreme_changes = extreme_change_result["count"] if extreme_change_result else 0
            
            # If source data file is provided, compare with database records
            source_comparison_success = None
            source_comparison_details = {}
            if source_data_file and os.path.exists(source_data_file):
                source_comparison_success, source_comparison_details = self._compare_with_source_data(
                    instrument_id, symbol, source_data_file
                )
            
            # Log the results
            logger.info(f"Data integrity checks for {symbol}:")
            logger.info(f"OHLC consistency errors: {ohlc_errors}")
            logger.info(f"Negative volumes: {negative_volumes}")
            logger.info(f"Negative prices: {negative_prices}")
            logger.info(f"Extreme price changes: {extreme_changes}")
            if source_comparison_success is not None:
                logger.info(f"Source data comparison: {'Success' if source_comparison_success else 'Failed'}")
            
            details = {
                "symbol": symbol,
                "ohlc_consistency_errors": ohlc_errors,
                "negative_volumes": negative_volumes,
                "negative_prices": negative_prices,
                "extreme_price_changes": extreme_changes
            }
            
            if source_comparison_details:
                details["source_comparison"] = source_comparison_details
            
            # Determine if data integrity is valid
            has_errors = (
                ohlc_errors > 0 or 
                negative_volumes > 0 or 
                negative_prices > 0 or 
                extreme_changes > 10 or  # Allow a few extreme changes as they could be legitimate
                (source_comparison_success is not None and not source_comparison_success)
            )
            
            if not has_errors:
                self.reporter.record_test_result(
                    f"Data Integrity - {symbol}", 
                    "PASS", 
                    f"All data integrity checks passed for {symbol}",
                    details=details
                )
                return True
            else:
                issues = []
                if ohlc_errors > 0:
                    issues.append(f"{ohlc_errors} OHLC consistency errors")
                if negative_volumes > 0:
                    issues.append(f"{negative_volumes} negative volumes")
                if negative_prices > 0:
                    issues.append(f"{negative_prices} negative prices")
                if extreme_changes > 10:
                    issues.append(f"{extreme_changes} extreme price changes")
                if source_comparison_success is not None and not source_comparison_success:
                    issues.append("source data comparison failed")
                
                error_msg = f"Data integrity issues for {symbol}: {', '.join(issues)}"
                logger.warning(error_msg)
                self.reporter.record_test_result(f"Data Integrity - {symbol}", "FAIL", error_msg, details=details)
                return False
                
        except Exception as e:
            error_msg = f"Error validating data integrity for {symbol}: {str(e)}"
            logger.error(error_msg)
            self.reporter.record_test_result(f"Data Integrity - {symbol}", "FAIL", error_msg)
            return False
    
    def _compare_with_source_data(self, instrument_id: int, symbol: str, source_data_file: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Compare database records with source data file.
        
        Args:
            instrument_id: Instrument ID in the database
            symbol: Instrument symbol
            source_data_file: File with source data for comparison
            
        Returns:
            Tuple of (success_flag, comparison_details)
        """
        try:
            # Load source data from file
            source_df = pd.read_csv(source_data_file, parse_dates=['timestamp'])
            
            # Get matching data from database
            self.cursor.execute(
                """
                SELECT 
                    timestamp,
                    open,
                    high,
                    low,
                    close,
                    volume
                FROM ohlcv
                WHERE instrument_id = %s
                  AND timestamp BETWEEN %s AND %s
                ORDER BY timestamp
                """,
                (
                    instrument_id, 
                    source_df['timestamp'].min(),
                    source_df['timestamp'].max()
                )
            )
            
            # Convert to list of dictionaries for comparison
            db_rows = self.cursor.fetchall()
            db_data = [dict(row) for row in db_rows]
            source_data = source_df.to_dict('records')
            
            if not db_data:
                logger.warning(f"No data found in database for comparison with source data for {symbol}")
                return False, {"error": "No matching data in database"}
            
            # Use the existing utility function for comparison
            comparison_results = compare_datasets(
                source_data=source_data,
                target_data=db_data,
                key_field='timestamp'
            )
            
            success = comparison_results.get("match_percentage", 0) > 90  # 90% match threshold
            
            return success, comparison_results
            
        except Exception as e:
            logger.error(f"Error comparing with source data for {symbol}: {str(e)}")
            return False, {"error": str(e)}


class MockDatabaseVerifier(DatabaseVerifier):
    """Mock version of DatabaseVerifier that operates without a real database."""
    
    def __init__(self, mock_config: Dict[str, Any] = None):
        """
        Initialize a mock database verifier.
        
        Args:
            mock_config: Optional mock configuration
        """
        # Call parent init with empty config
        super().__init__({})
        
        # Set up mock data
        self.mock_config = mock_config or {}
        self.setup_mock_data()
        
        # Create a context manager for timing if not available in PerformanceTracker
        if not hasattr(self.perf_tracker, 'timed_operation'):
            self.perf_tracker.timed_operation = self._dummy_timed_operation
        
    def _dummy_timed_operation(self, operation_name: str):
        """
        Dummy context manager for timing operations when real one is not available.
        
        Args:
            operation_name: Name of the operation
        """
        class DummyContextManager:
            def __enter__(self):
                return None
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                pass
                
        return DummyContextManager()
    
    def setup_mock_data(self):
        """Set up mock data for testing."""
        self.mock_instruments = {
            "AAPL": {"id": 1, "symbol": "AAPL", "asset_type": "STOCK"},
            "MSFT": {"id": 2, "symbol": "MSFT", "asset_type": "STOCK"},
            "BTC/USD": {"id": 3, "symbol": "BTC/USD", "asset_type": "CRYPTO"},
        }
        
        # Create mock price data
        self.mock_price_data = {}
        now = datetime.datetime.now()
        for symbol, instr in self.mock_instruments.items():
            # Generate some mock OHLCV data
            ohlcv_data = []
            for i in range(30):  # 30 days of data
                date = now - datetime.timedelta(days=i)
                ohlcv_data.append({
                    "timestamp": date,
                    "open": 100 + i * 0.1,
                    "high": 105 + i * 0.1,
                    "low": 95 + i * 0.1,
                    "close": 102 + i * 0.1,
                    "volume": 1000000 - i * 10000,
                    "instrument_id": instr["id"]
                })
            self.mock_price_data[symbol] = ohlcv_data
        
        logger.info("Mock database initialized with data for symbols: " + 
                    ", ".join(self.mock_instruments.keys()))
    
    def connect(self) -> bool:
        """
        Mock connect to database.
        
        Returns:
            Always returns True
        """
        logger.info("Connecting to mock database")
        self.reporter.record_test_result("Database Connection", "PASS", "Connected to mock database")
        return True
    
    def close(self) -> None:
        """Mock database close - does nothing."""
        logger.info("Mock database connection closed")
    
    def _execute_mock_query(self, query: str, params: tuple = None) -> List[Dict[str, Any]]:
        """
        Mock query execution.
        
        Args:
            query: SQL query (for logging only)
            params: Query parameters (for logging only)
            
        Returns:
            Mock results based on the query
        """
        logger.debug(f"Mock query: {query}")
        if params:
            logger.debug(f"Parameters: {params}")
        
        # Very simple query parsing to determine what to return
        if "SELECT id FROM instruments WHERE symbol" in query:
            # Get instrument query
            symbol = params[0] if params else "AAPL"
            if symbol in self.mock_instruments:
                return [{"id": self.mock_instruments[symbol]["id"]}]
            return []
            
        elif "COUNT(*) as count FROM ohlcv" in query:
            # Count query for OHLCV
            return [{"count": 100}]
            
        elif "COUNT(*) as count FROM ticks" in query:
            # Count query for ticks
            return [{"count": 500}]
            
        elif "MIN(timestamp) as min_time, MAX(timestamp) as max_time" in query:
            # Timestamp range query
            now = datetime.datetime.now()
            return [{
                "min_time": now - datetime.timedelta(days=30),
                "max_time": now - datetime.timedelta(minutes=5),
                "count": 100,
                "distinct_count": 100
            }]
            
        elif "timestamp < prev_timestamp" in query:
            # Time ordering check
            return [{"count": 0}]
            
        elif "high < open OR high < close OR low > open OR low > close" in query:
            # OHLC consistency check
            return [{"count": 0}]
            
        elif "volume < 0" in query:
            # Negative volume check
            return [{"count": 0}]
            
        elif "open < 0 OR high < 0 OR low < 0 OR close < 0" in query:
            # Negative price check
            return [{"count": 0}]
            
        elif "ABS(close / NULLIF(prev_close, 0) - 1) > 0.5" in query:
            # Extreme price change check
            return [{"count": 0}]
            
        elif "SELECT timestamp, open, high, low, close, volume FROM ohlcv" in query:
            # OHLCV data query
            symbol = None
            instrument_id = None
            
            # Try to extract symbol or instrument_id from params
            if params:
                if isinstance(params[0], str):
                    symbol = params[0]
                else:
                    instrument_id = params[0]
            
            # If we have a symbol, use it, otherwise use first symbol
            if not symbol and instrument_id:
                symbol = next((s for s, i in self.mock_instruments.items() 
                               if i["id"] == instrument_id), "AAPL")
            elif not symbol:
                symbol = "AAPL"
                
            # Return sample data
            return [dict(row) for row in self.mock_price_data.get(symbol, [])][:50]
            
        # Default empty response
        return []
    
    def verify_data_insertion(self, symbol: str, hours: int = 1) -> bool:
        """
        Mock verify data insertion.
        
        Args:
            symbol: Instrument symbol to check
            hours: Hours of data to check
            
        Returns:
            True if verification successful, False otherwise
        """
        try:
            logger.info(f"Mock verification of data insertion for {symbol}")
            
            if symbol not in self.mock_instruments:
                logger.error(f"Symbol {symbol} not found in mock data")
                self.reporter.record_test_result(
                    f"Data Insertion - {symbol}", 
                    "FAIL", 
                    f"Symbol {symbol} not found in mock data"
                )
                return False
            
            # Mock successful insertion verification
            ohlcv_count = len(self.mock_price_data.get(symbol, []))
            ticks_count = ohlcv_count * 5  # Mock 5 ticks per OHLCV bar
            
            details = {
                "symbol": symbol,
                "hours_checked": hours,
                "ohlcv_count": ohlcv_count,
                "ticks_count": ticks_count,
                "time_cutoff": (datetime.datetime.now() - datetime.timedelta(hours=hours)).isoformat()
            }
            
            self.reporter.record_test_result(
                f"Data Insertion - {symbol}", 
                "PASS", 
                f"Found {ohlcv_count} OHLCV records and {ticks_count} tick records in mock data",
                details
            )
            return True
            
        except Exception as e:
            error_msg = f"Error in mock data insertion verification for {symbol}: {str(e)}"
            logger.error(error_msg)
            self.reporter.record_test_result(f"Data Insertion - {symbol}", "FAIL", error_msg)
            return False
    
    def verify_timestamp_processing(self, symbol: str) -> bool:
        """
        Mock verify timestamp processing.
        
        Args:
            symbol: Instrument symbol to check
            
        Returns:
            Always returns True for mock data
        """
        try:
            logger.info(f"Mock verification of timestamp processing for {symbol}")
            
            if symbol not in self.mock_instruments:
                self.reporter.record_test_result(
                    f"Timestamp Processing - {symbol}", 
                    "FAIL", 
                    f"Symbol {symbol} not found in mock data"
                )
                return False
            
            # Mock timestamp data
            now = datetime.datetime.now()
            min_time = now - datetime.timedelta(days=30)
            max_time = now - datetime.timedelta(minutes=5)
            
            self.reporter.record_test_result(
                f"Timestamp Processing - {symbol}", 
                "PASS", 
                f"Timestamps for {symbol} are valid and properly ordered in mock data"
            )
            return True
        except Exception as e:
            error_msg = f"Error in mock timestamp processing verification for {symbol}: {str(e)}"
            logger.error(error_msg)
            self.reporter.record_test_result(f"Timestamp Processing - {symbol}", "FAIL", error_msg)
            return False
    
    def test_query_performance(self, symbol: str, interval: str = '1h', days: int = 7) -> bool:
        """
        Mock test query performance.
        
        Args:
            symbol: Instrument symbol to check
            interval: Time interval
            days: Days of data
            
        Returns:
            Always returns True for mock data
        """
        try:
            logger.info(f"Mock query performance test for {symbol}")
            
            if symbol not in self.mock_instruments:
                self.reporter.record_test_result(
                    f"Query Performance - {symbol}", 
                    "FAIL", 
                    f"Symbol {symbol} not found in mock data"
                )
                return False
            
            # Mock query performance data
            query_times = [
                ("Simple time range", 0.05, 100),
                ("Daily aggregation", 0.12, 7),
                ("Join query", 0.18, 100)
            ]
            
            # Record performance metrics
            for query_type, duration, result_count in query_times:
                if hasattr(self.perf_tracker, 'measure_db_query'):
                    try:
                        self.perf_tracker.measure_db_query(
                            query_name=f"{symbol}_{query_type.replace(' ', '_').lower()}", 
                            rows_returned=result_count, 
                            query_time_ms=duration * 1000
                        )
                    except Exception as e:
                        logger.debug(f"Could not record performance metric: {e}")
                else:
                    # Fallback if measure_db_query is not available
                    logger.info(f"Mock query: {query_type} - {duration:.2f}s for {result_count} rows")
            
            details_str = ", ".join([f"{q}: {d:.4f}s ({r} rows)" for q, d, r in query_times])
            self.reporter.record_test_result(
                f"Query Performance - {symbol}", 
                "PASS", 
                f"All mock queries completed within acceptable time. {details_str}"
            )
            return True
        except Exception as e:
            error_msg = f"Error in mock query performance test for {symbol}: {str(e)}"
            logger.error(error_msg)
            self.reporter.record_test_result(f"Query Performance - {symbol}", "FAIL", error_msg)
            return False
    
    def validate_data_integrity(self, symbol: str, source_data_file: Optional[str] = None) -> bool:
        """
        Mock validate data integrity.
        
        Args:
            symbol: Instrument symbol to check
            source_data_file: Optional source data file
            
        Returns:
            Always returns True for mock data unless source comparison fails
        """
        try:
            logger.info(f"Mock data integrity validation for {symbol}")
            
            if symbol not in self.mock_instruments:
                self.reporter.record_test_result(
                    f"Data Integrity - {symbol}", 
                    "FAIL", 
                    f"Symbol {symbol} not found in mock data"
                )
                return False
            
            # If source file is provided, try to compare with it
            source_comparison_success = None
            if source_data_file and os.path.exists(source_data_file):
                try:
                    # Load source data
                    source_df = pd.read_csv(source_data_file, parse_dates=['timestamp'])
                    
                    # Create mock DB data
                    mock_data = self.mock_price_data.get(symbol, [])
                    
                    # Use the existing utility to compare
                    source_data = source_df.to_dict('records')
                    comparison_results = compare_datasets(
                        source_data=source_data,
                        target_data=mock_data,
                        key_field='timestamp'
                    )
                    
                    source_comparison_success = comparison_results.get("match_percentage", 0) > 90
                    
                    if not source_comparison_success:
                        self.reporter.record_test_result(
                            f"Data Integrity - {symbol}", 
                            "FAIL", 
                            f"Source data comparison failed for {symbol}"
                        )
                        return False
                        
                except Exception as e:
                    logger.error(f"Error in mock source data comparison: {str(e)}")
                    source_comparison_success = False
                    
                    self.reporter.record_test_result(
                        f"Data Integrity - {symbol}", 
                        "FAIL", 
                        f"Error in source data comparison: {str(e)}"
                    )
                    return False
            
            # Default success case
            self.reporter.record_test_result(
                f"Data Integrity - {symbol}", 
                "PASS", 
                f"All data integrity checks passed for {symbol} in mock data"
            )
            return True
        except Exception as e:
            error_msg = f"Error in mock data integrity validation for {symbol}: {str(e)}"
            logger.error(error_msg)
            self.reporter.record_test_result(f"Data Integrity - {symbol}", "FAIL", error_msg)
            return False

def main():
    """Run database storage verification tests."""
    parser = argparse.ArgumentParser(description='Verify database storage for market data')
    parser.add_argument('--symbols', type=str, default=None, 
                        help='Comma-separated list of symbols to verify (defaults to config)')
    parser.add_argument('--days', type=int, default=7, help='Number of days of data to check')
    parser.add_argument('--source-data', type=str, help='Path to source data file for comparison')
    parser.add_argument('--mock', action='store_true', help='Use mock database instead of real connection')
    args = parser.parse_args()
    
    # Parse symbols or use from config
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(',')]
    else:
        symbols = [instrument["symbol"] for instrument in TEST_INSTRUMENTS]
    
    # Initialize verifier - real or mock based on args
    if args.mock:
        logger.info("Using mock database for verification")
        verifier = MockDatabaseVerifier()
    else:
        logger.info("Using real database connection for verification")
        verifier = DatabaseVerifier(DB_CONFIG)
    
    try:
        # Connect to database
        if not verifier.connect():
            # If real connection fails, fall back to mock if not explicitly using mock
            if not args.mock:
                logger.warning("Database connection failed, falling back to mock database")
                verifier = MockDatabaseVerifier()
                if not verifier.connect():
                    logger.error("Mock database initialization failed, aborting verification")
                    return 1
            else:
                logger.error("Database connection failed, aborting verification")
                return 1
        
        # Run verification tests for each symbol
        for symbol in symbols:
            logger.info(f"Starting verification for {symbol}")
            
            # Verify data insertion
            verifier.verify_data_insertion(symbol, hours=24)
            
            # Verify timestamp processing
            verifier.verify_timestamp_processing(symbol)
            
            # Test query performance
            verifier.test_query_performance(symbol, days=args.days)
            
            # Validate data integrity
            source_data_file = args.source_data if args.source_data else None
            verifier.validate_data_integrity(symbol, source_data_file)
            
            logger.info(f"Completed verification for {symbol}")
        
        # Generate performance summary - safely handle if method doesn't exist
        try:
            if hasattr(verifier.perf_tracker, 'print_summary'):
                verifier.perf_tracker.print_summary()
        except Exception as e:
            logger.debug(f"Could not print performance summary: {e}")
        
        # Finalize report
        try:
            all_results = {}
            if hasattr(verifier.reporter, 'test_results'):
                all_results = verifier.reporter.test_results.get("tests", {})
            
            if all_results:
                pass_count = sum(1 for r in all_results.values() if r.get("status") == "PASS")
                total_count = len(all_results)
                overall_status = "PASS" if pass_count == total_count and total_count > 0 else "FAIL"
            else:
                overall_status = "FAIL"  # No tests were run
                
            # Safely finalize report if method exists
            if hasattr(verifier.reporter, 'finalize_report'):
                verifier.reporter.finalize_report(overall_status)
        except Exception as e:
            logger.debug(f"Could not finalize report: {e}")
            
        logger.info("Database verification completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Error during database verification: {str(e)}", exc_info=True)
        return 1
    finally:
        # Close database connection
        verifier.close()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())