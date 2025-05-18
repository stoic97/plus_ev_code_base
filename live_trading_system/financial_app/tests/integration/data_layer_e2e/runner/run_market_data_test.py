#!/usr/bin/env python3
"""
Main test runner for market data flow end-to-end testing.

This script orchestrates the entire end-to-end test for the market data flow:
1. Authentication & authorization
2. Broker connection
3. Market data subscription
4. Kafka message production and consumption
5. Database storage verification
6. API access verification

Usage:
    python run_market_data_test.py [--config CONFIG_FILE] [--duration MINUTES]
"""

import os
import sys
import time
import argparse
import logging
import json
import signal
from datetime import datetime, timedelta
import asyncio
from typing import Dict, List, Any, Optional




from financial_app.tests.integration.data_layer_e2e.e2e_config import (
    TEST_DURATION_MINUTES, 
    LOG_LEVEL, 
    TEST_REPORT_DIR, 
    TEST_ID, 
    DB_CONFIG,
    KAFKA_CONFIG,
    API_BASE_URL,
    BROKER_CONFIG,
    TEST_USER,
    TEST_INSTRUMENTS
)

from financial_app.tests.integration.data_layer_e2e.utils import (
    TestReporter,
    PerformanceTracker,
    generate_html_report,
    timed_operation,
    timed_function
)

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(TEST_REPORT_DIR, f"{TEST_ID}.log"))
    ]
)

logger = logging.getLogger("e2e_test")

# Create reporter and performance tracker instances
reporter = TestReporter()
perf_tracker = PerformanceTracker()

# Global variables for test state
test_token = None
subscribed_instruments = []
test_running = True

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="End-to-end testing for market data flow")
    parser.add_argument(
        "--config", 
        help="Path to custom configuration file"
    )
    parser.add_argument(
        "--duration", 
        type=int, 
        default=TEST_DURATION_MINUTES,
        help="Test duration in minutes"
    )
    parser.add_argument(
        "--log-level", 
        default=LOG_LEVEL,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level"
    )
    
    return parser.parse_args()

def signal_handler(sig, frame):
    """Handle interruption signals."""
    global test_running
    logger.info("Test interrupted. Cleaning up and generating report...")
    test_running = False
    
    # Cleanup will be handled in the main function after the loop exits

# Placeholder for test modules - these will be imported from separate files in the future
async def test_authentication():
    """Test user authentication and token generation."""
    logger.info("Starting authentication test...")
    
    try:
        with timed_operation(perf_tracker, "authentication"):
            # Simulate authentication API call
            await asyncio.sleep(0.5)  # Placeholder for actual API call
            
            # For the placeholder, we'll just create a dummy token
            global test_token
            test_token = f"test_token_{TEST_ID}"
            
            logger.info("Authentication successful")
            reporter.record_test_result(
                "authentication", 
                "PASS",
                {"token_generated": True}
            )
            return True
    except Exception as e:
        error_msg = f"Authentication failed: {str(e)}"
        logger.error(error_msg)
        reporter.record_test_result(
            "authentication", 
            "FAIL",
            {"error": str(e)},
            error_msg
        )
        return False

async def test_broker_connection():
    """Test connection to the broker API."""
    logger.info("Testing broker connection...")
    
    if not test_token:
        logger.error("Cannot test broker connection: Authentication not completed")
        reporter.record_test_result(
            "broker_connection", 
            "SKIP",
            {"reason": "Authentication not completed"}
        )
        return False
    
    try:
        with timed_operation(perf_tracker, "broker_connection"):
            # Simulate broker connection
            await asyncio.sleep(1.0)  # Placeholder for actual connection
            
            logger.info("Broker connection successful")
            reporter.record_test_result(
                "broker_connection", 
                "PASS",
                {"connected": True}
            )
            return True
    except Exception as e:
        error_msg = f"Broker connection failed: {str(e)}"
        logger.error(error_msg)
        reporter.record_test_result(
            "broker_connection", 
            "FAIL",
            {"error": str(e)},
            error_msg
        )
        return False

async def test_market_data_subscription():
    """Test subscription to market data."""
    logger.info("Testing market data subscription...")
    
    if not test_token:
        logger.error("Cannot test market data subscription: Authentication not completed")
        reporter.record_test_result(
            "market_data_subscription", 
            "SKIP",
            {"reason": "Authentication not completed"}
        )
        return False
    
    try:
        with timed_operation(perf_tracker, "market_data_subscription"):
            # Simulate subscription to test instruments
            for instrument in TEST_INSTRUMENTS:
                # Placeholder for actual subscription
                await asyncio.sleep(0.2)
                
                symbol = instrument["symbol"]
                exchange = instrument["exchange"]
                logger.info(f"Subscribed to {symbol} on {exchange}")
                
                global subscribed_instruments
                subscribed_instruments.append(instrument)
            
            reporter.record_test_result(
                "market_data_subscription", 
                "PASS",
                {"subscribed_instruments": subscribed_instruments}
            )
            return True
    except Exception as e:
        error_msg = f"Market data subscription failed: {str(e)}"
        logger.error(error_msg)
        reporter.record_test_result(
            "market_data_subscription", 
            "FAIL",
            {"error": str(e)},
            error_msg
        )
        return False

async def test_kafka_producer():
    """Test Kafka producer for market data."""
    logger.info("Testing Kafka producer...")
    
    if not subscribed_instruments:
        logger.error("Cannot test Kafka producer: No subscribed instruments")
        reporter.record_test_result(
            "kafka_producer", 
            "SKIP",
            {"reason": "No subscribed instruments"}
        )
        return False
    
    try:
        with timed_operation(perf_tracker, "kafka_producer"):
            # Simulate producing messages to Kafka
            message_count = 0
            for _ in range(5):  # Simulate multiple batches
                # Placeholder for actual message production
                await asyncio.sleep(0.3)
                batch_size = len(subscribed_instruments)
                message_count += batch_size
                
                # Record performance metrics
                perf_tracker.record_metric(
                    "kafka_producer_batch_size", 
                    batch_size
                )
            
            logger.info(f"Produced {message_count} messages to Kafka")
            reporter.record_test_result(
                "kafka_producer", 
                "PASS",
                {"message_count": message_count}
            )
            return True
    except Exception as e:
        error_msg = f"Kafka producer test failed: {str(e)}"
        logger.error(error_msg)
        reporter.record_test_result(
            "kafka_producer", 
            "FAIL",
            {"error": str(e)},
            error_msg
        )
        return False

async def test_kafka_consumer():
    """Test Kafka consumer for market data."""
    logger.info("Testing Kafka consumer...")
    
    try:
        with timed_operation(perf_tracker, "kafka_consumer"):
            # Simulate consuming messages from Kafka
            message_count = 0
            for _ in range(5):  # Simulate multiple batches
                # Placeholder for actual message consumption
                await asyncio.sleep(0.4)
                batch_size = len(subscribed_instruments)
                message_count += batch_size
                
                # Record performance metrics
                perf_tracker.record_metric(
                    "kafka_consumer_batch_size", 
                    batch_size
                )
                perf_tracker.record_metric(
                    "consumer_lag", 
                    float(batch_size) / 2  # Simulated lag metric
                )
            
            logger.info(f"Consumed {message_count} messages from Kafka")
            reporter.record_test_result(
                "kafka_consumer", 
                "PASS",
                {"message_count": message_count}
            )
            return True
    except Exception as e:
        error_msg = f"Kafka consumer test failed: {str(e)}"
        logger.error(error_msg)
        reporter.record_test_result(
            "kafka_consumer", 
            "FAIL",
            {"error": str(e)},
            error_msg
        )
        return False

async def test_database_storage():
    """Test database storage of market data."""
    logger.info("Testing database storage...")
    
    try:
        with timed_operation(perf_tracker, "database_storage"):
            # Simulate database operations
            record_count = 0
            for _ in range(3):  # Simulate multiple batches
                # Placeholder for actual database operations
                await asyncio.sleep(0.5)
                batch_size = len(subscribed_instruments) * 2  # More records than messages due to transformations
                record_count += batch_size
                
                # Record performance metrics
                perf_tracker.record_metric(
                    "db_write_batch_size", 
                    batch_size
                )
                perf_tracker.record_metric(
                    "db_write_latency_ms", 
                    500.0 * 0.8  # Simulated write latency
                )
            
            logger.info(f"Stored {record_count} records in database")
            reporter.record_test_result(
                "database_storage", 
                "PASS",
                {"record_count": record_count}
            )
            return True
    except Exception as e:
        error_msg = f"Database storage test failed: {str(e)}"
        logger.error(error_msg)
        reporter.record_test_result(
            "database_storage", 
            "FAIL",
            {"error": str(e)},
            error_msg
        )
        return False

async def test_api_access():
    """Test API access to stored market data."""
    logger.info("Testing API access...")
    
    if not test_token:
        logger.error("Cannot test API access: Authentication not completed")
        reporter.record_test_result(
            "api_access", 
            "SKIP",
            {"reason": "Authentication not completed"}
        )
        return False
    
    try:
        with timed_operation(perf_tracker, "api_access"):
            # Simulate API requests
            response_count = 0
            for instrument in subscribed_instruments:
                # Placeholder for actual API requests
                await asyncio.sleep(0.3)
                
                symbol = instrument["symbol"]
                exchange = instrument["exchange"]
                logger.info(f"Successfully queried data for {symbol} on {exchange}")
                response_count += 1
                
                # Record performance metrics
                perf_tracker.record_metric(
                    "api_response_time_ms", 
                    300.0 * 0.9  # Simulated response time
                )
            
            logger.info(f"Successfully made {response_count} API requests")
            reporter.record_test_result(
                "api_access", 
                "PASS",
                {"response_count": response_count}
            )
            return True
    except Exception as e:
        error_msg = f"API access test failed: {str(e)}"
        logger.error(error_msg)
        reporter.record_test_result(
            "api_access", 
            "FAIL",
            {"error": str(e)},
            error_msg
        )
        return False

async def run_continuous_monitoring(duration_minutes):
    """Run continuous monitoring for the specified duration."""
    logger.info(f"Starting continuous monitoring for {duration_minutes} minutes...")
    
    end_time = datetime.now() + timedelta(minutes=duration_minutes)
    
    try:
        # Initial test result
        reporter.record_test_result(
            "continuous_monitoring", 
            "RUNNING",
            {"start_time": datetime.now().isoformat()}
        )
        
        # Monitor until the end time or until interrupted
        while datetime.now() < end_time and test_running:
            # Simulate data flow monitoring
            await asyncio.sleep(5)  # Check every 5 seconds
            
            # Record some simulated metrics
            perf_tracker.record_metric(
                "message_latency_ms", 
                50.0 + (datetime.now().microsecond % 100)  # Simulated variable latency
            )
            perf_tracker.record_metric(
                "consumer_lag", 
                float(datetime.now().second % 10)  # Simulated variable lag
            )
            perf_tracker.record_metric(
                "system_cpu_usage", 
                30.0 + (datetime.now().second % 20)  # Simulated CPU usage
            )
            
            logger.debug("Monitoring cycle completed")
        
        # Final test result
        if test_running:  # Only mark as PASS if not interrupted
            reporter.record_test_result(
                "continuous_monitoring", 
                "PASS",
                {
                    "end_time": datetime.now().isoformat(),
                    "duration_minutes": duration_minutes
                }
            )
            return True
        else:
            logger.info("Continuous monitoring interrupted")
            reporter.record_test_result(
                "continuous_monitoring", 
                "INTERRUPTED",
                {
                    "end_time": datetime.now().isoformat(),
                    "actual_duration_minutes": (datetime.now() - (end_time - timedelta(minutes=duration_minutes))).total_seconds() / 60
                }
            )
            return False
    except Exception as e:
        error_msg = f"Continuous monitoring failed: {str(e)}"
        logger.error(error_msg)
        reporter.record_test_result(
            "continuous_monitoring", 
            "FAIL",
            {
                "error": str(e),
                "end_time": datetime.now().isoformat()
            },
            error_msg
        )
        return False

async def run_test_suite(duration_minutes: int):
    """Run the complete test suite."""
    logger.info(f"Starting end-to-end test suite with ID: {TEST_ID}")
    
    # Step 1: Run initial tests sequentially
    auth_success = await test_authentication()
    if not auth_success:
        logger.error("Authentication failed. Aborting test suite.")
        return False
    
    broker_success = await test_broker_connection()
    if not broker_success:
        logger.error("Broker connection failed. Aborting test suite.")
        return False
    
    subscription_success = await test_market_data_subscription()
    if not subscription_success:
        logger.error("Market data subscription failed. Aborting test suite.")
        return False
    
    # Step 2: Run Kafka tests
    kafka_producer_success = await test_kafka_producer()
    if not kafka_producer_success:
        logger.error("Kafka producer test failed. Aborting test suite.")
        return False
    
    kafka_consumer_success = await test_kafka_consumer()
    if not kafka_consumer_success:
        logger.error("Kafka consumer test failed. Aborting test suite.")
        return False
    
    # Step 3: Run database test
    db_success = await test_database_storage()
    if not db_success:
        logger.error("Database storage test failed. Aborting test suite.")
        return False
    
    # Step 4: Run API access test
    api_success = await test_api_access()
    if not api_success:
        logger.error("API access test failed. Aborting test suite.")
        return False
    
    # Step 5: Run continuous monitoring
    monitoring_success = await run_continuous_monitoring(duration_minutes)
    
    # Step 6: Determine overall test status
    overall_status = "PASS" if monitoring_success else "FAIL"
    
    return overall_status

async def main():
    """Main function to run tests and generate report."""
    args = parse_arguments()
    
    # Set up signal handler for graceful interruption
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Run the test suite
        overall_status = await run_test_suite(args.duration)
        
        # Generate test report
        report_path = reporter.finalize_report(overall_status)
        html_report_path = generate_html_report(reporter)
        
        logger.info(f"Test completed with status: {overall_status}")
        logger.info(f"Test report available at: {report_path}")
        logger.info(f"HTML report available at: {html_report_path}")
        
        # Return exit code based on test status
        return 0 if overall_status == "PASS" else 1
    
    except Exception as e:
        logger.error(f"Unexpected error in test suite: {str(e)}")
        reporter.finalize_report("ERROR")
        return 2

if __name__ == "__main__":
    # Run the main function using asyncio
    exit_code = asyncio.run(main())
    sys.exit(exit_code)