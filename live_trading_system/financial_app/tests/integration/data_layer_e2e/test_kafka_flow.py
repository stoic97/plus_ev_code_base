#!/usr/bin/env python3
"""
End-to-End Kafka Flow Testing Module

This module coordinates the testing of the complete Kafka flow including both 
producer and consumer components, though in this implementation we're focusing
on the producer verification aspects.

The test can run with real Kafka or fall back to a mock implementation when
Kafka is not available.
"""

import logging
import argparse
import time
import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

# Import configuration
from financial_app.tests.integration.data_layer_e2e import e2e_config

# Import test components
from financial_app.tests.integration.data_layer_e2e.test_kafka_producer import run_kafka_producer_verification
from financial_app.tests.integration.data_layer_e2e.utils.reporting import TestReporter, generate_html_report
from financial_app.tests.integration.data_layer_e2e.utils.performance import PerformanceTracker, timed_operation



# Configure logging
logger = logging.getLogger(__name__)

def run_kafka_flow_test() -> Dict[str, Any]:
    """
    Run the complete Kafka flow test.
    
    Returns:
        Dict containing test results and metrics
    """
    logger.info("Initializing Kafka flow test")
    
    # Initialize reporter and performance tracker
    reporter = TestReporter()
    perf_tracker = PerformanceTracker()
    
    # Initialize result tracking
    results = {
        "timestamp": datetime.now().isoformat(),
        "producer_verification": {
            "success": False,
            "metrics": {}
        },
        # Consumer verification would go here in future implementation
        "overall_success": False
    }
    
    # Log test configuration
    logger.info(f"Bootstrap servers: {e2e_config.KAFKA_CONFIG['bootstrap_servers']}")
    logger.info(f"Market data topic: {e2e_config.KAFKA_CONFIG['market_data_topic']}")
    logger.info(f"Error topic: {e2e_config.KAFKA_CONFIG['error_topic']}")
    
    # Step 1: Kafka Producer Verification
    logger.info("=== Kafka Producer Verification ===")
    
    with timed_operation(perf_tracker, "producer_verification"):
        producer_success = run_kafka_producer_verification()
    
    results["producer_verification"]["success"] = producer_success
    
    if not producer_success:
        logger.error("Kafka producer verification failed - stopping Kafka flow test")
        reporter.record_test_result(
            "kafka_flow_test",
            "FAIL",
            {"producer_verification": "FAIL"},
            "Kafka producer verification failed"
        )
        results["overall_success"] = False
        
        # Finalize report
        reporter.finalize_report("FAIL")
        
        return results
    
    logger.info("Kafka producer verification passed")
    
    # Step 2: Kafka Consumer Testing (to be implemented in a future task)
    # This would be a separate function call similar to run_kafka_producer_verification
    # For now, we'll consider the test successful if the producer verification passed
    
    # Overall success
    results["overall_success"] = producer_success
    
    # Record final test result
    status = "PASS" if results["overall_success"] else "FAIL"
    reporter.record_test_result(
        "kafka_flow_test",
        status,
        {
            "producer_verification": "PASS" if producer_success else "FAIL",
            "consumer_verification": "NOT_IMPLEMENTED"
        }
    )
    
    # Finalize report
    reporter.finalize_report(status)
    
    # Generate HTML report
    html_report_path = generate_html_report(reporter)
    logger.info(f"HTML report generated: {html_report_path}")
    
    if results["overall_success"]:
        logger.info("=== Kafka Flow Test: PASSED ===")
    else:
        logger.error("=== Kafka Flow Test: FAILED ===")
    
    return results


if __name__ == "__main__":
    # Configure command-line arguments
    parser = argparse.ArgumentParser(description="Run Kafka flow end-to-end tests")
    
    # Kafka override options
    parser.add_argument(
        "--bootstrap-servers", 
        default=None,
        help="Comma-separated list of Kafka bootstrap servers"
    )
    parser.add_argument(
        "--market-data-topic", 
        default=None,
        help="Kafka topic for market data"
    )
    
    # Mock mode option
    parser.add_argument(
        "--use-mock", 
        action="store_true",
        help="Force use of mock Kafka implementation"
    )
    
    # Logging options
    parser.add_argument(
        "--log-level", 
        default=e2e_config.LOG_LEVEL,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Set mock flag from command line
    if args.use_mock:
        os.environ["USE_MOCK_KAFKA"] = "true"
        logger.info("Using mock Kafka implementation (set by command line)")
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Apply command-line overrides to config
    if args.bootstrap_servers:
        e2e_config.KAFKA_CONFIG["bootstrap_servers"] = args.bootstrap_servers
    
    if args.market_data_topic:
        e2e_config.KAFKA_CONFIG["market_data_topic"] = args.market_data_topic
    
    # Generate log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(e2e_config.TEST_REPORT_DIR, f"kafka_flow_test_{timestamp}.log")
    
    # Configure file logging
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)
    
    # Run the test
    logger.info("Starting Kafka flow end-to-end test")
    results = run_kafka_flow_test()
    
    # Exit with appropriate code
    exit(0 if results["overall_success"] else 1)