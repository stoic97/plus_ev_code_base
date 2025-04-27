"""
Script to run API access tests for market data flow.

This script runs the API access tests independently or as part of a larger test suite.
It handles test execution, reporting, and performance metrics collection.
"""

import argparse
import logging
import sys
import os
from datetime import datetime

# Add project root to path if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from financial_app.tests.integration.data_layer_e2e.utils.reporting import TestReporter, generate_html_report
from financial_app.tests.integration.data_layer_e2e.utils.performance import PerformanceTracker
from financial_app.tests.integration.data_layer_e2e.test_api_access import run_api_access_tests
from financial_app.tests.integration.data_layer_e2e.e2e_config import LOG_LEVEL, MOCK_MODE
from financial_app.tests.integration.data_layer_e2e.utils.service_check import get_service_availability


def setup_logging(log_level_str: str, log_file: str = None) -> None:
    """
    Set up logging configuration.
    
    Args:
        log_level_str: Log level as string ('DEBUG', 'INFO', etc.)
        log_file: Optional log file path
    """
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers
    )


def main() -> int:
    """
    Main entry point for running API access tests.
    
    Handles command-line arguments, sets up logging, runs the tests,
    and generates reports.
    
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run API access tests for market data flow")
    parser.add_argument("--log-level", default=LOG_LEVEL, help="Logging level")
    parser.add_argument("--log-file", help="Log file path")
    parser.add_argument("--report-dir", help="Directory for test reports")
    parser.add_argument("--html-report", action="store_true", help="Generate HTML report")
    parser.add_argument("--mock-mode", choices=["always", "never", "auto"], default=MOCK_MODE,
                      help="Mock mode: 'always' to always use mocks, 'never' to never use mocks, 'auto' to use real services if available")
    args = parser.parse_args()
    
    # Set up logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = args.log_file or f"logs/api_tests_{timestamp}.log"
    setup_logging(args.log_level, log_file)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting API access tests")
    
    # Override mock mode from command line if specified
    if args.mock_mode:
        os.environ["MOCK_MODE"] = args.mock_mode
        logger.info(f"Mock mode set to '{args.mock_mode}' from command line")
    
    # Check service availability
    service_status = get_service_availability()
    logger.info("Service availability status:")
    logger.info(f"  API: {'Available' if service_status['api'] else 'Not available'}")
    logger.info(f"  Database: {'Available' if service_status['database'] else 'Not available'}")
    logger.info(f"  Kafka: {'Available' if service_status['kafka'] else 'Not available'}")
    
    # Create test reporter and performance tracker
    reporter = TestReporter()
    performance_tracker = PerformanceTracker()
    
    try:
        # Run API access tests
        success = run_api_access_tests(reporter, performance_tracker)
        
        # Finalize report
        report_file = reporter.finalize_report("PASS" if success else "FAIL")
        
        # Generate HTML report if requested
        if args.html_report:
            html_report = generate_html_report(reporter)
            logger.info(f"HTML report generated: {html_report}")
        
        # Print performance metrics
        logger.info("Performance Metrics:")
        for metric, stats in performance_tracker.get_all_metrics().items():
            logger.info(f"{metric}: avg={stats['mean']:.2f}ms, p95={stats['p95']:.2f}ms")
        
        # Check for performance violations
        violations = performance_tracker.check_thresholds()
        if violations:
            logger.warning(f"Performance threshold violations detected: {len(violations)}")
            for violation in violations:
                logger.warning(f"Threshold violation: {violation['metric']} = {violation['max_value']} (threshold: {violation['threshold']})")
        
        logger.info(f"API access tests completed with {'SUCCESS' if success else 'FAILURE'}")
        return 0 if success else 1
        
    except Exception as e:
        logger.error(f"Error running API access tests: {str(e)}", exc_info=True)
        reporter.record_test_result("api_tests", "FAIL", error=f"Unhandled exception: {str(e)}")
        reporter.finalize_report("FAIL")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)