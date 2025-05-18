#!/usr/bin/env python3
"""
Main runner script for Kafka consumer end-to-end testing.
This script sets up the test environment and executes the Kafka consumer tests.
"""

import os
import sys
import logging
import argparse
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("kafka_test_runner")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Kafka Consumer End-to-End Test Runner')
    
    parser.add_argument(
        '--use-mock', 
        action='store_true',
        help='Force the use of mock implementations even if real services are available'
    )
    
    parser.add_argument(
        '--bootstrap-servers',
        type=str,
        help='Kafka bootstrap servers (overrides config)'
    )
    
    parser.add_argument(
        '--test-duration',
        type=int,
        default=None,
        help='Test duration in minutes (overrides config)'
    )
    
    parser.add_argument(
        '--report-dir',
        type=str,
        default=None,
        help='Directory to store test reports (overrides config)'
    )
    
    return parser.parse_args()

def setup_environment(args):
    """Set up the test environment based on arguments"""
    # Import project-specific modules
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    
    # Import the configuration
    from financial_app.tests.integration.data_layer_e2e.e2e_config import (
        KAFKA_CONFIG, TEST_REPORT_DIR, TEST_DURATION_MINUTES
    )
    
    # Override config with command line arguments if provided
    if args.bootstrap_servers:
        KAFKA_CONFIG['bootstrap_servers'] = args.bootstrap_servers
        
    if args.test_duration:
        TEST_DURATION_MINUTES = args.test_duration
        
    if args.report_dir:
        TEST_REPORT_DIR = args.report_dir
    
    # Make sure directories exist
    os.makedirs(TEST_REPORT_DIR, exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # If forcing mock mode, patch imports
    if args.use_mock:
        logger.info("Forcing mock mode for all tests")
        patch_imports_with_mocks()
    
    return {
        'KAFKA_CONFIG': KAFKA_CONFIG, 
        'TEST_REPORT_DIR': TEST_REPORT_DIR, 
        'TEST_DURATION_MINUTES': TEST_DURATION_MINUTES
    }

def patch_imports_with_mocks():
    """Patch real modules with mock implementations"""
    import sys
    from unittest import mock
    
    # Import our mock implementation
    from financial_app.tests.integration.data_layer_e2e.mocks.mock_kafka import (
        MockConsumerManager, MockProducer
    )
    
    # Mock the Kafka module
    kafka_mock = mock.MagicMock()
    kafka_admin_mock = mock.MagicMock()
    
    # Set up the mocks with our implementations
    kafka_mock.KafkaConsumer = MockConsumerManager
    kafka_mock.KafkaProducer = MockProducer
    
    # Patch the modules
    sys.modules['kafka'] = kafka_mock
    sys.modules['kafka.admin'] = kafka_admin_mock
    
    # Ensure any attempt to import real consumer will use our mock
    class ImportBlocker(object):
        def find_module(self, fullname, path=None):
            if fullname == 'app.kafka.consumer':
                return self
            return None
            
        def load_module(self, fullname):
            if fullname in sys.modules:
                return sys.modules[fullname]
            
            module = mock.MagicMock()
            module.KafkaConsumerManager = MockConsumerManager
            sys.modules[fullname] = module
            return module
    
    # Register the import blocker
    sys.meta_path.insert(0, ImportBlocker())
    
    logger.info("Imports patched with mock implementations")

def setup_mock_data():
    """Set up mock data for testing"""
    try:
        logger.info("Setting up mock data for testing")
        from financial_app.tests.integration.data_layer_e2e.mocks.mock_kafka import publish_mock_market_data
        from financial_app.tests.integration.data_layer_e2e.e2e_config import KAFKA_CONFIG
        
        # Publish mock data to the Kafka topic
        publish_mock_market_data(
            topic=KAFKA_CONFIG['market_data_topic'],
            num_messages=500,  # Generate enough data for tests
            include_errors=True  # Include some errors for testing
        )
    except Exception as e:
        logger.warning(f"Error setting up mock data: {str(e)}")

def run_kafka_consumer_tests():
    """Run the Kafka consumer tests"""
    try:
        # Now import the test module
        from financial_app.tests.integration.data_layer_e2e.tests.test_kafka_consumer import KafkaConsumerTest
        
        # Create and run the tests
        consumer_test = KafkaConsumerTest()
        success = consumer_test.run_all_tests()
        
        return success
    except Exception as e:
        logger.error(f"Error running Kafka consumer tests: {str(e)}", exc_info=True)
        return False

def main():
    """Main entry point"""
    start_time = datetime.now()
    logger.info(f"Starting Kafka consumer E2E tests at {start_time.isoformat()}")
    
    # Parse arguments
    args = parse_args()
    
    # Set up the environment
    config = setup_environment(args)
    
    # Set up mock data if using mocks
    if args.use_mock:
        setup_mock_data()
    
    # Run the tests
    success = run_kafka_consumer_tests()
    
    # Log results
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    logger.info(f"Tests completed in {duration:.2f} seconds with {'SUCCESS' if success else 'FAILURE'}")
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()