#!/usr/bin/env python
"""
Authentication & Security Testing Runner

This script serves as a simple entry point for running the authentication and
security tests for the trading application.

Usage:
    python run_auth_test.py [--verbose]

Author: Your Name
Date: April 24, 2025
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# Add the project root to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import test modules
from financial_app.tests.integration.data_layer_e2e.tests.test_auth_security import AuthSecurityTester

def setup_logging(verbose=False):
    """Set up logging configuration for the test runner."""
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Set up logging
    log_file = os.path.join(log_dir, f"auth_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger("AuthTestRunner")
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run authentication and security tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    return parser.parse_args()

def main():
    """Main function to run the authentication and security tests."""
    args = parse_args()
    logger = setup_logging(args.verbose)
    
    logger.info("Starting Authentication & Security Tests")
    print("\n" + "="*80)
    print("AUTHENTICATION & SECURITY TESTING")
    print("="*80)
    
    try:
        # Run authentication and security tests
        tester = AuthSecurityTester()
        success = tester.run_all_tests()
        
        if success:
            message = "All authentication and security tests passed"
            logger.info(message)
            print(f"\n✅ {message}")
            print("="*80)
            return 0
        else:
            message = "Some authentication and security tests failed"
            logger.error(message)
            print(f"\n❌ {message}")
            print("="*80)
            return 1
            
    except Exception as e:
        message = f"Error running tests: {str(e)}"
        logger.exception(message)
        print(f"\n❌ {message}")
        print("="*80)
        return 1

if __name__ == "__main__":
    sys.exit(main())