#!/usr/bin/env python
"""
Script to run database verification tests.

This script provides a convenient way to launch the database
verification tests with default settings.
"""

import os
import sys
import argparse
import logging
from datetime import datetime

from financial_app.tests.integration.data_layer_e2e.tests.test_db_storage import main as run_verify_db_storage
from financial_app.tests.integration.data_layer_e2e.e2e_config import TEST_REPORT_DIR


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run database verification tests')
    parser.add_argument('--symbols', type=str, help='Comma-separated list of symbols to verify')
    parser.add_argument('--days', type=int, default=7, help='Number of days of data to check')
    parser.add_argument('--source-data', type=str, help='Path to source data file for comparison')
    parser.add_argument('--report-dir', type=str, default=TEST_REPORT_DIR, help='Directory to save reports')
    parser.add_argument('--mock', action='store_true', help='Force use of mock database instead of real connection')
    
    args = parser.parse_args()
    
    # Create report directory
    os.makedirs(args.report_dir, exist_ok=True)
    
    # Set up logging
    log_file = os.path.join(
        args.report_dir, 
        f"db_verification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting database verification tests")
    
    # Build arguments for the verification script
    verify_args = []
    
    if args.symbols:
        verify_args.extend(['--symbols', args.symbols])
    
    if args.days:
        verify_args.extend(['--days', str(args.days)])
    
    if args.source_data:
        verify_args.extend(['--source-data', args.source_data])
        
    if args.mock:
        verify_args.append('--mock')
    
    # Run the verification script with the provided arguments
    old_argv = sys.argv
    try:
        sys.argv = [sys.argv[0]] + verify_args
        exit_code = run_verify_db_storage()
        
        if exit_code == 0:
            logger.info("Database verification completed successfully")
        else:
            logger.error(f"Database verification failed with exit code {exit_code}")
        
        sys.exit(exit_code)
    except Exception as e:
        logger.error(f"Error running database verification: {str(e)}", exc_info=True)
        sys.exit(1)
    finally:
        sys.argv = old_argv