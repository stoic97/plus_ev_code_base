"""
Broker connection tests for the end-to-end testing framework.

This module implements comprehensive tests for broker API connectivity,
including connection establishment, authentication, reconnection capability,
and secure connection verification.
"""

import os
import time
import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
from functools import wraps

# Application imports
from financial_app.app.services.fyers_client import FyersClient
from financial_app.tests.integration.data_layer_e2e.e2e_config import (
    BROKER_CONFIG,
    TEST_USER
)
from financial_app.tests.integration.data_layer_e2e.utils import (
    timed_operation,
    timed_function
)


# Custom wrappers for timed operations when performance tracker is None
def safe_timed_operation(perf_tracker, operation_name):
    class DummyContext:
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
    
    if perf_tracker is None:
        return DummyContext()
    else:
        return timed_operation(perf_tracker, operation_name)

def safe_timed_function(operation_name):
    def decorator(func):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            try:
                if self.perf_tracker:
                    with timed_operation(self.perf_tracker, operation_name):
                        return await func(self, *args, **kwargs)
                else:
                    return await func(self, *args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {operation_name}: {str(e)}")
                raise
        return wrapper
    return decorator

logger = logging.getLogger("broker_connection_test")

class BrokerConnectionTester:
    """Tester class for broker connection functionality."""
    
    def __init__(self, 
                 config_override: Optional[Dict[str, Any]] = None,
                 perf_tracker: Any = None,
                 reporter: Any = None):
        """
        Initialize the broker connection tester.
        
        Args:
            config_override: Optional configuration overrides
            perf_tracker: Performance tracker instance
            reporter: Test reporter instance
        """
        self.config = config_override or BROKER_CONFIG
        self.perf_tracker = perf_tracker
        self.reporter = reporter
        self.broker = None
        self.test_run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
    async def setup(self) -> bool:
        """
        Set up the broker connection test environment.
        
        Returns:
            bool: True if setup successful, False otherwise
        """
        logger.info("Setting up broker connection test environment")
        try:
            # Use a test-specific config path if needed
            test_config_path = os.getenv("TEST_BROKER_CONFIG_PATH", "config/broker_config.yaml")
            
            # Initialize broker client but don't connect yet
            self.broker = FyersClient(config_path=test_config_path)
            logger.info(f"Successfully initialized broker client with config: {test_config_path}")
            return True
        except Exception as e:
            logger.error(f"Broker test setup failed: {str(e)}")
            if self.reporter:
                self.reporter.record_test_result(
                    "broker_connection_setup", 
                    "FAIL",
                    {"error": str(e)},
                    f"Setup error: {str(e)}"
                )
            return False
    
    @safe_timed_function("broker_connection_test")
    async def test_connection_establishment(self) -> bool:
        """
        Test basic connection to the broker API.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        logger.info("Testing broker connection establishment")
        try:
            # We need to run the synchronous connect method in a separate thread
            # to avoid blocking the event loop
            connection_result = await asyncio.to_thread(self.broker.connect)
            
            if connection_result:
                logger.info("Connection established successfully")
                if self.reporter:
                    self.reporter.record_test_result(
                        "broker_connection_establishment", 
                        "PASS",
                        {"connected": True}
                    )
                return True
            else:
                logger.error("Failed to establish connection")
                if self.reporter:
                    self.reporter.record_test_result(
                        "broker_connection_establishment", 
                        "FAIL",
                        {"connected": False},
                        "Failed to establish connection to broker API"
                    )
                return False
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            if self.reporter:
                self.reporter.record_test_result(
                    "broker_connection_establishment", 
                    "FAIL",
                    {"error": str(e)},
                    f"Connection error: {str(e)}"
                )
            return False
    
    @safe_timed_function("broker_connection_test")
    async def test_authentication(self) -> bool:
        """
        Test authentication with the broker API.
        
        Returns:
            bool: True if authentication successful, False otherwise
        """
        logger.info("Testing broker authentication")
        try:
            # We need to run the synchronous get_profile method in a separate thread
            profile = await asyncio.to_thread(self.broker.get_profile)
            
            if profile and len(profile) > 0:
                logger.info(f"Authentication successful for user: {profile.get('name', 'Unknown')}")
                if self.reporter:
                    self.reporter.record_test_result(
                        "broker_authentication", 
                        "PASS",
                        {
                            "authenticated": True,
                            "user": profile.get('name', 'Unknown')
                        }
                    )
                return True
            else:
                logger.error("Authentication failed or returned empty profile")
                if self.reporter:
                    self.reporter.record_test_result(
                        "broker_authentication", 
                        "FAIL",
                        {"authenticated": False},
                        "Authentication failed or returned empty profile"
                    )
                return False
        except Exception as e:
            logger.error(f"Authentication test failed: {str(e)}")
            if self.reporter:
                self.reporter.record_test_result(
                    "broker_authentication", 
                    "FAIL",
                    {"error": str(e)},
                    f"Authentication error: {str(e)}"
                )
            return False
    
    @safe_timed_function("broker_connection_test")
    async def test_reconnection(self) -> bool:
        """
        Test reconnection capability on failure.
        
        Returns:
            bool: True if reconnection successful, False otherwise
        """
        logger.info("Testing broker reconnection capability")
        try:
            # Simulate disconnection
            logger.info("Simulating broker disconnection")
            await asyncio.to_thread(self._simulate_disconnection)
            
            # Check if disconnection worked
            try:
                # This should fail if disconnection worked properly
                profile_after_disconnect = await asyncio.to_thread(self.broker.get_profile)
                if profile_after_disconnect and len(profile_after_disconnect) > 0:
                    logger.warning("Failed to properly simulate disconnection")
                    # We'll continue with the test anyway
            except Exception:
                # This exception means disconnection worked as expected
                logger.info("Disconnection simulation successful")
            
            # Attempt reconnection
            logger.info("Attempting reconnection")
            reconnect_result = await asyncio.to_thread(self.broker.connect)
            
            # Verify reconnection by checking profile access
            profile_after_reconnect = await asyncio.to_thread(self.broker.get_profile)
            reconnect_successful = reconnect_result and profile_after_reconnect and len(profile_after_reconnect) > 0
            
            if reconnect_successful:
                logger.info("Reconnection successful")
                if self.reporter:
                    self.reporter.record_test_result(
                        "broker_reconnection", 
                        "PASS",
                        {"reconnected": True}
                    )
                return True
            else:
                logger.error("Failed to reconnect")
                if self.reporter:
                    self.reporter.record_test_result(
                        "broker_reconnection", 
                        "FAIL",
                        {"reconnected": False},
                        "Failed to reconnect to broker API"
                    )
                return False
        except Exception as e:
            logger.error(f"Reconnection test failed: {str(e)}")
            if self.reporter:
                self.reporter.record_test_result(
                    "broker_reconnection", 
                    "FAIL",
                    {"error": str(e)},
                    f"Reconnection error: {str(e)}"
                )
            return False
    
    def _simulate_disconnection(self):
        """Simulate a broker disconnection for testing reconnection."""
        # Clear access token
        self.broker.access_token = None
        
        # Clear any session cookies/state
        self.broker.session = self.broker.session.__class__()
        
        # Wait briefly to ensure disconnection is processed
        time.sleep(1)
    
    @safe_timed_function("broker_connection_test")
    async def test_secure_connection(self) -> bool:
        """
        Test that connection is using secure protocols.
        
        Returns:
            bool: True if connection is secure, False otherwise
        """
        logger.info("Testing secure connection")
        try:
            # Check if the base URL uses HTTPS
            is_https = self.broker.base_url.startswith('https://')
            
            # Check if the WebSocket URL uses WSS
            is_wss = self.broker.ws_url.startswith('wss://')
            
            # Both connection types should be secure
            is_secure = is_https and is_wss
            
            if is_secure:
                logger.info("Secure connection verified (HTTPS and WSS)")
                if self.reporter:
                    self.reporter.record_test_result(
                        "broker_secure_connection", 
                        "PASS",
                        {
                            "secure": True,
                            "https": is_https,
                            "wss": is_wss
                        }
                    )
                return True
            else:
                # Log specifically what's not secure
                if not is_https:
                    logger.error("REST API connection is not secure (not using HTTPS)")
                if not is_wss:
                    logger.error("WebSocket connection is not secure (not using WSS)")
                
                if self.reporter:
                    self.reporter.record_test_result(
                        "broker_secure_connection", 
                        "FAIL",
                        {
                            "secure": False,
                            "https": is_https,
                            "wss": is_wss
                        },
                        "Connection is not fully secure (should use HTTPS and WSS)"
                    )
                return False
        except Exception as e:
            logger.error(f"Secure connection test failed: {str(e)}")
            if self.reporter:
                self.reporter.record_test_result(
                    "broker_secure_connection", 
                    "FAIL",
                    {"error": str(e)},
                    f"Secure connection error: {str(e)}"
                )
            return False
    
    async def run_all_tests(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Run all broker connection tests.
        
        Returns:
            Tuple[bool, Dict[str, Any]]: Success status and detailed results
        """
        logger.info(f"Starting broker connection tests (Run ID: {self.test_run_id})")
        
        test_results = {
            'connection_established': False,
            'authentication_success': False,
            'reconnection_success': False,
            'secure_connection': False,
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'total_duration': None,
            'errors': []
        }
        
        # Setup
        if not await self.setup():
            logger.error("Test setup failed. Aborting tests.")
            test_results['errors'].append("Setup failed")
            test_results['end_time'] = datetime.now().isoformat()
            return False, test_results
        
        # Tests to run in sequence
        tests = [
            self.test_connection_establishment,
            self.test_authentication,
            self.test_reconnection,
            self.test_secure_connection
        ]
        
        # Test names for result tracking
        test_names = [
            'connection_established',
            'authentication_success',
            'reconnection_success',
            'secure_connection'
        ]
        
        # Run tests
        for i, test in enumerate(tests):
            try:
                # Run the test
                test_success = await test()
                test_results[test_names[i]] = test_success
                
                # Stop further tests if a critical test fails
                if i <= 1 and not test_success:  # Connection and authentication are critical
                    logger.error(f"Critical test failed: {test.__name__}. Aborting remaining tests.")
                    break
                    
            except Exception as e:
                logger.error(f"Error during {test.__name__}: {str(e)}")
                test_results['errors'].append(f"Error in {test.__name__}: {str(e)}")
                test_results[test_names[i]] = False
        
        # Record completion time
        test_results['end_time'] = datetime.now().isoformat()
        
        # Calculate duration
        start_time = datetime.fromisoformat(test_results['start_time'])
        end_time = datetime.fromisoformat(test_results['end_time'])
        test_results['total_duration'] = (end_time - start_time).total_seconds()
        
        # Determine overall success
        success = all([
            test_results['connection_established'],
            test_results['authentication_success'],
            test_results['reconnection_success'],
            test_results['secure_connection']
        ])
        
        # Log summary
        passed_tests = sum([
            test_results['connection_established'],
            test_results['authentication_success'],
            test_results['reconnection_success'],
            test_results['secure_connection']
        ])
        
        logger.info(f"Broker connection tests completed: {passed_tests}/4 tests passed")
        if success:
            logger.info("All broker connection tests PASSED")
        else:
            logger.error("Some broker connection tests FAILED")
        
        return success, test_results


# Standalone test function for integration with the main test runner
async def test_broker_connection(perf_tracker=None, reporter=None) -> bool:
    """
    Test connection to the broker API.
    
    This function is designed to be called from the main test runner.
    
    Args:
        perf_tracker: Optional performance tracker
        reporter: Optional test reporter
        
    Returns:
        bool: True if all broker tests passed, False otherwise
    """
    logger.info("Running comprehensive broker connection tests...")
    
    try:
        # Check if perf_tracker is None before using timed_operation
        if perf_tracker:
            with timed_operation(perf_tracker, "broker_connection_suite"):
                # Create and run tester
                tester = BrokerConnectionTester(
                    perf_tracker=perf_tracker,
                    reporter=reporter
                )
                
                success, results = await tester.run_all_tests()
                
                # Record overall result in reporter if provided
                if reporter:
                    status = "PASS" if success else "FAIL"
                    reporter.record_test_result(
                        "broker_connection", 
                        status,
                        results
                    )
                
                return success
        else:
            # Run without performance tracking when perf_tracker is None
            tester = BrokerConnectionTester(
                perf_tracker=None,
                reporter=reporter
            )
            
            success, results = await tester.run_all_tests()
            
            # Record overall result in reporter if provided
            if reporter:
                status = "PASS" if success else "FAIL"
                reporter.record_test_result(
                    "broker_connection", 
                    status,
                    results
                )
            
            return success
            
    except Exception as e:
        error_msg = f"Broker connection testing failed: {str(e)}"
        logger.error(error_msg)
        
        if reporter:
            reporter.record_test_result(
                "broker_connection", 
                "FAIL",
                {"error": str(e)},
                error_msg
            )
        
        return False


# For standalone execution
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the test without performance tracking
    asyncio.run(test_broker_connection(perf_tracker=None, reporter=None))