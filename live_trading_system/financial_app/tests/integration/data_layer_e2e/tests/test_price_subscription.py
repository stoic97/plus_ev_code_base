"""
Market Data Subscription Test

This module tests the market data subscription functionality, including:
- Subscribing to test instruments
- Verifying subscription acknowledgment
- Confirming data flow initiation
- Testing subscription management (add/remove instruments)
"""

import asyncio
import logging
import time
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Add project root to path to fix imports
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import the config
from financial_app.tests.integration.data_layer_e2e.e2e_config import (
    TEST_INSTRUMENTS,
    BROKER_CONFIG,
    TEST_REPORT_DIR,
    TEST_ID
)

# Define timeouts
SUBSCRIPTION_TIMEOUT = 20  # Maximum time to wait for subscription acknowledgment
DATA_VALIDATION_TIMEOUT = 60  # Maximum time to wait for data validation
USE_MOCK_SERVICES = os.environ.get("USE_MOCK_SERVICES", "true").lower() in ["true", "1", "yes"]

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(TEST_REPORT_DIR, f"market_data_subscription_{TEST_ID}.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('market_data_subscription_test')

# Try to import real services first, fall back to mock implementations if not available
try:
    if USE_MOCK_SERVICES:
        raise ImportError("Using mock services as specified by environment variable")
        
    # Try to import the real services
    logger.info("Attempting to use real service implementations...")
    
    # Adjust these imports to match your actual project structure
    from financial_app.services.broker_service import BrokerService
    from financial_app.services.auth_service import AuthService
    from financial_app.models.market_data import MarketDataSubscription
    
    logger.info("Successfully imported real service implementations")
    USING_MOCK = False
except ImportError as e:
    logger.warning(f"Could not import real services: {str(e)}")
    logger.info("Falling back to mock implementations")
    USING_MOCK = True
    
    # Mock classes used when real implementations are not available
# If we're using mocks, we already defined MockMarketDataSubscription above
# This is needed in case we're using real services but the real MarketDataSubscription 
# doesn't have the attributes our test expects
if not USING_MOCK:
    # Adapt the real MarketDataSubscription if needed
    original_init = MarketDataSubscription.__init__
    
    def enhanced_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # Ensure these attributes exist for our test code
        if not hasattr(self, 'symbol') and hasattr(self, 'instrument') and isinstance(self.instrument, dict):
            self.symbol = self.instrument.get('symbol')
        if not hasattr(self, 'exchange') and hasattr(self, 'instrument') and isinstance(self.instrument, dict):
            self.exchange = self.instrument.get('exchange')
    
    # Only patch if the real class exists and has different behavior
    try:
        MarketDataSubscription.__init__ = enhanced_init
        logger.info("Enhanced MarketDataSubscription with additional attributes")
    except Exception as e:
        logger.warning(f"Could not enhance MarketDataSubscription: {e}")

class MockBrokerService:
    """Mock broker service for testing."""
    
    def __init__(self):
        """Initialize the broker service with configuration."""
        self.config = BROKER_CONFIG
        self.connected = False
        self.authenticated = False
        logger.info(f"Initialized mock broker service with config: {self.config}")
        
    async def authenticate(self, auth_token):
        """Mock authentication with broker."""
        logger.info(f"[MOCK] Authenticating with broker using token: {auth_token}")
        await asyncio.sleep(0.5)  # Simulate network delay
        self.authenticated = True
        return True
        
    async def subscribe_market_data(self, instrument, callback):
        """Mock subscription to market data."""
        symbol = instrument.get('symbol')
        exchange = instrument.get('exchange')
        logger.info(f"[MOCK] Subscribing to market data for {symbol} on {exchange}")
        
        # Return a mock subscription object
        subscription_id = f"sub_{symbol}_{int(time.time())}"
        subscription = MockMarketDataSubscription(id=subscription_id, instrument=instrument)
        
        # Simulate sending some data after a delay
        asyncio.create_task(self._send_mock_data(instrument, callback))
        
        return subscription
        
    async def _send_mock_data(self, instrument, callback):
        """Send mock market data to callback."""
        import random
        
        symbol = instrument.get('symbol')
        exchange = instrument.get('exchange')
        
        # Wait a bit before sending data
        await asyncio.sleep(1)
        
        # Set base price based on symbol
        if symbol == "AAPL":
            base_price = 190.0
        elif symbol == "MSFT":
            base_price = 420.0
        elif symbol == "BTC/USD":
            base_price = 68000.0
        else:
            base_price = 100.0
        
        # Send several mock data points
        for i in range(5):
            # Create realistic price movement
            price_change = random.uniform(-1.0, 1.0) * (base_price * 0.002)  # 0.2% movement
            current_price = base_price + price_change
            
            data = {
                'symbol': symbol,
                'exchange': exchange,
                'price': round(current_price, 2),
                'timestamp': datetime.now().timestamp(),
                'volume': random.randint(100, 1000),
                'bid': round(current_price - 0.1, 2),
                'ask': round(current_price + 0.1, 2),
                'sequence': i + 1
            }
            
            logger.debug(f"[MOCK] Sending data for {symbol}: {data}")
            await callback(instrument, data)
            await asyncio.sleep(0.5)
            
    async def check_subscription_status(self, subscription_id):
        """Mock check of subscription status."""
        logger.info(f"[MOCK] Checking subscription status for ID: {subscription_id}")
        await asyncio.sleep(0.2)  # Simulate network delay
        # Return a mock status object
        return type('Status', (), {'confirmed': True})
        
    async def unsubscribe_market_data(self, subscription_id):
        """Mock unsubscription from market data."""
        logger.info(f"[MOCK] Unsubscribing from market data for subscription ID: {subscription_id}")
        await asyncio.sleep(0.3)  # Simulate network delay
        return True
        
    async def close(self):
        """Mock close of broker connection."""
        logger.info("[MOCK] Closing broker connection")
        self.connected = False
        self.authenticated = False
        return True

class MockAuthService:
    """Mock authentication service for testing."""
    
    async def login(self, username, password):
        """Mock login function."""
        logger.info(f"[MOCK] Logging in user: {username}")
        return "mock_auth_token_12345"

class MockMarketDataSubscription:
    """Mock market data subscription model."""
    
    def __init__(self, id, instrument):
        self.id = id
        self.instrument = instrument
        self.symbol = instrument.get('symbol')
        self.exchange = instrument.get('exchange')
        
    def __str__(self):
        return f"Subscription({self.id}: {self.symbol} on {self.exchange})"

# Service factory to get the appropriate implementation
def get_broker_service():
    """Get broker service implementation based on availability."""
    if USING_MOCK:
        return MockBrokerService()
    else:
        return BrokerService()

def get_auth_service():
    """Get auth service implementation based on availability."""
    if USING_MOCK:
        return MockAuthService()
    else:
        return AuthService()

# Validation functions
def validate_market_data_format(data):
    """
    Validate the format of market data.
    
    Args:
        data: Market data to validate
        
    Returns:
        bool: True if the data format is valid
    """
    required_fields = ['symbol', 'price', 'timestamp']
    return all(field in data for field in required_fields)

def report_test_result(test_name, success):
    """
    Report test result.
    
    Args:
        test_name: Name of the test
        success: Whether the test passed
    """
    result = "PASSED" if success else "FAILED"
    logger.info(f"TEST RESULT: {test_name} - {result}")
    
    # Write to report file
    report_path = os.path.join(TEST_REPORT_DIR, f"test_results_{TEST_ID}.txt")
    with open(report_path, "a") as f:
        f.write(f"{datetime.now().isoformat()} - {test_name}: {result}\n")

class MarketDataSubscriptionTest:
    """Test class for market data subscription functionality."""
    
    def __init__(self, auth_token: str):
        """
        Initialize the test with authentication token.
        
        Args:
            auth_token: Authentication token from previous authentication step
        """
        self.auth_token = auth_token
        self.broker_service = get_broker_service()
        self.auth_service = get_auth_service()
        self.subscriptions = {}
        self.received_data = {}
        self.subscription_confirmed = {}
        self.data_received_event = asyncio.Event()
        
        logger.info(f"Using {'mock' if USING_MOCK else 'real'} service implementations")
        
    async def setup(self):
        """Set up the test by authenticating with the broker."""
        logger.info("Setting up market data subscription test")
        
        # Authenticate with the broker using the token
        try:
            await self.broker_service.authenticate(self.auth_token)
            logger.info("Successfully authenticated with broker")
        except Exception as e:
            logger.error(f"Failed to authenticate with broker: {str(e)}")
            raise
            
    async def subscribe_to_instruments(self, instruments: List[Dict]) -> bool:
        """
        Subscribe to the provided list of instruments.
        
        Args:
            instruments: List of instrument dictionaries with symbol and exchange
            
        Returns:
            bool: True if all subscriptions were successful
        """
        logger.info(f"Subscribing to instruments: {instruments}")
        
        all_successful = True
        
        for instrument in instruments:
            symbol = instrument.get('symbol')
            try:
                # Create subscription for the instrument
                subscription = await self.broker_service.subscribe_market_data(
                    instrument=instrument,
                    callback=self._data_callback
                )
                
                self.subscriptions[symbol] = subscription
                self.subscription_confirmed[symbol] = False
                
                logger.info(f"Subscription request sent for {symbol}")
            except Exception as e:
                logger.error(f"Failed to subscribe to {symbol}: {str(e)}")
                all_successful = False
                
        return all_successful
    
    async def verify_subscription_acknowledgment(self, timeout: int = SUBSCRIPTION_TIMEOUT) -> Dict[str, bool]:
        """
        Verify that all subscriptions have been acknowledged by the broker.
        
        Args:
            timeout: Maximum time to wait for acknowledgment in seconds
            
        Returns:
            Dict mapping instrument symbols to acknowledgment status
        """
        logger.info(f"Verifying subscription acknowledgments with {timeout}s timeout")
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Check if all subscriptions have been confirmed
            if all(self.subscription_confirmed.values()):
                logger.info("All subscriptions confirmed")
                break
                
            # Check each subscription's status
            for symbol, subscription in self.subscriptions.items():
                if not self.subscription_confirmed[symbol]:
                    # Check if subscription is confirmed
                    status = await self.broker_service.check_subscription_status(subscription.id)
                    
                    if status.confirmed:
                        logger.info(f"Subscription confirmed for {symbol}")
                        self.subscription_confirmed[symbol] = True
            
            # Wait a bit before checking again
            await asyncio.sleep(1)
        
        # Return final acknowledgment status
        return self.subscription_confirmed
    
    async def confirm_data_flow(self, timeout: int = DATA_VALIDATION_TIMEOUT) -> Dict[str, bool]:
        """
        Confirm that data is flowing for each instrument subscription.
        
        Args:
            timeout: Maximum time to wait for data in seconds
            
        Returns:
            Dict mapping instrument symbols to data flow status
        """
        logger.info(f"Confirming data flow with {timeout}s timeout")
        
        data_received = {symbol: False for symbol in self.subscriptions}
        
        # Wait for data to be received
        try:
            # Wait for the data_received_event to be set
            await asyncio.wait_for(self.data_received_event.wait(), timeout)
            
            # Check which instruments have received data
            for symbol, data_list in self.received_data.items():
                if data_list:
                    data_received[symbol] = True
                    logger.info(f"Data received for {symbol}: {len(data_list)} messages")
                else:
                    logger.warning(f"No data received for {symbol}")
        except asyncio.TimeoutError:
            logger.warning(f"Timeout waiting for market data after {timeout} seconds")
        
        return data_received
    
    async def test_subscription_management(self) -> bool:
        """
        Test adding and removing subscriptions dynamically.
        
        Returns:
            bool: True if subscription management tests passed
        """
        logger.info("Testing subscription management (add/remove)")
        
        try:
            # Get one of the instruments to unsubscribe from
            if not self.subscriptions:
                logger.error("No active subscriptions to test with")
                return False
                
            test_symbol = list(self.subscriptions.keys())[0]
            test_subscription = self.subscriptions[test_symbol]
            
            # Find the corresponding instrument
            test_instrument = next((i for i in TEST_INSTRUMENTS if i['symbol'] == test_symbol), None)
            if not test_instrument:
                logger.error(f"Could not find instrument data for {test_symbol}")
                return False
            
            # Unsubscribe from the instrument
            logger.info(f"Unsubscribing from {test_symbol}")
            await self.broker_service.unsubscribe_market_data(test_subscription.id)
            
            # Wait a bit to ensure the unsubscribe takes effect
            await asyncio.sleep(2)
            
            # Confirm no more data is flowing for this instrument
            # Reset the received data for this instrument
            original_data_count = len(self.received_data.get(test_symbol, []))
            self.received_data[test_symbol] = []
            
            # Wait a bit to see if any new data arrives
            await asyncio.sleep(5)
            
            unsubscribe_successful = len(self.received_data[test_symbol]) == 0
            
            if unsubscribe_successful:
                logger.info(f"Successfully unsubscribed from {test_symbol}")
            else:
                logger.warning(f"Data still received after unsubscribing from {test_symbol}")
            
            # Resubscribe to the instrument
            logger.info(f"Resubscribing to {test_symbol}")
            subscription = await self.broker_service.subscribe_market_data(
                instrument=test_instrument,
                callback=self._data_callback
            )
            
            self.subscriptions[test_symbol] = subscription
            self.subscription_confirmed[test_symbol] = False
            
            # Verify resubscription acknowledgment
            await asyncio.sleep(2)
            status = await self.broker_service.check_subscription_status(subscription.id)
            resubscribe_successful = status.confirmed
            
            if resubscribe_successful:
                logger.info(f"Successfully resubscribed to {test_symbol}")
            else:
                logger.warning(f"Failed to resubscribe to {test_symbol}")
            
            # Check that data flows again after resubscription
            # Reset the event
            self.data_received_event.clear()
            
            # Wait for new data
            await asyncio.wait_for(self.data_received_event.wait(), 10)
            
            # Check if we received new data
            data_flowing_again = len(self.received_data[test_symbol]) > 0
            
            if data_flowing_again:
                logger.info(f"Data flowing again for {test_symbol} after resubscription")
            else:
                logger.warning(f"No data received after resubscription to {test_symbol}")
            
            return unsubscribe_successful and resubscribe_successful and data_flowing_again
        except Exception as e:
            logger.error(f"Error in subscription management test: {str(e)}")
            return False
    
    async def _data_callback(self, instrument, data):
        """
        Callback function for receiving market data.
        
        Args:
            instrument: Instrument dictionary with symbol and exchange
            data: Market data for the instrument
        """
        symbol = instrument.get('symbol')
        
        # Initialize the list for this instrument if it doesn't exist
        if symbol not in self.received_data:
            self.received_data[symbol] = []
            
        # Add the data to the list
        self.received_data[symbol].append(data)
        
        # Validate the data format
        is_valid = validate_market_data_format(data)
        
        if not is_valid:
            logger.warning(f"Received invalid market data format for {symbol}: {data}")
        
        # Set the event to indicate data was received
        self.data_received_event.set()
    
    async def teardown(self):
        """Clean up resources after the test."""
        logger.info("Tearing down market data subscription test")
        
        # Unsubscribe from all instruments
        for symbol, subscription in self.subscriptions.items():
            try:
                await self.broker_service.unsubscribe_market_data(subscription.id)
                logger.info(f"Unsubscribed from {symbol}")
            except Exception as e:
                logger.warning(f"Failed to unsubscribe from {symbol}: {str(e)}")
        
        # Close the broker connection
        await self.broker_service.close()
        logger.info("Broker connection closed")

async def run_test(auth_token: str) -> bool:
    """
    Run the market data subscription test.
    
    Args:
        auth_token: Authentication token from previous authentication step
        
    Returns:
        bool: True if test passed, False otherwise
    """
    logger.info(f"Starting market data subscription test with auth token: {auth_token}")
    
    test = MarketDataSubscriptionTest(auth_token)
    
    try:
        # Setup the test
        await test.setup()
        
        # Subscribe to test instruments
        subscription_success = await test.subscribe_to_instruments(TEST_INSTRUMENTS)
        if not subscription_success:
            logger.error("Failed to subscribe to test instruments")
            return False
            
        # Verify subscription acknowledgment
        acknowledgments = await test.verify_subscription_acknowledgment()
        all_acknowledged = all(acknowledgments.values())
        
        if not all_acknowledged:
            unconfirmed = [symbol for symbol, confirmed in acknowledgments.items() if not confirmed]
            logger.error(f"Not all subscriptions were acknowledged: {unconfirmed}")
            return False
            
        # Confirm data flow
        data_flow = await test.confirm_data_flow()
        all_data_flowing = all(data_flow.values())
        
        if not all_data_flowing:
            no_data = [symbol for symbol, flowing in data_flow.items() if not flowing]
            logger.error(f"Data not flowing for all instruments: {no_data}")
            return False
            
        # Test subscription management
        management_success = await test.test_subscription_management()
        
        if not management_success:
            logger.error("Subscription management test failed")
            return False
            
        # All tests passed
        logger.info("Market data subscription test passed successfully")
        return True
    except Exception as e:
        logger.error(f"Market data subscription test failed with exception: {str(e)}")
        return False
    finally:
        # Always tear down
        await test.teardown()

if __name__ == "__main__":
    # This allows the script to be run directly for testing
    # Get authentication token from previous test or use environment variable
    auth_token = os.environ.get("E2E_AUTH_TOKEN", "sample_auth_token")
    
    logger.info(f"=== STARTING MARKET DATA SUBSCRIPTION TEST ===")
    logger.info(f"Test ID: {TEST_ID}")
    logger.info(f"Using test instruments: {TEST_INSTRUMENTS}")
    logger.info(f"Service mode: {'MOCK' if USING_MOCK else 'REAL'}")
    
    # Run the test
    success = asyncio.run(run_test(auth_token))
    
    # Report the result
    report_test_result("Market Data Subscription Test", success)
    
    # Log completion
    logger.info(f"=== MARKET DATA SUBSCRIPTION TEST COMPLETED: {'PASSED' if success else 'FAILED'} ===")
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)