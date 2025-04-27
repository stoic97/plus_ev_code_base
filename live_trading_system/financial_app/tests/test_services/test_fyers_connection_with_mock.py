import sys
import os
import logging
import time
import json
import argparse
from typing import Dict, Any, List

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("fyers_test")

# Add the project root to Python path to help with imports
current_dir = os.path.dirname(os.path.abspath(__file__))  # /test_services directory
tests_dir = os.path.dirname(current_dir)                  # /tests directory
financial_app_dir = os.path.dirname(tests_dir)            # /financial_app directory
root_dir = os.path.dirname(financial_app_dir)             # /live_trading_system directory

# Add both the root and financial_app directories to the path
sys.path.insert(0, root_dir)
sys.path.insert(0, financial_app_dir)

# Now try to import the FyersClient
try:
    # First try the absolute import
    from financial_app.app.services.fyers_client import FyersClient
    logger.info("Successfully imported FyersClient")
except ImportError as e:
    logger.error(f"Failed to import FyersClient: {e}")
    sys.exit(1)

# Create a mock class that extends FyersClient
class MockFyersClient(FyersClient):
    """Mock FyersClient for testing without relying on the actual Fyers API"""
    
    def connect(self) -> bool:
        """Override to mock a successful connection"""
        logger.info("Using mock connection to Fyers API")
        self.access_token = "mock_access_token_for_testing"
        return True
    
    def get_profile(self) -> Dict[str, Any]:
        """Return mock profile data"""
        logger.info("Returning mock profile data")
        return {
            "name": "Mock User",
            "email": "mock.user@example.com",
            "client_id": self.broker_config["api_id"],
            "account_type": "Trading"
        }
    
    def get_market_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Return mock market data for the requested symbols"""
        logger.info(f"Returning mock market data for symbols: {symbols}")
        mock_data = {}
        
        # Current timestamp for "now" data
        current_time = int(time.time())
        
        for symbol in symbols:
            # Generate some realistic-looking test data based on the symbol
            if "CRUDEOIL" in symbol:
                base_price = 82.45
            elif "GOLD" in symbol:
                base_price = 2320.75
            elif "NIFTY" in symbol:
                base_price = 23450.60
            elif "USDINR" in symbol:
                base_price = 83.25
            else:
                base_price = 100.00
            
            # Add some small random variations to make the data look more realistic
            import random
            variation = random.uniform(-2.0, 2.0)
            price = base_price + variation
            
            mock_data[symbol] = {
                "symbol": symbol,
                "ltp": round(price, 2),  # Last traded price
                "open": round(price * 0.99, 2),
                "high": round(price * 1.02, 2),
                "low": round(price * 0.98, 2),
                "close": round(price * 0.995, 2),
                "volume": int(random.uniform(100000, 2000000)),
                "timestamp": current_time,
                "exchange": "MCX" if "MCX:" in symbol else "NSE",
                "instrument_type": "FUTURES" if "FUTURES" in symbol else "EQUITY"
            }
        
        return mock_data

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test Fyers API connection with optional mocking")
    parser.add_argument("--mock", action="store_true", help="Use mock data instead of real API")
    args = parser.parse_args()
    
    try:
        logger.info("Testing Fyers API connection...")
        
        # Find config file
        config_path = os.path.join(financial_app_dir, "config", "broker_config.yaml")
        if not os.path.exists(config_path):
            logger.error(f"Config file not found at {config_path}")
            return 1
            
        logger.info(f"Using config file at: {config_path}")
        
        # Create the appropriate client based on mode
        if args.mock:
            logger.info("MOCK MODE: Using simulated Fyers API responses")
            client = MockFyersClient(config_path=config_path)
        else:
            logger.info("LIVE MODE: Connecting to actual Fyers API")
            client = FyersClient(config_path=config_path)
        
        # Attempt to connect
        logger.info("Connecting to Fyers API...")
        connected = client.connect()
        
        if not connected:
            logger.error("Failed to connect to Fyers API")
            return 1
            
        logger.info("Successfully connected to Fyers API!")
        
        # Get profile info
        profile = client.get_profile()
        logger.info(f"User profile: {json.dumps(profile, indent=2)}")
        
        # Test fetching market data
        symbols = ["MCX:CRUDEOIL-FUTURES", "NSE:NIFTY-INDEX"]  # Add any symbols you're interested in
        logger.info(f"Fetching market data for symbols: {symbols}")
        
        market_data = client.get_market_data(symbols)
        
        if market_data:
            logger.info("Successfully received market data:")
            logger.info(f"{json.dumps(market_data, indent=2)}")
            return 0
        else:
            logger.error("Failed to retrieve market data")
            return 1
            
    except Exception as e:
        logger.error(f"Error during Fyers API test: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())