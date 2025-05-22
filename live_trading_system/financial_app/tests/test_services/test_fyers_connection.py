import sys
import os
import logging

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
    try:
        from financial_app.app.services.fyers_client import FyersClient
        logger.info("Successfully imported FyersClient via financial_app.app.services")
    except ImportError:
        # Then try the relative import if we're already in financial_app
        from app.services.fyers_client import FyersClient
        logger.info("Successfully imported FyersClient via app.services")
except ImportError as e:
    logger.error(f"Failed to import FyersClient: {e}")
    logger.error(f"Current Python path: {sys.path}")
    sys.exit(1)

def main():
    try:
        logger.info("Testing Fyers API connection...")
        
        # Try multiple possible locations for the config file
        possible_config_paths = [
            os.path.join(financial_app_dir, "config", "broker_config.yaml"),
            os.path.join(root_dir, "financial_app", "config", "broker_config.yaml"),
            "financial_app/config/broker_config.yaml",
            "config/broker_config.yaml"
        ]
        
        config_path = None
        for path in possible_config_paths:
            if os.path.exists(path):
                config_path = path
                logger.info(f"Found config file at: {config_path}")
                break
                
        if not config_path:
            logger.error("Could not find broker_config.yaml")
            return 1
        
        # Create the Fyers client
        logger.info(f"Creating FyersClient with config path: {config_path}")
        client = FyersClient(config_path=config_path)
        
        # Attempt to connect (this should handle authentication)
        logger.info("Connecting to Fyers API...")
        connected = client.connect()
        
        if not connected:
            logger.error("Failed to connect to Fyers API")
            return 1
            
        logger.info("Successfully connected to Fyers API!")
        
        # Test fetching market data
        symbols = ["MCX:CRUDEOIL-FUTURES"]  # Adjust as needed for your specific contract
        logger.info(f"Fetching market data for symbols: {symbols}")
        
        market_data = client.get_market_data(symbols)
        
        if market_data:
            logger.info("Successfully received market data:")
            logger.info(f"{market_data}")
            return 0
        else:
            logger.error("Failed to retrieve market data")
            return 1
            
    except Exception as e:
        logger.error(f"Error during Fyers API test: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())

# Example output when running the script with mock data:
# (venv) PS D:\e backup\web development\plusEv\plusEV_code_base\plus_ev_code_base\live_trading_system> python financial_app/tests/test_services/test_fyers_connection.py --mock

#     (venv) PS D:\e backup\web development\plusEv\plusEV_code_base\plus_ev_code_base\live_trading_system> python financial_app/tests/test_services/test_fyers_connection_with_mock.py --mock
# 2025-04-23 15:49:28,614 - fyers_test - INFO - Successfully imported FyersClient
# 2025-04-23 15:49:28,621 - fyers_test - INFO - Testing Fyers API connection...
# 2025-04-23 15:49:28,622 - fyers_test - INFO - Using config file at: D:\e backup\web development\plusEv\plusEV_code_base\plus_ev_code_base\live_trading_system\financial_app\config\broker_config.yaml
# 2025-04-23 15:49:28,622 - fyers_test - INFO - MOCK MODE: Using simulated Fyers API responses
# 2025-04-23 15:49:28,627 - fyers_test - INFO - Connecting to Fyers API...
# 2025-04-23 15:49:28,628 - fyers_test - INFO - Using mock connection to Fyers API
# 2025-04-23 15:49:28,628 - fyers_test - INFO - Successfully connected to Fyers API!
# 2025-04-23 15:49:28,628 - fyers_test - INFO - Returning mock profile data
# 2025-04-23 15:49:28,628 - fyers_test - INFO - User profile: {
#   "name": "Mock User",
#   "email": "mock.user@example.com",
#   "client_id": "GBJMHA44CH-100",
#   "account_type": "Trading"
# }
# 2025-04-23 15:49:28,628 - fyers_test - INFO - Fetching market data for symbols: ['MCX:CRUDEOIL-FUTURES', 'NSE:NIFTY-INDEX']
# 2025-04-23 15:49:28,628 - fyers_test - INFO - Returning mock market data for symbols: ['MCX:CRUDEOIL-FUTURES', 'NSE:NIFTY-INDEX']
# 2025-04-23 15:49:28,628 - fyers_test - INFO - Successfully received market data:
# 2025-04-23 15:49:28,628 - fyers_test - INFO - {
#   "MCX:CRUDEOIL-FUTURES": {
#     "symbol": "MCX:CRUDEOIL-FUTURES",
#     "ltp": 81.52,
#     "open": 80.71,
#     "high": 83.15,
#     "low": 79.89,
#     "close": 81.11,
#     "volume": 350196,
#     "timestamp": 1745403568,
#     "exchange": "MCX",
#     "instrument_type": "FUTURES"
#   },
#   "NSE:NIFTY-INDEX": {
#     "symbol": "NSE:NIFTY-INDEX",
#     "ltp": 23451.32,
#     "open": 23216.81,
#     "high": 23920.35,
#     "low": 22982.3,
#     "close": 23334.07,
#     "volume": 1407117,
#     "timestamp": 1745403568,
#     "exchange": "NSE",
#     "instrument_type": "EQUITY"
#   }
# }