"""
End-to-end test configuration for market data flow testing.
Contains all parameters and settings needed for the test suite.
"""

import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# General test settings
TEST_DURATION_MINUTES = 30
LOG_LEVEL = "INFO"
TEST_REPORT_DIR = os.path.join(os.path.dirname(__file__), "reports")
TEST_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

# print(f"DEBUG: USE_MOCK_SERVICES from env = {os.environ.get('USE_MOCK_SERVICES', 'NOT SET')}")
# Modify your USE_MOCK_SERVICES variable to work with the new hybrid approach
# Convert "true"/"false" string to the appropriate mock mode
USE_MOCK_SERVICES = os.environ.get("USE_MOCK_SERVICES", "false").lower()
if USE_MOCK_SERVICES == "true":
    MOCK_MODE = "always"
elif USE_MOCK_SERVICES == "false":
    MOCK_MODE = "never" 
else:
    MOCK_MODE = "auto"  # Default fallback to auto mode

# Database connection settings
DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "port": int(os.environ.get("DB_PORT", "5432")),
    "database": os.environ.get("DB_NAME", "trading_db"),
    "user": os.environ.get("DB_USER", "test_user"),
    "password": os.environ.get("DB_PASSWORD", "test_password"),
}

# Kafka configuration
KAFKA_CONFIG = {
    "bootstrap_servers": os.environ.get("KAFKA_SERVERS", "localhost:9092"),
    "group_id": "e2e_test_consumer",
    "auto_offset_reset": "earliest",
    "market_data_topic": "market_data",
    "error_topic": "market_data_errors",
}

# API settings
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000/api")
API_TIMEOUT = 30  # seconds

# Broker API settings
BROKER_CONFIG = {
    "api_key": os.environ.get("BROKER_API_KEY", "test_api_key"),
    "api_secret": os.environ.get("BROKER_API_SECRET", "test_api_secret"),
    "base_url": os.environ.get("BROKER_BASE_URL", "https://api.testbroker.com"),
    "timeout": 30,  # seconds
}

# Test user credentials
TEST_USER = {
    "username": os.environ.get("TEST_USER", "test_user@example.com"),
    "password": os.environ.get("TEST_PASSWORD", "test_password"),
}

# Test instruments to subscribe to
TEST_INSTRUMENTS = [
    {"symbol": "AAPL", "exchange": "NASDAQ"},
    {"symbol": "MSFT", "exchange": "NASDAQ"},
    {"symbol": "BTC/USD", "exchange": "COINBASE"},
]

# Performance thresholds
PERFORMANCE_THRESHOLDS = {
    "max_message_latency_ms": 500,
    "max_db_write_latency_ms": 200,
    "max_api_response_time_ms": 300,
    "max_consumer_lag": 100,
}

# Test retry settings
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5

# Date range for historical data testing
END_DATE = datetime.now()
START_DATE = END_DATE - timedelta(days=1)

# Create report directory if it doesn't exist
os.makedirs(TEST_REPORT_DIR, exist_ok=True)