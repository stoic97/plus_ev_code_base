# tests/test_consumers/conftest.py

import pytest
import json
from unittest.mock import MagicMock

@pytest.fixture
def mock_kafka_consumer():
    """Create a mock Kafka consumer for testing."""
    mock_consumer = MagicMock()
    
    # Configure the mock to return sample messages
    mock_consumer.poll.return_value = {
        'topic1': [
            MagicMock(
                value=json.dumps({
                    'symbol': 'AAPL', 
                    'price': 150.25, 
                    'timestamp': '2023-01-01T12:00:00Z'
                }).encode(),
                key=b'AAPL',
                offset=0,
                partition=0,
                topic='topic1'
            )
        ]
    }
    
    return mock_consumer

@pytest.fixture
def sample_ohlcv_data():
    """Return sample OHLCV data for testing."""
    return {
        'symbol': 'AAPL',
        'open': 150.25,
        'high': 152.50,
        'low': 149.75,
        'close': 151.80,
        'volume': 10000000,
        'timestamp': '2023-01-01T12:00:00Z'
    }

@pytest.fixture
def sample_trade_data():
    """Return sample trade data for testing."""
    return {
        'symbol': 'AAPL',
        'price': 151.50,
        'quantity': 100,
        'side': 'buy',
        'timestamp': '2023-01-01T12:00:00Z',
        'trade_id': '12345'
    }

@pytest.fixture
def sample_orderbook_data():
    """Return sample orderbook data for testing."""
    return {
        'symbol': 'AAPL',
        'bids': [
            [150.00, 500],
            [149.95, 1000],
        ],
        'asks': [
            [150.10, 800],
            [150.15, 1200],
        ],
        'timestamp': '2023-01-01T12:00:00Z'
    }