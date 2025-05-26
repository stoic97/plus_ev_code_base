"""
Unit Tests for Fyers WebSocket Protocol

This module provides comprehensive tests for the WebSocket protocol definitions,
including message types, data structures, and validation.
"""
import pytest
from datetime import datetime, timezone
from typing import Dict, Any

from app.providers.fyers.fyers_ws_protocol import (
    WebSocketMessageType,
    WebSocketConnectionState,
    WebSocketConfig,
    SubscriptionRequest,
    MarketDataUpdate,
    MarketDepthUpdate,
    WebSocketError
)


class TestWebSocketMessageType:
    """Test WebSocket message type enumeration."""
    
    def test_message_types(self):
        """Test all message type values."""
        assert WebSocketMessageType.SYMBOL_FEED.value == "sf"
        assert WebSocketMessageType.INDEX_FEED.value == "if"
        assert WebSocketMessageType.DEPTH.value == "dp"
        assert WebSocketMessageType.CONNECTION.value == "cn"
        assert WebSocketMessageType.SUBSCRIBE.value == "sub"
        assert WebSocketMessageType.ERROR.value == "error"
    
    def test_message_type_comparison(self):
        """Test message type comparison."""
        assert WebSocketMessageType.SYMBOL_FEED == WebSocketMessageType.SYMBOL_FEED
        assert WebSocketMessageType.SYMBOL_FEED != WebSocketMessageType.DEPTH
    
    def test_message_type_from_string(self):
        """Test creating message type from string."""
        assert WebSocketMessageType("sf") == WebSocketMessageType.SYMBOL_FEED
        assert WebSocketMessageType("dp") == WebSocketMessageType.DEPTH
        
        with pytest.raises(ValueError):
            WebSocketMessageType("invalid")


class TestWebSocketConnectionState:
    """Test WebSocket connection state enumeration."""
    
    def test_connection_states(self):
        """Test all connection states."""
        states = [
            WebSocketConnectionState.DISCONNECTED,
            WebSocketConnectionState.CONNECTING,
            WebSocketConnectionState.AUTHENTICATING,
            WebSocketConnectionState.CONNECTED,
            WebSocketConnectionState.RECONNECTING,
            WebSocketConnectionState.ERROR
        ]
        
        # Ensure all states have unique values
        state_values = [state.value for state in states]
        assert len(state_values) == len(set(state_values))
    
    def test_state_comparison(self):
        """Test connection state comparison."""
        assert WebSocketConnectionState.CONNECTED == WebSocketConnectionState.CONNECTED
        assert WebSocketConnectionState.CONNECTED != WebSocketConnectionState.DISCONNECTED
        assert WebSocketConnectionState.CONNECTED.value > WebSocketConnectionState.CONNECTING.value


class TestWebSocketConfig:
    """Test WebSocket configuration dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = WebSocketConfig()
        
        assert config.base_url == "wss://api.fyers.in/socket/v2/"
        assert config.max_retries == 50
        assert config.retry_delay == 1.0
        assert config.max_retry_delay == 30.0
        assert config.heartbeat_interval == 30.0
        assert config.connection_timeout == 10.0
        assert config.max_symbols_per_connection == 5000
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = WebSocketConfig(
            base_url="wss://test.example.com",
            max_retries=10,
            retry_delay=2.0,
            max_retry_delay=20.0,
            heartbeat_interval=15.0,
            connection_timeout=5.0,
            max_symbols_per_connection=1000
        )
        
        assert config.base_url == "wss://test.example.com"
        assert config.max_retries == 10
        assert config.retry_delay == 2.0
        assert config.max_retry_delay == 20.0
        assert config.heartbeat_interval == 15.0
        assert config.connection_timeout == 5.0
        assert config.max_symbols_per_connection == 1000


class TestSubscriptionRequest:
    """Test subscription request message."""
    
    def test_default_subscription(self):
        """Test default subscription request."""
        symbols = ["NSE:SBIN-EQ", "NSE:RELIANCE-EQ"]
        request = SubscriptionRequest(symbols=symbols)
        
        assert request.symbols == symbols
        assert request.data_type == WebSocketMessageType.SYMBOL_FEED
    
    def test_custom_subscription(self):
        """Test custom subscription request."""
        symbols = ["NSE:SBIN-EQ"]
        request = SubscriptionRequest(
            symbols=symbols,
            data_type=WebSocketMessageType.DEPTH
        )
        
        assert request.symbols == symbols
        assert request.data_type == WebSocketMessageType.DEPTH
    
    def test_to_dict(self):
        """Test conversion to dictionary format."""
        symbols = ["NSE:SBIN-EQ", "NSE:RELIANCE-EQ"]
        request = SubscriptionRequest(symbols=symbols)
        
        result = request.to_dict()
        assert result == {
            "type": "sf",
            "symbols": symbols
        }


class TestMarketDataUpdate:
    """Test market data update message."""
    
    def test_from_valid_message(self):
        """Test creating update from valid message."""
        message = {
            "symbol": "NSE:SBIN-EQ",
            "timestamp": 1640995200000,  # 2022-01-01 00:00:00
            "ltp": 500.50,
            "volume": 1000,
            "open_price": 495.0,
            "high_price": 505.0,
            "low_price": 490.0,
            "close_price": 500.50,
            "change": 5.50,
            "change_percentage": 1.1
        }
        
        update = MarketDataUpdate.from_message(message)
        
        assert update.symbol == "NSE:SBIN-EQ"
        assert update.timestamp == datetime.fromtimestamp(1640995200)
        assert update.last_price == 500.50
        assert update.volume == 1000
        assert update.open == 495.0
        assert update.high == 505.0
        assert update.low == 490.0
        assert update.close == 500.50
        assert update.change == 5.50
        assert update.change_percent == 1.1
    
    def test_from_partial_message(self):
        """Test creating update from partial message."""
        message = {
            "symbol": "NSE:SBIN-EQ",
            "ltp": 500.50
        }
        
        update = MarketDataUpdate.from_message(message)
        
        assert update.symbol == "NSE:SBIN-EQ"
        assert update.last_price == 500.50
        assert update.volume == 0  # Default value
        assert update.open == 0.0  # Default value
    
    def test_from_invalid_message(self):
        """Test creating update from invalid message."""
        message = {
            "symbol": "NSE:SBIN-EQ",
            "ltp": "invalid"  # Invalid price
        }
        
        with pytest.raises(ValueError):
            MarketDataUpdate.from_message(message)


class TestMarketDepthUpdate:
    """Test market depth update message."""
    
    def test_from_valid_message(self):
        """Test creating depth update from valid message."""
        message = {
            "symbol": "NSE:SBIN-EQ",
            "timestamp": 1640995200000,
            "bids": [[500.0, 100], [499.0, 200]],
            "asks": [[501.0, 150], [502.0, 250]]
        }
        
        update = MarketDepthUpdate.from_message(message)
        
        assert update.symbol == "NSE:SBIN-EQ"
        assert update.timestamp == datetime.fromtimestamp(1640995200)
        assert len(update.bids) == 2
        assert len(update.asks) == 2
        assert update.bids[0] == {"price": 500.0, "quantity": 100}
        assert update.asks[0] == {"price": 501.0, "quantity": 150}
    
    def test_from_empty_message(self):
        """Test creating depth update from empty message."""
        message = {
            "symbol": "NSE:SBIN-EQ"
        }
        
        update = MarketDepthUpdate.from_message(message)
        
        assert update.symbol == "NSE:SBIN-EQ"
        assert len(update.bids) == 0
        assert len(update.asks) == 0
    
    def test_from_invalid_message(self):
        """Test creating depth update from invalid message."""
        message = {
            "symbol": "NSE:SBIN-EQ",
            "bids": "invalid"  # Invalid bids format
        }
        
        with pytest.raises(TypeError):
            MarketDepthUpdate.from_message(message)


class TestWebSocketError:
    """Test WebSocket error message."""
    
    def test_create_error(self):
        """Test creating error message."""
        error = WebSocketError(code=1001, message="Test error")
        
        assert error.code == 1001
        assert error.message == "Test error"
        assert isinstance(error.timestamp, datetime)
    
    def test_from_valid_message(self):
        """Test creating error from valid message."""
        message = {
            "code": 1001,
            "message": "Test error",
            "timestamp": 1640995200
        }
        
        error = WebSocketError.from_message(message)
        
        assert error.code == 1001
        assert error.message == "Test error"
        assert error.timestamp == datetime.fromtimestamp(1640995200)
    
    def test_from_partial_message(self):
        """Test creating error from partial message."""
        message = {
            "code": 1001
        }
        
        error = WebSocketError.from_message(message)
        
        assert error.code == 1001
        assert error.message == "Unknown error"
        assert isinstance(error.timestamp, datetime)
    
    def test_custom_timestamp(self):
        """Test error with custom timestamp."""
        timestamp = datetime.now(timezone.utc)
        error = WebSocketError(code=1001, message="Test error", timestamp=timestamp)
        
        assert error.timestamp == timestamp


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 