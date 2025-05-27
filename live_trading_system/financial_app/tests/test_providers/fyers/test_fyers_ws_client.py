"""
Unit Tests for Fyers WebSocket Client

This module provides comprehensive tests for the WebSocket client,
including market data handling, caching, callbacks, and error handling.
"""
import pytest
import pytest_asyncio
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timezone
from pydantic import SecretStr

from app.providers.fyers.fyers_ws_client import FyersWebSocketClient
from app.providers.fyers.fyers_ws_protocol import (
    WebSocketConfig,
    WebSocketMessageType,
    MarketDataUpdate,
    MarketDepthUpdate,
    WebSocketError
)
from app.providers.fyers.fyers_settings import FyersSettings
from app.providers.fyers.fyers_auth import FyersAuth
from app.providers.base.provider import MarketServiceProtocol
from app.providers.base.cache import MarketAwareStrategicCache


# ===========================================
# FIXTURES
# ===========================================

@pytest.fixture
def mock_settings():
    """Mock Fyers settings."""
    settings = Mock(spec=FyersSettings)
    settings.API_BASE_URL = "https://api.fyers.in/api/v2"
    settings.ACCESS_TOKEN = SecretStr("test_token")
    return settings


@pytest.fixture
def mock_auth():
    """Mock authentication service."""
    auth = AsyncMock(spec=FyersAuth)
    auth.ensure_token.return_value = True
    auth.access_token = "test_token"
    return auth


@pytest.fixture
def mock_market_service():
    """Mock market service."""
    service = Mock(spec=MarketServiceProtocol)
    service.is_market_open.return_value = True
    service.get_market_state.return_value = "open"
    return service


@pytest.fixture
def mock_cache():
    """Mock strategic cache."""
    cache = AsyncMock(spec=MarketAwareStrategicCache)
    cache.get.return_value = None
    cache.set.return_value = None
    cache.get_cache_performance.return_value = {"hit_ratio": 0.8}
    return cache


@pytest.fixture
def mock_connection():
    """Mock WebSocket connection manager."""
    connection = AsyncMock()
    connection.connect = AsyncMock()
    connection.disconnect = AsyncMock()
    connection.subscribe = AsyncMock()
    connection.unsubscribe = AsyncMock()
    connection.get_connection_stats.return_value = {
        "state": "CONNECTED",
        "metrics": {"messages_received": 100}
    }
    return connection


@pytest_asyncio.fixture
async def client(mock_settings, mock_auth, mock_market_service, mock_cache, mock_connection):
    """Create WebSocket client instance."""
    with patch('app.providers.fyers.fyers_ws_client.WebSocketConnectionManager', return_value=mock_connection):
        client = FyersWebSocketClient(
            settings=mock_settings,
            auth=mock_auth,
            market_service=mock_market_service,
            cache=mock_cache
        )
        yield client
        await client.disconnect()


# ===========================================
# INITIALIZATION TESTS
# ===========================================

class TestInitialization:
    """Test client initialization."""
    
    def test_default_initialization(self, mock_settings):
        """Test initialization with default values."""
        client = FyersWebSocketClient(settings=mock_settings)
        
        assert client.settings == mock_settings
        assert isinstance(client.auth, FyersAuth)
        assert client.cache is None
        assert len(client.callbacks) == 4  # market_data, market_depth, connection, error
    
    def test_custom_initialization(self, mock_settings, mock_auth, mock_market_service, mock_cache):
        """Test initialization with custom components."""
        client = FyersWebSocketClient(
            settings=mock_settings,
            auth=mock_auth,
            market_service=mock_market_service,
            cache=mock_cache
        )
        
        assert client.auth == mock_auth
        assert client.market_service == mock_market_service
        assert client.cache == mock_cache


# ===========================================
# CONNECTION TESTS
# ===========================================

class TestConnection:
    """Test connection management."""
    
    @pytest.mark.asyncio
    async def test_connect(self, client, mock_connection):
        """Test connection establishment."""
        await client.connect()
        mock_connection.connect.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_disconnect(self, client, mock_connection):
        """Test connection termination."""
        await client.disconnect()
        mock_connection.disconnect.assert_called_once()


# ===========================================
# SUBSCRIPTION TESTS
# ===========================================

class TestSubscription:
    """Test subscription management."""
    
    @pytest.mark.asyncio
    async def test_subscribe_market_data(self, client, mock_connection):
        """Test market data subscription."""
        symbols = ["NSE:SBIN-EQ", "NSE:RELIANCE-EQ"]
        await client.subscribe_market_data(symbols)
        
        mock_connection.subscribe.assert_called_once_with(
            symbols, WebSocketMessageType.SYMBOL_FEED
        )
    
    @pytest.mark.asyncio
    async def test_subscribe_market_depth(self, client, mock_connection):
        """Test market depth subscription."""
        symbols = ["NSE:SBIN-EQ"]
        await client.subscribe_market_depth(symbols)
        
        mock_connection.subscribe.assert_called_once_with(
            symbols, WebSocketMessageType.DEPTH
        )
    
    @pytest.mark.asyncio
    async def test_unsubscribe(self, client, mock_connection):
        """Test unsubscription."""
        symbols = ["NSE:SBIN-EQ"]
        await client.unsubscribe(symbols)
        
        mock_connection.unsubscribe.assert_called_once_with(symbols)


# ===========================================
# CALLBACK TESTS
# ===========================================

class TestCallbacks:
    """Test callback registration and handling."""
    
    def test_register_market_data_callback(self, client):
        """Test market data callback registration."""
        callback = Mock()
        client.on_market_data(callback)
        
        assert callback in client.callbacks["market_data"]
    
    def test_register_market_depth_callback(self, client):
        """Test market depth callback registration."""
        callback = Mock()
        client.on_market_depth(callback)
        
        assert callback in client.callbacks["market_depth"]
    
    def test_register_connection_callback(self, client):
        """Test connection callback registration."""
        callback = Mock()
        client.on_connection(callback)
        
        assert callback in client.callbacks["connection"]
    
    def test_register_error_callback(self, client):
        """Test error callback registration."""
        callback = Mock()
        client.on_error(callback)
        
        assert callback in client.callbacks["error"]


# ===========================================
# MESSAGE HANDLING TESTS
# ===========================================

class TestMessageHandling:
    """Test message handling and processing."""
    
    @pytest.mark.asyncio
    async def test_handle_market_data(self, client):
        """Test market data message handling."""
        callback = Mock()
        client.on_market_data(callback)
        
        message = {
            "symbol": "NSE:SBIN-EQ",
            "timestamp": int(datetime.now().timestamp() * 1000),
            "ltp": 500.50,
            "volume": 1000
        }
        
        await client._handle_market_data(message)
        
        callback.assert_called_once()
        assert isinstance(callback.call_args[0][0], MarketDataUpdate)
    
    @pytest.mark.asyncio
    async def test_handle_market_depth(self, client):
        """Test market depth message handling."""
        callback = Mock()
        client.on_market_depth(callback)
        
        message = {
            "symbol": "NSE:SBIN-EQ",
            "timestamp": int(datetime.now().timestamp() * 1000),
            "bids": [[500.0, 100]],
            "asks": [[501.0, 150]]
        }
        
        await client._handle_market_depth(message)
        
        callback.assert_called_once()
        assert isinstance(callback.call_args[0][0], MarketDepthUpdate)
    
    @pytest.mark.asyncio
    async def test_handle_error(self, client):
        """Test error message handling."""
        callback = Mock()
        client.on_error(callback)
        
        message = {
            "code": 1001,
            "message": "Test error"
        }
        
        await client._handle_error(message)
        
        callback.assert_called_once()
        assert isinstance(callback.call_args[0][0], WebSocketError)


# ===========================================
# CACHE INTEGRATION TESTS
# ===========================================

class TestCacheIntegration:
    """Test cache integration."""
    
    @pytest.mark.asyncio
    async def test_cache_update(self, client, mock_cache):
        """Test cache update with market data."""
        # Setup mock cache to simulate a cache hit
        mock_cache.get.return_value = {
            "symbol": "NSE:SBIN-EQ",
            "last_price": 500.0,
            "volume": 900
        }
        
        message = {
            "symbol": "NSE:SBIN-EQ",
            "timestamp": int(datetime.now().timestamp() * 1000),
            "ltp": 500.50,
            "volume": 1000
        }
        
        await client._handle_market_data(message)
        
        mock_cache.set.assert_called_once()
        assert client.metrics["cache_hits"] == 1
    
    @pytest.mark.asyncio
    async def test_cache_error(self, client, mock_cache):
        """Test cache error handling."""
        # Setup mock cache to simulate a cache miss
        mock_cache.get.return_value = None
        mock_cache.set.side_effect = Exception("Cache error")
        
        message = {
            "symbol": "NSE:SBIN-EQ",
            "timestamp": int(datetime.now().timestamp() * 1000),
            "ltp": 500.50,
            "volume": 1000
        }
        
        await client._handle_market_data(message)
        
        assert client.metrics["cache_misses"] == 1


# ===========================================
# METRICS AND MONITORING TESTS
# ===========================================

class TestMetrics:
    """Test metrics collection and reporting."""
    
    @pytest.mark.asyncio
    async def test_get_metrics(self, client, mock_connection, mock_cache):
        """Test metrics retrieval."""
        # Setup connection stats
        mock_connection.get_connection_stats.return_value = {
            "state": "CONNECTED",
            "messages_received": 100,
            "messages_sent": 50
        }
        
        # Get metrics
        metrics = await client.get_metrics()
        
        # Verify metrics
        assert "messages_processed" in metrics
        assert "market_data_updates" in metrics
        assert "depth_updates" in metrics
        assert "errors" in metrics
        assert "cache_hits" in metrics
        assert "cache_misses" in metrics
        assert metrics["messages_received"] == 100
        assert metrics["messages_sent"] == 50
        assert metrics["state"] == "CONNECTED"
        
        # Verify cache metrics
        assert "cache_hit_ratio" in metrics
        assert "cache_eviction_rate" in metrics
        assert "cache_entries" in metrics
    
    @pytest.mark.asyncio
    async def test_metrics_update(self, client):
        """Test metrics update during message handling."""
        # Create test message
        message = {
            "symbol": "NSE:SBIN-EQ",
            "last_price": 100.0,
            "volume": 1000,
            "timestamp": datetime.now().timestamp(),
            "open": 99.0,
            "high": 101.0,
            "low": 98.0,
            "close": 100.0,
            "change": 1.0,
            "change_percent": 1.0
        }
        
        # Handle message
        await client._handle_market_data(message)
        
        # Get metrics
        metrics = await client.get_metrics()
        
        # Verify metrics update
        assert metrics["messages_processed"] == 1
        assert metrics["market_data_updates"] == 1


# ===========================================
# ERROR HANDLING TESTS
# ===========================================

class TestErrorHandling:
    """Test error handling scenarios."""
    
    @pytest.mark.asyncio
    async def test_invalid_market_data(self, client):
        """Test handling of invalid market data."""
        message = {
            "symbol": "NSE:SBIN-EQ",
            "ltp": "invalid"  # Invalid price
        }
        
        await client._handle_market_data(message)
        
        assert client.metrics["errors"] > 0
    
    @pytest.mark.asyncio
    async def test_callback_error(self, client):
        """Test handling of callback errors."""
        callback = Mock(side_effect=Exception("Callback error"))
        client.on_market_data(callback)
        
        message = {
            "symbol": "NSE:SBIN-EQ",
            "timestamp": int(datetime.now().timestamp() * 1000),
            "ltp": 500.50,
            "volume": 1000
        }
        
        # Should not raise exception
        await client._handle_market_data(message)


# ===========================================
# CLEANUP AND RESOURCE MANAGEMENT
# ===========================================

@pytest_asyncio.fixture(autouse=True)
async def cleanup_tasks():
    """Cleanup any remaining tasks after each test."""
    yield
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    for task in tasks:
        if not task.done():
            task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 