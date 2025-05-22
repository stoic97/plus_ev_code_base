"""
Unit tests for the base provider module.

These tests verify the behavior of the abstract base provider class,
including rate limiting, error handling, and retry logic.
"""

import pytest
import asyncio
import sys
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime, timedelta

# Import the error classes from the provider module directly 
# (they'll be moved to a dedicated errors module later)
from app.providers.base.provider import (
    BaseProvider, RateLimiter, ConnectionState, SubscriptionType, rate_limited,
    ProviderError, ConnectionError, RateLimitError, AuthenticationError, DataNotFoundError
)
from app.providers.config.provider_settings import BaseProviderSettings

# Create a concrete test implementation of BaseProvider for testing
class TestProvider(BaseProvider):
    """Concrete implementation of BaseProvider for testing."""
    
    def __init__(self, settings=None):
        settings = settings or BaseProviderSettings()
        super().__init__(settings, "test_provider")
        self.test_rate_limiter = RateLimiter(calls_per_second=10)
        self.connect_called = False
        self.disconnect_called = False
    
    async def connect(self):
        self.connect_called = True
        self.connection_state = ConnectionState.CONNECTED
    
    async def disconnect(self):
        self.disconnect_called = True
        self.connection_state = ConnectionState.DISCONNECTED
    
    async def get_ohlcv(self, symbol, interval, start_time=None, end_time=None, limit=None):
        return [{"timestamp": datetime.now(), "open": 100, "high": 101, "low": 99, "close": 100.5, "volume": 1000}]
    
    async def get_orderbook(self, symbol, depth=None):
        return {"bids": [[100, 10], [99, 20]], "asks": [[101, 15], [102, 25]]}
    
    async def get_trades(self, symbol, start_time=None, end_time=None, limit=None):
        return [{"id": "123", "timestamp": datetime.now(), "price": 100, "volume": 10, "side": "buy"}]
    
    async def get_quote(self, symbol):
        return {"bid": 99.5, "ask": 100.5, "last": 100, "volume": 5000}
    
    async def subscribe_to_trades(self, symbol, callback):
        self.active_subscriptions.add((symbol, SubscriptionType.TRADES))
    
    async def subscribe_to_orderbook(self, symbol, callback, depth=None):
        self.active_subscriptions.add((symbol, SubscriptionType.ORDERBOOK))
    
    async def subscribe_to_quotes(self, symbol, callback):
        self.active_subscriptions.add((symbol, SubscriptionType.QUOTES))
    
    async def unsubscribe(self, symbol, subscription_type):
        self.active_subscriptions.discard((symbol, subscription_type))
    
    async def health_check(self):
        return {"status": "ok", "connected": True}
    
    @rate_limited("test_rate_limiter")
    async def rate_limited_method(self):
        return "success"


class TestRateLimiter:
    """Tests for the RateLimiter class."""
    
    @pytest.mark.asyncio
    async def test_rate_limiter_allows_within_limit(self):
        """Test that RateLimiter allows requests within the limit."""
        limiter = RateLimiter(calls_per_second=10)
        
        # Should not block
        start_time = asyncio.get_event_loop().time()
        await limiter.acquire()
        elapsed = asyncio.get_event_loop().time() - start_time
        
        assert elapsed < 0.1, "Rate limiter should not block within limits"
    
    @pytest.mark.asyncio
    async def test_rate_limiter_delays_over_limit(self):
        """Test that RateLimiter delays requests over the limit."""
        limiter = RateLimiter(calls_per_second=2, burst_limit=2)
        
        # Use all tokens
        await limiter.acquire(2)
        
        # Should delay
        start_time = asyncio.get_event_loop().time()
        await limiter.acquire()
        elapsed = asyncio.get_event_loop().time() - start_time
        
        assert elapsed >= 0.45, "Rate limiter should delay when over limit"
    
    @pytest.mark.asyncio
    async def test_rate_limiter_refills_over_time(self):
        """Test that RateLimiter refills tokens over time."""
        limiter = RateLimiter(calls_per_second=10, burst_limit=10)
        
        # Use all tokens
        await limiter.acquire(10)
        assert limiter.tokens == 0
        
        # Wait for refill
        await asyncio.sleep(0.2)
        limiter._refill()
        
        # Should have refilled some tokens
        assert limiter.tokens > 0, "Tokens should refill over time"


class TestBaseProvider:
    """Tests for the BaseProvider class."""
    
    @pytest.fixture
    def provider(self):
        """Create a test provider instance."""
        return TestProvider()
    
    @pytest.mark.asyncio
    async def test_connect_disconnect(self, provider):
        """Test connect and disconnect methods."""
        await provider.connect()
        assert provider.connect_called
        assert provider.connection_state == ConnectionState.CONNECTED
        
        await provider.disconnect()
        assert provider.disconnect_called
        assert provider.connection_state == ConnectionState.DISCONNECTED
    
    @pytest.mark.asyncio
    async def test_subscription_management(self, provider):
        """Test subscription management."""
        symbol = "AAPL"
        callback = AsyncMock()
        
        # Subscribe to different data types
        await provider.subscribe_to_trades(symbol, callback)
        await provider.subscribe_to_orderbook(symbol, callback)
        await provider.subscribe_to_quotes(symbol, callback)
        
        # Check subscriptions
        assert (symbol, SubscriptionType.TRADES) in provider.active_subscriptions
        assert (symbol, SubscriptionType.ORDERBOOK) in provider.active_subscriptions
        assert (symbol, SubscriptionType.QUOTES) in provider.active_subscriptions
        
        # Unsubscribe
        await provider.unsubscribe(symbol, SubscriptionType.TRADES)
        assert (symbol, SubscriptionType.TRADES) not in provider.active_subscriptions
        
        # Other subscriptions should still be active
        assert (symbol, SubscriptionType.ORDERBOOK) in provider.active_subscriptions
        assert (symbol, SubscriptionType.QUOTES) in provider.active_subscriptions
    
    @pytest.mark.asyncio
    async def test_rate_limited_decorator(self, provider):
        """Test rate_limited decorator."""
        # Mock the acquire method
        with patch.object(provider.test_rate_limiter, 'acquire', new_callable=AsyncMock) as mock_acquire:
            result = await provider.rate_limited_method()
            
            # Should have called acquire
            mock_acquire.assert_called_once()
            assert result == "success"
    
    @pytest.mark.asyncio
    async def test_with_error_handling(self, provider):
        """Test with_error_handling method."""
        # Test successful execution
        async def success_func():
            return "success"
        
        result = await provider.with_error_handling(success_func)
        assert result == "success"
        
        # Test provider error
        async def provider_error_func():
            raise ProviderError("Test error")
        
        with pytest.raises(ProviderError):
            await provider.with_error_handling(provider_error_func)
        
        # Test timeout error
        async def timeout_error_func():
            raise asyncio.TimeoutError("Timeout")
        
        with pytest.raises(ConnectionError):
            await provider.with_error_handling(timeout_error_func)
        
        # Test unexpected error
        async def unexpected_error_func():
            raise ValueError("Unexpected")
        
        with pytest.raises(ProviderError):
            await provider.with_error_handling(unexpected_error_func)
    
    @pytest.mark.asyncio
    async def test_with_retries(self, provider):
        """Test with_retries method."""
        # Mock function that succeeds on the second try
        mock_func = AsyncMock()
        mock_func.side_effect = [ConnectionError("First attempt failed"), "success"]
        
        result = await provider.with_retries(mock_func, max_retries=2)
        assert result == "success"
        assert mock_func.call_count == 2
        
        # Mock function that always fails
        mock_func = AsyncMock(side_effect=ConnectionError("Always fails"))
        
        with pytest.raises(ConnectionError):
            await provider.with_retries(mock_func, max_retries=2)
        assert mock_func.call_count == 3  # Initial attempt + 2 retries
        
        # Test non-retryable error
        mock_func = AsyncMock(side_effect=ValueError("Non-retryable"))
        
        with pytest.raises(ValueError):
            await provider.with_retries(mock_func, max_retries=2)
        assert mock_func.call_count == 1  # Should not retry
    
    @pytest.mark.asyncio
    async def test_provider_methods(self, provider):
        """Test basic provider methods."""
        # Test get_ohlcv
        ohlcv = await provider.get_ohlcv("AAPL", "1d")
        assert isinstance(ohlcv, list)
        assert "open" in ohlcv[0]
        
        # Test get_orderbook
        orderbook = await provider.get_orderbook("AAPL")
        assert "bids" in orderbook
        assert "asks" in orderbook
        
        # Test get_trades
        trades = await provider.get_trades("AAPL")
        assert isinstance(trades, list)
        assert "price" in trades[0]
        
        # Test get_quote
        quote = await provider.get_quote("AAPL")
        assert "bid" in quote
        assert "ask" in quote
        
        # Test health_check
        health = await provider.health_check()
        assert health["status"] == "ok"
        assert health["connected"] is True