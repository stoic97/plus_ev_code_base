"""
Abstract base provider for market data.

This module defines the base interface for all market data providers,
establishing a consistent API regardless of the underlying data source.
"""

import logging
import time
import asyncio
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, TypeVar, Generic, Callable, Protocol, runtime_checkable
from functools import wraps

# Import provider settings
from app.providers.config.provider_settings import BaseProviderSettings

@runtime_checkable
class MarketServiceProtocol(Protocol):
    """Protocol defining the interface for market services."""
    
    def is_market_open(self) -> bool:
        """Check if the market is currently open."""
        ...
    
    def get_market_state(self) -> str:
        """Get the current market state."""
        ...
    
    def get_trading_hours(self) -> Dict[str, Any]:
        """Get the trading hours information."""
        ...
    
    def get_market_holidays(self) -> List[datetime]:
        """Get the list of market holidays."""
        ...

class NoOpMarketService:
    """A no-operation implementation of MarketServiceProtocol."""
    
    def is_market_open(self) -> bool:
        """Always return True."""
        return True
    
    def get_market_state(self) -> str:
        """Always return 'open'."""
        return "open"
    
    def get_trading_hours(self) -> Dict[str, Any]:
        """Return dummy trading hours."""
        return {
            "open": "09:15",
            "close": "15:30",
            "pre_market": "09:00",
            "post_market": "15:45"
        }
    
    def get_market_holidays(self) -> List[datetime]:
        """Return empty list of holidays."""
        return []

# Error classes - these will be moved to a dedicated module later
class ProviderError(Exception):
    """Base class for provider errors."""
    pass

class ConnectionError(ProviderError):
    """Connection error."""
    pass

class AuthenticationError(ProviderError):
    """Authentication error."""
    pass

class RateLimitError(ProviderError):
    """Rate limit error."""
    pass

class DataNotFoundError(ProviderError):
    """Data not found error."""
    pass

# Set up logging
logger = logging.getLogger(__name__)

# Type variables
T = TypeVar('T')


class SubscriptionType(str, Enum):
    """Types of market data subscriptions."""
    QUOTES = "quotes"
    TRADES = "trades"
    ORDERBOOK = "orderbook"
    OHLC = "ohlc"


class ConnectionState(str, Enum):
    """Connection states for providers."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


class RateLimiter:
    """
    Rate limiter for API requests.
    
    Uses a token bucket algorithm to enforce rate limits.
    """
    
    def __init__(self, calls_per_second: float, burst_limit: Optional[int] = None):
        """
        Initialize a rate limiter.
        
        Args:
            calls_per_second: Number of calls allowed per second
            burst_limit: Maximum burst size (defaults to calls_per_second * 2)
        """
        self.calls_per_second = calls_per_second
        self.burst_limit = burst_limit or int(calls_per_second * 2)
        self.tokens = self.burst_limit
        self.last_refill_time = time.monotonic()
        self._lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1) -> None:
        """
        Acquire tokens from the bucket, waiting if necessary.
        
        Args:
            tokens: Number of tokens to acquire
        """
        async with self._lock:
            await self._wait_for_tokens(tokens)
    
    async def _wait_for_tokens(self, tokens: int) -> None:
        """
        Wait until the requested number of tokens are available.
        
        Args:
            tokens: Number of tokens to wait for
        """
        self._refill()
        
        # If we don't have enough tokens, calculate wait time
        if tokens > self.tokens:
            # Calculate time to wait for enough tokens
            tokens_needed = tokens - self.tokens
            wait_time = tokens_needed / self.calls_per_second
            logger.debug(f"Rate limit reached, waiting {wait_time:.2f}s for {tokens_needed} tokens")
            
            # Wait and then refill
            await asyncio.sleep(wait_time)
            self._refill()
        
        # Consume tokens
        self.tokens -= tokens
    
    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_refill_time
        
        # Add new tokens based on elapsed time
        new_tokens = elapsed * self.calls_per_second
        if new_tokens > 0:
            self.tokens = min(self.burst_limit, self.tokens + new_tokens)
            self.last_refill_time = now


def rate_limited(limiter_attr: str):
    """
    Decorator for rate limiting provider methods.
    
    Args:
        limiter_attr: Attribute name of the rate limiter to use
        
    Returns:
        Decorated method
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            # Get the rate limiter instance
            limiter = getattr(self, limiter_attr, None)
            if limiter is None:
                logger.warning(f"Rate limiter '{limiter_attr}' not found, proceeding without rate limiting")
                return await func(self, *args, **kwargs)
            
            # Acquire a token from the rate limiter
            await limiter.acquire()
            
            # Call the original function
            return await func(self, *args, **kwargs)
        return wrapper
    return decorator


class BaseProvider(ABC):
    """
    Abstract base class for market data providers.
    
    Defines the interface that all provider implementations must follow.
    """
    
    def __init__(self, settings: BaseProviderSettings, provider_name: str):
        """
        Initialize a base provider.
        
        Args:
            settings: Provider-specific settings
            provider_name: Name of the provider for logging/metrics
        """
        self.settings = settings
        self.provider_name = provider_name
        self.connection_state = ConnectionState.DISCONNECTED
        self.active_subscriptions: Set[Tuple[str, SubscriptionType]] = set()
        
        # Set up rate limiters
        self.default_rate_limiter = RateLimiter(
            calls_per_second=self.settings.RATE_LIMIT_CALLS / self.settings.RATE_LIMIT_PERIOD
        )
        
        self._initialize_metrics()
        logger.info(f"Initialized {provider_name} provider")
    
    def _initialize_metrics(self) -> None:
        """Initialize provider metrics."""
        # This will be implemented in the monitoring module later
        pass
    
    @abstractmethod
    async def connect(self) -> None:
        """
        Establish connection to the provider API.
        
        This method should handle authentication and connection setup.
        
        Raises:
            ConnectionError: If connection fails
            AuthenticationError: If authentication fails
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """
        Disconnect from the provider API.
        
        This method should clean up any open connections and resources.
        """
        pass
    
    @abstractmethod
    async def get_ohlcv(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get OHLCV candlestick data for a symbol.
        
        Args:
            symbol: Instrument symbol in provider's format
            interval: Time interval (e.g., '1m', '1h', '1d')
            start_time: Start of time range (optional)
            end_time: End of time range (optional)
            limit: Maximum number of candles to return (optional)
            
        Returns:
            List of OHLCV data points
            
        Raises:
            ProviderError: If the request fails
            DataNotFoundError: If data is not available
        """
        pass
    
    @abstractmethod
    async def get_orderbook(
        self,
        symbol: str,
        depth: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get order book data for a symbol.
        
        Args:
            symbol: Instrument symbol in provider's format
            depth: Depth of order book to return (optional)
            
        Returns:
            Order book data with bids and asks
            
        Raises:
            ProviderError: If the request fails
            DataNotFoundError: If data is not available
        """
        pass
    
    @abstractmethod
    async def get_trades(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get recent trades for a symbol.
        
        Args:
            symbol: Instrument symbol in provider's format
            start_time: Start of time range (optional)
            end_time: End of time range (optional)
            limit: Maximum number of trades to return (optional)
            
        Returns:
            List of trade data points
            
        Raises:
            ProviderError: If the request fails
            DataNotFoundError: If data is not available
        """
        pass
    
    @abstractmethod
    async def get_quote(
        self,
        symbol: str
    ) -> Dict[str, Any]:
        """
        Get current quote for a symbol.
        
        Args:
            symbol: Instrument symbol in provider's format
            
        Returns:
            Quote data with bid, ask, etc.
            
        Raises:
            ProviderError: If the request fails
            DataNotFoundError: If data is not available
        """
        pass
    
    @abstractmethod
    async def subscribe_to_trades(
        self,
        symbol: str,
        callback: Callable[[Dict[str, Any]], None]
    ) -> None:
        """
        Subscribe to real-time trade updates.
        
        Args:
            symbol: Instrument symbol in provider's format
            callback: Function to call with trade updates
            
        Raises:
            ProviderError: If subscription fails
        """
        pass
    
    @abstractmethod
    async def subscribe_to_orderbook(
        self,
        symbol: str,
        callback: Callable[[Dict[str, Any]], None],
        depth: Optional[int] = None
    ) -> None:
        """
        Subscribe to real-time order book updates.
        
        Args:
            symbol: Instrument symbol in provider's format
            callback: Function to call with order book updates
            depth: Depth of order book to subscribe to (optional)
            
        Raises:
            ProviderError: If subscription fails
        """
        pass
    
    @abstractmethod
    async def subscribe_to_quotes(
        self,
        symbol: str,
        callback: Callable[[Dict[str, Any]], None]
    ) -> None:
        """
        Subscribe to real-time quote updates.
        
        Args:
            symbol: Instrument symbol in provider's format
            callback: Function to call with quote updates
            
        Raises:
            ProviderError: If subscription fails
        """
        pass
    
    @abstractmethod
    async def unsubscribe(
        self,
        symbol: str,
        subscription_type: SubscriptionType
    ) -> None:
        """
        Unsubscribe from real-time data updates.
        
        Args:
            symbol: Instrument symbol in provider's format
            subscription_type: Type of subscription to cancel
            
        Raises:
            ProviderError: If unsubscription fails
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        Check the health of the provider connection.
        
        Returns:
            Health status information
            
        Raises:
            ConnectionError: If connection check fails
        """
        pass
    
    async def authenticate(self) -> None:
        """
        Authenticate with the provider API.
        
        This method can be overridden by providers that need a custom auth flow.
        
        Raises:
            AuthenticationError: If authentication fails
        """
        pass
    
    async def with_error_handling(self, func: Callable[[], T]) -> T:
        """
        Execute a function with standardized error handling.
        
        Args:
            func: Function to execute
            
        Returns:
            Function result
            
        Raises:
            ProviderError: If the function raises an error
        """
        try:
            return await func()
        except ProviderError:
            # Re-raise provider errors as-is
            raise
        except asyncio.TimeoutError as e:
            logger.error(f"Timeout error in {self.provider_name}: {e}")
            raise ConnectionError(f"Timeout error: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error in {self.provider_name}: {e}", exc_info=True)
            raise ProviderError(f"Unexpected error: {str(e)}") from e
    
    async def with_retries(
        self,
        func: Callable[[], T],
        max_retries: Optional[int] = None,
        retry_delay: Optional[float] = None,
        retryable_errors: Optional[List[type]] = None
    ) -> T:
        """
        Execute a function with automatic retries.
        
        Args:
            func: Function to execute
            max_retries: Maximum number of retries (defaults to settings.MAX_RETRIES)
            retry_delay: Delay between retries (defaults to settings.RETRY_BACKOFF)
            retryable_errors: Error types that should be retried
            
        Returns:
            Function result
            
        Raises:
            ProviderError: If all retries fail
        """
        max_attempts = (max_retries or self.settings.MAX_RETRIES) + 1
        delay = retry_delay or self.settings.RETRY_BACKOFF
        retryable = retryable_errors or [ConnectionError, RateLimitError]
        
        last_error = None
        for attempt in range(1, max_attempts + 1):
            try:
                result = await func()
                
                # Log retry success if we had to retry
                if attempt > 1:
                    logger.info(f"Operation succeeded on attempt {attempt}/{max_attempts}")
                
                return result
                
            except tuple(retryable) as e:
                last_error = e
                if attempt < max_attempts:
                    wait_time = delay * (2 ** (attempt - 1))  # Exponential backoff
                    logger.warning(
                        f"Retryable error on attempt {attempt}/{max_attempts}, "
                        f"retrying in {wait_time:.2f}s: {e}"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"All {max_attempts} attempts failed: {e}")
                    raise
            except Exception as e:
                # Non-retryable error
                logger.error(f"Non-retryable error: {e}")
                raise
        
        # This should never happen, but just in case
        raise ProviderError(f"Failed after {max_attempts} attempts: {last_error}")
    
    def track_metric(self, metric: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None) -> None:
        """
        Track a provider metric.
        
        Args:
            metric: Metric name
            value: Metric value
            tags: Additional tags for the metric
        """
        # This will be implemented in the monitoring module later
        pass
    
    def format_symbol(self, symbol: str) -> str:
        """
        Format a symbol into the provider's expected format.
        
        This method can be overridden by providers that need custom symbol formatting.
        
        Args:
            symbol: Input symbol
            
        Returns:
            Formatted symbol
        """
        return symbol