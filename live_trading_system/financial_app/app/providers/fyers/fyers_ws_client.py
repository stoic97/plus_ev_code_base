"""
Fyers WebSocket Client

This module provides a market-aware WebSocket client for the Fyers API,
optimized for medium-frequency trading with robust error handling and
automatic reconnection.
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime

from .fyers_ws_protocol import (
    WebSocketConfig,
    WebSocketMessageType,
    MarketDataUpdate,
    MarketDepthUpdate,
    WebSocketError,
    WebSocketMessageHandler
)
from .fyers_ws_manager import WebSocketConnectionManager
from .fyers_auth import FyersAuth
from .fyers_settings import FyersSettings
from .fyers_rest_client import NoOpMarketService
from ..base.provider import MarketServiceProtocol
from ..base.cache import MarketAwareStrategicCache

# Setup logging
logger = logging.getLogger(__name__)


class FyersWebSocketClient:
    """
    Market-aware WebSocket client for Fyers API.
    
    Features:
    - Market data streaming with automatic reconnection
    - Market-aware connection management
    - Integration with strategic cache
    - Comprehensive error handling
    - Performance monitoring
    """
    
    def __init__(
        self,
        settings: FyersSettings,
        auth: Optional[FyersAuth] = None,
        market_service: Optional[MarketServiceProtocol] = None,
        cache: Optional[MarketAwareStrategicCache] = None,
        config: Optional[WebSocketConfig] = None
    ):
        """
        Initialize the WebSocket client.
        
        Args:
            settings: Fyers API settings
            auth: Optional authentication service (created if not provided)
            market_service: Optional market service for market awareness
            cache: Optional strategic cache for data caching
            config: Optional WebSocket configuration
        """
        self.settings = settings
        self.auth = auth or FyersAuth(settings)
        self.market_service = market_service or NoOpMarketService()
        self.cache = cache
        
        # Initialize connection manager
        self.connection = WebSocketConnectionManager(
            auth=self.auth,
            market_service=self.market_service,
            config=config
        )
        
        # Callback registry
        self.callbacks: Dict[str, List[Callable]] = {
            "market_data": [],
            "market_depth": [],
            "connection": [],
            "error": []
        }
        
        # Setup message handlers
        self._setup_message_handlers()
        
        # Performance metrics
        self.metrics = {
            "messages_processed": 0,
            "market_data_updates": 0,
            "depth_updates": 0,
            "errors": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
    
    async def connect(self) -> None:
        """
        Establish WebSocket connection.
        """
        await self.connection.connect()
    
    async def disconnect(self) -> None:
        """
        Close WebSocket connection.
        """
        await self.connection.disconnect()
    
    async def subscribe_market_data(self, symbols: List[str]) -> None:
        """
        Subscribe to market data for specified symbols.
        
        Args:
            symbols: List of symbols to subscribe to
        """
        await self.connection.subscribe(symbols, WebSocketMessageType.SYMBOL_FEED)
    
    async def subscribe_market_depth(self, symbols: List[str]) -> None:
        """
        Subscribe to market depth data for specified symbols.
        
        Args:
            symbols: List of symbols to subscribe to
        """
        await self.connection.subscribe(symbols, WebSocketMessageType.DEPTH)
    
    async def unsubscribe(self, symbols: List[str]) -> None:
        """
        Unsubscribe from data for specified symbols.
        
        Args:
            symbols: List of symbols to unsubscribe from
        """
        await self.connection.unsubscribe(symbols)
    
    def on_market_data(self, callback: Callable[[MarketDataUpdate], None]) -> None:
        """
        Register callback for market data updates.
        
        Args:
            callback: Function to call with market data updates
        """
        self.callbacks["market_data"].append(callback)
    
    def on_market_depth(self, callback: Callable[[MarketDepthUpdate], None]) -> None:
        """
        Register callback for market depth updates.
        
        Args:
            callback: Function to call with market depth updates
        """
        self.callbacks["market_depth"].append(callback)
    
    def on_connection(self, callback: Callable[[str], None]) -> None:
        """
        Register callback for connection state changes.
        
        Args:
            callback: Function to call with connection state updates
        """
        self.callbacks["connection"].append(callback)
    
    def on_error(self, callback: Callable[[WebSocketError], None]) -> None:
        """
        Register callback for error handling.
        
        Args:
            callback: Function to call with error information
        """
        self.callbacks["error"].append(callback)
    
    def _setup_message_handlers(self) -> None:
        """
        Setup WebSocket message handlers.
        """
        # Market data handler
        self.connection.add_message_handler(
            WebSocketMessageType.SYMBOL_FEED,
            self._handle_market_data
        )
        
        # Market depth handler
        self.connection.add_message_handler(
            WebSocketMessageType.DEPTH,
            self._handle_market_depth
        )
        
        # Error handler
        self.connection.add_message_handler(
            WebSocketMessageType.ERROR,
            self._handle_error
        )
    
    async def _handle_market_data(self, message: Dict[str, Any]) -> None:
        """
        Handle market data update messages.
        
        Args:
            message: Market data message
        """
        try:
            update = MarketDataUpdate.from_message(message)
            self.metrics["market_data_updates"] += 1
            
            # Update cache if available
            if self.cache:
                await self._update_cache(update)
            
            # Notify callbacks
            for callback in self.callbacks["market_data"]:
                try:
                    callback(update)
                except Exception as e:
                    logger.error(f"Error in market data callback: {e}")
                    self.metrics["errors"] += 1
            
            self.metrics["messages_processed"] += 1
            
        except Exception as e:
            logger.error(f"Error handling market data message: {e}")
            self.metrics["errors"] += 1
    
    async def _handle_market_depth(self, message: Dict[str, Any]) -> None:
        """
        Handle market depth update messages.
        
        Args:
            message: Market depth message
        """
        try:
            update = MarketDepthUpdate.from_message(message)
            self.metrics["depth_updates"] += 1
            
            # Notify callbacks
            for callback in self.callbacks["market_depth"]:
                try:
                    callback(update)
                except Exception as e:
                    logger.error(f"Market depth callback error: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error processing market depth: {str(e)}")
            self.metrics["errors"] += 1
    
    async def _handle_error(self, message: Dict[str, Any]) -> None:
        """
        Handle error messages.
        
        Args:
            message: Error message
        """
        try:
            error = WebSocketError.from_message(message)
            self.metrics["errors"] += 1
            
            # Notify callbacks
            for callback in self.callbacks["error"]:
                try:
                    callback(error)
                except Exception as e:
                    logger.error(f"Error callback error: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error processing error message: {str(e)}")
    
    async def _update_cache(self, update: MarketDataUpdate) -> None:
        """
        Update cache with market data.
        
        Args:
            update: Market data update
        """
        try:
            if not self.cache:
                return
            
            # Check if we have this symbol in cache
            cache_key = f"market_data:{update.symbol}"
            existing = self.cache.get(cache_key)
            
            # Update metrics before potential error
            if existing is not None:
                self.metrics["cache_hits"] += 1
            else:
                self.metrics["cache_misses"] += 1
            
            # Update cache
            self.cache.set(
                key=cache_key,
                value=update.to_dict(),
                data_type="quotes"
            )
            
        except Exception as e:
            logger.error(f"Cache update error: {e}")
            self.metrics["errors"] += 1
    
    async def get_metrics(self) -> Dict[str, Any]:
        """
        Get client metrics.
        
        Returns:
            Dictionary of metrics
        """
        metrics = self.metrics.copy()
        
        # Add connection stats
        connection_stats = await self.connection.get_connection_stats()
        metrics.update(connection_stats)
        
        # Add cache stats if available
        if self.cache:
            cache_stats = self.cache.get_cache_performance()
            metrics.update({
                "cache_hit_ratio": cache_stats.get("hit_ratio", 0.0),
                "cache_eviction_rate": cache_stats.get("eviction_rate", 0.0),
                "cache_entries": len(self.cache.cache) if hasattr(self.cache, "cache") else 0
            })
        
        return metrics 