"""
WebSocket Connection Manager for Fyers API

This module handles WebSocket connection lifecycle, reconnection strategies,
and subscription management with market-aware features.
"""
import asyncio
import logging
import json
from typing import Dict, List, Optional, Set, Any, Callable
from datetime import datetime, timedelta
import aiohttp
from aiohttp.client_exceptions import ClientError

from .fyers_ws_protocol import (
    WebSocketConfig,
    WebSocketConnectionState,
    WebSocketMessageType,
    SubscriptionRequest,
    WebSocketError,
    WebSocketMessageHandler
)
from .fyers_auth import FyersAuth
from ..base.provider import MarketServiceProtocol

# Setup logging
logger = logging.getLogger(__name__)


class WebSocketConnectionManager:
    """
    Manages WebSocket connection lifecycle with market-aware features.
    
    This class handles:
    - Connection establishment and authentication
    - Automatic reconnection with exponential backoff
    - Market-aware connection management
    - Subscription state tracking
    - Message routing
    """
    
    def __init__(
        self,
        auth: FyersAuth,
        market_service: MarketServiceProtocol,
        config: Optional[WebSocketConfig] = None,
        message_handlers: Optional[Dict[WebSocketMessageType, List[WebSocketMessageHandler]]] = None
    ):
        """
        Initialize the WebSocket connection manager.
        
        Args:
            auth: FyersAuth instance for authentication
            market_service: Market service for market-aware features
            config: Optional WebSocket configuration
            message_handlers: Optional message handler mapping
        """
        self.auth = auth
        self.market_service = market_service
        self.config = config or WebSocketConfig()
        self.message_handlers = message_handlers or {}
        
        # Connection state
        self.state = WebSocketConnectionState.DISCONNECTED
        self.ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self.session: Optional[aiohttp.ClientSession] = None
        self.retry_count = 0
        self.last_heartbeat = datetime.utcnow()
        
        # Subscription state
        self.active_subscriptions: Set[str] = set()
        self.subscription_queue: asyncio.Queue = asyncio.Queue()
        
        # Tasks
        self._connection_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._message_task: Optional[asyncio.Task] = None
        self._subscription_task: Optional[asyncio.Task] = None
        
        # Control flags
        self._running = False
        
        # Performance metrics
        self.metrics = {
            "messages_received": 0,
            "messages_processed": 0,
            "connection_attempts": 0,
            "successful_connections": 0,
            "failed_connections": 0,
            "market_open_connections": 0,
            "market_closed_connections": 0
        }
    
    async def connect(self) -> None:
        """
        Establish WebSocket connection with automatic reconnection.
        """
        if self.state in (WebSocketConnectionState.CONNECTING, WebSocketConnectionState.CONNECTED):
            return
        
        self.state = WebSocketConnectionState.CONNECTING
        self.metrics["connection_attempts"] += 1
        
        try:
            # Ensure we have a valid token
            if not await self.auth.ensure_token():
                self.state = WebSocketConnectionState.ERROR
                raise WebSocketError(code=-8, message="Authentication failed")
            
            # Create session if needed
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            # Connect to WebSocket
            self.ws = await self.session.ws_connect(
                self.config.base_url,
                timeout=self.config.connection_timeout,
                heartbeat=self.config.heartbeat_interval
            )
            
            # Authenticate
            await self._authenticate()
            
            # Start background tasks
            self._running = True
            self._start_background_tasks()
            
            self.state = WebSocketConnectionState.CONNECTED
            self.retry_count = 0
            self.metrics["successful_connections"] += 1
            
            # Update market-aware metrics
            symbol = next(iter(self.active_subscriptions)) if self.active_subscriptions else None
            if symbol and self.market_service.is_market_open(symbol):
                self.metrics["market_open_connections"] += 1
            else:
                self.metrics["market_closed_connections"] += 1
            
            logger.info("WebSocket connection established successfully")
        
        except WebSocketError as e:
            self.metrics["failed_connections"] += 1
            self.state = WebSocketConnectionState.ERROR
            logger.error(f"WebSocket connection failed: {str(e)}")
            raise  # Re-raise WebSocketError
        except Exception as e:
            self.metrics["failed_connections"] += 1
            self.state = WebSocketConnectionState.ERROR
            logger.error(f"WebSocket connection failed: {str(e)}")
            
            # Attempt reconnection
            await self._handle_reconnection()
    
    async def disconnect(self) -> None:
        """
        Gracefully close the WebSocket connection.
        """
        # Stop background tasks
        self._running = False
        
        # Cancel background tasks
        for task in [self._connection_task, self._heartbeat_task, 
                    self._message_task, self._subscription_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Close WebSocket connection
        if self.ws:
            await self.ws.close()
            self.ws = None
        
        # Close session
        if self.session:
            await self.session.close()
            self.session = None
        
        self.state = WebSocketConnectionState.DISCONNECTED
        logger.info("WebSocket connection closed")
    
    async def subscribe(self, symbols: List[str], data_type: WebSocketMessageType = WebSocketMessageType.SYMBOL_FEED) -> None:
        """
        Subscribe to market data for specified symbols.
        
        Args:
            symbols: List of symbols to subscribe to
            data_type: Type of data to subscribe to
        """
        if not symbols:
            return
        
        # Validate subscription limit
        if len(self.active_subscriptions) + len(symbols) > self.config.max_symbols_per_connection:
            raise ValueError(f"Subscription limit ({self.config.max_symbols_per_connection}) exceeded")
        
        # Create subscription request
        request = SubscriptionRequest(symbols=symbols, data_type=data_type)
        
        # Add to subscription queue
        await self.subscription_queue.put(request)
        
        # Update active subscriptions
        self.active_subscriptions.update(symbols)
    
    async def unsubscribe(self, symbols: List[str]) -> None:
        """
        Unsubscribe from market data for specified symbols.
        
        Args:
            symbols: List of symbols to unsubscribe from
        """
        if not symbols:
            return
        
        if self.ws and self.state == WebSocketConnectionState.CONNECTED:
            message = {
                "type": "unsubscribe",
                "symbols": symbols
            }
            await self.ws.send_json(message)
        
        # Update active subscriptions
        self.active_subscriptions.difference_update(symbols)
    
    def add_message_handler(self, message_type: WebSocketMessageType, handler: WebSocketMessageHandler) -> None:
        """
        Add a message handler for a specific message type.
        
        Args:
            message_type: Type of message to handle
            handler: Callback function to handle messages
        """
        if message_type not in self.message_handlers:
            self.message_handlers[message_type] = []
        self.message_handlers[message_type].append(handler)
    
    def remove_message_handler(self, message_type: WebSocketMessageType, handler: WebSocketMessageHandler) -> None:
        """
        Remove a message handler for a specific message type.
        
        Args:
            message_type: Type of message to handle
            handler: Callback function to remove
        """
        if message_type in self.message_handlers:
            try:
                self.message_handlers[message_type].remove(handler)
            except ValueError:
                pass
    
    async def _authenticate(self) -> None:
        """
        Authenticate WebSocket connection using access token.
        """
        self.state = WebSocketConnectionState.AUTHENTICATING
        
        auth_message = {
            "type": "authenticate",
            "token": self.auth.access_token
        }
        
        await self.ws.send_json(auth_message)
        
        # Wait for authentication response
        response = await self.ws.receive_json()
        
        if response.get("s") != "ok":
            raise WebSocketError(
                code=response.get("code", -1),
                message=response.get("message", "Authentication failed")
            )
    
    def _start_background_tasks(self) -> None:
        """
        Start background tasks for connection management.
        """
        if not self._heartbeat_task or self._heartbeat_task.done():
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        
        if not self._message_task or self._message_task.done():
            self._message_task = asyncio.create_task(self._message_loop())
        
        if not self._subscription_task or self._subscription_task.done():
            self._subscription_task = asyncio.create_task(self._subscription_loop())
    
    async def _heartbeat_loop(self) -> None:
        """
        Send periodic heartbeat messages to keep connection alive.
        """
        while self._running:
            try:
                if self.ws and self.state == WebSocketConnectionState.CONNECTED:
                    # Check if we've missed too many heartbeats
                    if datetime.utcnow() - self.last_heartbeat > timedelta(seconds=self.config.heartbeat_interval * 2):
                        logger.warning("Missed heartbeat, initiating reconnection")
                        await self._handle_reconnection()
                        continue
                    
                    # Send heartbeat
                    await self.ws.ping()
                    self.last_heartbeat = datetime.utcnow()
                
                await asyncio.sleep(self.config.heartbeat_interval)
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {str(e)}")
                await self._handle_reconnection()
    
    async def _message_loop(self) -> None:
        """
        Process incoming WebSocket messages.
        """
        while self._running:
            try:
                if not self.ws:
                    await asyncio.sleep(0.1)
                    continue
                
                try:
                    # Use wait_for to prevent infinite blocking
                    message = await asyncio.wait_for(
                        self.ws.receive_json(),
                        timeout=self.config.heartbeat_interval * 2
                    )
                    self.metrics["messages_received"] += 1
                    
                    # Process message
                    await self._process_message(message)
                except (asyncio.TimeoutError, aiohttp.ClientConnectionError) as e:
                    # Connection closed or timed out
                    logger.warning(f"WebSocket connection issue: {str(e)}")
                    await self._handle_reconnection()
                    break
            
            except asyncio.CancelledError:
                logger.info("Message loop cancelled")
                break
            except Exception as e:
                logger.error(f"Message processing error: {str(e)}")
                await self._handle_reconnection()
                break
    
    async def _subscription_loop(self) -> None:
        """
        Process subscription requests from queue.
        """
        while self._running:
            try:
                # Wait for subscription request
                request = await self.subscription_queue.get()
                
                if self.ws and self.state == WebSocketConnectionState.CONNECTED:
                    await self.ws.send_json(request.to_dict())
                
                self.subscription_queue.task_done()
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Subscription error: {str(e)}")
    
    async def _process_message(self, message: Dict[str, Any]) -> None:
        """
        Process incoming message and route to appropriate handlers.
        
        Args:
            message: Received WebSocket message
        """
        try:
            message_type = message.get("type", "")
            
            # Route message to registered handlers
            if message_type in self.message_handlers:
                for handler in self.message_handlers[message_type]:
                    try:
                        await handler(message)
                    except Exception as e:
                        logger.error(f"Handler error for {message_type}: {str(e)}")
            
            # Update metrics after successful processing
            self.metrics["messages_processed"] += 1
        
        except Exception as e:
            logger.error(f"Message processing error: {str(e)}")
            # Don't increment metrics for failed processing
    
    async def _handle_reconnection(self) -> None:
        """
        Handle connection failures with exponential backoff.
        """
        if self.retry_count >= self.config.max_retries:
            logger.error("Max retry attempts reached")
            self.state = WebSocketConnectionState.ERROR
            return
        
        self.state = WebSocketConnectionState.RECONNECTING
        self.retry_count += 1
        
        # Calculate delay with exponential backoff
        delay = min(
            self.config.retry_delay * (2 ** (self.retry_count - 1)),
            self.config.max_retry_delay
        )
        
        # Adjust delay based on market state
        symbol = next(iter(self.active_subscriptions)) if self.active_subscriptions else None
        if symbol:
            if self.market_service.is_market_open(symbol):
                # Faster reconnection during market hours
                delay = delay * 0.5
            else:
                # Slower reconnection outside market hours
                delay = delay * 2.0
        
        logger.info(f"Attempting reconnection in {delay:.1f} seconds (attempt {self.retry_count})")
        await asyncio.sleep(delay)
        
        try:
            # Attempt reconnection
            await self.connect()
        except WebSocketError:
            # If it's a WebSocket error, we want to stop retrying
            self.state = WebSocketConnectionState.ERROR
            raise
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """
        Get connection statistics and metrics.
        
        Returns:
            Dictionary containing connection stats and metrics
        """
        return {
            "state": self.state.value,
            "retry_count": self.retry_count,
            "active_subscriptions": len(self.active_subscriptions),
            "metrics": self.metrics
        } 