"""
Base WebSocket client for market data providers.

This module implements a reusable WebSocket client with connection management,
automatic reconnection, heartbeat handling, and subscription tracking
for all market data providers.
"""

import logging
import json
import asyncio
import ssl
import time
import random
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable, TypeVar
from urllib.parse import urlparse

import aiohttp
from aiohttp import ClientSession, WSMsgType, ClientWebSocketResponse

from app.providers.base.provider import (
    ProviderError, ConnectionError, AuthenticationError, 
    RateLimitError, ConnectionState
)
from app.providers.config.provider_settings import BaseProviderSettings

# Set up logging
logger = logging.getLogger(__name__)

# Type variables
T = TypeVar('T')
MessageHandler = Callable[[Dict[str, Any]], None]


class WebSocketError(ProviderError):
    """Error from WebSocket connection."""
    
    def __init__(
        self, 
        message: str, 
        code: Optional[int] = None,
        data: Optional[Any] = None
    ):
        """
        Initialize a WebSocket error.
        
        Args:
            message: Error message
            code: Error code (optional)
            data: Additional error data (optional)
        """
        self.code = code
        self.data = data
        super().__init__(message)


class SubscriptionError(WebSocketError):
    """Error subscribing to a channel."""
    pass


class WebSocketClient:
    """
    Base WebSocket client for real-time market data.
    
    Provides a reusable foundation for provider-specific WebSocket clients
    with automatic reconnection, heartbeat handling, and subscription tracking.
    """
    
    def __init__(
        self, 
        url: str,
        settings: BaseProviderSettings,
        headers: Optional[Dict[str, str]] = None,
        connection_hooks: Optional[Dict[str, Callable]] = None,
        message_handlers: Optional[Dict[str, MessageHandler]] = None,
        auto_reconnect: bool = True,
    ):
        """
        Initialize a WebSocket client.
        
        Args:
            url: WebSocket URL
            settings: Provider settings
            headers: Default headers for connection
            connection_hooks: Functions to call during connection lifecycle
            message_handlers: Handler functions for different message types
            auto_reconnect: Whether to automatically reconnect
        """
        self.url = url
        self.settings = settings
        self.headers = headers or {}
        self.connection_hooks = connection_hooks or {}
        self.message_handlers = message_handlers or {}
        self.auto_reconnect = auto_reconnect
        
        # Connection state
        self.state = ConnectionState.DISCONNECTED
        self.connection_attempt = 0
        self.last_message_time: Optional[float] = None
        self.last_heartbeat_time: Optional[float] = None
        self.subscriptions: Set[Tuple[str, str]] = set()  # (channel, symbol) pairs
        
        # WebSocket objects
        self._session: Optional[ClientSession] = None
        self._ws: Optional[ClientWebSocketResponse] = None
        self._tasks: List[asyncio.Task] = []
        self._closing = False
        self._reconnect_delay = 1.0  # Initial reconnect delay
        self._max_reconnect_delay = 60.0  # Maximum reconnect delay
        
        # Connection lock to prevent multiple concurrent connection attempts
        self._connection_lock = asyncio.Lock()
        
        logger.debug(f"Initialized WebSocket client for {url}")
    
    async def __aenter__(self) -> "WebSocketClient":
        """Enter async context manager, connecting to WebSocket."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context manager, closing the connection."""
        await self.close()
    
    async def connect(self) -> None:
        """
        Connect to the WebSocket server.
        
        This method will establish a WebSocket connection and start
        background tasks for message handling and heartbeat.
        
        Raises:
            ConnectionError: If connection fails
            AuthenticationError: If authentication fails
        """
        # Use a lock to prevent multiple concurrent connection attempts
        async with self._connection_lock:
            if self.state in (ConnectionState.CONNECTED, ConnectionState.CONNECTING):
                logger.debug("Already connected or connecting, skipping connect")
                return
            
            # Set state to connecting
            self.state = ConnectionState.CONNECTING
            self.connection_attempt += 1
            self._closing = False
            
            logger.info(f"Connecting to WebSocket at {self.url} (attempt {self.connection_attempt})")
            
            try:
                # Create SSL context with recommended security settings
                ssl_context = ssl.create_default_context()
                
                # Create session if needed
                if self._session is None or self._session.closed:
                    # Create session with proper timeout settings
                    timeout = aiohttp.ClientTimeout(
                        total=self.settings.REQUEST_TIMEOUT,
                        connect=self.settings.CONNECTION_TIMEOUT
                    )
                    
                    self._session = ClientSession(
                        timeout=timeout,
                        headers=self.headers.copy()
                    )
                
                # Create WebSocket connection
                parsed_url = urlparse(self.url)
                is_secure = parsed_url.scheme == "wss"
                
                self._ws = await self._session.ws_connect(
                    self.url,
                    ssl=ssl_context if is_secure else None,
                    heartbeat=None,  # We'll handle heartbeats ourselves
                    compress=15,  # Enable compression with default settings
                    autoclose=False,  # We'll handle closing
                    max_msg_size=0,  # No limit on message size
                )
                
                # Mark as connected
                self.state = ConnectionState.CONNECTED
                self._reconnect_delay = 1.0  # Reset reconnect delay on successful connection
                self.last_message_time = time.monotonic()
                
                # Call on_connect hook if defined
                if "on_connect" in self.connection_hooks:
                    await self.connection_hooks["on_connect"](self)
                
                # Start background tasks
                self._tasks.append(asyncio.create_task(self._message_handler()))
                self._tasks.append(asyncio.create_task(self._heartbeat_handler()))
                
                logger.info(f"Connected to WebSocket at {self.url}")
                
                # Resubscribe to channels if needed
                if self.subscriptions:
                    await self._resubscribe()
                    
            except aiohttp.ClientError as e:
                self.state = ConnectionState.ERROR
                logger.error(f"WebSocket connection error: {e}")
                raise ConnectionError(f"Failed to connect to WebSocket: {e}")
            except asyncio.TimeoutError:
                self.state = ConnectionState.ERROR
                logger.error("WebSocket connection timeout")
                raise ConnectionError("Connection timeout")
            except Exception as e:
                self.state = ConnectionState.ERROR
                logger.error(f"Unexpected error connecting to WebSocket: {e}", exc_info=True)
                raise ConnectionError(f"Failed to connect to WebSocket: {e}")
    
    async def close(self) -> None:
        """
        Close the WebSocket connection.
        
        This method will close the WebSocket connection and clean up
        any background tasks.
        """
        if self._closing:
            logger.debug("Already closing, skipping close")
            return
            
        self._closing = True
        logger.info("Closing WebSocket connection")
        
        # Set state to disconnected to prevent reconnection attempts
        old_state = self.state
        self.state = ConnectionState.DISCONNECTED
        
        # Call on_close hook if defined and we were connected
        if old_state == ConnectionState.CONNECTED and "on_close" in self.connection_hooks:
            try:
                await self.connection_hooks["on_close"](self)
            except Exception as e:
                logger.error(f"Error in on_close hook: {e}")
        
        # Cancel all background tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()
        
        self._tasks.clear()
        
        # Close WebSocket connection
        if self._ws is not None:
            if not self._ws.closed:
                await self._ws.close()
            self._ws = None
        
        # Close session
        if self._session is not None:
            await self._session.close()
            self._session = None
        
        logger.info("WebSocket connection closed")
    
    async def _message_handler(self) -> None:
        """
        Background task to handle incoming WebSocket messages.
        
        This task runs until the connection is closed or an error occurs.
        """
        if self._ws is None:
            logger.error("WebSocket connection not established")
            return
            
        try:
            async for msg in self._ws:
                # Update last message time
                self.last_message_time = time.monotonic()
                
                if msg.type == WSMsgType.TEXT:
                    await self._handle_text_message(msg.data)
                elif msg.type == WSMsgType.BINARY:
                    await self._handle_binary_message(msg.data)
                elif msg.type == WSMsgType.ERROR:
                    logger.error(f"WebSocket connection closed with error: {self._ws.exception()}")
                    break
                elif msg.type == WSMsgType.CLOSE:
                    logger.info(f"WebSocket connection closed: {msg.data}, {msg.extra}")
                    break
                elif msg.type == WSMsgType.CLOSED:
                    logger.info("WebSocket connection is already closed")
                    break
                elif msg.type == WSMsgType.CLOSING:
                    logger.info("WebSocket connection is closing")
                    break
                else:
                    logger.warning(f"Received unexpected WebSocket message type: {msg.type}")
            
            logger.info("WebSocket message handler exiting")
            
            # Connection closed, attempt reconnect if needed
            if self.auto_reconnect and not self._closing:
                await self._reconnect()
            else:
                self.state = ConnectionState.DISCONNECTED
                
        except asyncio.CancelledError:
            logger.debug("WebSocket message handler cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in WebSocket message handler: {e}", exc_info=True)
            
            # Connection error, attempt reconnect if needed
            if self.auto_reconnect and not self._closing:
                await self._reconnect()
            else:
                self.state = ConnectionState.ERROR
    
    async def _heartbeat_handler(self) -> None:
        """
        Background task to handle WebSocket heartbeats.
        
        This task runs until the connection is closed, sending periodic
        heartbeats and checking for connection timeouts.
        """
        try:
            while not self._closing and self._ws is not None and not self._ws.closed:
                now = time.monotonic()
                
                # Check if we need to send a heartbeat
                if (self.last_heartbeat_time is None or 
                    now - self.last_heartbeat_time >= self.settings.WEBSOCKET_PING_INTERVAL):
                    try:
                        # Send heartbeat
                        await self._send_heartbeat()
                        self.last_heartbeat_time = now
                    except Exception as e:
                        logger.error(f"Error sending heartbeat: {e}")
                        # If we can't send a heartbeat, the connection might be dead
                        # Break and let the reconnect logic handle it
                        break
                
                # Check for connection timeout
                if (self.last_message_time is not None and 
                    now - self.last_message_time > self.settings.WEBSOCKET_PING_INTERVAL * 3):
                    logger.warning(
                        f"No messages received for {now - self.last_message_time:.1f}s, "
                        f"exceeding timeout of {self.settings.WEBSOCKET_PING_INTERVAL * 3}s"
                    )
                    # Connection is dead, break and let the reconnect logic handle it
                    break
                
                # Sleep until next heartbeat
                await asyncio.sleep(1)
            
            logger.info("WebSocket heartbeat handler exiting")
            
            # Connection closed, attempt reconnect if needed
            if self.auto_reconnect and not self._closing:
                await self._reconnect()
                
        except asyncio.CancelledError:
            logger.debug("WebSocket heartbeat handler cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in WebSocket heartbeat handler: {e}", exc_info=True)
            
            # Connection error, attempt reconnect if needed
            if self.auto_reconnect and not self._closing:
                await self._reconnect()
    
    async def _send_heartbeat(self) -> None:
        """
        Send a heartbeat message to keep the connection alive.
        
        This method should be overridden by provider-specific implementations
        if they require a custom heartbeat format.
        """
        if self._ws is None or self._ws.closed:
            raise ConnectionError("WebSocket connection not established")
        
        # Default implementation sends a simple ping frame
        await self._ws.ping()
    
    async def _reconnect(self) -> None:
        """
        Attempt to reconnect to the WebSocket server.
        
        This method will attempt to reconnect with exponential backoff,
        respecting the maximum number of reconnect attempts.
        """
        if self._closing:
            logger.debug("Connection is closing, skipping reconnect")
            return
            
        if self.state == ConnectionState.RECONNECTING:
            logger.debug("Already reconnecting, skipping duplicate reconnect")
            return
            
        # Set state to reconnecting
        self.state = ConnectionState.RECONNECTING
        
        # Calculate reconnect delay with jitter
        delay = min(self._reconnect_delay * (1.0 + random.uniform(-0.1, 0.1)), self._max_reconnect_delay)
        logger.info(f"Reconnecting in {delay:.1f}s (attempt {self.connection_attempt + 1})")
        
        # Wait before reconnecting
        await asyncio.sleep(delay)
        
        # Increase reconnect delay for next attempt
        self._reconnect_delay = min(self._reconnect_delay * 1.5, self._max_reconnect_delay)
        
        try:
            # Close existing connection if needed
            if self._ws is not None and not self._ws.closed:
                try:
                    await self._ws.close()
                except Exception as e:
                    logger.error(f"Error closing WebSocket during reconnect: {e}")
                finally:
                    self._ws = None
            
            # Clear tasks list
            for task in self._tasks:
                if not task.done():
                    task.cancel()
            
            self._tasks.clear()
            
            # Reconnect
            await self.connect()
            
        except Exception as e:
            logger.error(f"Failed to reconnect: {e}")
            
            # Set error state
            self.state = ConnectionState.ERROR
            
            # Try again if we haven't reached the maximum reconnect attempts
            if (self.settings.WEBSOCKET_MAX_RECONNECTS == 0 or 
                self.connection_attempt < self.settings.WEBSOCKET_MAX_RECONNECTS):
                # Schedule another reconnect attempt
                asyncio.create_task(self._reconnect())
            else:
                logger.error(
                    f"Exceeded maximum reconnect attempts ({self.settings.WEBSOCKET_MAX_RECONNECTS}), "
                    "giving up"
                )
    
    async def _resubscribe(self) -> None:
        """
        Resubscribe to all active channels after reconnection.
        
        This method should be overridden by provider-specific implementations
        to handle resubscription in the format expected by the provider.
        """
        if not self.subscriptions:
            logger.debug("No subscriptions to restore")
            return
            
        logger.info(f"Resubscribing to {len(self.subscriptions)} channels")
        
        # Make a copy of subscriptions to avoid modification during iteration
        subscriptions = list(self.subscriptions)
        
        # Resubscribe to all channels
        for channel, symbol in subscriptions:
            try:
                # Call the appropriate subscribe method based on channel type
                if channel == "trades":
                    await self.subscribe_to_trades(symbol)
                elif channel == "orderbook":
                    await self.subscribe_to_orderbook(symbol)
                elif channel == "quotes":
                    await self.subscribe_to_quotes(symbol)
                else:
                    logger.warning(f"Unknown channel type for resubscription: {channel}")
            except Exception as e:
                logger.error(f"Failed to resubscribe to {channel}/{symbol}: {e}")
    
    async def _handle_text_message(self, data: str) -> None:
        """
        Handle a text message from the WebSocket.
        
        Args:
            data: Message data as a string
        """
        try:
            # Parse JSON data
            message = json.loads(data)
            
            # Call message handlers based on message type
            await self._dispatch_message(message)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse WebSocket message as JSON: {e}")
            logger.debug(f"Raw message: {data[:200]}...")
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}", exc_info=True)
    
    async def _handle_binary_message(self, data: bytes) -> None:
        """
        Handle a binary message from the WebSocket.
        
        Args:
            data: Message data as bytes
        """
        logger.debug(f"Received binary message of {len(data)} bytes")
        
        try:
            # Try to decode as JSON
            message = json.loads(data.decode('utf-8'))
            await self._dispatch_message(message)
        except Exception as e:
            logger.error(f"Error handling binary WebSocket message: {e}")
            
            # Call binary message handler if defined
            if "binary" in self.message_handlers:
                try:
                    self.message_handlers["binary"](data)
                except Exception as e:
                    logger.error(f"Error in binary message handler: {e}")
    
    async def _dispatch_message(self, message: Dict[str, Any]) -> None:
        """
        Dispatch a message to the appropriate handler based on its type.
        
        This method should be overridden by provider-specific implementations
        to handle message routing based on the provider's message format.
        
        Args:
            message: Parsed message data
        """
        # Default implementation tries to determine message type from message structure
        message_type = None
        
        # Check for a type field using common names
        for type_field in ["type", "event", "e", "method", "msg_type", "messageType"]:
            if type_field in message:
                message_type = str(message[type_field])
                break
        
        if message_type is None:
            logger.warning(f"Could not determine message type: {message}")
            
            # Call default handler if defined
            if "default" in self.message_handlers:
                await self._call_handler("default", message)
            return
        
        # Call the appropriate handler based on message type
        if message_type in self.message_handlers:
            await self._call_handler(message_type, message)
        else:
            logger.debug(f"No handler for message type: {message_type}")
            
            # Call default handler if defined
            if "default" in self.message_handlers:
                await self._call_handler("default", message)
    
    async def _call_handler(self, handler_name: str, message: Dict[str, Any]) -> None:
        """
        Call a message handler with proper error handling.
        
        Args:
            handler_name: Name of the handler to call
            message: Message to pass to the handler
        """
        handler = self.message_handlers.get(handler_name)
        if handler is None:
            logger.warning(f"Handler '{handler_name}' not found")
            return
            
        try:
            # Check if handler is a coroutine function
            if asyncio.iscoroutinefunction(handler):
                await handler(message)
            else:
                handler(message)
        except Exception as e:
            logger.error(f"Error in message handler '{handler_name}': {e}", exc_info=True)
    
    async def send_message(self, message: Union[Dict[str, Any], str, bytes]) -> None:
        """
        Send a message over the WebSocket connection.
        
        Args:
            message: Message to send (dict, string, or bytes)
            
        Raises:
            ConnectionError: If connection is not established
        """
        if self._ws is None or self._ws.closed:
            raise ConnectionError("WebSocket connection not established")
            
        try:
            if isinstance(message, dict):
                # Convert dict to JSON string
                await self._ws.send_str(json.dumps(message))
            elif isinstance(message, str):
                # Send as text
                await self._ws.send_str(message)
            elif isinstance(message, bytes):
                # Send as binary
                await self._ws.send_bytes(message)
            else:
                raise ValueError(f"Unsupported message type: {type(message)}")
                
        except aiohttp.ClientError as e:
            logger.error(f"Error sending WebSocket message: {e}")
            raise ConnectionError(f"Failed to send message: {e}")
    
    async def subscribe_to_trades(self, symbol: str) -> None:
        """
        Subscribe to real-time trade updates.
        
        This method should be overridden by provider-specific implementations
        to handle trade subscription in the format expected by the provider.
        
        Args:
            symbol: Instrument symbol
            
        Raises:
            SubscriptionError: If subscription fails
        """
        # Add to subscriptions list
        self.subscriptions.add(("trades", symbol))
        logger.debug(f"Added trade subscription for {symbol}")
    
    async def subscribe_to_orderbook(self, symbol: str, depth: Optional[int] = None) -> None:
        """
        Subscribe to real-time order book updates.
        
        This method should be overridden by provider-specific implementations
        to handle order book subscription in the format expected by the provider.
        
        Args:
            symbol: Instrument symbol
            depth: Depth of order book to subscribe to (optional)
            
        Raises:
            SubscriptionError: If subscription fails
        """
        # Add to subscriptions list
        self.subscriptions.add(("orderbook", symbol))
        logger.debug(f"Added orderbook subscription for {symbol}")
    
    async def subscribe_to_quotes(self, symbol: str) -> None:
        """
        Subscribe to real-time quote updates.
        
        This method should be overridden by provider-specific implementations
        to handle quote subscription in the format expected by the provider.
        
        Args:
            symbol: Instrument symbol
            
        Raises:
            SubscriptionError: If subscription fails
        """
        # Add to subscriptions list
        self.subscriptions.add(("quotes", symbol))
        logger.debug(f"Added quote subscription for {symbol}")
    
    async def unsubscribe(self, symbol: str, channel: str) -> None:
        """
        Unsubscribe from real-time data updates.
        
        This method should be overridden by provider-specific implementations
        to handle unsubscription in the format expected by the provider.
        
        Args:
            symbol: Instrument symbol
            channel: Channel type ('trades', 'orderbook', 'quotes')
            
        Raises:
            SubscriptionError: If unsubscription fails
        """
        # Remove from subscriptions list
        self.subscriptions.discard((channel, symbol))
        logger.debug(f"Removed {channel} subscription for {symbol}")
    
    def set_message_handler(self, message_type: str, handler: MessageHandler) -> None:
        """
        Set a handler for a specific message type.
        
        Args:
            message_type: Type of message to handle
            handler: Handler function
        """
        self.message_handlers[message_type] = handler
        logger.debug(f"Set message handler for type: {message_type}")
    
    def remove_message_handler(self, message_type: str) -> None:
        """
        Remove a message handler.
        
        Args:
            message_type: Type of message handler to remove
        """
        if message_type in self.message_handlers:
            del self.message_handlers[message_type]
            logger.debug(f"Removed message handler for type: {message_type}")
    
    def is_connected(self) -> bool:
        """
        Check if the WebSocket connection is established.
        
        Returns:
            True if connected, False otherwise
        """
        return (
            self.state == ConnectionState.CONNECTED and
            self._ws is not None and
            not self._ws.closed
        )
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get connection status information.
        
        Returns:
            Dictionary with status details
        """
        return {
            "state": self.state.value,
            "connected": self.is_connected(),
            "connection_attempt": self.connection_attempt,
            "last_message_time": self.last_message_time,
            "last_heartbeat_time": self.last_heartbeat_time,
            "subscription_count": len(self.subscriptions),
            "subscriptions": list(self.subscriptions)
        }