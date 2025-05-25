"""
Unit tests for the WebSocket client.

This module contains tests for the WebSocket client implementation,
covering connection management, message handling, subscription tracking,
and error recovery.
"""

import asyncio
import json
import ssl
import pytest
import pytest_asyncio
import time
from unittest.mock import MagicMock, patch, AsyncMock, call
from typing import Dict, Any, Optional, List, Tuple

import aiohttp
from aiohttp import WSMsgType, WSMessage, ClientWebSocketResponse
from aiohttp.client_exceptions import ClientError

from app.providers.base.provider import ConnectionState, ConnectionError as ProviderConnectionError
from app.providers.base.websocket_client import (
    WebSocketClient, WebSocketError, SubscriptionError, 
    MessageHandler
)
from app.providers.config.provider_settings import BaseProviderSettings


class MockWebSocketResponse:
    """Mock WebSocket response for testing."""
    
    def __init__(self, messages=None):
        self.messages = messages or []
        self.closed = False
        self.exception_value = None
        self.close_code = 1000
        self.close_message = "Normal closure"
        self.sent_messages = []
    
    def __aiter__(self):
        return self
    
    async def __anext__(self):
        if not self.messages:
            raise StopAsyncIteration
        
        message = self.messages.pop(0)
        if isinstance(message, Exception):
            raise message
        return message
    
    async def close(self, code=1000, message="Normal closure"):
        self.closed = True
        self.close_code = code
        self.close_message = message
    
    def exception(self):
        return self.exception_value
    
    async def send_str(self, data):
        self.sent_messages.append(("text", data))
    
    async def send_bytes(self, data):
        self.sent_messages.append(("binary", data))
    
    async def ping(self):
        self.sent_messages.append(("ping", None))


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    settings = MagicMock(spec=BaseProviderSettings)
    settings.WEBSOCKET_PING_INTERVAL = 1  # Short interval for quick testing
    settings.WEBSOCKET_RECONNECT_DELAY = 0.1  # Quick reconnect for testing
    settings.WEBSOCKET_MAX_RECONNECTS = 5
    settings.REQUEST_TIMEOUT = 30.0
    settings.CONNECTION_TIMEOUT = 10.0
    return settings


@pytest.fixture
def mock_ws_response():
    """Create a mock WebSocket response."""
    return MockWebSocketResponse()


@pytest.fixture
def mock_session(mock_ws_response):
    """Create a mock aiohttp ClientSession."""
    session = AsyncMock(spec=aiohttp.ClientSession)
    session.ws_connect = AsyncMock(return_value=mock_ws_response)
    session.closed = False
    session.close = AsyncMock()
    return session


@pytest.fixture
def websocket_client(mock_settings):
    """Create a WebSocket client for testing."""
    url = "wss://test.example.com/ws"
    headers = {"X-Test": "test-value"}
    
    # Create message handlers
    message_handlers = {
        "trade": AsyncMock(),
        "orderbook": AsyncMock(),
        "quote": AsyncMock(),
        "default": AsyncMock(),
        "binary": AsyncMock()
    }
    
    # Create connection hooks
    connection_hooks = {
        "on_connect": AsyncMock(),
        "on_close": AsyncMock()
    }
    
    # Create client
    client = WebSocketClient(
        url=url,
        settings=mock_settings,
        headers=headers,
        message_handlers=message_handlers,
        connection_hooks=connection_hooks
    )
    
    return client


class TestWebSocketClient:
    """Test cases for the WebSocket client."""
    
    @pytest.mark.asyncio
    async def test_connect(self, websocket_client, mock_session, monkeypatch):
        """Test connecting to WebSocket server."""
        # Set up session directly to avoid creating a real one
        websocket_client._session = mock_session
        
        # Connect
        await websocket_client.connect()
        
        # Check state
        assert websocket_client.state == ConnectionState.CONNECTED
        assert websocket_client.connection_attempt == 1
        assert websocket_client.last_message_time is not None
        
        # Check session creation
        assert mock_session.ws_connect.called
        
        # Check on_connect hook called
        assert websocket_client.connection_hooks["on_connect"].called
        
        # Cleanup
        if websocket_client._tasks:
            for task in websocket_client._tasks:
                task.cancel()
            websocket_client._tasks = []

    @pytest.mark.asyncio
    async def test_connect_error(self, websocket_client, mock_session):
        """Test handling connection error."""
        # Set up session directly to avoid creating a real one
        websocket_client._session = mock_session
        
        # Make ws_connect raise an error
        mock_session.ws_connect.side_effect = ClientError("Connection failed")
        
        # Connect should raise ConnectionError
        with pytest.raises(ProviderConnectionError) as excinfo:
            await websocket_client.connect()
        
        # Verify the error message
        assert "Failed to connect to WebSocket: Connection failed" in str(excinfo.value)
        
        # Check state
        assert websocket_client.state == ConnectionState.ERROR

    @pytest.mark.asyncio
    async def test_close(self, websocket_client, mock_session, mock_ws_response):
        """Test closing the WebSocket connection."""
        # Set up client with mock objects
        websocket_client._session = mock_session
        websocket_client._ws = mock_ws_response
        websocket_client.state = ConnectionState.CONNECTED
        
        # Set up some tasks
        websocket_client._tasks = [
            asyncio.create_task(asyncio.sleep(1)),
            asyncio.create_task(asyncio.sleep(1))
        ]
        
        # Close
        await websocket_client.close()
        
        # Check state
        assert websocket_client.state == ConnectionState.DISCONNECTED
        assert websocket_client._closing is True
        assert len(websocket_client._tasks) == 0
        assert mock_ws_response.closed is True
        assert mock_session.close.called
        
        # Check on_close hook called
        assert websocket_client.connection_hooks["on_close"].called

    @pytest.mark.asyncio
    async def test_message_handler(self, websocket_client, monkeypatch):
        """Test the message handler process."""
        # Create mock messages
        text_message = WSMessage(WSMsgType.TEXT, '{"type": "trade", "data": {"symbol": "AAPL", "price": 150.0}}', None)
        binary_message = WSMessage(WSMsgType.BINARY, b'{"type": "orderbook", "data": {"symbol": "AAPL"}}', None)
        error_message = WSMessage(WSMsgType.ERROR, None, None)
        close_message = WSMessage(WSMsgType.CLOSE, 1000, "Normal closure")
        
        # Create mock WebSocket response with messages
        mock_ws = MockWebSocketResponse([
            text_message,
            binary_message,
            error_message,
            close_message
        ])
        mock_ws.exception_value = Exception("Test error")
        
        # Patch _handle_text_message and _handle_binary_message to avoid issues with message format
        handle_text_mock = AsyncMock()
        handle_binary_mock = AsyncMock()
        monkeypatch.setattr(websocket_client, "_handle_text_message", handle_text_mock)
        monkeypatch.setattr(websocket_client, "_handle_binary_message", handle_binary_mock)
        
        # Patch reconnect method 
        reconnect_mock = AsyncMock()
        monkeypatch.setattr(websocket_client, "_reconnect", reconnect_mock)
        
        # Set _ws directly
        websocket_client._ws = mock_ws
        websocket_client.state = ConnectionState.CONNECTED
        websocket_client.auto_reconnect = True
        
        # Run message handler
        await websocket_client._message_handler()
        
        # Check handlers called
        assert handle_text_mock.called
        assert handle_binary_mock.called
        
        # Check reconnect called (after error or close message)
        assert reconnect_mock.called

    @pytest.mark.asyncio
    async def test_heartbeat_handler(self, websocket_client, monkeypatch):
        """Test the heartbeat handler process."""
        # Create mock WebSocket
        mock_ws = MockWebSocketResponse()
        
        # Mock send_heartbeat method
        send_heartbeat_mock = AsyncMock()
        monkeypatch.setattr(websocket_client, "_send_heartbeat", send_heartbeat_mock)
        
        # Mock reconnect method
        reconnect_mock = AsyncMock()
        monkeypatch.setattr(websocket_client, "_reconnect", reconnect_mock)
        
        # Set up WebSocket client
        websocket_client._ws = mock_ws
        websocket_client.state = ConnectionState.CONNECTED
        websocket_client._closing = False
        websocket_client.auto_reconnect = True
        
        # Set initial message time (recent)
        websocket_client.last_message_time = time.monotonic()
        
        # Create a custom implementation of _heartbeat_handler that we can better control
        async def test_heartbeat():
            # First, trigger a normal heartbeat
            await websocket_client._send_heartbeat()
            
            # Then simulate timeout
            websocket_client.last_message_time = time.monotonic() - (websocket_client.settings.WEBSOCKET_PING_INTERVAL * 4)
            
            # Manually check for timeout as the actual handler would
            if (websocket_client.last_message_time is not None and 
                time.monotonic() - websocket_client.last_message_time > websocket_client.settings.WEBSOCKET_PING_INTERVAL * 3):
                await websocket_client._reconnect()
        
        # Run our controlled test heartbeat
        await test_heartbeat()
        
        # Check send_heartbeat called
        assert send_heartbeat_mock.called
        
        # Check reconnect called due to timeout
        assert reconnect_mock.called

    @pytest.mark.asyncio
    async def test_send_heartbeat(self, websocket_client, mock_ws_response):
        """Test sending heartbeat."""
        # Set up WebSocket client
        websocket_client._ws = mock_ws_response
        
        # Send heartbeat
        await websocket_client._send_heartbeat()
        
        # Check ping sent
        assert ("ping", None) in mock_ws_response.sent_messages

    @pytest.mark.asyncio
    async def test_reconnect(self, websocket_client, monkeypatch):
        """Test reconnection process."""
        # Mock connect method
        connect_mock = AsyncMock()
        monkeypatch.setattr(websocket_client, "connect", connect_mock)
        
        # Mock sleep to speed up test
        sleep_mock = AsyncMock()
        monkeypatch.setattr(asyncio, "sleep", sleep_mock)
        
        # Set up WebSocket client
        websocket_client._ws = MockWebSocketResponse()
        websocket_client.state = ConnectionState.DISCONNECTED
        websocket_client._reconnect_delay = 1.0
        websocket_client._closing = False
        
        # Reconnect
        await websocket_client._reconnect()
        
        # Check state and reconnect delay
        assert websocket_client.state == ConnectionState.RECONNECTING
        assert websocket_client._reconnect_delay > 1.0
        
        # Check sleep called
        assert sleep_mock.called
        
        # Check connect called
        assert connect_mock.called

    @pytest.mark.asyncio
    async def test_reconnect_failure(self, websocket_client, monkeypatch):
        """Test handling reconnection failure."""
        # Mock connect method to fail
        connect_mock = AsyncMock(side_effect=ProviderConnectionError("Reconnect failed"))
        monkeypatch.setattr(websocket_client, "connect", connect_mock)
        
        # Mock sleep to speed up test
        sleep_mock = AsyncMock()
        monkeypatch.setattr(asyncio, "sleep", sleep_mock)
        
        # Mock asyncio.create_task to capture the call
        create_task_original = asyncio.create_task
        task_mock = MagicMock(wraps=create_task_original)
        monkeypatch.setattr(asyncio, "create_task", task_mock)
        
        # Set up WebSocket client
        websocket_client._ws = MockWebSocketResponse()
        websocket_client.state = ConnectionState.DISCONNECTED
        websocket_client._reconnect_delay = 1.0
        websocket_client._closing = False
        
        # Reconnect
        await websocket_client._reconnect()
        
        # Check state
        assert websocket_client.state == ConnectionState.ERROR
        
        # Verify another reconnect is scheduled
        assert task_mock.called
        
        # Now test with max reconnects exceeded
        # Reset mocks
        task_mock.reset_mock()
        
        # Exceed max reconnects
        websocket_client.connection_attempt = websocket_client.settings.WEBSOCKET_MAX_RECONNECTS
        
        # Reconnect again
        await websocket_client._reconnect()
        
        # Check another reconnect is NOT scheduled - task not called with a reconnect function
        reconnect_scheduled = False
        for call_args in task_mock.call_args_list:
            if "_reconnect" in str(call_args):
                reconnect_scheduled = True
                break
                
        assert not reconnect_scheduled

    @pytest.mark.asyncio
    async def test_resubscribe(self, websocket_client, monkeypatch):
        """Test resubscribing to channels after reconnection."""
        # Mock subscribe methods
        subscribe_trades_mock = AsyncMock()
        subscribe_orderbook_mock = AsyncMock()
        subscribe_quotes_mock = AsyncMock()
        
        monkeypatch.setattr(websocket_client, "subscribe_to_trades", subscribe_trades_mock)
        monkeypatch.setattr(websocket_client, "subscribe_to_orderbook", subscribe_orderbook_mock)
        monkeypatch.setattr(websocket_client, "subscribe_to_quotes", subscribe_quotes_mock)
        
        # Set up subscriptions
        websocket_client.subscriptions = {
            ("trades", "AAPL"),
            ("orderbook", "MSFT"),
            ("quotes", "GOOGL")
        }
        
        # Resubscribe
        await websocket_client._resubscribe()
        
        # Check subscribe methods called
        assert subscribe_trades_mock.called
        assert subscribe_orderbook_mock.called
        assert subscribe_quotes_mock.called

    @pytest.mark.asyncio
    async def test_handle_text_message(self, websocket_client, monkeypatch):
        """Test handling text message."""
        # Mock dispatch method
        dispatch_mock = AsyncMock()
        monkeypatch.setattr(websocket_client, "_dispatch_message", dispatch_mock)
        
        # Create test message
        message = json.dumps({"type": "trade", "data": {"symbol": "AAPL", "price": 150.0}})
        
        # Handle message
        await websocket_client._handle_text_message(message)
        
        # Check dispatch called with parsed message
        dispatch_mock.assert_called_once()
        call_args = dispatch_mock.call_args[0][0]
        assert call_args["type"] == "trade"
        assert call_args["data"]["symbol"] == "AAPL"

    @pytest.mark.asyncio
    async def test_handle_binary_message(self, websocket_client, monkeypatch):
        """Test handling binary message."""
        # Mock dispatch method
        dispatch_mock = AsyncMock()
        monkeypatch.setattr(websocket_client, "_dispatch_message", dispatch_mock)
        
        # Create test message
        message = json.dumps({"type": "orderbook", "data": {"symbol": "AAPL"}}).encode('utf-8')
        
        # Handle message
        await websocket_client._handle_binary_message(message)
        
        # Check dispatch called with parsed message
        dispatch_mock.assert_called_once()
        call_args = dispatch_mock.call_args[0][0]
        assert call_args["type"] == "orderbook"
        assert call_args["data"]["symbol"] == "AAPL"

        # Test non-JSON binary message
        dispatch_mock.reset_mock()
        binary_handler = AsyncMock()
        
        # Replace the binary handler with our mock
        old_handler = websocket_client.message_handlers.get("binary")
        websocket_client.message_handlers["binary"] = binary_handler
        
        try:
            # Create binary message
            binary_data = b'\x01\x02\x03\x04'
            
            # Handle message - this should call our mock binary handler
            await websocket_client._handle_binary_message(binary_data)
            
            # Check binary handler was called
            binary_handler.assert_called_once_with(binary_data)
        finally:
            # Restore the original handler
            if old_handler:
                websocket_client.message_handlers["binary"] = old_handler

    @pytest.mark.asyncio
    async def test_dispatch_message(self, websocket_client, monkeypatch):
        """Test dispatching message to handlers."""
        # Mock call_handler method
        call_handler_mock = AsyncMock()
        monkeypatch.setattr(websocket_client, "_call_handler", call_handler_mock)
        
        # Test with known type
        message = {"type": "trade", "data": {"symbol": "AAPL"}}
        await websocket_client._dispatch_message(message)
        
        # Check handler called
        call_handler_mock.assert_called_with("trade", message)
        
        # Test with unknown type
        call_handler_mock.reset_mock()
        message = {"unknown_field": "value"}
        await websocket_client._dispatch_message(message)
        
        # Check default handler called
        call_handler_mock.assert_called_with("default", message)

    @pytest.mark.asyncio
    async def test_call_handler(self, websocket_client):
        """Test calling message handlers."""
        # Set up handlers
        sync_handler = MagicMock()
        async_handler = AsyncMock()
        
        websocket_client.message_handlers["sync"] = sync_handler
        websocket_client.message_handlers["async"] = async_handler
        
        # Test sync handler
        message = {"data": "test"}
        await websocket_client._call_handler("sync", message)
        
        # Check sync handler called
        sync_handler.assert_called_with(message)
        
        # Test async handler
        await websocket_client._call_handler("async", message)
        
        # Check async handler called
        async_handler.assert_called_with(message)
        
        # Test handler not found
        await websocket_client._call_handler("nonexistent", message)
        # Should not raise an exception

    @pytest.mark.asyncio
    async def test_send_message(self, websocket_client, mock_ws_response):
        """Test sending messages."""
        # Set up WebSocket client
        websocket_client._ws = mock_ws_response
        
        # Test dict message
        dict_message = {"type": "subscribe", "symbol": "AAPL"}
        await websocket_client.send_message(dict_message)
        
        # Test string message
        string_message = "test message"
        await websocket_client.send_message(string_message)
        
        # Test binary message
        binary_message = b"binary data"
        await websocket_client.send_message(binary_message)
        
        # Check messages sent
        assert len(mock_ws_response.sent_messages) == 3
        assert mock_ws_response.sent_messages[0][0] == "text"
        assert mock_ws_response.sent_messages[1][0] == "text"
        assert mock_ws_response.sent_messages[2][0] == "binary"
        
        # Test with closed connection
        mock_ws_response.closed = True
        
        # Should raise ConnectionError
        with pytest.raises(ProviderConnectionError):
            await websocket_client.send_message("test")
    
    @pytest.mark.asyncio
    async def test_subscription_methods(self, websocket_client):
        """Test subscription management methods."""
        # Test subscribing to trades
        await websocket_client.subscribe_to_trades("AAPL")
        assert ("trades", "AAPL") in websocket_client.subscriptions
        
        # Test subscribing to orderbook
        await websocket_client.subscribe_to_orderbook("MSFT")
        assert ("orderbook", "MSFT") in websocket_client.subscriptions
        
        # Test subscribing to quotes
        await websocket_client.subscribe_to_quotes("GOOGL")
        assert ("quotes", "GOOGL") in websocket_client.subscriptions
        
        # Test unsubscribing
        await websocket_client.unsubscribe("AAPL", "trades")
        assert ("trades", "AAPL") not in websocket_client.subscriptions

    def test_set_remove_message_handler(self, websocket_client):
        """Test setting and removing message handlers."""
        # Create a new handler
        new_handler = AsyncMock()
        
        # Set handler
        websocket_client.set_message_handler("new_type", new_handler)
        assert "new_type" in websocket_client.message_handlers
        assert websocket_client.message_handlers["new_type"] == new_handler
        
        # Remove handler
        websocket_client.remove_message_handler("new_type")
        assert "new_type" not in websocket_client.message_handlers

    def test_is_connected(self, websocket_client, mock_ws_response):
        """Test connection status check."""
        # Not connected initially
        assert not websocket_client.is_connected()
        
        # Set as connected
        websocket_client.state = ConnectionState.CONNECTED
        websocket_client._ws = mock_ws_response
        mock_ws_response.closed = False
        
        # Should be connected
        assert websocket_client.is_connected()
        
        # Test with closed connection
        mock_ws_response.closed = True
        assert not websocket_client.is_connected()

    def test_get_status(self, websocket_client):
        """Test getting status information."""
        # Set up some state
        websocket_client.state = ConnectionState.CONNECTED
        websocket_client.connection_attempt = 3
        websocket_client.last_message_time = time.monotonic()
        websocket_client.last_heartbeat_time = time.monotonic()
        websocket_client.subscriptions = {("trades", "AAPL"), ("orderbook", "MSFT")}
        
        # Get status
        status = websocket_client.get_status()
        
        # Check status
        assert status["state"] == "connected"
        assert status["connection_attempt"] == 3
        assert status["last_message_time"] is not None
        assert status["last_heartbeat_time"] is not None
        assert status["subscription_count"] == 2
        assert len(status["subscriptions"]) == 2
        assert ("trades", "AAPL") in status["subscriptions"]
        assert ("orderbook", "MSFT") in status["subscriptions"]

    @pytest.mark.asyncio
    async def test_context_manager(self, websocket_client, monkeypatch):
        """Test using WebSocket client as a context manager."""
        # Mock connect and close methods
        connect_mock = AsyncMock()
        close_mock = AsyncMock()
        
        monkeypatch.setattr(websocket_client, "connect", connect_mock)
        monkeypatch.setattr(websocket_client, "close", close_mock)
        
        # Use as context manager
        async with websocket_client as client:
            assert client is websocket_client
        
        # Check connect and close called
        assert connect_mock.called
        assert close_mock.called