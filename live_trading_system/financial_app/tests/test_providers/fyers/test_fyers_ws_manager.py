"""
Unit Tests for Fyers WebSocket Connection Manager

This module provides comprehensive tests for the WebSocket connection manager,
including connection lifecycle, reconnection, subscription management, and error handling.
"""
import pytest
import pytest_asyncio
import asyncio
import aiohttp
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta

from app.providers.fyers.fyers_ws_manager import WebSocketConnectionManager
from app.providers.fyers.fyers_ws_protocol import (
    WebSocketConfig,
    WebSocketConnectionState,
    WebSocketMessageType,
    WebSocketError
)
from app.providers.fyers.fyers_auth import FyersAuth
from app.providers.base.provider import MarketServiceProtocol

# Test timeout in seconds
TEST_TIMEOUT = 2

# ===========================================
# FIXTURES
# ===========================================

@pytest_asyncio.fixture
async def mock_auth():
    """Mock authentication service."""
    auth = AsyncMock(spec=FyersAuth)
    auth.ensure_token.return_value = True
    auth.access_token = "test_token"
    return auth


@pytest_asyncio.fixture
async def mock_market_service():
    """Mock market service."""
    service = Mock(spec=MarketServiceProtocol)
    service.is_market_open.return_value = True
    service.get_market_state.return_value = "open"
    return service


@pytest_asyncio.fixture
async def mock_ws():
    """Mock WebSocket connection with controlled message flow."""
    ws = AsyncMock()
    ws.send_json = AsyncMock()
    
    # Create a message queue for controlled message flow
    message_queue = asyncio.Queue()
    
    async def controlled_receive():
        try:
            return await asyncio.wait_for(message_queue.get(), timeout=0.1)
        except asyncio.TimeoutError:
            # Simulate connection close on timeout
            raise aiohttp.ClientConnectionError("Connection closed")
    
    # Replace receive_json with our controlled version
    ws.receive_json = controlled_receive
    
    # Store the queue on the mock for test control
    ws.message_queue = message_queue
    
    # Add initial auth success message
    await message_queue.put({"s": "ok"})
    
    ws.close = AsyncMock()
    ws.ping = AsyncMock()
    ws.closed = False
    ws.__aenter__ = AsyncMock(return_value=ws)
    ws.__aexit__ = AsyncMock()
    return ws


@pytest_asyncio.fixture
async def mock_session(mock_ws):
    """Mock aiohttp ClientSession."""
    session = AsyncMock()
    session.ws_connect = AsyncMock(return_value=mock_ws)
    session.close = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock()
    return session


@pytest_asyncio.fixture
async def config():
    """WebSocket configuration for testing."""
    return WebSocketConfig(
        base_url="wss://test.example.com",
        max_retries=3,
        retry_delay=0.01,  # Use shorter delays for testing
        max_retry_delay=0.03,
        heartbeat_interval=0.01,
        connection_timeout=0.1
    )


@pytest_asyncio.fixture
async def manager(mock_auth, mock_market_service, config, mock_session):
    """Create WebSocket manager instance with mocked session."""
    with patch('aiohttp.ClientSession', return_value=mock_session):
        manager = WebSocketConnectionManager(
            auth=mock_auth,
            market_service=mock_market_service,
            config=config
        )
        manager.session = mock_session
        try:
            yield manager
        finally:
            # Ensure we stop all background tasks
            manager._running = False
            # Wait a bit for tasks to notice the running flag change
            await asyncio.sleep(0.1)
            # Force disconnect
            await manager.disconnect()
            # Cancel any remaining tasks
            tasks = [t for t in [manager._connection_task, manager._heartbeat_task,
                               manager._message_task, manager._subscription_task]
                    if t and not t.done()]
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)


# ===========================================
# CONNECTION TESTS
# ===========================================

@pytest.mark.asyncio
class TestConnection:
    """Test WebSocket connection management."""
    
    @pytest.fixture(autouse=True)
    async def setup_patches(self, mock_session, mock_ws):
        """Setup patches for all tests in this class."""
        with patch('aiohttp.ClientSession', return_value=mock_session), \
             patch.object(mock_session, 'ws_connect', return_value=mock_ws):
            yield
    
    async def test_successful_connection(self, manager, mock_ws, mock_session):
        """Test successful WebSocket connection."""
        try:
            await asyncio.wait_for(manager.connect(), timeout=TEST_TIMEOUT)
            
            # Verify immediate connection state
            assert manager.state == WebSocketConnectionState.CONNECTED
            assert manager.ws == mock_ws
            assert manager.retry_count == 0
            mock_session.ws_connect.assert_called_once()
            
            # Verify background tasks are running
            assert manager._heartbeat_task and not manager._heartbeat_task.done()
            assert manager._message_task and not manager._message_task.done()
            assert manager._subscription_task and not manager._subscription_task.done()
            
            # Add a test message and verify it's processed
            await mock_ws.message_queue.put({
                "type": "sf",
                "data": {"symbol": "TEST"}
            })
            
            # Small delay to allow message processing
            await asyncio.sleep(0.1)
        finally:
            # Ensure cleanup
            manager._running = False
            await asyncio.sleep(0.1)
    
    async def test_connection_auth_failure(self, manager, mock_auth):
        """Test connection failure due to authentication."""
        mock_auth.ensure_token.return_value = False
        
        with pytest.raises(WebSocketError) as exc_info:
            await asyncio.wait_for(manager.connect(), timeout=TEST_TIMEOUT)
        
        assert manager.state == WebSocketConnectionState.ERROR
        assert exc_info.value.code == -8
    
    async def test_connection_network_error(self, manager, mock_session):
        """Test connection failure due to network error."""
        mock_session.ws_connect.side_effect = aiohttp.ClientError("Network error")
        
        await asyncio.wait_for(manager.connect(), timeout=TEST_TIMEOUT)
        
        assert manager.state == WebSocketConnectionState.ERROR
        assert manager.retry_count > 0
    
    async def test_reconnection_backoff(self, manager, mock_session, mock_ws):
        """Test exponential backoff during reconnection."""
        mock_session.ws_connect.side_effect = [
            aiohttp.ClientError("Error 1"),
            aiohttp.ClientError("Error 2"),
            mock_ws  # Successful on third try
        ]
        
        start_time = datetime.utcnow()
        await asyncio.wait_for(manager.connect(), timeout=1)
        end_time = datetime.utcnow()
        
        assert manager.retry_count > 0
        assert (end_time - start_time) > timedelta(seconds=0.02)  # At least 2 retries
    
    async def test_max_retries_exceeded(self, manager, mock_session):
        """Test behavior when max retries is exceeded."""
        mock_session.ws_connect.side_effect = aiohttp.ClientError("Network error")
        
        await asyncio.wait_for(manager.connect(), timeout=1)
        
        assert manager.retry_count >= manager.config.max_retries
        assert manager.state == WebSocketConnectionState.ERROR


# ===========================================
# SUBSCRIPTION TESTS
# ===========================================

@pytest.mark.asyncio
class TestSubscription:
    """Test subscription management."""
    
    @pytest.fixture(autouse=True)
    def setup_patches(self, mock_session, mock_ws):
        """Setup patches for all tests in this class."""
        with patch('aiohttp.ClientSession', return_value=mock_session), \
             patch.object(mock_session, 'ws_connect', return_value=mock_ws):
            yield
    
    async def test_subscribe_success(self, manager, mock_ws):
        """Test successful subscription."""
        manager.ws = mock_ws
        manager.state = WebSocketConnectionState.CONNECTED
        manager._running = True
        
        # Start subscription task
        manager._subscription_task = asyncio.create_task(manager._subscription_loop())
        
        try:
            symbols = ["NSE:SBIN-EQ", "NSE:RELIANCE-EQ"]
            await manager.subscribe(symbols)
            
            # Wait for subscription to be processed
            await asyncio.sleep(0.1)
            
            mock_ws.send_json.assert_called_once()
            assert all(symbol in manager.active_subscriptions for symbol in symbols)
        finally:
            manager._running = False
            if manager._subscription_task:
                manager._subscription_task.cancel()
                try:
                    await manager._subscription_task
                except asyncio.CancelledError:
                    pass
    
    async def test_subscribe_limit_exceeded(self, manager):
        """Test subscription limit enforcement."""
        symbols = [f"NSE:STOCK{i}-EQ" for i in range(6000)]  # Exceeds 5000 limit
        
        with pytest.raises(ValueError):
            await asyncio.wait_for(manager.subscribe(symbols), timeout=1)
    
    async def test_unsubscribe_success(self, manager, mock_ws):
        """Test successful unsubscription."""
        manager.ws = mock_ws
        manager.state = WebSocketConnectionState.CONNECTED
        manager.active_subscriptions.add("NSE:SBIN-EQ")
        
        await asyncio.wait_for(
            asyncio.gather(
                manager.unsubscribe(["NSE:SBIN-EQ"]),
                asyncio.sleep(0.1)  # Allow unsubscription to process
            ),
            timeout=1
        )
        
        assert "NSE:SBIN-EQ" not in manager.active_subscriptions
        mock_ws.send_json.assert_called_once()
    
    async def test_subscription_queue(self, manager, mock_ws):
        """Test subscription queue processing."""
        manager.ws = mock_ws
        manager.state = WebSocketConnectionState.CONNECTED
        manager._running = True
        
        # Start subscription task
        manager._subscription_task = asyncio.create_task(manager._subscription_loop())
        
        try:
            symbols1 = ["NSE:SBIN-EQ"]
            symbols2 = ["NSE:RELIANCE-EQ"]
            
            await asyncio.gather(
                manager.subscribe(symbols1),
                manager.subscribe(symbols2)
            )
            
            # Wait for subscriptions to be processed
            await asyncio.sleep(0.1)
            
            assert mock_ws.send_json.call_count == 2
            assert all(symbol in manager.active_subscriptions 
                      for symbol in symbols1 + symbols2)
        finally:
            manager._running = False
            if manager._subscription_task:
                manager._subscription_task.cancel()
                try:
                    await manager._subscription_task
                except asyncio.CancelledError:
                    pass


# ===========================================
# MESSAGE HANDLING TESTS
# ===========================================

class TestMessageHandling:
    """Test message handling and routing."""
    
    @pytest.fixture(autouse=True)
    def setup_patches(self, mock_session, mock_ws):
        """Setup patches for all tests in this class."""
        with patch('aiohttp.ClientSession', return_value=mock_session), \
             patch.object(mock_session, 'ws_connect', return_value=mock_ws):
            yield
    
    @pytest.mark.asyncio
    async def test_message_routing(self, manager):
        """Test message routing to handlers."""
        handler = AsyncMock()
        manager.add_message_handler(WebSocketMessageType.SYMBOL_FEED, handler)
        
        message = {
            "type": "sf",
            "data": {"symbol": "NSE:SBIN-EQ"}
        }
        
        await asyncio.wait_for(manager._process_message(message), timeout=1)
        
        handler.assert_called_once_with(message)
    
    @pytest.mark.asyncio
    async def test_multiple_handlers(self, manager):
        """Test multiple handlers for same message type."""
        handler1 = AsyncMock()
        handler2 = AsyncMock()
        manager.add_message_handler(WebSocketMessageType.SYMBOL_FEED, handler1)
        manager.add_message_handler(WebSocketMessageType.SYMBOL_FEED, handler2)
        
        message = {
            "type": "sf",
            "data": {"symbol": "NSE:SBIN-EQ"}
        }
        
        await asyncio.wait_for(manager._process_message(message), timeout=1)
        
        handler1.assert_called_once_with(message)
        handler2.assert_called_once_with(message)
    
    @pytest.mark.asyncio
    async def test_handler_error(self, manager):
        """Test handler error handling."""
        handler = AsyncMock(side_effect=Exception("Handler error"))
        manager.add_message_handler(WebSocketMessageType.SYMBOL_FEED, handler)
        
        message = {
            "type": "sf",
            "data": {"symbol": "NSE:SBIN-EQ"}
        }
        
        await asyncio.wait_for(manager._process_message(message), timeout=1)
    
    def test_remove_handler(self, manager):
        """Test handler removal."""
        handler = AsyncMock()
        manager.add_message_handler(WebSocketMessageType.SYMBOL_FEED, handler)
        manager.remove_message_handler(WebSocketMessageType.SYMBOL_FEED, handler)
        
        assert handler not in manager.message_handlers.get(WebSocketMessageType.SYMBOL_FEED, [])


# ===========================================
# HEARTBEAT AND MAINTENANCE TESTS
# ===========================================

@pytest.mark.asyncio
class TestHeartbeat:
    """Test heartbeat and connection maintenance."""
    
    @pytest.fixture(autouse=True)
    def setup_patches(self, mock_session, mock_ws):
        """Setup patches for all tests in this class."""
        with patch('aiohttp.ClientSession', return_value=mock_session), \
             patch.object(mock_session, 'ws_connect', return_value=mock_ws):
            yield
    
    async def test_heartbeat_success(self, manager, mock_ws):
        """Test successful heartbeat."""
        manager.ws = mock_ws
        manager.state = WebSocketConnectionState.CONNECTED
        manager._running = True
        
        # Start heartbeat task
        manager._heartbeat_task = asyncio.create_task(manager._heartbeat_loop())
        
        try:
            # Wait for heartbeat to be sent
            await asyncio.sleep(manager.config.heartbeat_interval * 2)
            
            assert mock_ws.ping.called
            assert (datetime.utcnow() - manager.last_heartbeat) < timedelta(seconds=1)
        finally:
            manager._running = False
            if manager._heartbeat_task:
                manager._heartbeat_task.cancel()
                try:
                    await manager._heartbeat_task
                except asyncio.CancelledError:
                    pass

    async def test_missed_heartbeat(self, manager, mock_ws):
        """Test missed heartbeat detection."""
        manager.ws = mock_ws
        manager.state = WebSocketConnectionState.CONNECTED
        manager._running = True
        manager.last_heartbeat = datetime.utcnow() - timedelta(seconds=100)
        
        # Mock ping to raise an error
        mock_ws.ping.side_effect = aiohttp.ClientError("Connection lost")
        
        # Start heartbeat task
        manager._heartbeat_task = asyncio.create_task(manager._heartbeat_loop())
        
        try:
            # Wait for heartbeat to fail
            await asyncio.sleep(manager.config.heartbeat_interval * 2)
            
            assert manager.state == WebSocketConnectionState.RECONNECTING
            assert manager.retry_count > 0
        finally:
            manager._running = False
            if manager._heartbeat_task:
                manager._heartbeat_task.cancel()
                try:
                    await manager._heartbeat_task
                except asyncio.CancelledError:
                    pass


# ===========================================
# MARKET AWARENESS TESTS
# ===========================================

@pytest.mark.asyncio
class TestMarketAwareness:
    """Test market-aware features."""
    
    @pytest.fixture(autouse=True)
    def setup_patches(self, mock_session, mock_ws):
        """Setup patches for all tests in this class."""
        with patch('aiohttp.ClientSession', return_value=mock_session), \
             patch.object(mock_session, 'ws_connect', return_value=mock_ws):
            yield
    
    async def test_market_open_reconnection(self, manager, mock_market_service, mock_session):
        """Test faster reconnection during market hours."""
        mock_market_service.is_market_open.return_value = True
        manager.retry_count = 0
        manager.config.max_retries = 5  # Ensure we have enough retries
        
        # Add a test symbol to active subscriptions
        test_symbol = "NSE:SBIN-EQ"
        manager.active_subscriptions.add(test_symbol)
        
        # Mock connection failure
        mock_session.ws_connect.side_effect = aiohttp.ClientError("Test error")
        
        # Force a reconnection
        manager.state = WebSocketConnectionState.CONNECTED  # Set initial state
        await manager._handle_reconnection()
        
        # Verify market-aware behavior
        assert mock_market_service.is_market_open.called
        mock_market_service.is_market_open.assert_called_with(test_symbol)

    async def test_market_closed_reconnection(self, manager, mock_market_service, mock_session):
        """Test slower reconnection outside market hours."""
        mock_market_service.is_market_open.return_value = False
        manager.retry_count = 0
        manager.config.max_retries = 5  # Ensure we have enough retries
        
        # Add a test symbol to active subscriptions
        test_symbol = "NSE:SBIN-EQ"
        manager.active_subscriptions.add(test_symbol)
        
        # Mock connection failure
        mock_session.ws_connect.side_effect = aiohttp.ClientError("Test error")
        
        # Force a reconnection
        manager.state = WebSocketConnectionState.CONNECTED  # Set initial state
        await manager._handle_reconnection()
        
        # Verify market-aware behavior
        assert mock_market_service.is_market_open.called
        mock_market_service.is_market_open.assert_called_with(test_symbol)


# ===========================================
# METRICS AND MONITORING TESTS
# ===========================================

class TestMetrics:
    """Test performance metrics and monitoring."""
    
    @pytest.fixture(autouse=True)
    def setup_patches(self, mock_session, mock_ws):
        """Setup patches for all tests in this class."""
        with patch('aiohttp.ClientSession', return_value=mock_session), \
             patch.object(mock_session, 'ws_connect', return_value=mock_ws):
            yield
    
    def test_connection_metrics(self, manager):
        """Test connection metrics tracking."""
        stats = manager.get_connection_stats()
        
        assert "state" in stats
        assert "retry_count" in stats
        assert "active_subscriptions" in stats
        assert "metrics" in stats
        assert "messages_received" in stats["metrics"]
    
    @pytest.mark.asyncio
    async def test_message_metrics(self, manager):
        """Test message processing metrics."""
        # Add a test handler to ensure message is processed
        test_handler = AsyncMock()
        manager.add_message_handler(WebSocketMessageType.SYMBOL_FEED, test_handler)
        
        initial_count = manager.metrics["messages_processed"]
        
        # Process a test message
        await manager._process_message({
            "type": WebSocketMessageType.SYMBOL_FEED,
            "data": {"symbol": "NSE:SBIN-EQ"}
        })
        
        # Verify metrics and handler call
        assert manager.metrics["messages_processed"] == initial_count + 1
        test_handler.assert_called_once()


# ===========================================
# CLEANUP AND RESOURCE MANAGEMENT
# ===========================================

@pytest_asyncio.fixture(autouse=True)
async def cleanup_tasks():
    """Cleanup any remaining tasks after each test."""
    try:
        yield
    finally:
        # Get all tasks except the current one
        tasks = [t for t in asyncio.all_tasks() 
                if t is not asyncio.current_task() and not t.done()]
        
        # Cancel all tasks
        for task in tasks:
            task.cancel()
        
        # Wait for all tasks to complete with a timeout
        if tasks:
            try:
                async with asyncio.timeout(1.0):  # 1 second timeout for cleanup
                    await asyncio.gather(*tasks, return_exceptions=True)
            except asyncio.TimeoutError:
                print("Warning: Some tasks did not cleanup within timeout")


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 