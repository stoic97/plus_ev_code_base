"""
Fyers Provider Implementation

This module implements the BaseProvider abstract class for Fyers broker,
providing market data access, real-time streaming, and authentication.
"""

import logging
import asyncio
import json
import time
import os
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Type
from datetime import datetime, timedelta
from urllib.parse import urlparse, parse_qs

from app.providers.base.provider import (
    BaseProvider, ConnectionState, SubscriptionType,
    ProviderError, ConnectionError, AuthenticationError, RateLimitError, DataNotFoundError
)
from app.providers.config.provider_settings import (
    FyersSettings, get_settings_for_provider, ProviderType
)
from app.providers.fyers.fyers_rest_client import FyersRestClient
from app.providers.fyers.fyers_websocket_client import FyersWebSocketClient
from app.providers.fyers.fyers_auth import FyersAuth
from app.providers.fyers.fyers_mapper import map_interval, map_symbol, transform_ohlcv, transform_orderbook, transform_quote

# Configure logging
logger = logging.getLogger(__name__)


class FyersProvider(BaseProvider):
    """
    Fyers market data provider implementation.
    
    This provider connects to Fyers API to obtain market data,
    subscribe to real-time updates, and handle authentication.
    """
    
    def __init__(
        self, 
        settings: Optional[FyersSettings] = None, 
        provider_name: str = "fyers",
        auth_mode: str = "auto"
    ):
        """
        Initialize Fyers provider.
        
        Args:
            settings: Fyers-specific settings (optional)
            provider_name: Name identifier for the provider
            auth_mode: Authentication mode ("auto", "headless", or "manual")
        """
        # Initialize base provider
        super().__init__(
            settings or get_settings_for_provider(ProviderType.FYERS), 
            provider_name
        )
        
        # Authentication and connection state
        self.auth_mode = auth_mode
        self._auth_manager: Optional[FyersAuth] = None
        self._rest_client: Optional[FyersRestClient] = None
        self._ws_client: Optional[FyersWebSocketClient] = None
        self._connected = False
        
        # For authentication callback server
        self._auth_server = None
        
        # Rate limiting settings
        self._setup_rate_limiters()
        
        logger.info(f"Fyers provider initialized with auth mode: {auth_mode}")
    
    def _setup_rate_limiters(self) -> None:
        """Set up specialized rate limiters for different endpoints"""
        # Specialized rate limiters for different endpoints
        settings = self.settings
        if hasattr(settings, 'MARKET_DEPTH_RATE_LIMIT'):
            self.market_depth_limiter = RateLimiter(calls_per_second=settings.MARKET_DEPTH_RATE_LIMIT)
        
        if hasattr(settings, 'HISTORICAL_DATA_RATE_LIMIT'):
            self.historical_data_limiter = RateLimiter(calls_per_second=settings.HISTORICAL_DATA_RATE_LIMIT)
        
        if hasattr(settings, 'QUOTES_RATE_LIMIT'):
            self.quotes_rate_limiter = RateLimiter(calls_per_second=settings.QUOTES_RATE_LIMIT)
    
    async def connect(self) -> None:
        """
        Establish connection to Fyers API with authentication.
        
        This method handles the authentication flow based on the configured
        auth_mode and initializes the REST and WebSocket clients.
        
        Raises:
            ConnectionError: If connection fails
            AuthenticationError: If authentication fails
        """
        try:
            self.connection_state = ConnectionState.CONNECTING
            
            # Initialize auth manager if needed
            if not self._auth_manager:
                self._auth_manager = FyersAuth(self.settings)
            
            # Check if we have a valid token already
            if self._auth_manager.has_valid_token():
                logger.info("Using existing valid token")
                await self._init_clients()
                return
            
            # Try authentication based on mode
            if self.auth_mode == "auto":
                # Try headless first, fall back to manual if it fails
                try:
                    await self._authenticate_headless()
                except Exception as e:
                    logger.warning(f"Headless authentication failed: {e}")
                    await self._authenticate_manual()
            elif self.auth_mode == "headless":
                await self._authenticate_headless()
            else:  # manual mode
                await self._authenticate_manual()
            
            # Initialize clients after authentication
            await self._init_clients()
            
        except Exception as e:
            self.connection_state = ConnectionState.ERROR
            raise ConnectionError(f"Failed to connect to Fyers: {e}")
    
    async def disconnect(self) -> None:
        """
        Disconnect from Fyers API.
        
        Closes WebSocket connections and cleans up resources.
        """
        logger.info("Disconnecting from Fyers API")
        
        # Cleanup authentication server if active
        if self._auth_server:
            await self._cleanup_auth_server()
        
        # Close WebSocket connection if active
        if self._ws_client:
            await self._ws_client.close()
            self._ws_client = None
        
        # Update state
        self._connected = False
        self.connection_state = ConnectionState.DISCONNECTED
        
        logger.info("Disconnected from Fyers API")
    
    async def _authenticate_headless(self) -> None:
        """
        Authenticate using headless browser automation.
        
        Uses Selenium to automate the login process without user interaction.
        
        Raises:
            AuthenticationError: If headless authentication fails
        """
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.chrome.service import Service
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
            from webdriver_manager.chrome import ChromeDriverManager
            
            logger.info("Starting headless browser authentication")
            
            # Create the auth URL
            auth_url = self._auth_manager.get_auth_url()
            
            # Set up headless browser
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            
            # Set up the driver with automatic ChromeDriver management
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=chrome_options)
            
            try:
                # Navigate to auth URL
                driver.get(auth_url)
                logger.debug(f"Navigated to auth URL: {auth_url}")
                
                # Wait for login form and fill it
                WebDriverWait(driver, 30).until(
                    EC.presence_of_element_located((By.ID, "fyers_id"))
                )
                
                logger.debug("Login form loaded, filling credentials")
                
                # Fill username
                username_field = driver.find_element(By.ID, "fyers_id")
                username_field.clear()
                username_field.send_keys(self.settings.USERNAME)
                
                # Fill password
                password_field = driver.find_element(By.ID, "password")
                password_field.clear()
                password_field.send_keys(self.settings.PASSWORD.get_secret_value())
                
                # Submit form
                submit_button = driver.find_element(By.ID, "submit-login-btn")
                submit_button.click()
                
                logger.debug("Login form submitted, waiting for redirect")
                
                # Wait for potential 2FA if enabled
                try:
                    totp_field = WebDriverWait(driver, 5).until(
                        EC.presence_of_element_located((By.ID, "totp"))
                    )
                    
                    if totp_field:
                        logger.debug("2FA detected, waiting for user input")
                        if not hasattr(self.settings, 'TOTP_KEY'):
                            print("\n" + "="*80)
                            print("FYERS 2FA AUTHENTICATION REQUIRED")
                            print("="*80)
                            print("\nPlease enter the 2FA code from your authenticator app:")
                            totp_code = input("TOTP Code: ").strip()
                        else:
                            # Use pyotp if available and TOTP key is configured
                            import pyotp
                            totp = pyotp.TOTP(self.settings.TOTP_KEY.get_secret_value())
                            totp_code = totp.now()
                            
                        totp_field.send_keys(totp_code)
                        verify_button = driver.find_element(By.ID, "verify-totp-btn")
                        verify_button.click()
                except Exception as e:
                    logger.debug(f"No 2FA detected or error: {e}")
                
                # Wait for redirect to our callback URL
                logger.debug(f"Waiting for redirect to: {self.settings.REDIRECT_URI}")
                WebDriverWait(driver, 60).until(
                    lambda d: self.settings.REDIRECT_URI in d.current_url
                )
                
                # Extract auth code from URL
                current_url = driver.current_url
                logger.debug(f"Redirected to: {current_url}")
                auth_code = self._extract_auth_code(current_url)
                
                if not auth_code:
                    raise AuthenticationError("Failed to extract auth code from redirect URL")
                
                logger.debug("Auth code extracted successfully")
                
                # Exchange auth code for access token
                await self._auth_manager.get_access_token_from_code(auth_code)
                logger.info("Headless authentication successful")
                
            finally:
                # Clean up
                driver.quit()
                
        except ImportError as e:
            logger.error(f"Required package not installed: {e}")
            raise AuthenticationError(f"Required package not installed for headless auth: {e}")
        except Exception as e:
            logger.error(f"Headless authentication failed: {e}", exc_info=True)
            raise AuthenticationError(f"Headless authentication failed: {e}")
    
    async def _authenticate_manual(self) -> None:
        """
        Fallback to manual authentication with user input.
        
        Provides instructions for the user to complete the authentication
        flow manually in a browser, with options for automatic callback
        or manual code entry.
        
        Raises:
            AuthenticationError: If manual authentication fails
        """
        logger.info("Starting manual authentication process")
        
        # Start a local server to catch the redirect if needed
        await self._start_auth_callback_server()
        
        # Get the auth URL
        auth_url = self._auth_manager.get_auth_url()
        
        # Print instructions for the user
        print("\n" + "="*80)
        print("FYERS MANUAL AUTHENTICATION REQUIRED")
        print("="*80)
        print("\nPlease follow these steps to authenticate:")
        print("1. Open this URL in your browser:")
        print(f"\n{auth_url}\n")
        print("2. Log in with your Fyers credentials")
        print("3. Authorize the application when prompted")
        
        # Option 1: Wait for callback to our local server
        auth_code = None
        if self._auth_server:
            print("\nWaiting for redirect to complete automatically...")
            try:
                auth_code = await asyncio.wait_for(self._auth_server.wait_for_code(), 
                                                  timeout=300)
                print("Received auth code automatically.")
            except asyncio.TimeoutError:
                print("\nTimeout waiting for automatic redirect.")
        
        # Option 2: Manual input as fallback
        if not auth_code:
            print("\nAfter authorization, you will be redirected to a URL.")
            print("Look for the 'code=' parameter in that URL.")
            print("Example: https://your-redirect-uri/?code=eyJ0eX...&state=...")
            print("\nCopy the FULL URL and paste it below:")
            
            redirect_url = input("\nEnter the full redirect URL: ").strip()
            auth_code = self._extract_auth_code(redirect_url)
        
        if not auth_code:
            raise AuthenticationError("Failed to get valid authorization code")
        
        # Exchange the code for an access token
        await self._auth_manager.get_access_token_from_code(auth_code)
        print("\nAuthentication successful!")
        logger.info("Manual authentication successful")
    
    def _extract_auth_code(self, url: str) -> Optional[str]:
        """
        Extract auth code from redirect URL.
        
        Args:
            url: Redirect URL containing the auth code
        
        Returns:
            Extracted auth code or None if not found
        """
        try:
            parsed_url = urlparse(url)
            query_params = parse_qs(parsed_url.query)
            
            if 'code' in query_params:
                return query_params['code'][0]
            return None
        except Exception as e:
            logger.error(f"Failed to extract auth code: {e}")
            return None
    
    async def _start_auth_callback_server(self) -> None:
        """
        Start a local web server to handle the OAuth callback.
        
        This creates a temporary HTTP server that listens for the redirect
        from Fyers after successful authentication, extracting the auth code.
        """
        try:
            # Only import if needed
            from aiohttp import web
            
            # Define the code future
            code_future = asyncio.Future()
            
            # Create a handler for the callback
            async def callback_handler(request):
                code = request.query.get('code')
                if code:
                    if not code_future.done():
                        code_future.set_result(code)
                    return web.Response(text="Authentication successful! You can close this window.")
                return web.Response(text="Authentication failed. No authorization code received.")
            
            # Create the app and add the route
            app = web.Application()
            app.router.add_get('/', callback_handler)
            
            # Parse the redirect URI to get the port
            parsed_uri = urlparse(self.settings.REDIRECT_URI)
            host = parsed_uri.hostname or 'localhost'
            port = parsed_uri.port or 8000
            
            # Start the server
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, host, port)
            await site.start()
            
            # Set up server object for cleanup
            self._auth_server = SimpleNamespace(
                runner=runner,
                wait_for_code=lambda: code_future
            )
            
            logger.info(f"Auth callback server started on {host}:{port}")
            
        except ImportError:
            logger.warning("aiohttp not installed. Manual code entry will be required.")
            self._auth_server = None
        except Exception as e:
            logger.error(f"Failed to start callback server: {e}")
            self._auth_server = None
    
    async def _cleanup_auth_server(self) -> None:
        """Clean up the auth callback server"""
        if self._auth_server and hasattr(self._auth_server, 'runner'):
            await self._auth_server.runner.cleanup()
            self._auth_server = None
    
    async def _init_clients(self) -> None:
        """
        Initialize REST and WebSocket clients with authenticated token.
        
        Sets up the API clients with the proper authentication and
        verifies the connection is working.
        """
        # Get authentication headers
        auth_headers = self._auth_manager.get_auth_headers()
        
        # Initialize REST client
        self._rest_client = FyersRestClient(
            base_url=self.settings.API_BASE_URL,
            settings=self.settings,
            headers=auth_headers
        )
        
        # Check connection with a basic API call
        await self.health_check()
        
        # Update state
        self._connected = True
        self.connection_state = ConnectionState.CONNECTED
        logger.info("Fyers API clients initialized successfully")
    
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
        # Ensure connection
        if not self._connected:
            await self.connect()
        
        # Format parameters for Fyers API
        formatted_symbol = self.format_symbol(symbol)
        formatted_interval = map_interval(interval)
        
        # Default to current time if end_time not provided
        if not end_time:
            end_time = datetime.now()
        
        # Default start_time based on interval if not provided
        if not start_time:
            if interval.lower() in ['1d', 'd']:
                # For daily candles, default to 100 days
                start_time = end_time - timedelta(days=100)
            else:
                # For intraday candles, default to 5 days
                start_time = end_time - timedelta(days=5)
        
        # Format dates as required by the API
        from_date = start_time.strftime("%Y-%m-%d")
        to_date = end_time.strftime("%Y-%m-%d")
        
        # Prepare parameters for the API call
        params = {
            "symbol": formatted_symbol,
            "resolution": formatted_interval,
            "date_format": 1,  # Use date format
            "range_from": from_date,
            "range_to": to_date,
            "cont_flag": 1  # For continuous data
        }
        
        if hasattr(self, 'historical_data_limiter'):
            # Acquire rate limiting token for historical data
            await self.historical_data_limiter.acquire()
        
        # Make the API call with error handling and retries
        try:
            response = await self.with_retries(
                lambda: self._rest_client.get("data/history", params)
            )
            
            # Check for error response
            if response.get("s") == "error":
                error_msg = response.get("message", "Unknown error")
                raise DataNotFoundError(f"Failed to get OHLCV data: {error_msg}")
            
            # Transform to internal format
            result = transform_ohlcv(response)
            
            # Apply limit if specified
            if limit and len(result) > limit:
                result = result[-limit:]
            
            return result
            
        except Exception as e:
            if isinstance(e, DataNotFoundError):
                raise
            logger.error(f"Failed to get OHLCV data: {e}", exc_info=True)
            raise ProviderError(f"Failed to get OHLCV data: {e}")
    
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
        # Ensure connection
        if not self._connected:
            await self.connect()
        
        # Format symbol for Fyers API
        formatted_symbol = self.format_symbol(symbol)
        
        # Prepare parameters
        params = {
            "symbol": formatted_symbol,
            "ohlcv_flag": 1  # Include OHLCV data
        }
        
        if hasattr(self, 'market_depth_limiter'):
            # Acquire rate limiting token for market depth
            await self.market_depth_limiter.acquire()
        
        # Make the API call with error handling and retries
        try:
            response = await self.with_retries(
                lambda: self._rest_client.get("data/depth", params)
            )
            
            # Check for error response
            if response.get("s") == "error":
                error_msg = response.get("message", "Unknown error")
                raise DataNotFoundError(f"Failed to get orderbook data: {error_msg}")
            
            # Transform to internal format
            result = transform_orderbook(response, symbol, depth)
            return result
            
        except Exception as e:
            if isinstance(e, DataNotFoundError):
                raise
            logger.error(f"Failed to get orderbook data: {e}", exc_info=True)
            raise ProviderError(f"Failed to get orderbook data: {e}")
    
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
        # Ensure connection
        if not self._connected:
            await self.connect()
        
        # Note: Fyers doesn't have a dedicated API for trade history
        # We'll return an empty list for now and implement later if needed
        logger.warning("Trade history API not available in Fyers")
        return []
    
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
        # Ensure connection
        if not self._connected:
            await self.connect()
        
        # Format symbol for Fyers API
        formatted_symbol = self.format_symbol(symbol)
        
        # Prepare parameters
        params = {
            "symbols": formatted_symbol
        }
        
        if hasattr(self, 'quotes_rate_limiter'):
            # Acquire rate limiting token for quotes
            await self.quotes_rate_limiter.acquire()
        
        # Make the API call with error handling and retries
        try:
            response = await self.with_retries(
                lambda: self._rest_client.get("data/quotes", params)
            )
            
            # Check for error response
            if response.get("s") == "error":
                error_msg = response.get("message", "Unknown error")
                raise DataNotFoundError(f"Failed to get quote data: {error_msg}")
            
            # Transform to internal format
            result = transform_quote(response, symbol)
            return result
            
        except Exception as e:
            if isinstance(e, DataNotFoundError):
                raise
            logger.error(f"Failed to get quote data: {e}", exc_info=True)
            raise ProviderError(f"Failed to get quote data: {e}")
    
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
        # Ensure connection
        if not self._connected:
            await self.connect()
        
        # Initialize WebSocket if needed
        if not self._ws_client:
            await self._init_websocket()
        
        # Format symbol for Fyers API
        formatted_symbol = self.format_symbol(symbol)
        
        # Set up message handler for this data type
        self._ws_client.set_message_handler("trade", callback)
        
        # Subscribe to the symbol
        await self._ws_client.subscribe_to_trades(formatted_symbol)
        
        # Add to active subscriptions
        self.active_subscriptions.add((SubscriptionType.TRADES, symbol))
        logger.info(f"Subscribed to trades for {symbol}")
    
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
        # Ensure connection
        if not self._connected:
            await self.connect()
        
        # Initialize WebSocket if needed
        if not self._ws_client:
            await self._init_websocket()
        
        # Format symbol for Fyers API
        formatted_symbol = self.format_symbol(symbol)
        
        # Set up message handler for this data type
        self._ws_client.set_message_handler("orderbook", callback)
        
        # Use configured depth or default
        actual_depth = depth or self.settings.ORDERBOOK_DEPTH
        
        # Subscribe to the symbol
        await self._ws_client.subscribe_to_orderbook(formatted_symbol, actual_depth)
        
        # Add to active subscriptions
        self.active_subscriptions.add((SubscriptionType.ORDERBOOK, symbol))
        logger.info(f"Subscribed to orderbook for {symbol} with depth {actual_depth}")
    
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
        # Ensure connection
        if not self._connected:
            await self.connect()
        
        # Initialize WebSocket if needed
        if not self._ws_client:
            await self._init_websocket()
        
        # Format symbol for Fyers API
        formatted_symbol = self.format_symbol(symbol)
        
        # Set up message handler for this data type
        self._ws_client.set_message_handler("quotes", callback)
        
        # Subscribe to the symbol
        await self._ws_client.subscribe_to_quotes(formatted_symbol)
        
        # Add to active subscriptions
        self.active_subscriptions.add((SubscriptionType.QUOTES, symbol))
        logger.info(f"Subscribed to quotes for {symbol}")
    
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
        # Check if WebSocket is initialized
        if not self._ws_client:
            logger.warning(f"No active WebSocket connection to unsubscribe {symbol}")
            return
        
        # Format symbol for Fyers API
        formatted_symbol = self.format_symbol(symbol)
        
        # Unsubscribe based on subscription type
        try:
            await self._ws_client.unsubscribe(formatted_symbol, str(subscription_type.value))
            
            # Remove from active subscriptions
            self.active_subscriptions.discard((subscription_type, symbol))
            logger.info(f"Unsubscribed from {subscription_type.value} for {symbol}")
            
        except Exception as e:
            logger.error(f"Failed to unsubscribe {symbol}: {e}")
            raise ProviderError(f"Failed to unsubscribe: {e}")
    
    async def _init_websocket(self) -> None:
        """
        Initialize the WebSocket connection.
        
        Sets up the WebSocket client with proper authentication and
        message handlers.
        """
        # Check if auth manager is set up
        if not self._auth_manager:
            raise ProviderError("Authentication manager not initialized")
        
        # Get access token for WebSocket
        access_token = self._auth_manager.access_token
        
        # Create WebSocket client
        self._ws_client = FyersWebSocketClient(
            url=self.settings.WEBSOCKET_URL,
            settings=self.settings,
            headers={
                "Authorization": f"{self.settings.APP_ID}:{access_token}"
            },
            connection_hooks={
                "on_connect": self._on_ws_connect,
                "on_close": self._on_ws_close
            },
            message_handlers={
                "default": self._handle_default_message
            }
        )
        
        # Connect to WebSocket
        await self._ws_client.connect()
        logger.info("WebSocket connection initialized")
    
    async def _on_ws_connect(self, client) -> None:
        """
        Callback for WebSocket connection established.
        
        Args:
            client: WebSocket client instance
        """
        logger.info("WebSocket connection established")
        
        # Resubscribe to active subscriptions if any
        if self.active_subscriptions:
            await self._resubscribe()
    
    async def _on_ws_close(self, client) -> None:
        """
        Callback for WebSocket connection closed.
        
        Args:
            client: WebSocket client instance
        """
        logger.info("WebSocket connection closed")
    
    async def _resubscribe(self) -> None:
        """
        Resubscribe to all active channels after reconnection.
        
        Restores all subscriptions after a WebSocket reconnection.
        """
        if not self.active_subscriptions:
            logger.debug("No subscriptions to restore")
            return
        
        logger.info(f"Resubscribing to {len(self.active_subscriptions)} channels")
        
        # Make a copy of subscriptions to avoid modification during iteration
        subscriptions = list(self.active_subscriptions)
        
        # Resubscribe to all channels
        for sub_type, symbol in subscriptions:
            try:
                if sub_type == SubscriptionType.TRADES:
                    await self._ws_client.subscribe_to_trades(self.format_symbol(symbol))
                elif sub_type == SubscriptionType.ORDERBOOK:
                    await self._ws_client.subscribe_to_orderbook(self.format_symbol(symbol))
                elif sub_type == SubscriptionType.QUOTES:
                    await self._ws_client.subscribe_to_quotes(self.format_symbol(symbol))
                else:
                    logger.warning(f"Unknown subscription type for resubscription: {sub_type}")
            except Exception as e:
                logger.error(f"Failed to resubscribe to {sub_type}/{symbol}: {e}")
    
    def _handle_default_message(self, message: Dict[str, Any]) -> None:
        """
        Default handler for WebSocket messages.
        
        Args:
            message: Message received from WebSocket
        """
        logger.debug(f"Received unhandled WebSocket message: {message}")
    
    def format_symbol(self, symbol: str) -> str:
        """
        Format a symbol into the provider's expected format.
        
        Args:
            symbol: Input symbol
            
        Returns:
            Formatted symbol
        """
        return map_symbol(symbol)
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check the health of the provider connection.
        
        Returns:
            Health status information
            
        Raises:
            ConnectionError: If connection check fails
        """
        try:
            # Check if we have a REST client
            if not self._rest_client:
                return {
                    "status": "error",
                    "message": "REST client not initialized",
                    "rest_connected": False,
                    "websocket_connected": False
                }
            
            # Make a simple API call to verify connection
            profile = await self.with_retries(
                lambda: self._rest_client.get("profile")
            )
            
            # Check if the response is valid
            if profile.get("s") != "ok" or "data" not in profile:
                return {
                    "status": "error",
                    "message": "Failed to get profile data",
                    "rest_connected": False,
                    "websocket_connected": self._ws_client is not None and self._ws_client.is_connected()
                }
            
            # Check WebSocket status if initialized
            ws_status = self._ws_client is not None and self._ws_client.is_connected()
            
            return {
                "status": "ok",
                "rest_connected": True,
                "websocket_connected": ws_status,
                "user_id": profile.get("data", {}).get("fy_id", "unknown"),
                "user_name": profile.get("data", {}).get("name", "unknown")
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}", exc_info=True)
            return {
                "status": "error",
                "message": str(e),
                "rest_connected": False,
                "websocket_connected": False
            }