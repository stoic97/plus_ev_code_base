"""
Fyers Authentication Module

This module handles authentication with the Fyers API, including token
acquisition, validation, management and expiry monitoring using ORM models.
"""

import logging
import asyncio
import base64
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, Callable, Union, List
from urllib.parse import urlencode

import aiohttp
from pydantic import SecretStr
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from app.providers.base.provider import AuthenticationError, ConnectionError, RateLimitError
from app.providers.fyers.fyers_settings import FyersSettings
from app.models.fyers_tokens import FyersToken
from app.core.database import get_db

# Set up logging
logger = logging.getLogger(__name__)

# Authentication endpoints and parameters
AUTH_ENDPOINT = "validate-authcode"
AUTHORIZE_ENDPOINT = "generate-authcode"
VALIDATE_TOKEN_ENDPOINT = "validate-token"
TOKEN_GRANT_TYPE = "authorization_code"
REFRESH_GRANT_TYPE = "refresh_token"
TOKEN_PARAM_CLIENT_ID = "client_id"
TOKEN_PARAM_REDIRECT_URI = "redirect_uri"
TOKEN_PARAM_RESPONSE_TYPE = "response_type"
TOKEN_PARAM_STATE = "state"
RESPONSE_TYPE = "code"

# Error response codes
ERROR_EXPIRED_TOKEN = -8
ERROR_INVALID_TOKEN = -15
ERROR_AUTH_FAILED = -16
ERROR_TOKEN_ERROR = -17


class FyersAuth:
    """Authentication handler for Fyers API using ORM models."""
    
    def __init__(self, settings: FyersSettings, session_factory=None, discord_webhooks: List[str] = None):
        """
        Initialize Fyers authentication handler.
        
        Args:
            settings: Fyers-specific settings
            session_factory: Database session factory (defaults to get_db)
            discord_webhooks: Optional list of Discord webhook URLs for notifications
        """
        self.settings = settings
        self.access_token: Optional[str] = None
        self.refresh_token_value: Optional[str] = None  # Renamed to avoid conflict
        self.token_expiry: Optional[datetime] = None
        self._lock = asyncio.Lock()  # Thread safety for token operations
        self._session: Optional[aiohttp.ClientSession] = None
        
        # Database session factory
        self.session_factory = session_factory or get_db
        
        # Discord webhook URLs for notifications
        self.discord_webhooks = discord_webhooks or []
        
        # Initialize from settings if token is provided
        if settings.ACCESS_TOKEN:
            token_value = settings.ACCESS_TOKEN.get_secret_value() if isinstance(settings.ACCESS_TOKEN, SecretStr) else settings.ACCESS_TOKEN
            if token_value:
                self.set_token_manually(token_value)
    
    async def initialize(self) -> bool:
        """
        Initialize authentication handler and attempt to load token from database.
        
        Returns:
            True if successfully loaded a valid token, False otherwise
        """
        # First try loading from database
        try:
            loaded = await self.load_token_from_db()
            if loaded and await self.has_valid_token():
                return True
        except Exception as e:
            logger.error(f"Error loading token from database: {e}")
            
        # If no token or invalid, use the one from settings if available
        if self.settings.ACCESS_TOKEN and not self.access_token:
            self.set_token_manually(self.settings.ACCESS_TOKEN.get_secret_value())
            return await self.has_valid_token()
            
        return False
    
    async def _ensure_session(self) -> None:
        """Ensure HTTP session exists."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"Content-Type": "application/json"}
            )
    
    async def close(self) -> None:
        """Close HTTP session if open."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
    
    async def get_auth_url(self, state: Optional[str] = None) -> str:
        """
        Generate the authorization URL for user login.
        
        Args:
            state: Optional state parameter for CSRF protection
            
        Returns:
            URL to redirect user for authentication
        """
        params = {
            TOKEN_PARAM_CLIENT_ID: self.settings.APP_ID,
            TOKEN_PARAM_REDIRECT_URI: str(self.settings.REDIRECT_URI),
            TOKEN_PARAM_RESPONSE_TYPE: RESPONSE_TYPE,
            TOKEN_PARAM_STATE: state or "fyers_auth_state"
        }
        
        auth_url = f"{self.settings.AUTH_BASE_URL}{AUTHORIZE_ENDPOINT}?{urlencode(params)}"
        logger.debug(f"Generated auth URL: {auth_url}")
        return auth_url
    
    async def get_access_token_from_code(self, auth_code: str) -> bool:
        """
        Exchange authorization code for access token.
        
        Args:
            auth_code: Authorization code from redirect
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            AuthenticationError: If authentication fails
            ConnectionError: If connection to Fyers API fails
        """
        if not auth_code:
            raise ValueError("Authorization code cannot be empty")
        
        async with self._lock:
            try:
                logger.info("Exchanging authorization code for access token")
                await self._ensure_session()
                
                data = {
                    "grant_type": TOKEN_GRANT_TYPE,
                    "appIdHash": self.settings.generate_app_id_hash(),
                    "code": auth_code
                }
                
                # Sanitize for logging
                safe_data = self._sanitize_log_data(data)
                logger.debug(f"Token request data: {safe_data}")
                
                response = await self._make_auth_request(AUTH_ENDPOINT, data)
                
                # Check for errors
                if response.get("s") != "ok":
                    error_msg = response.get("message", "Unknown error")
                    error_code = response.get("code", 0)
                    logger.error(f"Token acquisition failed: {error_code} - {error_msg}")
                    raise AuthenticationError(f"Failed to acquire token: {error_msg}")
                
                # Extract token and update state
                access_token = response.get("access_token")
                refresh_token = response.get("refresh_token", None)
                
                if not access_token:
                    raise AuthenticationError("No access token in response")
                
                # Parse expiry or set default (24 hours)
                expiry = self.parse_token_expiry(access_token)
                if not expiry:
                    expiry = datetime.now() + timedelta(hours=24)
                
                self._update_token_state(access_token, refresh_token, expiry)
                
                # Save token to database
                await self.save_token_to_db()
                
                logger.info(f"Successfully acquired access token, valid until {expiry}")
                return True
                
            except aiohttp.ClientError as e:
                logger.error(f"Connection error during token acquisition: {e}")
                raise ConnectionError(f"Failed to connect to Fyers auth service: {e}")
            except (RateLimitError, AuthenticationError):
                # Re-raise these specific errors
                raise
            except Exception as e:
                logger.error(f"Unexpected error during token acquisition: {e}", exc_info=True)
                raise AuthenticationError(f"Token acquisition failed: {e}")
    
    async def validate_token(self, token: Optional[str] = None) -> bool:
        """
        Verify if a token is valid with Fyers API.
        
        Args:
            token: Token to validate (uses stored token if None)
            
        Returns:
            True if token is valid, False otherwise
        """
        token_to_check = token or self.access_token
        if not token_to_check:
            return False
        
        try:
            logger.debug("Validating token with Fyers API")
            await self._ensure_session()
            
            # For validation, we simply make a profile request
            headers = {"Authorization": f"{self.settings.APP_ID}:{token_to_check}"}
            
            url = f"{self.settings.API_BASE_URL}profile"
            async with self._session.get(url, headers=headers) as response:
                if response.status != 200:
                    logger.warning(f"Token validation failed: HTTP {response.status}")
                    return False
                
                data = await response.json()
                return data.get("s") == "ok"
                
        except Exception as e:
            logger.error(f"Error validating token: {e}")
            return False
    
    async def has_valid_token(self) -> bool:
        """
        Check if current token is valid and not near expiry.
        
        Returns:
            True if token is valid and not near expiry, False otherwise
        """
        if not self.access_token or not self.token_expiry:
            return False
        
        # Check if already expired
        now = datetime.now()
        if now >= self.token_expiry:
            logger.warning("Token has expired")
            return False
        
        # Check if approaching expiry
        expiry_threshold = now + timedelta(seconds=self.settings.TOKEN_RENEWAL_MARGIN)
        if self.token_expiry <= expiry_threshold:
            logger.warning(f"Token will expire soon: {self.token_expiry}")
            return False
        
        return True
    
    async def refresh_token_async(self) -> bool:
        """
        Refresh access token using refresh token or PIN.
        
        Returns:
            True if successful, False otherwise
            
        Raises:
            AuthenticationError: If refresh fails
        """
        if not self.refresh_token_value:
            logger.warning("No refresh token available, cannot refresh")
            return False
        
        try:
            logger.info("Refreshing access token")
            await self._ensure_session()
            
            data = {
                "grant_type": REFRESH_GRANT_TYPE,
                "appIdHash": self.settings.generate_app_id_hash(),
                "refresh_token": self.refresh_token_value
            }
            
            # Add PIN if available
            if self.settings.PIN:
                data["pin"] = self.settings.PIN.get_secret_value()
            
            response = await self._make_auth_request("validate-refresh-token", data)
            
            # Check for errors
            if response.get("s") != "ok":
                error_msg = response.get("message", "Unknown error")
                error_code = response.get("code", 0)
                logger.error(f"Token refresh failed: {error_code} - {error_msg}")
                raise AuthenticationError(f"Failed to refresh token: {error_msg}")
            
            # Extract token and update state
            access_token = response.get("access_token")
            
            if not access_token:
                raise AuthenticationError("No access token in refresh response")
            
            # Parse expiry or set default (24 hours)
            expiry = self.parse_token_expiry(access_token)
            if not expiry:
                expiry = datetime.now() + timedelta(hours=24)
            
            self._update_token_state(access_token, self.refresh_token_value, expiry)
            
            # Save token to database
            await self.save_token_to_db()
            
            logger.info(f"Successfully refreshed access token, valid until {expiry}")
            return True
            
        except Exception as e:
            logger.error(f"Error refreshing token: {e}")
            if not isinstance(e, AuthenticationError):
                raise AuthenticationError(f"Token refresh failed: {e}")
            raise
    
    async def ensure_token(self) -> bool:
        """
        Ensure valid token exists, refreshing if needed and possible.
        
        Returns:
            True if valid token is available, False otherwise
        """
        async with self._lock:
            # If token is valid, return immediately
            if await self.has_valid_token():
                return True
            
            # If refresh token available, try to refresh
            if self.refresh_token_value:
                try:
                    success = await self.refresh_token_async()
                    if success:
                        return True
                except Exception as e:
                    logger.error(f"Token refresh failed: {e}")
            
            # Load from database as last resort
            if not self.access_token:
                try:
                    if await self.load_token_from_db():
                        return await self.has_valid_token()
                except Exception as e:
                    logger.error(f"Failed to load token from database: {e}")
            
            # No valid token and couldn't refresh
            return False
    
    def get_auth_headers(self) -> Dict[str, str]:
        """
        Get authorization headers for API requests.
        
        Returns:
            Dictionary with Authorization header if token available
            
        Raises:
            AuthenticationError: If no valid token is available
        """
        if not self.access_token:
            raise AuthenticationError("No access token available")
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"{self.settings.APP_ID}:{self.access_token}"
        }
        return headers
    
    def set_token_manually(self, access_token: str, 
                          refresh_token: Optional[str] = None,
                          expiry: Optional[datetime] = None) -> None:
        """
        Manually set access token for testing or development.
        
        Args:
            access_token: The access token string
            refresh_token: Optional refresh token
            expiry: Optional expiry datetime (default: 24 hours from now)
        """
        if not access_token:
            raise ValueError("Access token cannot be empty")
        
        # Parse expiry from token or use provided/default
        token_expiry = self.parse_token_expiry(access_token)
        if not token_expiry:
            token_expiry = expiry or (datetime.now() + timedelta(hours=24))
        
        self._update_token_state(access_token, refresh_token, token_expiry)
        logger.info(f"Manually set access token, valid until {token_expiry}")
        
        # Don't create async task in sync context - let caller handle saving
    
    async def clear_token(self) -> None:
        """Clear current token data."""
        async with self._lock:
            self.access_token = None
            self.refresh_token_value = None
            self.token_expiry = None
            logger.info("Cleared token data")
            
            # Remove from database
            try:
                # Use sync database session in async context
                def delete_token():
                    session = next(self.session_factory())
                    try:
                        success = FyersToken.delete_token(session, self.settings.APP_ID)
                        if success:
                            logger.info(f"Removed token from database for {self.settings.APP_ID}")
                        return success
                    finally:
                        session.close()
                
                # Run in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, delete_token)
                
            except Exception as e:
                logger.error(f"Failed to remove token from database: {e}")
    
    def get_token_info(self) -> Dict[str, Any]:
        """
        Get information about current token.
        
        Returns:
            Dictionary with token information
        """
        info = {
            "has_token": self.access_token is not None,
            "has_refresh_token": self.refresh_token_value is not None,
            "expiry": self.token_expiry,
            "is_valid": False
        }
        
        if self.access_token and self.token_expiry:
            now = datetime.now()
            info["is_valid"] = now < self.token_expiry
            info["time_remaining"] = (self.token_expiry - now).total_seconds() if info["is_valid"] else 0
            
        return info
    
    async def check_token_expiry(self) -> Optional[timedelta]:
        """
        Check if token is about to expire.
        
        Returns:
            Time remaining until expiry, or None if no valid token
        """
        if not self.access_token or not self.token_expiry:
            return None
            
        now = datetime.now()
        if now >= self.token_expiry:
            logger.warning("Token has already expired")
            return timedelta(0)
            
        time_remaining = self.token_expiry - now
        
        # Alert if token will expire soon
        expiry_threshold = timedelta(seconds=self.settings.TOKEN_RENEWAL_MARGIN)
        if time_remaining <= expiry_threshold:
            logger.warning(f"Token will expire in {time_remaining}. Renewal required soon.")
            
        return time_remaining
    
    async def send_discord_alert(self, message: str, level: str = "info") -> Dict[str, Any]:
        """
        Send an alert to Discord webhooks.
        
        Args:
            message: Message to send
            level: Alert level ('info', 'warning', 'error')
            
        Returns:
            Dictionary with results for each webhook
        """
        if not self.discord_webhooks:
            logger.debug("No Discord webhooks configured, skipping alert")
            return {"success": False, "reason": "No webhooks configured"}
        
        # Color based on level
        colors = {
            "info": 3447003,     # Blue
            "warning": 16776960, # Yellow
            "error": 15158332    # Red
        }
        color = colors.get(level, colors["info"])
        
        # Create Discord embed
        payload = {
            "embeds": [{
                "title": f"Fyers Authentication {level.title()}",
                "description": message,
                "color": color,
                "timestamp": datetime.now().isoformat()
            }]
        }
        
        results = {}
        
        # Send to all webhooks
        await self._ensure_session()
        for i, webhook in enumerate(self.discord_webhooks):
            try:
                async with self._session.post(webhook, json=payload) as response:
                    success = 200 <= response.status < 300
                    results[f"webhook_{i}"] = {
                        "success": success,
                        "status": response.status
                    }
                    
                    if not success:
                        response_text = await response.text()
                        logger.error(f"Failed to send Discord alert: HTTP {response.status} - {response_text}")
                    else:
                        logger.info(f"Sent Discord alert: {level}")
                        
            except Exception as e:
                logger.error(f"Error sending Discord alert: {e}")
                results[f"webhook_{i}"] = {
                    "success": False,
                    "error": str(e)
                }
                
        return results
    
    def get_renewal_instructions(self) -> Dict[str, Any]:
        """
        Get instructions for token renewal.
        
        Returns:
            Dictionary with renewal instructions including auth URL
        """
        # Create event loop if needed to run async method
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        auth_url = loop.run_until_complete(self.get_auth_url())
        
        return {
            "message": "Please follow these steps to renew your token:",
            "steps": [
                "1. Visit the authorization URL in your browser",
                "2. Complete the login and authorization process",
                "3. Copy the authorization code from the redirect URL (code= parameter)",
                "4. Call get_access_token_from_code() with the new code"
            ],
            "auth_url": auth_url
        }
    
    async def save_token_to_db(self) -> bool:
        """
        Save current token data to database using ORM.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.access_token or not self.token_expiry:
            logger.warning("No token data to save to database")
            return False
            
        try:
            # Use sync database session in async context
            def save_token():
                session = next(self.session_factory())
                try:
                    success = FyersToken.save_token(
                        session,
                        self.settings.APP_ID,
                        self.access_token,
                        self.refresh_token_value,
                        self.token_expiry
                    )
                    return success
                finally:
                    session.close()
            
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(None, save_token)
            
            if success:
                logger.info(f"Saved token to database for {self.settings.APP_ID}")
            return success
            
        except Exception as e:
            logger.error(f"Failed to save token to database: {e}")
            return False
    
    async def load_token_from_db(self) -> bool:
        """
        Load token data from database using ORM.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Use sync database session in async context
            def load_token():
                session = next(self.session_factory())
                try:
                    token = FyersToken.get_by_app_id(session, self.settings.APP_ID)
                    if token and token.is_active:
                        return {
                            'access_token': token.access_token,
                            'refresh_token': token.refresh_token,
                            'expiry': token.expiry
                        }
                    return None
                finally:
                    session.close()
            
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            token_data = await loop.run_in_executor(None, load_token)
            
            if not token_data:
                logger.info(f"No token found in database for {self.settings.APP_ID}")
                return False
                
            self._update_token_state(
                token_data['access_token'],
                token_data['refresh_token'],
                token_data['expiry']
            )
            
            logger.info(f"Loaded token from database for {self.settings.APP_ID}, valid until {token_data['expiry']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load token from database: {e}")
            return False
    
    def parse_token_expiry(self, token: str) -> Optional[datetime]:
        """
        Extract expiry time from JWT token.
        
        Args:
            token: JWT token
            
        Returns:
            Expiry datetime or None if parsing fails
        """
        try:
            # JWT tokens are base64 encoded in 3 parts separated by dots
            parts = token.split('.')
            if len(parts) != 3:
                logger.warning("Invalid JWT token format")
                return None
            
            # Decode the payload (middle part)
            # Padding may be necessary for base64 decoding
            padding = '=' * (4 - len(parts[1]) % 4) if len(parts[1]) % 4 != 0 else ''
            payload = base64.b64decode(parts[1] + padding).decode('utf-8')
            claims = json.loads(payload)
            
            # Extract expiry time (standard field is 'exp')
            if 'exp' not in claims:
                logger.warning("No expiry claim in token")
                return None
            
            # Convert epoch time to datetime
            expiry = datetime.fromtimestamp(claims['exp'])
            return expiry
            
        except Exception as e:
            logger.error(f"Error parsing token expiry: {e}")
            return None
    
    def _update_token_state(self, access_token: str, refresh_token: Optional[str], expiry: datetime) -> None:
        """
        Update internal token state with new token.
        
        Args:
            access_token: New access token
            refresh_token: New refresh token (optional)
            expiry: Token expiry datetime
        """
        self.access_token = access_token
        self.refresh_token_value = refresh_token
        self.token_expiry = expiry
    
    async def _make_auth_request(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make HTTP request to auth endpoints.
        
        Args:
            endpoint: API endpoint path
            data: Request payload
            
        Returns:
            API response as dictionary
            
        Raises:
            ConnectionError: If connection fails
            AuthenticationError: If authentication fails
            RateLimitError: If rate limit is exceeded
        """
        await self._ensure_session()
        
        url = f"{self.settings.AUTH_BASE_URL}{endpoint}"
        
        try:
            async with self._session.post(url, json=data) as response:
                response_text = await response.text()
                
                # Handle HTTP errors
                if response.status == 429:
                    logger.error(f"Rate limit exceeded: {response_text}")
                    raise RateLimitError("Fyers API rate limit exceeded")
                
                if response.status != 200:
                    logger.error(f"Auth request failed (HTTP {response.status}): {response_text}")
                    raise ConnectionError(f"Auth request failed with status {response.status}: {response_text}")
                
                # Parse response
                try:
                    result = json.loads(response_text)
                    return result
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON response: {response_text}")
                    raise ConnectionError(f"Invalid response from auth server: {e}")
                
        except aiohttp.ClientError as e:
            logger.error(f"Connection error during auth request: {e}")
            raise ConnectionError(f"Failed to connect to Fyers auth service: {e}")
    
    def _sanitize_log_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize sensitive data for logging.
        
        Args:
            data: Data to sanitize
            
        Returns:
            Sanitized data with sensitive fields masked
        """
        safe_data = data.copy()
        
        # Mask sensitive fields
        sensitive_keys = ['code', 'appIdHash', 'refresh_token', 'pin', 'password']
        for key in sensitive_keys:
            if key in safe_data:
                safe_data[key] = '********'
        
        return safe_data