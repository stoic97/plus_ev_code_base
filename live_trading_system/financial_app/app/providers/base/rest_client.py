"""
Base REST API client for market data providers.

This module implements a reusable REST client with connection pooling, 
retry mechanisms, rate limiting, and standardized error handling
for all market data providers.
"""

import logging
import json
import time
import asyncio
from typing import Any, Dict, List, Optional, Union, Callable, Type, TypeVar, cast
from urllib.parse import urljoin

import aiohttp
from aiohttp import ClientSession, ClientResponse, TCPConnector
from pydantic import BaseModel, ValidationError

from app.providers.base.provider import (
    ProviderError, ConnectionError, AuthenticationError, 
    RateLimitError, DataNotFoundError, BaseProvider, RateLimiter
)
from app.providers.config.provider_settings import BaseProviderSettings

# Set up logging
logger = logging.getLogger(__name__)

# Type variables
T = TypeVar('T')
ResponseType = TypeVar('ResponseType', bound=BaseModel)


class RequestError(ProviderError):
    """Error from REST API request."""
    
    def __init__(
        self, 
        message: str, 
        status_code: Optional[int] = None,
        response_text: Optional[str] = None
    ):
        """
        Initialize a request error.
        
        Args:
            message: Error message
            status_code: HTTP status code (optional)
            response_text: Response text (optional)
        """
        self.status_code = status_code
        self.response_text = response_text
        super().__init__(message)


class RestClient:
    """
    Base REST API client with connection pooling and error handling.
    
    Provides a reusable foundation for provider-specific REST clients
    with consistent error handling, retry logic, and connection management.
    """
    
    def __init__(
        self, 
        base_url: str,
        settings: BaseProviderSettings,
        headers: Optional[Dict[str, str]] = None,
        request_hooks: Optional[Dict[str, Callable]] = None,
        response_hooks: Optional[Dict[str, Callable]] = None
    ):
        """
        Initialize a REST client.
        
        Args:
            base_url: Base URL for API requests
            settings: Provider settings
            headers: Default headers for all requests
            request_hooks: Functions to call before sending requests
            response_hooks: Functions to call after receiving responses
        """
        self.base_url = base_url.rstrip("/") + "/"
        self.settings = settings
        self.default_headers = headers or {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        # Hooks for request/response processing
        self.request_hooks = request_hooks or {}
        self.response_hooks = response_hooks or {}
        
        # Session management
        self._session: Optional[ClientSession] = None
        self._connector: Optional[TCPConnector] = None
        
        # Rate limiter
        self.rate_limiter = RateLimiter(
            calls_per_second=settings.RATE_LIMIT_CALLS / settings.RATE_LIMIT_PERIOD
        )
        
        # Various endpoint-specific rate limiters can be added by subclasses
        
        logger.debug(f"Initialized REST client for {base_url}")
    
    async def __aenter__(self) -> "RestClient":
        """Enter async context manager, ensuring session is created."""
        await self.ensure_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context manager, closing the session."""
        await self.close()
    
    async def ensure_session(self) -> None:
        """
        Ensure that an HTTP session exists, creating one if necessary.
        """
        if self._session is None or self._session.closed:
            # Create connection pooling connector
            self._connector = TCPConnector(
                limit=100,  # Connection pool size
                limit_per_host=20,  # Connections per host
                enable_cleanup_closed=True,
                keepalive_timeout=30,  # Keep connections alive for 30 seconds
            )
            
            # Create session with connector
            timeout = aiohttp.ClientTimeout(
                total=self.settings.REQUEST_TIMEOUT,
                connect=self.settings.CONNECTION_TIMEOUT
            )
            
            self._session = ClientSession(
                connector=self._connector,
                timeout=timeout,
                headers=self.default_headers.copy(),
                raise_for_status=False,  # We'll handle status codes ourselves
            )
            
            logger.debug("Created new HTTP session with connection pooling")
    
    async def close(self) -> None:
        """
        Close the HTTP session.
        """
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
            logger.debug("Closed HTTP session")
    
    def get_full_url(self, endpoint: str) -> str:
        """
        Get the full URL for an endpoint.
        
        Args:
            endpoint: API endpoint path
            
        Returns:
            Full URL
        """
        # Strip leading slash if present to avoid double slashes
        endpoint = endpoint.lstrip("/")
        return urljoin(self.base_url, endpoint)
    
    def _prepare_headers(self, headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """
        Prepare headers for a request by combining defaults with provided headers.
        
        Args:
            headers: Additional headers for this request
            
        Returns:
            Combined headers
        """
        result = self.default_headers.copy()
        if headers:
            result.update(headers)
        return result
    
    async def _process_response(
        self, 
        response: ClientResponse
    ) -> Dict[str, Any]:
        """
        Process an HTTP response, handling errors and extracting JSON.
        
        Args:
            response: HTTP response
            
        Returns:
            JSON response data
            
        Raises:
            RequestError: If the request failed
            ConnectionError: For network errors
            AuthenticationError: For authentication failures
            RateLimitError: For rate limit errors
            DataNotFoundError: If requested data wasn't found
        """
        status_code = response.status
        content_type = response.headers.get("Content-Type", "")
        
        # Call response hooks if defined
        hook_name = f"status_{status_code}"
        if hook_name in self.response_hooks:
            await self.response_hooks[hook_name](response)
        
        # Process based on status code
        if 200 <= status_code < 300:
            # Successful response
            if "application/json" in content_type:
                try:
                    return await response.json()
                except json.JSONDecodeError as e:
                    text = await response.text()
                    logger.error(f"Failed to parse JSON response: {text[:200]}...")
                    raise RequestError(f"Invalid JSON response: {e}", status_code, text)
            else:
                # Not JSON, return as text
                text = await response.text()
                return {"text": text}
        
        # Error response - try to parse response for error details
        error_text = await response.text()
        error_data = {}
        
        try:
            if "application/json" in content_type:
                error_data = json.loads(error_text)
        except json.JSONDecodeError:
            # Not valid JSON, use text as is
            pass
        
        # Extract error message if possible
        error_message = error_data.get("message", "Unknown error")
        if not error_message or error_message == "Unknown error":
            # Try common error fields
            for field in ["error", "error_description", "errorMessage", "msg"]:
                if field in error_data:
                    error_message = error_data[field]
                    break
        
        # Map status codes to specific error types
        if status_code == 401 or status_code == 403:
            raise AuthenticationError(f"Authentication failed: {error_message}", status_code, error_text)
        elif status_code == 404:
            raise DataNotFoundError(f"Data not found: {error_message}", status_code, error_text)
        elif status_code == 429:
            retry_after = response.headers.get("Retry-After")
            message = f"Rate limit exceeded: {error_message}"
            if retry_after:
                message += f", retry after {retry_after} seconds"
            raise RateLimitError(message, status_code, error_text)
        else:
            # Generic error
            raise RequestError(f"Request failed ({status_code}): {error_message}", status_code, error_text)
    
    async def _handle_rate_limiting(self, endpoint: str) -> None:
        """
        Handle rate limiting for an endpoint.
        
        Args:
            endpoint: API endpoint being accessed
            
        Raises:
            RateLimitError: If rate limiting is enabled but tokens not available
        """
        if not self.settings.RATE_LIMIT_ENABLED:
            return
        
        # By default use the global rate limiter
        limiter = self.rate_limiter
        
        # Special case for endpoint-specific rate limiters
        # Subclasses can add their own limiters and override this method
        
        # Acquire a token from the rate limiter
        try:
            await limiter.acquire()
        except Exception as e:
            logger.error(f"Rate limiting error: {e}")
            raise RateLimitError(f"Failed to acquire rate limiting token: {e}")
    
    async def request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        skip_rate_limiting: bool = False,
    ) -> Dict[str, Any]:
        """
        Make an HTTP request to the API.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            params: Query parameters
            data: Request data (will be converted to JSON)
            headers: Additional headers
            timeout: Request timeout override
            skip_rate_limiting: Whether to skip rate limiting
            
        Returns:
            Response data
            
        Raises:
            ConnectionError: For network errors
            RequestError: If the request failed
            Various specific error types for different failure modes
        """
        await self.ensure_session()
        assert self._session is not None, "Session not initialized"
        
        # Handle rate limiting
        if not skip_rate_limiting:
            await self._handle_rate_limiting(endpoint)
        
        # Build full URL
        url = self.get_full_url(endpoint)
        
        # Prepare headers
        request_headers = self._prepare_headers(headers)
        
        # Prepare request
        request_kwargs = {
            "params": params,
            "headers": request_headers,
        }
        
        # Add data if provided
        if data is not None:
            if isinstance(data, dict) or isinstance(data, list):
                request_kwargs["json"] = data
            else:
                request_kwargs["data"] = data
        
        # Set custom timeout if specified
        if timeout is not None:
            request_kwargs["timeout"] = aiohttp.ClientTimeout(total=timeout)
        
        # Log request details at debug level
        logger.debug(f"Making {method} request to {url}")
        if self.settings.DEBUG_MODE:
            logger.debug(f"Request params: {params}")
            if isinstance(data, dict) or isinstance(data, list):
                # Mask sensitive data like passwords, tokens, etc.
                safe_data = self._mask_sensitive_data(data)
                logger.debug(f"Request data: {safe_data}")
        
        # Call pre-request hooks
        hook_name = f"before_{method.lower()}"
        if hook_name in self.request_hooks:
            await self.request_hooks[hook_name](url, request_kwargs)
        
        # Make the request with error handling
        start_time = time.monotonic()
        try:
            async with self._session.request(method, url, **request_kwargs) as response:
                # Process the response
                result = await self._process_response(response)
                
                # Log timing at debug level
                request_time = time.monotonic() - start_time
                logger.debug(f"{method} request to {endpoint} completed in {request_time:.3f}s")
                
                return result
                
        except aiohttp.ClientError as e:
            request_time = time.monotonic() - start_time
            logger.error(f"{method} request to {endpoint} failed after {request_time:.3f}s: {e}")
            
            if isinstance(e, aiohttp.ClientConnectorError):
                raise ConnectionError(f"Connection error: {e}")
            elif isinstance(e, aiohttp.ClientResponseError):
                raise RequestError(f"Response error: {e}")
            elif isinstance(e, aiohttp.ClientPayloadError):
                raise RequestError(f"Payload error: {e}")
            elif isinstance(e, aiohttp.ClientOSError):
                raise ConnectionError(f"OS error: {e}")
            elif isinstance(e, aiohttp.ServerDisconnectedError):
                raise ConnectionError("Server disconnected")
            elif isinstance(e, aiohttp.ServerTimeoutError):
                raise ConnectionError("Server timeout")
            else:
                raise ConnectionError(f"Request failed: {e}")
        except asyncio.TimeoutError:
            request_time = time.monotonic() - start_time
            logger.error(f"{method} request to {endpoint} timed out after {request_time:.3f}s")
            raise ConnectionError("Request timed out")
    
    def _mask_sensitive_data(self, data: Union[Dict[str, Any], List[Any]]) -> Union[Dict[str, Any], List[Any]]:
        """
        Mask sensitive data for logging.
        
        Args:
            data: Data to mask
            
        Returns:
            Masked data
        """
        if isinstance(data, dict):
            result = {}
            for key, value in data.items():
                # Mask sensitive fields
                if any(sensitive in key.lower() for sensitive in [
                    "password", "secret", "token", "key", "auth", "cred", "security"
                ]):
                    result[key] = "********"
                elif isinstance(value, (dict, list)):
                    result[key] = self._mask_sensitive_data(value)
                else:
                    result[key] = value
            return result
        elif isinstance(data, list):
            return [self._mask_sensitive_data(item) if isinstance(item, (dict, list)) else item for item in data]
        else:
            return data
    
    async def get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        skip_rate_limiting: bool = False,
    ) -> Dict[str, Any]:
        """
        Make a GET request.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            headers: Additional headers
            timeout: Request timeout override
            skip_rate_limiting: Whether to skip rate limiting
            
        Returns:
            Response data
        """
        return await self.request(
            "GET", 
            endpoint, 
            params=params, 
            headers=headers, 
            timeout=timeout,
            skip_rate_limiting=skip_rate_limiting
        )
    
    async def post(
        self,
        endpoint: str,
        data: Optional[Any] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        skip_rate_limiting: bool = False,
    ) -> Dict[str, Any]:
        """
        Make a POST request.
        
        Args:
            endpoint: API endpoint
            data: Request data
            params: Query parameters
            headers: Additional headers
            timeout: Request timeout override
            skip_rate_limiting: Whether to skip rate limiting
            
        Returns:
            Response data
        """
        return await self.request(
            "POST", 
            endpoint, 
            params=params, 
            data=data, 
            headers=headers, 
            timeout=timeout,
            skip_rate_limiting=skip_rate_limiting
        )
    
    async def put(
        self,
        endpoint: str,
        data: Optional[Any] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        skip_rate_limiting: bool = False,
    ) -> Dict[str, Any]:
        """
        Make a PUT request.
        
        Args:
            endpoint: API endpoint
            data: Request data
            params: Query parameters
            headers: Additional headers
            timeout: Request timeout override
            skip_rate_limiting: Whether to skip rate limiting
            
        Returns:
            Response data
        """
        return await self.request(
            "PUT", 
            endpoint, 
            params=params, 
            data=data, 
            headers=headers, 
            timeout=timeout,
            skip_rate_limiting=skip_rate_limiting
        )
    
    async def delete(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        skip_rate_limiting: bool = False,
    ) -> Dict[str, Any]:
        """
        Make a DELETE request.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            headers: Additional headers
            timeout: Request timeout override
            skip_rate_limiting: Whether to skip rate limiting
            
        Returns:
            Response data
        """
        return await self.request(
            "DELETE", 
            endpoint, 
            params=params, 
            headers=headers, 
            timeout=timeout,
            skip_rate_limiting=skip_rate_limiting
        )
    
    async def with_retries(
        self,
        func: Callable[[], T],
        max_retries: Optional[int] = None,
        retry_delay: Optional[float] = None,
        retryable_errors: Optional[List[Type[Exception]]] = None
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
        retryable = retryable_errors or [ConnectionError, RequestError, RateLimitError]
        
        # Remove non-retryable errors from the list
        if AuthenticationError in retryable:
            retryable.remove(AuthenticationError)
        if DataNotFoundError in retryable:
            retryable.remove(DataNotFoundError)
        
        last_error = None
        for attempt in range(1, max_attempts + 1):
            try:
                return await func()
            except tuple(retryable) as e:
                last_error = e
                if attempt < max_attempts:
                    # For rate limit errors, use the Retry-After header if available
                    if isinstance(e, RateLimitError) and hasattr(e, "response_headers"):
                        retry_after = e.response_headers.get("Retry-After")
                        if retry_after:
                            try:
                                wait_time = float(retry_after)
                            except (ValueError, TypeError):
                                wait_time = delay * (2 ** (attempt - 1))
                        else:
                            wait_time = delay * (2 ** (attempt - 1))
                    else:
                        wait_time = delay * (2 ** (attempt - 1))
                    
                    logger.warning(
                        f"Request failed on attempt {attempt}/{max_attempts}, "
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
    
    async def get_with_model(
        self,
        endpoint: str,
        model_class: Type[ResponseType],
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> ResponseType:
        """
        Make a GET request and parse the response into a Pydantic model.
        
        Args:
            endpoint: API endpoint
            model_class: Pydantic model class to parse response into
            params: Query parameters
            headers: Additional headers
            timeout: Request timeout override
            
        Returns:
            Parsed model instance
            
        Raises:
            ProviderError: If request fails or response parsing fails
        """
        result = await self.get(endpoint, params, headers, timeout)
        
        try:
            return model_class.model_validate(result)
        except ValidationError as e:
            logger.error(f"Failed to parse response into {model_class.__name__}: {e}")
            raise RequestError(f"Invalid response format: {e}")
    
    async def post_with_model(
        self,
        endpoint: str,
        model_class: Type[ResponseType],
        data: Optional[Any] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> ResponseType:
        """
        Make a POST request and parse the response into a Pydantic model.
        
        Args:
            endpoint: API endpoint
            model_class: Pydantic model class to parse response into
            data: Request data
            params: Query parameters
            headers: Additional headers
            timeout: Request timeout override
            
        Returns:
            Parsed model instance
            
        Raises:
            ProviderError: If request fails or response parsing fails
        """
        result = await self.post(endpoint, data, params, headers, timeout)
        
        try:
            return model_class.model_validate(result)
        except ValidationError as e:
            logger.error(f"Failed to parse response into {model_class.__name__}: {e}")
            raise RequestError(f"Invalid response format: {e}")