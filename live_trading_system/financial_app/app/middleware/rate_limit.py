"""
Rate Limiting Middleware for Trading Strategies Application.

This module provides robust request rate limiting functionality with:
- Redis-based distributed rate limiting
- In-memory fallback when Redis is unavailable
- Flexible client identification (IP or API key)
- Configurable limits per endpoint or client type
- Standard rate limit response headers
"""

import time
import logging
import ipaddress
from datetime import datetime
from typing import Dict, Optional, Tuple, Callable, Any, List, Union
import asyncio
from fastapi import Request, Response, Depends
from fastapi.routing import APIRoute
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

# Import core modules
from app.core.config import get_settings
from app.core.error_handling import RateLimitExceededError, ErrorTracker, ErrorCategory
from app.core.database import get_redis_db, RedisDB, DatabaseType, get_db_instance

# Set up logging
logger = logging.getLogger(__name__)

# In-memory fallback storage for rate limits when Redis is unavailable
_local_rate_limits: Dict[str, Dict[str, Union[int, float]]] = {}


class RateLimitSettings:
    """Rate limit settings configuration container."""
    
    def __init__(self):
        """Initialize rate limit settings from application config."""
        self.settings = get_settings()
        
        # Default rate limits
        self.default_rate_limit = self.settings.security.RATE_LIMIT_DEFAULT or 100
        self.default_rate_window = self.settings.security.RATE_LIMIT_WINDOW_SECONDS or 60
        
        # API endpoints can have different rate limits
        self.api_rate_limit = self.settings.security.API_RATE_LIMIT or 200
        self.api_rate_window = self.settings.security.API_RATE_WINDOW_SECONDS or 60
        
        # Authentication endpoints often need stricter limits
        self.auth_rate_limit = self.settings.security.AUTH_RATE_LIMIT or 20
        self.auth_rate_window = self.settings.security.AUTH_RATE_WINDOW_SECONDS or 60
        
        # Trusted clients (internal services, admin users) can have higher limits
        self.trusted_rate_limit = self.settings.security.TRUSTED_RATE_LIMIT or 1000
        self.trusted_rate_window = self.settings.security.TRUSTED_RATE_WINDOW_SECONDS or 60
        
        # Redis key configuration
        self.redis_key_prefix = "rate_limit:"
        self.redis_key_ttl = max(self.default_rate_window, 
                                 self.api_rate_window,
                                 self.auth_rate_window,
                                 self.trusted_rate_window) * 2  # TTL longer than window
        
        # Path patterns for specific rate limit rules
        self.auth_paths = ["/api/v1/auth/", "/login", "/register", "/password-reset"]
        self.excluded_paths = ["/health", "/metrics", "/docs", "/redoc", "/openapi.json"]
        self.trusted_ips = self._parse_trusted_ips()
        self.trusted_api_keys = self.settings.security.TRUSTED_API_KEYS or []
    
    def _parse_trusted_ips(self) -> List[ipaddress.IPv4Network]:
        """Parse trusted IP addresses or networks from settings."""
        trusted_ips = []
        raw_ips = self.settings.security.TRUSTED_IPS or []
        
        for ip_str in raw_ips:
            try:
                # Check if it's a CIDR notation (network)
                if "/" in ip_str:
                    trusted_ips.append(ipaddress.IPv4Network(ip_str))
                else:
                    # Convert single IP to network with /32 mask
                    trusted_ips.append(ipaddress.IPv4Network(f"{ip_str}/32"))
            except ValueError as e:
                logger.warning(f"Invalid trusted IP address or network: {ip_str}. Error: {e}")
        
        return trusted_ips


class RateLimiter:
    """
    Rate limiter implementation with Redis storage and in-memory fallback.
    """
    
    def __init__(self):
        """Initialize the rate limiter."""
        self.settings = RateLimitSettings()
        self.redis: Optional[RedisDB] = None
        self._initialize_redis()
    
    def _initialize_redis(self) -> None:
        """Initialize Redis connection for rate limiting."""
        try:
            self.redis = get_db_instance(DatabaseType.REDIS)
            if not self.redis.is_connected:
                self.redis.connect()
        except Exception as e:
            logger.warning(f"Failed to initialize Redis for rate limiting: {e}")
            self.redis = None
    
    def _get_redis_key(self, client_id: str, window: int) -> str:
        """
        Generate Redis key for a specific client and time window.
        
        Args:
            client_id: Client identifier (IP or API key)
            window: Time window in seconds
            
        Returns:
            Redis key string
        """
        return f"{self.settings.redis_key_prefix}{client_id}:{window}"
    
    def _extract_client_id(self, request: Request) -> str:
        """
        Extract client identifier from request.
        Tries API key first, then falls back to IP address.
        
        Args:
            request: FastAPI request object
            
        Returns:
            Client identifier string
        """
        # Try to get API key from header or query parameter
        api_key = None
        auth_header = request.headers.get("Authorization")
        
        if auth_header and auth_header.startswith("Bearer "):
            api_key = auth_header.replace("Bearer ", "")
        else:
            # Try to get from query parameter
            api_key = request.query_params.get("api_key")
        
        # If API key found, use it as client identifier
        if api_key:
            return f"api:{api_key}"
        
        # Otherwise use IP address
        client_ip = self._get_client_ip(request)
        return f"ip:{client_ip}"
    
    def _get_client_ip(self, request: Request) -> str:
        """
        Get client IP address from request.
        Handles X-Forwarded-For header for proxied requests.
        
        Args:
            request: FastAPI request object
            
        Returns:
            Client IP address
        """
        # Check X-Forwarded-For header (common with reverse proxies)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # The client IP is the first address in the list
            return forwarded_for.split(",")[0].strip()
        
        # Fall back to direct client IP
        return str(request.client.host) if request.client else "unknown"
    
    def _is_trusted_client(self, request: Request, client_id: str) -> bool:
        """
        Check if the client is in the trusted list.
        
        Args:
            request: FastAPI request object
            client_id: Extracted client identifier
            
        Returns:
            True if client is trusted, False otherwise
        """
        # Check if using API key
        if client_id.startswith("api:"):
            api_key = client_id.replace("api:", "")
            return api_key in self.settings.trusted_api_keys
        
        # Check if IP is trusted
        client_ip = self._get_client_ip(request)
        try:
            ip_obj = ipaddress.IPv4Address(client_ip)
            for trusted_net in self.settings.trusted_ips:
                if ip_obj in trusted_net:
                    return True
        except ValueError:
            # If IP parsing fails, don't trust the client
            pass
        
        return False
    
    def _should_skip_rate_limiting(self, request: Request) -> bool:
        """
        Check if rate limiting should be skipped for this request.
        
        Args:
            request: FastAPI request object
            
        Returns:
            True if rate limiting should be skipped, False otherwise
        """
        # Skip for excluded paths
        path = request.url.path
        return any(excluded in path for excluded in self.settings.excluded_paths)
    
    def _get_rate_limit_for_request(self, request: Request, is_trusted: bool) -> Tuple[int, int]:
        """
        Determine appropriate rate limit for the request.
        
        Args:
            request: FastAPI request object
            is_trusted: Whether the client is trusted
            
        Returns:
            Tuple of (rate_limit, window_seconds)
        """
        # Trusted clients get higher limits
        if is_trusted:
            return (self.settings.trusted_rate_limit, self.settings.trusted_rate_window)
        
        # Check for authentication paths
        path = request.url.path
        if any(auth_path in path for auth_path in self.settings.auth_paths):
            return (self.settings.auth_rate_limit, self.settings.auth_rate_window)
        
        # Check for API paths
        if path.startswith("/api/"):
            return (self.settings.api_rate_limit, self.settings.api_rate_window)
        
        # Default rate limit
        return (self.settings.default_rate_limit, self.settings.default_rate_window)
    
    async def _check_rate_limit_redis(
        self, 
        client_id: str, 
        rate_limit: int, 
        window: int
    ) -> Tuple[bool, int, int]:
        """
        Check rate limit using Redis storage.
        
        Args:
            client_id: Client identifier
            rate_limit: Maximum requests allowed
            window: Time window in seconds
            
        Returns:
            Tuple of (is_allowed, remaining_requests, reset_time)
        """
        key = self._get_redis_key(client_id, window)
        pipe = self.redis.client.pipeline()
        
        # Current time
        now = time.time()
        window_start = now - window
        
        try:
            # Add the current request timestamp
            pipe.zadd(key, {str(now): now})
            
            # Remove timestamps outside the current window
            pipe.zremrangebyscore(key, 0, window_start)
            
            # Count requests in current window
            pipe.zcard(key)
            
            # Set expiration on the key
            pipe.expire(key, self.settings.redis_key_ttl)
            
            # Execute pipeline
            _, _, request_count, _ = await pipe.execute()
            
            # Calculate reset time and remaining requests
            reset_time = int(now) + window
            remaining = max(0, rate_limit - request_count)
            
            # Check if limit exceeded
            is_allowed = request_count <= rate_limit
            
            return (is_allowed, remaining, reset_time)
        
        except Exception as e:
            logger.error(f"Redis rate limit check failed: {e}")
            # Fall back to local in-memory check
            return await self._check_rate_limit_local(client_id, rate_limit, window)
    
    async def _check_rate_limit_local(
        self, 
        client_id: str, 
        rate_limit: int, 
        window: int
    ) -> Tuple[bool, int, int]:
        """
        Check rate limit using local in-memory storage (fallback).
        
        Args:
            client_id: Client identifier
            rate_limit: Maximum requests allowed
            window: Time window in seconds
            
        Returns:
            Tuple of (is_allowed, remaining_requests, reset_time)
        """
        global _local_rate_limits
        
        # Key for the rate limit entry
        key = f"{client_id}:{window}"
        now = time.time()
        
        # Initialize if not present
        if key not in _local_rate_limits:
            _local_rate_limits[key] = {
                "count": 0,
                "reset_at": now + window,
                "timestamps": []
            }
        
        # Check if window expired and reset if needed
        if now > _local_rate_limits[key]["reset_at"]:
            _local_rate_limits[key] = {
                "count": 0,
                "reset_at": now + window,
                "timestamps": []
            }
        
        # Add request timestamp
        timestamps = _local_rate_limits[key]["timestamps"]
        timestamps.append(now)
        
        # Remove expired timestamps
        window_start = now - window
        _local_rate_limits[key]["timestamps"] = [t for t in timestamps if t > window_start]
        
        # Update count
        request_count = len(_local_rate_limits[key]["timestamps"])
        _local_rate_limits[key]["count"] = request_count
        
        # Get remaining and reset time
        reset_time = int(_local_rate_limits[key]["reset_at"])
        remaining = max(0, rate_limit - request_count)
        
        # Check if limit exceeded
        is_allowed = request_count <= rate_limit
        
        return (is_allowed, remaining, reset_time)
    
    async def check_rate_limit(
        self, 
        request: Request
    ) -> Tuple[bool, int, int, int, int]:
        """
        Check if request is within rate limits.
        
        Args:
            request: FastAPI request object
            
        Returns:
            Tuple of (is_allowed, limit, remaining, reset_time, window)
        """
        # Skip rate limiting for excluded paths
        if self._should_skip_rate_limiting(request):
            return (True, 0, 0, 0, 0)
        
        # Get client identifier
        client_id = self._extract_client_id(request)
        
        # Check if trusted client
        is_trusted = self._is_trusted_client(request, client_id)
        
        # Get applicable rate limit
        limit, window = self._get_rate_limit_for_request(request, is_trusted)
        
        # Use Redis if available, otherwise use local storage
        if self.redis and self.redis.is_connected:
            is_allowed, remaining, reset_time = await self._check_rate_limit_redis(client_id, limit, window)
        else:
            is_allowed, remaining, reset_time = await self._check_rate_limit_local(client_id, limit, window)
        
        return (is_allowed, limit, remaining, reset_time, window)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware for enforcing rate limits on API requests.
    """
    
    def __init__(self, app: ASGIApp):
        """Initialize rate limit middleware."""
        super().__init__(app)
        self.rate_limiter = RateLimiter()
        logger.info("Rate limiting middleware initialized")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process the request and apply rate limiting.
        
        Args:
            request: FastAPI request
            call_next: Next middleware in chain
            
        Returns:
            Response object
        """
        # Skip middleware for OPTIONS requests (CORS preflight)
        if request.method == "OPTIONS":
            return await call_next(request)
        
        # Check rate limit for this request
        is_allowed, limit, remaining, reset_time, window = await self.rate_limiter.check_rate_limit(request)
        
        # If rate limit check was skipped, proceed without adding headers
        if limit == 0:
            return await call_next(request)
        
        if not is_allowed:
            # Track rate limit exceeded error
            ErrorTracker.track_error(
                RateLimitExceededError(
                    message=f"Rate limit exceeded: {limit} requests per {window} seconds"
                ),
                category=ErrorCategory.RATE_LIMIT
            )
            
            # Create error response
            error = RateLimitExceededError(
                message=f"Rate limit exceeded. Try again in {reset_time - int(time.time())} seconds."
            )
            
            # Convert to HTTP exception
            http_exception = error.to_http_exception()
            
            # Create response
            response = Response(
                content={"detail": http_exception.detail},
                status_code=http_exception.status_code,
                media_type="application/json"
            )
            
            # Add rate limit headers
            response.headers["X-RateLimit-Limit"] = str(limit)
            response.headers["X-RateLimit-Remaining"] = "0"
            response.headers["X-RateLimit-Reset"] = str(reset_time)
            response.headers["Retry-After"] = str(reset_time - int(time.time()))
            
            return response
        
        # Proceed with request processing
        response = await call_next(request)
        
        # Add rate limit headers to the response
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(reset_time)
        
        return response


# Dependency for endpoint-specific rate limiting 
# Can be used with FastAPI's Depends() in route definitions
async def check_rate_limit(
    request: Request,
    limit: Optional[int] = None,
    window: Optional[int] = None
) -> None:
    """
    Check rate limit for a specific endpoint.
    Raises RateLimitExceededError if limit is exceeded.
    
    Args:
        request: FastAPI request
        limit: Optional custom limit for this endpoint
        window: Optional custom window for this endpoint
        
    Raises:
        RateLimitExceededError: If rate limit is exceeded
    """
    # Create limiter instance
    limiter = RateLimiter()
    
    # Check rate limit
    is_allowed, default_limit, remaining, reset_time, default_window = await limiter.check_rate_limit(request)
    
    # Apply custom limits if provided
    actual_limit = limit or default_limit
    actual_window = window or default_window
    
    if not is_allowed:
        # Rate limit exceeded, raise error with retry information
        retry_seconds = max(1, reset_time - int(time.time()))
        raise RateLimitExceededError(
            message=f"Rate limit exceeded. Maximum {actual_limit} requests per {actual_window} seconds allowed. "
                    f"Please try again in {retry_seconds} seconds."
        )