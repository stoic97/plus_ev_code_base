"""
Authentication middleware for the Trading Strategies Application.

This middleware handles:
- Setting up user context from authentication performed by endpoints
- Protecting non-public endpoints that lack explicit auth dependencies
- Providing API key authentication for service-to-service communication
"""

import logging
from typing import Optional

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.status import HTTP_401_UNAUTHORIZED

from app.core.error_handling import AuthenticationError

logger = logging.getLogger(__name__)


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Lightweight middleware that ensures authentication for protected paths.
    
    This middleware complements the direct security.py usage by endpoints
    by ensuring that protected paths have authentication.
    """
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """
        Process the request, setting up auth context if needed.
        
        Args:
            request: The incoming request
            call_next: The next middleware or route handler
            
        Returns:
            The response from the next handler
        """
        # Skip auth for certain paths
        if self._should_skip_auth(request.url.path):
            # For public endpoints, ensure auth state is cleared
            request.state.auth_type = None
            request.state.user = None
            request.state.roles = []
            return await call_next(request)
        
        # For protected endpoints, check if auth header exists
        # The actual auth is handled by the security dependencies in the routes
        auth_header = request.headers.get("Authorization", "")
        
        # Check if the endpoint requires authentication but has no auth header
        if self._requires_auth(request.url.path, request.method) and not auth_header:
            return self._auth_error_response("Authentication required")
        
        # Continue to the endpoint, which will use security.py dependencies
        # for actual authentication
        response = await call_next(request)
        
        return response
    
    def _extract_token(self, auth_header: str) -> Optional[str]:
        """
        Extract JWT token from Authorization header.
        
        Args:
            auth_header: Authorization header value
            
        Returns:
            JWT token or None if not found
        """
        if auth_header.startswith("Bearer "):
            return auth_header.replace("Bearer ", "").strip()
        return None
    
    def _auth_error_response(self, detail: str) -> JSONResponse:
        """
        Create a standardized authentication error response.
        
        Args:
            detail: Error detail message
            
        Returns:
            JSON response with error details
        """
        return JSONResponse(
            status_code=HTTP_401_UNAUTHORIZED,
            content={
                "detail": detail,
                "type": "authentication_error"
            },
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    def _should_skip_auth(self, path: str) -> bool:
        """
        Determine if authentication should be skipped for a path.
        
        Args:
            path: Request path
            
        Returns:
            True if authentication should be skipped
        """
        # Public paths that don't require authentication
        public_paths = [
            "/health",
            "/api/v1/auth/token",
            "/api/v1/auth/login",      #  ADD THIS LINE!
            "/api/v1/auth/register",
            "/api/v1/auth/refresh",    #  ADD THIS TOO (for token refresh)
            "/api/v1/auth/password-reset-request",  #  ADD THIS (for password reset)
            "/docs",
            "/redoc",
            "/openapi.json",
            "/",  #  ADD ROOT PATH
        ]
        
        for public_path in public_paths:
            if path.startswith(public_path):
                return True
        
        return False
    
    def _requires_auth(self, path: str, method: str) -> bool:
        """
        Determine if a path requires authentication.
        
        Args:
            path: Request path
            method: HTTP method
            
        Returns:
            True if authentication is required
        """
        # API paths that require authentication
        if path.startswith("/api/v1/"):
            # Some endpoints might be public even under /api/
            public_endpoints = [
                "/api/v1/auth/token",
                "/api/v1/auth/login",
                "/api/v1/auth/register",
                "/api/v1/auth/refresh",
                "/api/v1/auth/password-reset-request",
                "/api/v1/market-data/public",
            ]
            
            for endpoint in public_endpoints:
                if path.startswith(endpoint):
                    return False
            
            # All other API endpoints require authentication
            return True
        
        return False