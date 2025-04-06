"""
Error handling middleware for the Trading Strategies Application.

This middleware provides centralized error processing and creates
consistent error responses across the API.
"""

import logging
import time
import traceback
from typing import Dict, Any

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.status import HTTP_500_INTERNAL_SERVER_ERROR

from app.core.error_handling import AppError, ErrorCategory, track_error

logger = logging.getLogger(__name__)


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for handling and standardizing errors in the application.
    
    This middleware catches exceptions that occur during request processing
    and formats them into consistent error responses.
    """
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """
        Process the request and handle any errors.
        
        Args:
            request: The incoming request
            call_next: The next middleware or route handler
            
        Returns:
            The response with standardized error handling
        """
        start_time = time.time()
        request_id = request.headers.get("X-Request-ID", "")
        
        # Add request_id to the request state for logging
        request.state.request_id = request_id
        
        try:
            response = await call_next(request)
            
            # Log timing information for performance monitoring
            process_time = (time.time() - start_time) * 1000
            response.headers["X-Process-Time"] = f"{process_time:.2f}ms"
            
            # If we have a request_id, include it in the response
            if request_id:
                response.headers["X-Request-ID"] = request_id
            
            return response
            
        except AppError as app_err:
            # Log and track our custom application errors
            app_err.log_error()
            track_error(app_err)
            
            # Convert to HTTP response
            return self._create_error_response(
                app_err.status_code,
                app_err.detail.message,
                app_err.detail.error_code,
                request_id,
                app_err.detail.error_category
            )
            
        except Exception as e:
            # For unknown exceptions, log the full traceback
            logger.error(
                f"Unhandled exception in request {request.method} {request.url.path}: {str(e)}",
                exc_info=True
            )
            
            # Track the error for monitoring
            track_error(e, ErrorCategory.UNKNOWN)
            
            # Get traceback for debugging
            error_traceback = traceback.format_exc()
            
            # In production, don't send traceback details to clients
            include_traceback = request.app.debug
            
            return self._create_error_response(
                HTTP_500_INTERNAL_SERVER_ERROR,
                f"Internal server error: {str(e)}",
                "internal_error",
                request_id,
                ErrorCategory.UNKNOWN,
                error_traceback if include_traceback else None
            )
    
    def _create_error_response(
        self,
        status_code: int,
        message: str,
        error_code: str = None,
        request_id: str = None,
        error_category: str = None,
        traceback: str = None,
    ) -> JSONResponse:
        """
        Create a standardized error response.
        
        Args:
            status_code: HTTP status code
            message: Error message
            error_code: Optional error code
            request_id: Optional request ID for tracing
            error_category: Error category for classification
            traceback: Optional traceback for debugging
            
        Returns:
            JSON response with error details
        """
        error_body: Dict[str, Any] = {
            "detail": message,
            "type": error_category or "unknown",
        }
        
        if error_code:
            error_body["code"] = error_code
            
        if request_id:
            error_body["request_id"] = request_id
            
        if traceback:
            error_body["traceback"] = traceback
        
        response = JSONResponse(
            status_code=status_code,
            content=error_body
        )
        
        # Add request ID to headers for easier debugging
        if request_id:
            response.headers["X-Request-ID"] = request_id
        
        return response