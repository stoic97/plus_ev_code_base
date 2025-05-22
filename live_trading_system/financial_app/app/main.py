"""
Main application entry point for the Trading Strategies Application.
Sets up FastAPI app, routers, middleware, and event handlers.
"""

import logging
import uuid
from contextlib import asynccontextmanager
from typing import Callable, Dict, Optional

from fastapi import FastAPI, Request, Response, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.api.router import api_router
from app.core.config import settings
from app.core.database import MongoDB, PostgresDB, RedisDB, TimescaleDB
from app.core.error_handling import (
    DatabaseConnectionError,
    OperationalError,
    ValidationError,
    AuthenticationError,
    RateLimitExceededError,
)
from app.middleware.auth import AuthMiddleware
from app.middleware.error_middleware import ErrorHandlingMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Database connection state
db_state = {
    "postgres": {"connected": False, "required": True, "instance": None},
    "timescale": {"connected": False, "required": True, "instance": None},
    "mongodb": {"connected": False, "required": True, "instance": None},
    "redis": {"connected": False, "required": False, "instance": None},  # Redis is optional
}


# Request ID middleware
class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = str(uuid.uuid4())
        # Add request_id to request state
        request.state.request_id = request_id
        
        # Add request_id to response headers
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle events for startup and shutdown.
    Ensures proper database connections and cleanup.
    """
    # Startup: connect to databases
    logger.info("Starting application...")
    
    # Initialize database instances
    db_state["postgres"]["instance"] = PostgresDB()
    db_state["timescale"]["instance"] = TimescaleDB()
    db_state["mongodb"]["instance"] = MongoDB()
    db_state["redis"]["instance"] = RedisDB()
    
    # Connect to required databases
    critical_failure = False
    for db_name, db_info in db_state.items():
        db_instance = db_info["instance"]
        required = db_info["required"]
        
        try:
            logger.info(f"Connecting to {db_name}...")
            db_instance.connect()
            db_info["connected"] = True
            logger.info(f"Successfully connected to {db_name}")
        except Exception as e:
            logger.error(f"Error connecting to {db_name}: {e}")
            if required:
                critical_failure = True
                logger.critical(f"Failed to connect to required database: {db_name}")
            else:
                logger.warning(f"Non-critical database {db_name} is unavailable")
    
    # Fail fast if a critical database connection failed
    if critical_failure:
        logger.critical("Application startup failed due to missing critical database connections")
        raise DatabaseConnectionError("Failed to connect to one or more required databases")
    
    logger.info("Application started successfully")
    yield
    
    # Shutdown: close database connections
    logger.info("Shutting down application...")
    for db_name, db_info in db_state.items():
        if db_info["connected"] and db_info["instance"]:
            try:
                db_info["instance"].disconnect()
                logger.info(f"Disconnected from {db_name}")
            except Exception as e:
                logger.error(f"Error disconnecting from {db_name}: {e}")
    
    logger.info("All database connections closed")


# Database dependency functions
def get_postgres_db():
    """Dependency for PostgreSQL database access."""
    db = db_state["postgres"]["instance"]
    if not db or not db.is_connected:
        raise DatabaseConnectionError("PostgreSQL database is not connected")
    return db


def get_timescale_db():
    """Dependency for TimescaleDB database access."""
    db = db_state["timescale"]["instance"]
    if not db or not db.is_connected:
        raise DatabaseConnectionError("TimescaleDB database is not connected")
    return db


def get_mongodb_db():
    """Dependency for MongoDB database access."""
    db = db_state["mongodb"]["instance"]
    if not db or not db.is_connected:
        raise DatabaseConnectionError("MongoDB database is not connected")
    return db


def get_redis_db():
    """Dependency for Redis database access with fallback."""
    db = db_state["redis"]["instance"]
    if not db or not db.is_connected:
        logger.warning("Redis is not available, using fallback")
        return None  # Allow routes to implement fallback logic
    return db


def create_application() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="Trading Strategies Application API",
        lifespan=lifespan,
        debug=settings.DEBUG,
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.security.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add custom middleware (order matters - first added is last executed)
    app.add_middleware(ErrorHandlingMiddleware)
    app.add_middleware(AuthMiddleware)
    app.add_middleware(RequestIDMiddleware)
    
    # Register exception handlers
    @app.exception_handler(ValidationError)
    async def validation_exception_handler(request: Request, exc: ValidationError):
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": str(exc), "type": "validation_error", "request_id": getattr(request.state, "request_id", None)},
        )
    
    @app.exception_handler(AuthenticationError)
    async def auth_exception_handler(request: Request, exc: AuthenticationError):
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"detail": str(exc), "type": "authentication_error", "request_id": getattr(request.state, "request_id", None)},
        )
    
    @app.exception_handler(RateLimitExceededError)
    async def rate_limit_exception_handler(request: Request, exc: RateLimitExceededError):
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={"detail": str(exc), "type": "rate_limit_exceeded", "request_id": getattr(request.state, "request_id", None)},
        )
    
    @app.exception_handler(OperationalError)
    async def operational_exception_handler(request: Request, exc: OperationalError):
        logger.error(f"Operational error: {exc}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": str(exc), "type": "operational_error", "request_id": getattr(request.state, "request_id", None)},
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "An unexpected error occurred", "type": "server_error", "request_id": getattr(request.state, "request_id", None)},
        )
    
    # Include API router
    app.include_router(api_router, prefix=settings.API_PREFIX)
    
    # Add health check endpoint
    @app.get("/health", tags=["Health"])
    async def health_check():
        """Perform health check of critical system components."""
        health_status = {
            "status": "ok",
            "components": {},
            "timestamp": None,  # Will be added by middleware
        }
        
        # Check each database
        all_critical_healthy = True
        
        for db_name, db_info in db_state.items():
            is_healthy = False
            if db_info["connected"] and db_info["instance"]:
                try:
                    is_healthy = db_info["instance"].check_health()
                except Exception as e:
                    logger.error(f"Error checking health of {db_name}: {e}")
            
            health_status["components"][db_name] = {
                "status": "healthy" if is_healthy else "unhealthy",
                "required": db_info["required"]
            }
            
            # Update overall status if a critical component is unhealthy
            if db_info["required"] and not is_healthy:
                all_critical_healthy = False
        
        # Set overall status
        if not all_critical_healthy:
            health_status["status"] = "critical"
            return JSONResponse(status_code=503, content=health_status)
        
        # Check if any non-critical components are down
        if any(not info["components"][db]["status"] == "healthy" 
               for db, info in health_status["components"].items() 
               if not info["required"]):
            health_status["status"] = "degraded"
        
        return health_status
    
    @app.get("/", tags=["Root"])
    async def root():
        """Root endpoint that redirects to documentation."""
        return {"message": f"Welcome to {settings.APP_NAME}. See /docs for API documentation."}
    
    return app


app = create_application()


if __name__ == "__main__":
    import uvicorn
    
    # Run the application with uvicorn when script is executed directly
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        workers=settings.performance.WORKERS if not settings.DEBUG else 1,
    )