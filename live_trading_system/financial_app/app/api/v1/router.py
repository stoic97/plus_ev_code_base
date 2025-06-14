"""
Version 1 API router for the Trading Strategies Application.

This module configures the API endpoints for version 1 of the API.
Updated to use paper_trading_complete for all trading endpoints.
"""

from fastapi import APIRouter

# Import endpoint modules
from app.api.v1.endpoints import (
    auth,
    health,
)

# Try to import paper_trading_complete which has all trading endpoints
try:
    from app.api.v1.endpoints import paper_trading_complete
    PAPER_TRADING_COMPLETE_AVAILABLE = True
except ImportError:
    PAPER_TRADING_COMPLETE_AVAILABLE = False
    # Fallback imports
    try:
        from app.api.v1.endpoints import (
            strategy_management,
            analytics,
        )
        FALLBACK_AVAILABLE = True
    except ImportError:
        FALLBACK_AVAILABLE = False

# Create the v1 API router
api_router = APIRouter()

# Include auth and health routers
api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
api_router.include_router(health.router, prefix="/health", tags=["Health"])

# Use paper_trading_complete if available, otherwise use fallback
if PAPER_TRADING_COMPLETE_AVAILABLE:
    # Include paper_trading_complete router which has ALL endpoints we need
    api_router.include_router(
        paper_trading_complete.router, 
        prefix="",  # No prefix as endpoints already have their paths
        tags=["Paper Trading"]
    )
elif FALLBACK_AVAILABLE:
    # Fallback to original setup if paper_trading_complete is not available
    api_router.include_router(
        strategy_management.router, 
        prefix="/strategies", 
        tags=["Trading Strategies"]
    )
    api_router.include_router(
        analytics.router,
        prefix="",
        tags=["Analytics"]
    )

# Future endpoint routers can be added here
# api_router.include_router(users.router, prefix="/users", tags=["Users"])
# api_router.include_router(market_data.router, prefix="/market-data", tags=["Market Data"])