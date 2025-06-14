"""
Version 1 API router for the Trading Strategies Application.

This module configures the API endpoints for version 1 of the API.
"""

from fastapi import APIRouter



# Import endpoint modules
from app.api.v1.endpoints import (
    auth,
    health,
    strategy_management,
)


# Create the v1 API router
api_router = APIRouter()

# Include endpoint routers
api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
api_router.include_router(health.router, prefix="/health", tags=["Health"])

# Add the strategy management router
api_router.include_router(strategy_management.router, prefix="/strategies", tags=["Trading Strategies"])


from app.api.v1.endpoints import analytics

api_router.include_router(
    analytics.router,
    prefix="/analytics",
    tags=["analytics"]
)



# Add more endpoint routers here as they are implemented
# Example:
# api_router.include_router(users.router, prefix="/users", tags=["Users"])
# api_router.include_router(market_data.router, prefix="/market-data", tags=["Market Data"])
# api_router.include_router(orders.router, prefix="/orders", tags=["Orders"])