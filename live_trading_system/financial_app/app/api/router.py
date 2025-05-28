"""
Main API router for the Trading Strategies Application.

This module configures the API routes and endpoints for the application.
"""

from fastapi import APIRouter

# Import versioned routers
from app.api.v1.router import api_router as api_v1_router

# Main API router
api_router = APIRouter(prefix="/api/v1")



# Include versioned routers
api_router.include_router(api_v1_router)