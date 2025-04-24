"""
Main API router for the Trading Strategies Application.

This module configures the API routes and endpoints for the application.
"""

from fastapi import APIRouter

# Import versioned routers
from app.api.v1.router import api_router as api_v1_router

# Main API router
api_router = APIRouter()

# Include versioned routers
api_router.include_router(api_v1_router)

@api_router.get("/debug/routes")
async def debug_routes():
    """List all registered routes for debugging."""
    # You can log or return this for debugging
    return {"message": "API router is configured"}