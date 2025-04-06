"""
Health check endpoints for monitoring system components.
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Dict, Any

from app.core.database import MongoDB, PostgresDB, RedisDB, TimescaleDB

router = APIRouter()

# Get database instances
postgres_db = PostgresDB()
timescale_db = TimescaleDB()
mongo_db = MongoDB()
redis_db = RedisDB()


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str
    components: Dict[str, bool]
    details: Dict[str, Any] = {}


@router.get("/", response_model=HealthResponse)
async def health_check():
    """
    Check the health of all system components.
    
    Returns:
        Health status of all components
    """
    components = {
        "postgresql": postgres_db.check_health(),
        "timescaledb": timescale_db.check_health(),
        "mongodb": mongo_db.check_health(),
        "redis": redis_db.check_health(),
    }
    
    # Get more detailed status information
    details = {
        "postgresql": postgres_db.get_status(),
        "timescaledb": timescale_db.get_status(),
        "mongodb": mongo_db.get_status(),
        "redis": redis_db.get_status(),
    }
    
    # Overall status is healthy only if all components are healthy
    overall_status = "healthy" if all(components.values()) else "degraded"
    
    return HealthResponse(
        status=overall_status,
        components=components,
        details=details
    )


@router.get("/liveness")
async def liveness_check():
    """
    Simple liveness check to verify the API is responsive.
    
    Returns:
        Basic status message
    """
    return {"status": "alive"}


@router.get("/readiness")
async def readiness_check():
    """
    Readiness check to verify the application can handle requests.
    
    Verifies that essential services are available.
    
    Returns:
        Readiness status
    """
    # Check only critical components for readiness
    postgres_healthy = postgres_db.check_health()
    redis_healthy = redis_db.check_health()
    
    is_ready = postgres_healthy and redis_healthy
    
    if not is_ready:
        response = {"status": "not_ready", "details": {
            "postgresql": postgres_healthy,
            "redis": redis_healthy,
        }}
        return response
    
    return {"status": "ready"}