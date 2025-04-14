"""
Database migration integration for FastAPI.

This module provides utilities to check and run migrations during application startup,
as well as migration health check endpoints.
"""

import logging
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel

from app.core.security import get_current_user
from app.db.migrations.helpers.db_init import (
    check_all_databases, 
    initialize_database,
    run_migrations
)
from app.models.user import UserAuth

# Set up logging
logger = logging.getLogger(__name__)


# Pydantic models for API responses
class MigrationStatus(BaseModel):
    """Migration status response model."""
    database: str
    current: bool
    message: str


class MigrationResponse(BaseModel):
    """Migration response model."""
    ok: bool
    statuses: List[MigrationStatus]


class MigrationRunResponse(BaseModel):
    """Migration run response model."""
    success: bool
    database: str
    message: str


def get_migration_router() -> APIRouter:
    """
    Get FastAPI router for migration endpoints.
    
    Returns:
        APIRouter with migration endpoints
    """
    router = APIRouter(prefix="/migrations", tags=["Migrations"])
    
    @router.get("/status", response_model=MigrationResponse)
    async def migration_status(
        current_user: UserAuth = Depends(get_current_user)
    ) -> MigrationResponse:
        """
        Get database migration status.
        
        Args:
            current_user: Current authenticated user (must have admin role)
        
        Returns:
            Migration status response
        """
        # Check if user has admin role
        if not any(role.name == "admin" for role in current_user.roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to view migration status"
            )
        
        # Check all databases
        all_ok, statuses = check_all_databases()
        
        # Convert to response model
        response_statuses = [
            MigrationStatus(
                database=db,
                current="up to date" in status_msg.lower(),
                message=status_msg
            )
            for db, status_msg in statuses.items()
        ]
        
        return MigrationResponse(
            ok=all_ok,
            statuses=response_statuses
        )
    
    @router.post("/run", response_model=MigrationRunResponse)
    async def run_migration(
        database: str,
        current_user: UserAuth = Depends(get_current_user)
    ) -> MigrationRunResponse:
        """
        Run database migrations.
        
        Args:
            database: Database to run migrations for ('postgres' or 'timescale')
            current_user: Current authenticated user (must have admin role)
        
        Returns:
            Migration run response
        """
        # Check if user has admin role
        if not any(role.name == "admin" for role in current_user.roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to run migrations"
            )
        
        # Validate database
        if database not in ["postgres", "timescale"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid database. Must be 'postgres' or 'timescale'"
            )
        
        # Run migrations
        try:
            success = run_migrations(database)
            
            if success:
                message = f"Successfully migrated {database} database"
            else:
                message = f"Failed to migrate {database} database. Check logs for details."
            
            return MigrationRunResponse(
                success=success,
                database=database,
                message=message
            )
        
        except Exception as e:
            logger.error(f"Error running migrations for {database}: {e}")
            
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Migration error: {str(e)}"
            )
    
    @router.get("/health", status_code=status.HTTP_200_OK)
    async def migration_health() -> Response:
        """
        Check migration health for all databases.
        
        Returns:
            HTTP 200 if all migrations are up to date, HTTP 503 otherwise
        """
        # Check all databases
        all_ok, _ = check_all_databases()
        
        if all_ok:
            return Response(status_code=status.HTTP_200_OK)
        else:
            return Response(
                content="Migrations not up to date",
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE
            )
    
    return router


async def run_startup_migrations(auto_migrate: bool = True) -> None:
    """
    Run database migrations on application startup.
    
    Args:
        auto_migrate: Automatically run migrations if needed
    """
    logger.info("Checking database migrations on startup...")
    
    # Check all databases
    all_ok, statuses = check_all_databases(auto_upgrade=auto_migrate)
    
    for db, status_msg in statuses.items():
        logger.info(f"{db}: {status_msg}")
    
    if not all_ok and not auto_migrate:
        logger.warning("Database migrations are not up to date. Consider running migrations.")