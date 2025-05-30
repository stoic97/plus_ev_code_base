"""
API Performance Tracking System

This module provides a decorator-based approach to track API endpoint performance metrics.
It integrates with the application's existing database infrastructure and supports both
sync and async endpoints.

Features:
- Tracks response time, status code, and endpoint details
- Uses existing database session management
- Supports both sync and async endpoints
- Optional file-based backup storage (CSV/Excel)
- Comprehensive error handling and logging
"""

import time
import logging
import json
import asyncio
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable, Dict, Optional, Union
import pandas as pd
from fastapi import Request, Response
from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.sql import text

from app.core.database import Base, get_db, DatabaseType, db_session

# Set up logging
logger = logging.getLogger(__name__)

class APIPerformanceMetric(Base):
    """SQLAlchemy model for storing API performance metrics."""
    __tablename__ = "api_performance_metrics"

    id = Column(Integer, primary_key=True, index=True)
    endpoint = Column(String, index=True)
    method = Column(String, index=True)
    status_code = Column(Integer)
    response_time = Column(Float)  # in seconds
    timestamp = Column(DateTime, default=datetime.utcnow)
    error = Column(String, nullable=True)

def ensure_metrics_table():
    """
    Ensure the metrics table exists in the database.
    This is called automatically when the module is imported.
    """
    try:
        logger.info("Attempting to create/verify api_performance_metrics table...")
        with db_session(DatabaseType.POSTGRESQL) as session:
            # Create all tables (this is safe to call multiple times)
            Base.metadata.create_all(bind=session.bind)
            
            # Verify table exists by trying to query it
            result = session.execute("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'api_performance_metrics')").scalar()
            if result:
                logger.info("api_performance_metrics table exists")
            else:
                logger.error("api_performance_metrics table was not created successfully")
                
    except Exception as e:
        logger.error(f"Failed to create metrics table: {e}")
        # Don't raise the exception - we want the application to continue running
        # even if performance tracking isn't available

def track_performance():
    """
    Decorator for tracking API endpoint performance.
    
    Usage:
        @app.get("/users")
        @track_performance()
        async def get_users():
            ...
    
    The decorator will:
    1. Record the start time
    2. Execute the endpoint
    3. Calculate the response time
    4. Store the metrics in the database
    5. Optionally export to CSV/Excel if configured
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            # Extract request object from args/kwargs
            request = next((arg for arg in args if isinstance(arg, Request)), 
                         kwargs.get('request'))
            
            if not request:
                logger.warning(f"No request object found for {func.__name__}")
                return await func(*args, **kwargs)
            
            logger.info(f"Starting performance tracking for async endpoint: {request.url.path}")
            start_time = time.time()
            error_info = None
            response = None
            
            try:
                # Execute the endpoint
                response = await func(*args, **kwargs)
                return response
            
            except Exception as e:
                error_info = str(e)
                logger.error(f"Error in endpoint {request.url.path}: {error_info}")
                raise
            
            finally:
                try:
                    # Calculate response time
                    response_time = time.time() - start_time
                    
                    # Get status code (if available)
                    status_code = None
                    if response is not None:
                        if isinstance(response, Response):
                            status_code = response.status_code
                        elif hasattr(response, 'status_code'):
                            status_code = response.status_code
                    
                    logger.info(f"Recording metric for {request.url.path} - Time: {response_time:.3f}s, Status: {status_code}")
                    
                    # Store in database using direct SQL
                    try:
                        with db_session(DatabaseType.POSTGRESQL) as session:
                            # Verify connection
                            logger.info("Verifying database connection...")
                            conn_check = session.execute(text("SELECT 1")).scalar()
                            logger.info(f"Database connection check: {conn_check}")
                            
                            # Verify table exists
                            logger.info("Verifying metrics table exists...")
                            table_check = session.execute(
                                text("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'api_performance_metrics')")
                            ).scalar()
                            logger.info(f"Table exists check: {table_check}")
                            
                            if not table_check:
                                logger.error("Table does not exist! Attempting to create...")
                                Base.metadata.create_all(bind=session.bind)
                            
                            # Start transaction
                            session.begin()
                            
                            # Prepare insert SQL
                            insert_sql = text("""
                                INSERT INTO api_performance_metrics 
                                (endpoint, method, status_code, response_time, timestamp, error)
                                VALUES (:endpoint, :method, :status_code, :response_time, :timestamp, :error)
                            """)
                            
                            # Prepare parameters
                            params = {
                                "endpoint": str(request.url.path),
                                "method": request.method,
                                "status_code": status_code,
                                "response_time": response_time,
                                "timestamp": datetime.utcnow(),
                                "error": error_info
                            }
                            
                            logger.info(f"Executing SQL: {insert_sql}\nWith params: {params}")
                            
                            # Execute insert
                            result = session.execute(insert_sql, params)
                            
                            # Commit transaction
                            session.commit()
                            logger.info(f"Successfully recorded performance metric. Rows affected: {result.rowcount}")
                            
                            # Verify the insert
                            verify_sql = text("""
                                SELECT COUNT(*) FROM api_performance_metrics 
                                WHERE endpoint = :endpoint 
                                AND method = :method 
                                AND response_time = :response_time
                            """)
                            verify_count = session.execute(verify_sql, params).scalar()
                            logger.info(f"Verification query shows {verify_count} matching records")
                            
                    except Exception as db_error:
                        logger.error(f"Database error while recording metric: {str(db_error)}")
                        if 'session' in locals():
                            session.rollback()
                    
                except Exception as e:
                    logger.error(f"Failed to record performance metric: {e}")
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            # Extract request object from args/kwargs
            request = next((arg for arg in args if isinstance(arg, Request)), 
                         kwargs.get('request'))
            
            if not request:
                logger.warning(f"No request object found for {func.__name__}")
                return func(*args, **kwargs)
            
            logger.info(f"Starting performance tracking for sync endpoint: {request.url.path}")
            start_time = time.time()
            error_info = None
            response = None
            
            try:
                # Execute the endpoint
                response = func(*args, **kwargs)
                return response
            
            except Exception as e:
                error_info = str(e)
                logger.error(f"Error in endpoint {request.url.path}: {error_info}")
                raise
            
            finally:
                try:
                    # Calculate response time
                    response_time = time.time() - start_time
                    
                    # Get status code (if available)
                    status_code = None
                    if response is not None:
                        if isinstance(response, Response):
                            status_code = response.status_code
                        elif hasattr(response, 'status_code'):
                            status_code = response.status_code
                    
                    logger.info(f"Recording metric for {request.url.path} - Time: {response_time:.3f}s, Status: {status_code}")
                    
                    # Store in database using direct SQL
                    try:
                        with db_session(DatabaseType.POSTGRESQL) as session:
                            # Verify connection
                            logger.info("Verifying database connection...")
                            conn_check = session.execute(text("SELECT 1")).scalar()
                            logger.info(f"Database connection check: {conn_check}")
                            
                            # Verify table exists
                            logger.info("Verifying metrics table exists...")
                            table_check = session.execute(
                                text("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'api_performance_metrics')")
                            ).scalar()
                            logger.info(f"Table exists check: {table_check}")
                            
                            if not table_check:
                                logger.error("Table does not exist! Attempting to create...")
                                Base.metadata.create_all(bind=session.bind)
                            
                            # Start transaction
                            session.begin()
                            
                            # Prepare insert SQL
                            insert_sql = text("""
                                INSERT INTO api_performance_metrics 
                                (endpoint, method, status_code, response_time, timestamp, error)
                                VALUES (:endpoint, :method, :status_code, :response_time, :timestamp, :error)
                            """)
                            
                            # Prepare parameters
                            params = {
                                "endpoint": str(request.url.path),
                                "method": request.method,
                                "status_code": status_code,
                                "response_time": response_time,
                                "timestamp": datetime.utcnow(),
                                "error": error_info
                            }
                            
                            logger.info(f"Executing SQL: {insert_sql}\nWith params: {params}")
                            
                            # Execute insert
                            result = session.execute(insert_sql, params)
                            
                            # Commit transaction
                            session.commit()
                            logger.info(f"Successfully recorded performance metric. Rows affected: {result.rowcount}")
                            
                            # Verify the insert
                            verify_sql = text("""
                                SELECT COUNT(*) FROM api_performance_metrics 
                                WHERE endpoint = :endpoint 
                                AND method = :method 
                                AND response_time = :response_time
                            """)
                            verify_count = session.execute(verify_sql, params).scalar()
                            logger.info(f"Verification query shows {verify_count} matching records")
                            
                    except Exception as db_error:
                        logger.error(f"Database error while recording metric: {str(db_error)}")
                        if 'session' in locals():
                            session.rollback()
                    
                except Exception as e:
                    logger.error(f"Failed to record performance metric: {e}")
        
        # Return appropriate wrapper based on whether the function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator

def export_metrics_to_csv(filepath: str, days: Optional[int] = None):
    """
    Export performance metrics to a CSV file.
    
    Args:
        filepath: Path to save the CSV file
        days: Optional number of days to limit the export (None for all data)
    """
    try:
        with db_session(DatabaseType.POSTGRESQL) as session:
            query = session.query(APIPerformanceMetric)
            
            if days:
                cutoff = datetime.utcnow() - timedelta(days=days)
                query = query.filter(APIPerformanceMetric.timestamp >= cutoff)
            
            metrics = query.all()
            
            if not metrics:
                logger.warning("No metrics found to export")
                return
            
            # Convert to pandas DataFrame
            data = [{
                'endpoint': m.endpoint,
                'method': m.method,
                'status_code': m.status_code,
                'response_time': m.response_time,
                'timestamp': m.timestamp,
                'error': m.error
            } for m in metrics]
            
            df = pd.DataFrame(data)
            df.to_csv(filepath, index=False)
            logger.info(f"Exported {len(metrics)} metrics to {filepath}")
            
    except Exception as e:
        logger.error(f"Failed to export metrics: {e}")
        raise

# Ensure the metrics table exists when the module is imported
ensure_metrics_table() 