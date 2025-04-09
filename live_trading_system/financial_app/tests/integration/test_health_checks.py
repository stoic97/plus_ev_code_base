"""
Integration tests for health check system.

Tests the integration between the health check system and other
components of the application, such as database connections and APIs.
"""

import os
import unittest
from unittest.mock import patch, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.database import DatabaseType, PostgresDB, TimescaleDB, MongoDB, RedisDB
from app.monitoring.health_checks import (
    get_system_health,
    check_health,
    check_database_components,
    HealthStatus
)


#################################################
# Setup for Integration Tests
#################################################

# Set environment variable to use mocks in database.py
os.environ["TESTING"] = "True"

# Create a simple FastAPI app for testing
app = FastAPI()

@app.get("/health")
async def health_endpoint():
    """Simple health endpoint using the health check system."""
    return get_system_health()

@app.get("/health/{component}")
async def component_health_endpoint(component: str):
    """Component-specific health endpoint."""
    result = check_health(component=component)
    return result.to_dict()

@app.get("/health/category/{category}")
async def category_health_endpoint(category: str):
    """Category-specific health endpoint."""
    results = check_health(category=category)
    return {comp: result.to_dict() for comp, result in results.items()}

# Create a test client
client = TestClient(app)


#################################################
# Database Integration Tests
#################################################

class TestDatabaseIntegration(unittest.TestCase):
    """Test integration with database connections."""
    
    @patch('app.core.database.get_db_instance')
    def test_database_health_integration(self, mock_get_db_instance):
        """Test integration with database connections."""
        # Mock PostgreSQL database
        mock_postgres = MagicMock(spec=PostgresDB)
        mock_postgres.check_health.return_value = True
        mock_postgres.is_connected = True
        mock_postgres.get_status.return_value = {
            "name": "PostgresDB",
            "connected": True,
            "pool_size": 10,
            "pool_checkedin": 8,
            "pool_checkedout": 2
        }
        
        # Mock TimescaleDB database
        mock_timescale = MagicMock(spec=TimescaleDB)
        mock_timescale.check_health.return_value = True
        mock_timescale.is_connected = True
        mock_timescale.get_status.return_value = {
            "name": "TimescaleDB",
            "connected": True,
            "pool_size": 10,
            "pool_checkedin": 8,
            "pool_checkedout": 2
        }
        
        # Set up session mock for TimescaleDB
        session_mock = MagicMock()
        session_cm_mock = MagicMock()
        session_cm_mock.__enter__.return_value = session_mock
        mock_timescale.session.return_value = session_cm_mock
        
        # Mock execute result for TimescaleDB extension check
        execute_mock = MagicMock()
        session_mock.execute.return_value = execute_mock
        execute_mock.fetchall.return_value = [("timescaledb",)]
        
        # Mock MongoDB database
        mock_mongo = MagicMock(spec=MongoDB)
        mock_mongo.check_health.return_value = True
        mock_mongo.is_connected = True
        mock_mongo.get_status.return_value = {
            "name": "MongoDB",
            "connected": True,
            "server_version": "4.4.0",
            "connections": 10,
            "uptime_seconds": 3600
        }
        
        # Mock Redis database
        mock_redis = MagicMock(spec=RedisDB)
        mock_redis.check_health.return_value = True
        mock_redis.is_connected = True
        mock_redis.get_status.return_value = {
            "name": "RedisDB",
            "connected": True
        }
        mock_redis.set.return_value = True
        mock_redis.get.return_value = "test_value"
        mock_redis.delete.return_value = 1
        
        # Set up database instance mock to return different mocks based on database type
        def get_db_side_effect(db_type):
            if db_type == DatabaseType.POSTGRESQL:
                return mock_postgres
            elif db_type == DatabaseType.TIMESCALEDB:
                return mock_timescale
            elif db_type == DatabaseType.MONGODB:
                return mock_mongo
            elif db_type == DatabaseType.REDIS:
                return mock_redis
            return MagicMock()
        
        mock_get_db_instance.side_effect = get_db_side_effect
        
        # Test database health checks
        results = check_database_components()
        
        # Verify all databases were checked
        self.assertEqual(len(results), 4)
        self.assertEqual(results["postgresql"].status, HealthStatus.HEALTHY)
        self.assertEqual(results["timescaledb"].status, HealthStatus.HEALTHY)
        self.assertEqual(results["mongodb"].status, HealthStatus.HEALTHY)
        self.assertEqual(results["redis"].status, HealthStatus.HEALTHY)
        
        # Verify database methods were called
        mock_postgres.check_health.assert_called_once()
        mock_timescale.check_health.assert_called_once()
        mock_mongo.check_health.assert_called_once()
        mock_redis.check_health.assert_called_once()
        
        # Verify status was retrieved
        mock_postgres.get_status.assert_called_once()
        mock_timescale.get_status.assert_called_once()
        mock_mongo.get_status.assert_called_once()
        mock_redis.get_status.assert_called_once()


#################################################
# API Integration Tests
#################################################

class TestAPIIntegration(unittest.TestCase):
    """Test integration with API endpoints."""
    
    @patch('app.monitoring.health_checks.get_system_health')
    def test_health_endpoint(self, mock_get_system_health):
        """Test health endpoint."""
        # Mock system health
        mock_get_system_health.return_value = {
            "status": HealthStatus.HEALTHY,
            "timestamp": "2023-01-01T12:00:00",
            "components": {
                "postgresql": {
                    "component": "postgresql",
                    "status": HealthStatus.HEALTHY,
                    "timestamp": "2023-01-01T12:00:00",
                    "details": {"is_connected": True}
                }
            }
        }
        
        # Test health endpoint
        response = client.get("/health")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], HealthStatus.HEALTHY)
        self.assertIn("timestamp", data)
        self.assertIn("components", data)
        self.assertIn("postgresql", data["components"])
        
        # Verify mock was called
        mock_get_system_health.assert_called_once()
    
    @patch('app.monitoring.health_checks.check_health')
    def test_component_health_endpoint(self, mock_check_health):
        """Test component health endpoint."""
        # Mock component health
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {
            "component": "postgresql",
            "status": HealthStatus.HEALTHY,
            "timestamp": "2023-01-01T12:00:00",
            "details": {"is_connected": True}
        }
        mock_check_health.return_value = mock_result
        
        # Test component health endpoint
        response = client.get("/health/postgresql")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["component"], "postgresql")
        self.assertEqual(data["status"], HealthStatus.HEALTHY)
        
        # Verify mock was called with correct args
        mock_check_health.assert_called_once_with(component="postgresql")
    
    @patch('app.monitoring.health_checks.check_health')
    def test_category_health_endpoint(self, mock_check_health):
        """Test category health endpoint."""
        # Mock category health
        mock_result1 = MagicMock()
        mock_result1.to_dict.return_value = {
            "component": "postgresql",
            "status": HealthStatus.HEALTHY,
            "timestamp": "2023-01-01T12:00:00",
            "details": {"is_connected": True}
        }
        
        mock_result2 = MagicMock()
        mock_result2.to_dict.return_value = {
            "component": "redis",
            "status": HealthStatus.HEALTHY,
            "timestamp": "2023-01-01T12:00:00",
            "details": {"is_connected": True}
        }
        
        mock_check_health.return_value = {
            "postgresql": mock_result1,
            "redis": mock_result2
        }
        
        # Test category health endpoint
        response = client.get("/health/category/database")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("postgresql", data)
        self.assertIn("redis", data)
        
        # Verify mock was called with correct args
        mock_check_health.assert_called_once_with(category="database")


if __name__ == "__main__":
    unittest.main()