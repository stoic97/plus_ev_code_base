"""
Unit tests for health check system.

Tests the components of the app/monitoring/health_checks.py module,
including the health status definitions, caching, registry, and
individual health checks.
"""

import asyncio
import datetime
import time
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from app.core.database import DatabaseType
from app.core.error_handling import AppError
from app.monitoring.health_checks import (
    # Health status definitions
    HealthStatus,
    HealthCheckResult,
    
    # Caching and configuration
    cached_health_check,
    HealthCheckConfig,
    get_health_config,
    
    # Registry
    HealthCheckRegistry,
    get_registry,
    
    # API functions
    check_health,
    get_system_health,
    check_critical_components,
    
    # Database health checks
    check_postgres_health,
    check_timescaledb_health,
    check_mongodb_health,
    check_redis_health,
    
    # Other health checks
    check_kafka_health,
    check_system_resources,
    check_app_status,
    
    # Aggregation functions
    check_database_components,
    check_all_async,
    
    # Registration
    register_default_checks,
)


#################################################
# Health Status Tests
#################################################

class TestHealthStatus(unittest.TestCase):
    """Test health status enum and result class."""
    
    def test_health_status_values(self):
        """Test health status enum values."""
        self.assertEqual(HealthStatus.HEALTHY, "healthy")
        self.assertEqual(HealthStatus.DEGRADED, "degraded")
        self.assertEqual(HealthStatus.UNHEALTHY, "unhealthy")
        self.assertEqual(HealthStatus.UNKNOWN, "unknown")
    
    def test_health_check_result_init(self):
        """Test health check result initialization."""
        # Test with minimal arguments
        result = HealthCheckResult(
            component="test",
            status=HealthStatus.HEALTHY
        )
        self.assertEqual(result.component, "test")
        self.assertEqual(result.status, HealthStatus.HEALTHY)
        self.assertEqual(result.details, {})
        self.assertIsNone(result.error)
        self.assertIsInstance(result.timestamp, datetime.datetime)
        
        # Test with all arguments
        error = Exception("Test error")
        timestamp = datetime.datetime(2023, 1, 1, 12, 0, 0)
        details = {"key": "value"}
        
        result = HealthCheckResult(
            component="test",
            status=HealthStatus.UNHEALTHY,
            details=details,
            error=error,
            timestamp=timestamp
        )
        self.assertEqual(result.component, "test")
        self.assertEqual(result.status, HealthStatus.UNHEALTHY)
        self.assertEqual(result.details, details)
        self.assertEqual(result.error, error)
        self.assertEqual(result.timestamp, timestamp)
    
    def test_health_check_result_to_dict(self):
        """Test health check result to_dict method."""
        # Test without error
        result = HealthCheckResult(
            component="test",
            status=HealthStatus.HEALTHY,
            details={"key": "value"},
            timestamp=datetime.datetime(2023, 1, 1, 12, 0, 0)
        )
        
        expected = {
            "component": "test",
            "status": HealthStatus.HEALTHY,
            "timestamp": "2023-01-01T12:00:00",
            "details": {"key": "value"}
        }
        
        self.assertEqual(result.to_dict(), expected)
        
        # Test with error
        error = Exception("Test error")
        result = HealthCheckResult(
            component="test",
            status=HealthStatus.UNHEALTHY,
            details={"key": "value"},
            error=error,
            timestamp=datetime.datetime(2023, 1, 1, 12, 0, 0)
        )
        
        expected = {
            "component": "test",
            "status": HealthStatus.UNHEALTHY,
            "timestamp": "2023-01-01T12:00:00",
            "details": {"key": "value"},
            "error": str(error)
        }
        
        self.assertEqual(result.to_dict(), expected)


#################################################
# Caching Tests
#################################################

class TestCachingSystem(unittest.TestCase):
    """Test health check caching system."""
    
    def test_cached_health_check_decorator(self):
        """Test the cached_health_check decorator."""
        call_count = 0
        
        @cached_health_check(ttl_seconds=1)
        def test_check():
            nonlocal call_count
            call_count += 1
            return HealthCheckResult(
                component="test",
                status=HealthStatus.HEALTHY
            )
        
        # First call should execute the function
        result1 = test_check()
        self.assertEqual(call_count, 1)
        self.assertEqual(result1.status, HealthStatus.HEALTHY)
        
        # Second call within TTL should use cached result
        result2 = test_check()
        self.assertEqual(call_count, 1)  # Still 1
        self.assertEqual(result2.status, HealthStatus.HEALTHY)
        
        # Wait for TTL to expire
        time.sleep(1.1)
        
        # Call after TTL should execute the function again
        result3 = test_check()
        self.assertEqual(call_count, 2)
        self.assertEqual(result3.status, HealthStatus.HEALTHY)
        
        # Test force_refresh
        result4 = test_check.force_refresh()
        self.assertEqual(call_count, 3)
        self.assertEqual(result4.status, HealthStatus.HEALTHY)
    
    @patch('app.monitoring.health_checks.get_settings')
    def test_health_config(self, mock_get_settings):
        """Test health check configuration."""
        # Create mock settings
        mock_settings = MagicMock()
        mock_settings.monitoring = MagicMock()
        mock_settings.monitoring.DATABASE_CHECK_INTERVAL = 120
        mock_settings.monitoring.REDIS_CHECK_INTERVAL = 60
        mock_get_settings.return_value = mock_settings
        
        # Reset singleton for testing
        import app.monitoring.health_checks
        app.monitoring.health_checks._health_config = None
        
        # Get config
        config = get_health_config()
        
        # Check default values
        self.assertEqual(config.database_check_interval, 120)
        self.assertEqual(config.redis_check_interval, 60)
        self.assertEqual(config.memory_warning_threshold, 80)
        
        # Should return the same instance (singleton)
        config2 = get_health_config()
        self.assertIs(config, config2)


#################################################
# Registry Tests
#################################################

class TestHealthCheckRegistry(unittest.TestCase):
    """Test health check registry."""
    
    def setUp(self):
        """Set up test case."""
        self.registry = HealthCheckRegistry()
        
        # Define some test health check functions
        def check_healthy():
            return HealthCheckResult(
                component="healthy_component",
                status=HealthStatus.HEALTHY
            )
        
        def check_degraded():
            return HealthCheckResult(
                component="degraded_component",
                status=HealthStatus.DEGRADED
            )
        
        def check_unhealthy():
            return HealthCheckResult(
                component="unhealthy_component",
                status=HealthStatus.UNHEALTHY
            )
        
        def check_error():
            raise Exception("Test error")
        
        # Register test health checks
        self.registry.register(
            component="healthy_component",
            categories=["test", "critical"],
            check_func=check_healthy
        )
        
        self.registry.register(
            component="degraded_component",
            categories=["test"],
            check_func=check_degraded
        )
        
        self.registry.register(
            component="unhealthy_component",
            categories=["test", "critical"],
            check_func=check_unhealthy
        )
        
        self.registry.register(
            component="error_component",
            categories=["test"],
            check_func=check_error
        )
    
    def test_register_and_get_components(self):
        """Test component registration and retrieval."""
        components = self.registry.get_check_components()
        self.assertEqual(len(components), 4)
        self.assertIn("healthy_component", components)
        self.assertIn("degraded_component", components)
        self.assertIn("unhealthy_component", components)
        self.assertIn("error_component", components)
    
    def test_get_categories(self):
        """Test category retrieval."""
        categories = self.registry.get_categories()
        self.assertEqual(len(categories), 2)
        self.assertIn("test", categories)
        self.assertIn("critical", categories)
    
    def test_get_components_by_category(self):
        """Test getting components by category."""
        test_components = self.registry.get_components_by_category("test")
        self.assertEqual(len(test_components), 4)
        
        critical_components = self.registry.get_components_by_category("critical")
        self.assertEqual(len(critical_components), 2)
        self.assertIn("healthy_component", critical_components)
        self.assertIn("unhealthy_component", critical_components)
        
        # Test non-existent category
        empty_components = self.registry.get_components_by_category("nonexistent")
        self.assertEqual(len(empty_components), 0)
    
    def test_check_component(self):
        """Test checking a specific component."""
        # Test healthy component
        result = self.registry.check_component("healthy_component")
        self.assertEqual(result.status, HealthStatus.HEALTHY)
        
        # Test degraded component
        result = self.registry.check_component("degraded_component")
        self.assertEqual(result.status, HealthStatus.DEGRADED)
        
        # Test unhealthy component
        result = self.registry.check_component("unhealthy_component")
        self.assertEqual(result.status, HealthStatus.UNHEALTHY)
        
        # Test component with error
        result = self.registry.check_component("error_component")
        self.assertEqual(result.status, HealthStatus.UNHEALTHY)
        self.assertIsNotNone(result.error)
        
        # Test non-existent component
        with self.assertRaises(ValueError):
            self.registry.check_component("nonexistent")
    
    def test_check_category(self):
        """Test checking a category of components."""
        # Test critical category
        results = self.registry.check_category("critical")
        self.assertEqual(len(results), 2)
        self.assertEqual(results["healthy_component"].status, HealthStatus.HEALTHY)
        self.assertEqual(results["unhealthy_component"].status, HealthStatus.UNHEALTHY)
        
        # Test non-existent category
        results = self.registry.check_category("nonexistent")
        self.assertEqual(len(results), 0)
    
    def test_check_all(self):
        """Test checking all components."""
        results = self.registry.check_all()
        self.assertEqual(len(results), 4)
        self.assertEqual(results["healthy_component"].status, HealthStatus.HEALTHY)
        self.assertEqual(results["degraded_component"].status, HealthStatus.DEGRADED)
        self.assertEqual(results["unhealthy_component"].status, HealthStatus.UNHEALTHY)
        self.assertEqual(results["error_component"].status, HealthStatus.UNHEALTHY)
    
    def test_check_system(self):
        """Test system status check."""
        # Reset system status
        self.registry._system_status = HealthStatus.UNKNOWN
        
        # Check system
        status, results = self.registry.check_system()
        
        # With our test components (1 healthy, 1 degraded, 2 unhealthy)
        # System status should be UNHEALTHY
        self.assertEqual(status, HealthStatus.UNHEALTHY)
        self.assertEqual(len(results), 4)
    
    def test_update_system_status(self):
        """Test system status update logic."""
        # Test with no results
        self.registry._update_system_status({})
        self.assertEqual(self.registry._system_status, HealthStatus.UNKNOWN)
        
        # Test with all healthy
        self.registry._update_system_status({
            "comp1": HealthCheckResult("comp1", HealthStatus.HEALTHY),
            "comp2": HealthCheckResult("comp2", HealthStatus.HEALTHY)
        })
        self.assertEqual(self.registry._system_status, HealthStatus.HEALTHY)
        
        # Test with some degraded
        self.registry._update_system_status({
            "comp1": HealthCheckResult("comp1", HealthStatus.HEALTHY),
            "comp2": HealthCheckResult("comp2", HealthStatus.DEGRADED)
        })
        self.assertEqual(self.registry._system_status, HealthStatus.DEGRADED)
        
        # Test with a few unhealthy (less than 25%)
        self.registry._update_system_status({
            "comp1": HealthCheckResult("comp1", HealthStatus.HEALTHY),
            "comp2": HealthCheckResult("comp2", HealthStatus.HEALTHY),
            "comp3": HealthCheckResult("comp3", HealthStatus.HEALTHY),
            "comp4": HealthCheckResult("comp4", HealthStatus.HEALTHY),
            "comp5": HealthCheckResult("comp5", HealthStatus.UNHEALTHY)
        })
        self.assertEqual(self.registry._system_status, HealthStatus.DEGRADED)
        
        # Test with many unhealthy (more than 25%)
        self.registry._update_system_status({
            "comp1": HealthCheckResult("comp1", HealthStatus.HEALTHY),
            "comp2": HealthCheckResult("comp2", HealthStatus.HEALTHY),
            "comp3": HealthCheckResult("comp3", HealthStatus.UNHEALTHY),
            "comp4": HealthCheckResult("comp4", HealthStatus.UNHEALTHY)
        })
        self.assertEqual(self.registry._system_status, HealthStatus.UNHEALTHY)
        
        # Test with all unknown
        self.registry._update_system_status({
            "comp1": HealthCheckResult("comp1", HealthStatus.UNKNOWN),
            "comp2": HealthCheckResult("comp2", HealthStatus.UNKNOWN)
        })
        self.assertEqual(self.registry._system_status, HealthStatus.UNKNOWN)


#################################################
# API Function Tests
#################################################

class TestAPIFunctions(unittest.TestCase):
    """Test public API functions."""
    
    @patch('app.monitoring.health_checks.get_registry')
    def test_check_health(self, mock_get_registry):
        """Test check_health function."""
        # Mock registry
        mock_registry = MagicMock()
        mock_get_registry.return_value = mock_registry
        
        # Test check_health with component
        mock_result = HealthCheckResult("test", HealthStatus.HEALTHY)
        mock_registry.check_component.return_value = mock_result
        
        result = check_health(component="test")
        self.assertEqual(result, mock_result)
        mock_registry.check_component.assert_called_once_with("test")
        
        # Test check_health with category
        mock_registry.reset_mock()
        mock_results = {"comp1": mock_result}
        mock_registry.check_category.return_value = mock_results
        
        result = check_health(category="test_category")
        self.assertEqual(result, mock_results)
        mock_registry.check_category.assert_called_once_with("test_category")
        
        # Test check_health with no args
        mock_registry.reset_mock()
        mock_registry.check_all.return_value = mock_results
        
        result = check_health()
        self.assertEqual(result, mock_results)
        mock_registry.check_all.assert_called_once()
    
    @patch('app.monitoring.health_checks.get_registry')
    def test_get_system_health(self, mock_get_registry):
        """Test get_system_health function."""
        # Mock registry
        mock_registry = MagicMock()
        mock_get_registry.return_value = mock_registry
        
        # Mock results
        mock_result = HealthCheckResult("test", HealthStatus.HEALTHY)
        mock_registry.check_system.return_value = (HealthStatus.HEALTHY, {"test": mock_result})
        
        # Test get_system_health
        result = get_system_health()
        
        self.assertEqual(result["status"], HealthStatus.HEALTHY)
        self.assertIn("timestamp", result)
        self.assertIn("components", result)
        self.assertIn("test", result["components"])
        
        # Test with force_refresh
        mock_registry.reset_mock()
        result = get_system_health(force_refresh=True)
        
        mock_registry.check_system.assert_called_once_with(True)
    
    @patch('app.monitoring.health_checks.get_registry')
    @patch('app.monitoring.health_checks.log_critical_error')
    def test_check_critical_components(self, mock_log_critical, mock_get_registry):
        """Test check_critical_components function."""
        # Mock registry
        mock_registry = MagicMock()
        mock_get_registry.return_value = mock_registry
        
        # Test with all healthy components
        mock_registry.check_category.return_value = {
            "comp1": HealthCheckResult("comp1", HealthStatus.HEALTHY),
            "comp2": HealthCheckResult("comp2", HealthStatus.HEALTHY)
        }
        
        result = check_critical_components()
        
        self.assertEqual(result["status"], HealthStatus.HEALTHY)
        self.assertIn("timestamp", result)
        self.assertIn("components", result)
        self.assertEqual(len(result["components"]), 2)
        mock_log_critical.assert_not_called()
        
        # Test with unhealthy components
        mock_registry.reset_mock()
        mock_log_critical.reset_mock()
        
        mock_registry.check_category.return_value = {
            "comp1": HealthCheckResult("comp1", HealthStatus.HEALTHY),
            "comp2": HealthCheckResult("comp2", HealthStatus.UNHEALTHY)
        }
        
        result = check_critical_components()
        
        self.assertEqual(result["status"], HealthStatus.UNHEALTHY)
        self.assertIn("timestamp", result)
        self.assertIn("components", result)
        self.assertEqual(len(result["components"]), 2)
        mock_log_critical.assert_called_once()


#################################################
# Database Health Check Tests
#################################################

class TestDatabaseHealthChecks(unittest.TestCase):
    """Test database health checks."""
    
    @patch('app.monitoring.health_checks.get_db_instance')
    def test_check_postgres_health(self, mock_get_db_instance):
        """Test PostgreSQL health check."""
        # Mock PostgreSQL database
        mock_db = MagicMock()
        mock_db.check_health.return_value = True
        mock_db.is_connected = True
        mock_db.get_status.return_value = {
            "name": "PostgresDB",
            "connected": True,
            "pool_size": 10,
            "pool_checkedin": 8,
            "pool_checkedout": 2
        }
        mock_get_db_instance.return_value = mock_db
        
        # Test check_postgres_health
        result = check_postgres_health()
        
        self.assertEqual(result.component, "postgresql")
        self.assertEqual(result.status, HealthStatus.HEALTHY)
        self.assertTrue(result.details["is_connected"])
        self.assertIn("pool_info", result.details)
        self.assertIn("connection_usage_pct", result.details)
        
        # Test with unhealthy status
        mock_db.check_health.return_value = False
        
        result = check_postgres_health()
        
        self.assertEqual(result.status, HealthStatus.UNHEALTHY)
        
        # Test with high connection usage
        mock_db.check_health.return_value = True
        mock_db.get_status.return_value = {
            "name": "PostgresDB",
            "connected": True,
            "pool_size": 10,
            "pool_checkedin": 1,
            "pool_checkedout": 9
        }
        
        result = check_postgres_health()
        
        self.assertEqual(result.status, HealthStatus.DEGRADED)
        
        # Test with exception
        mock_db.check_health.side_effect = Exception("Test error")
        
        result = check_postgres_health()
        
        self.assertEqual(result.status, HealthStatus.UNHEALTHY)
        self.assertIn("error", result.details)
    
    @patch('app.monitoring.health_checks.get_db_instance')
    def test_check_redis_health(self, mock_get_db_instance):
        """Test Redis health check."""
        # Mock Redis database
        mock_db = MagicMock()
        mock_db.check_health.return_value = True
        mock_db.is_connected = True
        mock_db.get_status.return_value = {
            "name": "RedisDB",
            "connected": True
        }
        mock_db.set.return_value = True
        mock_db.get.return_value = "test_value"
        mock_db.delete.return_value = 1
        mock_get_db_instance.return_value = mock_db
        
        # Test check_redis_health
        result = check_redis_health()
        
        self.assertEqual(result.component, "redis")
        self.assertEqual(result.status, HealthStatus.HEALTHY)
        self.assertTrue(result.details["is_connected"])
        self.assertIn("operation_time_ms", result.details)
        self.assertIn("values_match", result.details)
        
        # Test with values not matching
        mock_db.get.return_value = "wrong_value"
        
        result = check_redis_health()
        
        self.assertEqual(result.status, HealthStatus.DEGRADED)
        self.assertEqual(result.details["message"], "Redis data integrity issue")
        
        # Test with slow operation
        mock_db.get.return_value = "test_value"
        
        # Mock slow set/get operation
        original_time = time.time
        time_counter = [0]
        
        def mock_time():
            if time_counter[0] == 0:
                time_counter[0] = 1
                return 0
            else:
                return 0.2  # 200ms
        
        with patch('time.time', side_effect=mock_time):
            result = check_redis_health()
            
            self.assertEqual(result.status, HealthStatus.DEGRADED)
            self.assertEqual(result.details["message"], "Redis operations are slow")
        
        # Test with connection failure
        mock_db.check_health.return_value = False
        
        result = check_redis_health()
        
        self.assertEqual(result.status, HealthStatus.UNHEALTHY)


#################################################
# Kafka Health Check Tests
#################################################

class TestKafkaHealthCheck(unittest.TestCase):
    """Test Kafka health checks."""
    
    @patch('app.monitoring.health_checks.get_settings')
    def test_check_kafka_health(self, mock_get_settings):
        """Test Kafka health check."""
        # Mock settings
        mock_settings = MagicMock()
        mock_kafka_settings = MagicMock()
        mock_kafka_settings.BOOTSTRAP_SERVERS = ["localhost:9092"]
        mock_kafka_settings.MARKET_DATA_TOPIC = "market-data"
        mock_kafka_settings.SIGNAL_TOPIC = "trading-signals"
        mock_settings.kafka = mock_kafka_settings
        mock_get_settings.return_value = mock_settings
        
        # Test check_kafka_health
        result = check_kafka_health()
        
        self.assertEqual(result.component, "kafka")
        self.assertEqual(result.status, HealthStatus.HEALTHY)
        self.assertEqual(result.details["bootstrap_servers"], ["localhost:9092"])
        self.assertEqual(result.details["topics"], ["market-data", "trading-signals"])
        
        # Test with no bootstrap servers
        mock_kafka_settings.BOOTSTRAP_SERVERS = []
        
        result = check_kafka_health()
        
        self.assertEqual(result.status, HealthStatus.UNHEALTHY)
        self.assertEqual(result.details["message"], "No Kafka bootstrap servers configured")


#################################################
# System Resource Tests
#################################################

class TestSystemResourceCheck(unittest.TestCase):
    """Test system resource check."""
    
    @patch('app.monitoring.health_checks.get_health_config')
    def test_check_system_resources(self, mock_get_health_config):
        """Test system resource check."""
        # Mock config
        mock_config = MagicMock()
        mock_config.cpu_warning_threshold = 80
        mock_config.memory_warning_threshold = 80
        mock_config.disk_warning_threshold = 80
        mock_get_health_config.return_value = mock_config
        
        # Test with all resources healthy
        result = check_system_resources()
        
        self.assertEqual(result.component, "system_resources")
        self.assertEqual(result.status, HealthStatus.HEALTHY)
        self.assertIn("cpu_usage_pct", result.details)
        self.assertIn("memory_usage_pct", result.details)
        self.assertIn("disk_usage_pct", result.details)
        
        # Test with implementation exception
        with patch('app.monitoring.health_checks.check_system_resources.force_refresh',
                  side_effect=Exception("Test error")):
            result = check_system_resources.force_refresh()
            
            self.assertEqual(result.status, HealthStatus.UNHEALTHY)
            self.assertIn("error", result.details)


#################################################
# Application Status Test
#################################################

class TestAppStatusCheck(unittest.TestCase):
    """Test application status check."""
    
    @patch('app.monitoring.health_checks.get_settings')
    def test_check_app_status(self, mock_get_settings):
        """Test application status check."""
        # Mock settings
        mock_settings = MagicMock()
        mock_settings.APP_NAME = "Trading Strategies Application"
        mock_settings.APP_VERSION = "0.1.0"
        mock_settings.ENV = "development"
        mock_settings.DEBUG = True
        mock_get_settings.return_value = mock_settings
        
        # Test check_app_status
        result = check_app_status()
        
        self.assertEqual(result.component, "application")
        self.assertEqual(result.status, HealthStatus.HEALTHY)
        self.assertEqual(result.details["name"], "Trading Strategies Application")
        self.assertEqual(result.details["version"], "0.1.0")
        self.assertEqual(result.details["environment"], "development")
        self.assertTrue(result.details["debug_mode"])


#################################################
# Aggregate Function Tests
#################################################

class TestAggregationFunctions(unittest.TestCase):
    """Test health check aggregation functions."""
    
    @patch('app.monitoring.health_checks.check_postgres_health')
    @patch('app.monitoring.health_checks.check_timescaledb_health')
    @patch('app.monitoring.health_checks.check_mongodb_health')
    @patch('app.monitoring.health_checks.check_redis_health')
    def test_check_database_components(
        self, mock_check_redis, mock_check_mongodb, 
        mock_check_timescaledb, mock_check_postgres
    ):
        """Test database components check."""
        # Mock health check results
        postgres_result = HealthCheckResult("postgresql", HealthStatus.HEALTHY)
        timescaledb_result = HealthCheckResult("timescaledb", HealthStatus.HEALTHY)
        mongodb_result = HealthCheckResult("mongodb", HealthStatus.HEALTHY)
        redis_result = HealthCheckResult("redis", HealthStatus.HEALTHY)
        
        mock_check_postgres.return_value = postgres_result
        mock_check_timescaledb.return_value = timescaledb_result
        mock_check_mongodb.return_value = mongodb_result
        mock_check_redis.return_value = redis_result
        
        # Test check_database_components
        results = check_database_components()
        
        self.assertEqual(len(results), 4)
        self.assertEqual(results["postgresql"], postgres_result)
        self.assertEqual(results["timescaledb"], timescaledb_result)
        self.assertEqual(results["mongodb"], mongodb_result)
        self.assertEqual(results["redis"], redis_result)
    
    @pytest.mark.asyncio
    @patch('app.monitoring.health_checks.get_registry')
    async def test_check_all_async(self, mock_get_registry):
        """Test asynchronous check of all components."""
        # Mock registry
        mock_registry = MagicMock()
        mock_get_registry.return_value = mock_registry
        
        # Mock component checks
        mock_registry.get_check_components.return_value = ["comp1", "comp2", "comp3"]
        
        def mock_check_component(component):
            if component == "comp1":
                return HealthCheckResult(component, HealthStatus.HEALTHY)
            elif component == "comp2":
                return HealthCheckResult(component, HealthStatus.DEGRADED)
            elif component == "comp3":
                raise Exception("Test error")
        
        mock_registry.check_component.side_effect = mock_check_component
        
        # Test check_all_async
        results = await check_all_async()
        
        self.assertEqual(len(results), 3)
        self.assertEqual(results["comp1"].status, HealthStatus.HEALTHY)
        self.assertEqual(results["comp2"].status, HealthStatus.DEGRADED)
        self.assertEqual(results["comp3"].status, HealthStatus.UNHEALTHY)
        self.assertIn("error", results["comp3"].details)


#################################################
# Registration Test
#################################################

class TestRegistration(unittest.TestCase):
    """Test health check registration."""
    
    def test_register_default_checks(self):
        """Test default health check registration."""
        # Create registry
        registry = HealthCheckRegistry()
        
        # Register default checks
        register_default_checks(registry)
        
        # Verify registered components
        components = registry.get_check_components()
        self.assertIn("postgresql", components)
        self.assertIn("timescaledb", components)
        self.assertIn("mongodb", components)
        self.assertIn("redis", components)
        self.assertIn("kafka", components)
        self.assertIn("system_resources", components)
        self.assertIn("application", components)
        
        # Verify categories
        categories = registry.get_categories()
        self.assertIn("database", categories)
        self.assertIn("critical", categories)
        self.assertIn("cache", categories)
        self.assertIn("messaging", categories)
        self.assertIn("system", categories)
        self.assertIn("maintenance", categories)
        self.assertIn("application", categories)
        self.assertIn("metadata", categories)
        
        # Verify critical components
        critical_components = registry.get_components_by_category("critical")
        self.assertIn("postgresql", critical_components)
        self.assertIn("timescaledb", critical_components)
        self.assertIn("mongodb", critical_components)
        self.assertIn("kafka", critical_components)
    
    @patch('app.monitoring.health_checks._registry_instance', None)
    @patch('app.monitoring.health_checks.register_default_checks')
    def test_get_registry(self, mock_register_default_checks):
        """Test get_registry singleton function."""
        # Reset singleton for testing
        import app.monitoring.health_checks
        app.monitoring.health_checks._registry_instance = None
        
        # Get registry
        registry1 = get_registry()
        
        # Verify registration happened
        mock_register_default_checks.assert_called_once()
        
        # Get registry again - should be same instance
        registry2 = get_registry()
        
        # Verify same instance returned
        self.assertIs(registry1, registry2)
        
        # Verify register_default_checks not called again
        mock_register_default_checks.assert_called_once()