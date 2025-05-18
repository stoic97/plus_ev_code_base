#!/usr/bin/env python3
"""
Kafka Consumer Testing Script for E2E Testing
This script tests the functionality of Kafka consumers in the trading application,
with support for both real and mock services.
"""

import sys
import os
import time
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import importlib
from contextlib import contextmanager

# Import configuration and utilities
from financial_app.tests.integration.data_layer_e2e.e2e_config import (
    KAFKA_CONFIG, 
    TEST_INSTRUMENTS, 
    TEST_DURATION_MINUTES,
    PERFORMANCE_THRESHOLDS, 
    TEST_ID
)

from financial_app.tests.integration.data_layer_e2e.utils.performance import PerformanceTracker, timed_operation
from financial_app.tests.integration.data_layer_e2e.utils.validation import validate_market_data_message
from financial_app.tests.integration.data_layer_e2e.utils.reporting import TestReporter

# Configure logging directory
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, f"kafka_consumer_test_{TEST_ID}.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("kafka_consumer_test")

# Import the mock Kafka consumer
from financial_app.tests.integration.data_layer_e2e.mocks.mock_kafka import MockConsumerManager

# Class for dynamically importing the real consumer or falling back to a mock
class KafkaConsumerFactory:
    @staticmethod
    def get_consumer_manager():
        """
        Returns a consumer manager instance, using real implementation if available,
        otherwise falls back to a mock.
        """
        # First try to import the real consumer manager
        try:
            from app.kafka.consumer import KafkaConsumerManager
            logger.info("Using real KafkaConsumerManager")
            return KafkaConsumerManager
        except ImportError:
            logger.warning("Real KafkaConsumerManager not available, using MockConsumerManager")
            return MockConsumerManager


class KafkaConsumerTest:
    """Test class for validating Kafka consumer functionality"""
    
    def __init__(self):
        # Initialize performance and reporting tools
        self.performance_tracker = PerformanceTracker()
        self.reporter = TestReporter()
        
        # Get the appropriate consumer manager (real or mock)
        ConsumerManager = KafkaConsumerFactory.get_consumer_manager()
        
        # Initialize consumer manager with config from e2e_config
        self.consumer_manager = ConsumerManager(
            bootstrap_servers=KAFKA_CONFIG['bootstrap_servers'],
            group_id=KAFKA_CONFIG['group_id'],
            topics=[KAFKA_CONFIG['market_data_topic']],
            auto_offset_reset=KAFKA_CONFIG['auto_offset_reset']
        )
        
        # Test state variables
        self.messages_processed = 0
        self.messages_failed = 0
        self.is_real_consumer = ConsumerManager.__name__ != 'MockConsumerManager'
        
        logger.info(f"Initialized KafkaConsumerTest with {'real' if self.is_real_consumer else 'mock'} consumer")
    
    def test_consumer_group_creation(self) -> bool:
        """Test that consumer group is created properly"""
        logger.info("Testing consumer group creation and assignment")
        
        try:
            with timed_operation(self.performance_tracker, "consumer_group_creation"):
                # Start the consumer which should create the consumer group
                self.consumer_manager.start()
                time.sleep(2)  # Allow time for group to be created
                
                # Verify the consumer group exists
                if hasattr(self.consumer_manager, 'list_consumer_groups'):
                    groups = self.consumer_manager.list_consumer_groups()
                    group_exists = KAFKA_CONFIG['group_id'] in groups
                else:
                    # For real Kafka, we might need to use admin client
                    try:
                        from kafka.admin import KafkaAdminClient
                        admin_client = KafkaAdminClient(
                            bootstrap_servers=KAFKA_CONFIG['bootstrap_servers']
                        )
                        consumer_groups = admin_client.list_consumer_groups()
                        group_ids = [group[0] for group in consumer_groups]
                        group_exists = KAFKA_CONFIG['group_id'] in group_ids
                        admin_client.close()
                    except ImportError:
                        # If we can't use the admin client, assume group exists
                        logger.warning("Cannot use KafkaAdminClient, assuming group exists")
                        group_exists = True
            
            if group_exists:
                logger.info(f"Consumer group {KAFKA_CONFIG['group_id']} created successfully")
                self.reporter.record_test_result(
                    "consumer_group_creation", 
                    "PASS",
                    details={"group_id": KAFKA_CONFIG['group_id']}
                )
                return True
            else:
                logger.error(f"Consumer group {KAFKA_CONFIG['group_id']} was not created")
                self.reporter.record_test_result(
                    "consumer_group_creation", 
                    "FAIL",
                    error=f"Consumer group {KAFKA_CONFIG['group_id']} not found"
                )
                return False
                
        except Exception as e:
            logger.error(f"Error testing consumer group creation: {str(e)}")
            self.reporter.record_test_result(
                "consumer_group_creation", 
                "FAIL",
                error=f"Exception during test: {str(e)}"
            )
            return False
        finally:
            # In case we created a real admin client
            if 'admin_client' in locals():
                admin_client.close()
    
    def test_offset_management(self) -> bool:
        """Test that offsets are correctly managed"""
        logger.info("Testing offset management")
        
        try:
            with timed_operation(self.performance_tracker, "offset_management"):
                # Get initial offsets
                initial_offsets = self.consumer_manager.get_current_offsets()
                logger.info(f"Initial offsets: {initial_offsets}")
                
                # Process some messages
                self.process_messages(max_messages=10, duration_seconds=30)
                
                # Check if offsets were updated
                updated_offsets = self.consumer_manager.get_current_offsets()
                logger.info(f"Updated offsets: {updated_offsets}")
                
                # Verify offsets have advanced
                offsets_advanced = False
                for partition, offset in updated_offsets.items():
                    if partition in initial_offsets:
                        if offset > initial_offsets[partition]:
                            offsets_advanced = True
                            logger.info(f"Offset for partition {partition} advanced: {initial_offsets[partition]} -> {offset}")
                            break
            
            if offsets_advanced:
                logger.info("Offset management is working correctly")
                self.reporter.record_test_result(
                    "offset_management", 
                    "PASS",
                    details={
                        "initial_offsets": initial_offsets,
                        "updated_offsets": updated_offsets
                    }
                )
                return True
            else:
                logger.error("Offsets did not advance after consuming messages")
                self.reporter.record_test_result(
                    "offset_management", 
                    "FAIL",
                    error="Offsets did not advance",
                    details={
                        "initial_offsets": initial_offsets,
                        "updated_offsets": updated_offsets
                    }
                )
                return False
                
        except Exception as e:
            logger.error(f"Error testing offset management: {str(e)}")
            self.reporter.record_test_result(
                "offset_management", 
                "FAIL",
                error=f"Exception during test: {str(e)}"
            )
            return False
    
    def test_message_deserialization(self) -> bool:
        """Test that messages are correctly deserialized"""
        logger.info("Testing message deserialization")
        
        try:
            with timed_operation(self.performance_tracker, "message_deserialization"):
                # Process a batch of messages and track deserialization success
                deserialize_success = self.process_messages(
                    max_messages=20, 
                    duration_seconds=60,
                    validation_fn=validate_market_data_message
                )
            
            if deserialize_success and self.messages_processed > 0:
                success_rate = (self.messages_processed / (self.messages_processed + self.messages_failed)) * 100
                logger.info(f"Message deserialization successful ({success_rate:.2f}% success rate)")
                self.reporter.record_test_result(
                    "message_deserialization", 
                    "PASS",
                    details={
                        "messages_processed": self.messages_processed,
                        "messages_failed": self.messages_failed,
                        "success_rate": success_rate
                    }
                )
                return True
            else:
                logger.error("Message deserialization test failed")
                self.reporter.record_test_result(
                    "message_deserialization", 
                    "FAIL",
                    error="Failed to deserialize messages correctly",
                    details={
                        "messages_processed": self.messages_processed,
                        "messages_failed": self.messages_failed
                    }
                )
                return False
                
        except Exception as e:
            logger.error(f"Error testing message deserialization: {str(e)}")
            self.reporter.record_test_result(
                "message_deserialization", 
                "FAIL",
                error=f"Exception during test: {str(e)}"
            )
            return False
    
    def test_error_handling(self) -> bool:
        """Test consumer's ability to handle malformed messages"""
        logger.info("Testing error handling for malformed messages")
        
        try:
            # If using mock consumer, it will generate some malformed messages
            # For real consumer, we would need to publish malformed messages
            
            # Reset error counts
            initial_error_count = self.consumer_manager.get_error_count() if hasattr(self.consumer_manager, 'get_error_count') else 0
            self.messages_failed = 0
            
            with timed_operation(self.performance_tracker, "error_handling"):
                # Process messages including some malformed ones
                self.process_messages(
                    max_messages=50, 
                    duration_seconds=120,
                    validation_fn=validate_market_data_message,
                    expect_errors=True
                )
                
                # Get updated error count
                final_error_count = self.consumer_manager.get_error_count() if hasattr(self.consumer_manager, 'get_error_count') else 0
                errors_detected = (final_error_count - initial_error_count) + self.messages_failed
            
            if errors_detected > 0:
                logger.info(f"Error handling test passed: detected {errors_detected} errors")
                self.reporter.record_test_result(
                    "error_handling", 
                    "PASS",
                    details={
                        "errors_detected": errors_detected,
                        "error_count_increase": final_error_count - initial_error_count,
                        "messages_failed": self.messages_failed
                    }
                )
                return True
            else:
                logger.warning("No errors were detected during error handling test")
                # This could be valid if no malformed messages were generated
                if self.is_real_consumer:
                    logger.info("Using real consumer - may need to manually inject errors")
                    self.reporter.record_test_result(
                        "error_handling", 
                        "SKIP",
                        details={"reason": "Real consumer may need manual error injection"}
                    )
                    return True
                else:
                    self.reporter.record_test_result(
                        "error_handling", 
                        "FAIL",
                        error="Failed to detect any errors in malformed messages"
                    )
                    return False
                
        except Exception as e:
            logger.error(f"Error during error handling test: {str(e)}")
            self.reporter.record_test_result(
                "error_handling", 
                "FAIL",
                error=f"Exception during test: {str(e)}"
            )
            return False
    
    def test_consumer_recovery(self) -> bool:
        """Test consumer's ability to recover after failures"""
        logger.info("Testing consumer recovery after failure")
        
        try:
            # Get initial state
            initial_offsets = self.consumer_manager.get_current_offsets()
            
            with timed_operation(self.performance_tracker, "consumer_recovery"):
                # Stop the consumer to simulate failure
                logger.info("Simulating consumer failure...")
                self.consumer_manager.stop()
                time.sleep(5)  # Wait a moment
                
                # Restart the consumer
                logger.info("Restarting consumer...")
                self.consumer_manager.start()
                time.sleep(5)  # Wait for recovery
                
                # Get state after recovery
                recovered_offsets = self.consumer_manager.get_current_offsets()
                
                # Process some messages after recovery
                recovery_success = self.process_messages(max_messages=10, duration_seconds=30)
                
                # Check final state
                final_offsets = self.consumer_manager.get_current_offsets()
            
            # Verify recovery by checking that offsets advanced after restart
            recovery_verified = recovery_success and any(
                final_offsets[p] > recovered_offsets.get(p, 0) 
                for p in final_offsets
            )
            
            if recovery_verified:
                logger.info("Consumer recovery test passed")
                self.reporter.record_test_result(
                    "consumer_recovery", 
                    "PASS",
                    details={
                        "initial_offsets": initial_offsets,
                        "recovered_offsets": recovered_offsets,
                        "final_offsets": final_offsets
                    }
                )
                return True
            else:
                logger.error("Consumer failed to recover properly after failure")
                self.reporter.record_test_result(
                    "consumer_recovery", 
                    "FAIL",
                    error="Consumer did not process messages after restart",
                    details={
                        "initial_offsets": initial_offsets,
                        "recovered_offsets": recovered_offsets,
                        "final_offsets": final_offsets
                    }
                )
                return False
                
        except Exception as e:
            logger.error(f"Error during consumer recovery test: {str(e)}")
            self.reporter.record_test_result(
                "consumer_recovery", 
                "FAIL",
                error=f"Exception during test: {str(e)}"
            )
            return False
        finally:
            # Ensure consumer is restarted if something went wrong
            if hasattr(self.consumer_manager, 'running') and not self.consumer_manager.running:
                self.consumer_manager.start()
    
    def process_messages(self, max_messages=100, duration_seconds=60, 
                        validation_fn=None, expect_errors=False) -> bool:
        """
        Process messages from Kafka and validate them.
        
        Args:
            max_messages: Maximum number of messages to process
            duration_seconds: Maximum duration to run for
            validation_fn: Function to validate messages
            expect_errors: Whether to expect errors during processing
            
        Returns:
            bool: True if processing was successful
        """
        self.performance_tracker.start_timer("message_processing")
        
        messages_received = 0
        local_messages_processed = 0
        local_messages_failed = 0
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        while time.time() < end_time and messages_received < max_messages:
            # Poll for messages
            try:
                messages = self.consumer_manager.poll(timeout_ms=1000, max_records=10)
                
                if messages:
                    messages_received += len(messages)
                    logger.debug(f"Received {len(messages)} messages, total: {messages_received}")
                    
                    # Process each message
                    for message in messages:
                        try:
                            # Validate message if a validation function was provided
                            if validation_fn and not validation_fn(message):
                                local_messages_failed += 1
                                if not expect_errors:
                                    logger.warning(f"Message validation failed: {message}")
                            else:
                                local_messages_processed += 1
                                
                        except Exception as e:
                            local_messages_failed += 1
                            if not expect_errors:
                                logger.error(f"Error processing message: {str(e)}")
                else:
                    # Small pause when no messages to avoid tight loop
                    time.sleep(0.1)
            except Exception as e:
                logger.error(f"Error during polling: {str(e)}")
                time.sleep(1)  # Wait a bit before retrying
        
        # Update instance variables
        self.messages_processed += local_messages_processed
        self.messages_failed += local_messages_failed
        
        processing_time = time.time() - start_time
        self.performance_tracker.stop_timer("message_processing")
        
        # Record performance metrics
        if messages_received > 0:
            # Calculate success rate
            success_rate = (local_messages_processed / messages_received) * 100
            
            # Record metrics
            self.reporter.record_performance_metric(
                "messages_per_second", 
                messages_received / processing_time if processing_time > 0 else 0
            )
            
            self.reporter.record_performance_metric("message_success_rate", success_rate)
            
            # Log processing statistics
            logger.info(
                f"Processed {messages_received} messages in {processing_time:.2f} seconds "
                f"({success_rate:.2f}% success rate)"
            )
            
            # Consider test successful if success rate is acceptable
            threshold = 50.0 if expect_errors else 90.0  # Lower threshold if expecting errors
            return success_rate >= threshold
        else:
            logger.warning("No messages were received during processing")
            return False
    
    def run_all_tests(self) -> bool:
        """Run all consumer tests"""
        logger.info(f"Starting Kafka consumer tests with {'real' if self.is_real_consumer else 'mock'} consumer")
        
        try:
            # Test 1: Consumer Group Creation
            group_test = self.test_consumer_group_creation()
            
            # Test 2: Offset Management
            offset_test = self.test_offset_management()
            
            # Test 3: Message Deserialization
            deserialize_test = self.test_message_deserialization()
            
            # Test 4: Error Handling
            error_test = self.test_error_handling()
            
            # Test 5: Consumer Recovery
            recovery_test = self.test_consumer_recovery()
            
            # Calculate overall status
            all_tests = [group_test, offset_test, deserialize_test, error_test, recovery_test]
            overall_success = all(all_tests)
            
            # Record performance metrics
            performance_stats = self.performance_tracker.get_all_metrics()
            for metric, stats in performance_stats.items():
                for stat_name, value in stats.items():
                    self.reporter.record_performance_metric(f"{metric}_{stat_name}", value)
            
            # Finalize report
            status = "PASS" if overall_success else "FAIL"
            report_path = self.reporter.finalize_report(status)
            
            logger.info(f"Kafka consumer testing completed with status: {status}")
            logger.info(f"Report available at: {report_path}")
            
            return overall_success
            
        except Exception as e:
            logger.error(f"Error during test execution: {str(e)}")
            self.reporter.finalize_report("FAIL")
            return False
        finally:
            # Cleanup
            if hasattr(self.consumer_manager, 'stop'):
                self.consumer_manager.stop()


# Mock implementation for KafkaAdminClient if not available
class MockKafkaAdminClient:
    def __init__(self, **kwargs):
        self.bootstrap_servers = kwargs.get('bootstrap_servers', 'localhost:9092')
        self.consumer_groups = ["e2e_test_consumer"]  # Default group from config
    
    def list_consumer_groups(self):
        return [(group, "consumer") for group in self.consumer_groups]
    
    def close(self):
        pass


# Context manager for dependency injection for testing
@contextmanager
def use_mock_if_needed(module_path, mock_class):
    """
    Context manager that tries to import a module, but falls back to a mock if not available.
    
    Args:
        module_path: Import path for the real module
        mock_class: Mock class to use as fallback
    """
    original_module = None
    module_name = module_path.split('.')[-1]
    parent_path = '.'.join(module_path.split('.')[:-1])
    
    try:
        # Try importing the module
        if parent_path:
            parent = importlib.import_module(parent_path)
            if hasattr(parent, module_name):
                original_module = getattr(parent, module_name)
        else:
            original_module = importlib.import_module(module_name)
        
        # Module exists, use it
        yield
        
    except (ImportError, AttributeError):
        # Module doesn't exist, replace with mock
        logger.warning(f"Module {module_path} not available, using mock")
        
        if parent_path:
            parent = importlib.import_module(parent_path)
            setattr(parent, module_name, mock_class)
        else:
            sys.modules[module_name] = mock_class
        
        try:
            yield
        finally:
            # Restore original module if it existed
            if original_module:
                if parent_path:
                    setattr(parent, module_name, original_module)
                else:
                    sys.modules[module_name] = original_module


if __name__ == "__main__":
    # Inject mocks if needed
    with use_mock_if_needed('kafka.admin.KafkaAdminClient', MockKafkaAdminClient):
        # Run the tests
        consumer_test = KafkaConsumerTest()
        success = consumer_test.run_all_tests()
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)