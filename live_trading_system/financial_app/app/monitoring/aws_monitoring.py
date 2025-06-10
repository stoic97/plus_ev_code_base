"""
Fixed AWS X-Ray and CloudWatch Integration for FastAPI Trading Application

This module provides comprehensive monitoring setup for your algo trading API
using AWS native services for performance tracking with proper error handling.

Save as: financial_app/app/monitoring/aws_monitoring.py
"""

import os
import json
import logging
import time
from typing import Optional, Dict, Any
from datetime import datetime

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
import boto3
from botocore.exceptions import ClientError

# AWS X-Ray imports with proper error handling
try:
    from aws_xray_sdk.core import xray_recorder, patch_all
    from aws_xray_sdk.core.context import Context
    XRAY_AVAILABLE = True
except ImportError:
    XRAY_AVAILABLE = False
    print("X-Ray SDK not available, monitoring will work without X-Ray")

# AWS CloudWatch imports with proper error handling
try:
    import watchtower
    WATCHTOWER_AVAILABLE = True
except ImportError:
    WATCHTOWER_AVAILABLE = False
    print("Watchtower not available, will use standard logging")

# Configure X-Ray only if available
if XRAY_AVAILABLE:
    xray_recorder.configure(
        context_missing='LOG_ERROR',  # Don't crash if X-Ray context is missing
        plugins=('EC2Plugin', 'ECSPlugin'),  # Auto-detect AWS environment
        daemon_address='127.0.0.1:2000'  # X-Ray daemon address
    )
    # Patch AWS services and HTTP libraries for automatic tracing
    patch_all()

# Configure logging
logger = logging.getLogger(__name__)

class AWSMonitoringSetup:
    """
    Centralized AWS monitoring configuration for the trading application.
    """
    
    def __init__(self, app_name: str = "algo-trading-api"):
        self.app_name = app_name
        self.region = os.getenv('AWS_REGION', 'us-east-1')
        self.environment = os.getenv('ENVIRONMENT', 'development')
        
        # Initialize AWS clients with error handling
        try:
            self.cloudwatch = boto3.client('cloudwatch', region_name=self.region)
            self.logs_client = boto3.client('logs', region_name=self.region)
        except Exception as e:
            logger.error(f"Failed to initialize AWS clients: {str(e)}")
            raise
        
        # Setup CloudWatch logging
        self._setup_cloudwatch_logging()
        
    def _setup_cloudwatch_logging(self):
        """Setup CloudWatch logging handler with proper error handling."""
        try:
            if not WATCHTOWER_AVAILABLE:
                logger.warning("Watchtower not available, falling back to console logging")
                logging.basicConfig(level=logging.INFO)
                return
                
            # Create CloudWatch log handler
            cloudwatch_handler = watchtower.CloudWatchLogHandler(
                log_group=f'/aws/fastapi/{self.app_name}',
                stream_name=f'{self.environment}-{datetime.now().strftime("%Y-%m-%d")}',
                region_name=self.region
            )
            
            # Configure formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            cloudwatch_handler.setFormatter(formatter)
            
            # Add handler to logger
            logger.addHandler(cloudwatch_handler)
            logger.setLevel(logging.INFO)
            
            logger.info(f"CloudWatch logging initialized for {self.app_name}")
            
        except Exception as e:
            logger.error(f"Failed to setup CloudWatch logging: {str(e)}")
            # Fall back to console logging
            logging.basicConfig(level=logging.INFO)

    def send_custom_metric(
        self, 
        metric_name: str, 
        value: float, 
        unit: str = 'Count',
        dimensions: Optional[Dict[str, str]] = None
    ):
        """
        Send custom metrics to CloudWatch with proper error handling.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            unit: Metric unit (Count, Seconds, Percent, etc.)
            dimensions: Additional dimensions for the metric
        """
        try:
            metric_data = {
                'MetricName': metric_name,
                'Value': value,
                'Unit': unit,
                'Timestamp': datetime.utcnow()
            }
            
            if dimensions:
                metric_data['Dimensions'] = [
                    {'Name': k, 'Value': v} for k, v in dimensions.items()
                ]
            
            self.cloudwatch.put_metric_data(
                Namespace=f'{self.app_name}/Trading',
                MetricData=[metric_data]
            )
            
        except ClientError as e:
            logger.error(f"Failed to send metric {metric_name}: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error sending metric {metric_name}: {str(e)}")

# Global monitoring instance
monitoring = AWSMonitoringSetup()

def setup_xray_middleware(app: FastAPI):
    """
    Setup X-Ray middleware for FastAPI application with proper error handling.
    
    Args:
        app: FastAPI application instance
    """
    if not XRAY_AVAILABLE:
        logger.warning("X-Ray not available, skipping X-Ray middleware setup")
        return
        
    try:
        # Custom X-Ray middleware implementation
        @app.middleware("http")
        async def xray_middleware(request: Request, call_next):
            # Create X-Ray segment for this request
            segment_name = f"{request.method} {request.url.path}"
            
            try:
                with xray_recorder.in_segment(segment_name) as segment:
                    # Add request metadata
                    if segment:
                        segment.put_http_meta('request', {
                            'method': request.method,
                            'url': str(request.url),
                            'user_agent': request.headers.get('user-agent', ''),
                            'remote_addr': request.client.host if request.client else None
                        })
                    
                    try:
                        # Process the request
                        response = await call_next(request)
                        
                        # Add response metadata
                        if segment:
                            segment.put_http_meta('response', {
                                'status': response.status_code,
                                'content_length': response.headers.get('content-length', 0)
                            })
                        
                        return response
                        
                    except Exception as e:
                        # Add exception to segment
                        if segment:
                            segment.add_exception(e)
                        raise
            except Exception as e:
                # If X-Ray fails, still process the request
                logger.warning(f"X-Ray segment creation failed: {str(e)}")
                return await call_next(request)
        
        logger.info("X-Ray middleware configured successfully")
        
    except Exception as e:
        logger.error(f"Failed to setup X-Ray middleware: {str(e)}")

async def trading_metrics_middleware(request: Request, call_next):
    """
    Custom middleware to track trading-specific metrics alongside X-Ray.
    This captures business metrics that X-Ray doesn't track.
    """
    start_time = time.time()
    
    # Extract trading-specific context
    user_id = None
    endpoint_type = None
    
    try:
        # Extract user ID from JWT token (if present)
        auth_header = request.headers.get('authorization')
        if auth_header:
            # You'd implement JWT decode here
            # user_id = decode_jwt_user_id(auth_header)
            pass
        
        # Classify endpoint type for business metrics
        path = request.url.path
        if '/auth/' in path:
            endpoint_type = 'authentication'
        elif '/trading/' in path:
            endpoint_type = 'trading'
        elif '/portfolio/' in path:
            endpoint_type = 'portfolio'
        elif '/market-data/' in path:
            endpoint_type = 'market_data'
        else:
            endpoint_type = 'other'
            
    except Exception as e:
        logger.warning(f"Failed to extract trading context: {str(e)}")
    
    # Add custom segments to X-Ray trace with proper error handling
    try:
        if XRAY_AVAILABLE:
            with xray_recorder.in_subsegment(f'trading_middleware_{endpoint_type}'):
                # Add metadata to X-Ray trace
                current_subsegment = xray_recorder.current_subsegment()
                if current_subsegment:
                    current_subsegment.put_metadata('user_id', user_id)
                    current_subsegment.put_metadata('endpoint_type', endpoint_type)
                    current_subsegment.put_metadata('environment', monitoring.environment)
                
                # Execute the request
                response = await call_next(request)
                
                # Calculate response time
                response_time = time.time() - start_time
                
                # Send custom CloudWatch metrics for trading operations
                if endpoint_type == 'trading':
                    monitoring.send_custom_metric(
                        metric_name='TradingEndpointResponseTime',
                        value=response_time,
                        unit='Seconds',
                        dimensions={
                            'Environment': monitoring.environment,
                            'Endpoint': path,
                            'StatusCode': str(response.status_code)
                        }
                    )
                
                # Track authentication metrics
                if endpoint_type == 'authentication':
                    monitoring.send_custom_metric(
                        metric_name='AuthenticationRequests',
                        value=1,
                        unit='Count',
                        dimensions={
                            'Environment': monitoring.environment,
                            'Method': request.method,
                            'StatusCode': str(response.status_code)
                        }
                    )
                
                # Add response time to X-Ray
                if current_subsegment:
                    current_subsegment.put_annotation('response_time', response_time)
                    current_subsegment.put_annotation('status_code', response.status_code)
                
                return response
        else:
            # If X-Ray is not available, still execute the request with metrics
            response = await call_next(request)
            response_time = time.time() - start_time
            
            # Send metrics without X-Ray
            if endpoint_type == 'trading':
                monitoring.send_custom_metric(
                    metric_name='TradingEndpointResponseTime',
                    value=response_time,
                    unit='Seconds',
                    dimensions={
                        'Environment': monitoring.environment,
                        'Endpoint': path,
                        'StatusCode': str(response.status_code)
                    }
                )
            
            if endpoint_type == 'authentication':
                monitoring.send_custom_metric(
                    metric_name='AuthenticationRequests',
                    value=1,
                    unit='Count',
                    dimensions={
                        'Environment': monitoring.environment,
                        'Method': request.method,
                        'StatusCode': str(response.status_code)
                    }
                )
            
            return response
    
    except Exception as e:
        # If X-Ray fails, still process the request
        logger.warning(f"X-Ray subsegment failed: {str(e)}")
        return await call_next(request)

def configure_xray_sampling():
    """
    Configure X-Ray sampling rules for optimal cost and performance.
    """
    if not XRAY_AVAILABLE:
        logger.warning("X-Ray not available, skipping sampling configuration")
        return
        
    sampling_rules = {
        "version": 2,
        "default": {
            "fixed_target": 1,      # Always sample 1 request per second
            "rate": 0.1             # Sample 10% of additional requests
        },
        "rules": [
            {
                "description": "Trading endpoints - high sampling",
                "service_name": "algo-trading-api",
                "http_method": "*",
                "url_path": "/api/v1/trading/*",
                "fixed_target": 2,
                "rate": 0.5         # Sample 50% of trading requests
            },
            {
                "description": "Health checks - minimal sampling",
                "service_name": "algo-trading-api", 
                "http_method": "GET",
                "url_path": "/health",
                "fixed_target": 0,
                "rate": 0.01        # Sample only 1% of health checks
            }
        ]
    }
    
    # Save sampling rules (you can upload this to X-Ray console)
    try:
        with open('xray-sampling-rules.json', 'w') as f:
            json.dump(sampling_rules, f, indent=2)
        
        logger.info("X-Ray sampling rules configured")
    except Exception as e:
        logger.error(f"Failed to save sampling rules: {str(e)}")

def setup_cloudwatch_alarms():
    """
    Setup CloudWatch alarms for critical trading metrics.
    """
    alarms = [
        {
            'AlarmName': f'{monitoring.app_name}-HighResponseTime',
            'ComparisonOperator': 'GreaterThanThreshold',
            'EvaluationPeriods': 2,
            'MetricName': 'ResponseTime',
            'Namespace': 'AWS/X-Ray',
            'Period': 300,
            'Statistic': 'Average',
            'Threshold': 5.0,  # 5 seconds
            'ActionsEnabled': True,
            'AlarmDescription': 'Alert when API response time is high',
            'Dimensions': [
                {
                    'Name': 'ServiceName',
                    'Value': monitoring.app_name
                }
            ]
        },
        {
            'AlarmName': f'{monitoring.app_name}-HighErrorRate',
            'ComparisonOperator': 'GreaterThanThreshold',
            'EvaluationPeriods': 2,
            'MetricName': 'ErrorRate',
            'Namespace': 'AWS/X-Ray',
            'Period': 300,
            'Statistic': 'Average',
            'Threshold': 0.05,  # 5% error rate
            'ActionsEnabled': True,
            'AlarmDescription': 'Alert when API error rate is high'
        }
    ]
    
    try:
        for alarm in alarms:
            monitoring.cloudwatch.put_metric_alarm(**alarm)
        
        logger.info("CloudWatch alarms configured")
        
    except ClientError as e:
        logger.error(f"Failed to setup CloudWatch alarms: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error setting up CloudWatch alarms: {str(e)}")

# Utility functions for trading-specific monitoring
def track_trade_execution(
    user_id: str,
    symbol: str, 
    quantity: float,
    trade_type: str,
    execution_time: float,
    success: bool
):
    """
    Track specific trading operations for business intelligence.
    """
    try:
        if XRAY_AVAILABLE:
            with xray_recorder.in_subsegment('trade_execution'):
                # Add trading context to X-Ray
                current_subsegment = xray_recorder.current_subsegment()
                if current_subsegment:
                    current_subsegment.put_metadata('trade_details', {
                        'user_id': user_id,
                        'symbol': symbol,
                        'quantity': quantity,
                        'trade_type': trade_type,
                        'success': success
                    })
                
                # Send custom metrics
                monitoring.send_custom_metric(
                    metric_name='TradeExecutionTime',
                    value=execution_time,
                    unit='Seconds',
                    dimensions={
                        'Symbol': symbol,
                        'TradeType': trade_type,
                        'Success': str(success)
                    }
                )
                
                monitoring.send_custom_metric(
                    metric_name='TradeVolume',
                    value=quantity,
                    unit='Count',
                    dimensions={
                        'Symbol': symbol,
                        'TradeType': trade_type
                    }
                )
        else:
            # Send metrics without X-Ray
            monitoring.send_custom_metric(
                metric_name='TradeExecutionTime',
                value=execution_time,
                unit='Seconds',
                dimensions={
                    'Symbol': symbol,
                    'TradeType': trade_type,
                    'Success': str(success)
                }
            )
            
            monitoring.send_custom_metric(
                metric_name='TradeVolume',
                value=quantity,
                unit='Count',
                dimensions={
                    'Symbol': symbol,
                    'TradeType': trade_type
                }
            )
            
    except Exception as e:
        logger.error(f"Failed to track trade execution: {str(e)}")

def track_portfolio_update(user_id: str, portfolio_value: float):
    """
    Track portfolio value changes for risk monitoring.
    """
    try:
        monitoring.send_custom_metric(
            metric_name='PortfolioValue',
            value=portfolio_value,
            unit='None',
            dimensions={
                'UserId': user_id,
                'Environment': monitoring.environment
            }
        )
    except Exception as e:
        logger.error(f"Failed to track portfolio update: {str(e)}")

def track_api_performance(endpoint: str, response_time: float, status_code: int):
    """
    Track API performance metrics for all endpoints.
    """
    try:
        monitoring.send_custom_metric(
            metric_name='APIResponseTime',
            value=response_time,
            unit='Seconds',
            dimensions={
                'Endpoint': endpoint,
                'StatusCode': str(status_code),
                'Environment': monitoring.environment
            }
        )
        
        # Track success/failure rates
        is_success = 200 <= status_code < 400
        monitoring.send_custom_metric(
            metric_name='APIRequests',
            value=1,
            unit='Count',
            dimensions={
                'Endpoint': endpoint,
                'Success': str(is_success),
                'Environment': monitoring.environment
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to track API performance: {str(e)}")

def track_user_activity(user_id: str, activity_type: str, metadata: Dict[str, Any] = None):
    """
    Track user activity for analytics and monitoring.
    """
    try:
        if XRAY_AVAILABLE:
            with xray_recorder.in_subsegment('user_activity'):
                current_subsegment = xray_recorder.current_subsegment()
                if current_subsegment:
                    current_subsegment.put_metadata('user_activity', {
                        'user_id': user_id,
                        'activity_type': activity_type,
                        'metadata': metadata or {}
                    })
        
        monitoring.send_custom_metric(
            metric_name='UserActivity',
            value=1,
            unit='Count',
            dimensions={
                'ActivityType': activity_type,
                'Environment': monitoring.environment
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to track user activity: {str(e)}")

def track_system_health():
    """
    Track overall system health metrics.
    """
    try:
        import psutil
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        monitoring.send_custom_metric(
            metric_name='SystemCPUUsage',
            value=cpu_percent,
            unit='Percent',
            dimensions={'Environment': monitoring.environment}
        )
        
        # Memory usage
        memory = psutil.virtual_memory()
        monitoring.send_custom_metric(
            metric_name='SystemMemoryUsage',
            value=memory.percent,
            unit='Percent',
            dimensions={'Environment': monitoring.environment}
        )
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        monitoring.send_custom_metric(
            metric_name='SystemDiskUsage',
            value=disk_percent,
            unit='Percent',
            dimensions={'Environment': monitoring.environment}
        )
        
    except ImportError:
        logger.warning("psutil not available, cannot track system health")
    except Exception as e:
        logger.error(f"Failed to track system health: {str(e)}")

def create_custom_dashboard():
    """
    Create a custom CloudWatch dashboard for trading metrics.
    """
    try:
        dashboard_body = {
            "widgets": [
                {
                    "type": "metric",
                    "properties": {
                        "metrics": [
                            [f"{monitoring.app_name}/Trading", "TradeExecutionTime"],
                            [f"{monitoring.app_name}/Trading", "TradingEndpointResponseTime"],
                            [f"{monitoring.app_name}/Trading", "APIResponseTime"]
                        ],
                        "period": 300,
                        "stat": "Average",
                        "region": monitoring.region,
                        "title": "API Performance"
                    }
                },
                {
                    "type": "metric",
                    "properties": {
                        "metrics": [
                            [f"{monitoring.app_name}/Trading", "TradeVolume"],
                            [f"{monitoring.app_name}/Trading", "AuthenticationRequests"],
                            [f"{monitoring.app_name}/Trading", "APIRequests"]
                        ],
                        "period": 300,
                        "stat": "Sum",
                        "region": monitoring.region,
                        "title": "Trading Volume & Activity"
                    }
                }
            ]
        }
        
        monitoring.cloudwatch.put_dashboard(
            DashboardName=f'{monitoring.app_name}-Trading-Dashboard',
            DashboardBody=json.dumps(dashboard_body)
        )
        
        logger.info("Custom CloudWatch dashboard created")
        
    except Exception as e:
        logger.error(f"Failed to create custom dashboard: {str(e)}")

# Context manager for manual tracing
class CustomTrace:
    """
    Context manager for manual X-Ray tracing when middleware isn't enough.
    """
    
    def __init__(self, trace_name: str, metadata: Dict[str, Any] = None):
        self.trace_name = trace_name
        self.metadata = metadata or {}
        self.subsegment = None
        
    def __enter__(self):
        if XRAY_AVAILABLE:
            try:
                self.subsegment = xray_recorder.begin_subsegment(self.trace_name)
                if self.subsegment and self.metadata:
                    for key, value in self.metadata.items():
                        self.subsegment.put_metadata(key, value)
                return self.subsegment
            except Exception as e:
                logger.warning(f"Failed to start custom trace {self.trace_name}: {str(e)}")
                return None
        return None
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if XRAY_AVAILABLE and self.subsegment:
            try:
                if exc_type:
                    self.subsegment.add_exception(exc_val)
                xray_recorder.end_subsegment()
            except Exception as e:
                logger.warning(f"Failed to end custom trace {self.trace_name}: {str(e)}")

# Decorator for automatic function tracing
def trace_function(trace_name: str = None, include_args: bool = False):
    """
    Decorator to automatically trace function execution.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            name = trace_name or f"{func.__module__}.{func.__name__}"
            metadata = {}
            
            if include_args and args:
                metadata['args'] = str(args)
            if include_args and kwargs:
                metadata['kwargs'] = str(kwargs)
                
            with CustomTrace(name, metadata):
                return func(*args, **kwargs)
        return wrapper
    return decorator

# Health check function for monitoring
def monitoring_health_check():
    """
    Perform a health check of the monitoring system itself.
    """
    health_status = {
        'aws_monitoring': 'healthy',
        'xray': 'available' if XRAY_AVAILABLE else 'unavailable',
        'cloudwatch': 'healthy',
        'watchtower': 'available' if WATCHTOWER_AVAILABLE else 'unavailable',
        'timestamp': datetime.utcnow().isoformat()
    }
    
    try:
        # Test CloudWatch connectivity
        monitoring.send_custom_metric('MonitoringHealthCheck', 1, 'Count')
        
        # Test X-Ray if available
        if XRAY_AVAILABLE:
            with CustomTrace('health_check'):
                pass
                
    except Exception as e:
        health_status['aws_monitoring'] = 'unhealthy'
        health_status['error'] = str(e)
        logger.error(f"Monitoring health check failed: {str(e)}")
    
    return health_status

# Initialize monitoring on module import
def initialize_monitoring():
    """
    Initialize monitoring system with proper error handling.
    """
    try:
        logger.info("Initializing AWS monitoring system")
        
        # Send startup metric
        monitoring.send_custom_metric('SystemStartup', 1, 'Count')
        
        # Create dashboard if in production
        if monitoring.environment == 'production':
            create_custom_dashboard()
            
        logger.info("AWS monitoring system initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize monitoring system: {str(e)}")

# Auto-initialize when module is imported
initialize_monitoring()