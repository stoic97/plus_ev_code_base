"""
Utility functions for checking service availability.

This module provides functions to check if services are running and available
for testing, allowing tests to fall back to mocks when necessary.
"""

import logging
import socket
import requests
from typing import Dict, Any, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

def check_api_available(api_url: str, timeout: int = 5) -> bool:
    """
    Check if an API endpoint is available.
    
    Args:
        api_url: Base URL of the API
        timeout: Connection timeout in seconds
        
    Returns:
        bool: True if the API is available, False otherwise
    """
    try:
        # Parse URL to get host and port
        parsed_url = urlparse(api_url)
        host = parsed_url.hostname
        
        # If no scheme, assume http
        if not parsed_url.scheme:
            api_url = f"http://{api_url}"
        
        # Try to connect using a simple HEAD request
        response = requests.head(api_url, timeout=timeout)
        logger.debug(f"API check to {api_url} returned status code {response.status_code}")
        
        # Consider any response (even 4xx) as the service being available
        return True
    except requests.RequestException as e:
        logger.debug(f"API at {api_url} is not available: {str(e)}")
        return False
    except Exception as e:
        logger.debug(f"Error checking API availability: {str(e)}")
        return False

def check_db_available(host: str, port: int, timeout: int = 5) -> bool:
    """
    Check if a database server is available.
    
    Args:
        host: Database server hostname
        port: Database server port
        timeout: Connection timeout in seconds
        
    Returns:
        bool: True if the database is available, False otherwise
    """
    try:
        # Try to open a socket connection to the database
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        
        # If result is 0, the connection was successful
        is_available = (result == 0)
        logger.debug(f"Database at {host}:{port} is {'available' if is_available else 'not available'}")
        return is_available
    except Exception as e:
        logger.debug(f"Error checking database availability: {str(e)}")
        return False

def check_kafka_available(bootstrap_servers: str, timeout: int = 5) -> bool:
    """
    Check if Kafka is available.
    
    Args:
        bootstrap_servers: Kafka bootstrap servers (comma-separated list)
        timeout: Connection timeout in seconds
        
    Returns:
        bool: True if Kafka is available, False otherwise
    """
    try:
        # Split servers and check each one
        servers = bootstrap_servers.split(',')
        for server in servers:
            # Parse host:port from server
            if ':' in server:
                host, port_str = server.strip().split(':')
                port = int(port_str)
            else:
                host = server.strip()
                port = 9092  # Default Kafka port
            
            # Try to connect to the server
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            sock.close()
            
            # If any server is available, return True
            if result == 0:
                logger.debug(f"Kafka at {host}:{port} is available")
                return True
        
        logger.debug(f"No Kafka servers are available in {bootstrap_servers}")
        return False
    except Exception as e:
        logger.debug(f"Error checking Kafka availability: {str(e)}")
        return False

def get_service_availability() -> Dict[str, bool]:
    """
    Check availability of all services needed for testing.
    
    Returns:
        Dictionary mapping service names to availability status
    """
    from financial_app.tests.integration.data_layer_e2e.e2e_config import (
        API_BASE_URL, DB_CONFIG, KAFKA_CONFIG
    )
    
    return {
        "api": check_api_available(API_BASE_URL),
        "database": check_db_available(DB_CONFIG["host"], DB_CONFIG["port"]),
        "kafka": check_kafka_available(KAFKA_CONFIG["bootstrap_servers"])
    }