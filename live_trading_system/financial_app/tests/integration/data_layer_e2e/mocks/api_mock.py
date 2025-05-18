"""
Mock data and responses for API testing.

This module provides mock implementations for API responses when
the real API server is not available.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class APIMock:
    """
    Provides mock API responses for testing.
    """
    
    @staticmethod
    def get_auth_token_response():
        """Get mock authentication token response."""
        return {
            "token": "mock_test_token_for_e2e_tests",
            "user_id": "test_user_id",
            "expires_at": (datetime.now() + timedelta(hours=1)).isoformat()
        }
    
    @staticmethod
    def get_latest_data_response(symbol: str, exchange: str) -> Dict[str, Any]:
        """
        Get mock response for latest market data endpoint.
        
        Args:
            symbol: Instrument symbol
            exchange: Exchange code
            
        Returns:
            Mock response data
        """
        now = datetime.now()
        
        # Generate realistic mock data based on symbol
        base_price = 0
        if symbol == "AAPL":
            base_price = 150.25
        elif symbol == "MSFT":
            base_price = 300.50
        elif symbol == "BTC/USD":
            base_price = 35000.00
        else:
            base_price = 100.00
        
        # Create mock data points with slight variations
        data_points = []
        for i in range(10):
            timestamp = now - timedelta(minutes=i)
            price_adjustment = (i * 0.05)  # Small price movement
            volume = 1000 - (i * 100)  # Decreasing volume
            
            data_points.append({
                "symbol": symbol,
                "exchange": exchange,
                "price": base_price - price_adjustment,
                "volume": max(100, volume),
                "timestamp": timestamp.isoformat(),
                "source": "mock_data"
            })
        
        return {
            "data": data_points,
            "metadata": {
                "count": len(data_points),
                "page": 1,
                "total_pages": 1
            }
        }
    
    @staticmethod
    def get_historical_data_response(
        symbol: str, 
        exchange: str, 
        start_time: datetime, 
        end_time: datetime,
        interval: str = "1m"
    ) -> Dict[str, Any]:
        """
        Get mock response for historical market data endpoint.
        
        Args:
            symbol: Instrument symbol
            exchange: Exchange code
            start_time: Start time for data range
            end_time: End time for data range
            interval: Data interval (e.g., "1m", "5m", "1h")
            
        Returns:
            Mock response data
        """
        # Determine interval in minutes
        interval_minutes = 1
        if interval.endswith('m'):
            interval_minutes = int(interval[:-1])
        elif interval.endswith('h'):
            interval_minutes = int(interval[:-1]) * 60
        
        # Generate data points at specified interval
        data_points = []
        current_time = start_time
        
        # Generate realistic mock data based on symbol
        base_price = 0
        if symbol == "AAPL":
            base_price = 150.25
        elif symbol == "MSFT":
            base_price = 300.50
        elif symbol == "BTC/USD":
            base_price = 35000.00
        else:
            base_price = 100.00
        
        # Generate mock data points with a simple sine wave pattern
        i = 0
        while current_time <= end_time:
            # Create price movement pattern
            price_adjustment = (i % 10) * 0.25  # Simple price movement
            if i % 20 >= 10:  # Make it oscillate
                price_adjustment = -price_adjustment
                
            volume = 1000 + (i % 5) * 200  # Varying volume
            
            data_points.append({
                "symbol": symbol,
                "exchange": exchange,
                "price": base_price + price_adjustment,
                "volume": volume,
                "timestamp": current_time.isoformat(),
                "source": "mock_data"
            })
            
            current_time += timedelta(minutes=interval_minutes)
            i += 1
            
            # Limit to reasonable number of data points
            if len(data_points) >= 1000:
                break
        
        return {
            "data": data_points,
            "metadata": {
                "count": len(data_points),
                "page": 1,
                "total_pages": 1,
                "interval": interval,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat()
            }
        }
    
    @staticmethod
    def get_error_response(error_code: str, message: str, status_code: int = 400) -> Dict[str, Any]:
        """
        Get mock error response.
        
        Args:
            error_code: Error code
            message: Error message
            status_code: HTTP status code
            
        Returns:
            Mock error response
        """
        return {
            "error": {
                "code": error_code,
                "message": message,
                "status": status_code
            }
        }