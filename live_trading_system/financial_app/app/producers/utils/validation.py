"""
Message validation utilities.
"""

from typing import Dict, Any, List, Tuple
from datetime import datetime

def validate_orderbook_message(data: Dict[str, Any]) -> None:
    """
    Validate orderbook message format.
    
    Args:
        data: Orderbook message to validate
        
    Raises:
        ValueError: If message is invalid
    """
    required_fields = ['symbol', 'bids', 'asks', 'timestamp']
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")
    
    if not isinstance(data['bids'], list) or not isinstance(data['asks'], list):
        raise ValueError("bids and asks must be lists")
    
    for bid in data['bids']:
        if not isinstance(bid, dict) or 'price' not in bid or 'quantity' not in bid:
            raise ValueError("Invalid bid format")
        if not isinstance(bid['price'], (int, float)) or not isinstance(bid['quantity'], (int, float)):
            raise ValueError("Price and quantity must be numbers")
    
    for ask in data['asks']:
        if not isinstance(ask, dict) or 'price' not in ask or 'quantity' not in ask:
            raise ValueError("Invalid ask format")
        if not isinstance(ask['price'], (int, float)) or not isinstance(ask['quantity'], (int, float)):
            raise ValueError("Price and quantity must be numbers")

def validate_ohlcv_message(data: Dict[str, Any]) -> None:
    """
    Validate OHLCV message format.
    
    Args:
        data: OHLCV message to validate
        
    Raises:
        ValueError: If message is invalid
    """
    required_fields = ['symbol', 'open', 'high', 'low', 'close', 'volume', 'timestamp', 'interval']
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")
    
    numeric_fields = ['open', 'high', 'low', 'close', 'volume']
    for field in numeric_fields:
        if not isinstance(data[field], (int, float)):
            raise ValueError(f"{field} must be a number")
    
    if not isinstance(data['interval'], str):
        raise ValueError("interval must be a string") 