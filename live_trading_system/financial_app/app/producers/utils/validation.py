"""
Message validation utilities.

This module provides utilities for validating message structure and content
before publishing to Kafka.
"""

from typing import Any, Dict
from datetime import datetime

from app.producers.base.error import ValidationError

def validate_ohlcv_message(data: Dict[str, Any]) -> None:
    """
    Validate OHLCV message structure and content.
    
    Args:
        data: OHLCV message to validate
        
    Raises:
        ValidationError: If validation fails
    """
    # Required fields
    required_fields = ['symbol', 'open', 'high', 'low', 'close', 'volume', 'interval']
    for field in required_fields:
        if field not in data:
            raise ValidationError(f"Missing required field: {field}")
    
    # Validate symbol
    if not isinstance(data['symbol'], str) or not data['symbol']:
        raise ValidationError("Symbol must be a non-empty string")
    
    # Validate numeric fields
    numeric_fields = ['open', 'high', 'low', 'close', 'volume']
    for field in numeric_fields:
        if not isinstance(data[field], (int, float)):
            raise ValidationError(f"{field} must be a number")
        if data[field] < 0:
            raise ValidationError(f"{field} must be non-negative")
    
    # Validate price relationships
    if data['high'] < data['low']:
        raise ValidationError("High price cannot be less than low price")
    if data['open'] < data['low'] or data['open'] > data['high']:
        raise ValidationError("Open price must be between low and high")
    if data['close'] < data['low'] or data['close'] > data['high']:
        raise ValidationError("Close price must be between low and high")
    
    # Validate interval
    valid_intervals = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w', '1M']
    if data['interval'] not in valid_intervals:
        raise ValidationError(f"Invalid interval. Must be one of: {valid_intervals}")
    
    # Validate optional fields
    if 'vwap' in data:
        if not isinstance(data['vwap'], (int, float)) or data['vwap'] < 0:
            raise ValidationError("VWAP must be a non-negative number")
    
    if 'trades_count' in data:
        if not isinstance(data['trades_count'], int) or data['trades_count'] < 0:
            raise ValidationError("Trades count must be a non-negative integer")
    
    if 'open_interest' in data:
        if not isinstance(data['open_interest'], (int, float)) or data['open_interest'] < 0:
            raise ValidationError("Open interest must be a non-negative number")
    
    if 'adjusted_close' in data:
        if not isinstance(data['adjusted_close'], (int, float)) or data['adjusted_close'] < 0:
            raise ValidationError("Adjusted close must be a non-negative number")
    
    # Validate timestamp
    if 'timestamp' in data:
        if not isinstance(data['timestamp'], (int, float)):
            raise ValidationError("Timestamp must be a number")
        if data['timestamp'] < 0:
            raise ValidationError("Timestamp must be non-negative")

def validate_orderbook_message(data: Dict[str, Any]) -> None:
    """
    Validate orderbook message structure and content.
    
    Args:
        data: Orderbook message to validate
        
    Raises:
        ValidationError: If validation fails
    """
    # Required fields
    required_fields = ['symbol', 'bids', 'asks', 'timestamp']
    for field in required_fields:
        if field not in data:
            raise ValidationError(f"Missing required field: {field}")
    
    # Validate symbol
    if not isinstance(data['symbol'], str) or not data['symbol']:
        raise ValidationError("Symbol must be a non-empty string")
    
    # Validate bids and asks
    if not isinstance(data['bids'], list) or not isinstance(data['asks'], list):
        raise ValidationError("Bids and asks must be lists")
    
    # Validate price levels
    for side, levels in [('bids', data['bids']), ('asks', data['asks'])]:
        for level in levels:
            if not isinstance(level, list) or len(level) != 2:
                raise ValidationError(f"Invalid {side} level format")
            price, quantity = level
            if not isinstance(price, (int, float)) or price < 0:
                raise ValidationError(f"Invalid price in {side}")
            if not isinstance(quantity, (int, float)) or quantity < 0:
                raise ValidationError(f"Invalid quantity in {side}")
    
    # Validate price relationships
    if data['bids'] and data['asks']:
        best_bid = max(price for price, _ in data['bids'])
        best_ask = min(price for price, _ in data['asks'])
        if best_bid >= best_ask:
            raise ValidationError("Best bid must be less than best ask")
    
    # Validate timestamp
    if not isinstance(data['timestamp'], (int, float)):
        raise ValidationError("Timestamp must be a number")
    if data['timestamp'] < 0:
        raise ValidationError("Timestamp must be non-negative") 