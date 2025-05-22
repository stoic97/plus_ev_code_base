"""
Data validation utilities for Kafka consumers.

This module provides validation functions for market data messages,
ensuring data integrity and consistency before processing.
"""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from datetime import datetime
from decimal import Decimal
import json

from app.consumers.base.error import ValidationError

# Set up logging
logger = logging.getLogger(__name__)


# Required fields for different message types
OHLCV_REQUIRED_FIELDS = {'symbol', 'open', 'high', 'low', 'close', 'volume', 'timestamp', 'interval'}
TRADE_REQUIRED_FIELDS = {'symbol', 'price', 'volume', 'timestamp'}
ORDERBOOK_REQUIRED_FIELDS = {'symbol', 'timestamp', 'bids', 'asks'}


def validate_message_structure(message: Dict[str, Any], required_fields: Set[str]) -> None:
    """
    Validate that a message contains all required fields.
    
    Args:
        message: Message to validate
        required_fields: Set of required field names
        
    Raises:
        ValidationError: If the message is missing required fields
    """
    missing_fields = required_fields - set(message.keys())
    if missing_fields:
        raise ValidationError(f"Missing required fields: {', '.join(missing_fields)}")


def validate_numeric_field(value: Any, field_name: str, min_value: Optional[float] = None, 
                          max_value: Optional[float] = None, allow_zero: bool = True) -> None:
    """
    Validate a numeric field.
    
    Args:
        value: Field value to validate
        field_name: Name of the field for error messages
        min_value: Minimum allowed value (optional)
        max_value: Maximum allowed value (optional)
        allow_zero: Whether zero is allowed
        
    Raises:
        ValidationError: If the field has an invalid value
    """
    # Check if numeric
    try:
        # Convert to Decimal for precise comparison
        if isinstance(value, (str, float, int)):
            num_value = Decimal(str(value))
        elif isinstance(value, Decimal):
            num_value = value
        else:
            raise ValidationError(f"Field '{field_name}' has invalid type: {type(value).__name__}")
    except Exception as e:
        raise ValidationError(f"Field '{field_name}' has non-numeric value: {value}")
    
    # Check for NaN
    if num_value != num_value:  # NaN check
        raise ValidationError(f"Field '{field_name}' has NaN value")
    
    # Check for zero value
    if not allow_zero and num_value == 0:
        raise ValidationError(f"Field '{field_name}' cannot be zero")
    
    # Check for negative value
    if min_value is not None and num_value < min_value:
        raise ValidationError(f"Field '{field_name}' is below minimum value: {num_value} < {min_value}")
    
    # Check for maximum value
    if max_value is not None and num_value > max_value:
        raise ValidationError(f"Field '{field_name}' exceeds maximum value: {num_value} > {max_value}")


def validate_timestamp(timestamp: Any, field_name: str = 'timestamp') -> None:
    """
    Validate a timestamp field.
    
    Args:
        timestamp: Timestamp value to validate (epoch milliseconds or ISO format)
        field_name: Name of the field for error messages
        
    Raises:
        ValidationError: If the timestamp is invalid
    """
    try:
        # Handle different timestamp formats
        if isinstance(timestamp, (int, float)):
            # Epoch timestamp (milliseconds or seconds)
            # If milliseconds (13 digits), convert to seconds
            if timestamp > 1e12:  # Assume milliseconds if very large
                timestamp = timestamp / 1000.0
            
            # Create datetime from epoch
            dt = datetime.fromtimestamp(timestamp)
            
            # Sanity check: not in the distant past or future
            now = datetime.now()
            if dt.year < 2000 or dt.year > now.year + 1:
                raise ValidationError(f"Field '{field_name}' has unreasonable date: {dt.year}")
                
        elif isinstance(timestamp, str):
            # ISO format string
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except ValueError:
                try:
                    # Try different format
                    dt = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%fZ")
                except ValueError:
                    raise ValidationError(f"Field '{field_name}' has invalid datetime format: {timestamp}")
                
            now = datetime.now()
            if dt.year < 2000 or dt.year > now.year + 1:
                raise ValidationError(f"Field '{field_name}' has unreasonable date: {dt.year}")
        else:
            raise ValidationError(f"Field '{field_name}' has invalid type: {type(timestamp).__name__}")
            
    except Exception as e:
        if isinstance(e, ValidationError):
            raise
        raise ValidationError(f"Field '{field_name}' has invalid value: {timestamp}")


def validate_symbol(symbol: str) -> None:
    """
    Validate a trading symbol.
    
    Args:
        symbol: Symbol to validate
        
    Raises:
        ValidationError: If the symbol is invalid
    """
    if not isinstance(symbol, str):
        raise ValidationError(f"Symbol must be a string, got {type(symbol).__name__}")
    
    if not symbol:
        raise ValidationError("Symbol cannot be empty")
    
    # Check for reasonable length
    if len(symbol) > 20:
        raise ValidationError(f"Symbol is too long: {len(symbol)} characters")
    
    # Basic format check - your requirements may vary
    allowed_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-/:")
    if not all(c in allowed_chars for c in symbol.upper()):
        raise ValidationError(f"Symbol contains invalid characters: {symbol}")


def validate_ohlcv_consistency(message: Dict[str, Any]) -> None:
    """
    Validate internal consistency of OHLCV data.
    
    Args:
        message: OHLCV message to validate
        
    Raises:
        ValidationError: If the OHLCV data is inconsistent
    """
    # Extract price values
    try:
        open_price = Decimal(str(message['open']))
        high_price = Decimal(str(message['high']))
        low_price = Decimal(str(message['low']))
        close_price = Decimal(str(message['close']))
    except Exception as e:
        raise ValidationError(f"Invalid price values in OHLCV data: {e}")
    
    # Check price relationships
    if high_price < low_price:
        raise ValidationError(f"High price ({high_price}) is less than low price ({low_price})")
        
    if high_price < open_price:
        raise ValidationError(f"High price ({high_price}) is less than open price ({open_price})")
        
    if high_price < close_price:
        raise ValidationError(f"High price ({high_price}) is less than close price ({close_price})")
        
    if low_price > open_price:
        raise ValidationError(f"Low price ({low_price}) is greater than open price ({open_price})")
        
    if low_price > close_price:
        raise ValidationError(f"Low price ({low_price}) is greater than close price ({close_price})")


def validate_orderbook_structure(orderbook: Dict[str, Any]) -> None:
    """
    Validate the structure of an order book message.
    
    Args:
        orderbook: Order book message to validate
        
    Raises:
        ValidationError: If the order book structure is invalid
    """
    # Check bids and asks are lists
    if not isinstance(orderbook.get('bids'), list):
        raise ValidationError("Order book bids must be a list")
        
    if not isinstance(orderbook.get('asks'), list):
        raise ValidationError("Order book asks must be a list")
    
    # Validate each bid and ask entry
    for i, bid in enumerate(orderbook['bids']):
        if not isinstance(bid, list) or len(bid) < 2:
            raise ValidationError(f"Invalid bid at index {i}: must be [price, volume] list")
        validate_numeric_field(bid[0], 'bid_price')
        validate_numeric_field(bid[1], 'bid_volume')
    
    for i, ask in enumerate(orderbook['asks']):
        if not isinstance(ask, list) or len(ask) < 2:
            raise ValidationError(f"Invalid ask at index {i}: must be [price, volume] list")
        validate_numeric_field(ask[0], 'ask_price')
        validate_numeric_field(ask[1], 'ask_volume')
    
    # Check bid/ask ordering (optional)
    if orderbook['bids'] and orderbook['asks']:
        highest_bid = Decimal(str(orderbook['bids'][0][0]))
        lowest_ask = Decimal(str(orderbook['asks'][0][0]))
        
        if highest_bid >= lowest_ask:
            logger.warning(f"Crossed order book: highest bid {highest_bid} >= lowest ask {lowest_ask}")


def validate_ohlcv_message(message: Dict[str, Any]) -> None:
    """
    Validate an OHLCV message.
    
    Args:
        message: OHLCV message to validate
        
    Raises:
        ValidationError: If the message is invalid
    """
    # Check required fields
    validate_message_structure(message, OHLCV_REQUIRED_FIELDS)
    
    # Validate individual fields
    validate_symbol(message['symbol'])
    validate_timestamp(message['timestamp'])
    
    # Validate price values
    validate_numeric_field(message['open'], 'open')
    validate_numeric_field(message['high'], 'high')
    validate_numeric_field(message['low'], 'low')
    validate_numeric_field(message['close'], 'close')
    validate_numeric_field(message['volume'], 'volume', min_value=0)
    
    # Check interval
    interval = message.get('interval')
    if not isinstance(interval, str):
        raise ValidationError(f"Interval must be a string, got {type(interval).__name__}")
    
    # Check internal consistency
    validate_ohlcv_consistency(message)


def validate_trade_message(message: Dict[str, Any]) -> None:
    """
    Validate a trade message.
    
    Args:
        message: Trade message to validate
        
    Raises:
        ValidationError: If the message is invalid
    """
    # Check required fields
    validate_message_structure(message, TRADE_REQUIRED_FIELDS)
    
    # Validate individual fields
    validate_symbol(message['symbol'])
    validate_timestamp(message['timestamp'])
    validate_numeric_field(message['price'], 'price', min_value=0, allow_zero=False)
    validate_numeric_field(message['volume'], 'volume', min_value=0, allow_zero=False)
    
    # Validate trade side if present
    side = message.get('side')
    if side is not None and side not in ('buy', 'sell'):
        raise ValidationError(f"Invalid trade side: {side}, must be 'buy' or 'sell'")


def validate_orderbook_message(message: Dict[str, Any]) -> None:
    """
    Validate an order book message.
    
    Args:
        message: Order book message to validate
        
    Raises:
        ValidationError: If the message is invalid
    """
    # Check required fields
    validate_message_structure(message, ORDERBOOK_REQUIRED_FIELDS)
    
    # Validate individual fields
    validate_symbol(message['symbol'])
    validate_timestamp(message['timestamp'])
    
    # Validate order book structure
    validate_orderbook_structure(message)
    
    # Validate depth if present
    depth = message.get('depth')
    if depth is not None:
        if not isinstance(depth, int) or depth <= 0:
            raise ValidationError(f"Invalid depth value: {depth}, must be a positive integer")


def get_validator_for_message_type(message_type: str) -> Callable[[Dict[str, Any]], None]:
    """
    Get validator function for a specific message type.
    
    Args:
        message_type: Type of message ('ohlcv', 'trade', 'orderbook')
        
    Returns:
        Validator function
        
    Raises:
        ValueError: If an unsupported message type is provided
    """
    validators = {
        'ohlcv': validate_ohlcv_message,
        'trade': validate_trade_message,
        'orderbook': validate_orderbook_message
    }
    
    if message_type not in validators:
        raise ValueError(f"Unsupported message type: {message_type}")
    
    return validators[message_type]