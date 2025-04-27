"""
Utility functions for data validation in end-to-end tests.
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

logger = logging.getLogger(__name__)

def validate_market_data_message(message: Dict[str, Any]) -> bool:
    """
    Validates that a market data message contains all required fields
    and that the data types are correct.
    
    Args:
        message: The market data message to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    required_fields = ['symbol', 'exchange', 'timestamp', 'price', 'volume']
    
    # Check for required fields
    for field in required_fields:
        if field not in message:
            logger.error(f"Missing required field: {field}")
            return False
    
    # Validate data types
    try:
        assert isinstance(message['symbol'], str), "Symbol must be a string"
        assert isinstance(message['exchange'], str), "Exchange must be a string"
        
        # Timestamp could be string or int
        if isinstance(message['timestamp'], str):
            datetime.fromisoformat(message['timestamp'].replace('Z', '+00:00'))
        elif isinstance(message['timestamp'], (int, float)):
            datetime.fromtimestamp(message['timestamp'] / 1000 if message['timestamp'] > 1e10 else message['timestamp'])
        else:
            logger.error("Timestamp must be string or number")
            return False
        
        assert isinstance(message['price'], (int, float)), "Price must be a number"
        assert isinstance(message['volume'], (int, float)), "Volume must be a number"
        
        # Additional validations
        assert message['price'] > 0, "Price must be positive"
        assert message['volume'] >= 0, "Volume cannot be negative"
        
    except AssertionError as e:
        logger.error(f"Validation error: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Error validating message: {str(e)}")
        return False
    
    return True

def validate_database_record(record: Dict[str, Any], source_message: Dict[str, Any]) -> bool:
    """
    Validates that a database record matches the source message.
    
    Args:
        record: The database record
        source_message: The original message from the broker
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        # Check essential fields match
        assert record['symbol'] == source_message['symbol'], "Symbol mismatch"
        assert record['exchange'] == source_message['exchange'], "Exchange mismatch"
        
        # Price should match within a small delta (floating point comparison)
        price_delta = abs(float(record['price']) - float(source_message['price']))
        assert price_delta < 0.0001, f"Price mismatch: DB={record['price']}, Source={source_message['price']}"
        
        # Volume should match exactly
        assert float(record['volume']) == float(source_message['volume']), "Volume mismatch"
        
        # Timestamps should be close (allowing for processing time)
        # This depends on how timestamps are stored in your system
        
    except AssertionError as e:
        logger.error(f"Record validation error: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Error validating database record: {str(e)}")
        return False
    
    return True

def compare_datasets(source_data: List[Dict[str, Any]], 
                     target_data: List[Dict[str, Any]],
                     key_field: str = 'id') -> Dict[str, Any]:
    """
    Compares two datasets and returns statistics about the comparison.
    
    Args:
        source_data: The source dataset
        target_data: The target dataset to compare against
        key_field: The field to use as the unique key for matching records
        
    Returns:
        dict: Statistics about the comparison
    """
    if not source_data or not target_data:
        return {
            "source_count": len(source_data),
            "target_count": len(target_data),
            "match_count": 0,
            "missing_count": len(source_data),
            "extra_count": len(target_data),
            "match_percentage": 0.0
        }
    
    # Create dictionaries for faster lookup
    source_dict = {item[key_field]: item for item in source_data if key_field in item}
    target_dict = {item[key_field]: item for item in target_data if key_field in item}
    
    # Find matches, missing and extra items
    matching_keys = set(source_dict.keys()) & set(target_dict.keys())
    missing_keys = set(source_dict.keys()) - set(target_dict.keys())
    extra_keys = set(target_dict.keys()) - set(source_dict.keys())
    
    # Calculate statistics
    match_count = len(matching_keys)
    match_percentage = (match_count / len(source_dict) * 100) if source_dict else 0
    
    return {
        "source_count": len(source_data),
        "target_count": len(target_data),
        "match_count": match_count,
        "missing_count": len(missing_keys),
        "extra_count": len(extra_keys),
        "match_percentage": match_percentage,
        "missing_keys": list(missing_keys)[:10],  # List first 10 missing keys
        "extra_keys": list(extra_keys)[:10]  # List first 10 extra keys
    }

def validate_api_response(response: Dict[str, Any], expected_schema: Dict[str, Any]) -> bool:
    """
    Validates an API response against an expected schema.
    
    Args:
        response: The API response to validate
        expected_schema: The expected schema
        
    Returns:
        bool: True if valid, False otherwise
    """
    # This is a simple implementation - for production, consider using
    # a schema validation library like jsonschema
    
    def validate_object(obj: Any, schema: Dict[str, Any], path: str = "") -> List[str]:
        errors = []
        
        # Check required fields
        for field in schema.get('required', []):
            if field not in obj:
                errors.append(f"{path}.{field} is required but missing")
        
        # Check properties
        for field, field_schema in schema.get('properties', {}).items():
            if field in obj:
                field_path = f"{path}.{field}" if path else field
                
                # Check type
                field_type = field_schema.get('type')
                if field_type:
                    if field_type == 'string' and not isinstance(obj[field], str):
                        errors.append(f"{field_path} should be a string")
                    elif field_type == 'number' and not isinstance(obj[field], (int, float)):
                        errors.append(f"{field_path} should be a number")
                    elif field_type == 'integer' and not isinstance(obj[field], int):
                        errors.append(f"{field_path} should be an integer")
                    elif field_type == 'boolean' and not isinstance(obj[field], bool):
                        errors.append(f"{field_path} should be a boolean")
                    elif field_type == 'array' and not isinstance(obj[field], list):
                        errors.append(f"{field_path} should be an array")
                    elif field_type == 'object' and not isinstance(obj[field], dict):
                        errors.append(f"{field_path} should be an object")
                
                # Recursive validation for objects
                if field_type == 'object' and isinstance(obj[field], dict):
                    errors.extend(validate_object(obj[field], field_schema, field_path))
                
                # Recursive validation for arrays
                if field_type == 'array' and isinstance(obj[field], list):
                    if 'items' in field_schema:
                        for i, item in enumerate(obj[field]):
                            if field_schema['items'].get('type') == 'object':
                                errors.extend(validate_object(item, field_schema['items'], f"{field_path}[{i}]"))
        
        return errors
    
    errors = validate_object(response, expected_schema)
    
    if errors:
        for error in errors:
            logger.error(f"API response validation error: {error}")
        return False
    
    return True