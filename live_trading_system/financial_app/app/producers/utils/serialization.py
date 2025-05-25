"""
Message serialization utilities.

This module provides utilities for serializing messages to various formats
for Kafka publishing.
"""

import json
from typing import Any, Dict
from datetime import datetime

from app.producers.base.error import SerializationError

def serialize_json(data: Dict[str, Any]) -> bytes:
    """
    Serialize data to JSON bytes.
    
    Args:
        data: Data to serialize
        
    Returns:
        Serialized data as bytes
        
    Raises:
        SerializationError: If serialization fails
    """
    try:
        return json.dumps(data, default=_json_serializer).encode('utf-8')
    except Exception as e:
        raise SerializationError(f"Failed to serialize data to JSON: {e}")

def _json_serializer(obj: Any) -> Any:
    """
    Custom JSON serializer for handling special types.
    
    Args:
        obj: Object to serialize
        
    Returns:
        JSON-serializable representation of the object
    """
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable") 