"""
Message serialization utilities.
"""

import json
from typing import Dict, Any

def serialize_json(data: Dict[str, Any]) -> bytes:
    """
    Serialize data to JSON bytes.
    
    Args:
        data: Data to serialize
        
    Returns:
        Serialized data as bytes
    """
    return json.dumps(data).encode('utf-8') 