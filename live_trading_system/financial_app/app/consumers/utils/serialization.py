"""
Message serialization/deserialization utilities.

This module provides serialization and deserialization functions
for Kafka messages using various formats (JSON, Avro, etc.).
"""

import json
import logging
from typing import Any, Dict, List, Optional, Union, Callable
from confluent_kafka import Message
from app.consumers.base.error import DeserializationError

try:
    import avro.schema
    from avro.io import DatumReader, BinaryDecoder
    import io
    AVRO_AVAILABLE = True
except ImportError:
    AVRO_AVAILABLE = False

try:
    import fastavro
    FASTAVRO_AVAILABLE = True
except ImportError:
    FASTAVRO_AVAILABLE = False


# Set up logging
logger = logging.getLogger(__name__)


def deserialize_json(message: Message) -> Dict[str, Any]:
    """
    Deserialize a JSON message.
    
    Args:
        message: Kafka message with JSON payload
        
    Returns:
        Deserialized JSON object
        
    Raises:
        DeserializationError: If the message cannot be deserialized
    """
    if not message.value():
        raise DeserializationError("Empty message payload")
    
    try:
        # Decode bytes to string if necessary
        if isinstance(message.value(), bytes):
            payload = message.value().decode('utf-8')
        else:
            payload = message.value()
            
        # Parse JSON
        return json.loads(payload)
    except UnicodeDecodeError as e:
        raise DeserializationError(f"Failed to decode message payload: {e}")
    except json.JSONDecodeError as e:
        raise DeserializationError(f"Invalid JSON payload: {e}")


def deserialize_avro(message: Message, schema: Optional[Union[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Deserialize an Avro message.
    
    Args:
        message: Kafka message with Avro payload
        schema: Avro schema for deserialization (string or dict)
        
    Returns:
        Deserialized Avro object
        
    Raises:
        DeserializationError: If the message cannot be deserialized
    """
    if not AVRO_AVAILABLE and not FASTAVRO_AVAILABLE:
        raise DeserializationError("Avro deserialization requires avro or fastavro package")
    
    if not message.value():
        raise DeserializationError("Empty message payload")
    
    try:
        # Try using fastavro first if available (much faster)
        if FASTAVRO_AVAILABLE:
            # Handle schema in different formats
            if isinstance(schema, str):
                parsed_schema = fastavro.parse_schema(json.loads(schema))
            elif isinstance(schema, dict):
                parsed_schema = fastavro.parse_schema(schema)
            else:
                parsed_schema = None
            
            # Deserialize using fastavro
            if parsed_schema:
                return next(fastavro.schemaless_reader(io.BytesIO(message.value()), parsed_schema))
            else:
                # Try reading without schema
                return next(fastavro.reader(io.BytesIO(message.value())))
                
        # Fall back to standard avro if fastavro not available
        elif AVRO_AVAILABLE:
            # Parse schema if provided
            if isinstance(schema, str):
                parsed_schema = avro.schema.parse(schema)
            elif isinstance(schema, dict):
                parsed_schema = avro.schema.parse(json.dumps(schema))
            else:
                raise DeserializationError("Avro schema required for deserialization")
            
            # Create reader and decoder
            reader = DatumReader(parsed_schema)
            decoder = BinaryDecoder(io.BytesIO(message.value()))
            
            # Read and return data
            return reader.read(decoder)
            
    except Exception as e:
        raise DeserializationError(f"Failed to deserialize Avro message: {e}")


def deserialize_string(message: Message, encoding: str = 'utf-8') -> str:
    """
    Deserialize a string message.
    
    Args:
        message: Kafka message with string payload
        encoding: Character encoding (default: utf-8)
        
    Returns:
        Deserialized string
        
    Raises:
        DeserializationError: If the message cannot be deserialized
    """
    if not message.value():
        return ""
    
    try:
        return message.value().decode(encoding)
    except UnicodeDecodeError as e:
        raise DeserializationError(f"Failed to decode string message: {e}")


def deserialize_bytes(message: Message) -> bytes:
    """
    Return raw bytes payload.
    
    Args:
        message: Kafka message
        
    Returns:
        Raw bytes payload
        
    Raises:
        DeserializationError: If the message payload is empty
    """
    if not message.value():
        raise DeserializationError("Empty message payload")
    
    return message.value()


def get_deserializer(serialization_type: str) -> Callable[[Message], Any]:
    """
    Get deserializer function based on serialization type.
    
    Args:
        serialization_type: Type of serialization ('json', 'avro', 'string', 'bytes')
        
    Returns:
        Deserializer function
        
    Raises:
        ValueError: If an unsupported serialization type is provided
    """
    deserializers = {
        'json': deserialize_json,
        'string': deserialize_string,
        'bytes': deserialize_bytes
    }
    
    if serialization_type not in deserializers:
        raise ValueError(f"Unsupported serialization type: {serialization_type}")
    
    return deserializers[serialization_type]


def serialize_to_json(data: Any) -> bytes:
    """
    Serialize data to JSON bytes.
    
    Args:
        data: Data to serialize
        
    Returns:
        JSON bytes
    """
    return json.dumps(data).encode('utf-8')


def serialize_to_string(data: str) -> bytes:
    """
    Serialize string to bytes.
    
    Args:
        data: String to serialize
        
    Returns:
        String bytes
    """
    if isinstance(data, str):
        return data.encode('utf-8')
    return str(data).encode('utf-8')


def get_serializer(serialization_type: str) -> Callable[[Any], bytes]:
    """
    Get serializer function based on serialization type.
    
    Args:
        serialization_type: Type of serialization ('json', 'string', 'bytes')
        
    Returns:
        Serializer function
        
    Raises:
        ValueError: If an unsupported serialization type is provided
    """
    serializers = {
        'json': serialize_to_json,
        'string': serialize_to_string,
    }
    
    if serialization_type == 'bytes':
        return lambda x: x if isinstance(x, bytes) else bytes(x)
    
    if serialization_type not in serializers:
        raise ValueError(f"Unsupported serialization type: {serialization_type}")
    
    return serializers[serialization_type]