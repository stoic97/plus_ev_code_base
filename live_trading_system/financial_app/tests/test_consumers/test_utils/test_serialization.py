import pytest
import json
from io import BytesIO

from app.consumers.utils import serialization
from app.consumers.base.error import DeserializationError

# Dummy Kafka Message stub
test_logger = serialization.logger
class DummyMessage:
    def __init__(self, val):
        self._val = val
    def value(self):
        return self._val

# -------------------- JSON Deserialization --------------------

def test_deserialize_json_success_bytes():
    data = {"key": "value", "num": 1}
    payload = json.dumps(data).encode('utf-8')
    msg = DummyMessage(payload)
    assert serialization.deserialize_json(msg) == data


def test_deserialize_json_success_str():
    data = {"a": [1, 2, 3]}
    payload = json.dumps(data)
    msg = DummyMessage(payload)
    assert serialization.deserialize_json(msg) == data


def test_deserialize_json_empty_payload():
    msg = DummyMessage(None)
    with pytest.raises(DeserializationError) as exc:
        serialization.deserialize_json(msg)
    assert "Empty message payload" in str(exc.value)


def test_deserialize_json_invalid_utf8():
    invalid = b'\xff'
    msg = DummyMessage(invalid)
    with pytest.raises(DeserializationError) as exc:
        serialization.deserialize_json(msg)
    assert "Failed to decode message payload" in str(exc.value)


def test_deserialize_json_invalid_json():
    msg = DummyMessage(b'{"a": 1,,}')
    with pytest.raises(DeserializationError) as exc:
        serialization.deserialize_json(msg)
    assert "Invalid JSON payload" in str(exc.value)

# -------------------- Avro Deserialization --------------------

@pytest.mark.skipif(
    not serialization.FASTAVRO_AVAILABLE,
    reason="fastavro is not available"
)
def test_deserialize_avro_fastavro_with_schema():
    import fastavro
    from fastavro import schemaless_writer

    # Define a simple record schema
    schema = {
        "name": "TestRecord",
        "type": "record",
        "fields": [{"name": "f1", "type": "string"}]
    }
    record = {"f1": "hello_avro"}

    # Serialize the record
    buffer = BytesIO()
    schemaless_writer(buffer, schema, record)
    payload = buffer.getvalue()

    msg = DummyMessage(payload)
    result = serialization.deserialize_avro(msg, schema)
    assert result == record

@pytest.mark.skipif(
    serialization.FASTAVRO_AVAILABLE or serialization.AVRO_AVAILABLE,
    reason="requires no avro libs available"
)
def test_deserialize_avro_no_libraries(monkeypatch):
    # Simulate neither avro nor fastavro available
    monkeypatch.setattr(serialization, 'FASTAVRO_AVAILABLE', False)
    monkeypatch.setattr(serialization, 'AVRO_AVAILABLE', False)

    msg = DummyMessage(b'data')
    with pytest.raises(DeserializationError) as exc:
        serialization.deserialize_avro(msg, schema=None)
    assert "Avro deserialization requires avro" in str(exc.value)

@pytest.mark.skipif(
    not serialization.FASTAVRO_AVAILABLE,
    reason="fastavro is not available"
)
def test_deserialize_avro_empty_payload(monkeypatch):
    msg = DummyMessage(None)
    with pytest.raises(DeserializationError) as exc:
        serialization.deserialize_avro(msg, schema={})
    assert "Empty message payload" in str(exc.value)

# -------------------- String Deserialization --------------------

def test_deserialize_string_success():
    s = "hello world"
    msg = DummyMessage(s.encode('utf-8'))
    assert serialization.deserialize_string(msg) == s


def test_deserialize_string_empty():
    msg = DummyMessage(None)
    assert serialization.deserialize_string(msg) == ""


def test_deserialize_string_unicode_error():
    msg = DummyMessage(b'\xff\xff')
    with pytest.raises(DeserializationError) as exc:
        serialization.deserialize_string(msg)
    assert "Failed to decode string message" in str(exc.value)

# -------------------- Bytes Deserialization --------------------

def test_deserialize_bytes_success():
    data = b'\x00\x01\x02'
    msg = DummyMessage(data)
    assert serialization.deserialize_bytes(msg) == data


def test_deserialize_bytes_empty():
    msg = DummyMessage(None)
    with pytest.raises(DeserializationError) as exc:
        serialization.deserialize_bytes(msg)
    assert "Empty message payload" in str(exc.value)

# -------------------- Deserializer Factory --------------------

def test_get_deserializer_valid():
    assert serialization.get_deserializer('json') == serialization.deserialize_json
    assert serialization.get_deserializer('string') == serialization.deserialize_string
    assert serialization.get_deserializer('bytes') == serialization.deserialize_bytes


def test_get_deserializer_invalid():
    with pytest.raises(ValueError) as exc:
        serialization.get_deserializer('xml')
    assert "Unsupported serialization type" in str(exc.value)

# -------------------- JSON Serialization --------------------

def test_serialize_to_json():
    data = {'x': 1, 'y': [1, 2, 3]}
    result = serialization.serialize_to_json(data)
    assert isinstance(result, bytes)
    assert json.loads(result.decode('utf-8')) == data

# -------------------- String Serialization --------------------

def test_serialize_to_string_str():
    s = "test_string"
    assert serialization.serialize_to_string(s) == b"test_string"


def test_serialize_to_string_non_str():
    num = 12345
    assert serialization.serialize_to_string(num) == b"12345"

# -------------------- Serializer Factory --------------------

def test_get_serializer_json():
    func = serialization.get_serializer('json')
    data = {'a': 1}
    assert func(data) == serialization.serialize_to_json(data)


def test_get_serializer_string():
    func = serialization.get_serializer('string')
    assert func("hi") == b"hi"


def test_get_serializer_bytes():
    func = serialization.get_serializer('bytes')
    b = b'ab'
    assert func(b) == b
    # test bytearray conversion
    ba = bytearray(b'cd')
    assert func(ba) == bytes(ba)


def test_get_serializer_invalid():
    with pytest.raises(ValueError) as exc:
        serialization.get_serializer('xml')
    assert "Unsupported serialization type" in str(exc.value)
