import json
import pytest
from datetime import datetime, timezone

import app.consumers.market_data.trade_consumer as tc_module
from app.consumers.market_data.trade_consumer import TradeConsumer
from app.consumers.base.error import DeserializationError, ProcessingError, ValidationError

# Dummy classes for dependencies
class DummyInstrument:
    def __init__(self, symbol, id=123):
        self.symbol = symbol
        self.id = id

class DummyRepository:
    def __init__(self):
        self.saved_batches = []
        self.instruments = {}

    def save_tick_batch(self, tick_batch):
        self.saved_batches.append(tick_batch)

    def get_or_create_instrument(self, symbol):
        if symbol not in self.instruments:
            self.instruments[symbol] = DummyInstrument(symbol)
        return self.instruments[symbol]

class DummyOffsetManager:
    def __init__(self, *args, **kwargs):
        self.tracked = []
        self.committed = 0

    def track_message(self, msg):
        self.tracked.append(msg)

    def should_commit(self):
        return True

    def commit(self, async_commit=True):
        self.committed += 1

class DummyMetrics:
    def __init__(self):
        self.processed = 0
        self.failed = 0
    def record_message_processed(self, _):
        self.processed += 1
    def record_message_failed(self):
        self.failed += 1

class DummyHealthCheck:
    def __init__(self):
        self.processed = 0
        self.errors = 0
    def record_message_processed(self, _):
        self.processed += 1
    def record_error(self):
        self.errors += 1

# No-op circuit breaker decorator
def noop_circuit(name):
    def decorator(func):
        return func
    return decorator

@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch):
    # Patch OffsetManager
    monkeypatch.setattr(tc_module, 'OffsetManager', DummyOffsetManager)
    # Patch metrics registry
    monkeypatch.setattr(tc_module, 'get_metrics_registry', lambda: type('R', (), {
        'register_consumer': lambda *args, **kwargs: DummyMetrics()
    })())
    # Patch health manager
    monkeypatch.setattr(tc_module, 'get_health_manager', lambda: type('H', (), {
        'register_consumer': lambda *args, **kwargs: DummyHealthCheck()
    })())
    # Patch circuit_breaker to no-op
    monkeypatch.setattr(tc_module, 'circuit_breaker', noop_circuit)
    # Default JSON deserializer and validator
    monkeypatch.setattr(tc_module, 'deserialize_json', lambda m: json.loads(m.value()))
    monkeypatch.setattr(tc_module, 'validate_trade_message', lambda x: None)
    # Patch Tick model to dummy to bypass validation
    class DummyTick:
        def __init__(self, instrument_id, timestamp, price, volume, source, source_timestamp):
            self.instrument_id = instrument_id
            self.timestamp = timestamp
            self.price = price
            self.volume = volume
            self.source = source
            self.source_timestamp = source_timestamp
            self.trade_data = {}
        def __repr__(self):
            return f"DummyTick({instrument_id}, {timestamp}, {price}, {volume})"
    monkeypatch.setattr(tc_module, 'Tick', DummyTick)
    yield

# Helper to create a dummy Kafka Message with JSON payload
class DummyMsg:
    def __init__(self, payload):
        self._payload = payload
    def value(self):
        if isinstance(self._payload, (dict, list)):
            return json.dumps(self._payload).encode('utf-8')
        return self._payload.encode('utf-8')
    def __repr__(self):
        return f"DummyMsg({self._payload})"

@pytest.fixture
def consumer():
    repo = DummyRepository()
    cons = TradeConsumer(batch_size=1, batch_timeout_ms=100000, repository=repo)
    return cons

# Deserialization error tests

def test_deserialize_json_error(monkeypatch, consumer):
    err = json.JSONDecodeError("msg", "doc", 0)
    monkeypatch.setattr(tc_module, 'deserialize_json', lambda m: (_ for _ in ()).throw(err))
    with pytest.raises(DeserializationError) as ei:
        consumer._deserialize_message(DummyMsg('{}'))
    assert "Invalid JSON in trade message" in str(ei.value)


def test_deserialize_generic_error(monkeypatch, consumer):
    exc = Exception("oops")
    monkeypatch.setattr(tc_module, 'deserialize_json', lambda m: (_ for _ in ()).throw(exc))
    with pytest.raises(DeserializationError) as ei:
        consumer._deserialize_message(DummyMsg('{}'))
    assert "Failed to deserialize trade message" in str(ei.value)


def test_deserialize_validation_error(monkeypatch, consumer):
    data = {'symbol': 'X', 'timestamp': 0, 'price': 1, 'volume': 1}
    monkeypatch.setattr(tc_module, 'deserialize_json', lambda m: data)
    monkeypatch.setattr(tc_module, 'validate_trade_message', lambda m: (_ for _ in ()).throw(ValidationError("invalid")))
    with pytest.raises(DeserializationError) as ei:
        consumer._deserialize_message(DummyMsg(json.dumps(data)))
    assert "Invalid trade message" in str(ei.value)

# Tests for tick creation

def test_create_tick_numeric_timestamp(consumer):
    msg = {
        'symbol': 'SYM',
        'timestamp': 1620000000000,
        'price': 100.5,
        'volume': 20,
        'trade_id': 'T1',
        'side': 'buy',
        'extra': 'data'
    }
    tick = consumer._create_tick_from_message(msg)
    assert tick.instrument_id == consumer.repository.get_or_create_instrument('SYM').id
    expected = datetime.fromtimestamp(1620000000)
    assert abs((tick.timestamp - expected).total_seconds()) < 1
    assert tick.trade_id == 'T1'
    assert tick.side == 'buy'
    assert tick.trade_data == {'extra': 'data'}


def test_create_tick_iso_timestamp(consumer):
    iso = '2025-04-17T12:00:00Z'
    msg = {'symbol': 'SYM', 'timestamp': iso, 'price': 50, 'volume': 10}
    tick = consumer._create_tick_from_message(msg)
    expected = datetime(2025, 4, 17, 12, 0, tzinfo=timezone.utc)
    assert tick.timestamp == expected


def test_create_tick_missing_field(consumer):
    msg = {'timestamp': 0, 'price': 1, 'volume': 1}
    with pytest.raises(ProcessingError) as ei:
        consumer._create_tick_from_message(msg)
    assert "Missing required field" in str(ei.value)

# Tests for processing messages and batch

def test_process_message_saves_batch_and_commits(consumer):
    data = {'symbol': 'A', 'timestamp': 1000, 'price': 2, 'volume': 3}
    raw_msg = DummyMsg(data)
    consumer.process_message(data, raw_msg)
    assert len(consumer.repository.saved_batches) == 1
    saved_tick = consumer.repository.saved_batches[0][0]
    assert saved_tick.price == data['price']
    assert consumer.offset_manager.tracked == [raw_msg]
    assert consumer.offset_manager.committed == 1
    assert consumer.metrics.processed == 1
    assert consumer.health_check.processed == 1


def test_process_message_on_error(monkeypatch, consumer):
    data = {'symbol': 'A', 'timestamp': 0, 'price': 1, 'volume': 1}
    raw_msg = DummyMsg(data)
    monkeypatch.setattr(consumer.repository, 'save_tick_batch', lambda x: (_ for _ in ()).throw(Exception('db error')))
    with pytest.raises(ProcessingError):
        consumer.process_message(data, raw_msg)
    assert consumer.metrics.failed == 1
    assert consumer.health_check.errors == 1

# Tests for on_stop behavior

def test_on_stop_with_pending_batch(monkeypatch, consumer):
    called = {'processed': 0, 'committed': 0}
    consumer._batch.append({'symbol':'X','timestamp':0,'price':0,'volume':0})
    monkeypatch.setattr(consumer, '_process_batch', lambda: called.update(processed=called['processed']+1))
    monkeypatch.setattr(consumer.offset_manager, 'commit', lambda async_commit=False: called.update(committed=called['committed']+1))

    consumer.on_stop()
    assert called['processed'] == 1
    assert called['committed'] == 1
