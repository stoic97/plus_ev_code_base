import pytest
from decimal import Decimal
from datetime import datetime, timedelta

from app.consumers.utils.validation import (
    validate_message_structure,
    validate_numeric_field,
    validate_timestamp,
    validate_symbol,
    validate_ohlcv_consistency,
    validate_orderbook_structure,
    validate_ohlcv_message,
    validate_trade_message,
    validate_orderbook_message,
    get_validator_for_message_type,
    OHLCV_REQUIRED_FIELDS,
    TRADE_REQUIRED_FIELDS,
    ORDERBOOK_REQUIRED_FIELDS
)
from app.consumers.base.error import ValidationError

# Sample valid messages for reuse
good_ohlcv = {
    'symbol': 'BTCUSD',
    'open': '100.0',
    'high': 110.0,
    'low': 90.0,
    'close': Decimal('105.0'),
    'volume': 1000,
    'timestamp': int(datetime.now().timestamp() * 1000),
    'interval': '1m'
}

good_trade = {
    'symbol': 'ETHUSD',
    'price': 200.5,
    'volume': '10',
    'timestamp': datetime.now().isoformat()
}

good_orderbook = {
    'symbol': 'XRPUSD',
    'timestamp': datetime.now().isoformat(),
    'bids': [[1.0, 100], [0.9, 200]],
    'asks': [[1.1, 100], [1.2, 50]]
}

# validate_message_structure

def test_validate_message_structure_pass():
    validate_message_structure({'a': 1, 'b': 2}, {'a', 'b'})


def test_validate_message_structure_missing():
    with pytest.raises(ValidationError) as exc:
        validate_message_structure({'a': 1}, {'a', 'b'})
    assert 'Missing required fields' in str(exc.value)

# validate_numeric_field

def test_validate_numeric_field_valid():
    validate_numeric_field('5.5', 'test')
    validate_numeric_field(0, 'zero', allow_zero=True)


def test_validate_numeric_field_zero_not_allowed():
    with pytest.raises(ValidationError) as exc:
        validate_numeric_field(0, 'zero', allow_zero=False)
    assert "cannot be zero" in str(exc.value)


def test_validate_numeric_field_non_numeric():
    with pytest.raises(ValidationError):
        validate_numeric_field('abc', 'test')


def test_validate_numeric_field_below_min():
    with pytest.raises(ValidationError) as exc:
        validate_numeric_field(-1, 'neg', min_value=0)
    assert "below minimum value" in str(exc.value)


def test_validate_numeric_field_above_max():
    with pytest.raises(ValidationError) as exc:
        validate_numeric_field(11, 'high', max_value=10)
    assert "exceeds maximum value" in str(exc.value)

# validate_timestamp

def test_validate_timestamp_epoch_seconds():
    now = datetime.now().timestamp()
    validate_timestamp(now)


def test_validate_timestamp_epoch_millis():
    now_ms = datetime.now().timestamp() * 1000
    validate_timestamp(now_ms)


def test_validate_timestamp_iso_string():
    iso = datetime.now().isoformat()
    validate_timestamp(iso)


def test_validate_timestamp_iso_z_format():
    z = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    validate_timestamp(z)


def test_validate_timestamp_invalid_type():
    with pytest.raises(ValidationError):
        validate_timestamp({'ts': 123})


def test_validate_timestamp_unreasonable_past():
    # Year 1970 < 2000
    with pytest.raises(ValidationError):
        validate_timestamp(123)


def test_validate_timestamp_unreasonable_future():
    # Year 3000
    future = datetime(3000, 1, 1).isoformat()
    with pytest.raises(ValidationError):
        validate_timestamp(future)

# validate_symbol

def test_validate_symbol_valid():
    validate_symbol('AAPL')
    validate_symbol('BTC-USD')


def test_validate_symbol_non_string():
    with pytest.raises(ValidationError):
        validate_symbol(123)


def test_validate_symbol_empty():
    with pytest.raises(ValidationError):
        validate_symbol('')


def test_validate_symbol_too_long():
    with pytest.raises(ValidationError):
        validate_symbol('A' * 21)


def test_validate_symbol_invalid_chars():
    with pytest.raises(ValidationError):
        validate_symbol('BAD$SYM')

# validate_ohlcv_consistency

def test_validate_ohlcv_consistency_valid():
    validate_ohlcv_consistency(good_ohlcv)


def test_validate_ohlcv_consistency_high_less_low():
    bad = dict(good_ohlcv)
    bad['high'] = 80
    with pytest.raises(ValidationError):
        validate_ohlcv_consistency(bad)


def test_validate_ohlcv_consistency_invalid_prices():
    bad = dict(good_ohlcv)
    bad['open'] = 'abc'
    with pytest.raises(ValidationError):
        validate_ohlcv_consistency(bad)

# validate_orderbook_structure

def test_validate_orderbook_structure_valid(capsys):
    # Should not raise
    validate_orderbook_structure(good_orderbook)


def test_validate_orderbook_structure_bids_not_list():
    bad = dict(good_orderbook)
    bad['bids'] = 'notalist'
    with pytest.raises(ValidationError):
        validate_orderbook_structure(bad)


def test_validate_orderbook_structure_ask_entry_invalid():
    bad = dict(good_orderbook)
    bad['asks'] = [['bad']]
    with pytest.raises(ValidationError):
        validate_orderbook_structure(bad)

# validate_ohlcv_message

def test_validate_ohlcv_message_valid():
    validate_ohlcv_message(good_ohlcv)


def test_validate_ohlcv_message_missing_field():
    bad = dict(good_ohlcv)
    bad.pop('open')
    with pytest.raises(ValidationError):
        validate_ohlcv_message(bad)


def test_validate_ohlcv_message_bad_interval():
    bad = dict(good_ohlcv)
    bad['interval'] = 5
    with pytest.raises(ValidationError):
        validate_ohlcv_message(bad)

# validate_trade_message

def test_validate_trade_message_valid():
    validate_trade_message(good_trade)


def test_validate_trade_message_missing_field():
    bad = dict(good_trade)
    bad.pop('price')
    with pytest.raises(ValidationError):
        validate_trade_message(bad)


def test_validate_trade_message_zero_price():
    bad = dict(good_trade)
    bad['price'] = 0
    with pytest.raises(ValidationError):
        validate_trade_message(bad)


def test_validate_trade_message_invalid_side():
    bad = dict(good_trade)
    bad['side'] = 'hold'
    with pytest.raises(ValidationError):
        validate_trade_message(bad)

# validate_orderbook_message

def test_validate_orderbook_message_valid():
    validate_orderbook_message(good_orderbook)


def test_validate_orderbook_message_missing_field():
    bad = dict(good_orderbook)
    bad.pop('asks')
    with pytest.raises(ValidationError):
        validate_orderbook_message(bad)


def test_validate_orderbook_message_invalid_depth():
    bad = dict(good_orderbook)
    bad['depth'] = 0
    with pytest.raises(ValidationError):
        validate_orderbook_message(bad)

# get_validator_for_message_type

def test_get_validator_for_message_type_valid():
    assert get_validator_for_message_type('ohlcv') == validate_ohlcv_message
    assert get_validator_for_message_type('trade') == validate_trade_message
    assert get_validator_for_message_type('orderbook') == validate_orderbook_message


def test_get_validator_for_message_type_invalid():
    with pytest.raises(ValueError):
        get_validator_for_message_type('unknown')
