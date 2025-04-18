import threading
import pytest

from app.consumers.utils.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerError,
    CircuitState,
    circuit_breaker as cb_decorator,
)

# Alias module for time monkeypatching
import app.consumers.utils.circuit_breaker as cb_module

class DummyTime:
    """Simple dummy time provider for controlling time in tests."""
    def __init__(self, start: float = 1.0):  # start non-zero to avoid falsy zero timestamp
        self._t = start

    def time(self) -> float:
        return self._t

    def advance(self, seconds: float):
        self._t += seconds

@pytest.fixture(autouse=True)
def clear_registry():
    # Reset any existing breakers before each test
    CircuitBreaker._instances.clear()
    yield
    CircuitBreaker._instances.clear()

@pytest.fixture()
def dummy_time(monkeypatch):
    dt = DummyTime(start=1.0)
    # Patch the time.time in the circuit_breaker module
    monkeypatch.setattr(cb_module.time, 'time', dt.time)
    return dt


def test_initial_state_and_registry():
    # Create a new breaker
    cb = CircuitBreaker('test')
    assert cb.name == 'test'
    assert cb.state == CircuitState.CLOSED
    assert cb.failure_count == 0
    # get should return same instance
    cb2 = CircuitBreaker.get('test')
    assert cb is cb2
    # get_all should include it
    all_b = CircuitBreaker.get_all()
    assert 'test' in all_b and all_b['test'] is cb


def test_reset_and_reset_all_and_record_success(dummy_time):
    cb = CircuitBreaker('rtest', failure_threshold=1)
    # Force open
    cb.record_failure()
    assert cb.state == CircuitState.OPEN
    # Reset single
    cb.reset()
    assert cb.state == CircuitState.CLOSED
    assert cb.failure_count == 0
    # Test half_open to closed via record_success
    cb.state = CircuitState.HALF_OPEN
    cb.failure_count = 5
    cb.record_success()
    assert cb.state == CircuitState.CLOSED
    assert cb.failure_count == 0
    # Reset all ensures registry cleared instances reset
    cb2 = CircuitBreaker('another', failure_threshold=1)
    # open cb2 too
    cb2.record_failure()
    CircuitBreaker.reset_all()
    assert cb.state == CircuitState.CLOSED
    assert cb2.state == CircuitState.CLOSED


def test_record_failure_and_excluded_and_reopen(dummy_time):
    cb = CircuitBreaker('failtest', failure_threshold=3)
    # record 2 failures
    cb.record_failure()
    cb.record_failure()
    assert cb.failure_count == 2
    assert cb.state == CircuitState.CLOSED
    # excluded exceptions
    cb_exc = CircuitBreaker('exc', excluded_exceptions=[KeyError], failure_threshold=1)
    cb_exc.record_failure(KeyError('ignore'))
    assert cb_exc.failure_count == 0
    # threshold breach
    cb.record_failure()
    assert cb.failure_count == 3
    assert cb.state == CircuitState.OPEN
    # simulate half_open then failure reopens
    cb.state = CircuitState.HALF_OPEN
    cb.record_failure(Exception('fail'))
    assert cb.state == CircuitState.OPEN


def test_allow_request_transitions(dummy_time):
    cb = CircuitBreaker('allow', failure_threshold=1, recovery_timeout=10, half_open_timeout=5)
    # closed always allow
    assert cb.allow_request()
    # open within timeout
    cb.record_failure()  # now open, last_failure_time=1
    assert not cb.allow_request()
    # after recovery_timeout passes
    dummy_time.advance(11)
    allowed = cb.allow_request()
    assert allowed
    assert cb.state == CircuitState.HALF_OPEN
    first_test_time = cb.last_test_time
    # half-open before half_open_timeout
    assert not cb.allow_request()
    # after half_open_timeout
    dummy_time.advance(6)
    assert cb.allow_request()
    # ensure last_test_time updated
    assert cb.last_test_time != first_test_time


def test_execute_and_decorator(dummy_time):
    cb = CircuitBreaker('exec', failure_threshold=1, recovery_timeout=5)

    # successful execution
    def success_fn(x):
        return x * 2
    result = cb.execute(success_fn, 3)
    assert result == 6
    assert cb.state == CircuitState.CLOSED

    # failing execution raises underlying exception
    def fail_fn():
        raise ValueError('oops')
    with pytest.raises(ValueError):
        cb.execute(fail_fn)
    assert cb.failure_count == 1
    assert cb.state == CircuitState.OPEN

    # open circuit raises CircuitBreakerError
    with pytest.raises(CircuitBreakerError) as excinfo:
        cb.execute(success_fn, 5)
    assert "Circuit 'exec' is open" in str(excinfo.value)

    # test decorator syntax
    cb2 = CircuitBreaker('dec', failure_threshold=1, recovery_timeout=2)
    @cb2
    def decorated(x):  # CircuitBreaker.__call__ used
        return x + 1
    assert decorated(4) == 5


def test_get_status_and_factory_updates(dummy_time):
    # Create default
    default = CircuitBreaker.get('stat')
    status = default.get_status()
    assert status['name'] == 'stat'
    assert status['state'] == CircuitState.CLOSED.value
    assert status['failure_count'] == 0
    # Use decorator factory to update settings
    @cb_decorator('stat', failure_threshold=2, recovery_timeout=7, half_open_timeout=3)
    def foo():
        return 'bar'
    # Underlying instance should be updated
    inst = CircuitBreaker.get('stat')
    assert inst.failure_threshold == 2
    assert inst.recovery_timeout == 7
    assert inst.half_open_timeout == 3
    # Execution still works
    assert foo() == 'bar'
