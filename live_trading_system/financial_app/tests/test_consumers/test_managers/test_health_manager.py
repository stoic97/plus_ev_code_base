import pytest
import time
from unittest.mock import MagicMock

from confluent_kafka import Consumer, KafkaException
import app.consumers.managers.health_manager as hm

from app.consumers.managers.health_manager import (
    HealthStatus,
    HealthCheckConfig,
    ConsumerHealthCheck,
    HealthManager,
    get_health_manager,
)


class DummyOffsetManager:
    def __init__(self, lag_dict):
        self._lag = lag_dict

    def get_consumer_lag(self):
        return self._lag


@pytest.fixture
def consumer():
    return MagicMock(spec=Consumer)


@pytest.fixture
def health_check(monkeypatch, consumer):
    # custom config for faster thresholds
    cfg = HealthCheckConfig(
        max_lag_messages=5,
        critical_lag_messages=10,
        max_idle_seconds=2,
        critical_idle_seconds=4,
        min_throughput_messages_per_second=1.0,
        low_throughput_messages_per_second=0.5,
        max_error_rate_percent=10.0,
        critical_error_rate_percent=20.0
    )
    hc = ConsumerHealthCheck(
        consumer_id="cid",
        consumer=consumer,
        offset_manager=None,
        config=cfg,
        health_check_interval_seconds=1
    )
    # Force check to run on first call
    hc.last_health_check_time = 0
    return hc


def test_record_message_processed_and_limit(health_check):
    assert health_check.messages_processed == 0
    # simulate 1005 messages to hit the 1000-limit
    for i in range(1005):
        health_check.record_message_processed(processing_time_ms=i)
    assert health_check.messages_processed == 1005
    # processing_times should keep only the last 1000
    assert len(health_check.processing_times) == 1000
    # last_message_time updated
    assert health_check.last_message_time is not None


def test_record_error(health_check):
    assert health_check.errors_count == 0
    health_check.record_error()
    health_check.record_error()
    assert health_check.errors_count == 2


def test_add_alert_callback(health_check):
    cb = lambda sev, details: None
    health_check.add_alert_callback(cb)
    # Protected attr _alert_callbacks
    assert cb in health_check._alert_callbacks


def test_run_health_check_degraded_on_low_throughput(monkeypatch, health_check):
    # advance time so interval passes
    monkeypatch.setattr(hm.time, "time", lambda: 2.0)
    # no messages => throughput 0 < low threshold 0.5
    status = health_check.run_health_check()
    assert status == HealthStatus.DEGRADED
    assert any("Low throughput" in issue for issue in health_check.health_issues)


def test_run_health_check_healthy_when_good_throughput(monkeypatch, health_check):
    # record some messages before the check
    health_check.messages_processed = 3
    health_check.messages_processed_last_check = 0
    monkeypatch.setattr(hm.time, "time", lambda: 2.0)
    status = health_check.run_health_check()
    assert status == HealthStatus.HEALTHY
    assert health_check.health_issues == []


def test_run_health_check_stalled_on_idle(monkeypatch, health_check):
    # simulate last message long ago
    health_check.last_message_time = 0.0
    # no extra messages => throughput 0
    monkeypatch.setattr(hm.time, "time", lambda: 7.0)
    # register a spy callback
    alerts = []
    health_check.add_alert_callback(lambda sev, det: alerts.append((sev, det)))
    status = health_check.run_health_check()
    assert status == HealthStatus.STALLED
    # should have an idle/stalled issue
    assert any("stalled" in issue.lower() for issue in health_check.health_issues)
    # callback should have been invoked with warning
    assert alerts and alerts[0][0] == "warning"
    # details include consumer_id
    assert alerts[0][1]["consumer_id"] == "cid"


def test_run_health_check_unhealthy_on_critical_error_rate(monkeypatch, health_check):
    # simulate no messages, two errors
    health_check.messages_processed_last_check = 0
    health_check.messages_processed = 0
    health_check.errors_last_check = 0
    health_check.errors_count = 2
    monkeypatch.setattr(hm.time, "time", lambda: 2.0)
    alerts = []
    health_check.add_alert_callback(lambda sev, det: alerts.append((sev, det)))
    status = health_check.run_health_check()
    # error_rate = 100% > critical_error_rate_percent
    assert status == HealthStatus.UNHEALTHY
    assert any("Critical error rate" in issue for issue in health_check.health_issues)
    assert alerts and alerts[0][0] == "critical"


def test_run_health_check_unhealthy_on_critical_lag(monkeypatch, consumer):
    # attach offset manager returning high lag
    hc = ConsumerHealthCheck(
        consumer_id="cid2",
        consumer=consumer,
        offset_manager=DummyOffsetManager({("t", 0): 15}),
        config=HealthCheckConfig(
            max_lag_messages=5,
            critical_lag_messages=10
        ),
        health_check_interval_seconds=1
    )
    hc.last_health_check_time = 0
    monkeypatch.setattr(hm.time, "time", lambda: 2.0)
    alerts = []
    hc.add_alert_callback(lambda sev, det: alerts.append((sev, det)))
    status = hc.run_health_check()
    assert status == HealthStatus.UNHEALTHY
    assert any("Critical consumer lag" in issue for issue in hc.health_issues)
    assert alerts and alerts[0][0] == "critical"


def test_get_health_info(monkeypatch, health_check):
    # simulate healthy state
    health_check.messages_processed = 2
    health_check.errors_count = 1
    health_check.last_message_time = time.time() - 1
    monkeypatch.setattr(hm.time, "time", lambda: health_check.last_message_time + 2)
    # force a check so status updates
    health_check.last_health_check_time = 0
    info = health_check.get_health_info()
    assert info["consumer_id"] == "cid"
    assert "status" in info and isinstance(info["status"], str)
    assert "messages_processed" in info and info["messages_processed"] == 2
    assert "errors_count" in info and info["errors_count"] == 1
    assert "avg_processing_time_ms" in info


def test_health_manager_register_and_get():
    mgr = HealthManager()
    c = MagicMock(spec=Consumer)
    hc1 = mgr.register_consumer("a", c)
    assert mgr.get_consumer_health("a") is hc1
    assert mgr.get_consumer_health("missing") is None


def test_health_manager_run_all_and_overall(monkeypatch):
    mgr = HealthManager()
    # create two fake checks
    hc1 = MagicMock(run_health_check=lambda: HealthStatus.DEGRADED, status=HealthStatus.DEGRADED)
    hc2 = MagicMock(run_health_check=lambda: HealthStatus.UNHEALTHY, status=HealthStatus.UNHEALTHY)
    mgr._health_checks = {"c1": hc1, "c2": hc2}
    results = mgr.run_all_health_checks()
    assert results == {"c1": HealthStatus.DEGRADED, "c2": HealthStatus.UNHEALTHY}
    # overall should pick the worst
    assert mgr.get_overall_health() == HealthStatus.UNHEALTHY

    # if only healthy
    mgr._health_checks = {"c1": MagicMock(run_health_check=lambda: HealthStatus.HEALTHY, status=HealthStatus.HEALTHY)}
    assert mgr.get_overall_health() == HealthStatus.HEALTHY

    # no consumers
    mgr._health_checks = {}
    assert mgr.get_overall_health() == HealthStatus.UNKNOWN


def test_health_manager_alert_propagation():
    mgr = HealthManager()
    calls = []
    cb = lambda sev, det: calls.append((sev, det))
    mgr.add_alert_callback(cb)

    # new health check should have the callback
    c = MagicMock(spec=Consumer)
    hc = mgr.register_consumer("z", c)
    assert cb in hc._alert_callbacks

    # triggering an alert inside hc should call cb
    hc._send_alerts("info", {"foo": "bar"})
    assert calls and calls[-1][0] == "info"


def test_health_manager_summaries(monkeypatch):
    mgr = HealthManager()
    # stub two checks
    hc1 = MagicMock(status=HealthStatus.DEGRADED)
    hc2 = MagicMock(status=HealthStatus.UNHEALTHY)
    mgr._health_checks = {"c1": hc1, "c2": hc2}

    summary = mgr.get_health_summary()
    assert "timestamp" in summary
    assert summary["overall_status"] in {"degraded", "unhealthy", "healthy", "unknown", "stalled"}
    assert summary["consumer_count"] == 2
    assert isinstance(summary["consumer_statuses"], dict)
    assert "c1" in summary["consumer_statuses"]
    assert isinstance(summary["unhealthy_consumers"], list)
    assert isinstance(summary["degraded_consumers"], list)

    # detailed info
    # stub get_health_info
    hc1.get_health_info.return_value = {"a": 1}
    hc2.get_health_info.return_value = {"b": 2}
    detail = mgr.get_detailed_health_info()
    assert detail == {"c1": {"a": 1}, "c2": {"b": 2}}


def test_global_get_health_manager():
    gm1 = get_health_manager()
    gm2 = get_health_manager()
    assert isinstance(gm1, HealthManager)
    assert gm1 is gm2
