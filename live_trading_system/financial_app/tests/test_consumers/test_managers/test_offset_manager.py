import pytest
import time
from types import SimpleNamespace
from unittest.mock import MagicMock
from confluent_kafka import TopicPartition, KafkaException

from app.consumers.managers.offset_manager import OffsetManager
from app.consumers.base.error import CommitError


class DummyMessage:
    """Minimal stand‑in for confluent_kafka.Message."""
    def __init__(self, topic, partition, offset):
        self._topic = topic
        self._partition = partition
        self._offset = offset

    def topic(self):
        return self._topic

    def partition(self):
        return self._partition

    def offset(self):
        return self._offset


@pytest.fixture
def consumer():
    return MagicMock()


@pytest.fixture
def manager(consumer):
    # small thresholds for fast tests
    return OffsetManager(
        consumer=consumer,
        auto_commit=False,
        commit_interval_ms=100,
        commit_threshold=3,
    )


def test_track_message_and_uncommitted_count(manager):
    msg = DummyMessage("t", 0, 5)
    manager.track_message(msg)
    # offset stored is +1
    assert manager._offsets == {("t", 0): 6}
    assert manager._uncommitted_count == 1

    # tracking a smaller offset does not overwrite
    manager.track_message(DummyMessage("t", 0, 3))
    assert manager._offsets == {("t", 0): 6}
    assert manager._uncommitted_count == 2


def test_track_message_autocommit(consumer):
    mgr = OffsetManager(consumer=consumer, auto_commit=True)
    mgr.track_message(DummyMessage("x", 1, 10))
    assert mgr._offsets == {}
    assert mgr._uncommitted_count == 0


def test_should_commit_threshold(manager):
    manager._uncommitted_count = 3
    assert manager.should_commit()


def test_should_commit_time(manager):
    # simulate last commit far in the past
    now_ms = time.time() * 1000
    manager._last_commit_time = now_ms - 200
    manager._uncommitted_count = 0
    assert manager.should_commit()


def test_should_commit_false(manager):
    manager._uncommitted_count = 1
    manager._last_commit_time = time.time() * 1000
    assert not manager.should_commit()


def test_commit_no_offsets(manager, consumer):
    manager.commit()
    consumer.commit.assert_not_called()


def test_commit_success(manager, consumer):
    # track two partitions
    manager.track_message(DummyMessage("a", 0, 1))
    manager.track_message(DummyMessage("b", 1, 2))

    manager.commit(async_commit=False)

    # verify call once
    consumer.commit.assert_called_once()
    # extract the offsets passed
    kwargs = consumer.commit.call_args.kwargs
    tps = kwargs["offsets"]
    seen = {(tp.topic, tp.partition, tp.offset) for tp in tps}
    assert seen == {("a", 0, 2), ("b", 1, 3)}

    # uncommitted count reset
    assert manager._uncommitted_count == 0


def test_commit_exception(manager, consumer):
    consumer.commit.side_effect = KafkaException("boom")
    manager.track_message(DummyMessage("t", 0, 0))
    with pytest.raises(CommitError):
        manager.commit()


def test_commit_message_success(manager, consumer):
    # pre‑populate tracking
    manager._offsets = {("t", 0): 5}
    manager._uncommitted_count = 1

    msg = DummyMessage("t", 0, 4)
    manager.commit_message(msg, async_commit=False)

    consumer.commit.assert_called_once_with(message=msg, asynchronous=False)
    # offset removed and count decremented
    assert manager._offsets == {}
    assert manager._uncommitted_count == 0


def test_commit_message_exception(manager, consumer):
    consumer.commit.side_effect = KafkaException("err")
    with pytest.raises(CommitError):
        manager.commit_message(DummyMessage("x", 1, 1))


def test_get_consumer_lag_success(manager, consumer):
    # assignment returns two partitions
    tp0 = TopicPartition("t", 0)
    tp1 = TopicPartition("t", 1)
    consumer.assignment.return_value = [tp0, tp1]

    # watermark offsets for each
    consumer.get_watermark_offsets.side_effect = [(0, 10), (0, 20)]
    # position returns list with .offset
    p0 = SimpleNamespace(offset=7)
    p1 = SimpleNamespace(offset=15)
    consumer.position.return_value = [p0, p1]

    lag = manager.get_consumer_lag()
    assert lag == {("t", 0): 3, ("t", 1): 5}


def test_get_consumer_lag_exception(manager, consumer):
    consumer.assignment.side_effect = KafkaException("fail")
    assert manager.get_consumer_lag() == {}


def test_reset_offsets_latest(manager, consumer):
    # set some state
    manager._offsets = {("t", 0): 10}
    manager._uncommitted_count = 5
    consumer.assignment.return_value = [TopicPartition("t", 0)]

    manager.reset_offsets("latest")

    consumer.seek_to_end.assert_called_once()
    assert manager._offsets == {}
    assert manager._uncommitted_count == 0


def test_reset_offsets_earliest(manager, consumer):
    manager._offsets = {("x", 1): 20}
    manager._uncommitted_count = 2
    consumer.assignment.return_value = []

    manager.reset_offsets("earliest")

    consumer.seek_to_beginning.assert_called_once()
    assert manager._offsets == {}
    assert manager._uncommitted_count == 0


def test_reset_offsets_invalid_strategy(manager):
    with pytest.raises(ValueError):
        manager.reset_offsets("middle")


def test_reset_offsets_exception(manager, consumer):
    consumer.assignment.return_value = []
    consumer.seek_to_end.side_effect = KafkaException("oops")
    with pytest.raises(CommitError):
        manager.reset_offsets("latest")
