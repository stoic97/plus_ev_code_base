"""
Kafka offset management utilities.

This module provides utilities for managing Kafka consumer offsets,
including manual commit strategies and offset storage.
"""

import logging
import threading
import time
from typing import Dict, List, Tuple

from confluent_kafka import Consumer, TopicPartition, KafkaException, Message

from app.consumers.base.error import CommitError

# Set up logging
logger = logging.getLogger(__name__)


class OffsetManager:
    """
    Manager for Kafka consumer offsets.
    
    Provides utilities for tracking and committing consumer offsets
    with various commit strategies.
    """
    
    def __init__(
        self,
        consumer: Consumer,
        auto_commit: bool = False,
        commit_interval_ms: int = 5000,
        commit_threshold: int = 100,
    ):
        """
        Initialize a new offset manager.
        
        Args:
            consumer: Kafka consumer instance
            auto_commit: Whether auto-commit is enabled
            commit_interval_ms: Interval between commits in ms
            commit_threshold: Number of messages before committing
        """
        self.consumer = consumer
        self.auto_commit = auto_commit
        self.commit_interval_ms = commit_interval_ms
        self.commit_threshold = commit_threshold
        
        # Offset tracking
        self._offsets: Dict[Tuple[str, int], int] = {}  # (topic, partition) -> offset
        self._uncommitted_count = 0
        self._last_commit_time = time.time() * 1000  # milliseconds
        
        # Thread safety
        self._lock = threading.RLock()
    
    def track_message(self, message: Message) -> None:
        """
        Track a message's offset for later commit.
        
        Args:
            message: Kafka message to track
        """
        if self.auto_commit:
            return  # No need to track if auto-commit is enabled
            
        with self._lock:
            topic = message.topic()
            partition = message.partition()
            offset = message.offset() + 1
            key = (topic, partition)
            # store only highest offset
            if key not in self._offsets or offset > self._offsets[key]:
                self._offsets[key] = offset
            self._uncommitted_count += 1
    
    def should_commit(self) -> bool:
        """
        Check if offsets should be committed based on commit strategy.
        """
        if self.auto_commit:
            return False
            
        with self._lock:
            if self._uncommitted_count >= self.commit_threshold:
                return True
            now = time.time() * 1000
            if now - self._last_commit_time >= self.commit_interval_ms:
                return True
            return False
    
    def commit(self, async_commit: bool = True) -> None:
        """
        Commit tracked offsets to Kafka.
        """
        if self.auto_commit:
            return
            
        with self._lock:
            if not self._offsets:
                return
            topic_partitions = [
                TopicPartition(topic, partition, offset)
                for (topic, partition), offset in self._offsets.items()
            ]
            try:
                self.consumer.commit(offsets=topic_partitions, asynchronous=async_commit)
                # reset
                self._uncommitted_count = 0
                self._last_commit_time = time.time() * 1000
                # avoid repr issues
                logger.debug(f"Committed {len(topic_partitions)} offsets")
            except KafkaException as e:
                raise CommitError(f"Failed to commit offsets: {e}")
    
    def commit_message(self, message: Message, async_commit: bool = True) -> None:
        """
        Commit a specific message's offset.
        """
        if self.auto_commit:
            return
            
        try:
            self.consumer.commit(message=message, asynchronous=async_commit)
            with self._lock:
                topic = message.topic()
                partition = message.partition()
                offset = message.offset() + 1
                key = (topic, partition)
                if key in self._offsets and self._offsets[key] <= offset:
                    del self._offsets[key]
                self._uncommitted_count = max(0, self._uncommitted_count - 1)
            logger.debug(f"Committed offset for {topic}:{partition} at {message.offset()}")
        except KafkaException as e:
            raise CommitError(f"Failed to commit message offset: {e}")
    
    def get_consumer_lag(self) -> Dict[Tuple[str, int], int]:
        """
        Get consumer lag for each partition.
        """
        lag: Dict[Tuple[str, int], int] = {}
        try:
            assignment = self.consumer.assignment()
            # gather highs
            highs: List[int] = []
            for tp in assignment:
                _, high = self.consumer.get_watermark_offsets(tp)
                highs.append(high)
            # batch positions
            positions = self.consumer.position(assignment)
            for tp, high, pos in zip(assignment, highs, positions):
                lag[(tp.topic, tp.partition)] = high - pos.offset
        except KafkaException as e:
            logger.error(f"Failed to get consumer lag: {e}")
        return lag
    
    def reset_offsets(self, strategy: str = 'latest') -> None:
        """
        Reset consumer offsets using the specified strategy.
        """
        try:
            assignment = self.consumer.assignment()
            if strategy == 'latest':
                self.consumer.seek_to_end()
                logger.info(f"Reset offsets to latest for {len(assignment)} partitions")
            elif strategy == 'earliest':
                self.consumer.seek_to_beginning()
                logger.info(f"Reset offsets to earliest for {len(assignment)} partitions")
            else:
                raise ValueError(f"Unsupported offset reset strategy: {strategy}")
            with self._lock:
                self._offsets = {}
                self._uncommitted_count = 0
                self._last_commit_time = time.time() * 1000
        except KafkaException as e:
            raise CommitError(f"Failed to reset offsets: {e}")
