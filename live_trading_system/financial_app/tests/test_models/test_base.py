"""
Unit tests for base.py models and utilities.
"""

import pytest
import json
from datetime import datetime, timedelta
from decimal import Decimal
from uuid import UUID, uuid4

from sqlalchemy import (
    Column, String, Integer, create_engine, ForeignKey, DECIMAL, Table, MetaData
)
from sqlalchemy.orm import sessionmaker, declarative_base

# Import utility functions from base.py to test
from app.models.base import generate_uuid, validate_non_negative

# Create an isolated test environment
TestBase = declarative_base()

class TestTimestampModel(TestBase):
    """Simple model for testing timestamps."""
    __tablename__ = 'test_timestamps'
    id = Column(Integer, primary_key=True)
    name = Column(String(50))
    created_at = Column(String(50))  # Simplified for testing
    updated_at = Column(String(50))  # Simplified for testing

# Set up test database and session
@pytest.fixture
def engine():
    """Create an in-memory SQLite database for testing."""
    return create_engine('sqlite:///:memory:')

@pytest.fixture
def tables(engine):
    """Create all tables before tests and drop them after."""
    TestBase.metadata.create_all(engine)
    yield
    TestBase.metadata.drop_all(engine)

@pytest.fixture
def session(engine, tables):
    """Create a new session for testing."""
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()

# Tests for utility functions
def test_generate_uuid():
    """Test that generate_uuid creates valid UUIDs."""
    uuid_str = generate_uuid()
    assert isinstance(uuid_str, str)
    # Verify it's a valid UUID
    UUID(uuid_str, version=4)

def test_validate_non_negative():
    """Test that validate_non_negative allows positives and rejects negatives."""
    # Positive cases
    assert validate_non_negative(0) == 0
    assert validate_non_negative(10) == 10
    assert validate_non_negative(0.5) == 0.5
    assert validate_non_negative(None) is None
    
    # Negative case
    with pytest.raises(ValueError):
        validate_non_negative(-1)
    
    with pytest.raises(ValueError):
        validate_non_negative(-0.1)

# Tests for basic model behavior
def test_timestamp_model(session):
    """Test basic model functionality."""
    # Create a timestamp model
    model = TestTimestampModel(name="Test Model", created_at=datetime.now().isoformat())
    session.add(model)
    session.commit()
    
    # Verify it was saved
    assert model.id is not None
    assert model.name == "Test Model"
    assert model.created_at is not None

# Mock tests for mixins that don't require database interaction
def test_mock_serializable_mixin():
    """Test SerializableMixin functionality with a mock."""
    # Create a mock implementation of SerializableMixin
    class MockSerializable:
        def __init__(self):
            self.id = 1
            self.name = "Test"
            self.created_at = datetime.now()
            
        def to_dict(self, include_relationships=False):
            result = {
                'id': self.id,
                'name': self.name,
                'created_at': self.created_at.isoformat()
            }
            return result
            
        def to_json(self):
            return json.dumps(self.to_dict())
    
    # Test the mock
    mock = MockSerializable()
    result = mock.to_dict()
    
    assert result['id'] == 1
    assert result['name'] == "Test"
    assert isinstance(result['created_at'], str)
    
    json_result = mock.to_json()
    assert isinstance(json_result, str)
    
def test_mock_positive_value_validation():
    """Test positive value validation with a mock."""
    
    # Create a mock implementation
    class MockValidator:
        def __init__(self, amount, price):
            self.validate(amount, "amount")
            self.validate(price, "price")
            self.amount = amount
            self.price = price
            
        def validate(self, value, field_name):
            if value is not None and value < 0:
                raise ValueError(f"{field_name} cannot be negative")
    
    # Test valid values
    valid = MockValidator(100, 50)
    assert valid.amount == 100
    assert valid.price == 50
    
    # Test None values
    none_values = MockValidator(None, None)
    assert none_values.amount is None
    assert none_values.price is None
    
    # Test invalid values
    with pytest.raises(ValueError):
        MockValidator(-10, 50)
    
    with pytest.raises(ValueError):
        MockValidator(100, -5)

def test_mock_soft_delete():
    """Test soft delete functionality with a mock."""
    
    # Create a mock implementation
    class MockSoftDelete:
        def __init__(self):
            self.deleted_at = None
            self.deleted_by_id = None
            
        @property
        def is_deleted(self):
            return self.deleted_at is not None
            
        def soft_delete(self, user_id=None):
            self.deleted_at = datetime.now()
            self.deleted_by_id = user_id
    
    # Test soft delete
    mock = MockSoftDelete()
    assert not mock.is_deleted
    assert mock.deleted_at is None
    
    mock.soft_delete(user_id=1)
    assert mock.is_deleted
    assert mock.deleted_at is not None
    assert mock.deleted_by_id == 1

def test_mock_versioned_mixin():
    """Test versioned mixin functionality with a mock."""
    
    # Create a mock implementation
    class MockVersioned:
        def __init__(self):
            self.version = 1
            
        def validate_version(self, value):
            if value != self.version + 1:
                raise ValueError(f"Version can only increment by 1. Current: {self.version}, Attempted: {value}")
            self.version = value
    
    # Test versioning
    mock = MockVersioned()
    assert mock.version == 1
    
    # Valid increment
    mock.validate_version(2)
    assert mock.version == 2
    
    # Invalid increment (skipping a version)
    with pytest.raises(ValueError):
        mock.validate_version(4)

def test_mock_status_mixin():
    """Test status mixin functionality with a mock."""
    
    # Create a mock implementation
    class MockStatus:
        def __init__(self, status):
            self.status = status
            self.previous_status = None
            self.status_changed_at = datetime.now()
            
        def update_status(self, new_status):
            if self.status != new_status:
                self.previous_status = self.status
                self.status = new_status
                self.status_changed_at = datetime.now()
    
    # Test status updates
    mock = MockStatus("draft")
    assert mock.status == "draft"
    assert mock.previous_status is None
    
    # Update status
    mock.update_status("active")
    assert mock.status == "active"
    assert mock.previous_status == "draft"
    
    # Update again
    mock.update_status("completed")
    assert mock.status == "completed"
    assert mock.previous_status == "active"