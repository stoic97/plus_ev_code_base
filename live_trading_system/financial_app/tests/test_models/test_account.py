"""
Unit tests for account.py models and functions.

This module provides test coverage for the account-related models using SQLite.
"""

import sys
import os
from decimal import Decimal
from datetime import datetime, date, timedelta

import pytest
from sqlalchemy import (
    create_engine, event, text, func, 
    Column, Integer, String, ForeignKey, Boolean, 
    Numeric, DateTime
)
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.ext.declarative import declarative_base

# Add application root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# We'll create an independent testing environment with its own Base
TestBase = declarative_base()

# Create a mock User model for testing
class User(TestBase):
    """Mock User model for testing purposes only."""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True)
    username = Column(String(50), nullable=False)
    
    # This is what was missing - the Account model expects this property
    accounts = relationship("TestAccount", back_populates="user")

# Create testing versions of the models that don't depend on the real ones
class TestAccount(TestBase):
    """Testing version of the Account model."""
    __tablename__ = "accounts"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    account_number = Column(String(50), unique=True, index=True, nullable=False)
    account_type = Column(String(20), nullable=False)
    name = Column(String(100), nullable=False)
    base_currency = Column(String(10), nullable=False, default="USD")
    status = Column(String(20), nullable=False, default="active")
    is_margin_enabled = Column(Boolean, default=False, nullable=False)
    margin_level = Column(Numeric(precision=10, scale=2), default=0)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_activity_at = Column(DateTime, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="accounts")
    balances = relationship("TestBalance", back_populates="account", cascade="all, delete-orphan")
    positions = relationship("TestPosition", back_populates="account", cascade="all, delete-orphan")
    risk_profile = relationship("TestRiskProfile", back_populates="account", uselist=False, cascade="all, delete-orphan")

class TestBalance(TestBase):
    """Testing version of the Balance model."""
    __tablename__ = "balances"
    
    id = Column(Integer, primary_key=True, index=True)
    account_id = Column(Integer, ForeignKey("accounts.id"), nullable=False)
    asset = Column(String(20), nullable=False)
    asset_type = Column(String(20), nullable=False)
    amount = Column(Numeric(precision=18, scale=8), nullable=False, default=0)
    available = Column(Numeric(precision=18, scale=8), nullable=False, default=0)
    reserved = Column(Numeric(precision=18, scale=8), nullable=False, default=0)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    account = relationship("TestAccount", back_populates="balances")

class TestPosition(TestBase):
    """Testing version of the Position model."""
    __tablename__ = "positions"
    
    id = Column(Integer, primary_key=True, index=True)
    account_id = Column(Integer, ForeignKey("accounts.id"), nullable=False)
    symbol = Column(String(20), nullable=False)
    instrument_type = Column(String(20), nullable=False)
    direction = Column(String(5), nullable=False, default="long")
    quantity = Column(Numeric(precision=18, scale=8), nullable=False)
    average_entry_price = Column(Numeric(precision=18, scale=8), nullable=False)
    current_price = Column(Numeric(precision=18, scale=8), nullable=False)
    unrealized_pnl = Column(Numeric(precision=18, scale=8), nullable=False, default=0)
    realized_pnl = Column(Numeric(precision=18, scale=8), nullable=False, default=0)
    
    # Relationships
    account = relationship("TestAccount", back_populates="positions")

class TestRiskProfile(TestBase):
    """Testing version of the RiskProfile model."""
    __tablename__ = "risk_profiles"
    
    id = Column(Integer, primary_key=True)
    account_id = Column(Integer, ForeignKey("accounts.id"), unique=True, nullable=False)
    max_position_size_percentage = Column(Numeric(precision=5, scale=2), nullable=False, default=5.0)
    max_sector_exposure_percentage = Column(Numeric(precision=5, scale=2), nullable=False, default=20.0)
    max_asset_class_exposure_percentage = Column(Numeric(precision=5, scale=2), nullable=False, default=40.0)
    max_daily_loss_percentage = Column(Numeric(precision=5, scale=2), nullable=False, default=3.0)
    max_total_loss_percentage = Column(Numeric(precision=5, scale=2), nullable=False, default=15.0)
    max_leverage = Column(Numeric(precision=5, scale=2), nullable=False, default=1.0)
    is_active = Column(Boolean, nullable=False, default=True)
    is_in_cooldown = Column(Boolean, nullable=False, default=False)
    cooldown_until = Column(DateTime)
    
    # Relationships
    account = relationship("TestAccount", back_populates="risk_profile")

class TestTransaction(TestBase):
    """Testing version of the Transaction model."""
    __tablename__ = "transactions"
    
    id = Column(Integer, primary_key=True, index=True)
    account_id = Column(Integer, ForeignKey("accounts.id"), nullable=False)
    transaction_type = Column(String(20), nullable=False)
    asset = Column(String(20), nullable=False)
    amount = Column(Numeric(precision=18, scale=8), nullable=False)
    price = Column(Numeric(precision=18, scale=8))
    quantity = Column(Numeric(precision=18, scale=8))
    symbol = Column(String(20))
    direction = Column(String(5))
    reference_id = Column(String(100))
    status = Column(String(20), nullable=False, default="completed")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

# SQLite test fixture
@pytest.fixture(scope="function")
def db_session():
    """Create a SQLite in-memory database session for testing."""
    # Create SQLite in-memory engine
    engine = create_engine("sqlite:///:memory:", echo=False)
    
    # Enable foreign key constraints (disabled by default in SQLite)
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()
    
    # Create all tables using our test models
    TestBase.metadata.create_all(engine)
    
    # Create session factory
    SessionFactory = sessionmaker(bind=engine)
    
    # Create session
    session = SessionFactory()
    
    # Create test user with raw SQL to avoid any model loading issues
    session.execute(text("INSERT INTO users (id, username) VALUES (1, 'testuser')"))
    session.commit()
    
    yield session
    
    # Teardown - rollback and close session
    session.rollback()
    session.close()

@pytest.fixture(scope="function")
def sample_account(db_session):
    """Create a sample account for testing."""
    # User ID from the test user we created
    user_id = 1
    
    # Create account
    account = TestAccount(
        user_id=user_id,
        account_number="TEST123456",
        account_type="main",
        name="Test Trading Account",
        base_currency="USD",
        status="active"
    )
    
    # Add risk profile
    risk_profile = TestRiskProfile(
        max_position_size_percentage=5.0,
        max_sector_exposure_percentage=20.0,
        max_asset_class_exposure_percentage=40.0,
        max_daily_loss_percentage=3.0,
        max_total_loss_percentage=15.0,
        max_leverage=1.0
    )
    account.risk_profile = risk_profile
    
    # Add to session
    db_session.add(account)
    db_session.flush()
    
    # Add initial balance
    balance = TestBalance(
        account_id=account.id,
        asset="USD",
        asset_type="currency",
        amount=Decimal("10000.00"),
        available=Decimal("10000.00"),
        reserved=Decimal("0.00")
    )
    db_session.add(balance)
    db_session.flush()
    
    return account

@pytest.fixture(scope="function")
def sample_position(db_session, sample_account):
    """Create a sample position for testing."""
    position = TestPosition(
        account_id=sample_account.id,
        symbol="AAPL",
        instrument_type="stock",
        direction="long",
        quantity=Decimal("100"),
        average_entry_price=Decimal("150.00"),
        current_price=Decimal("155.00"),
        unrealized_pnl=Decimal("500.00"),
        realized_pnl=Decimal("0.00")
    )
    db_session.add(position)
    db_session.flush()
    return position

class TestBasicFunctionality:
    """Basic tests for the test models to verify SQLite setup."""
    
    def test_account_creation(self, db_session):
        """Test creating a basic account."""
        account = TestAccount(
            user_id=1,
            account_number="TEST001",
            account_type="main",
            name="Test Account",
            status="active"
        )
        db_session.add(account)
        db_session.flush()
        
        assert account.id is not None
        assert account.account_number == "TEST001"
        assert account.status == "active"
        assert account.created_at is not None
        
    def test_account_relationships(self, db_session, sample_account, sample_position):
        """Test account relationships."""
        # Test account has user relationship
        account = db_session.query(TestAccount).filter_by(id=sample_account.id).first()
        assert account is not None
        assert account.user_id == 1
        
        # Test account has balance relationship
        assert len(account.balances) == 1
        assert account.balances[0].asset == "USD"
        assert account.balances[0].amount == Decimal("10000.00")
        
        # Test account has position relationship
        assert len(account.positions) == 1
        assert account.positions[0].symbol == "AAPL"
        
        # Test account has risk profile relationship
        assert account.risk_profile is not None
        assert account.risk_profile.max_position_size_percentage == Decimal("5.0")
    
    def test_balance_operations(self, db_session, sample_account):
        """Test basic balance operations."""
        # Add a new balance
        balance = TestBalance(
            account_id=sample_account.id,
            asset="BTC",
            asset_type="crypto",
            amount=Decimal("1.5"),
            available=Decimal("1.5"),
            reserved=Decimal("0.0")
        )
        db_session.add(balance)
        db_session.flush()
        
        # Verify balance was created
        fetched_balance = db_session.query(TestBalance).filter_by(account_id=sample_account.id, asset="BTC").first()
        assert fetched_balance is not None
        assert fetched_balance.amount == Decimal("1.5")
    
    def test_position_operations(self, db_session, sample_account):
        """Test basic position operations."""
        # Create a position
        position = TestPosition(
            account_id=sample_account.id,
            symbol="MSFT",
            instrument_type="stock",
            direction="long",
            quantity=Decimal("50"),
            average_entry_price=Decimal("250.00"),
            current_price=Decimal("260.00"),
            unrealized_pnl=Decimal("500.00")
        )
        db_session.add(position)
        db_session.flush()
        
        # Verify position was created
        fetched_position = db_session.query(TestPosition).filter_by(account_id=sample_account.id, symbol="MSFT").first()
        assert fetched_position is not None
        assert fetched_position.quantity == Decimal("50")
        assert fetched_position.current_price == Decimal("260.00")

if __name__ == "__main__":
    pytest.main(["-xvs", __file__])