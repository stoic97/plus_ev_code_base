"""
Unit tests for account.py models and functions.

This module provides comprehensive test coverage for the account-related models,
including Account, Balance, Position, Transaction, and RiskProfile,
as well as the utility functions for account operations.

Each test class focuses on a specific model or functionality, and follows
this basic structure:
1. Setup test data and DB session
2. Test model initialization with valid/invalid data
3. Test model methods and properties
4. Test relationships between models
5. Test database constraints and validations
6. Test complex business logic and edge cases
"""

import sys
import os
import unittest
from decimal import Decimal
from datetime import datetime, date, timedelta
from unittest import mock

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from pytest_postgresql import factories

# Add application root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import models and functions to test
from app.models.account import (
    Base, Account, Balance, Position, Transaction, RiskProfile, Tag,
    create_transaction, trade_execution, get_account_risk_summary
)
# Import database utilities
from app.core.database import DatabaseType


# PostgreSQL test fixtures

# Define PostgreSQL factory for testing
postgresql_my_proc = factories.postgresql_proc(
    port=None,  # Use dynamic port to avoid conflicts
)
postgresql_my = factories.postgresql('postgresql_my_proc')

@pytest.fixture(scope="function")
def db_session(postgresql_my):
    """Create a PostgreSQL database session for testing."""
    # Get connection details from pytest-postgresql
    db_url = (
        f"postgresql://{postgresql_my.info.user}:"
        f"{postgresql_my.info.password}@{postgresql_my.info.host}:"
        f"{postgresql_my.info.port}/{postgresql_my.info.dbname}"
    )
    
    # Create PostgreSQL engine
    engine = create_engine(db_url, echo=False)
    
    # Create all tables
    Base.metadata.create_all(engine)
    
    # Create session factory
    SessionFactory = sessionmaker(bind=engine)
    
    # Create session
    session = SessionFactory()
    
    # Setup any mock data or database state here
    
    yield session
    
    # Teardown - rollback and close session
    session.rollback()
    session.close()
    
    # Drop all tables to clean up
    Base.metadata.drop_all(engine)


@pytest.fixture(scope="function")
def sample_account(db_session):
    """Create a sample account for testing."""
    # Create a mock user first (in real app, this would be a User model)
    user_id = 1  # Mock user ID
    
    # Create account
    account = Account(
        user_id=user_id,
        account_number="TEST123456",
        account_type="main",
        name="Test Trading Account",
        base_currency="USD",
        status="active"
    )
    
    # Add risk profile
    risk_profile = RiskProfile(
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
    balance = Balance(
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
    position = Position(
        account_id=sample_account.id,
        symbol="AAPL",
        instrument_type="stock",
        direction="long",
        quantity=Decimal("100"),
        average_entry_price=Decimal("150.00"),
        current_price=Decimal("155.00"),
        unrealized_pnl=Decimal("500.00"),
        realized_pnl=Decimal("0.00"),
        strategy_id="test_strategy"
    )
    db_session.add(position)
    db_session.flush()
    return position

# Rest of the test classes remain the same
# Actual test classes

class TestAccount:
    """Tests for the Account model."""
    
    def test_account_creation(self, db_session):
        """Test creating a basic account."""
        account = Account(
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
        
    def test_account_status_validation(self, db_session):
        """Test that account status validation works."""
        account = Account(
            user_id=1,
            account_number="TEST002",
            account_type="main",
            name="Test Account",
            status="active"
        )
        db_session.add(account)
        db_session.flush()
        
        # Valid status change
        account.status = "suspended"
        db_session.flush()
        assert account.status == "suspended"
        
        # Invalid status
        with pytest.raises(ValueError):
            account.status = "invalid_status"
            
    def test_total_equity_calculation(self, db_session, sample_account, sample_position):
        """Test the total_equity hybrid property."""
        # Expected: cash balance + position market value
        expected_equity = Decimal("10000.00") + (Decimal("100") * Decimal("155.00"))
        assert sample_account.total_equity == expected_equity
        
        # Add another position
        position2 = Position(
            account_id=sample_account.id,
            symbol="MSFT",
            instrument_type="stock",
            direction="long",
            quantity=Decimal("50"),
            average_entry_price=Decimal("250.00"),
            current_price=Decimal("260.00"),
            unrealized_pnl=Decimal("500.00")
        )
        db_session.add(position2)
        db_session.flush()
        
        # Recalculate expected equity
        expected_equity = Decimal("10000.00") + (Decimal("100") * Decimal("155.00")) + (Decimal("50") * Decimal("260.00"))
        assert sample_account.total_equity == expected_equity
        
    def test_get_position_by_symbol(self, db_session, sample_account, sample_position):
        """Test the get_position_by_symbol method."""
        # Test finding existing position
        position = sample_account.get_position_by_symbol("AAPL")
        assert position is not None
        assert position.symbol == "AAPL"
        
        # Test position not found
        position = sample_account.get_position_by_symbol("NONEXISTENT")
        assert position is None
        
    def test_get_balance_by_asset(self, db_session, sample_account):
        """Test the get_balance_by_asset method."""
        # Test finding existing balance
        balance = sample_account.get_balance_by_asset("USD")
        assert balance is not None
        assert balance.asset == "USD"
        
        # Test balance not found
        balance = sample_account.get_balance_by_asset("NONEXISTENT")
        assert balance is None
        
    def test_daily_pnl_calculation(self, db_session, sample_account, sample_position):
        """Test the daily_pnl property calculation."""
        # Add some transactions for today
        today = date.today()
        
        # Create a trade transaction
        transaction = Transaction(
            account_id=sample_account.id,
            transaction_type="trade",
            asset="AAPL",
            amount=Decimal("300.00"),
            created_at=datetime.now()
        )
        db_session.add(transaction)
        
        # Create a fee transaction
        fee_transaction = Transaction(
            account_id=sample_account.id,
            transaction_type="fee",
            asset="USD",
            amount=Decimal("-10.00"),
            created_at=datetime.now()
        )
        db_session.add(transaction)
        db_session.flush()
        
        # Should still be at limit
        assert risk_profile.has_reached_daily_trade_limit(db_session) is True
        
    def test_check_position_size_limit(self, db_session, sample_account):
        """Test the check_position_size_limit method."""
        risk_profile = sample_account.risk_profile
        
        # Set position size limit to 10% of equity
        risk_profile.max_position_size_percentage = Decimal("10.0")
        
        # Sample account has $10,000 and a position worth $15,500
        # Total equity = $25,500
        # 10% of equity = $2,550
        
        # Test with position below limit
        assert risk_profile.check_position_size_limit(
            symbol="MSFT",
            quantity=Decimal("10"),
            price=Decimal("250.00"),
            instrument_type="stock",
            session=db_session
        ) is True
        
        # Test with position at limit
        assert risk_profile.check_position_size_limit(
            symbol="MSFT",
            quantity=Decimal("102"),  # ~$25,500 (total equity) * 0.1 / $250 = 10.2
            price=Decimal("250.00"),
            instrument_type="stock",
            session=db_session
        ) is True
        
        # Test with position exceeding limit
        assert risk_profile.check_position_size_limit(
            symbol="MSFT",
            quantity=Decimal("103"),  # Just over the limit
            price=Decimal("250.00"),
            instrument_type="stock",
            session=db_session
        ) is False
        
        # Test with absolute limit override
        risk_profile.max_position_size_absolute = Decimal("2000.00")
        
        # Now even a position at 8% should fail because of absolute limit
        assert risk_profile.check_position_size_limit(
            symbol="MSFT",
            quantity=Decimal("9"),  # $2,250 > $2,000 absolute limit
            price=Decimal("250.00"),
            instrument_type="stock",
            session=db_session
        ) is False


class TestUtilityFunctions:
    """Tests for the utility functions."""
    
    def test_create_transaction(self, db_session, sample_account):
        """Test the create_transaction utility function."""
        # Create a deposit transaction
        transaction = create_transaction(
            session=db_session,
            account_id=sample_account.id,
            transaction_type="deposit",
            asset="USD",
            amount=Decimal("5000.00"),
            reference_id="DEP123456",
            description="Test deposit"
        )
        db_session.flush()
        
        # Verify transaction created
        assert transaction.id is not None
        assert transaction.transaction_type == "deposit"
        assert transaction.amount == Decimal("5000.00")
        
        # Verify balance updated
        balance = sample_account.get_balance_by_asset("USD")
        assert balance is not None
        assert balance.amount == Decimal("15000.00")  # 10000 + 5000
        
        # Create a withdrawal transaction
        transaction = create_transaction(
            session=db_session,
            account_id=sample_account.id,
            transaction_type="withdrawal",
            asset="USD",
            amount=Decimal("-2000.00"),
            reference_id="WIT123456",
            description="Test withdrawal"
        )
        db_session.flush()
        
        # Verify balance updated
        balance = sample_account.get_balance_by_asset("USD")
        assert balance is not None
        assert balance.amount == Decimal("13000.00")  # 15000 - 2000
        
        # Create a transaction for a new asset
        transaction = create_transaction(
            session=db_session,
            account_id=sample_account.id,
            transaction_type="deposit",
            asset="BTC",
            amount=Decimal("1.5"),
            reference_id="DEP123457",
            description="Test crypto deposit"
        )
        db_session.flush()
        
        # Verify new balance created
        balance = sample_account.get_balance_by_asset("BTC")
        assert balance is not None
        assert balance.amount == Decimal("1.5")
        
    def test_trade_execution(self, db_session, sample_account):
        """Test the trade_execution utility function."""
        # Execute a buy trade for a new position
        transaction, position = trade_execution(
            session=db_session,
            account_id=sample_account.id,
            symbol="MSFT",
            instrument_type="stock",
            direction="buy",
            quantity=Decimal("10"),
            price=Decimal("250.00"),
            fee=Decimal("5.00"),
            strategy_id="test_strategy",
            stop_loss=Decimal("240.00"),
            take_profit=Decimal("275.00"),
            reference_id="ORD123456",
            counterparty="NASDAQ"
        )
        db_session.flush()
        
        # Verify transaction created
        assert transaction.id is not None
        assert transaction.transaction_type == "trade"
        assert transaction.symbol == "MSFT"
        assert transaction.direction == "buy"
        assert transaction.amount == Decimal("-2500.00")  # 10 * 250 = 2500 (negative for buy)
        
        # Verify position created
        assert position is not None
        assert position.symbol == "MSFT"
        assert position.direction == "long"
        assert position.quantity == Decimal("10")
        assert position.average_entry_price == Decimal("250.00")
        assert position.stop_loss == Decimal("240.00")
        assert position.take_profit == Decimal("275.00")
        
        # Execute another buy trade for the same position
        transaction, position = trade_execution(
            session=db_session,
            account_id=sample_account.id,
            symbol="MSFT",
            instrument_type="stock",
            direction="buy",
            quantity=Decimal("5"),
            price=Decimal("260.00"),
            fee=Decimal("3.00"),
            strategy_id="test_strategy",
            reference_id="ORD123457",
            counterparty="NASDAQ"
        )
        db_session.flush()
        
        # Verify position updated with new average price
        assert position.quantity == Decimal("15")
        expected_avg_price = (Decimal("10") * Decimal("250.00") + Decimal("5") * Decimal("260.00")) / Decimal("15")
        assert position.average_entry_price == expected_avg_price
        
        # Execute a sell trade for partial position closure
        transaction, position = trade_execution(
            session=db_session,
            account_id=sample_account.id,
            symbol="MSFT",
            instrument_type="stock",
            direction="sell",
            quantity=Decimal("7"),
            price=Decimal("265.00"),
            fee=Decimal("4.00"),
            strategy_id="test_strategy",
            reference_id="ORD123458",
            counterparty="NASDAQ"
        )
        db_session.flush()
        
        # Verify position updated
        assert position.quantity == Decimal("8")
        assert position.realized_pnl > Decimal("0")  # Should have profit from selling at higher price
        
        # Execute a sell trade for complete position closure
        transaction, position = trade_execution(
            session=db_session,
            account_id=sample_account.id,
            symbol="MSFT",
            instrument_type="stock",
            direction="sell",
            quantity=Decimal("8"),
            price=Decimal("270.00"),
            fee=Decimal("4.00"),
            strategy_id="test_strategy",
            reference_id="ORD123459",
            counterparty="NASDAQ"
        )
        db_session.flush()
        
        # Position should be None after complete closure
        assert position is None
        
        # Verify position no longer exists in database
        position = sample_account.get_position_by_symbol("MSFT")
        assert position is None
        
    def test_trade_execution_with_risk_checks(self, db_session, sample_account):
        """Test that trade_execution enforces risk limits."""
        # Set very small position size limit (1% of equity)
        sample_account.risk_profile.max_position_size_percentage = Decimal("1.0")
        db_session.flush()
        
        # Try to execute a trade exceeding the limit
        # Account equity is about $25,500, so 1% is $255
        with pytest.raises(ValueError, match="exceeds position size limits"):
            trade_execution(
                session=db_session,
                account_id=sample_account.id,
                symbol="MSFT",
                instrument_type="stock",
                direction="buy",
                quantity=Decimal("2"),  # 2 * 300 = $600, which exceeds 1% of equity
                price=Decimal("300.00"),
                strategy_id="test_strategy"
            )
        
        # Set account to cooldown mode
        sample_account.risk_profile.is_in_cooldown = True
        sample_account.risk_profile.cooldown_until = datetime.now() + timedelta(hours=1)
        db_session.flush()
        
        # Try to execute a trade during cooldown
        with pytest.raises(ValueError, match="Trading not allowed"):
            trade_execution(
                session=db_session,
                account_id=sample_account.id,
                symbol="MSFT",
                instrument_type="stock",
                direction="buy",
                quantity=Decimal("0.5"),  # Small enough to be within limits
                price=Decimal("300.00"),
                strategy_id="test_strategy"
            )
            
    def test_get_account_risk_summary(self, db_session, sample_account, sample_position):
        """Test the get_account_risk_summary function."""
        # Add a trade for today to test daily P&L
        transaction = Transaction(
            account_id=sample_account.id,
            transaction_type="trade",
            asset="USD",
            amount=Decimal("500.00"),
            created_at=datetime.now()
        )
        db_session.add(transaction)
        db_session.flush()
        
        # Get risk summary
        risk_summary = get_account_risk_summary(db_session, sample_account.id)
        
        # Verify summary fields
        assert risk_summary["account_id"] == sample_account.id
        assert risk_summary["account_name"] == sample_account.name
        assert abs(risk_summary["total_equity"] - 25500.0) < 0.01  # ~$10,000 + (100 * $155)
        assert risk_summary["positions_count"] == 1
        assert "stock" in risk_summary["asset_class_exposure"]
        assert risk_summary["risk_status"] == "active"
        
        # Set account close to daily loss warning
        sample_account.risk_profile.max_daily_loss_percentage = Decimal("3.0")
        
        # Add a losing trade that brings daily P&L to about -2.5% (close to warning threshold)
        transaction = Transaction(
            account_id=sample_account.id,
            transaction_type="trade",
            asset="USD",
            amount=Decimal("-650.00"),  # Net daily P&L: 500 - 650 = -150 (~-0.6% of equity)
            created_at=datetime.now()
        )
        db_session.add(transaction)
        db_session.flush()
        
        # Get updated risk summary
        risk_summary = get_account_risk_summary(db_session, sample_account.id)
        
        # Daily loss should be negative now, but not at warning level yet
        assert risk_summary["daily_pnl"] < 0
        assert risk_summary["daily_loss_warning"] is False
        
        # Add a bigger loss to trigger warning
        transaction = Transaction(
            account_id=sample_account.id,
            transaction_type="trade",
            asset="USD",
            amount=Decimal("-500.00"),  # Net daily P&L: -150 - 500 = -650 (over -2% of equity)
            created_at=datetime.now()
        )
        db_session.add(transaction)
        db_session.flush()
        
        # Get updated risk summary
        risk_summary = get_account_risk_summary(db_session, sample_account.id)
        
        # Daily loss warning should be triggered now
        assert risk_summary["daily_loss_warning"] is True
        
    def test_margin_account_risk_metrics(self, db_session):
        """Test risk metrics for margin accounts."""
        # Create a margin account
        margin_account = Account(
            user_id=1,
            account_number="MARGIN001",
            account_type="margin",
            name="Margin Trading Account",
            base_currency="USD",
            status="active",
            is_margin_enabled=True,
            margin_level=Decimal("40.00")  # 40% margin used
        )
        
        # Add risk profile
        risk_profile = RiskProfile(
            max_position_size_percentage=10.0,
            max_leverage=Decimal("2.0"),
            margin_call_level=Decimal("75.0"),
            liquidation_level=Decimal("90.0")
        )
        margin_account.risk_profile = risk_profile
        
        # Add to session
        db_session.add(margin_account)
        db_session.flush()
        
        # Add balance
        balance = Balance(
            account_id=margin_account.id,
            asset="USD",
            asset_type="currency",
            amount=Decimal("10000.00"),
            available=Decimal("6000.00"),
            reserved=Decimal("4000.00")
        )
        db_session.add(balance)
        db_session.flush()
        
        # Get risk summary
        risk_summary = get_account_risk_summary(db_session, margin_account.id)
        
        # Verify margin-specific metrics
        assert risk_summary["margin_call_warning"] is False  # 40% < 70% threshold
        
        # Update margin level to warning level
        margin_account.margin_level = Decimal("72.00")
        db_session.flush()
        
        # Get updated risk summary
        risk_summary = get_account_risk_summary(db_session, margin_account.id)
        
        # Margin call warning should be triggered now
        assert risk_summary["margin_call_warning"] is True


# Additional integration tests to ensure models work together correctly

class TestAccountIntegration:
    """Integration tests for account models working together."""
    
    def test_full_trading_lifecycle(self, db_session):
        """
        Test a complete trading lifecycle from account creation to trade execution
        with position management and risk enforcement.
        """
        # Create a new account
        account = Account(
            user_id=1,
            account_number="INTTEST001",
            account_type="main",
            name="Integration Test Account",
            base_currency="USD",
            status="active"
        )
        
        # Add risk profile
        risk_profile = RiskProfile(
            max_position_size_percentage=20.0,
            max_sector_exposure_percentage=40.0,
            max_asset_class_exposure_percentage=80.0,
            max_daily_loss_percentage=5.0,
            max_total_loss_percentage=15.0,
            max_leverage=1.0
        )
        account.risk_profile = risk_profile
        
        # Add to session
        db_session.add(account)
        db_session.flush()
        
        # Step 1: Initial funding
        create_transaction(
            session=db_session,
            account_id=account.id,
            transaction_type="deposit",
            asset="USD",
            amount=Decimal("50000.00"),
            reference_id="DEP-INT-001",
            description="Initial account funding"
        )
        db_session.flush()
        
        # Verify balance created
        balance = account.get_balance_by_asset("USD")
        assert balance is not None
        assert balance.amount == Decimal("50000.00")
        
        # Step 2: First trade - buy a position
        transaction1, position1 = trade_execution(
            session=db_session,
            account_id=account.id,
            symbol="AAPL",
            instrument_type="stock",
            direction="buy",
            quantity=Decimal("200"),
            price=Decimal("150.00"),
            fee=Decimal("15.00"),
            strategy_id="value_strategy",
            stop_loss=Decimal("140.00"),
            take_profit=Decimal("180.00"),
            reference_id="ORD-INT-001"
        )
        db_session.flush()
        
        # Verify position created
        assert position1 is not None
        assert position1.symbol == "AAPL"
        assert position1.quantity == Decimal("200")
        
        # Verify transaction created
        assert transaction1 is not None
        assert transaction1.amount == Decimal("-30000.00")  # 200 * 150 = 30000 (negative for buy)
        
        # Step 3: Second trade - different position
        transaction2, position2 = trade_execution(
            session=db_session,
            account_id=account.id,
            symbol="MSFT",
            instrument_type="stock",
            direction="buy",
            quantity=Decimal("100"),
            price=Decimal("250.00"),
            fee=Decimal("12.50"),
            strategy_id="growth_strategy",
            stop_loss=Decimal("235.00"),
            take_profit=Decimal("275.00"),
            reference_id="ORD-INT-002"
        )
        db_session.flush()
        
        # Verify second position
        assert position2 is not None
        assert position2.symbol == "MSFT"
        
        # Step 4: Add to first position
        transaction3, position3 = trade_execution(
            session=db_session,
            account_id=account.id,
            symbol="AAPL",
            instrument_type="stock",
            direction="buy",
            quantity=Decimal("50"),
            price=Decimal("155.00"),
            fee=Decimal("7.75"),
            strategy_id="value_strategy",
            reference_id="ORD-INT-003"
        )
        db_session.flush()
        
        # Verify position updated
        assert position3 is not None
        assert position3.symbol == "AAPL"
        assert position3.quantity == Decimal("250")  # 200 + 50
        
        # Expected average price: (200 * 150 + 50 * 155) / 250 = 151
        expected_avg_price = (Decimal("200") * Decimal("150.00") + Decimal("50") * Decimal("155.00")) / Decimal("250")
        assert position3.average_entry_price == expected_avg_price
        
        # Step 5: Update position prices
        for pos in [position1, position2]:
            if pos.symbol == "AAPL":
                pos.update_price(Decimal("160.00"))  # Price increase
            elif pos.symbol == "MSFT":
                pos.update_price(Decimal("245.00"))  # Price decrease
        db_session.flush()
        
        # Get latest risk summary
        risk_summary = get_account_risk_summary(db_session, account.id)
        
        # Verify equity calculation
        # Cash: 50000 - 30000 - 25000 - 7750 - 15 - 12.50 - 7.75 = 17215.00
        # Positions: (250 * 160) + (100 * 245) = 40000 + 24500 = 64500
        # Total equity: 17215 + 64500 = 81715
        expected_equity = 50000 - 30000 - 25000 - 35
        expected_equity += 250 * 160
        expected_equity += 100 * 245
        
        assert abs(risk_summary["total_equity"] - expected_equity) < 100  # Allow for minor calculation differences
        
        # Verify position counts
        assert risk_summary["positions_count"] == 2
        
        # Step a position partially
        transaction4, position4 = trade_execution(
            session=db_session,
            account_id=account.id,
            symbol="AAPL",
            instrument_type="stock",
            direction="sell",
            quantity=Decimal("150"),
            price=Decimal("162.00"),
            fee=Decimal("9.25"),
            strategy_id="value_strategy",
            reference_id="ORD-INT-004"
        )
        db_session.flush()
        
        # Verify partial closure
        assert position4 is not None
        assert position4.quantity == Decimal("100")  # 250 - 150
        assert position4.realized_pnl > 0  # Should have profit from price increase
        
        # Step 7: Close all positions
        transaction5, position5 = trade_execution(
            session=db_session,
            account_id=account.id,
            symbol="AAPL",
            instrument_type="stock",
            direction="sell",
            quantity=Decimal("100"),
            price=Decimal("163.00"),
            fee=Decimal("8.15"),
            strategy_id="value_strategy",
            reference_id="ORD-INT-005"
        )
        db_session.flush()
        
        transaction6, position6 = trade_execution(
            session=db_session,
            account_id=account.id,
            symbol="MSFT",
            instrument_type="stock",
            direction="sell",
            quantity=Decimal("100"),
            price=Decimal("248.00"),
            fee=Decimal("12.40"),
            strategy_id="growth_strategy",
            reference_id="ORD-INT-006"
        )
        db_session.flush()
        
        # Verify all positions closed
        positions = db_session.query(Position).filter(Position.account_id == account.id).all()
        assert len(positions) == 0
        
        # Get final account state
        final_risk_summary = get_account_risk_summary(db_session, account.id)
        
        # Verify no positions
        assert final_risk_summary["positions_count"] == 0
        
        # Equity should now be just cash, which should be initial deposit plus gains/losses
        # This verifies that all P&L was properly accounted for
        assert final_risk_summary["cash_balance"] > 0


    if __name__ == "__main__":
        pytest.main(["-xvs", __file__])
        created_at=datetime.now()
        
        db_session.add(fee_transaction)
        db_session.flush()
        
        # Expected: realized P&L (300 - 10) + unrealized P&L (500)
        expected_daily_pnl = Decimal("300.00") + Decimal("-10.00") + Decimal("500.00")
        assert sample_account.daily_pnl == expected_daily_pnl


class TestBalance:
    """Tests for the Balance model."""
    
    def test_balance_creation(self, db_session, sample_account):
        """Test creating a balance record."""
        balance = Balance(
            account_id=sample_account.id,
            asset="EUR",
            asset_type="currency",
            amount=Decimal("5000.00"),
            available=Decimal("5000.00"),
            reserved=Decimal("0.00")
        )
        db_session.add(balance)
        db_session.flush()
        
        assert balance.id is not None
        assert balance.asset == "EUR"
        assert balance.amount == Decimal("5000.00")
        
    def test_amount_validation(self, db_session, sample_account):
        """Test amount validation and auto-calculation."""
        balance = Balance(
            account_id=sample_account.id,
            asset="BTC",
            asset_type="crypto",
            available=Decimal("1.5"),
            reserved=Decimal("0.5")
        )
        db_session.add(balance)
        db_session.flush()
        
        # Check that amount is calculated correctly
        assert balance.amount == Decimal("2.0")
        
        # Update available and check amount recalculation
        balance.available = Decimal("2.0")
        db_session.flush()
        assert balance.amount == Decimal("2.5")
        
        # Update reserved and check amount recalculation
        balance.reserved = Decimal("1.0")
        db_session.flush()
        assert balance.amount == Decimal("3.0")
        
    def test_negative_amount_validation(self, db_session, sample_account):
        """Test that negative amounts are rejected."""
        balance = Balance(
            account_id=sample_account.id,
            asset="BTC",
            asset_type="crypto",
            amount=Decimal("1.0"),
            available=Decimal("1.0"),
            reserved=Decimal("0.0")
        )
        db_session.add(balance)
        db_session.flush()
        
        # Attempt to set negative available
        with pytest.raises(ValueError):
            balance.available = Decimal("-0.5")
            
    def test_locked_percentage(self, db_session, sample_account):
        """Test the locked_percentage calculation."""
        balance = Balance(
            account_id=sample_account.id,
            asset="ETH",
            asset_type="crypto",
            amount=Decimal("10.0"),
            available=Decimal("7.5"),
            reserved=Decimal("2.5")
        )
        db_session.add(balance)
        db_session.flush()
        
        # Check locked percentage: (2.5 / 10.0) * 100 = 25%
        assert balance.locked_percentage == 25.0
        
        # Test with zero amount
        balance.available = Decimal("0.0")
        balance.reserved = Decimal("0.0")
        db_session.flush()
        assert balance.locked_percentage == 0.0
        
    def test_reserve_and_release(self, db_session, sample_account):
        """Test the reserve and release methods."""
        balance = Balance(
            account_id=sample_account.id,
            asset="XRP",
            asset_type="crypto",
            amount=Decimal("1000.0"),
            available=Decimal("1000.0"),
            reserved=Decimal("0.0")
        )
        db_session.add(balance)
        db_session.flush()
        
        # Test reserve
        success = balance.reserve(Decimal("300.0"))
        assert success is True
        assert balance.available == Decimal("700.0")
        assert balance.reserved == Decimal("300.0")
        assert balance.amount == Decimal("1000.0")
        
        # Test reserve with insufficient funds
        success = balance.reserve(Decimal("800.0"))
        assert success is False
        assert balance.available == Decimal("700.0")  # Unchanged
        
        # Test release
        success = balance.release(Decimal("100.0"))
        assert success is True
        assert balance.available == Decimal("800.0")
        assert balance.reserved == Decimal("200.0")
        assert balance.amount == Decimal("1000.0")
        
        # Test release with insufficient reserved
        success = balance.release(Decimal("300.0"))
        assert success is False
        assert balance.reserved == Decimal("200.0")  # Unchanged
        
    def test_add_method(self, db_session, sample_account):
        """Test the add method for increasing/decreasing available balance."""
        balance = Balance(
            account_id=sample_account.id,
            asset="LTC",
            asset_type="crypto",
            amount=Decimal("50.0"),
            available=Decimal("50.0"),
            reserved=Decimal("0.0")
        )
        db_session.add(balance)
        db_session.flush()
        
        # Test adding positive amount
        balance.add(Decimal("25.0"))
        assert balance.available == Decimal("75.0")
        assert balance.amount == Decimal("75.0")
        
        # Test adding negative amount (subtraction)
        balance.add(Decimal("-15.0"))
        assert balance.available == Decimal("60.0")
        assert balance.amount == Decimal("60.0")
        
        # Test adding negative amount that would make balance negative
        with pytest.raises(ValueError):
            balance.add(Decimal("-100.0"))


class TestPosition:
    """Tests for the Position model."""
    
    def test_position_creation(self, db_session, sample_account):
        """Test creating a position."""
        position = Position(
            account_id=sample_account.id,
            symbol="TSLA",
            instrument_type="stock",
            direction="long",
            quantity=Decimal("10"),
            average_entry_price=Decimal("800.00"),
            current_price=Decimal("820.00")
        )
        db_session.add(position)
        db_session.flush()
        
        assert position.id is not None
        assert position.symbol == "TSLA"
        assert position.direction == "long"
        
    def test_position_direction_validation(self, db_session, sample_account):
        """Test direction validation."""
        position = Position(
            account_id=sample_account.id,
            symbol="TSLA",
            instrument_type="stock",
            direction="long",
            quantity=Decimal("10"),
            average_entry_price=Decimal("800.00"),
            current_price=Decimal("820.00")
        )
        db_session.add(position)
        db_session.flush()
        
        # Valid direction change
        position.direction = "short"
        db_session.flush()
        assert position.direction == "short"
        
        # Invalid direction
        with pytest.raises(ValueError):
            position.direction = "invalid"
            
    def test_market_value_calculation(self, db_session, sample_account):
        """Test market value calculation for long and short positions."""
        # Long position
        long_position = Position(
            account_id=sample_account.id,
            symbol="AMZN",
            instrument_type="stock",
            direction="long",
            quantity=Decimal("5"),
            average_entry_price=Decimal("3000.00"),
            current_price=Decimal("3100.00")
        )
        db_session.add(long_position)
        
        # Short position
        short_position = Position(
            account_id=sample_account.id,
            symbol="GME",
            instrument_type="stock",
            direction="short",
            quantity=Decimal("20"),
            average_entry_price=Decimal("200.00"),
            current_price=Decimal("180.00")
        )
        db_session.add(short_position)
        db_session.flush()
        
        # Test long position market value: quantity * price (positive)
        assert long_position.market_value == Decimal("5") * Decimal("3100.00")
        
        # Test short position market value: quantity * price (negative)
        assert short_position.market_value == -(Decimal("20") * Decimal("180.00"))
        
    def test_pnl_percentage_calculation(self, db_session, sample_account):
        """Test P&L percentage calculation for long and short positions."""
        # Long position with profit
        long_profit = Position(
            account_id=sample_account.id,
            symbol="NFLX",
            instrument_type="stock",
            direction="long",
            quantity=Decimal("10"),
            average_entry_price=Decimal("500.00"),
            current_price=Decimal("550.00")
        )
        db_session.add(long_profit)
        
        # Long position with loss
        long_loss = Position(
            account_id=sample_account.id,
            symbol="FB",
            instrument_type="stock",
            direction="long",
            quantity=Decimal("15"),
            average_entry_price=Decimal("300.00"),
            current_price=Decimal("270.00")
        )
        db_session.add(long_loss)
        
        # Short position with profit
        short_profit = Position(
            account_id=sample_account.id,
            symbol="DASH",
            instrument_type="stock",
            direction="short",
            quantity=Decimal("25"),
            average_entry_price=Decimal("200.00"),
            current_price=Decimal("180.00")
        )
        db_session.add(short_profit)
        
        # Short position with loss
        short_loss = Position(
            account_id=sample_account.id,
            symbol="COIN",
            instrument_type="stock",
            direction="short",
            quantity=Decimal("5"),
            average_entry_price=Decimal("250.00"),
            current_price=Decimal("300.00")
        )
        db_session.add(short_loss)
        db_session.flush()
        
        # Test long position with profit: (550 - 500) / 500 * 100 = 10%
        assert long_profit.pnl_percentage == 10.0
        
        # Test long position with loss: (270 - 300) / 300 * 100 = -10%
        assert long_loss.pnl_percentage == -10.0
        
        # Test short position with profit: (200 - 180) / 200 * 100 = 10%
        assert short_profit.pnl_percentage == 10.0
        
        # Test short position with loss: (250 - 300) / 250 * 100 = -20%
        assert short_loss.pnl_percentage == -20.0
        
    def test_stop_loss_and_take_profit_checks(self, db_session, sample_account):
        """Test the stop loss and take profit checking methods."""
        # Long position
        long_position = Position(
            account_id=sample_account.id,
            symbol="AAPL",
            instrument_type="stock",
            direction="long",
            quantity=Decimal("100"),
            average_entry_price=Decimal("150.00"),
            current_price=Decimal("155.00"),
            stop_loss=Decimal("145.00"),
            take_profit=Decimal("165.00")
        )
        db_session.add(long_position)
        
        # Short position
        short_position = Position(
            account_id=sample_account.id,
            symbol="TSLA",
            instrument_type="stock",
            direction="short",
            quantity=Decimal("10"),
            average_entry_price=Decimal("800.00"),
            current_price=Decimal("780.00"),
            stop_loss=Decimal("820.00"),
            take_profit=Decimal("750.00")
        )
        db_session.add(short_position)
        db_session.flush()
        
        # Test long position stop loss
        assert long_position.would_hit_stop_loss(Decimal("144.99")) is True
        assert long_position.would_hit_stop_loss(Decimal("145.00")) is True
        assert long_position.would_hit_stop_loss(Decimal("145.01")) is False
        
        # Test long position take profit
        assert long_position.would_hit_take_profit(Decimal("164.99")) is False
        assert long_position.would_hit_take_profit(Decimal("165.00")) is True
        assert long_position.would_hit_take_profit(Decimal("165.01")) is True
        
        # Test short position stop loss
        assert short_position.would_hit_stop_loss(Decimal("819.99")) is False
        assert short_position.would_hit_stop_loss(Decimal("820.00")) is True
        assert short_position.would_hit_stop_loss(Decimal("820.01")) is True
        
        # Test short position take profit
        assert short_position.would_hit_take_profit(Decimal("750.01")) is False
        assert short_position.would_hit_take_profit(Decimal("750.00")) is True
        assert short_position.would_hit_take_profit(Decimal("749.99")) is True
        
    def test_update_price(self, db_session, sample_account):
        """Test the update_price method with P&L recalculation."""
        # Long position
        position = Position(
            account_id=sample_account.id,
            symbol="GOOGL",
            instrument_type="stock",
            direction="long",
            quantity=Decimal("5"),
            average_entry_price=Decimal("2000.00"),
            current_price=Decimal("2000.00"),
            unrealized_pnl=Decimal("0.00")
        )
        db_session.add(position)
        db_session.flush()
        
        # Update price with profit
        position.update_price(Decimal("2100.00"))
        assert position.current_price == Decimal("2100.00")
        assert position.unrealized_pnl == Decimal("500.00")  # 5 * (2100 - 2000) = 500
        
        # Update price with loss
        position.update_price(Decimal("1900.00"))
        assert position.current_price == Decimal("1900.00")
        assert position.unrealized_pnl == Decimal("-500.00")  # 5 * (1900 - 2000) = -500
        
    def test_add_quantity(self, db_session, sample_account):
        """Test adding to a position with average price recalculation."""
        position = Position(
            account_id=sample_account.id,
            symbol="NVDA",
            instrument_type="stock",
            direction="long",
            quantity=Decimal("10"),
            average_entry_price=Decimal("500.00"),
            current_price=Decimal("520.00")
        )
        db_session.add(position)
        db_session.flush()
        
        # Add to position at a different price
        position.add_quantity(Decimal("5"), Decimal("550.00"))
        
        # Check new quantity
        assert position.quantity == Decimal("15")
        
        # Check new average entry price: (10*500 + 5*550) / 15 = 516.67
        expected_avg_price = (Decimal("10") * Decimal("500.00") + Decimal("5") * Decimal("550.00")) / Decimal("15")
        assert position.average_entry_price == expected_avg_price
        
    def test_reduce_quantity(self, db_session, sample_account):
        """Test reducing a position and calculating realized P&L."""
        position = Position(
            account_id=sample_account.id,
            symbol="AMZN",
            instrument_type="stock",
            direction="long",
            quantity=Decimal("10"),
            average_entry_price=Decimal("3000.00"),
            current_price=Decimal("3200.00"),
            unrealized_pnl=Decimal("2000.00"),
            realized_pnl=Decimal("0.00")
        )
        db_session.add(position)
        db_session.flush()
        
        # Reduce position by 4 shares at current price
        realized_pnl = position.reduce_quantity(Decimal("4"), Decimal("3200.00"))
        
        # Check new quantity
        assert position.quantity == Decimal("6")
        
        # Check realized P&L: 4 * (3200 - 3000) = 800
        assert realized_pnl == Decimal("800.00")
        assert position.realized_pnl == Decimal("800.00")
        
        # Check unrealized P&L updated: 6 * (3200 - 3000) = 1200
        assert position.unrealized_pnl == Decimal("1200.00")
        
        # Try to reduce by more than remaining quantity
        with pytest.raises(ValueError):
            position.reduce_quantity(Decimal("7"), Decimal("3200.00"))


class TestTransaction:
    """Tests for the Transaction model."""
    
    def test_transaction_creation(self, db_session, sample_account):
        """Test creating various types of transactions."""
        # Deposit transaction
        deposit = Transaction(
            account_id=sample_account.id,
            transaction_type="deposit",
            asset="USD",
            amount=Decimal("5000.00"),
            status="completed",
            reference_id="DEP12345"
        )
        db_session.add(deposit)
        
        # Trade transaction
        trade = Transaction(
            account_id=sample_account.id,
            transaction_type="trade",
            asset="AAPL",
            amount=Decimal("-15000.00"),
            price=Decimal("150.00"),
            quantity=Decimal("100"),
            symbol="AAPL",
            direction="buy",
            fee=Decimal("7.50"),
            status="completed",
            reference_id="TRD12345"
        )
        db_session.add(trade)
        
        # Fee transaction
        fee = Transaction(
            account_id=sample_account.id,
            transaction_type="fee",
            asset="USD",
            amount=Decimal("-9.99"),
            status="completed",
            reference_id="FEE12345"
        )
        db_session.add(fee)
        db_session.flush()
        
        assert deposit.id is not None
        assert trade.id is not None
        assert fee.id is not None
        
    def test_transaction_type_validation(self, db_session, sample_account):
        """Test transaction type validation."""
        transaction = Transaction(
            account_id=sample_account.id,
            transaction_type="deposit",
            asset="USD",
            amount=Decimal("1000.00"),
            status="completed"
        )
        db_session.add(transaction)
        db_session.flush()
        
        # Valid type change
        transaction.transaction_type = "withdrawal"
        db_session.flush()
        assert transaction.transaction_type == "withdrawal"
        
        # Invalid type
        with pytest.raises(ValueError):
            transaction.transaction_type = "invalid_type"
            
    def test_transaction_status_validation(self, db_session, sample_account):
        """Test transaction status validation."""
        transaction = Transaction(
            account_id=sample_account.id,
            transaction_type="deposit",
            asset="USD",
            amount=Decimal("1000.00"),
            status="pending"
        )
        db_session.add(transaction)
        db_session.flush()
        
        # Valid status change
        transaction.status = "completed"
        db_session.flush()
        assert transaction.status == "completed"
        
        # Another valid status change
        transaction.status = "failed"
        db_session.flush()
        assert transaction.status == "failed"
        
        # Invalid status
        with pytest.raises(ValueError):
            transaction.status = "invalid_status"


class TestRiskProfile:
    """Tests for the RiskProfile model."""
    
    def test_risk_profile_creation(self, db_session, sample_account):
        """Test creating a risk profile."""
        # Risk profile was already created by the sample_account fixture
        risk_profile = sample_account.risk_profile
        
        assert risk_profile is not None
        assert risk_profile.max_position_size_percentage == Decimal("5.0")
        assert risk_profile.max_daily_loss_percentage == Decimal("3.0")
        
    def test_is_trade_allowed(self, db_session, sample_account):
        """Test the is_trade_allowed method."""
        risk_profile = sample_account.risk_profile
        
        # By default, trading should be allowed
        assert risk_profile.is_trade_allowed() is True
        
        # Test with risk management disabled
        risk_profile.is_active = False
        assert risk_profile.is_trade_allowed() is False
        
        # Test with cooldown
        risk_profile.is_active = True
        risk_profile.is_in_cooldown = True
        risk_profile.cooldown_until = datetime.now() + timedelta(hours=1)
        assert risk_profile.is_trade_allowed() is False
        
        # Test with expired cooldown
        risk_profile.cooldown_until = datetime.now() - timedelta(hours=1)
        assert risk_profile.is_trade_allowed() is True
        
    def test_has_reached_daily_trade_limit(self, db_session, sample_account):
        """Test the has_reached_daily_trade_limit method."""
        risk_profile = sample_account.risk_profile
        risk_profile.max_trades_per_day = 3
        
        # Initially should not have reached limit
        assert risk_profile.has_reached_daily_trade_limit(db_session) is False
        
        # Add 2 trades for today
        for i in range(2):
            transaction = Transaction(
                account_id=sample_account.id,
                transaction_type="trade",
                asset="AAPL",
                amount=Decimal("1000.00"),
                created_at=datetime.now()
            )
            db_session.add(transaction)
        db_session.flush()
        
        # Still under limit
        assert risk_profile.has_reached_daily_trade_limit(db_session) is False
        
        # Add 1 more trade for today
        transaction = Transaction(
            account_id=sample_account.id,
            transaction_type="trade",
            asset="MSFT",
            amount=Decimal("2000.00"),
            created_at=datetime.now()
        )
        db_session.add(transaction)
        db_session.flush()
        
        # Now should have reached limit
        assert risk_profile.has_reached_daily_trade_limit(db_session) is True
        
        # Add a non-trade transaction - shouldn't affect limit
        transaction = Transaction(
            account_id=sample_account.id,
            transaction_type="fee",
            asset="USD",
            amount=Decimal("-10.00"),
            created_at=datetime.now()
        )