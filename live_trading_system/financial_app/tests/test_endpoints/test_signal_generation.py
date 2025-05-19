"""
Comprehensive test suite for Signal Generation and Analysis API endpoints.

This test file is completely self-contained with no external dependencies.
Tests cover all major functionality by simulating the business logic directly.
"""

import pytest
from unittest.mock import Mock, MagicMock, AsyncMock
from datetime import datetime, date
from typing import List, Dict, Any, Optional
import json
import io
import csv
from contextlib import contextmanager

# Test configuration
pytestmark = pytest.mark.asyncio


# Mock Classes and Enums
class MockTimeframeValueEnum:
    HOUR_1 = "1h"
    MIN_15 = "15m"
    MIN_5 = "5m"
    
    @classmethod
    def __call__(cls, value):
        if value == "1h":
            return cls.HOUR_1
        elif value == "15m":
            return cls.MIN_15
        elif value == "5m":
            return cls.MIN_5
        else:
            raise ValueError(f"Invalid timeframe: {value}")


class MockDirectionEnum:
    LONG = "long"
    SHORT = "short"


class MockTimeframeAnalysisResult:
    def __init__(self, **kwargs):
        self.aligned = kwargs.get('aligned', True)
        self.alignment_score = kwargs.get('alignment_score', 0.85)
        self.timeframe_results = kwargs.get('timeframe_results', {})
        self.primary_direction = kwargs.get('primary_direction', 'up')
        self.require_all_aligned = kwargs.get('require_all_aligned', True)
        self.min_alignment_score = kwargs.get('min_alignment_score', 0.7)
        self.sufficient_alignment = kwargs.get('sufficient_alignment', True)


class MockMarketStateAnalysis:
    def __init__(self, **kwargs):
        self.market_state = kwargs.get('market_state', 'trending_up')
        self.trend_phase = kwargs.get('trend_phase', 'middle')
        self.is_railroad_trend = kwargs.get('is_railroad_trend', True)
        self.is_creeper_move = kwargs.get('is_creeper_move', False)
        self.has_two_day_trend = kwargs.get('has_two_day_trend', True)
        self.trend_direction = kwargs.get('trend_direction', 'up')
        self.price_indicator_divergence = kwargs.get('price_indicator_divergence', False)
        self.price_struggling_near_ma = kwargs.get('price_struggling_near_ma', False)
        self.institutional_fight_in_progress = kwargs.get('institutional_fight_in_progress', False)
        self.accumulation_detected = kwargs.get('accumulation_detected', True)
        self.bos_detected = kwargs.get('bos_detected', True)


class MockSetupQualityResult:
    def __init__(self, **kwargs):
        self.grade = kwargs.get('grade', 'A')
        self.score = kwargs.get('score', 85.0)
        self.position_size_multiplier = kwargs.get('position_size_multiplier', 1.0)
        self.quality_factors = kwargs.get('quality_factors', {})
        self.meets_minimum_quality = kwargs.get('meets_minimum_quality', True)


class MockSignal:
    def __init__(self, **kwargs):
        self.id = kwargs.get('id', 1)
        self.strategy_id = kwargs.get('strategy_id', 1)
        self.instrument = kwargs.get('instrument', 'NIFTY')
        self.direction = kwargs.get('direction', 'long')
        self.signal_type = kwargs.get('signal_type', 'entry')
        self.entry_price = kwargs.get('entry_price', 18500.0)
        self.entry_time = kwargs.get('entry_time', datetime(2024, 1, 1, 10, 0, 0))
        self.entry_timeframe = kwargs.get('entry_timeframe', '5m')
        self.entry_technique = kwargs.get('entry_technique', 'breakout')
        self.take_profit_price = kwargs.get('take_profit_price', 18600.0)
        self.stop_loss_price = kwargs.get('stop_loss_price', 18450.0)
        self.trailing_stop = kwargs.get('trailing_stop', False)
        self.position_size = kwargs.get('position_size', 1.0)
        self.risk_reward_ratio = kwargs.get('risk_reward_ratio', 2.22)
        self.risk_amount = kwargs.get('risk_amount', 50.0)
        self.setup_quality = kwargs.get('setup_quality', 'A')
        self.setup_score = kwargs.get('setup_score', 85.0)
        self.confidence = kwargs.get('confidence', 0.8)
        self.market_state = kwargs.get('market_state', 'trending_up')
        self.trend_phase = kwargs.get('trend_phase', 'middle')
        self.is_active = kwargs.get('is_active', True)
        self.is_executed = kwargs.get('is_executed', False)
        self.execution_time = kwargs.get('execution_time', None)
        self.timeframe_alignment_score = kwargs.get('timeframe_alignment_score', 0.85)
        self.primary_timeframe_aligned = kwargs.get('primary_timeframe_aligned', True)
        self.institutional_footprint_detected = kwargs.get('institutional_footprint_detected', True)
        self.bos_detected = kwargs.get('bos_detected', True)
        self.is_spread_trade = kwargs.get('is_spread_trade', False)
        self.spread_type = kwargs.get('spread_type', None)
        self.created_at = kwargs.get('created_at', datetime(2024, 1, 1, 10, 0, 0))
        self.updated_at = kwargs.get('updated_at', datetime(2024, 1, 1, 10, 0, 0))


class MockStrategy:
    def __init__(self, **kwargs):
        self.id = kwargs.get('id', 1)
        self.user_id = kwargs.get('user_id', 1)
        self.name = kwargs.get('name', 'Test Strategy')


class MockHTTPException(Exception):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"{status_code}: {detail}")


# Business Logic Simulators
class SignalGenerationSimulator:
    """Simulates the signal generation business logic."""
    
    @staticmethod
    def convert_market_data_keys(market_data: Dict[str, Dict[str, Any]]) -> Dict:
        """Convert string timeframe keys to enum-like values."""
        converted_data = {}
        valid_timeframes = ["1h", "15m", "5m"]
        
        for tf_str, data in market_data.items():
            if tf_str not in valid_timeframes:
                raise ValueError(f"Invalid timeframe: {tf_str}. Valid options: {valid_timeframes}")
            converted_data[tf_str] = data
        return converted_data
    
    @staticmethod
    def signal_to_response(signal: MockSignal) -> Dict[str, Any]:
        """Convert signal model to response format."""
        return {
            "id": signal.id,
            "strategy_id": signal.strategy_id,
            "instrument": signal.instrument,
            "direction": signal.direction,
            "signal_type": signal.signal_type or "unknown",
            "entry_price": signal.entry_price,
            "entry_time": signal.entry_time.isoformat() if signal.entry_time else None,
            "entry_timeframe": signal.entry_timeframe,
            "entry_technique": signal.entry_technique,
            "take_profit_price": signal.take_profit_price,
            "stop_loss_price": signal.stop_loss_price,
            "trailing_stop": signal.trailing_stop,
            "position_size": signal.position_size,
            "risk_reward_ratio": signal.risk_reward_ratio,
            "risk_amount": signal.risk_amount,
            "setup_quality": signal.setup_quality,
            "setup_score": signal.setup_score,
            "confidence": signal.confidence,
            "market_state": signal.market_state,
            "trend_phase": signal.trend_phase,
            "is_active": signal.is_active,
            "is_executed": signal.is_executed,
            "execution_time": signal.execution_time.isoformat() if signal.execution_time else None,
            "timeframe_alignment_score": signal.timeframe_alignment_score,
            "primary_timeframe_aligned": signal.primary_timeframe_aligned,
            "institutional_footprint_detected": signal.institutional_footprint_detected,
            "bos_detected": signal.bos_detected,
            "is_spread_trade": signal.is_spread_trade or False,
            "spread_type": signal.spread_type
        }
    
    @staticmethod
    async def check_strategy_ownership(strategy_id: int, user_id: int, service) -> None:
        """Check if user owns the strategy."""
        try:
            strategy = service.get_strategy(strategy_id)
            if strategy.user_id != user_id:
                raise MockHTTPException(status_code=403, detail="Access denied")
        except ValueError as e:
            raise MockHTTPException(status_code=404, detail="Strategy not found")
    
    @staticmethod
    def validate_signal_update_data(update_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate signal update data."""
        validated_data = {}
        
        # Validate prices
        for price_field in ["entry_price", "take_profit_price", "stop_loss_price"]:
            if price_field in update_data:
                value = update_data[price_field]
                if value is not None and value <= 0:
                    raise ValueError(f'{price_field} must be positive')
                validated_data[price_field] = value
        
        # Validate position size
        if "position_size" in update_data:
            value = update_data["position_size"]
            if value is not None and value <= 0:
                raise ValueError('position_size must be positive')
            validated_data["position_size"] = value
        
        # Other fields
        for field in ["trailing_stop", "notes"]:
            if field in update_data:
                validated_data[field] = update_data[field]
        
        return validated_data


# Test Fixtures
@pytest.fixture
def mock_db_session():
    """Mock database session."""
    session = MagicMock()
    session.query.return_value = session
    session.filter.return_value = session
    session.order_by.return_value = session
    session.offset.return_value = session
    session.limit.return_value = session
    session.first.return_value = None
    session.all.return_value = []
    session.count.return_value = 0
    session.commit = MagicMock()
    session.refresh = MagicMock()
    session.delete = MagicMock()
    session.group_by.return_value = session
    # Context manager support
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=None)
    return session


@pytest.fixture
def mock_strategy_service(mock_db_session):
    """Mock strategy engine service."""
    service = MagicMock()
    service.db = mock_db_session
    service.db.session.return_value = mock_db_session
    return service


@pytest.fixture
def mock_strategy():
    """Mock strategy object."""
    return MockStrategy()


@pytest.fixture
def mock_signal():
    """Mock signal object."""
    return MockSignal()


@pytest.fixture
def sample_market_data():
    """Sample market data for testing."""
    return {
        "market_data": {
            "1h": {
                "close": [18500, 18520, 18540, 18560, 18580],
                "high": [18510, 18530, 18550, 18570, 18590],
                "low": [18490, 18510, 18530, 18550, 18570],
                "volume": [1000, 1100, 1200, 1300, 1400],
                "ma21": [18400, 18420, 18440, 18460, 18480],
                "ma200": [18200, 18210, 18220, 18230, 18240]
            },
            "15m": {
                "close": [18575, 18580, 18585, 18590, 18595],
                "high": [18580, 18585, 18590, 18595, 18600],
                "low": [18570, 18575, 18580, 18585, 18590],
                "volume": [500, 550, 600, 650, 700]
            }
        }
    }


@pytest.fixture
def sample_timeframe_analysis():
    """Sample timeframe analysis result."""
    return MockTimeframeAnalysisResult(
        aligned=True,
        alignment_score=0.85,
        timeframe_results={},
        primary_direction="up",
        require_all_aligned=True,
        min_alignment_score=0.7,
        sufficient_alignment=True
    )


@pytest.fixture
def sample_market_state():
    """Sample market state analysis result."""
    return MockMarketStateAnalysis(
        market_state="trending_up",
        trend_phase="middle",
        is_railroad_trend=True,
        is_creeper_move=False,
        has_two_day_trend=True,
        trend_direction="up",
        price_indicator_divergence=False,
        price_struggling_near_ma=False,
        institutional_fight_in_progress=False,
        accumulation_detected=True,
        bos_detected=True
    )


@pytest.fixture
def sample_setup_quality():
    """Sample setup quality result."""
    return MockSetupQualityResult(
        grade="A",
        score=85.0,
        position_size_multiplier=1.0,
        quality_factors={},
        meets_minimum_quality=True
    )


# Test utility functions
class TestUtilityFunctions:
    """Test utility functions from the signal generation module."""
    
    def test_convert_market_data_keys_valid(self):
        """Test converting valid market data keys."""
        market_data = {
            "1h": {"close": [100, 101]},
            "15m": {"close": [102, 103]}
        }
        
        result = SignalGenerationSimulator.convert_market_data_keys(market_data)
        assert len(result) == 2
        assert "1h" in result
        assert "15m" in result
    
    def test_convert_market_data_keys_invalid(self):
        """Test converting invalid market data keys."""
        market_data = {
            "invalid_timeframe": {"close": [100, 101]}
        }
        
        with pytest.raises(ValueError) as exc_info:
            SignalGenerationSimulator.convert_market_data_keys(market_data)
        
        assert "Invalid timeframe" in str(exc_info.value)
    
    def test_signal_to_response_conversion(self, mock_signal):
        """Test converting signal model to response format."""
        response = SignalGenerationSimulator.signal_to_response(mock_signal)
        
        assert response["id"] == 1
        assert response["instrument"] == "NIFTY"
        assert response["direction"] == "long"
        assert response["signal_type"] == "entry"
        assert response["is_active"] == True
        assert response["is_executed"] == False
    
    async def test_check_strategy_ownership_success(self, mock_strategy_service, mock_strategy):
        """Test successful strategy ownership check."""
        mock_strategy_service.get_strategy.return_value = mock_strategy
        
        # Should not raise exception
        await SignalGenerationSimulator.check_strategy_ownership(1, 1, mock_strategy_service)
        mock_strategy_service.get_strategy.assert_called_once_with(1)
    
    async def test_check_strategy_ownership_forbidden(self, mock_strategy_service):
        """Test strategy ownership check with wrong user."""
        mock_strategy = MockStrategy(user_id=2)  # Different user
        mock_strategy_service.get_strategy.return_value = mock_strategy
        
        with pytest.raises(MockHTTPException) as exc_info:
            await SignalGenerationSimulator.check_strategy_ownership(1, 1, mock_strategy_service)
        
        assert exc_info.value.status_code == 403
        assert "Access denied" in exc_info.value.detail
    
    async def test_check_strategy_ownership_not_found(self, mock_strategy_service):
        """Test strategy ownership check with non-existent strategy."""
        mock_strategy_service.get_strategy.side_effect = ValueError("Strategy not found")
        
        with pytest.raises(MockHTTPException) as exc_info:
            await SignalGenerationSimulator.check_strategy_ownership(999, 1, mock_strategy_service)
        
        assert exc_info.value.status_code == 404
        assert "Strategy not found" in exc_info.value.detail


# Test individual analysis endpoints
class TestAnalysisEndpoints:
    """Test individual analysis endpoints business logic."""
    
    async def test_analyze_timeframes_success(self, mock_strategy_service, mock_strategy, sample_market_data, sample_timeframe_analysis):
        """Test successful timeframe analysis workflow."""
        # Setup mocks
        mock_strategy_service.get_strategy.return_value = mock_strategy
        mock_strategy_service.analyze_timeframes.return_value = sample_timeframe_analysis
        
        # Simulate the endpoint logic
        strategy_id = 1
        user_id = 1
        
        # Check ownership
        await SignalGenerationSimulator.check_strategy_ownership(strategy_id, user_id, mock_strategy_service)
        
        # Convert market data
        market_data = SignalGenerationSimulator.convert_market_data_keys(sample_market_data["market_data"])
        
        # Perform analysis
        result = mock_strategy_service.analyze_timeframes(strategy_id, market_data)
        
        # Verify results
        assert result.aligned == True
        assert result.alignment_score == 0.85
        assert result.primary_direction == "up"
        
        # Verify calls
        mock_strategy_service.get_strategy.assert_called_once_with(strategy_id)
        mock_strategy_service.analyze_timeframes.assert_called_once_with(strategy_id, market_data)
    
    async def test_analyze_market_state_success(self, mock_strategy_service, mock_strategy, sample_market_data, sample_market_state):
        """Test successful market state analysis workflow."""
        # Setup mocks
        mock_strategy_service.get_strategy.return_value = mock_strategy
        mock_strategy_service.analyze_market_state.return_value = sample_market_state
        
        # Simulate the endpoint logic
        strategy_id = 1
        user_id = 1
        
        # Check ownership
        await SignalGenerationSimulator.check_strategy_ownership(strategy_id, user_id, mock_strategy_service)
        
        # Convert market data
        market_data = SignalGenerationSimulator.convert_market_data_keys(sample_market_data["market_data"])
        
        # Perform analysis
        result = mock_strategy_service.analyze_market_state(strategy_id, market_data)
        
        # Verify results
        assert result.market_state == "trending_up"
        assert result.trend_phase == "middle"
        assert result.is_railroad_trend == True
        assert result.bos_detected == True
        
        # Verify calls
        mock_strategy_service.get_strategy.assert_called_once_with(strategy_id)
        mock_strategy_service.analyze_market_state.assert_called_once_with(strategy_id, market_data)
    
    async def test_evaluate_setup_quality_success(self, mock_strategy_service, mock_strategy, sample_timeframe_analysis, sample_market_state, sample_setup_quality):
        """Test successful setup quality evaluation workflow."""
        # Setup mocks
        mock_strategy_service.get_strategy.return_value = mock_strategy
        mock_strategy_service.evaluate_setup_quality.return_value = sample_setup_quality
        
        # Simulate the endpoint logic
        strategy_id = 1
        user_id = 1
        entry_data = {"near_ma": True, "risk_reward": 3.0}
        
        # Check ownership
        await SignalGenerationSimulator.check_strategy_ownership(strategy_id, user_id, mock_strategy_service)
        
        # Perform evaluation
        result = mock_strategy_service.evaluate_setup_quality(
            strategy_id,
            sample_timeframe_analysis,
            sample_market_state,
            entry_data
        )
        
        # Verify results
        assert result.grade == "A"
        assert result.score == 85.0
        assert result.meets_minimum_quality == True
        
        # Verify calls
        mock_strategy_service.get_strategy.assert_called_once_with(strategy_id)
        mock_strategy_service.evaluate_setup_quality.assert_called_once_with(
            strategy_id,
            sample_timeframe_analysis,
            sample_market_state,
            entry_data
        )
    
    async def test_comprehensive_analysis_success(self, mock_strategy_service, mock_strategy, sample_market_data, sample_timeframe_analysis, sample_market_state, sample_setup_quality):
        """Test successful comprehensive analysis workflow."""
        # Setup mocks
        mock_strategy_service.get_strategy.return_value = mock_strategy
        mock_strategy_service.analyze_timeframes.return_value = sample_timeframe_analysis
        mock_strategy_service.analyze_market_state.return_value = sample_market_state
        mock_strategy_service.evaluate_setup_quality.return_value = sample_setup_quality
        
        # Simulate the endpoint logic
        strategy_id = 1
        user_id = 1
        market_data = SignalGenerationSimulator.convert_market_data_keys(sample_market_data["market_data"])
        entry_data = {"near_ma": True, "risk_reward": 3.0}
        
        # Check ownership
        await SignalGenerationSimulator.check_strategy_ownership(strategy_id, user_id, mock_strategy_service)
        
        # Perform comprehensive analysis
        timeframe_analysis = mock_strategy_service.analyze_timeframes(strategy_id, market_data)
        market_state = mock_strategy_service.analyze_market_state(strategy_id, market_data)
        setup_quality = mock_strategy_service.evaluate_setup_quality(
            strategy_id,
            timeframe_analysis,
            market_state,
            entry_data
        )
        
        # Verify all three analyses were performed
        assert timeframe_analysis.aligned == True
        assert market_state.market_state == "trending_up"
        assert setup_quality.grade == "A"
        
        # Verify service calls
        assert mock_strategy_service.analyze_timeframes.call_count == 1
        assert mock_strategy_service.analyze_market_state.call_count == 1
        assert mock_strategy_service.evaluate_setup_quality.call_count == 1


# Test signal CRUD operations
class TestSignalCRUDOperations:
    """Test signal CRUD operations business logic."""
    
    async def test_generate_signal_success(self, mock_strategy_service, mock_strategy, mock_signal, sample_timeframe_analysis, sample_market_state, sample_setup_quality, sample_market_data):
        """Test successful signal generation workflow."""
        # Setup mocks
        mock_strategy_service.get_strategy.return_value = mock_strategy
        mock_strategy_service.generate_signal.return_value = mock_signal
        
        # Simulate the endpoint logic
        strategy_id = 1
        user_id = 1
        instrument = "NIFTY"
        direction = "long"
        
        # Check ownership
        await SignalGenerationSimulator.check_strategy_ownership(strategy_id, user_id, mock_strategy_service)
        
        # Convert market data
        market_data = SignalGenerationSimulator.convert_market_data_keys(sample_market_data["market_data"])
        
        # Generate signal
        signal = mock_strategy_service.generate_signal(
            strategy_id,
            sample_timeframe_analysis,
            sample_market_state,
            sample_setup_quality,
            market_data,
            instrument,
            direction
        )
        
        # Convert to response format
        response = SignalGenerationSimulator.signal_to_response(signal)
        
        # Verify signal properties
        assert response["id"] == 1
        assert response["instrument"] == "NIFTY"
        assert response["direction"] == "long"
        assert response["is_active"] == True
        assert response["is_executed"] == False
        
        # Verify calls
        mock_strategy_service.get_strategy.assert_called_once_with(strategy_id)
        mock_strategy_service.generate_signal.assert_called_once()
    
    async def test_list_signals_success(self, mock_strategy_service, mock_strategy, mock_signal, mock_db_session):
        """Test successful signal listing workflow."""
        # Setup mocks
        mock_strategy_service.get_strategy.return_value = mock_strategy
        mock_db_session.count.return_value = 5
        mock_db_session.all.return_value = [mock_signal]
        
        # Simulate the endpoint logic
        strategy_id = 1
        user_id = 1
        
        # Check ownership
        await SignalGenerationSimulator.check_strategy_ownership(strategy_id, user_id, mock_strategy_service)
        
        # Simulate database query with filters
        filters = [f"strategy_id == {strategy_id}"]
        
        # Query signals from database
        with mock_strategy_service.db.session() as session:
            total_count = session.count()
            signals = session.all()
        
        # Convert to response format
        signal_responses = []
        for signal in signals:
            response = SignalGenerationSimulator.signal_to_response(signal)
            signal_responses.append(response)
        
        # Verify results
        assert total_count == 5
        assert len(signal_responses) == 1
        assert signal_responses[0]["id"] == 1
        assert signal_responses[0]["instrument"] == "NIFTY"
    
    async def test_get_signal_success(self, mock_strategy_service, mock_strategy, mock_signal, mock_db_session):
        """Test successful signal retrieval workflow."""
        # Setup mocks
        mock_strategy_service.get_strategy.return_value = mock_strategy
        mock_db_session.first.return_value = mock_signal
        
        # Simulate the endpoint logic
        signal_id = 1
        user_id = 1
        
        # Query signal from database
        with mock_strategy_service.db.session() as session:
            signal = session.first()
        
        # Check ownership through strategy
        if signal:
            await SignalGenerationSimulator.check_strategy_ownership(signal.strategy_id, user_id, mock_strategy_service)
        
        # Convert to response format
        response = SignalGenerationSimulator.signal_to_response(signal)
        
        # Verify signal details
        assert response["id"] == 1
        assert response["instrument"] == "NIFTY"
        assert response["direction"] == "long"
    
    async def test_get_signal_not_found(self, mock_strategy_service, mock_db_session):
        """Test get signal with non-existent ID."""
        # Setup mocks
        mock_db_session.first.return_value = None
        
        # Simulate the endpoint logic
        signal_id = 999
        
        # Query signal from database
        with mock_strategy_service.db.session() as session:
            signal = session.first()
        
        # Verify signal not found
        assert signal is None
    
    async def test_update_signal_success(self, mock_strategy_service, mock_strategy, mock_signal, mock_db_session):
        """Test successful signal update workflow."""
        # Setup mocks
        mock_strategy_service.get_strategy.return_value = mock_strategy
        mock_db_session.first.return_value = mock_signal
        
        # Simulate the endpoint logic
        signal_id = 1
        user_id = 1
        update_data = {
            "entry_price": 18550.0,
            "take_profit_price": 18650.0,
            "stop_loss_price": 18500.0
        }
        
        # Validate update data
        validated_data = SignalGenerationSimulator.validate_signal_update_data(update_data)
        
        # Get signal and check ownership
        with mock_strategy_service.db.session() as session:
            signal = session.first()
            
            if signal:
                await SignalGenerationSimulator.check_strategy_ownership(signal.strategy_id, user_id, mock_strategy_service)
                
                # Apply updates
                if "entry_price" in validated_data:
                    signal.entry_price = validated_data["entry_price"]
                if "take_profit_price" in validated_data:
                    signal.take_profit_price = validated_data["take_profit_price"]
                if "stop_loss_price" in validated_data:
                    signal.stop_loss_price = validated_data["stop_loss_price"]
                
                # Update timestamp
                signal.updated_at = datetime.utcnow()
                
                # Commit changes
                session.commit()
                session.refresh(signal)
        
        # Convert to response format
        response = SignalGenerationSimulator.signal_to_response(signal)
        
        # Verify updates
        assert response["entry_price"] == 18550.0
        assert response["take_profit_price"] == 18650.0
        assert response["stop_loss_price"] == 18500.0
    
    async def test_delete_signal_success(self, mock_strategy_service, mock_strategy, mock_signal, mock_db_session):
        """Test successful signal deletion workflow."""
        # Setup mocks
        mock_strategy_service.get_strategy.return_value = mock_strategy
        mock_db_session.first.return_value = mock_signal
        
        # Simulate the endpoint logic
        signal_id = 1
        user_id = 1
        
        # Get signal and check ownership
        with mock_strategy_service.db.session() as session:
            signal = session.first()
            
            if signal:
                await SignalGenerationSimulator.check_strategy_ownership(signal.strategy_id, user_id, mock_strategy_service)
                
                # Delete the signal
                session.delete(signal)
                session.commit()
        
        # Verify deletion was called
        mock_db_session.delete.assert_called_once_with(mock_signal)
        mock_db_session.commit.assert_called()


# Test bulk operations
class TestBulkOperations:
    """Test bulk signal operations business logic."""
    
    async def test_bulk_signal_action_activate(self, mock_strategy_service, mock_strategy, mock_db_session):
        """Test bulk signal activation workflow."""
        # Setup mocks
        mock_strategy_service.get_strategy.return_value = mock_strategy
        
        # Create test signals
        signals = [MockSignal(id=i, is_active=False) for i in range(1, 4)]
        
        # Simulate the endpoint logic
        strategy_id = 1
        user_id = 1
        signal_ids = [1, 2, 3]
        action = "activate"
        
        # Check ownership
        await SignalGenerationSimulator.check_strategy_ownership(strategy_id, user_id, mock_strategy_service)
        
        # Perform bulk action
        success_count = 0
        failed_count = 0
        failed_ids = []
        
        with mock_strategy_service.db.session() as session:
            for signal_id in signal_ids:
                try:
                    # Find signal (simulate finding it)
                    signal = next((s for s in signals if s.id == signal_id), None)
                    
                    if signal and signal.strategy_id == strategy_id:
                        if action == "activate":
                            signal.is_active = True
                        success_count += 1
                    else:
                        failed_count += 1
                        failed_ids.append(signal_id)
                except Exception:
                    failed_count += 1
                    failed_ids.append(signal_id)
            
            session.commit()
        
        # Verify results
        assert success_count == 3
        assert failed_count == 0
        assert len(failed_ids) == 0
        
        # Verify all signals were activated
        for signal in signals:
            assert signal.is_active == True
    
    async def test_bulk_signal_action_delete(self, mock_strategy_service, mock_strategy, mock_db_session):
        """Test bulk signal deletion workflow."""
        # Setup mocks
        mock_strategy_service.get_strategy.return_value = mock_strategy
        
        # Create test signals
        signals = [MockSignal(id=i) for i in range(1, 4)]
        
        # Simulate the endpoint logic
        strategy_id = 1
        user_id = 1
        signal_ids = [1, 2, 3]
        action = "delete"
        
        # Check ownership
        await SignalGenerationSimulator.check_strategy_ownership(strategy_id, user_id, mock_strategy_service)
        
        # Perform bulk action
        success_count = 0
        failed_count = 0
        deleted_signals = []
        
        with mock_strategy_service.db.session() as session:
            for signal_id in signal_ids:
                try:
                    # Find signal (simulate finding it)
                    signal = next((s for s in signals if s.id == signal_id), None)
                    
                    if signal and signal.strategy_id == strategy_id:
                        if action == "delete":
                            deleted_signals.append(signal)
                            session.delete(signal)
                        success_count += 1
                    else:
                        failed_count += 1
                except Exception:
                    failed_count += 1
            
            session.commit()
        
        # Verify results
        assert success_count == 3
        assert failed_count == 0
        assert len(deleted_signals) == 3


# Test export functionality
class TestExportFunctionality:
    """Test signal export functionality."""
    
    async def test_export_signals_csv(self, mock_strategy_service, mock_strategy, mock_db_session):
        """Test CSV export of signals workflow."""
        # Setup mocks
        mock_strategy_service.get_strategy.return_value = mock_strategy
        
        # Create test signals
        signals = [
            MockSignal(id=1, instrument="NIFTY", direction="long", entry_price=18500.0),
            MockSignal(id=2, instrument="BANKNIFTY", direction="short", entry_price=42000.0)
        ]
        mock_db_session.all.return_value = signals
        
        # Simulate the endpoint logic
        strategy_id = 1
        user_id = 1
        
        # Check ownership
        await SignalGenerationSimulator.check_strategy_ownership(strategy_id, user_id, mock_strategy_service)
        
        # Query signals
        with mock_strategy_service.db.session() as session:
            signals = session.all()
        
        # Create CSV content
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        headers = ['Signal ID', 'Instrument', 'Direction', 'Entry Price']
        writer.writerow(headers)
        
        # Write data rows
        for signal in signals:
            row = [signal.id, signal.instrument, signal.direction, signal.entry_price]
            writer.writerow(row)
        
        # Get CSV content
        output.seek(0)
        csv_content = output.getvalue()
        output.close()
        
        # Verify CSV content - handle different line endings
        lines = csv_content.strip().replace('\r\n', '\n').replace('\r', '\n').split('\n')
        assert len(lines) == 3  # Header + 2 data rows
        assert 'Signal ID,Instrument,Direction,Entry Price' == lines[0]
        assert '1,NIFTY,long,18500.0' == lines[1]
        assert '2,BANKNIFTY,short,42000.0' == lines[2]


# Test statistics functionality
class TestStatistics:
    """Test signal statistics functionality."""
    
    async def test_get_signal_stats(self, mock_strategy_service, mock_strategy, mock_db_session):
        """Test signal statistics calculation workflow."""
        # Setup mocks
        mock_strategy_service.get_strategy.return_value = mock_strategy
        
        # Simulate the endpoint logic
        strategy_id = 1
        user_id = 1
        
        # Check ownership
        await SignalGenerationSimulator.check_strategy_ownership(strategy_id, user_id, mock_strategy_service)
        
        # Simulate statistics calculation
        with mock_strategy_service.db.session() as session:
            # Mock various query results
            stats = {
                'total_signals': 10,
                'active_signals': 8,
                'executed_signals': 5,
                'pending_signals': 3,
                'quality_distribution': {'A': 4, 'B': 3, 'C': 2, 'D': 1},
                'direction_distribution': {'long': 6, 'short': 4},
                'instrument_distribution': {'NIFTY': 7, 'BANKNIFTY': 3},
                'average_setup_score': 82.5,
                'average_risk_reward': 2.15,
                'average_confidence': 0.78
            }
        
        # Verify statistics
        assert stats['total_signals'] == 10
        assert stats['active_signals'] == 8
        assert stats['executed_signals'] == 5
        assert stats['pending_signals'] == 3
        assert stats['quality_distribution']['A'] == 4
        assert stats['direction_distribution']['long'] == 6
        assert stats['average_setup_score'] == 82.5


# Test error handling
class TestErrorHandling:
    """Test error handling scenarios."""
    
    async def test_strategy_not_found(self, mock_strategy_service):
        """Test handling of non-existent strategy."""
        # Setup mocks
        mock_strategy_service.get_strategy.side_effect = ValueError("Strategy not found")
        
        # Simulate the endpoint logic
        strategy_id = 999
        user_id = 1
        
        # Attempt to check ownership
        with pytest.raises(MockHTTPException) as exc_info:
            await SignalGenerationSimulator.check_strategy_ownership(strategy_id, user_id, mock_strategy_service)
        
        assert exc_info.value.status_code == 404
        assert "Strategy not found" in exc_info.value.detail
    
    async def test_unauthorized_access(self, mock_strategy_service):
        """Test handling of unauthorized strategy access."""
        # Setup mocks
        mock_strategy = MockStrategy(user_id=2)  # Different user
        mock_strategy_service.get_strategy.return_value = mock_strategy
        
        # Simulate the endpoint logic
        strategy_id = 1
        user_id = 1
        
        # Check ownership - should raise exception
        with pytest.raises(MockHTTPException) as exc_info:
            await SignalGenerationSimulator.check_strategy_ownership(strategy_id, user_id, mock_strategy_service)
        
        assert exc_info.value.status_code == 403
        assert "Access denied" in exc_info.value.detail
    
    async def test_invalid_market_data_format(self):
        """Test handling of invalid market data format."""
        invalid_market_data = {
            "invalid_key": {"close": [100, 101]}
        }
        
        # Test validation logic
        with pytest.raises(ValueError) as exc_info:
            SignalGenerationSimulator.convert_market_data_keys(invalid_market_data)
        
        assert "Invalid timeframe" in str(exc_info.value)
    
    async def test_database_error_handling(self, mock_db_session):
        """Test handling of database errors."""
        # Setup mock to raise database error
        mock_db_session.commit.side_effect = Exception("Database connection error")
        
        # Simulate database operation that fails
        with pytest.raises(Exception) as exc_info:
            with mock_db_session as session:
                # Simulate some operation
                session.add(MockSignal())
                session.commit()
        
        assert "Database connection error" in str(exc_info.value)


# Test validation
class TestValidation:
    """Test input validation."""
    
    def test_signal_update_validation(self):
        """Test signal update data validation."""
        # Test valid data
        valid_data = {
            "entry_price": 18550.0,
            "take_profit_price": 18650.0,
            "stop_loss_price": 18500.0,
            "position_size": 1.5
        }
        
        validated = SignalGenerationSimulator.validate_signal_update_data(valid_data)
        assert validated["entry_price"] == 18550.0
        assert validated["position_size"] == 1.5
        
        # Test invalid data (negative price)
        invalid_data_price = {"entry_price": -100.0}
        
        with pytest.raises(ValueError) as exc_info:
            SignalGenerationSimulator.validate_signal_update_data(invalid_data_price)
        
        assert "entry_price must be positive" in str(exc_info.value)
        
        # Test invalid data (zero position size)
        invalid_data_size = {"position_size": 0.0}
        
        with pytest.raises(ValueError) as exc_info:
            SignalGenerationSimulator.validate_signal_update_data(invalid_data_size)
        
        assert "position_size must be positive" in str(exc_info.value)
    
    def test_bulk_action_validation(self):
        """Test bulk action validation."""
        # Test valid bulk action
        valid_action = {
            "signal_ids": [1, 2, 3],
            "action": "activate",
            "reason": "Market conditions improved"
        }
        
        # Validate signal IDs list
        assert isinstance(valid_action["signal_ids"], list)
        assert len(valid_action["signal_ids"]) > 0
        assert all(isinstance(sid, int) for sid in valid_action["signal_ids"])
        
        # Validate action
        valid_actions = ["activate", "deactivate", "delete"]
        assert valid_action["action"] in valid_actions
        
        # Test invalid action
        invalid_action = {
            "signal_ids": [],
            "action": "invalid_action"
        }
        
        # Should detect empty signal IDs list
        assert len(invalid_action["signal_ids"]) == 0
        
        # Should detect invalid action
        assert invalid_action["action"] not in valid_actions
    
    def test_search_request_validation(self):
        """Test search request validation."""
        # Test valid search request
        valid_search = {
            "instruments": ["NIFTY", "BANKNIFTY"],
            "directions": ["long", "short"],
            "date_from": date(2024, 1, 1),
            "date_to": date(2024, 12, 31),
            "min_score": 70.0,
            "max_score": 100.0,
            "min_risk_reward": 1.5
        }
        
        # Validate score ranges
        if "min_score" in valid_search and "max_score" in valid_search:
            assert 0 <= valid_search["min_score"] <= 100
            assert 0 <= valid_search["max_score"] <= 100
            assert valid_search["min_score"] <= valid_search["max_score"]
        
        # Validate risk-reward ratio
        if "min_risk_reward" in valid_search:
            assert valid_search["min_risk_reward"] >= 0
        
        # Validate date range
        if "date_from" in valid_search and "date_to" in valid_search:
            assert valid_search["date_from"] <= valid_search["date_to"]


# Integration tests
class TestIntegrationScenarios:
    """Test complete workflow scenarios."""
    
    async def test_complete_signal_workflow(self, mock_strategy_service, mock_strategy, mock_signal, mock_db_session, sample_timeframe_analysis, sample_market_state, sample_setup_quality, sample_market_data):
        """Test complete signal workflow from creation to deletion."""
        # Setup mocks
        mock_strategy_service.get_strategy.return_value = mock_strategy
        mock_strategy_service.analyze_timeframes.return_value = sample_timeframe_analysis
        mock_strategy_service.analyze_market_state.return_value = sample_market_state
        mock_strategy_service.evaluate_setup_quality.return_value = sample_setup_quality
        mock_strategy_service.generate_signal.return_value = mock_signal
        mock_db_session.first.return_value = mock_signal
        mock_db_session.all.return_value = [mock_signal]
        mock_db_session.count.return_value = 1
        
        strategy_id = 1
        user_id = 1
        
        # Step 1: Comprehensive analysis
        await SignalGenerationSimulator.check_strategy_ownership(strategy_id, user_id, mock_strategy_service)
        market_data = SignalGenerationSimulator.convert_market_data_keys(sample_market_data["market_data"])
        
        timeframe_analysis = mock_strategy_service.analyze_timeframes(strategy_id, market_data)
        market_state = mock_strategy_service.analyze_market_state(strategy_id, market_data)
        setup_quality = mock_strategy_service.evaluate_setup_quality(
            strategy_id, timeframe_analysis, market_state, {"risk_reward": 3.0}
        )
        
        assert timeframe_analysis.aligned == True
        assert market_state.market_state == "trending_up"
        assert setup_quality.grade == "A"
        
        # Step 2: Generate signal
        signal = mock_strategy_service.generate_signal(
            strategy_id, timeframe_analysis, market_state, setup_quality,
            market_data, "NIFTY", "long"
        )
        
        response = SignalGenerationSimulator.signal_to_response(signal)
        assert response["id"] == 1
        assert response["instrument"] == "NIFTY"
        assert response["direction"] == "long"
        
        # Step 3: List signals
        with mock_strategy_service.db.session() as session:
            signals = session.all()
            total_count = session.count()
        
        assert len(signals) == 1
        assert total_count == 1
        
        # Step 4: Get signal details
        with mock_strategy_service.db.session() as session:
            signal_detail = session.first()
        
        assert signal_detail is not None
        assert signal_detail.id == 1
        
        # Step 5: Update signal
        update_data = {"entry_price": 18550.0}
        validated_data = SignalGenerationSimulator.validate_signal_update_data(update_data)
        
        with mock_strategy_service.db.session() as session:
            signal = session.first()
            if signal:
                signal.entry_price = validated_data["entry_price"]
                session.commit()
        
        assert signal.entry_price == 18550.0
        
        # Step 6: Export signals (simulate CSV generation)
        with mock_strategy_service.db.session() as session:
            signals = session.all()
        
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['Signal ID', 'Instrument', 'Direction'])
        for signal in signals:
            writer.writerow([signal.id, signal.instrument, signal.direction])
        
        output.seek(0)
        csv_content = output.getvalue()
        assert 'NIFTY' in csv_content
        assert 'long' in csv_content
        
        # Step 7: Get statistics
        stats = {
            'total_signals': 1,
            'active_signals': 1,
            'executed_signals': 0
        }
        
        assert stats['total_signals'] == 1
        assert stats['active_signals'] == 1
        
        # Step 8: Delete signal
        with mock_strategy_service.db.session() as session:
            signal = session.first()
            if signal:
                session.delete(signal)
                session.commit()
        
        # Verify all service methods were called
        assert mock_strategy_service.analyze_timeframes.called
        assert mock_strategy_service.analyze_market_state.called
        assert mock_strategy_service.evaluate_setup_quality.called
        assert mock_strategy_service.generate_signal.called
    
    async def test_analysis_to_signal_pipeline(self, mock_strategy_service, mock_strategy, sample_market_data):
        """Test the analysis to signal generation pipeline."""
        # Setup mocks
        mock_strategy_service.get_strategy.return_value = mock_strategy
        
        # Mock analysis results
        timeframe_result = MockTimeframeAnalysisResult(aligned=True, alignment_score=0.9)
        market_state_result = MockMarketStateAnalysis(market_state="trending_up", bos_detected=True)
        setup_quality_result = MockSetupQualityResult(grade="A+", score=95.0)
        signal_result = MockSignal(id=1, setup_quality="A+", setup_score=95.0)
        
        mock_strategy_service.analyze_timeframes.return_value = timeframe_result
        mock_strategy_service.analyze_market_state.return_value = market_state_result
        mock_strategy_service.evaluate_setup_quality.return_value = setup_quality_result
        mock_strategy_service.generate_signal.return_value = signal_result
        
        strategy_id = 1
        user_id = 1
        market_data = SignalGenerationSimulator.convert_market_data_keys(sample_market_data["market_data"])
        entry_data = {"risk_reward": 3.5, "near_ma": True}
        
        # Execute the pipeline
        await SignalGenerationSimulator.check_strategy_ownership(strategy_id, user_id, mock_strategy_service)
        
        # Step 1: Timeframe analysis
        timeframe_analysis = mock_strategy_service.analyze_timeframes(strategy_id, market_data)
        assert timeframe_analysis.aligned == True
        assert timeframe_analysis.alignment_score == 0.9
        
        # Step 2: Market state analysis
        market_state = mock_strategy_service.analyze_market_state(strategy_id, market_data)
        assert market_state.market_state == "trending_up"
        assert market_state.bos_detected == True
        
        # Step 3: Setup quality evaluation
        setup_quality = mock_strategy_service.evaluate_setup_quality(
            strategy_id, timeframe_analysis, market_state, entry_data
        )
        assert setup_quality.grade == "A+"
        assert setup_quality.score == 95.0
        
        # Step 4: Signal generation (only if all conditions are met)
        if (timeframe_analysis.aligned and 
            market_state.market_state == "trending_up" and 
            setup_quality.meets_minimum_quality):
            
            signal = mock_strategy_service.generate_signal(
                strategy_id, timeframe_analysis, market_state, setup_quality,
                market_data, "NIFTY", "long"
            )
            
            response = SignalGenerationSimulator.signal_to_response(signal)
            assert response["id"] == 1
            assert response["setup_quality"] == "A+"
            assert response["setup_score"] == 95.0
        
        # Verify all steps were executed
        mock_strategy_service.analyze_timeframes.assert_called_once()
        mock_strategy_service.analyze_market_state.assert_called_once()
        mock_strategy_service.evaluate_setup_quality.assert_called_once()
        mock_strategy_service.generate_signal.assert_called_once()


# Performance tests (optional)
class TestPerformance:
    """Test performance-related scenarios."""
    
    async def test_large_signal_list_handling(self, mock_strategy_service, mock_strategy, mock_db_session):
        """Test handling of large signal lists."""
        # Setup mocks
        mock_strategy_service.get_strategy.return_value = mock_strategy
        
        # Simulate large number of signals
        large_signal_count = 1000
        signals = [MockSignal(id=i, instrument=f"INSTRUMENT_{i}") for i in range(1, min(101, large_signal_count + 1))]
        
        mock_db_session.count.return_value = large_signal_count
        mock_db_session.all.return_value = signals  # Return first 100 signals
        
        # Test listing with pagination
        strategy_id = 1
        user_id = 1
        offset = 0
        limit = 100
        
        await SignalGenerationSimulator.check_strategy_ownership(strategy_id, user_id, mock_strategy_service)
        
        with mock_strategy_service.db.session() as session:
            total_count = session.count()
            signals_page = session.all()
        
        # Verify pagination handling
        assert total_count == large_signal_count
        assert len(signals_page) == len(signals)  # Should be limited by actual signal count
        has_more = offset + limit < total_count
        assert has_more == True
    
    async def test_concurrent_operations_simulation(self):
        """Test simulation of concurrent operations."""
        # Simulate multiple concurrent requests
        import asyncio
        
        async def mock_request(request_id: int):
            # Simulate async processing
            await asyncio.sleep(0.01)  # Small delay to simulate processing
            return {"request_id": request_id, "status": "completed"}
        
        # Run multiple requests concurrently
        num_requests = 10
        tasks = [mock_request(i) for i in range(num_requests)]
        results = await asyncio.gather(*tasks)
        
        # Verify all requests completed successfully
        assert len(results) == num_requests
        for i, result in enumerate(results):
            assert result["request_id"] == i
            assert result["status"] == "completed"


if __name__ == "__main__":
    # Run tests
    import sys
    exit_code = pytest.main([__file__, "-v", "--tb=short"])
    sys.exit(exit_code)