"""
Unit tests for FyersAuth class.
"""

import pytest
import asyncio
import json
import base64
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock

import aiohttp
from aioresponses import aioresponses

# Adjust imports based on your project structure
from app.providers.fyers.fyers_auth import FyersAuth
from app.providers.fyers.fyers_settings import FyersSettings
from app.providers.base.provider import AuthenticationError, ConnectionError, RateLimitError
from app.models.fyers_tokens import FyersToken


class TestFyersAuth:
    """Test cases for FyersAuth class."""
    
    @pytest.fixture
    def mock_settings(self):
        """Create mock Fyers settings."""
        settings = Mock(spec=FyersSettings)
        settings.APP_ID = "TEST123-100"
        settings.APP_SECRET = Mock()
        settings.APP_SECRET.get_secret_value.return_value = "test_secret"
        settings.ACCESS_TOKEN = None
        settings.PIN = None
        settings.REDIRECT_URI = "https://example.com/callback"
        settings.AUTH_BASE_URL = "https://api-t1.fyers.in/api/v3/"
        settings.API_BASE_URL = "https://api.fyers.in/api/v2/"
        settings.TOKEN_RENEWAL_MARGIN = 300
        settings.generate_app_id_hash.return_value = "test_hash"
        return settings
    
    @pytest.fixture
    def mock_session_factory(self):
        """Create mock database session factory."""
        session = Mock()
        session_factory = Mock(return_value=iter([session]))
        return session_factory, session
    
    @pytest.fixture
    def fyers_auth(self, mock_settings, mock_session_factory):
        """Create FyersAuth instance for testing."""
        session_factory, _ = mock_session_factory
        return FyersAuth(mock_settings, session_factory)
    
    @pytest.fixture
    def valid_jwt_token(self):
        """Create a valid JWT token for testing."""
        # Create JWT payload with expiry 1 hour from now
        exp_time = int((datetime.now() + timedelta(hours=1)).timestamp())
        payload = {"exp": exp_time, "sub": "test_user"}
        
        # Encode payload (simplified - not cryptographically valid)
        payload_bytes = json.dumps(payload).encode('utf-8')
        payload_b64 = base64.b64encode(payload_bytes).decode('utf-8')
        
        # Create fake JWT token
        return f"header.{payload_b64}.signature"


class TestInitialization(TestFyersAuth):
    """Test FyersAuth initialization."""
    
    def test_init_without_access_token(self, mock_settings, mock_session_factory):
        """Test initialization without access token."""
        session_factory, _ = mock_session_factory
        auth = FyersAuth(mock_settings, session_factory)
        
        assert auth.access_token is None
        assert auth.refresh_token_value is None
        assert auth.token_expiry is None
    
    def test_init_with_discord_webhooks(self, mock_settings, mock_session_factory):
        """Test initialization with Discord webhooks."""
        webhooks = ["https://discord.com/webhook1", "https://discord.com/webhook2"]
        session_factory, _ = mock_session_factory
        
        auth = FyersAuth(mock_settings, session_factory, discord_webhooks=webhooks)
        
        assert auth.discord_webhooks == webhooks


class TestAuthUrl(TestFyersAuth):
    """Test auth URL generation."""
    
    @pytest.mark.asyncio
    async def test_get_auth_url_default_state(self, fyers_auth):
        """Test auth URL generation with default state."""
        url = await fyers_auth.get_auth_url()
        
        assert "client_id=TEST123-100" in url
        assert "redirect_uri=https%3A%2F%2Fexample.com%2Fcallback" in url  # Fixed encoding
        assert "response_type=code" in url
        assert "state=fyers_auth_state" in url
        assert url.startswith("https://api-t1.fyers.in/api/v3/generate-authcode")
    
    @pytest.mark.asyncio
    async def test_get_auth_url_custom_state(self, fyers_auth):
        """Test auth URL generation with custom state."""
        url = await fyers_auth.get_auth_url("custom_state")
        
        assert "state=custom_state" in url


class TestTokenAcquisition(TestFyersAuth):
    """Test token acquisition and exchange."""
    
    @pytest.mark.asyncio
    async def test_get_access_token_from_code_success(self, fyers_auth, valid_jwt_token):
        """Test successful token acquisition from auth code."""
        auth_code = "test_auth_code"
        
        # Mock successful API response
        mock_response = {
            "s": "ok",
            "access_token": valid_jwt_token,
            "refresh_token": "refresh_token"
        }
        
        with aioresponses() as m:
            m.post(
                "https://api-t1.fyers.in/api/v3/validate-authcode",
                payload=mock_response
            )
            
            # Mock database save
            with patch.object(fyers_auth, 'save_token_to_db', return_value=True):
                result = await fyers_auth.get_access_token_from_code(auth_code)
        
        assert result is True
        assert fyers_auth.access_token == valid_jwt_token
        assert fyers_auth.refresh_token_value == "refresh_token"
        assert fyers_auth.token_expiry is not None
    
    @pytest.mark.asyncio
    async def test_get_access_token_from_code_error_response(self, fyers_auth):
        """Test token acquisition with error response."""
        auth_code = "test_auth_code"
        
        # Mock error response
        mock_response = {
            "s": "error",
            "code": -15,
            "message": "Invalid auth code"
        }
        
        with aioresponses() as m:
            m.post(
                "https://api-t1.fyers.in/api/v3/validate-authcode",
                payload=mock_response
            )
            
            with pytest.raises(AuthenticationError, match="Failed to acquire token"):
                await fyers_auth.get_access_token_from_code(auth_code)
    
    @pytest.mark.asyncio
    async def test_get_access_token_from_code_no_token(self, fyers_auth):
        """Test token acquisition with missing access token in response."""
        auth_code = "test_auth_code"
        
        mock_response = {
            "s": "ok",
            "message": "Success but no token"
        }
        
        with aioresponses() as m:
            m.post(
                "https://api-t1.fyers.in/api/v3/validate-authcode",
                payload=mock_response
            )
            
            with pytest.raises(AuthenticationError, match="No access token in response"):
                await fyers_auth.get_access_token_from_code(auth_code)
    
    @pytest.mark.asyncio
    async def test_get_access_token_from_code_rate_limit(self, fyers_auth):
        """Test token acquisition with rate limit error."""
        auth_code = "test_auth_code"
        
        with aioresponses() as m:
            m.post(
                "https://api-t1.fyers.in/api/v3/validate-authcode",
                status=429,
                payload={"error": "Rate limit exceeded"}
            )
            
            with pytest.raises(RateLimitError, match="Fyers API rate limit exceeded"):  # Fixed expectation
                await fyers_auth.get_access_token_from_code(auth_code)
    
    @pytest.mark.asyncio
    async def test_get_access_token_empty_code(self, fyers_auth):
        """Test token acquisition with empty auth code."""
        with pytest.raises(ValueError, match="Authorization code cannot be empty"):
            await fyers_auth.get_access_token_from_code("")


class TestTokenValidation(TestFyersAuth):
    """Test token validation."""
    
    @pytest.mark.asyncio
    async def test_validate_token_success(self, fyers_auth, valid_jwt_token):
        """Test successful token validation."""
        fyers_auth.access_token = valid_jwt_token
        
        mock_response = {"s": "ok", "data": {"name": "Test User"}}
        
        with aioresponses() as m:
            m.get(
                "https://api.fyers.in/api/v2/profile",
                payload=mock_response
            )
            
            result = await fyers_auth.validate_token()
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_validate_token_failure(self, fyers_auth, valid_jwt_token):
        """Test token validation failure."""
        fyers_auth.access_token = valid_jwt_token
        
        with aioresponses() as m:
            m.get(
                "https://api.fyers.in/api/v2/profile",
                status=401
            )
            
            result = await fyers_auth.validate_token()
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_validate_token_no_token(self, fyers_auth):
        """Test validation with no token."""
        result = await fyers_auth.validate_token()
        assert result is False
    
    @pytest.mark.asyncio
    async def test_validate_custom_token(self, fyers_auth, valid_jwt_token):
        """Test validation of custom token."""
        mock_response = {"s": "ok"}
        
        with aioresponses() as m:
            m.get(
                "https://api.fyers.in/api/v2/profile",
                payload=mock_response
            )
            
            result = await fyers_auth.validate_token(valid_jwt_token)
        
        assert result is True


class TestTokenValidity(TestFyersAuth):
    """Test token validity checking."""
    
    @pytest.mark.asyncio
    async def test_has_valid_token_true(self, fyers_auth, valid_jwt_token):
        """Test has_valid_token returns True for valid token."""
        fyers_auth.access_token = valid_jwt_token
        fyers_auth.token_expiry = datetime.now() + timedelta(hours=1)
        
        result = await fyers_auth.has_valid_token()
        assert result is True
    
    @pytest.mark.asyncio
    async def test_has_valid_token_expired(self, fyers_auth, valid_jwt_token):
        """Test has_valid_token returns False for expired token."""
        fyers_auth.access_token = valid_jwt_token
        fyers_auth.token_expiry = datetime.now() - timedelta(hours=1)
        
        result = await fyers_auth.has_valid_token()
        assert result is False
    
    @pytest.mark.asyncio
    async def test_has_valid_token_expiring_soon(self, fyers_auth, valid_jwt_token):
        """Test has_valid_token returns False for token expiring soon."""
        fyers_auth.access_token = valid_jwt_token
        fyers_auth.token_expiry = datetime.now() + timedelta(seconds=100)  # Less than renewal margin
        
        result = await fyers_auth.has_valid_token()
        assert result is False
    
    @pytest.mark.asyncio
    async def test_has_valid_token_no_token(self, fyers_auth):
        """Test has_valid_token returns False when no token."""
        result = await fyers_auth.has_valid_token()
        assert result is False


class TestTokenRefresh(TestFyersAuth):
    """Test token refresh functionality."""
    
    @pytest.mark.asyncio
    async def test_refresh_token_success(self, fyers_auth, valid_jwt_token):
        """Test successful token refresh."""
        fyers_auth.refresh_token_value = "refresh_token"  # Fixed attribute name
        
        mock_response = {
            "s": "ok",
            "access_token": valid_jwt_token
        }
        
        with aioresponses() as m:
            m.post(
                "https://api-t1.fyers.in/api/v3/validate-refresh-token",
                payload=mock_response
            )
            
            with patch.object(fyers_auth, 'save_token_to_db', return_value=True):
                result = await fyers_auth.refresh_token_async()  # Fixed method name
        
        assert result is True
        assert fyers_auth.access_token == valid_jwt_token
    
    @pytest.mark.asyncio
    async def test_refresh_token_no_refresh_token(self, fyers_auth):
        """Test refresh when no refresh token available."""
        result = await fyers_auth.refresh_token_async()  # Fixed method name
        assert result is False
    
    @pytest.mark.asyncio
    async def test_refresh_token_error(self, fyers_auth):
        """Test refresh token with error response."""
        fyers_auth.refresh_token_value = "refresh_token"  # Fixed attribute name
        
        mock_response = {
            "s": "error",
            "code": -8,
            "message": "Refresh token expired"
        }
        
        with aioresponses() as m:
            m.post(
                "https://api-t1.fyers.in/api/v3/validate-refresh-token",
                payload=mock_response
            )
            
            with pytest.raises(AuthenticationError, match="Failed to refresh token"):
                await fyers_auth.refresh_token_async()  # Fixed method name


class TestTokenEnsurance(TestFyersAuth):
    """Test token ensurance functionality."""
    
    @pytest.mark.asyncio
    async def test_ensure_token_valid_token(self, fyers_auth, valid_jwt_token):
        """Test ensure_token with valid token."""
        fyers_auth.access_token = valid_jwt_token
        fyers_auth.token_expiry = datetime.now() + timedelta(hours=1)
        
        result = await fyers_auth.ensure_token()
        assert result is True
    
    @pytest.mark.asyncio
    async def test_ensure_token_refresh_success(self, fyers_auth, valid_jwt_token):
        """Test ensure_token with successful refresh."""
        fyers_auth.access_token = "old_token"
        fyers_auth.refresh_token_value = "refresh_token"  # Fixed attribute name
        fyers_auth.token_expiry = datetime.now() - timedelta(minutes=1)  # Expired
        
        with patch.object(fyers_auth, 'refresh_token_async', return_value=True):  # Fixed method name
            result = await fyers_auth.ensure_token()
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_ensure_token_load_from_db(self, fyers_auth, valid_jwt_token):
        """Test ensure_token loading from database."""
        with patch.object(fyers_auth, 'load_token_from_db', return_value=True), \
             patch.object(fyers_auth, 'has_valid_token', return_value=True):
            result = await fyers_auth.ensure_token()
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_ensure_token_all_fail(self, fyers_auth):
        """Test ensure_token when all methods fail."""
        with patch.object(fyers_auth, 'has_valid_token', return_value=False), \
             patch.object(fyers_auth, 'refresh_token_async', return_value=False), \
             patch.object(fyers_auth, 'load_token_from_db', return_value=False):
            result = await fyers_auth.ensure_token()
        
        assert result is False


class TestAuthHeaders(TestFyersAuth):
    """Test authentication headers."""
    
    def test_get_auth_headers_success(self, fyers_auth):
        """Test getting auth headers with valid token."""
        fyers_auth.access_token = "test_token"
        
        headers = fyers_auth.get_auth_headers()
        
        assert headers["Authorization"] == "TEST123-100:test_token"
        assert headers["Content-Type"] == "application/json"
    
    def test_get_auth_headers_no_token(self, fyers_auth):
        """Test getting auth headers with no token."""
        with pytest.raises(AuthenticationError, match="No access token available"):
            fyers_auth.get_auth_headers()


class TestTokenManagement(TestFyersAuth):
    """Test token management functionality."""
    
    def test_set_token_manually(self, fyers_auth, valid_jwt_token):
        """Test manually setting token."""
        fyers_auth.set_token_manually(valid_jwt_token, "refresh", datetime.now() + timedelta(hours=1))
        
        assert fyers_auth.access_token == valid_jwt_token
        assert fyers_auth.refresh_token_value == "refresh"  # Fixed attribute name
        assert fyers_auth.token_expiry is not None
    
    def test_set_token_manually_empty(self, fyers_auth):
        """Test setting empty token."""
        with pytest.raises(ValueError, match="Access token cannot be empty"):
            fyers_auth.set_token_manually("")
    
    @pytest.mark.asyncio
    async def test_clear_token(self, fyers_auth, mock_session_factory):
        """Test clearing token."""
        _, session = mock_session_factory
        fyers_auth.access_token = "test_token"
        
        with patch.object(FyersToken, 'delete_token', return_value=True):
            await fyers_auth.clear_token()
        
        assert fyers_auth.access_token is None
        assert fyers_auth.refresh_token_value is None  # Fixed attribute name
        assert fyers_auth.token_expiry is None
    
    def test_get_token_info_with_token(self, fyers_auth, valid_jwt_token):
        """Test getting token info with valid token."""
        expiry = datetime.now() + timedelta(hours=1)
        fyers_auth.access_token = valid_jwt_token
        fyers_auth.refresh_token_value = "refresh"  # Fixed attribute name
        fyers_auth.token_expiry = expiry
        
        info = fyers_auth.get_token_info()
        
        assert info["has_token"] is True
        assert info["has_refresh_token"] is True
        assert info["is_valid"] is True
        assert info["time_remaining"] > 0
    
    def test_get_token_info_no_token(self, fyers_auth):
        """Test getting token info with no token."""
        info = fyers_auth.get_token_info()
        
        assert info["has_token"] is False
        assert info["has_refresh_token"] is False
        assert info["is_valid"] is False


class TestExpiryChecking(TestFyersAuth):
    """Test token expiry checking."""
    
    @pytest.mark.asyncio
    async def test_check_token_expiry_valid(self, fyers_auth, valid_jwt_token):
        """Test checking expiry for valid token."""
        expiry = datetime.now() + timedelta(hours=1)
        fyers_auth.access_token = valid_jwt_token
        fyers_auth.token_expiry = expiry
        
        remaining = await fyers_auth.check_token_expiry()
        
        assert remaining is not None
        assert remaining.total_seconds() > 3000  # About 1 hour
    
    @pytest.mark.asyncio
    async def test_check_token_expiry_expired(self, fyers_auth, valid_jwt_token):
        """Test checking expiry for expired token."""
        fyers_auth.access_token = valid_jwt_token
        fyers_auth.token_expiry = datetime.now() - timedelta(hours=1)
        
        remaining = await fyers_auth.check_token_expiry()
        
        assert remaining == timedelta(0)
    
    @pytest.mark.asyncio
    async def test_check_token_expiry_no_token(self, fyers_auth):
        """Test checking expiry with no token."""
        remaining = await fyers_auth.check_token_expiry()
        assert remaining is None


class TestDiscordAlerts(TestFyersAuth):
    """Test Discord alert functionality."""
    
    @pytest.mark.asyncio
    async def test_send_discord_alert_success(self, fyers_auth):
        """Test successful Discord alert."""
        fyers_auth.discord_webhooks = ["https://discord.com/webhook"]
        
        with aioresponses() as m:
            m.post("https://discord.com/webhook", status=200)
            
            result = await fyers_auth.send_discord_alert("Test message", "info")
        
        assert result["webhook_0"]["success"] is True
    
    @pytest.mark.asyncio
    async def test_send_discord_alert_no_webhooks(self, fyers_auth):
        """Test Discord alert with no webhooks."""
        result = await fyers_auth.send_discord_alert("Test message")
        
        assert result["success"] is False
        assert result["reason"] == "No webhooks configured"
    
    @pytest.mark.asyncio
    async def test_send_discord_alert_failure(self, fyers_auth):
        """Test Discord alert failure."""
        fyers_auth.discord_webhooks = ["https://discord.com/webhook"]
        
        with aioresponses() as m:
            m.post("https://discord.com/webhook", status=400)
            
            result = await fyers_auth.send_discord_alert("Test message")
        
        assert result["webhook_0"]["success"] is False


class TestDatabaseOperations(TestFyersAuth):
    """Test database operations."""
    
    @pytest.mark.asyncio
    async def test_save_token_to_db_success(self, fyers_auth, mock_session_factory, valid_jwt_token):
        """Test successful token save to database."""
        _, session = mock_session_factory
        fyers_auth.access_token = valid_jwt_token
        fyers_auth.refresh_token_value = "refresh"  # Fixed attribute name
        fyers_auth.token_expiry = datetime.now() + timedelta(hours=1)
        
        with patch.object(FyersToken, 'save_token', return_value=True):
            result = await fyers_auth.save_token_to_db()
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_save_token_to_db_no_token(self, fyers_auth):
        """Test saving with no token data."""
        result = await fyers_auth.save_token_to_db()
        assert result is False
    
    @pytest.mark.asyncio
    async def test_load_token_from_db_success(self, fyers_auth, mock_session_factory, valid_jwt_token):
        """Test successful token load from database."""
        _, session = mock_session_factory
        
        mock_token = Mock()
        mock_token.access_token = valid_jwt_token
        mock_token.refresh_token = "refresh"
        mock_token.expiry = datetime.now() + timedelta(hours=1)
        mock_token.is_active = True
        
        with patch.object(FyersToken, 'get_by_app_id', return_value=mock_token):
            result = await fyers_auth.load_token_from_db()
        
        assert result is True
        assert fyers_auth.access_token == valid_jwt_token
    
    @pytest.mark.asyncio
    async def test_load_token_from_db_not_found(self, fyers_auth, mock_session_factory):
        """Test loading token when not found in database."""
        _, session = mock_session_factory
        
        with patch.object(FyersToken, 'get_by_app_id', return_value=None):
            result = await fyers_auth.load_token_from_db()
        
        assert result is False


class TestTokenParsing(TestFyersAuth):
    """Test JWT token parsing."""
    
    def test_parse_token_expiry_valid(self, fyers_auth, valid_jwt_token):
        """Test parsing expiry from valid JWT token."""
        expiry = fyers_auth.parse_token_expiry(valid_jwt_token)
        
        assert expiry is not None
        assert isinstance(expiry, datetime)
        assert expiry > datetime.now()
    
    def test_parse_token_expiry_invalid_format(self, fyers_auth):
        """Test parsing expiry from invalid token format."""
        expiry = fyers_auth.parse_token_expiry("invalid.token")
        assert expiry is None
    
    def test_parse_token_expiry_no_exp_claim(self, fyers_auth):
        """Test parsing token without exp claim."""
        # Token without exp claim
        payload = {"sub": "test_user"}
        payload_bytes = json.dumps(payload).encode('utf-8')
        payload_b64 = base64.b64encode(payload_bytes).decode('utf-8')
        token = f"header.{payload_b64}.signature"
        
        expiry = fyers_auth.parse_token_expiry(token)
        assert expiry is None


class TestUtilityMethods(TestFyersAuth):
    """Test utility methods."""
    
    def test_sanitize_log_data(self, fyers_auth):
        """Test sanitizing sensitive data for logging."""
        data = {
            "code": "secret_code",
            "appIdHash": "secret_hash",
            "public_field": "public_value"
        }
        
        sanitized = fyers_auth._sanitize_log_data(data)
        
        assert sanitized["code"] == "********"
        assert sanitized["appIdHash"] == "********"
        assert sanitized["public_field"] == "public_value"
    
    def test_update_token_state(self, fyers_auth):
        """Test updating token state."""
        expiry = datetime.now() + timedelta(hours=1)
        
        fyers_auth._update_token_state("access", "refresh", expiry)
        
        assert fyers_auth.access_token == "access"
        assert fyers_auth.refresh_token_value == "refresh"  # Fixed attribute name
        assert fyers_auth.token_expiry == expiry
    
    def test_get_renewal_instructions(self, fyers_auth):
        """Test getting renewal instructions."""
        instructions = fyers_auth.get_renewal_instructions()
        
        assert "message" in instructions
        assert "steps" in instructions
        assert "auth_url" in instructions
        assert len(instructions["steps"]) == 4


class TestSessionManagement(TestFyersAuth):
    """Test HTTP session management."""
    
    @pytest.mark.asyncio
    async def test_ensure_session_creates_session(self, fyers_auth):
        """Test that ensure_session creates a new session."""
        await fyers_auth._ensure_session()
        
        assert fyers_auth._session is not None
        assert not fyers_auth._session.closed
    
    @pytest.mark.asyncio
    async def test_close_session(self, fyers_auth):
        """Test closing HTTP session."""
        await fyers_auth._ensure_session()
        session = fyers_auth._session
        
        await fyers_auth.close()
        
        assert session.closed
        assert fyers_auth._session is None