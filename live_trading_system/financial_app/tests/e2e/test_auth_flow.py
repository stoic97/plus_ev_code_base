# tests/e2e/test_auth_flow.py

import pytest
from fastapi.testclient import TestClient
import logging
from app.main import app

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create test client
client = TestClient(app)

def test_complete_user_journey():
    """Test a complete user journey from registration to authenticated actions."""
    logger.info("Starting complete user journey test")
    
    # Step 1: Register a new user
    register_response = client.post(
        "/api/v1/auth/register",
        json={
            "username": "journey_user",
            "email": "journey@example.com",
            "password": "JourneyP@ssw0rd123",
            "full_name": "Journey User"
        }
    )
    logger.info(f"Register response: {register_response.status_code} - {register_response.text}")
    assert register_response.status_code == 200
    assert register_response.json()["success"] is True
    
    # Step 2: Login with new user
    login_response = client.post(
        "/api/v1/auth/token",
        data={
            "username": "journey_user",
            "password": "JourneyP@ssw0rd123"
        }
    )
    logger.info(f"Login response: {login_response.status_code}")
    assert login_response.status_code == 200
    token_data = login_response.json()
    assert "access_token" in token_data
    
    # Step 3: Access protected resource
    headers = {"Authorization": f"Bearer {token_data['access_token']}"}
    me_response = client.get("/api/v1/auth/me", headers=headers)
    logger.info(f"Me response: {me_response.status_code}")
    assert me_response.status_code == 200
    assert me_response.json()["username"] == "journey_user"
    
    # Step 4: Refresh token
    refresh_response = client.post(
        "/api/v1/auth/refresh",
        json={"token": token_data["refresh_token"]}
    )
    logger.info(f"Refresh response: {refresh_response.status_code}")
    assert refresh_response.status_code == 200
    new_token_data = refresh_response.json()
    assert new_token_data["access_token"] != token_data["access_token"]
    
    # Step 5: Use new token to access protected resource
    new_headers = {"Authorization": f"Bearer {new_token_data['access_token']}"}
    me_response2 = client.get("/api/v1/auth/me", headers=new_headers)
    logger.info(f"Second me response: {me_response2.status_code}")
    assert me_response2.status_code == 200
    
    # Step 6: Logout
    logout_response = client.post("/api/v1/auth/logout", headers=headers)
    logger.info(f"Logout response: {logout_response.status_code}")
    assert logout_response.status_code == 200

def test_auth_failure_scenarios():
    """Test various authentication failure scenarios."""
    
    # Test invalid login
    login_response = client.post(
        "/api/v1/auth/token",
        data={
            "username": "nonexistent_user",
            "password": "WrongPassword123!"
        }
    )
    assert login_response.status_code == 401
    
    # Test accessing protected resource without token
    me_response = client.get("/api/v1/auth/me")
    assert me_response.status_code in (401, 403)  # Either is acceptable
    
    # Test with invalid token
    headers = {"Authorization": "Bearer invalid_token_here"}
    me_response = client.get("/api/v1/auth/me", headers=headers)
    assert me_response.status_code == 401
    
    # Test with expired refresh token
    refresh_response = client.post(
        "/api/v1/auth/refresh",
        json={"token": "expired_token_here"}
    )
    assert refresh_response.status_code == 401

def test_password_strength_requirements():
    """Test password strength requirements during registration."""
    
    # Test too short password
    response = client.post(
        "/api/v1/auth/register",
        json={
            "username": "password_test_user",
            "email": "password_test@example.com",
            "password": "Short123!",  # Too short
            "full_name": "Password Test"
        }
    )
    assert response.status_code in (400, 422)  # Either is acceptable
    
    # Test password without uppercase
    response = client.post(
        "/api/v1/auth/register",
        json={
            "username": "password_test_user",
            "email": "password_test@example.com",
            "password": "nouppercase123!",  # No uppercase
            "full_name": "Password Test"
        }
    )
    assert response.status_code in (400, 422)
    
    # Test password without special character
    response = client.post(
        "/api/v1/auth/register",
        json={
            "username": "password_test_user",
            "email": "password_test@example.com",
            "password": "NoSpecialChar123",  # No special character
            "full_name": "Password Test"
        }
    )
    assert response.status_code in (400, 422)

def test_path_discovery():
    """Helper test to discover working paths."""
    logger.info("Testing path discovery for register endpoint:")
    path_patterns = [
        "/register",
        "/auth/register",
        "/api/auth/register",
        "/api/v1/auth/register",
        "/v1/auth/register",
        "/auth",
        "/api/auth",
        "/api/v1/auth"
    ]
    
    for path in path_patterns:
        response = client.post(
            path,
            json={
                "username": "path_test_user",
                "email": "path_test@example.com", 
                "password": "PathTestP@ssw0rd123",
                "full_name": "Path Test User"
            }
        )
        logger.info(f"Path {path}: status {response.status_code}")
        
    # Get a list of all routes
    # This will only work if your app exposes this info
    routes_response = client.get("/")
    logger.info(f"Routes response: {routes_response.status_code} - {routes_response.text}")

def test_debug_paths():
    """Debug test to identify correct API paths."""
    # First check root and docs endpoints
    root_response = client.get("/")
    logger.info(f"Root response: {root_response.status_code} - {root_response.text}")
    
    docs_response = client.get("/docs")
    logger.info(f"Docs response: {docs_response.status_code}")
    
    # Check various API endpoints to find the correct path structure
    api_paths = [
        "/api/v1/auth/register",
        "/auth/register",
        "/api/v1/v1/auth/register",  # This might be the actual path
        "/api/auth/register",
        "/v1/auth/register",
        "/api/v1/debug/routes",  # Check the debug endpoint we added
    ]
    
    for path in api_paths:
        response = client.post(
            path,
            json={
                "username": "debug_user",
                "email": "debug@example.com",
                "password": "DebugPassword123!",
                "full_name": "Debug User"
            }
        )
        logger.info(f"POST {path}: status {response.status_code}")
