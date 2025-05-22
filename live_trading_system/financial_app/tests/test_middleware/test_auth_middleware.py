"""
Tests for the authentication middleware.
This test file covers:

- Testing public vs. protected endpoints
- Verifying request state is correctly set
- Testing token extraction from headers
- Checking error response format
- Testing different HTTP methods
- Validating various URL path patterns
"""
import pytest
from fastapi import FastAPI, Request, Response, status
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from app.middleware.auth import AuthMiddleware


# Create a mock endpoint for testing
async def mock_endpoint(request: Request):
    # Return request state information for testing
    return JSONResponse({
        "auth_type": getattr(request.state, "auth_type", None),
        "user": getattr(request.state, "user", None),
        "roles": getattr(request.state, "roles", None),
    })


# Mock auth endpoint that would be public
async def mock_auth_endpoint(request: Request):
    return JSONResponse({"status": "ok"})


@pytest.fixture
def app():
    """Create a test FastAPI app with the AuthMiddleware."""
    app = FastAPI()
    app.add_middleware(AuthMiddleware)
    
    # Add test routes
    app.add_api_route("/api/v1/protected", mock_endpoint)
    app.add_api_route("/api/v1/auth/token", mock_auth_endpoint, methods=["POST"])
    app.add_api_route("/health", mock_endpoint)
    app.add_api_route("/docs", mock_endpoint)
    
    return app


@pytest.fixture
def client(app):
    """Create a test client for the app."""
    return TestClient(app)


class TestAuthMiddleware:
    """Test suite for the AuthMiddleware."""
    
    def test_public_endpoints(self, client):
        """Test that public endpoints are accessible without authentication."""
        # Test health endpoint
        response = client.get("/health")
        assert response.status_code == 200
        
        # Test docs endpoint
        response = client.get("/docs")
        assert response.status_code == 200
        
        # Test auth token endpoint
        response = client.post("/api/v1/auth/token")
        assert response.status_code == 200
    
    def test_protected_endpoint_without_auth(self, client):
        """Test that protected endpoints require authentication."""
        response = client.get("/api/v1/protected")
        assert response.status_code == 401
        assert response.json()["type"] == "authentication_error"
        assert "Authentication required" in response.json()["detail"]
    
    def test_protected_endpoint_with_auth_header(self, client):
        """Test that protected endpoints are accessible with auth header."""
        # We're not testing actual token validation here, just that
        # the middleware allows requests with an auth header to pass through
        response = client.get(
            "/api/v1/protected", 
            headers={"Authorization": "Bearer some_token"}
        )
        
        # The middleware should let it through to the endpoint handler
        assert response.status_code == 200
    
    def test_request_state_for_public_endpoint(self, client):
        """Test that request state is properly set for public endpoints."""
        response = client.get("/health")
        assert response.status_code == 200
        
        # Check that auth state is cleared for public endpoints
        assert response.json()["auth_type"] is None
        assert response.json()["user"] is None
        assert response.json()["roles"] == []
    
    def test_various_public_path_patterns(self, client):
        """Test different path patterns that should be public."""
        public_paths = [
            "/health",
            "/health/details",  # Test subdirectory of public path
            "/docs",
            "/redoc",
            "/openapi.json",
            "/api/v1/auth/token",
            "/api/v1/auth/register",
        ]
        
        for path in public_paths:
            # Add dynamic routes for testing
            app = client.app
            app.add_api_route(path, mock_endpoint)
            
            response = client.get(path)
            assert response.status_code == 200, f"Path {path} should be public"
    
    def test_various_protected_path_patterns(self, client):
        """Test different path patterns that should be protected."""
        protected_paths = [
            "/api/v1/users",
            "/api/v1/strategies",
            "/api/v1/market-data/analysis",
            "/api/v1/orders",
        ]
        
        for path in protected_paths:
            # Add dynamic routes for testing
            app = client.app
            app.add_api_route(path, mock_endpoint)
            
            # Without auth header
            response = client.get(path)
            assert response.status_code == 401, f"Path {path} should require auth"
            
            # With auth header
            response = client.get(
                path, 
                headers={"Authorization": "Bearer some_token"}
            )
            assert response.status_code == 200, f"Path {path} should accept auth header"
    
    def test_extract_token(self, app):
        """Test that tokens are properly extracted from headers."""
        # Create an instance of the middleware for testing
        middleware = AuthMiddleware(app=app)
        
        # Test valid Bearer token
        token = middleware._extract_token("Bearer test_token")
        assert token == "test_token"
        
        # Test missing Bearer prefix
        token = middleware._extract_token("test_token")
        assert token is None
        
        # Test empty header
        token = middleware._extract_token("")
        assert token is None
    
    def test_auth_error_response(self, app):
        """Test the error response format."""
        middleware = AuthMiddleware(app=app)
        response = middleware._auth_error_response("Test error")
        
        assert response.status_code == 401
        assert response.headers["WWW-Authenticate"] == "Bearer"
        
        content = response.body
        # Need to convert bytes to JSON for assertion
        import json
        content_json = json.loads(content)
        
        assert content_json["detail"] == "Test error"
        assert content_json["type"] == "authentication_error"
    
    @pytest.mark.parametrize("method", ["GET", "POST", "PUT", "DELETE"])
    def test_http_methods(self, client, method):
        """Test that authentication is required regardless of HTTP method."""
        # Add dynamic route for testing if method is not GET
        app = client.app
        app.add_api_route("/api/v1/test_method", mock_endpoint, methods=[method])
        
        # Make request with specific method
        request_func = getattr(client, method.lower())
        response = request_func("/api/v1/test_method")
        
        assert response.status_code == 401, f"Method {method} should require auth"
        
        # With auth header
        response = request_func(
            "/api/v1/test_method", 
            headers={"Authorization": "Bearer token"}
        )
        assert response.status_code == 200, f"Method {method} should accept auth header"