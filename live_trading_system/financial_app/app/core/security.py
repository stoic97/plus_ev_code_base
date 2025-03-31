from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Union, List
import secrets
import hashlib
import base64
import hmac
import re
import logging
from jose import jwt, JWTError
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer, SecurityScopes
from pydantic import BaseModel, ValidationError

from app.core.config import settings
from app.core.database import DatabaseType, db_session, get_db

# Initialize logger
logger = logging.getLogger("security")

# Configure the password hashing context
# Using Argon2 (winner of the Password Hashing Competition) as primary,
# with bcrypt as fallback for legacy compatibility
pwd_context = CryptContext(
    schemes=["argon2", "bcrypt"],
    deprecated="auto",
    argon2__rounds=10,  # Controls computational cost
    argon2__memory_cost=65536,  # 64MB in KB, higher is more secure
    argon2__parallelism=4,  # Number of threads to use
    bcrypt__rounds=12  # Log2 of number of iterations
)

# OAuth2 token URL and scheme
OAUTH2_TOKEN_URL = f"{settings.API_PREFIX}/auth/token"
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl=OAUTH2_TOKEN_URL,
    scopes={
        "user": "Basic user access",
        "admin": "Administrator access",
        "trader": "Trading operations access",
        "analyst": "Market data analysis access",
        "system": "System operations access"
    }
)

# Rate limiting configuration
DEFAULT_RATE_LIMIT = 1000  # Default requests per minute
RATE_LIMITS = {
    "user": DEFAULT_RATE_LIMIT,
    "trader": DEFAULT_RATE_LIMIT * 2,  # Traders need higher limits for market data
    "analyst": DEFAULT_RATE_LIMIT * 3,  # Analysts need even higher limits for data processing
    "admin": DEFAULT_RATE_LIMIT * 5,    # Admins need the highest limits
    "system": DEFAULT_RATE_LIMIT * 10   # System operations have the highest limits
}

# In-memory token blacklist (should be replaced with Redis in production)
token_blacklist = set()

# In-memory rate limiting (should be replaced with Redis in production)
rate_limit_counters = {}

# Token model
class Token(BaseModel):
    access_token: str
    token_type: str
    expires_at: datetime
    scopes: List[str]

# Token data model
class TokenData(BaseModel):
    username: Optional[str] = None
    scopes: List[str] = []
    exp: Optional[datetime] = None

# User model for authentication
class UserAuth(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None
    scopes: List[str] = []
    hashed_password: str

# Password utilities
def get_password_hash(password: str) -> str:
    """Generate a secure hash of the password"""
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify the password against the hash"""
    return pwd_context.verify(plain_password, hashed_password)

def verify_password_strength(password: str) -> bool:
    """
    Verify password meets strength requirements:
    - At least 12 characters
    - Contains uppercase, lowercase, number, special character
    - Not a common password
    """
    # Basic length check
    if len(password) < 12:
        return False
    
    # Check for complexity
    if not re.search(r'[A-Z]', password):  # Uppercase
        return False
    if not re.search(r'[a-z]', password):  # Lowercase
        return False
    if not re.search(r'[0-9]', password):  # Number
        return False
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):  # Special char
        return False
    
    # Check against common passwords (simplified version - in production use a real list)
    common_passwords = {'password', 'password123', '123456', 'qwerty', 'admin'}
    if password.lower() in common_passwords:
        return False
    
    return True

# User authentication
def authenticate_user(db: Any, username: str, password: str) -> Optional[UserAuth]:
    """
    Authenticate a user by username and password
    
    In production, replace this with a real database lookup
    This is a simplified example using a hypothetical user model
    """
    # Example query to find user
    # user = db.query(User).filter(User.username == username).first()
    
    # Simulated user for demonstration
    user = UserAuth(
        username="test_user",
        email="test@example.com",
        full_name="Test User",
        disabled=False,
        scopes=["user", "trader"],
        hashed_password=get_password_hash("securepassword123!")
    )
    
    if username != user.username:
        return None
    
    if not verify_password(password, user.hashed_password):
        # Log failed login attempt
        logger.warning(f"Failed login attempt for user: {username}")
        return None
    
    return user

# JWT token creation
def create_access_token(
    data: Dict[str, Any], 
    expires_delta: Optional[timedelta] = None
) -> str:
    """Create a JWT access token with an optional expiration time"""
    to_encode = data.copy()
    
    # Set expiration time
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    
    # Create the JWT
    encoded_jwt = jwt.encode(
        to_encode, 
        settings.SECRET_KEY, 
        algorithm=settings.ALGORITHM
    )
    
    return encoded_jwt

# Token blacklisting
def blacklist_token(token: str) -> None:
    """Add a token to the blacklist (for logout)"""
    token_blacklist.add(token)
    
    # In production, use Redis with TTL
    # redis_client.set(f"token_blacklist:{token}", "1", ex=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60)

def is_token_blacklisted(token: str) -> bool:
    """Check if a token is blacklisted"""
    return token in token_blacklist
    
    # In production, use Redis
    # return redis_client.exists(f"token_blacklist:{token}")

# Rate limiting
def check_rate_limit(username: str, scope: str) -> bool:
    """
    Check if the user has exceeded their rate limit
    Returns True if request should be allowed, False if it should be blocked
    """
    current_time = datetime.utcnow().timestamp() // 60  # Current minute
    
    # Create key for rate limiting
    rate_key = f"{username}:{current_time}"
    
    # Get appropriate rate limit based on scope
    rate_limit = RATE_LIMITS.get(scope, DEFAULT_RATE_LIMIT)
    
    # Check and increment counter
    if rate_key not in rate_limit_counters:
        rate_limit_counters[rate_key] = 1
        return True
    
    if rate_limit_counters[rate_key] < rate_limit:
        rate_limit_counters[rate_key] += 1
        return True
    
    # Log rate limit exceeded
    logger.warning(f"Rate limit exceeded for user: {username}, scope: {scope}")
    return False

# Dependency for getting the current authenticated user
async def get_current_user(
    security_scopes: SecurityScopes,
    token: str = Depends(oauth2_scheme),
    db = Depends(get_db)
) -> UserAuth:
    """
    Validates the access token and returns the current user
    Also checks if the token has the required scopes
    """
    if security_scopes.scopes:
        authenticate_value = f'Bearer scope="{security_scopes.scope_str}"'
    else:
        authenticate_value = "Bearer"
    
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": authenticate_value},
    )
    
    # Check if token is blacklisted (logged out)
    if is_token_blacklisted(token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has been revoked",
            headers={"WWW-Authenticate": authenticate_value},
        )
    
    try:
        # Decode the JWT
        payload = jwt.decode(
            token, 
            settings.SECRET_KEY, 
            algorithms=[settings.ALGORITHM]
        )
        
        # Get username from payload
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        
        # Get token scopes
        token_scopes = payload.get("scopes", [])
        
        # Create token data
        token_data = TokenData(
            username=username,
            scopes=token_scopes,
            exp=datetime.fromtimestamp(payload.get("exp"))
        )
        
    except (JWTError, ValidationError):
        logger.warning("Invalid token attempt", exc_info=True)
        raise credentials_exception
    
    # Simulated user retrieval - in production, get from database
    # user = db.query(User).filter(User.username == token_data.username).first()
    user = UserAuth(
        username="test_user",
        email="test@example.com",
        full_name="Test User",
        disabled=False,
        scopes=["user", "trader"],
        hashed_password=get_password_hash("securepassword123!")
    )
    
    if user is None:
        raise credentials_exception
    
    if user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    
    # Check if token has required scopes
    for scope in security_scopes.scopes:
        if scope not in token_data.scopes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Not enough permissions. Required scope: {scope}",
                headers={"WWW-Authenticate": authenticate_value},
            )
    
    # Check rate limits based on maximum scope privilege
    max_scope = "user"  # Default scope
    for scope in ["system", "admin", "analyst", "trader", "user"]:
        if scope in token_data.scopes:
            max_scope = scope
            break
    
    if not check_rate_limit(username, max_scope):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded"
        )
    
    return user

# Dependency for getting the current active user
async def get_current_active_user(
    current_user: UserAuth = Depends(get_current_user)
) -> UserAuth:
    """Verifies that the current user is active"""
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# API key authentication for service-to-service communication
class APIKey(BaseModel):
    key_id: str
    api_key: str
    service_name: str
    scopes: List[str]
    created_at: datetime
    expires_at: Optional[datetime] = None

# In-memory API keys (should be replaced with database in production)
api_keys = {
    "market_data_service": APIKey(
        key_id="market_data_service",
        api_key=get_password_hash("test_api_key"),  # In production, use a strong random key
        service_name="Market Data Service",
        scopes=["system", "analyst"],
        created_at=datetime.utcnow(),
        expires_at=datetime.utcnow() + timedelta(days=365)
    )
}

def verify_api_key(key_id: str, api_key: str) -> Optional[APIKey]:
    """Verify an API key for service-to-service authentication"""
    if key_id not in api_keys:
        return None
        
    stored_key = api_keys[key_id]
    
    # Check if key has expired
    if stored_key.expires_at and stored_key.expires_at < datetime.utcnow():
        return None
    
    # Verify the API key (using password verification for consistency)
    if not verify_password(api_key, stored_key.api_key):
        return None
    
    return stored_key

# Middleware for API key authentication
async def api_key_auth(request: Request) -> Optional[APIKey]:
    """Middleware for API key authentication"""
    # Get API key from header
    key_id = request.headers.get("X-API-Key-ID")
    api_key = request.headers.get("X-API-Key")
    
    if not key_id or not api_key:
        return None
    
    return verify_api_key(key_id, api_key)

# CSRF protection
def generate_csrf_token() -> str:
    """Generate a CSRF token for form protection"""
    return secrets.token_hex(32)

def verify_csrf_token(request_token: str, session_token: str) -> bool:
    """Verify a CSRF token"""
    if not request_token or not session_token:
        return False
    
    # Use constant-time comparison to prevent timing attacks
    return hmac.compare_digest(request_token, session_token)

# Two-factor authentication
def generate_totp_secret() -> str:
    """Generate a secret for TOTP (Time-based One-Time Password)"""
    return base64.b32encode(secrets.token_bytes(20)).decode('utf-8')

def verify_totp(token: str, secret: str) -> bool:
    """
    Verify a TOTP token
    
    Note: This is a placeholder. In a real implementation, use a library like PyOTP
    """
    # Placeholder for actual TOTP verification
    return token == "123456"  # This should be replaced with real verification

# IP-based security
def is_ip_allowed(ip_address: str) -> bool:
    """Check if an IP address is allowed to access the system"""
    # Simplified IP verification - in production, use a more sophisticated approach
    allowed_networks = [
        "10.0.0.0/8",      # Private network
        "172.16.0.0/12",   # Private network
        "192.168.0.0/16",  # Private network
        "127.0.0.1/32"     # Localhost
    ]
    
    # Simplified check - in production, use a real CIDR matcher
    for network in allowed_networks:
        if ip_address.startswith(network.split('/')[0]):
            return True
    
    return False

# Audit logging
def log_auth_event(
    event_type: str, 
    username: str, 
    success: bool, 
    details: Optional[Dict[str, Any]] = None
) -> None:
    """Log an authentication or authorization event"""
    if not details:
        details = {}
    
    # Create audit log entry
    log_entry = {
        "event_type": event_type,
        "username": username,
        "success": success,
        "timestamp": datetime.utcnow().isoformat(),
        "details": details
    }
    
    # Log to application logger
    if success:
        logger.info(f"Auth event: {event_type} for user {username} succeeded", extra=log_entry)
    else:
        logger.warning(f"Auth event: {event_type} for user {username} failed", extra=log_entry)
    
    # In production, also log to database in the AUDIT_SCHEMA
    # with db_session(DatabaseType.POSTGRESQL) as session:
    #     audit_log = AuditLog(
    #         event_type=event_type,
    #         username=username,
    #         success=success,
    #         details=details,
    #         timestamp=datetime.utcnow()
    #     )
    #     session.add(audit_log)

# Login/logout functions to be used in API routes
def login_user(username: str, password: str, db: Any) -> Optional[Token]:
    """Log in a user and generate an access token"""
    user = authenticate_user(db, username, password)
    if not user:
        log_auth_event("login", username, False)
        return None
    
    # Generate token
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "scopes": user.scopes},
        expires_delta=access_token_expires
    )
    
    expiry_time = datetime.utcnow() + access_token_expires
    
    # Log successful login
    log_auth_event("login", username, True)
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_at=expiry_time,
        scopes=user.scopes
    )

def logout_user(token: str, username: str) -> bool:
    """Log out a user by blacklisting their token"""
    try:
        blacklist_token(token)
        log_auth_event("logout", username, True)
        return True
    except Exception as e:
        logger.error(f"Error during logout: {str(e)}", exc_info=True)
        log_auth_event("logout", username, False, {"error": str(e)})
        return False

# Password reset functions
def generate_password_reset_token(username: str) -> str:
    """Generate a token for password reset"""
    # Create a JWT with short expiration (15 minutes)
    expires = timedelta(minutes=15)
    token = create_access_token(
        data={"sub": username, "purpose": "password_reset"},
        expires_delta=expires
    )
    
    # Log password reset request
    log_auth_event("password_reset_request", username, True)
    
    return token

def verify_password_reset_token(token: str) -> Optional[str]:
    """Verify a password reset token and return the username"""
    try:
        payload = jwt.decode(
            token, 
            settings.SECRET_KEY, 
            algorithms=[settings.ALGORITHM]
        )
        
        username = payload.get("sub")
        purpose = payload.get("purpose")
        
        if not username or purpose != "password_reset":
            return None
        
        return username
    except JWTError:
        return None

# Security utility functions
def sanitize_input(input_str: str) -> str:
    """
    Sanitize user input to prevent injection attacks
    This is a simple example - in production, use a proper library
    """
    # Remove dangerous characters
    sanitized = re.sub(r'[<>"\'&;]', '', input_str)
    return sanitized

def secure_headers() -> Dict[str, str]:
    """Return secure headers for HTTP responses"""
    return {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Content-Security-Policy": "default-src 'self'; script-src 'self'",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Referrer-Policy": "no-referrer"
    }