"""
Security module for the FinTradeX trading platform.

Provides authentication, authorization, password management, and session handling
for institutional users. Designed for a small team of trusted users with
different permission levels for trading operations.

Features:
- JWT-based authentication with access and refresh tokens
- Role-based access control for different trading functions
- Password hashing and verification using bcrypt
- User management functions for small teams
- Audit logging for security events
- IP-based access restrictions
"""

import datetime
import ipaddress
import logging
from typing import Dict, List, Optional, Union, Any

from fastapi import Depends, HTTPException, Request, Security, status, APIRouter
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

from app.core.config import get_settings
from app.core.database import PostgresDB, get_db


# Create API router for authentication endpoints
# router = APIRouter(tags=["authentication"], prefix="/auth")

# Set up logging
logger = logging.getLogger(__name__)

# Initialize settings
settings = get_settings()


#################################################
# Data Models
#################################################

class User(BaseModel):
    """User model for authentication and authorization."""
    username: str
    email: str
    full_name: Optional[str] = None
    disabled: bool = False
    roles: List[str] = []


class Token(BaseModel):
    """JWT token response model."""
    access_token: str
    refresh_token: str
    token_type: str
    

class TokenData(BaseModel):
    """Data extracted from JWT token."""
    username: Optional[str] = None
    roles: List[str] = []
    exp: Optional[datetime.datetime] = None


class Roles:
    """Role definitions for role-based access control."""
    ADMIN = "admin"              # System administration, user management
    TRADER = "trader"            # Execute trades, manage positions 
    ANALYST = "analyst"          # View market data, run analysis
    RISK_MANAGER = "risk_manager" # Monitor risk, set trading limits
    OBSERVER = "observer"        # Read-only access to all data
    
    # Role groupings for convenience
    ALL_ROLES = [ADMIN, TRADER, ANALYST, RISK_MANAGER, OBSERVER]
    TRADING_ROLES = [ADMIN, TRADER]
    ANALYSIS_ROLES = [ADMIN, TRADER, ANALYST, RISK_MANAGER]
    MONITORING_ROLES = ALL_ROLES


#################################################
# Security Configuration
#################################################

# Password handling configuration
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 configuration with token URL
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token")


#################################################
# Password Management Functions
#################################################

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against its hash.
    
    Args:
        plain_password: Plain text password
        hashed_password: Hashed password from database
        
    Returns:
        True if password matches hash, False otherwise
    """
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """
    Generate a password hash using bcrypt.
    
    Args:
        password: Plain text password
        
    Returns:
        Hashed password
    """
    return pwd_context.hash(password)


#################################################
# JWT Token Functions
#################################################

def create_access_token(data: dict, expires_delta: Optional[datetime.timedelta] = None) -> str:
    """
    Create a new JWT access token.
    
    Args:
        data: Dictionary of data to encode in token
        expires_delta: Optional custom expiration time
        
    Returns:
        Encoded JWT token string
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.datetime.utcnow() + expires_delta
    else:
        expire = datetime.datetime.utcnow() + datetime.timedelta(
            minutes=settings.security.ACCESS_TOKEN_EXPIRE_MINUTES
        )
    
    to_encode.update({"exp": expire})
    return jwt.encode(
        to_encode, 
        settings.security.SECRET_KEY, 
        algorithm=settings.security.ALGORITHM
    )


def create_refresh_token(data: dict) -> str:
    """
    Create a new JWT refresh token with longer expiry.
    
    Args:
        data: Dictionary of data to encode in token
        
    Returns:
        Encoded JWT refresh token string
    """
    to_encode = data.copy()
    expire = datetime.datetime.utcnow() + datetime.timedelta(
        days=settings.security.REFRESH_TOKEN_EXPIRE_DAYS
    )
    to_encode.update({"exp": expire})
    return jwt.encode(
        to_encode, 
        settings.security.SECRET_KEY, 
        algorithm=settings.security.ALGORITHM
    )


#################################################
# User Authentication Functions
#################################################

def get_user(db: PostgresDB, username: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve user from database by username.
    
    Args:
        db: Database connection
        username: Username to lookup
        
    Returns:
        User dict if found, None otherwise
    """
    with db.session() as session:
        user_data = session.execute(
            """
            SELECT username, email, full_name, hashed_password, disabled, roles 
            FROM users WHERE username = :username
            """,
            {"username": username}
        ).fetchone()
        
        if user_data:
            # Convert roles from database format (potentially comma-delimited string)
            # to a list of strings
            roles = user_data.roles.split(",") if user_data.roles else []
            
            # Create user dict with all needed fields
            user_dict = {
                "username": user_data.username,
                "email": user_data.email,
                "full_name": user_data.full_name,
                "disabled": user_data.disabled,
                "hashed_password": user_data.hashed_password,
                "roles": roles
            }
            return user_dict
    return None


def authenticate_user(db: PostgresDB, username: str, password: str) -> Optional[User]:
    """
    Authenticate a user by username and password.
    
    Args:
        db: Database connection
        username: Username to authenticate
        password: Password to verify
        
    Returns:
        User object if authentication successful, None otherwise
    """
    user = get_user(db, username)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    
    # Remove hashed_password before returning user object
    user_dict = {k: v for k, v in user.items() if k != "hashed_password"}
    return User(**user_dict)


#################################################
# JWT Token Verification
#################################################

async def get_current_user(
    request: Request,
    db: PostgresDB = Depends(get_db),
    token: str = Depends(oauth2_scheme)
) -> User:
    """
    Extract and validate user from JWT token.
    
    Args:
        request: FastAPI request object
        db: Database connection
        token: JWT token from Authorization header
        
    Returns:
        User object if token is valid
        
    Raises:
        HTTPException: If token is invalid or user not found
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Check IP restriction if configured
        client_ip = request.client.host
        if not is_allowed_ip(client_ip):
            log_auth_event(
                db, 
                "ip_restricted", 
                "unknown", 
                False, 
                client_ip, 
                "Access attempt from restricted IP"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied from your IP address"
            )
        
        # Decode JWT token
        payload = jwt.decode(
            token, 
            settings.security.SECRET_KEY, 
            algorithms=[settings.security.ALGORITHM]
        )
        username = payload.get("sub")
        if username is None:
            raise credentials_exception
        
        # Extract token data
        token_data = TokenData(
            username=username,
            roles=payload.get("roles", []),
            exp=datetime.datetime.fromtimestamp(payload.get("exp"))
        )
    except jwt.JWTError:
        # Convert to HTTPException
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Get user from database
    user_data = get_user(db, username=token_data.username)
    if user_data is None:
        log_auth_event(
            db, 
            "user_not_found", 
            username, 
            False, 
            client_ip, 
            "User from token not found in database"
        )
        raise credentials_exception
    
    # Create a User object from user data, excluding the password hash
    user = User(**{k: v for k, v in user_data.items() if k != "hashed_password"})
    
    # Log successful authentication
    log_auth_event(
        db, 
        "token_validated", 
        username, 
        True, 
        client_ip
    )
    
    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Ensure user is active (not disabled).
    
    Args:
        current_user: User from token
        
    Returns:
        User object if user is active
        
    Raises:
        HTTPException: If user is disabled
    """
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


#################################################
# Role-Based Access Control
#################################################

def has_role(required_roles: List[str]):
    """
    Dependency for role-based access control.
    
    Args:
        required_roles: List of roles that are allowed
        
    Returns:
        Dependency function that validates user roles
    """
    async def role_checker(
        request: Request,
        current_user: User = Depends(get_current_user)
    ):
        for role in required_roles:
            if role in current_user.roles:
                return current_user
        
        # Log failed authorization
        log_auth_event(
            db=PostgresDB(settings=get_settings()),  # Create new connection
            event_type="authorization_failed",
            username=current_user.username,
            success=False,
            client_ip=request.client.host,
            details=f"Required roles: {required_roles}, User roles: {current_user.roles}"
        )
        
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )
    return role_checker


#################################################
# IP Restriction
#################################################

def is_allowed_ip(client_ip: str) -> bool:
    """
    Check if client IP is in allowed ranges.
    
    Args:
        client_ip: Client IP address
        
    Returns:
        True if IP is allowed, False otherwise
    """
    # If no IP restrictions configured, allow all
    if not hasattr(settings.security, 'ALLOWED_IP_RANGES'):
        return True
        
    allowed_ranges = settings.security.ALLOWED_IP_RANGES
    
    # If no ranges defined, allow all
    if not allowed_ranges:
        return True
    
    try:
        client_ip_obj = ipaddress.ip_address(client_ip)
        
        # Check each allowed range
        for ip_range in allowed_ranges:
            network = ipaddress.ip_network(ip_range, strict=False)
            if client_ip_obj in network:
                return True
                
        # IP not in any allowed range
        return False
    except ValueError:
        # Invalid IP address format
        logger.error(f"Invalid IP address format: {client_ip}")
        return False


#################################################
# Audit Logging
#################################################

def log_auth_event(
    db: PostgresDB,
    event_type: str,
    username: str,
    success: bool,
    client_ip: str,
    details: Optional[str] = None
) -> None:
    """
    Log an authentication or authorization event.
    
    Args:
        db: Database connection
        event_type: Type of event (login, logout, etc.)
        username: Username involved in event
        success: Whether event was successful
        client_ip: Client IP address
        details: Additional details about event
    """
    try:
        with db.session() as session:
            session.execute(
                """
                INSERT INTO auth_log 
                (event_type, username, success, client_ip, details, timestamp) 
                VALUES 
                (:event_type, :username, :success, :client_ip, :details, NOW())
                """,
                {
                    "event_type": event_type,
                    "username": username,
                    "success": success,
                    "client_ip": client_ip,
                    "details": details
                }
            )
            session.commit()
    except Exception as e:
        logger.error(f"Error logging auth event: {e}")


#################################################
# User Management Functions
#################################################

def create_user(
    db: PostgresDB, 
    username: str, 
    email: str, 
    password: str, 
    full_name: Optional[str] = None,
    roles: List[str] = []
) -> bool:
    """
    Create a new user in the database.
    
    Args:
        db: Database connection
        username: Username for new user
        email: Email address for new user
        password: Plain text password (will be hashed)
        full_name: Optional full name
        roles: List of role strings
        
    Returns:
        True if user created successfully, False otherwise
    """
    hashed_password = get_password_hash(password)
    
    try:
        with db.session() as session:
            # Check if user already exists
            existing = session.execute(
                "SELECT 1 FROM users WHERE username = :username",
                {"username": username}
            ).fetchone()
            
            if existing:
                logger.warning(f"Attempted to create user that already exists: {username}")
                return False
                
            # Insert new user
            session.execute(
                """
                INSERT INTO users (username, email, full_name, hashed_password, roles) 
                VALUES (:username, :email, :full_name, :hashed_password, :roles)
                """,
                {
                    "username": username,
                    "email": email,
                    "full_name": full_name,
                    "hashed_password": hashed_password,
                    "roles": ",".join(roles) if roles else ""
                }
            )
            session.commit()
            logger.info(f"Created new user: {username}")
            return True
    except Exception as e:
        logger.error(f"Error creating user: {e}")
        return False


def update_user_roles(
    db: PostgresDB, 
    username: str, 
    roles: List[str]
) -> bool:
    """
    Update a user's roles.
    
    Args:
        db: Database connection
        username: Username to update
        roles: New list of roles
        
    Returns:
        True if update successful, False otherwise
    """
    try:
        with db.session() as session:
            result = session.execute(
                """
                UPDATE users 
                SET roles = :roles
                WHERE username = :username
                """,
                {
                    "username": username,
                    "roles": ",".join(roles) if roles else ""
                }
            )
            session.commit()
            
            if result.rowcount > 0:
                logger.info(f"Updated roles for user {username}: {roles}")
                return True
            else:
                logger.warning(f"Attempted to update roles for non-existent user: {username}")
                return False
    except Exception as e:
        logger.error(f"Error updating user roles: {e}")
        return False


def disable_user(
    db: PostgresDB, 
    username: str
) -> bool:
    """
    Disable a user account.
    
    Args:
        db: Database connection
        username: Username to disable
        
    Returns:
        True if user disabled successfully, False otherwise
    """
    try:
        with db.session() as session:
            result = session.execute(
                """
                UPDATE users 
                SET disabled = TRUE
                WHERE username = :username
                """,
                {"username": username}
            )
            session.commit()
            
            if result.rowcount > 0:
                logger.info(f"Disabled user: {username}")
                return True
            else:
                logger.warning(f"Attempted to disable non-existent user: {username}")
                return False
    except Exception as e:
        logger.error(f"Error disabling user: {e}")
        return False


def enable_user(
    db: PostgresDB, 
    username: str
) -> bool:
    """
    Enable a disabled user account.
    
    Args:
        db: Database connection
        username: Username to enable
        
    Returns:
        True if user enabled successfully, False otherwise
    """
    try:
        with db.session() as session:
            result = session.execute(
                """
                UPDATE users 
                SET disabled = FALSE
                WHERE username = :username
                """,
                {"username": username}
            )
            session.commit()
            
            if result.rowcount > 0:
                logger.info(f"Enabled user: {username}")
                return True
            else:
                logger.warning(f"Attempted to enable non-existent user: {username}")
                return False
    except Exception as e:
        logger.error(f"Error enabling user: {e}")
        return False


def change_password(
    db: PostgresDB,
    username: str,
    current_password: str,
    new_password: str
) -> bool:
    """
    Change a user's password.
    
    Args:
        db: Database connection
        username: Username to update
        current_password: Current password for verification
        new_password: New password to set
        
    Returns:
        True if password changed successfully, False otherwise
    """
    # First authenticate with current password
    user = authenticate_user(db, username, current_password)
    if not user:
        logger.warning(f"Failed password change attempt for user: {username}")
        return False
    
    # Hash the new password
    new_hashed_password = get_password_hash(new_password)
    
    try:
        with db.session() as session:
            session.execute(
                """
                UPDATE users 
                SET hashed_password = :hashed_password
                WHERE username = :username
                """,
                {
                    "username": username,
                    "hashed_password": new_hashed_password
                }
            )
            session.commit()
            logger.info(f"Password changed for user: {username}")
            return True
    except Exception as e:
        logger.error(f"Error changing password: {e}")
        return False


def admin_reset_password(
    db: PostgresDB,
    admin_user: User,
    target_username: str,
    new_password: str
) -> bool:
    """
    Admin function to reset a user's password.
    
    Args:
        db: Database connection
        admin_user: Admin user performing the reset
        target_username: Username to update
        new_password: New password to set
        
    Returns:
        True if password reset successfully, False otherwise
    """
    # Verify admin has appropriate role
    if Roles.ADMIN not in admin_user.roles:
        logger.warning(f"Non-admin user {admin_user.username} attempted to reset password for {target_username}")
        return False
    
    # Hash the new password
    new_hashed_password = get_password_hash(new_password)
    
    try:
        with db.session() as session:
            result = session.execute(
                """
                UPDATE users 
                SET hashed_password = :hashed_password
                WHERE username = :username
                """,
                {
                    "username": target_username,
                    "hashed_password": new_hashed_password
                }
            )
            session.commit()
            
            if result.rowcount > 0:
                logger.info(f"Admin {admin_user.username} reset password for user: {target_username}")
                return True
            else:
                logger.warning(f"Admin {admin_user.username} attempted to reset password for non-existent user: {target_username}")
                return False
    except Exception as e:
        logger.error(f"Error resetting password: {e}")
        return False


#################################################
# FastAPI Endpoints (to be included in routes)
#################################################

# @router.post("/token", response_model=Token)
async def login_for_access_token(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: PostgresDB = Depends(get_db)
):
    """
    Generate JWT tokens from username and password.
    
    Args:
        request: FastAPI request object
        form_data: Username and password form data
        db: Database connection
        
    Returns:
        Token object with access and refresh tokens
        
    Raises:
        HTTPException: If authentication fails
    """
    client_ip = request.client.host
    
    # Check IP restriction
    if not is_allowed_ip(client_ip):
        log_auth_event(
            db, 
            "login_ip_restricted", 
            form_data.username, 
            False, 
            client_ip, 
            "Login attempt from restricted IP"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied from your IP address"
        )
    
    # Authenticate user
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        log_auth_event(
            db, 
            "login_failed", 
            form_data.username, 
            False, 
            client_ip, 
            "Incorrect username or password"
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check if user is disabled
    if user.disabled:
        log_auth_event(
            db, 
            "login_disabled", 
            form_data.username, 
            False, 
            client_ip, 
            "Account disabled"
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User account is disabled"
        )
    
    # Create access and refresh tokens
    access_token = create_access_token(
        data={"sub": user.username, "roles": user.roles}
    )
    refresh_token = create_refresh_token(
        data={"sub": user.username, "roles": user.roles}
    )
    
    # Log successful login
    log_auth_event(
        db, 
        "login_success", 
        user.username, 
        True, 
        client_ip
    )
    
    # Update last login timestamp
    with db.session() as session:
        session.execute(
            "UPDATE users SET last_login = NOW() WHERE username = :username",
            {"username": user.username}
        )
        session.commit()
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer"
    }


# @router.post("/refresh-token", response_model=Token)
async def refresh_access_token(
    request: Request,
    refresh_token: str,
    db: PostgresDB = Depends(get_db)
):
    """
    Generate new access token using refresh token.
    
    Args:
        request: FastAPI request object
        refresh_token: Current refresh token
        db: Database connection
        
    Returns:
        Token object with new access and refresh tokens
        
    Raises:
        HTTPException: If refresh token is invalid
    """
    client_ip = request.client.host
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    # Check IP restriction
    if not is_allowed_ip(client_ip):
        log_auth_event(
            db, 
            "refresh_ip_restricted", 
            "unknown", 
            False, 
            client_ip, 
            "Token refresh attempt from restricted IP"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied from your IP address"
        )
    
    try:
        # Decode refresh token
        payload = jwt.decode(
            refresh_token, 
            settings.security.SECRET_KEY, 
            algorithms=[settings.security.ALGORITHM]
        )
        username = payload.get("sub")
        if username is None:
            raise credentials_exception
        
        # Check token expiration
        exp = payload.get("exp")
        if not exp or datetime.datetime.fromtimestamp(exp) < datetime.datetime.utcnow():
            log_auth_event(
                db, 
                "refresh_token_expired", 
                username, 
                False, 
                client_ip,
                "Expired refresh token"
            )
            raise credentials_exception
        
        # Get user to verify it still exists and is active
        user_data = get_user(db, username)
        if not user_data:
            log_auth_event(
                db, 
                "refresh_user_not_found", 
                username, 
                False, 
                client_ip,
                "User from refresh token not found"
            )
            raise credentials_exception
            
        if user_data.get("disabled", False):
            log_auth_event(
                db, 
                "refresh_user_disabled", 
                username, 
                False, 
                client_ip,
                "User account is disabled"
            )
            raise credentials_exception
        
        # Create new tokens
        roles = user_data.get("roles", [])
        new_access_token = create_access_token(
            data={"sub": username, "roles": roles}
        )
        new_refresh_token = create_refresh_token(
            data={"sub": username, "roles": roles}
        )
        
        # Log successful token refresh
        log_auth_event(
            db, 
            "refresh_token_success", 
            username, 
            True, 
            client_ip
        )
        
        return {
            "access_token": new_access_token,
            "refresh_token": new_refresh_token,
            "token_type": "bearer"
        }
    except JWTError:
        log_auth_event(
            db, 
            "refresh_token_invalid", 
            "unknown", 
            False, 
            client_ip,
            "Invalid refresh token"
        )
        raise credentials_exception

#################################################
# Database Schema (for reference)
#################################################

"""
-- SQL Schema for security tables (place in migrations)
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    full_name VARCHAR(100),
    hashed_password VARCHAR(100) NOT NULL,
    disabled BOOLEAN DEFAULT FALSE,
    roles TEXT,  -- Comma-separated list of roles
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_login TIMESTAMP WITH TIME ZONE
);

CREATE TABLE auth_log (
    id SERIAL PRIMARY KEY,
    event_type VARCHAR(50) NOT NULL,  -- 'login', 'logout', 'permission_denied', etc.
    username VARCHAR(50) NOT NULL,
    success BOOLEAN NOT NULL,
    client_ip VARCHAR(50),
    details TEXT,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX auth_log_username_idx ON auth_log(username);
CREATE INDEX auth_log_timestamp_idx ON auth_log(timestamp);
"""