"""
Authentication endpoints for user login, registration, and token management.
"""

from datetime import datetime, timedelta
from typing import Optional
import re

from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordRequestForm

from ....core.config import settings
from ....core.database import PostgresDB, get_db
from ....core.security import (
    User,
    Token,
    authenticate_user,
    create_access_token,
    create_refresh_token,
    get_current_active_user,
    get_current_user,
    get_password_hash,
    verify_password,
    log_auth_event,
    create_user,
    Roles,
    refresh_access_token
)
from ....schemas.auth import (
    LoginResponse,
    PasswordResetRequest,
    RefreshTokenRequest,
    RegisterRequest,
    RegisterResponse,
    TokenResponse,
    UserResponse,
)
import logging
logger = logging.getLogger(__name__)

router = APIRouter()

# Get database instance
# db = PostgresDB()


@router.post("/token", response_model=TokenResponse)
async def login_for_access_token(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db_session: PostgresDB = Depends(get_db)
):
    """
    OAuth2 compatible token login, get an access token for future requests.
    
    Args:
        request: Request object
        form_data: OAuth2 password request form
        db_session: Database session
        
    Returns:
        Access token information
        
    Raises:
        HTTPException: If authentication fails
    """
    user = authenticate_user(db_session, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create tokens
    access_token = create_access_token(
        data={"sub": user.username, "roles": user.roles}
    )
    refresh_token = create_refresh_token(
        data={"sub": user.username, "roles": user.roles}
    )
    
    # Map to our token response model
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_at=datetime.utcnow() + timedelta(minutes=settings.security.ACCESS_TOKEN_EXPIRE_MINUTES),
        scopes=user.roles  # Using roles as scopes
    )


@router.post("/login", response_model=LoginResponse)
async def login(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db_session: PostgresDB = Depends(get_db)
):
    """
    Login endpoint with more detailed response.
    
    Args:
        request: Request object
        form_data: OAuth2 password request form
        db_session: Database session
        
    Returns:
        User information and access token
        
    Raises:
        HTTPException: If authentication fails
    """
    print("="*50)
    print("🔥 LOGIN ENDPOINT REACHED!")
    print(f"Username: {form_data.username}")
    print(f"Password length: {len(form_data.password)}")
    print("="*50)
    
    user = authenticate_user(db_session, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create tokens
    access_token = create_access_token(
        data={"sub": user.username, "roles": user.roles}
    )
    
    # Map to our login response model
    return LoginResponse(
        user=UserResponse(
            username=user.username,
            email=user.email,
            full_name=user.full_name,
            scopes=user.roles,  # Using roles as scopes
        ),
        access_token=access_token,
        token_type="bearer",
        expires_at=datetime.utcnow() + timedelta(minutes=settings.security.ACCESS_TOKEN_EXPIRE_MINUTES)
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    request: Request,
    refresh_request: RefreshTokenRequest,
    db_session: PostgresDB = Depends(get_db)
):
    """
    Refresh an access token.
    
    Args:
        request: Request object
        refresh_request: Token refresh request
        db_session: Database session
        
    Returns:
        New access token
        
    Raises:
        HTTPException: If token refresh fails
    """
    try:
        tokens = await refresh_access_token(
            request=request,
            refresh_token=refresh_request.token,
            db=db_session
        )
    
        # Convert to our response model
        return TokenResponse(
            access_token=tokens["access_token"],
            refresh_token=tokens["refresh_token"],
            token_type=tokens["token_type"],
            expires_at=datetime.utcnow() + timedelta(minutes=settings.security.ACCESS_TOKEN_EXPIRE_MINUTES),
            scopes=[]  # We would need to decode the token to get roles
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token for refresh",
            headers={"WWW-Authenticate": "Bearer"},
        )

@router.post("/register", response_model=RegisterResponse)
async def register(
    request: Request,
    register_request: RegisterRequest,
    db_session: PostgresDB = Depends(get_db)
):
    """
    Register a new user with comprehensive validation.
    """
    try:
        # First, let Pydantic validate basic requirements
        # The schema will already check min length, email format, etc.
        
        # Perform additional custom password strength check
        if not verify_password_strength(register_request.password):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "Password does not meet complexity requirements. "
                    "Must contain uppercase, lowercase, numbers, and special characters. "
                    "Minimum length is 12 characters."
                )
            )
        
        # Existing user creation logic
        success = create_user(
            db=db_session,
            username=register_request.username,
            email=register_request.email,
            password=register_request.password,
            full_name=register_request.full_name,
            roles=[Roles.OBSERVER]
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User registration failed. Username or email might already exist."
            )
        
        return RegisterResponse(
            username=register_request.username,
            email=register_request.email,
            success=True,
            message="User registered successfully"
        )
    
    except Exception as e:
        # Log the error for debugging
        logger.error(f"Registration error: {str(e)}")
        
        # Re-raise with appropriate status code
        if isinstance(e, HTTPException):
            raise
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An unexpected error occurred during registration"
            )


@router.post("/logout")
async def logout(
    request: Request,
    current_user: User = Depends(get_current_active_user),
    db_session: PostgresDB = Depends(get_db)
):
    """
    Log out a user securely.
    """
    try:
        # Log logout event
        log_auth_event(
            db=db_session,
            event_type="logout",
            username=current_user.username,
            success=True,
            client_ip=request.client.host
        )
        
        # Optional: Implement token blacklisting for enhanced security
        # This is a placeholder for a more robust token invalidation mechanism
        
        return {"message": "Successfully logged out"}
    
    except Exception as e:
        logger.error(f"Logout error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )

@router.get("/me", response_model=UserResponse)
async def get_user_me(
    current_user: User = Depends(get_current_active_user)
):
    """
    Get authenticated user's information securely.
    """
    return UserResponse(
        username=current_user.username,
        email=current_user.email,
        full_name=current_user.full_name,
        scopes=current_user.roles
    )

@router.post("/password-reset-request")
async def request_password_reset(
    request: Request,
    reset_request: PasswordResetRequest,
    db_session: PostgresDB = Depends(get_db)
):
    """
    Request a password reset.
    
    Args:
        request: Request object
        reset_request: Password reset request with email
        db_session: Database session
        
    Returns:
        Status message
    """
    # This is a simplified implementation
    # In a real implementation, you would:
    # 1. Verify the email exists in the database
    # 2. Generate a password reset token
    # 3. Send an email with a link containing the token
    
    # Log the attempt
    log_auth_event(
        db=db_session,
        event_type="password_reset_request",
        username=reset_request.email,  # Using email as username for this log
        success=True,
        client_ip=request.client.host
    )
    
    # For now, just return a success message
    return {
        "message": "If the email exists in our system, a password reset link has been sent."
    }



# Helper functions
def verify_password_strength(password: str) -> bool:
    """
    Verify password meets comprehensive strength requirements.
    
    Args:
        password: Password to check
        
    Returns:
        True if password is strong enough, False otherwise
    
    Checks:
    - Minimum length of 12 characters
    - Contains uppercase letters
    - Contains lowercase letters
    - Contains digits
    - Contains special characters
    - Prevents common weak patterns
    """
    # Minimum length check
    if len(password) < 12:
        return False
    
    # Complexity checks using regex
    has_upper = bool(re.search(r'[A-Z]', password))
    has_lower = bool(re.search(r'[a-z]', password))
    has_digit = bool(re.search(r'\d', password))
    has_special = bool(re.search(r'[!@#$%^&*(),.?":{}|<>]', password))
    
    # Expanded list of common weak patterns
    common_patterns = [
        'password', 
        '123456', 
        'qwerty', 
        'admin', 
        'letmein',
        'welcome',
        '12345678',
        'football',
        '123123',
        'dragon',
        'baseball',
        'abc123',
        'monkey',
        'master',
        'sunshine',
        'iloveyou'
    ]
    
    # Check for common patterns (case-insensitive)
    if any(pattern in password.lower() for pattern in common_patterns):
        return False
    
    # Ensure all complexity criteria are met
    complexity_criteria = [
        has_upper, 
        has_lower, 
        has_digit, 
        has_special
    ]
    
    return all(complexity_criteria)