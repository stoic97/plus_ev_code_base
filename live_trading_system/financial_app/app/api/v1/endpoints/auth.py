"""
Authentication endpoints for user login, registration, and token management.
"""

from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from app.core.config import settings
from app.core.database import PostgresDB
from app.core.security import (
    Token,
    UserAuth,
    authenticate_user,
    blacklist_token,
    create_access_token,
    get_current_active_user,
    get_current_user,
    get_password_hash,
    verify_password_strength,
)
from app.schemas.auth import (
    LoginResponse,
    PasswordResetRequest,
    RefreshTokenRequest,
    RegisterRequest,
    RegisterResponse,
    TokenResponse,
    UserResponse,
)

router = APIRouter()

# Get database instance
db = PostgresDB()


@router.post("/token", response_model=TokenResponse)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    OAuth2 compatible token login, get an access token for future requests.
    
    Args:
        form_data: OAuth2 password request form
        
    Returns:
        Access token information
        
    Raises:
        HTTPException: If authentication fails
    """
    with db.session() as session:
        user = authenticate_user(session, form_data.username, form_data.password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        access_token_expires = timedelta(minutes=settings.security.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.username, "scopes": user.scopes},
            expires_delta=access_token_expires
        )
        
        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            expires_at=datetime.utcnow() + access_token_expires,
            scopes=user.scopes
        )


@router.post("/login", response_model=LoginResponse)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Login endpoint with more detailed response.
    
    Args:
        form_data: OAuth2 password request form
        
    Returns:
        User information and access token
        
    Raises:
        HTTPException: If authentication fails
    """
    with db.session() as session:
        user = authenticate_user(session, form_data.username, form_data.password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        access_token_expires = timedelta(minutes=settings.security.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.username, "scopes": user.scopes},
            expires_delta=access_token_expires
        )
        
        return LoginResponse(
            user=UserResponse(
                username=user.username,
                email=user.email,
                full_name=user.full_name,
                scopes=user.scopes,
            ),
            access_token=access_token,
            token_type="bearer",
            expires_at=datetime.utcnow() + access_token_expires
        )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(refresh_request: RefreshTokenRequest):
    """
    Refresh an access token.
    
    Args:
        refresh_request: Token refresh request
        
    Returns:
        New access token
        
    Raises:
        HTTPException: If token refresh fails
    """
    # This is a placeholder implementation
    # In a real implementation, you would validate the refresh token
    # For now, we'll just issue a new token with the same user data
    
    try:
        user = await get_current_user(refresh_request.token)
        
        # Create a new access token
        access_token_expires = timedelta(minutes=settings.security.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.username, "scopes": user.scopes},
            expires_delta=access_token_expires
        )
        
        # Optionally blacklist the old token for security
        blacklist_token(refresh_request.token)
        
        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            expires_at=datetime.utcnow() + access_token_expires,
            scopes=user.scopes
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token for refresh",
            headers={"WWW-Authenticate": "Bearer"},
        )


@router.post("/register", response_model=RegisterResponse)
async def register(register_request: RegisterRequest):
    """
    Register a new user.
    
    Args:
        register_request: User registration data
        
    Returns:
        Result of registration
        
    Raises:
        HTTPException: If registration fails
    """
    # This is a simplified implementation
    # In a real implementation, you would create the user in the database
    
    # Check if password is strong enough
    if not verify_password_strength(register_request.password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Password is not strong enough. It must be at least 12 characters and contain "
                "uppercase letters, lowercase letters, numbers, and special characters."
            ),
        )
    
    # Check if username/email is already taken
    # This would check the database in a real implementation
    # For now, just assume it's available
    
    # Hash the password
    hashed_password = get_password_hash(register_request.password)
    
    # In a real implementation, you would save the user to the database here
    # with the hashed password
    
    return RegisterResponse(
        username=register_request.username,
        email=register_request.email,
        success=True,
        message="User registered successfully"
    )


@router.post("/logout")
async def logout(token: str, current_user: UserAuth = Depends(get_current_active_user)):
    """
    Log out a user by blacklisting their token.
    
    Args:
        token: Current access token
        current_user: Current authenticated user
        
    Returns:
        Logout status
    """
    blacklist_token(token)
    return {"message": "Successfully logged out"}


@router.post("/password-reset-request")
async def request_password_reset(reset_request: PasswordResetRequest):
    """
    Request a password reset.
    
    Args:
        reset_request: Password reset request with email
        
    Returns:
        Status message
    """
    # This is a simplified implementation
    # In a real implementation, you would:
    # 1. Verify the email exists in the database
    # 2. Generate a password reset token
    # 3. Send an email with a link containing the token
    
    # For now, just return a success message
    return {
        "message": "If the email exists in our system, a password reset link has been sent."
    }


@router.get("/me", response_model=UserResponse)
async def get_user_me(current_user: UserAuth = Depends(get_current_active_user)):
    """
    Get information about the current authenticated user.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        User information
    """
    return UserResponse(
        username=current_user.username,
        email=current_user.email,
        full_name=current_user.full_name,
        scopes=current_user.scopes,
    )