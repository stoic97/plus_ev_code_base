"""
User model and related types for the trading application.

This module defines SQLAlchemy models for user management, authentication,
and access control for the AI-powered trading system.
"""

import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Table, Text, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.core.database import Base
from app.core.security import get_password_hash, verify_password


# Association table for user-role many-to-many relationship
user_roles = Table(
    'user_roles',
    Base.metadata,
    Column('user_id', UUID(as_uuid=True), ForeignKey('users.id'), primary_key=True),
    Column('role_id', Integer, ForeignKey('roles.id'), primary_key=True)
)


class Role(Base):
    """Role model for role-based access control."""
    
    __tablename__ = 'roles'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(50), unique=True, index=True, nullable=False)
    description = Column(String(255))
    
    # Permissions stored as JSON for flexibility
    permissions = Column(JSON, nullable=False, default=dict)
    
    # Relationships
    users = relationship("User", secondary=user_roles, back_populates="roles")
    
    def __repr__(self):
        return f"<Role {self.name}>"


class User(Base):
    """User model for authentication and account management."""
    
    __tablename__ = 'users'
    
    # Basic identification
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(50), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    
    # Personal information
    full_name = Column(String(255))
    
    # Account status
    is_active = Column(Boolean, default=True, nullable=False)
    is_superuser = Column(Boolean, default=False, nullable=False)
    
    # Account timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    last_login_at = Column(DateTime)
    password_changed_at = Column(DateTime, default=datetime.utcnow)
    
    # Login security tracking
    failed_login_attempts = Column(Integer, default=0, nullable=False)
    last_failed_login_at = Column(DateTime)
    lockout_until = Column(DateTime)
    
    # Trading system specific settings
    trading_limits = Column(JSON, default=dict)  # Store position limits, max order size, etc.
    algorithm_access = Column(JSON, default=dict)  # Which algorithms user can access
    environment_access = Column(JSON, default=dict)  # dev/test/prod access flags
    
    # Emergency controls
    has_kill_switch_access = Column(Boolean, default=False, nullable=False)
    emergency_contact = Column(Boolean, default=False, nullable=False)
    
    # Relationships
    roles = relationship("Role", secondary=user_roles, back_populates="users")
    sessions = relationship("UserSession", back_populates="user", cascade="all, delete-orphan")
    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")
    password_resets = relationship("PasswordReset", back_populates="user", cascade="all, delete-orphan")
    audit_logs = relationship("AuditLog", back_populates="user")
    
    def __repr__(self):
        return f"<User {self.username}>"
    
    @property
    def is_locked_out(self) -> bool:
        """Check if the user account is locked out due to failed login attempts."""
        if not self.lockout_until:
            return False
        return datetime.utcnow() < self.lockout_until
    
    def set_password(self, password: str) -> None:
        """Set a new password for the user."""
        self.hashed_password = get_password_hash(password)
        self.password_changed_at = datetime.utcnow()
    
    def verify_password(self, password: str) -> bool:
        """Verify a password against the stored hash."""
        return verify_password(password, self.hashed_password)
    
    def record_successful_login(self) -> None:
        """Record a successful login attempt."""
        self.last_login_at = datetime.utcnow()
        self.failed_login_attempts = 0
        self.lockout_until = None
    
    def record_failed_login(self, max_attempts: int = 5, lockout_minutes: int = 15) -> None:
        """
        Record a failed login attempt and apply lockout policy.
        
        Args:
            max_attempts: Maximum allowed failed attempts before lockout
            lockout_minutes: Duration of lockout in minutes
        """
        now = datetime.utcnow()
        self.last_failed_login_at = now
        self.failed_login_attempts += 1
        
        if self.failed_login_attempts >= max_attempts:
            self.lockout_until = now + timedelta(minutes=lockout_minutes)
    
    def has_permission(self, permission: str) -> bool:
        """
        Check if user has a specific permission through any of their roles.
        
        Args:
            permission: Permission string to check
            
        Returns:
            True if user has the permission, False otherwise
        """
        # Superusers have all permissions
        if self.is_superuser:
            return True
        
        # Check each role for the permission
        for role in self.roles:
            role_permissions = role.permissions.get("permissions", [])
            if permission in role_permissions:
                return True
        
        return False
    
    def has_algorithm_access(self, algorithm_id: str) -> bool:
        """
        Check if user has access to a specific trading algorithm.
        
        Args:
            algorithm_id: The ID of the algorithm to check
            
        Returns:
            True if user has access, False otherwise
        """
        # Superusers have access to all algorithms
        if self.is_superuser:
            return True
            
        # Check specific algorithm access
        algorithms = self.algorithm_access.get("algorithms", [])
        return algorithm_id in algorithms
    
    def can_trade_in_environment(self, environment: str) -> bool:
        """
        Check if user can trade in a specific environment.
        
        Args:
            environment: The environment to check (e.g., 'dev', 'test', 'prod')
            
        Returns:
            True if user can trade in the environment, False otherwise
        """
        # Superusers have access to all environments
        if self.is_superuser:
            return True
            
        # Check environment access
        environments = self.environment_access.get("environments", [])
        return environment in environments
    
    def get_trading_limit(self, instrument_type: str) -> float:
        """
        Get the user's trading limit for a specific instrument type.
        
        Args:
            instrument_type: The type of financial instrument
            
        Returns:
            The trading limit amount
        """
        limits = self.trading_limits.get("limits", {})
        return limits.get(instrument_type, 0.0)
    
    def get_scopes(self) -> List[str]:
        """
        Get all permission scopes from the user's roles.
        These are used for JWT token generation.
        
        Returns:
            List of permission scope strings
        """
        # Basic scope for all active users
        scopes = ["user"]
        
        # Add role-based scopes
        for role in self.roles:
            scopes.append(f"role:{role.name}")
            
            # Add permission-based scopes
            role_permissions = role.permissions.get("permissions", [])
            for permission in role_permissions:
                scopes.append(f"permission:{permission}")
        
        # Add environment access scopes
        environments = self.environment_access.get("environments", [])
        for env in environments:
            scopes.append(f"env:{env}")
            
        # Add algorithm access scopes
        algorithms = self.algorithm_access.get("algorithms", [])
        for algo in algorithms:
            scopes.append(f"algorithm:{algo}")
        
        # Add special scopes
        if self.has_kill_switch_access:
            scopes.append("kill_switch")
            
        if self.emergency_contact:
            scopes.append("emergency_contact")
            
        # Add superuser scope if applicable
        if self.is_superuser:
            scopes.append("superuser")
            
        return scopes
    
    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """
        Convert user model to dictionary for API responses.
        
        Args:
            include_sensitive: Whether to include sensitive fields
            
        Returns:
            Dictionary representation of user
        """
        user_dict = {
            "id": str(self.id),
            "username": self.username,
            "email": self.email,
            "full_name": self.full_name,
            "is_active": self.is_active,
            "is_superuser": self.is_superuser,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_login_at": self.last_login_at.isoformat() if self.last_login_at else None,
            "roles": [role.name for role in self.roles],
            "has_kill_switch_access": self.has_kill_switch_access,
            "emergency_contact": self.emergency_contact
        }
        
        if include_sensitive:
            user_dict.update({
                "failed_login_attempts": self.failed_login_attempts,
                "last_failed_login_at": self.last_failed_login_at.isoformat() if self.last_failed_login_at else None,
                "lockout_until": self.lockout_until.isoformat() if self.lockout_until else None,
                "password_changed_at": self.password_changed_at.isoformat() if self.password_changed_at else None,
                "trading_limits": self.trading_limits,
                "algorithm_access": self.algorithm_access,
                "environment_access": self.environment_access
            })
            
        return user_dict


class UserSession(Base):
    """User session model for tracking active sessions."""
    
    __tablename__ = 'user_sessions'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    
    # Session information
    token_id = Column(String(255), unique=True, nullable=False)  # JWT jti claim
    ip_address = Column(String(45))  # IPv6 can be up to 45 chars
    user_agent = Column(String(255))
    
    # Session validity
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    revoked_at = Column(DateTime)
    is_revoked = Column(Boolean, default=False, nullable=False)
    
    # Environment information (for tracking where this session is active)
    environment = Column(String(50), default="prod")
    is_algorithmic_session = Column(Boolean, default=False)  # True for API/algorithm sessions
    
    # Relationships
    user = relationship("User", back_populates="sessions")
    
    @property
    def is_active(self) -> bool:
        """Check if the session is currently active."""
        now = datetime.utcnow()
        return (
            not self.is_revoked and 
            now < self.expires_at
        )
    
    def revoke(self, reason: str = "User logout") -> None:
        """
        Revoke the session.
        
        Args:
            reason: Reason for session revocation
        """
        self.is_revoked = True
        self.revoked_at = datetime.utcnow()


class APIKey(Base):
    """API key model for automated trading access."""
    
    __tablename__ = 'api_keys'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    
    # API key details
    name = Column(String(100), nullable=False)  # Descriptive name for this API key
    key_prefix = Column(String(10), nullable=False)  # First few chars of the key (for UI display)
    key_hash = Column(String(255), nullable=False)  # Hashed API key
    
    # Validity period
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=True)  # Null means no expiration
    revoked_at = Column(DateTime)
    is_revoked = Column(Boolean, default=False, nullable=False)
    
    # Usage limits and permissions
    permissions = Column(JSON, default=dict)  # Specific permissions for this key
    environment = Column(String(50), default="prod")  # Which environment this key is valid for
    rate_limit = Column(Integer, default=100)  # Requests per minute
    
    # Usage tracking
    last_used_at = Column(DateTime)
    use_count = Column(Integer, default=0)
    
    # Relationships
    user = relationship("User", back_populates="api_keys")
    
    @property
    def is_active(self) -> bool:
        """Check if the API key is currently active."""
        now = datetime.utcnow()
        if self.is_revoked:
            return False
        if self.expires_at and now > self.expires_at:
            return False
        return True
    
    def revoke(self) -> None:
        """Revoke the API key."""
        self.is_revoked = True
        self.revoked_at = datetime.utcnow()
    
    def record_usage(self) -> None:
        """Record usage of the API key."""
        self.last_used_at = datetime.utcnow()
        self.use_count += 1


class PasswordReset(Base):
    """Password reset token tracking."""
    
    __tablename__ = 'password_resets'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    
    # Token information
    token_hash = Column(String(255), nullable=False)  # Hashed reset token
    
    # Validity period
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    used_at = Column(DateTime)
    
    # Usage tracking
    is_used = Column(Boolean, default=False, nullable=False)
    ip_address = Column(String(45))
    user_agent = Column(String(255))
    
    # Relationships
    user = relationship("User", back_populates="password_resets")
    
    @property
    def is_valid(self) -> bool:
        """Check if the password reset token is still valid."""
        now = datetime.utcnow()
        return (
            not self.is_used and 
            now < self.expires_at
        )
    
    def use_token(self, ip_address: Optional[str] = None, user_agent: Optional[str] = None) -> None:
        """
        Mark the token as used.
        
        Args:
            ip_address: IP address where token was used
            user_agent: User agent where token was used
        """
        self.is_used = True
        self.used_at = datetime.utcnow()
        self.ip_address = ip_address
        self.user_agent = user_agent


class AuditLog(Base):
    """Audit log for tracking security-sensitive actions."""
    
    __tablename__ = 'audit_logs'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=True)  # Nullable for system actions
    
    # Action details
    action = Column(String(100), nullable=False)  # Type of action (login, trade, algorithm_deploy, etc.)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    ip_address = Column(String(45))
    user_agent = Column(String(255))
    
    # Target of the action
    target_type = Column(String(50))  # What type of object was acted upon
    target_id = Column(String(255))   # ID of the object acted upon
    
    # Additional details
    environment = Column(String(50))  # dev, test, prod
    details = Column(JSON)  # Any additional action-specific details
    
    # Relationships
    user = relationship("User", back_populates="audit_logs")