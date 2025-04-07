"""
Test settings module for the application.
Provides configuration for test environments.
"""

import os
from typing import Dict, Any

# Ensure we're in test mode
os.environ["TESTING"] = "True"

# Make sure we use testing configuration
def get_test_settings():
    from app.core.config import Settings
    
    class TestSettings(Settings):
        """Test settings override."""
        # Override database settings to use test databases or mocks
        class DBSettings:
            # PostgreSQL settings
            POSTGRES_URI: str = "postgresql://postgres:postgres@localhost:5432/test_db"
            POSTGRES_MIN_CONNECTIONS: int = 1
            POSTGRES_MAX_CONNECTIONS: int = 2
            POSTGRES_STATEMENT_TIMEOUT: int = 30000  # 30 seconds
            
            # Other DB settings...
        
        db: DBSettings = DBSettings()
    
    return TestSettings()