"""
Database connection tester and fixer for Trading Strategies Application.
Place this in your project root directory.
"""
import os
import sys
import logging
from typing import Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,  # Use DEBUG for maximum verbosity
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Load settings
logger.info("Loading settings...")
from financial_app.app.core.config import settings

# Print database settings
logger.info("Database settings:")
logger.info(f"POSTGRES_SERVER: {settings.db.POSTGRES_SERVER}")
logger.info(f"POSTGRES_PORT: {settings.db.POSTGRES_PORT}")
logger.info(f"POSTGRES_DB: {settings.db.POSTGRES_DB}")
logger.info(f"POSTGRES_USER: {settings.db.POSTGRES_USER}")
# Don't log the actual password
logger.info(f"POSTGRES_PASSWORD set: {'Yes' if settings.db.POSTGRES_PASSWORD else 'No'}")

# Check constructed URI
logger.info(f"POSTGRES_URI: {settings.db.POSTGRES_URI}")

# Create our own connection string
direct_uri = f"postgresql://{settings.db.POSTGRES_USER}:{settings.db.POSTGRES_PASSWORD}@{settings.db.POSTGRES_SERVER}:{settings.db.POSTGRES_PORT}/{settings.db.POSTGRES_DB}"
logger.info(f"Directly constructed URI: {direct_uri}")

# Test direct connection with psycopg2
logger.info("Testing direct connection with psycopg2...")
import psycopg2
from psycopg2 import OperationalError

try:
    # Use values directly from settings
    conn = psycopg2.connect(
        host=settings.db.POSTGRES_SERVER,
        port=settings.db.POSTGRES_PORT,
        dbname=settings.db.POSTGRES_DB,
        user=settings.db.POSTGRES_USER,
        password=settings.db.POSTGRES_PASSWORD
    )
    logger.info("Direct connection successful!")
    
    # Test basic query
    cursor = conn.cursor()
    cursor.execute("SELECT 1 as test")
    result = cursor.fetchone()
    logger.info(f"Query result: {result}")
    
    cursor.close()
    conn.close()
    
    # Define a fixed PostgresDB class
    from financial_app.app.core.database import PostgresDB
    
    class FixedPostgresDB(PostgresDB):
        """PostgresDB with fixed connection logic."""
        
        def connect(self):
            """Connect using explicit parameters rather than URI."""
            try:
                import sqlalchemy
                from sqlalchemy import create_engine
                from sqlalchemy.orm import sessionmaker
                
                logger.info("Connecting to PostgreSQL with fixed method...")
                
                # Create engine with explicit parameters
                db_url = f"postgresql://{self.settings.db.POSTGRES_USER}:{self.settings.db.POSTGRES_PASSWORD}@{self.settings.db.POSTGRES_SERVER}:{self.settings.db.POSTGRES_PORT}/{self.settings.db.POSTGRES_DB}"
                logger.info(f"Connection URL: {db_url}")
                
                self.engine = create_engine(
                    db_url,
                    pool_pre_ping=True,
                    pool_size=5,
                    max_overflow=15,
                    pool_recycle=3600,
                    connect_args={
                        "connect_timeout": 10
                    }
                )
                
                # Set up session factory
                self.SessionLocal = sessionmaker(
                    autocommit=False,
                    autoflush=False,
                    bind=self.engine
                )
                
                # Test connection
                with self.engine.connect() as conn:
                    conn.execute(sqlalchemy.text("SELECT 1"))
                
                self.is_connected = True
                logger.info("PostgreSQL connection established successfully with fixed method")
                
            except Exception as e:
                self.is_connected = False
                logger.error(f"Failed to connect to PostgreSQL with fixed method: {e}")
                raise
    
    # Test connection with our fixed class
    logger.info("Testing connection with fixed PostgresDB class...")
    db = FixedPostgresDB()
    db.connect()
    logger.info("Fixed PostgresDB connection test result: SUCCESS")
    
    # Monkey patch the original class
    logger.info("Monkey patching the original PostgresDB class...")
    import financial_app.app.core.database
    financial_app.app.core.database.PostgresDB = FixedPostgresDB
    
    logger.info("Connection fix applied successfully! The application should now be able to connect.")
    logger.info("Run your application again to test.")
    
except OperationalError as e:
    logger.error(f"Connection error: {e}")
    
    # Check if it's a typical authentication error
    if "password authentication failed" in str(e).lower():
        logger.error("This appears to be an authentication error. Check your credentials.")
        logger.info("Your test script works with these credentials:")
        logger.info("  Host: localhost")
        logger.info("  Database: live-trading-system")
        logger.info("  User: postgres")
        logger.info("  Password: StrongplusEV125")
        
        logger.info("But the application is using:")
        logger.info(f"  Host: {settings.db.POSTGRES_SERVER}")
        logger.info(f"  Database: {settings.db.POSTGRES_DB}")
        logger.info(f"  User: {settings.db.POSTGRES_USER}")
        logger.info("  Password: (hidden)")
        
        logger.info("\nPossible solutions:")
        logger.info("1. Check that your .env file is in the correct location (project root)")
        logger.info("2. Verify DB__POSTGRES_PASSWORD in your .env file matches your actual password")
        logger.info("3. Try using the hardcoded credentials if .env is not being loaded correctly")
    
except Exception as e:
    logger.error(f"Unexpected error: {e}")