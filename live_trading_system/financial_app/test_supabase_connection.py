# file: test_supabase_connection.py

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import psycopg2
from urllib.parse import urlparse

# Add the financial_app directory to the path
current_dir = Path(__file__).parent

# Load environment variables from .env file
env_path = current_dir / '.env'
sys.path.insert(0, str(current_dir))

# Check if the .env file exists

if env_path.exists():
    load_dotenv(env_path)
    print(f"Loaded environment from: {env_path}")
else:
    print(f"Warning: .env file not found at {env_path}")

def test_psycopg2_connection():
    """Test direct psycopg2 connection to debug SSL issues"""
    print("\n=== Testing psycopg2 Direct Connection ===")
    
    # Get environment variables
    db_server = os.getenv('DB__POSTGRES_SERVER', 'db.hwzljrsmrudtmaitfzbr.supabase.co')
    db_port = os.getenv('DB__POSTGRES_PORT', '5432')
    db_user = os.getenv('DB__POSTGRES_USER', 'postgres')
    db_password = os.getenv('DB__POSTGRES_PASSWORD')
    db_name = os.getenv('DB__POSTGRES_DB', 'postgres')
    
    if not db_password:
        print("ERROR: DB__POSTGRES_PASSWORD not found in environment")
        return
    
    try:
        # Connect with psycopg2 directly
        conn_params = {
            'host': db_server,
            'port': db_port,
            'user': db_user,
            'password': db_password,
            'database': db_name,
            'sslmode': 'require'
        }
        
        print(f"Connecting to {db_server}:{db_port} as {db_user}")
        conn = psycopg2.connect(**conn_params)
        cursor = conn.cursor()
        
        cursor.execute('SELECT version()')
        version = cursor.fetchone()[0]
        print(f"Success! PostgreSQL version: {version}")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"psycopg2 connection failed: {e}")

def test_sqlalchemy_connection():
    """Test SQLAlchemy connection with proper SSL"""
    print("\n=== Testing SQLAlchemy Connection ===")
    
    # Get environment variables
    db_server = os.getenv('DB__POSTGRES_SERVER', 'db.hwzljrsmrudtmaitfzbr.supabase.co')
    db_port = os.getenv('DB__POSTGRES_PORT', '5432')
    db_user = os.getenv('DB__POSTGRES_USER', 'postgres')
    db_password = os.getenv('DB__POSTGRES_PASSWORD')
    db_name = os.getenv('DB__POSTGRES_DB', 'postgres')
    
    if not db_password:
        print("ERROR: DB__POSTGRES_PASSWORD not found in environment")
        return
    
    # Build connection URL
    db_uri = f"postgresql://{db_user}:{db_password}@{db_server}:{db_port}/{db_name}"
    
    print(f"Connecting to: {db_server}:{db_port}")
    
    try:
        # Create engine with SSL
        engine = create_engine(
            db_uri,
            echo=False,
            connect_args={'sslmode': 'require'}
        )
        
        # Test connection
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version()"))
            version = result.fetchone()[0]
            print(f"Success! PostgreSQL version: {version}")
            
            # Check current time
            result = conn.execute(text("SELECT NOW()"))
            current_time = result.fetchone()[0]
            print(f"Server time: {current_time}")
        
        engine.dispose()
        
    except Exception as e:
        print(f"SQLAlchemy connection failed: {e}")

def test_environment_variables():
    """Check if environment variables are loaded correctly"""
    print("=== Environment Variables Check ===")
    
    env_vars = [
        'DB__POSTGRES_SERVER',
        'DB__POSTGRES_PORT',
        'DB__POSTGRES_USER',
        'DB__POSTGRES_PASSWORD',
        'DB__POSTGRES_DB',
        'DB__POSTGRES_URI'
    ]
    
    for var in env_vars:
        value = os.getenv(var)
        if value:
            if 'PASSWORD' in var:
                print(f"{var}: {'*' * 10} (exists)")
            else:
                print(f"{var}: {value}")
        else:
            print(f"{var}: NOT FOUND")

def test_uri_from_env():
    """Test using the complete URI from environment"""
    print("\n=== Testing URI from Environment ===")
    
    db_uri = os.getenv('DB__POSTGRES_URI')
    if not db_uri:
        print("ERROR: DB__POSTGRES_URI not found in environment")
        return
    
    try:
        # Create engine from URI
        engine = create_engine(db_uri, echo=False)
        
        # Test connection
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            print(f"Success! Connection works with URI from environment")
        
        engine.dispose()
        
    except Exception as e:
        print(f"URI connection failed: {e}")

if __name__ == "__main__":
    # Run all tests
    test_environment_variables()
    test_psycopg2_connection()
    test_sqlalchemy_connection()
    test_uri_from_env()
    
    # Try to import from the app if it exists
    try:
        from app.core.database import PostgresDB
        print("\n=== Testing App Database Module ===")
        db = PostgresDB()
        db.connect()
        print("App database connection successful!")
        db.disconnect()
    except ImportError as e:
        print(f"\nSkipping app module test - not found: {e}")
    except Exception as e:
        print(f"\nApp database test failed: {e}")