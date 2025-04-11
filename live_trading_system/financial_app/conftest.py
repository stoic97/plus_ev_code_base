import sys
import os
from pathlib import Path
import pytest
import socket
from pytest_postgresql import factories


# Add the financial_app directory to Python path
# Since conftest.py is in financial_app folder, we want the parent directory 
# (live_trading_system) on the path
app_dir = Path(__file__).parent
project_root = app_dir.parent
sys.path.insert(0, str(project_root))

# Setup test environment variables if they don't exist already
if not os.environ.get("TEST_POSTGRES_URI"):
    os.environ["TEST_POSTGRES_URI"] = "postgresql://postgres:postgres@localhost:5432/test_db"

if not os.environ.get("TEST_MONGO_URI"):
    os.environ["TEST_MONGO_URI"] = "mongodb://localhost:27017/test_db"

if not os.environ.get("TEST_REDIS_HOST"):
    os.environ["TEST_REDIS_HOST"] = "localhost"
    os.environ["TEST_REDIS_PORT"] = "6379"

def find_guaranteed_free_port():
    try:
        # Try a specific high port range
        for port in [15432, 25432, 35432, 45432]:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                result = s.connect_ex(('localhost', port))
                if result != 0:  # Port is available
                    print(f"Selected port: {port}")
                    return port
    except:
        pass
    
    # Fallback: Let OS choose
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('localhost', 0))
    port = s.getsockname()[1]
    s.close()
    print(f"Using OS-selected port: {port}")
    return port

# Set environment variable to force port
port = find_guaranteed_free_port()
os.environ['POSTGRESQL_PORT'] = str(port)

# Create a PostgreSQL factory for tests
postgresql_test_proc = factories.postgresql_proc(
    port=port
)
postgresql_test = factories.postgresql('postgresql_test_proc')

@pytest.fixture
def db_connection(postgresql_test):
    """Provide a raw database connection to the temporary PostgreSQL database."""
    conn = postgresql_test.cursor().connection
    yield conn
    conn.close()

@pytest.fixture
def db_session(postgresql_test):
    """Provide a SQLAlchemy session connected to the temporary database."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    
    # Build a connection string using the temporary database details
    connection_string = (
        f"postgresql://{postgresql_test.info.user}:"
        f"{postgresql_test.info.password}@{postgresql_test.info.host}:"
        f"{postgresql_test.info.port}/{postgresql_test.info.dbname}"
    )
    
    # Create an engine connected to the temporary database
    engine = create_engine(connection_string)
    
    # Import your models and create all tables defined in them
    # Since conftest.py is in the financial_app folder, we need a relative import
    from models import Base
    Base.metadata.create_all(engine)
    
    # Create and provide a session
    Session = sessionmaker(bind=engine)
    session = Session()
    
    yield session
    
    # Clean up after the test
    session.close()
    Base.metadata.drop_all(engine)

# Alternative fixture that uses your existing TEST_POSTGRES_URI
@pytest.fixture
def db_session_env():
    """Provide a SQLAlchemy session using the TEST_POSTGRES_URI environment variable."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    
    connection_string = os.environ.get("TEST_POSTGRES_URI")
    engine = create_engine(connection_string)
    
    # Import your models and create all tables
    from models import Base
    # Clear any existing tables to ensure clean state
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    
    # Create and provide a session
    Session = sessionmaker(bind=engine)
    session = Session()
    
    yield session
    
    # Clean up after the test
    session.close()
    Base.metadata.drop_all(engine)
