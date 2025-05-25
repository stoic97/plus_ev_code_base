import sys
import os
import asyncio
from pathlib import Path
import pytest
import socket
import tempfile
import subprocess
import shutil
from unittest.mock import MagicMock, patch
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
postgres_port = find_guaranteed_free_port()
os.environ['POSTGRESQL_PORT'] = str(postgres_port)

# Create a PostgreSQL factory for tests
postgresql_test_proc = factories.postgresql_proc(
    port=postgres_port
)
postgresql_test = factories.postgresql('postgresql_test_proc')

# Create a TimescaleDB port
timescale_port = find_guaranteed_free_port() if postgres_port != 45432 else 55432

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

# ==================== Migration Testing Fixtures ====================

@pytest.fixture(scope="function")
def event_loop_policy():
    """Return an event loop policy for testing."""
    return asyncio.get_event_loop_policy()

@pytest.fixture(scope="function")
def event_loop(event_loop_policy):
    """Create an instance of the default event loop for each test case."""
    loop = event_loop_policy.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def mock_settings():
    """
    Provides a mock settings object that can be used in tests.
    This isolates tests from the actual application settings.
    """
    # Create a mock settings object based on the actual Settings class
    from app.core.config import Settings
    settings = Settings()
    
    # Override database URIs with test-specific values
    settings.db.POSTGRES_URI = f"postgresql://postgres:postgres@localhost:{postgres_port}/test_db"
    settings.db.TIMESCALE_URI = f"postgresql://postgres:postgres@localhost:{timescale_port}/test_timescale_db"
    settings.db.MONGODB_URI = "mongodb://localhost:27017/test_db"
    
    # Return the modified settings
    return settings

@pytest.fixture
def mock_alembic_config():
    """
    Creates a mock Alembic configuration object for testing.
    """
    # Create a MagicMock that mimics alembic.config.Config
    config = MagicMock()
    
    # Set common attributes used in migration scripts
    config.get_main_option = MagicMock(return_value="app/db/migrations")
    config.get_section_option = MagicMock(return_value=None)
    config.set_main_option = MagicMock()
    config.config_file_name = "alembic.ini"
    
    return config

@pytest.fixture
def temp_migration_dir():
    """
    Creates a temporary directory structure for migration testing.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create migration directory structure
        migrations_dir = Path(temp_dir) / "migrations"
        versions_dir = migrations_dir / "versions"
        versions_dir.mkdir(parents=True)
        
        # Create a stub alembic.ini file
        alembic_ini = Path(temp_dir) / "alembic.ini"
        with open(alembic_ini, "w") as f:
            f.write("[alembic]\n")
            f.write(f"script_location = {migrations_dir}\n")
            f.write(f"sqlalchemy.url = postgresql://postgres:postgres@localhost:{postgres_port}/test_db\n")
        
        # Create a stub env.py file
        env_py = migrations_dir / "env.py"
        with open(env_py, "w") as f:
            f.write("# Stub env.py file for testing\n")
            f.write("from alembic import context\n")
            f.write("def run_migrations_online(): pass\n")
            f.write("def run_migrations_offline(): pass\n")
        
        yield temp_dir

@pytest.fixture
def mock_sqlalchemy_engine():
    """Mock SQLAlchemy engine to avoid real connections."""
    with patch('app.core.database.create_engine') as mock_create:
        # Configure mock engine
        mock_engine = MagicMock()
        
        # Mock the event system
        with patch('app.core.database.event') as mock_event:
            # Make listens_for a no-op that returns the function unchanged
            mock_event.listens_for = lambda target, event_name: lambda fn: fn
            
            # Mock sessionmaker
            mock_session_factory = MagicMock()
            mock_session = MagicMock()
            mock_session_factory.return_value = mock_session
            
            # Session context manager mock
            mock_session_ctx = MagicMock()
            mock_session.return_value = mock_session_ctx
            mock_session_ctx.__enter__.return_value = mock_session
            
            # Connection mock
            mock_connection = MagicMock()
            mock_result = MagicMock()
            mock_fetchone = MagicMock()
            
            # Set up the method chain
            mock_fetchone.test = 1
            mock_result.fetchone.return_value = mock_fetchone
            mock_session.execute.return_value = mock_result
            
            # Engine connection
            mock_conn_ctx = MagicMock()
            mock_conn_ctx.__enter__.return_value = mock_connection
            mock_engine.connect.return_value = mock_conn_ctx
            
            # Return values
            mock_create.return_value = mock_engine
            
            # Patch sessionmaker as well
            with patch('app.core.database.sessionmaker', return_value=mock_session_factory):
                yield mock_create

@pytest.fixture
def mock_postgres_db(mock_settings, mock_sqlalchemy_engine):
    """Provides a mocked PostgresDB instance."""
    with patch('app.core.database.PostgresDB._register_event_listeners', return_value=None):
        from app.core.database import PostgresDB
        db = PostgresDB(settings=mock_settings)
        db.connect()
        yield db
        db.disconnect()

@pytest.fixture
def mock_timescale_db(mock_settings, mock_sqlalchemy_engine):
    """Provides a mocked TimescaleDB instance."""
    with patch('app.core.database.TimescaleDB._register_event_listeners', return_value=None):
        from app.core.database import TimescaleDB
        db = TimescaleDB(settings=mock_settings)
        db.connect()
        yield db
        db.disconnect()

@pytest.fixture
def docker_postgres():
    """
    Fixture for starting PostgreSQL in Docker for integration tests.
    Skip this test if Docker is not available.
    """
    # Check if Docker is available
    try:
        subprocess.run(["docker", "--version"], check=True, capture_output=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        pytest.skip("Docker not available")
    
    # Start PostgreSQL container
    container_name = "test_postgres_migrations"
    subprocess.run([
        "docker", "run", "--rm", "-d",
        "--name", container_name,
        "-e", "POSTGRES_PASSWORD=postgres",
        "-e", "POSTGRES_DB=test_db",
        "-p", f"{postgres_port}:5432",
        "postgres:13"
    ], check=True)
    
    # Wait for PostgreSQL to start
    for _ in range(30):  # Wait up to 30 seconds
        try:
            result = subprocess.run([
                "docker", "exec", container_name,
                "pg_isready", "-U", "postgres"
            ], check=False, capture_output=True)
            if result.returncode == 0:
                break
        except subprocess.SubprocessError:
            pass
        import time
        time.sleep(1)
    
    # Modify environment variables for tests to use the Docker container
    old_env = os.environ.copy()
    os.environ["TEST_POSTGRES_URI"] = f"postgresql://postgres:postgres@localhost:{postgres_port}/test_db"
    
    yield
    
    # Stop and remove the container
    subprocess.run(["docker", "stop", container_name], check=False)
    
    # Restore environment variables
    os.environ.clear()
    os.environ.update(old_env)

@pytest.fixture
def docker_timescale():
    """
    Fixture for starting TimescaleDB in Docker for integration tests.
    Skip this test if Docker is not available.
    """
    # Check if Docker is available
    try:
        subprocess.run(["docker", "--version"], check=True, capture_output=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        pytest.skip("Docker not available")
    
    # Start TimescaleDB container
    container_name = "test_timescale_migrations"
    subprocess.run([
        "docker", "run", "--rm", "-d",
        "--name", container_name,
        "-e", "POSTGRES_PASSWORD=postgres",
        "-e", "POSTGRES_DB=test_timescale_db",
        "-p", f"{timescale_port}:5432",
        "timescale/timescaledb:latest-pg13"
    ], check=True)
    
    # Wait for TimescaleDB to start
    for _ in range(30):  # Wait up to 30 seconds
        try:
            result = subprocess.run([
                "docker", "exec", container_name,
                "pg_isready", "-U", "postgres"
            ], check=False, capture_output=True)
            if result.returncode == 0:
                break
        except subprocess.SubprocessError:
            pass
        import time
        time.sleep(1)
    
    # Modify environment variables for tests to use the Docker container
    old_env = os.environ.copy()
    os.environ["TEST_TIMESCALE_URI"] = f"postgresql://postgres:postgres@localhost:{timescale_port}/test_timescale_db"
    
    yield
    
    # Stop and remove the container
    subprocess.run(["docker", "stop", container_name], check=False)
    
    # Restore environment variables
    os.environ.clear()
    os.environ.update(old_env)

@pytest.fixture
def alembic_migration_scripts():
    """
    Mock migration file content for testing.
    """
    return {
        "create_tables": """
            from alembic import op
            import sqlalchemy as sa
            
            def upgrade():
                op.create_table(
                    'test_table',
                    sa.Column('id', sa.Integer, primary_key=True),
                    sa.Column('name', sa.String(50), nullable=False),
                    sa.Column('created_at', sa.DateTime, nullable=False)
                )
            
            def downgrade():
                op.drop_table('test_table')
        """,
        "add_column": """
            from alembic import op
            import sqlalchemy as sa
            
            def upgrade():
                op.add_column('test_table', sa.Column('description', sa.String(200)))
            
            def downgrade():
                op.drop_column('test_table', 'description')
        """,
        "timescale_hypertable": """
            from alembic import op
            import sqlalchemy as sa
            from app.db.migrations.helpers.timescale import create_hypertable
            
            def upgrade():
                # Create a time-series table
                op.create_table(
                    'metrics',
                    sa.Column('time', sa.DateTime, primary_key=True),
                    sa.Column('symbol', sa.String(20), primary_key=True),
                    sa.Column('value', sa.Float, nullable=False)
                )
                
                # Convert to hypertable
                create_hypertable('metrics', 'time')
            
            def downgrade():
                op.drop_table('metrics')
        """
    }