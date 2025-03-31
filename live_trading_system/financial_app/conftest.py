import sys
import os
from pathlib import Path

# Add the project root directory to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Setup test environment variables if they don't exist already
if not os.environ.get("TEST_POSTGRES_URI"):
    os.environ["TEST_POSTGRES_URI"] = "postgresql://postgres:postgres@localhost:5432/test_db"

if not os.environ.get("TEST_MONGO_URI"):
    os.environ["TEST_MONGO_URI"] = "mongodb://localhost:27017/test_db"

if not os.environ.get("TEST_REDIS_HOST"):
    os.environ["TEST_REDIS_HOST"] = "localhost"
    os.environ["TEST_REDIS_PORT"] = "6379"