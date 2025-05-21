import os
import psycopg2
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import contextlib
from io import StringIO
from alembic import command
from alembic.config import Config

# Load environment variables
load_dotenv('.env')
db_uri = os.getenv('DB__POSTGRES_URI')

print("=== CHECKING VERSION TABLE DISCREPANCY ===")

# 1. Direct database check
conn = psycopg2.connect(db_uri)
cursor = conn.cursor()

# Check alembic_version_postgres
cursor.execute("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'alembic_version_postgres')")
table_exists = cursor.fetchone()[0]
print(f"alembic_version_postgres table exists: {table_exists}")

if table_exists:
    cursor.execute("SELECT version_num FROM alembic_version_postgres")
    versions = [row[0] for row in cursor.fetchall()]
    print(f"Versions in alembic_version_postgres: {versions}")

# Check alembic_version
cursor.execute("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'alembic_version')")
table_exists = cursor.fetchone()[0]
print(f"alembic_version table exists: {table_exists}")

if table_exists:
    cursor.execute("SELECT version_num FROM alembic_version")
    versions = [row[0] for row in cursor.fetchall()]
    print(f"Versions in alembic_version: {versions}")

# Check what tables exist
cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_name LIKE 'alembic%'")
alembic_tables = [row[0] for row in cursor.fetchall()]
print(f"All alembic-related tables: {alembic_tables}")

# 2. Check what Alembic is using
print("\n=== CHECKING ALEMBIC CONFIGURATION ===")

# Set up Alembic config
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
alembic_ini = os.path.join(project_root, 'alembic.ini')

if os.path.exists(alembic_ini):
    config = Config(alembic_ini)
    
    # Check configuration
    script_location = config.get_main_option("script_location")
    print(f"Script location: {script_location}")
    
    # Set the database option
    cmd_opts = type('', (), {})()
    cmd_opts.x = ["database=postgres"]
    config.cmd_opts = cmd_opts
    
    # Get version table name from env.py
    with open(os.path.join(project_root, script_location, "env.py"), "r") as f:
        env_py = f.read()
        if "version_table" in env_py:
            print("version_table found in env.py")
            if "alembic_version_postgres" in env_py:
                print("Using version table: alembic_version_postgres")
            elif "alembic_version" in env_py:
                print("Using version table: alembic_version")
            else:
                print("Custom version table found")
        else:
            print("No version_table specification found in env.py")
    
    # Check what the current command sees
    print("\nRunning alembic current command:")
    output = StringIO()
    with contextlib.redirect_stdout(output):
        command.current(config, verbose=True)
    output_str = output.getvalue()
    print(output_str)

conn.close()