import psycopg2
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv('.env')

# Get database credentials
db_uri = os.getenv('DB__POSTGRES_URI')

# Connect to the database
conn = psycopg2.connect(db_uri)
conn.autocommit = True
cursor = conn.cursor()

# Check if migration is already applied
cursor.execute("SELECT version_num FROM alembic_version_postgres WHERE version_num = '91e93a42b21c'")
if cursor.fetchone():
    print("Resetting migration 91e93a42b21c...")
    cursor.execute("DELETE FROM alembic_version_postgres WHERE version_num = '91e93a42b21c'")
    print("Migration reset successfully")
else:
    print("Migration is not applied, no need to reset")

# Check if app_auth schema exists
cursor.execute("SELECT schema_name FROM information_schema.schemata WHERE schema_name = 'app_auth'")
if cursor.fetchone():
    print("Dropping app_auth schema...")
    cursor.execute("DROP SCHEMA IF EXISTS app_auth CASCADE")
    print("Schema dropped successfully")
else:
    print("app_auth schema doesn't exist, no need to drop")

conn.close()