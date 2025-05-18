import psycopg2
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv('.env')
db_uri = os.getenv('DB__POSTGRES_URI')

conn = psycopg2.connect(db_uri)
cursor = conn.cursor()

# Check migration status
cursor.execute("SELECT version_num FROM alembic_version_postgres WHERE version_num = '91e93a42b21c'")
migration_applied = cursor.fetchone() is not None
print(f"Migration '91e93a42b21c' applied: {migration_applied}")

# Check schema
cursor.execute("SELECT schema_name FROM information_schema.schemata WHERE schema_name = 'app_auth'")
schema_exists = cursor.fetchone() is not None
print(f"Schema 'app_auth' exists: {schema_exists}")

# Check tables
cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'app_auth'")
tables = [row[0] for row in cursor.fetchall()]
print(f"Tables in app_auth schema: {', '.join(tables) if tables else 'None'}")

conn.close()