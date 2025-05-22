import sys
import traceback
from sqlalchemy import create_engine, text

# Define your connection parameters
password = "Ck1Ge6xWwgXVOVM5"
database_url = f"postgresql+psycopg2://postgres:{password}@db.idktztdekhcvjqtzzzqj.supabase.co:5432/postgres"

print(f"Attempting to connect to: {database_url.replace(password, '****')}")

try:
    # Create the engine with verbose logging
    engine = create_engine(
        database_url,
        connect_args={"sslmode": "require"},
        echo=True  # This will log all SQL commands
    )
    
    print("Engine created, attempting connection...")
    
    # Try to connect and execute a simple query
    with engine.connect() as connection:
        print("Connection established!")
        result = connection.execute(text("SELECT NOW();"))
        row = result.fetchone()
        print(f"Current database time: {row[0]}")
except Exception as e:
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {str(e)}")
    print("Traceback:")
    traceback.print_exc(file=sys.stdout)
    
print("Script execution completed")