import psycopg2
from psycopg2 import OperationalError

def test_connection():
    # Use your .env file values
    params = {
        "host": "localhost",
        "database": "live-trading-system",  # Try connecting to the default database firstyucft
        "user": "postgres",
        "password": "StrongplusEV125"  # Your password from .env
    }

    print(f"Trying to connect with: {params}")
    
    try:
        connection = psycopg2.connect(**params)
        cursor = connection.cursor()
        cursor.execute("SELECT 1")
        print("PostgreSQL connection successful!")
        
        # Now try to see if your database exists
        cursor.execute("SELECT datname FROM pg_database;")
        databases = cursor.fetchall()
        print("Available databases:")
        for db in databases:
            print(f"- {db[0]}")
        
        cursor.close()
        connection.close()
        return True
    except OperationalError as e:
        print(f"Error connecting to PostgreSQL: {e}")
        return False

if __name__ == "__main__":
    test_connection()