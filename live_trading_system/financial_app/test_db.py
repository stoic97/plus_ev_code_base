import psycopg2
import os

# Get password from .env manually to simplify testing
password = "StrongplusEV125"  # This should match your .env file

try:
    # Connect to PostgreSQL
    conn = psycopg2.connect(
        host="localhost",
        database="live-trading-system",  # From your .env
        user="postgres",
        password=password
    )
    
    # Create a cursor
    cur = conn.cursor()
    
    # Execute a simple query
    cur.execute("SELECT 1")
    result = cur.fetchone()
    
    print(f"Connection successful! Query result: {result}")
    
    # Close cursor and connection
    cur.close()
    conn.close()
    
except Exception as e:
    print(f"Error connecting to PostgreSQL: {e}")