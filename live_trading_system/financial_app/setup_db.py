# setup_db.py
import os
import psycopg2
from sqlalchemy import create_engine, text
from app.core.database import Base
from app.core.config import get_settings

# Import all models to ensure they're registered with Base
import app.models
from app.models.account import Account, Balance, Position
from app.models.trading import Order, Execution, ActivePosition, Trade, OrderEvent, BracketOrder
from app.models.user import User, Role
from app.models.market_data import Instrument, OHLCV, Tick, OrderBookSnapshot

def setup_database():
    try:
        # Get database settings from application config
        settings = get_settings()
        
        # Create SQLAlchemy engine
        print(f"Connecting to {settings.db.POSTGRES_DB}")
        engine = create_engine(str(settings.db.POSTGRES_URI))
        
        # Create all tables
        Base.metadata.create_all(engine)
        print("Tables created successfully")
        
        # Test connection to the database
        with engine.connect() as connection:
            result = connection.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"))
            tables = [row[0] for row in result.fetchall()]
            print(f"Created tables: {', '.join(tables)}")
        
        print("Database setup completed successfully")
        
    except Exception as e:
        print(f"Error during database setup: {e}")

if __name__ == "__main__":
    setup_database()