"""
Import all models to register them with SQLAlchemy Base.
"""

# Import Base first
from app.core.database import Base

# Import models one by one to catch errors
try:
    from app.models.user import *
    print("✅ Imported user models")
except Exception as e:
    print(f"❌ Error importing user models: {e}")

try:
    from app.models.market_data import *
    print("✅ Imported market_data models")
except Exception as e:
    print(f"❌ Error importing market_data models: {e}")

try:
    from app.models.base import *
    print("✅ Imported base models")
except Exception as e:
    print(f"❌ Error importing base models: {e}")

try:
    from app.models.fyers_tokens import *
    print("✅ Imported fyers_tokens models")
except Exception as e:
    print(f"❌ Error importing fyers_tokens models: {e}")

# Skip problematic files for now
# from app.models.account import *
# from app.models.orders import *
# from app.models.strategy import *
# from app.models.trading import *

# Export Base for convenience
__all__ = ["Base"]