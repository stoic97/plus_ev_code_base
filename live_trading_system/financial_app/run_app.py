"""
Run script for the Trading Strategies Application.
Place this file in your project root directory.
"""
import os
import sys

# Get the absolute path to the project root directory
project_root = os.path.dirname(os.path.abspath(__file__))

# Add the project root to the Python path
sys.path.insert(0, project_root)

# Now we can import from the app directly
from financial_app.app.main import app, settings

if __name__ == "__main__":
    import uvicorn
    
    print(f"Starting {settings.APP_NAME} v{settings.APP_VERSION} from project root")
    
    # Run the application with the proper module path
    uvicorn.run(
        "financial_app.app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        workers=1  # Use 1 worker in debug mode for easier debugging
    )