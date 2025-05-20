import os
import sys
import logging

# Configure logging with enhanced format for debugging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler("app_debug.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# Get the absolute path to the project root directory
# This should point to the directory containing 'financial_app'
script_dir = os.path.dirname(os.path.abspath(__file__))

# If run_app.py is inside financial_app/, go up one level
if os.path.basename(script_dir) == 'financial_app':
    project_root = os.path.dirname(script_dir)
else:
    project_root = script_dir

# Add the project root to the Python path
sys.path.insert(0, project_root)

print(f"Script directory: {script_dir}")
print(f"Project root: {project_root}")
print(f"Looking for financial_app in: {os.path.join(project_root, 'financial_app')}")

# Verify the path exists
if not os.path.exists(os.path.join(project_root, 'financial_app')):
    print("Error: financial_app directory not found!")
    print(f"Contents of {project_root}:")
    try:
        for item in os.listdir(project_root):
            print(f"  - {item}")
    except Exception as e:
        print(f"  Error listing directory: {e}")
    sys.exit(1)

# Now we can import from the app directly
try:
    from financial_app.app.main import app, settings
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure financial_app/app/main.py exists and contains 'app' and 'settings'")
    sys.exit(1)

# Set specific loggers to DEBUG level for auth debugging
auth_logger = logging.getLogger("financial_app.app.core.security")
auth_logger.setLevel(logging.DEBUG)

db_logger = logging.getLogger("financial_app.app.core.database")
db_logger.setLevel(logging.DEBUG)

# Add logger for uvicorn to see request details
uvicorn_logger = logging.getLogger("uvicorn.access")
uvicorn_logger.setLevel(logging.DEBUG)

if __name__ == "__main__":
    import uvicorn
    
    logging.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION} from project root")
    logging.info(f"Debug mode: {settings.DEBUG}")
    logging.info(f"Log file: app_debug.log")
    
    # Run the application with the proper module path
    uvicorn.run(
        "financial_app.app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        workers=1,  # Use 1 worker in debug mode for easier debugging
        log_level="debug",  # Set Uvicorn log level to debug
        access_log=True,   # Enable access logs
        use_colors=False   # Disable colors so log file is clean
    )