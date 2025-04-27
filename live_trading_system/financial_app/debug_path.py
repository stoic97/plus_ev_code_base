import sys
import os

# Print Python version and path
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print("\nPython path:")
for path in sys.path:
    print(f"  - {path}")

# Try to import yaml and show its location
try:
    import yaml
    print(f"\nSuccessfully imported yaml from: {yaml.__file__}")
except ImportError:
    print("\nFailed to import yaml")

# Check current directory and list files
print(f"\nCurrent working directory: {os.getcwd()}")
print("\nFiles in current directory:")
for item in os.listdir('.'):
    print(f"  - {item}")

# Look for app directory
print("\nSearching for app directory and fyers_client.py...")
for root, dirs, files in os.walk('.', topdown=True):
    if 'app' in dirs:
        app_path = os.path.join(root, 'app')
        print(f"Found app directory at: {app_path}")
        try:
            if 'services' in os.listdir(app_path):
                services_path = os.path.join(app_path, 'services')
                print(f"  - Found services directory at: {services_path}")
                if 'fyers_client.py' in os.listdir(services_path):
                    print(f"    - Found fyers_client.py at: {os.path.join(services_path, 'fyers_client.py')}")
        except Exception as e:
            print(f"  - Error checking app directory: {e}")

# Check for config file
print("\nSearching for broker_config.yaml...")
config_paths = [
    "config/broker_config.yaml", 
    "broker_config.yaml",
    "../config/broker_config.yaml",
    "financial_app/config/broker_config.yaml"
]
for path in config_paths:
    if os.path.exists(path):
        print(f"  - Found config at: {os.path.abspath(path)}")