"""
Financial App Package

This package contains the core functionality for the live trading system.
"""

import os
import sys
from pathlib import Path

# Get the project root directory
project_root = Path(__file__).parent.parent

# Add the project root to Python path if not already there
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Add the financial_app directory to Python path if not already there
financial_app_path = project_root / "financial_app"
if str(financial_app_path) not in sys.path:
    sys.path.insert(0, str(financial_app_path))
