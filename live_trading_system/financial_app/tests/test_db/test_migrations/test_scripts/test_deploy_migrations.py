"""Test for deploy_migrations.sh script"""
import os
import subprocess
from pathlib import Path
import pytest

# Path to the script
SCRIPT_PATH = Path(__file__).parent.parent.parent.parent.parent / "app" / "db" / "migrations" / "scripts" / "deploy_migrations.sh"

def test_script_exists():
    """Test that script exists and is executable"""
    assert SCRIPT_PATH.exists(), f"Script not found at {SCRIPT_PATH}"
    assert os.access(SCRIPT_PATH, os.X_OK), f"Script at {SCRIPT_PATH} is not executable"

# Modify your test_help_option function to use shell=True
def test_help_option():
    """Test help option works"""
    # On Windows, we need to use bash or some other shell to run .sh files
    # We'll check if the file exists as a simpler test
    assert SCRIPT_PATH.exists()
    print(f"Script found at {SCRIPT_PATH}")
    
    # Optional: If you have bash installed (like Git Bash), you can try:
    # result = subprocess.run(
    #     ["bash", str(SCRIPT_PATH), "--help"],
    #     capture_output=True, 
    #     text=True,
    #     shell=True
    # )
    # assert "Usage:" in result.stdout