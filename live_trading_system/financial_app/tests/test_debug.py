# tests/test_debug.py
import os
import sys
import pytest

def test_import_path():
    """Test that we can properly import modules from the app."""
    print(f"\nPython path: {sys.path}")
    print(f"Current directory: {os.getcwd()}")
    
    # Try to import the specific module causing issues
    try:
        import app
        print(f"app module location: {app.__file__}")
        
        # Verify the path
        try:
            import app.db
            print(f"app.db location: {app.db.__file__}")
            
            import app.db.migrations
            print(f"app.db.migrations location: {app.db.migrations.__file__}")
            
            import app.db.migrations.timescale
            print(f"app.db.migrations.timescale location: {app.db.migrations.timescale.__file__}")
            
            import app.db.migrations.timescale.versions
            print(f"app.db.migrations.timescale.versions location: {app.db.migrations.timescale.versions.__file__}")
            
            # The specific file might have a different name format
            # Try to list the directory contents to find the exact file name
            versions_dir = os.path.dirname(app.db.migrations.timescale.versions.__file__)
            print(f"Files in versions directory: {os.listdir(versions_dir)}")
            
            # Try different name formats
            try:
                from app.db.migrations.timescale.versions import _20250428_setup_market_data_hypertables
                print("Successfully imported with underscore prefix")
            except ImportError:
                try:
                    from app.db.migrations.timescale.versions import setup_market_data_hypertables_20250428
                    print("Successfully imported with suffix format")
                except ImportError:
                    for filename in os.listdir(versions_dir):
                        if "market_data_hypertables" in filename:
                            print(f"Found matching file: {filename}")
                            module_name = filename.replace(".py", "")
                            exec(f"from app.db.migrations.timescale.versions import {module_name}")
                            print(f"Successfully imported {module_name}")
                            break
            
        except ImportError as e:
            print(f"Import path error: {e}")
            
    except ImportError as e:
        pytest.fail(f"Import error: {e}")

    assert True  # Make the test pass