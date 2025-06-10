#!/usr/bin/env python3
"""
Dependency Fix Script for AWS Monitoring Tests

This script checks your current package versions and installs/upgrades
packages as needed to work with the fixed test suite.

Run with: python dependency_fix_script.py
"""

import subprocess
import sys
import pkg_resources
from packaging import version

def run_command(command):
    """Run a shell command and return the result"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
    except Exception as e:
        return False, "", str(e)

def get_installed_version(package_name):
    """Get the installed version of a package"""
    try:
        return pkg_resources.get_distribution(package_name).version
    except pkg_resources.DistributionNotFound:
        return None

def check_and_install_packages():
    """Check and install required packages"""
    print("🔍 Checking AWS Monitoring Dependencies...")
    print("=" * 50)
    
    # Required packages with minimum versions
    required_packages = {
        'moto': '5.0.0',
        'watchtower': '3.0.0',
        'boto3': '1.26.0',
        'aws-xray-sdk': '2.10.0',
        'fastapi': '0.68.0',
        'pytest': '7.0.0',
        'pytest-asyncio': '0.20.0',
        'psutil': '5.8.0'
    }
    
    needs_install = []
    needs_upgrade = []
    
    for package, min_version in required_packages.items():
        current_version = get_installed_version(package)
        
        if current_version is None:
            print(f"❌ {package}: Not installed")
            needs_install.append(package)
        elif version.parse(current_version) < version.parse(min_version):
            print(f"⚠️  {package}: {current_version} (needs >= {min_version})")
            needs_upgrade.append(package)
        else:
            print(f"✅ {package}: {current_version}")
    
    # Install missing packages
    if needs_install:
        print(f"\n📦 Installing missing packages: {', '.join(needs_install)}")
        for package in needs_install:
            success, stdout, stderr = run_command(f"pip install {package}")
            if success:
                print(f"✅ Successfully installed {package}")
            else:
                print(f"❌ Failed to install {package}: {stderr}")
    
    # Upgrade outdated packages
    if needs_upgrade:
        print(f"\n⬆️  Upgrading outdated packages: {', '.join(needs_upgrade)}")
        for package in needs_upgrade:
            success, stdout, stderr = run_command(f"pip install --upgrade {package}")
            if success:
                print(f"✅ Successfully upgraded {package}")
            else:
                print(f"❌ Failed to upgrade {package}: {stderr}")
    
    if not needs_install and not needs_upgrade:
        print("\n🎉 All dependencies are up to date!")
    
    return len(needs_install) == 0 and len(needs_upgrade) == 0

def check_moto_version():
    """Check moto version and provide specific guidance"""
    print("\n🔍 Checking Moto Version...")
    print("=" * 30)
    
    moto_version = get_installed_version('moto')
    if moto_version:
        print(f"Current moto version: {moto_version}")
        
        if version.parse(moto_version) >= version.parse('5.0.0'):
            print("✅ Moto 5.x detected - using mock_aws (correct)")
            return True
        else:
            print("⚠️  Moto 4.x detected - needs upgrade to 5.x")
            print("   The test file is fixed to use mock_aws instead of specific service mocks")
            return False
    else:
        print("❌ Moto not installed")
        return False

def check_watchtower_version():
    """Check watchtower version and provide specific guidance"""
    print("\n🔍 Checking Watchtower Version...")
    print("=" * 35)
    
    watchtower_version = get_installed_version('watchtower')
    if watchtower_version:
        print(f"Current watchtower version: {watchtower_version}")
        print("✅ Fixed test uses CloudWatchLogHandler (correct import)")
        return True
    else:
        print("❌ Watchtower not installed")
        return False

def create_pytest_config():
    """Create pytest configuration for asyncio"""
    pytest_config = """[tool:pytest]
asyncio_mode = auto
asyncio_default_fixture_loop_scope = function
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short --durations=10
markers =
    asyncio: marks tests as asyncio (deselect with '-m "not asyncio"')
    slow: marks tests as slow (deselect with '-m "not slow"')
"""
    
    try:
        with open('pytest.ini', 'w') as f:
            f.write(pytest_config)
        print("✅ Created pytest.ini with asyncio configuration")
        return True
    except Exception as e:
        print(f"❌ Failed to create pytest.ini: {e}")
        return False

def main():
    """Main function"""
    print("🚀 AWS Monitoring Test Dependencies Fixer")
    print("=========================================\n")
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        print("❌ Python 3.8+ required for AWS monitoring tests")
        return False
    
    # Check and install packages
    packages_ok = check_and_install_packages()
    
    # Check specific versions
    moto_ok = check_moto_version()
    watchtower_ok = check_watchtower_version()
    
    # Create pytest config
    print("\n🔧 Setting up pytest configuration...")
    pytest_ok = create_pytest_config()
    
    # Summary
    print("\n📋 Summary")
    print("=" * 20)
    
    if packages_ok and moto_ok and watchtower_ok and pytest_ok:
        print("🎉 All checks passed! You can now run the tests:")
        print("   pytest tests/test_monitoring/test_aws_monitoring_fixed.py -v")
        
        print("\n💡 Quick test command:")
        print("   cd tests/test_monitoring")
        print("   pytest test_aws_monitoring_fixed.py::TestAWSMonitoringImports::test_aws_monitoring_imports_successfully -v")
        
        return True
    else:
        print("⚠️  Some issues need to be resolved:")
        if not packages_ok:
            print("   - Package installation/upgrade issues")
        if not moto_ok:
            print("   - Moto version issues")
        if not watchtower_ok:
            print("   - Watchtower not available")
        if not pytest_ok:
            print("   - Pytest configuration issues")
        
        print("\n🔧 Try running this script again after fixing the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)