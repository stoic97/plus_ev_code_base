#!/usr/bin/env python3

import pytest
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from termcolor import colored
import time

def run_tests() -> Tuple[Dict[str, List[str]], int]:
    """
    Run all tests and collect failures.
    
    Returns:
        Tuple of (failed_tests_dict, total_failed_count)
        where failed_tests_dict maps file paths to lists of failed test names
    """
    # Get the tests directory path
    tests_dir = Path(__file__).parent / 'tests'
    
    # Store failed tests by file
    failed_tests: Dict[str, List[str]] = {}
    total_failed = 0
    
    class FailureCollector:
        def pytest_runtest_logreport(self, report):
            if report.failed:
                # Extract file path and test name
                nodeid = report.nodeid
                file_path, test_name = nodeid.split("::", 1)
                
                if file_path not in failed_tests:
                    failed_tests[file_path] = []
                failed_tests[file_path].append(test_name)
                nonlocal total_failed
                total_failed += 1

    # Run pytest with our custom plugin
    pytest.main([
        str(tests_dir),
        '-v',
        '--tb=short',
        '--color=yes',
        '-p', 'no:warnings'
    ], plugins=[FailureCollector()])
    
    return failed_tests, total_failed

def print_results(failed_tests: Dict[str, List[str]], total_failed: int) -> None:
    """Print test results in a formatted way."""
    print("\n" + "="*80)
    print(colored("Test Execution Summary", "cyan", attrs=['bold']))
    print("="*80 + "\n")
    
    if not failed_tests:
        print(colored("✓ All tests passed!", "green", attrs=['bold']))
        return
    
    print(colored(f"✗ Failed Tests Summary ({total_failed} total failures):", "red", attrs=['bold']))
    print("-"*80)
    
    for file_path, failed_list in failed_tests.items():
        print(colored(f"\nFile: {file_path}", "yellow"))
        print(f"Failed Tests: {len(failed_list)}")
        for test_name in failed_list:
            print(f"  • {test_name}")
    
    print("\n" + "="*80)
    print(colored(f"Total Failed Tests: {total_failed}", "red", attrs=['bold']))
    print("="*80 + "\n")

def main():
    """Main entry point for the test runner."""
    start_time = time.time()
    
    print("\n" + "="*80)
    print(colored("Test Execution Started at " + time.strftime("%Y-%m-%d %H:%M:%S"), "cyan"))
    print("="*80 + "\n")
    
    failed_tests, total_failed = run_tests()
    
    end_time = time.time()
    duration = end_time - start_time
    
    print_results(failed_tests, total_failed)
    
    print(colored(f"Total Execution Time: {duration:.2f} seconds", "cyan"))
    
    # Return non-zero exit code if there were failures
    sys.exit(1 if total_failed > 0 else 0)

if __name__ == "__main__":
    main()
