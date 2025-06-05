#!/usr/bin/env python3

import pytest
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from termcolor import colored
import time

def run_tests() -> Tuple[Dict[str, List[str]], Dict[str, List[str]], int, int]:
    """
    Run all tests and collect failures.
    
    Returns:
        Tuple of (failed_tests_dict, collection_errors_dict, total_failed_count, total_collection_errors)
    """
    # Get the tests directory path
    tests_dir = Path(__file__).parent / 'tests'
    
    # Store failed tests and collection errors
    failed_tests: Dict[str, List[str]] = {}
    collection_errors: Dict[str, List[str]] = {}
    total_failed = 0
    total_collection_errors = 0
    
    class FailureCollector:
        def pytest_runtest_logreport(self, report):
            """Capture test execution failures"""
            if report.failed:
                nonlocal total_failed
                # Extract file path and test name
                nodeid = report.nodeid
                if "::" in nodeid:
                    file_path, test_name = nodeid.split("::", 1)
                else:
                    file_path, test_name = nodeid, "unknown_test"
                
                if file_path not in failed_tests:
                    failed_tests[file_path] = []
                failed_tests[file_path].append(test_name)
                total_failed += 1
        
        def pytest_collectreport(self, report):
            """Capture collection failures (import errors, etc.)"""
            if report.failed:
                nonlocal total_collection_errors
                # Get the file path from the collection report
                file_path = str(report.fspath) if hasattr(report, 'fspath') else str(report.nodeid)
                error_msg = str(report.longrepr) if report.longrepr else "Unknown collection error"
                
                if file_path not in collection_errors:
                    collection_errors[file_path] = []
                collection_errors[file_path].append(error_msg)
                total_collection_errors += 1

    # Create collector instance
    collector = FailureCollector()
    
    # Run pytest with our custom plugin
    exit_code = pytest.main([
        str(tests_dir),
        '-v',
        '--tb=short',
        '--color=yes',
        '-p', 'no:warnings'
    ], plugins=[collector])
    
    return failed_tests, collection_errors, total_failed, total_collection_errors

def print_results(failed_tests: Dict[str, List[str]], collection_errors: Dict[str, List[str]], 
                 total_failed: int, total_collection_errors: int) -> None:
    """Print test results in a formatted way."""
    print("\n" + "="*80)
    print(colored("Test Execution Summary", "cyan", attrs=['bold']))
    print("="*80 + "\n")
    
    # Check if we have any issues at all
    has_issues = bool(failed_tests or collection_errors)
    
    if not has_issues:
        print(colored("✓ All tests passed!", "green", attrs=['bold']))
        return
    
    # Print collection errors first (these prevent tests from running)
    if collection_errors:
        print(colored(f"✗ Collection Errors ({total_collection_errors} total):", "red", attrs=['bold']))
        print(colored("These errors prevent tests from being discovered/imported:", "yellow"))
        print("-"*80)
        
        for file_path, error_list in collection_errors.items():
            print(colored(f"\nFile: {file_path}", "red"))
            for i, error in enumerate(error_list, 1):
                # Show only the relevant part of the error
                error_lines = error.split('\n')
                relevant_error = next((line for line in error_lines if 'Error:' in line or 'Exception:' in line), error_lines[-1] if error_lines else error)
                print(f"  {i}. {relevant_error}")
    
    # Print test execution failures
    if failed_tests:
        if collection_errors:
            print("\n" + "-"*80)
        
        print(colored(f"✗ Test Execution Failures ({total_failed} total):", "red", attrs=['bold']))
        print("-"*80)
        
        for file_path, failed_list in failed_tests.items():
            print(colored(f"\nFile: {file_path}", "yellow"))
            print(f"Failed Tests: {len(failed_list)}")
            for test_name in failed_list:
                print(f"  • {test_name}")
    
    print("\n" + "="*80)
    if collection_errors and failed_tests:
        print(colored(f"Total Issues: {total_collection_errors} collection errors + {total_failed} test failures", "red", attrs=['bold']))
    elif collection_errors:
        print(colored(f"Total Collection Errors: {total_collection_errors}", "red", attrs=['bold']))
        print(colored("Note: Tests could not run due to collection errors", "yellow"))
    else:
        print(colored(f"Total Failed Tests: {total_failed}", "red", attrs=['bold']))
    print("="*80 + "\n")

def main():
    """Main entry point for the test runner."""
    start_time = time.time()
    
    print("\n" + "="*80)
    print(colored("Test Execution Started at " + time.strftime("%Y-%m-%d %H:%M:%S"), "cyan"))
    print("="*80 + "\n")
    
    failed_tests, collection_errors, total_failed, total_collection_errors = run_tests()
    
    end_time = time.time()
    duration = end_time - start_time
    
    print_results(failed_tests, collection_errors, total_failed, total_collection_errors)
    
    print(colored(f"Total Execution Time: {duration:.2f} seconds", "cyan"))
    
    # Return non-zero exit code if there were any issues
    has_issues = total_failed > 0 or total_collection_errors > 0
    sys.exit(1 if has_issues else 0)

if __name__ == "__main__":
    main()