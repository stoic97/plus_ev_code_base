"""
Utility functions for generating test reports.
"""

import os
import json
import logging
import csv
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import matplotlib.pyplot as plt
from pathlib import Path


from financial_app.tests.integration.data_layer_e2e.e2e_config import TEST_REPORT_DIR, TEST_ID

logger = logging.getLogger(__name__)

class TestReporter:
    """Class for handling test reporting functionality."""
    
    def __init__(self):
        self.report_dir = os.path.join(TEST_REPORT_DIR, TEST_ID)
        os.makedirs(self.report_dir, exist_ok=True)
        
        self.test_results = {
            "test_id": TEST_ID,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "duration_seconds": None,
            "overall_status": "RUNNING",
            "tests": {},
            "performance_metrics": {},
            "error_summary": []
        }
        
        # Initialize report files
        self.summary_file = os.path.join(self.report_dir, "summary.json")
        self.performance_csv = os.path.join(self.report_dir, "performance.csv")
        
        # Create CSV header
        with open(self.performance_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'metric', 'value'])
        
        logger.info(f"Test reporting initialized. Report directory: {self.report_dir}")
    
    def record_test_result(self, test_name: str, status: str, 
                          details: Optional[Dict[str, Any]] = None, 
                          error: Optional[str] = None) -> None:
        """
        Record the result of a test.
        
        Args:
            test_name: Name of the test
            status: Status of the test (PASS, FAIL, SKIP)
            details: Additional details about the test
            error: Error message if the test failed
        """
        timestamp = datetime.now().isoformat()
        
        self.test_results["tests"][test_name] = {
            "status": status,
            "timestamp": timestamp,
            "details": details or {},
        }
        
        if error and status == "FAIL":
            self.test_results["tests"][test_name]["error"] = error
            self.test_results["error_summary"].append({
                "test_name": test_name,
                "timestamp": timestamp,
                "error": error
            })
        
        # Save updated results
        self._save_summary()
        
        logger.info(f"Test '{test_name}' completed with status: {status}")
        if error:
            logger.error(f"Test '{test_name}' error: {error}")
    
    def record_performance_metric(self, metric_name: str, value: Union[int, float]) -> None:
        """
        Record a performance metric.
        
        Args:
            metric_name: Name of the metric
            value: Value of the metric
        """
        timestamp = datetime.now().isoformat()
        
        # Add to in-memory metrics
        if metric_name not in self.test_results["performance_metrics"]:
            self.test_results["performance_metrics"][metric_name] = []
        
        self.test_results["performance_metrics"][metric_name].append({
            "timestamp": timestamp,
            "value": value
        })
        
        # Append to CSV
        with open(self.performance_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, metric_name, value])
        
        logger.debug(f"Performance metric recorded: {metric_name}={value}")
    
    def finalize_report(self, overall_status: str) -> str:
        """
        Finalize the test report.
        
        Args:
            overall_status: Overall status of the test run
            
        Returns:
            str: Path to the summary report file
        """
        self.test_results["overall_status"] = overall_status
        self.test_results["end_time"] = datetime.now().isoformat()
        
        # Calculate duration
        start_time = datetime.fromisoformat(self.test_results["start_time"])
        end_time = datetime.fromisoformat(self.test_results["end_time"])
        self.test_results["duration_seconds"] = (end_time - start_time).total_seconds()
        
        # Generate performance charts
        self._generate_performance_charts()
        
        # Save final summary
        self._save_summary()
        
        logger.info(f"Test report finalized with overall status: {overall_status}")
        logger.info(f"Report available at: {self.report_dir}")
        
        return self.summary_file
    
    def _save_summary(self) -> None:
        """Save the current summary to the summary file."""
        with open(self.summary_file, 'w') as f:
            json.dump(self.test_results, f, indent=2)
    
    def _generate_performance_charts(self) -> None:
        """Generate charts for performance metrics."""
        if not self.test_results["performance_metrics"]:
            return
        
        charts_dir = os.path.join(self.report_dir, "charts")
        os.makedirs(charts_dir, exist_ok=True)
        
        for metric_name, measurements in self.test_results["performance_metrics"].items():
            if not measurements:
                continue
            
            plt.figure(figsize=(10, 6))
            
            timestamps = [datetime.fromisoformat(m["timestamp"]) for m in measurements]
            values = [m["value"] for m in measurements]
            
            plt.plot(timestamps, values)
            plt.title(f"{metric_name} over time")
            plt.xlabel("Time")
            plt.ylabel(metric_name)
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            chart_file = os.path.join(charts_dir, f"{metric_name.replace(' ', '_')}.png")
            plt.savefig(chart_file)
            plt.close()
            
            logger.debug(f"Performance chart generated: {chart_file}")

def generate_html_report(reporter: TestReporter) -> str:
    """
    Generate an HTML report from the test results.
    
    Args:
        reporter: TestReporter instance with results
        
    Returns:
        str: Path to the HTML report
    """
    html_report_path = os.path.join(reporter.report_dir, "report.html")
    
    # Load test results
    with open(reporter.summary_file, 'r') as f:
        results = json.load(f)
    
    # Basic HTML template
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>E2E Test Report - {results['test_id']}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #333; }}
            .summary {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; }}
            .pass {{ color: green; }}
            .fail {{ color: red; }}
            .skip {{ color: orange; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .charts {{ display: flex; flex-wrap: wrap; margin-top: 20px; }}
            .chart {{ margin: 10px; }}
        </style>
    </head>
    <body>
        <h1>End-to-End Test Report</h1>
        
        <div class="summary">
            <h2>Summary</h2>
            <p><strong>Test ID:</strong> {results['test_id']}</p>
            <p><strong>Status:</strong> <span class="{results['overall_status'].lower()}">{results['overall_status']}</span></p>
            <p><strong>Start Time:</strong> {results['start_time']}</p>
            <p><strong>End Time:</strong> {results['end_time']}</p>
            <p><strong>Duration:</strong> {results['duration_seconds']} seconds</p>
        </div>
        
        <h2>Test Results</h2>
        <table>
            <tr>
                <th>Test Name</th>
                <th>Status</th>
                <th>Timestamp</th>
                <th>Details</th>
            </tr>
    """
    
    # Add test results
    for test_name, test_info in results['tests'].items():
        status_class = test_info['status'].lower()
        details_str = json.dumps(test_info.get('details', {}), indent=2)
        
        html += f"""
            <tr>
                <td>{test_name}</td>
                <td class="{status_class}">{test_info['status']}</td>
                <td>{test_info['timestamp']}</td>
                <td><pre>{details_str}</pre></td>
            </tr>
        """
    
    html += """
        </table>
        
        <h2>Performance Metrics</h2>
    """
    
    # Add performance charts if available
    charts_dir = os.path.join(reporter.report_dir, "charts")
    if os.path.exists(charts_dir):
        html += '<div class="charts">'
        
        for chart_file in os.listdir(charts_dir):
            if chart_file.endswith('.png'):
                chart_path = f"charts/{chart_file}"
                metric_name = chart_file.replace('_', ' ').replace('.png', '')
                
                html += f"""
                    <div class="chart">
                        <h3>{metric_name}</h3>
                        <img src="{chart_path}" alt="{metric_name}" width="500">
                    </div>
                """
        
        html += '</div>'
    else:
        html += '<p>No performance metrics available.</p>'
    
    # Add error summary if there were errors
    if results['error_summary']:
        html += """
            <h2>Error Summary</h2>
            <table>
                <tr>
                    <th>Test Name</th>
                    <th>Timestamp</th>
                    <th>Error</th>
                </tr>
        """
        
        for error in results['error_summary']:
            html += f"""
                <tr>
                    <td>{error['test_name']}</td>
                    <td>{error['timestamp']}</td>
                    <td>{error['error']}</td>
                </tr>
            """
        
        html += '</table>'
    
    html += """
    </body>
    </html>
    """
    
    # Write HTML to file
    with open(html_report_path, 'w') as f:
        f.write(html)
    
    return html_report_path