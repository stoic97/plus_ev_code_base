"""
Command-line script to view and export API performance metrics.
"""

import argparse
from app.monitoring.view_metrics import view_latest_metrics, export_metrics_to_excel

def main():
    parser = argparse.ArgumentParser(description='View and export API performance metrics')
    parser.add_argument('--view', type=int, default=10, help='Number of latest metrics to view')
    parser.add_argument('--export', action='store_true', help='Export metrics to Excel')
    parser.add_argument('--days', type=int, help='Number of days to include in export')
    
    args = parser.parse_args()
    
    if args.view:
        view_latest_metrics(args.view)
    
    if args.export:
        export_metrics_to_excel(args.days)

if __name__ == "__main__":
    main() 