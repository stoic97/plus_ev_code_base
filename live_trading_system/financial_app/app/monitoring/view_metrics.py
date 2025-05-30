"""
Script to view and export API performance metrics.
"""

import os
import logging
from datetime import datetime, timedelta
from sqlalchemy import desc, text
from ..core.database import DatabaseType, db_session
from .performance_tracker import APIPerformanceMetric

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def verify_table_structure():
    """Verify the table structure in the database."""
    try:
        with db_session(DatabaseType.POSTGRESQL) as session:
            # Check if table exists
            table_exists = session.execute(
                text("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'api_performance_metrics')")
            ).scalar()
            
            if not table_exists:
                print("Table does not exist in the database!")
                return False
            
            # Get column information
            columns = session.execute(text("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'api_performance_metrics'
                ORDER BY ordinal_position;
            """)).fetchall()
            
            print("\nTable Structure:")
            print("-" * 50)
            for col in columns:
                print(f"{col[0]:<20} {col[1]}")
            
            # Get table stats
            stats = session.execute(text("""
                SELECT 
                    (SELECT COUNT(*) FROM api_performance_metrics) as total_rows,
                    (SELECT COUNT(*) FROM api_performance_metrics WHERE error IS NOT NULL) as error_count,
                    (SELECT COUNT(DISTINCT endpoint) FROM api_performance_metrics) as unique_endpoints
            """)).fetchone()
            
            print("\nTable Statistics:")
            print("-" * 50)
            print(f"Total Records: {stats[0]}")
            print(f"Error Records: {stats[1]}")
            print(f"Unique Endpoints: {stats[2]}")
            
            return True
    except Exception as e:
        print(f"Error verifying table structure: {str(e)}")
        logger.exception("Failed to verify table structure")
        return False

def view_latest_metrics(limit: int = 10):
    """View the most recent performance metrics."""
    try:
        print("\nVerifying database table structure...")
        if not verify_table_structure():
            return
        
        print("\nQuerying latest metrics...")
        with db_session(DatabaseType.POSTGRESQL) as session:
            # Try direct SQL query first
            raw_metrics = session.execute(text("""
                SELECT 
                    endpoint, 
                    method, 
                    status_code, 
                    response_time, 
                    timestamp,
                    error
                FROM api_performance_metrics 
                ORDER BY timestamp DESC 
                LIMIT :limit
            """), {"limit": limit}).fetchall()
            
            if not raw_metrics:
                print("No metrics found using direct SQL query")
                return
            
            print("\nLatest API Performance Metrics:")
            print("-" * 100)
            print(f"{'Timestamp':<25} {'Endpoint':<30} {'Method':<8} {'Status':<8} {'Time (s)':<10} {'Error'}")
            print("-" * 100)
            
            for metric in raw_metrics:
                error = metric.error[:50] + "..." if metric.error and len(metric.error) > 50 else metric.error or ""
                print(
                    f"{metric.timestamp.strftime('%Y-%m-%d %H:%M:%S'):<25} "
                    f"{metric.endpoint:<30} "
                    f"{metric.method:<8} "
                    f"{str(metric.status_code):<8} "
                    f"{metric.response_time:.3f}s{'':6} "
                    f"{error}"
                )
                
    except Exception as e:
        print(f"Error while viewing metrics: {str(e)}")
        logger.exception("Failed to view metrics")

def export_metrics_to_excel(days: int = None):
    """
    Export metrics to Excel file.
    
    Args:
        days: Optional number of days to limit the export (None for all data)
    """
    from .performance_tracker import export_metrics_to_csv
    import pandas as pd
    
    # Create exports directory if it doesn't exist
    os.makedirs("exports", exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"exports/api_metrics_{timestamp}.csv"
    excel_path = f"exports/api_metrics_{timestamp}.xlsx"
    
    # Export to CSV first
    export_metrics_to_csv(csv_path, days)
    
    # Convert to Excel with formatting
    df = pd.read_csv(csv_path)
    
    # Convert timestamp strings to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Create Excel writer
    writer = pd.ExcelWriter(excel_path, engine='xlsxwriter')
    
    # Write to Excel with formatting
    df.to_excel(writer, sheet_name='API Metrics', index=False)
    
    # Get workbook and worksheet objects
    workbook = writer.book
    worksheet = writer.sheets['API Metrics']
    
    # Add formats
    header_format = workbook.add_format({
        'bold': True,
        'text_wrap': True,
        'valign': 'top',
        'bg_color': '#D9EAD3',
        'border': 1
    })
    
    # Format the header row
    for col_num, value in enumerate(df.columns.values):
        worksheet.write(0, col_num, value, header_format)
    
    # Adjust column widths
    worksheet.set_column('A:A', 20)  # timestamp
    worksheet.set_column('B:B', 40)  # endpoint
    worksheet.set_column('C:C', 10)  # method
    worksheet.set_column('D:D', 10)  # status_code
    worksheet.set_column('E:E', 12)  # response_time
    worksheet.set_column('F:F', 50)  # error
    
    # Save the Excel file
    writer.close()
    
    # Remove the temporary CSV file
    os.remove(csv_path)
    
    print(f"\nMetrics exported to: {excel_path}")
    return excel_path

if __name__ == "__main__":
    # View latest metrics
    print("\nShowing latest 10 metrics:")
    view_latest_metrics(10)
    
    # Export to Excel
    print("\nExporting metrics to Excel...")
    export_metrics_to_excel() 