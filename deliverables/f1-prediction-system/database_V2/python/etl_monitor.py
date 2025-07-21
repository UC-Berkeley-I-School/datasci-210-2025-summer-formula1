#!/usr/bin/env python3
"""
ETL Pipeline Monitoring and Status Tool
"""

import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import argparse
import json
from pathlib import Path

class ETLMonitor:
    def __init__(self, db_path=None):
        self.db_path = db_path or Path("../deliverables/f1-prediction-system/database/data/f1_data.db")
    
    def get_cache_status(self, days=7):
        """Get cache status for recent runs"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = '''
                SELECT pipeline_run_id, start_time, end_time, status, 
                       records_processed, error_message,
                       ROUND((julianday(end_time) - julianday(start_time)) * 24 * 60, 2) as duration_minutes
                FROM cache_status 
                WHERE start_time >= datetime('now', '-{} days')
                ORDER BY start_time DESC
            '''.format(days)
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            return df
            
        except Exception as e:
            print(f"Error getting cache status: {e}")
            return None
    
    def get_data_summary(self):
        """Get summary of cached data"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            summaries = {}
            
            # Race results summary
            race_query = '''
                SELECT COUNT(*) as total_records,
                       COUNT(DISTINCT race_id) as unique_races,
                       COUNT(DISTINCT driver_id) as unique_drivers,
                       MIN(created_at) as oldest_record,
                       MAX(created_at) as newest_record
                FROM race_results
            '''
            summaries['race_results'] = pd.read_sql_query(race_query, conn).iloc[0].to_dict()
            
            # Driver features summary
            driver_query = '''
                SELECT COUNT(*) as total_records,
                       COUNT(DISTINCT driver_id) as unique_drivers,
                       COUNT(DISTINCT season) as seasons_covered,
                       MIN(created_at) as oldest_record,
                       MAX(created_at) as newest_record
                FROM driver_features
            '''
            summaries['driver_features'] = pd.read_sql_query(driver_query, conn).iloc[0].to_dict()
            
            # Constructor features summary
            constructor_query = '''
                SELECT COUNT(*) as total_records,
                       COUNT(DISTINCT constructor_id) as unique_constructors,
                       COUNT(DISTINCT season) as seasons_covered,
                       MIN(created_at) as oldest_record,
                       MAX(created_at) as newest_record
                FROM constructor_features
            '''
            summaries['constructor_features'] = pd.read_sql_query(constructor_query, conn).iloc[0].to_dict()
            
            conn.close()
            return summaries
            
        except Exception as e:
            print(f"Error getting data summary: {e}")
            return None
    
    def check_data_freshness(self, hours=24):
        """Check if data is fresh (updated within specified hours)"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = '''
                SELECT MAX(created_at) as latest_update,
                       ROUND((julianday('now') - julianday(MAX(created_at))) * 24, 2) as hours_since_update
                FROM (
                    SELECT created_at FROM race_results
                    UNION ALL
                    SELECT created_at FROM driver_features
                    UNION ALL
                    SELECT created_at FROM constructor_features
                )
            '''
            
            result = pd.read_sql_query(query, conn).iloc[0]
            conn.close()
            
            is_fresh = result['hours_since_update'] <= hours
            
            return {
                'is_fresh': is_fresh,
                'latest_update': result['latest_update'],
                'hours_since_update': result['hours_since_update'],
                'threshold_hours': hours
            }
            
        except Exception as e:
            print(f"Error checking data freshness: {e}")
            return None
    
    def get_failed_runs(self, days=7):
        """Get failed pipeline runs"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = '''
                SELECT pipeline_run_id, start_time, error_message
                FROM cache_status 
                WHERE status = 'FAILED' 
                AND start_time >= datetime('now', '-{} days')
                ORDER BY start_time DESC
            '''.format(days)
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            return df
            
        except Exception as e:
            print(f"Error getting failed runs: {e}")
            return None
    
    def display_status(self, days=7):
        """Display comprehensive status"""
        print("=" * 60)
        print("F1 ETL Pipeline Status Report")
        print("=" * 60)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Recent runs
        print("Recent Pipeline Runs:")
        print("-" * 40)
        cache_status = self.get_cache_status(days)
        if cache_status is not None and not cache_status.empty:
            print(cache_status.to_string(index=False))
        else:
            print("No recent runs found or error accessing data")
        print()
        
        # Data freshness
        print("Data Freshness:")
        print("-" * 40)
        freshness = self.check_data_freshness()
        if freshness:
            status = "✅ FRESH" if freshness['is_fresh'] else "⚠️  STALE"
            print(f"Status: {status}")
            print(f"Latest Update: {freshness['latest_update']}")
            print(f"Hours Since Update: {freshness['hours_since_update']}")
        else:
            print("Error checking data freshness")
        print()
        
        # Data summary
        print("Data Summary:")
        print("-" * 40)
        summary = self.get_data_summary()
        if summary:
            for table, stats in summary.items():
                print(f"\n{table.upper()}:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
        else:
            print("Error getting data summary")
        print()
        
        # Failed runs
        print("Recent Failed Runs:")
        print("-" * 40)
        failed_runs = self.get_failed_runs(days)
        if failed_runs is not None and not failed_runs.empty:
            print(failed_runs.to_string(index=False))
        else:
            print("No failed runs in recent days ✅")

def main():
    parser = argparse.ArgumentParser(description='Monitor F1 ETL Pipeline')
    parser.add_argument('--days', type=int, default=7, help='Days to look back for status')
    parser.add_argument('--db-path', type=str, help='Path to database file')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    
    args = parser.parse_args()
    
    monitor = ETLMonitor(args.db_path)
    
    if args.json:
        # JSON output for programmatic use
        result = {
            'cache_status': monitor.get_cache_status(args.days).to_dict('records') if monitor.get_cache_status(args.days) is not None else [],
            'data_summary': monitor.get_data_summary(),
            'data_freshness': monitor.check_data_freshness(),
            'failed_runs': monitor.get_failed_runs(args.days).to_dict('records') if monitor.get_failed_runs(args.days) is not None else []
        }
        print(json.dumps(result, indent=2, default=str))
    else:
        # Human-readable output
        monitor.display_status(args.days)

if __name__ == "__main__":
    main()