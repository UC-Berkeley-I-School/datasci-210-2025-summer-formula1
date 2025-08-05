#!/usr/bin/env python3
"""
Scheduler for ETL pipeline runs
"""

import schedule
import time
import subprocess
import logging
from datetime import datetime
import os
from pathlib import Path

class ETLScheduler:
    def __init__(self):
        self.setup_logging()
        self.script_path = Path(__file__).parent / "etl_cache_pipeline.py"
        
    def setup_logging(self):
        """Setup logging for scheduler"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('etl_scheduler.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def run_etl_pipeline(self):
        """Execute the ETL pipeline script"""
        try:
            self.logger.info("Starting scheduled ETL pipeline run...")
            
            # Run the ETL cache script
            result = subprocess.run([
                'python', str(self.script_path)
            ], capture_output=True, text=True, timeout=3600)  # 1 hour timeout
            
            if result.returncode == 0:
                self.logger.info("ETL pipeline completed successfully")
                self.logger.info(f"Output: {result.stdout}")
            else:
                self.logger.error(f"ETL pipeline failed with return code: {result.returncode}")
                self.logger.error(f"Error: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            self.logger.error("ETL pipeline timed out after 1 hour")
        except Exception as e:
            self.logger.error(f"Failed to run ETL pipeline: {str(e)}")
    
    def setup_schedule(self):
        """Setup the schedule based on configuration"""
        schedule_type = os.getenv('ETL_SCHEDULE', 'daily')
        
        if schedule_type == 'daily':
            # Run daily at 2 AM
            schedule.every().day.at("02:00").do(self.run_etl_pipeline)
            self.logger.info("Scheduled daily ETL runs at 2:00 AM")
            
        elif schedule_type == 'weekly':
            # Run weekly on Sunday at 1 AM
            schedule.every().sunday.at("01:00").do(self.run_etl_pipeline)
            self.logger.info("Scheduled weekly ETL runs on Sunday at 1:00 AM")
            
        elif schedule_type == 'hourly':
            # Run every hour (for testing/high-frequency updates)
            schedule.every().hour.do(self.run_etl_pipeline)
            self.logger.info("Scheduled hourly ETL runs")
            
        else:
            self.logger.warning(f"Unknown schedule type: {schedule_type}. No automatic scheduling.")
    
    def run_scheduler(self):
        """Main scheduler loop"""
        self.setup_schedule()
        
        self.logger.info("ETL Scheduler started. Waiting for scheduled runs...")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            self.logger.info("Scheduler stopped by user")
        except Exception as e:
            self.logger.error(f"Scheduler error: {str(e)}")

def main():
    """Main entry point"""
    scheduler = ETLScheduler()
    
    # Check if we want to run immediately or just start scheduler
    if len(os.sys.argv) > 1 and os.sys.argv[1] == '--run-now':
        scheduler.run_etl_pipeline()
    else:
        scheduler.run_scheduler()

if __name__ == "__main__":
    main()