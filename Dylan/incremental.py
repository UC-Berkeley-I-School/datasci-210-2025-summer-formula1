#!/usr/bin/env python3

import os
import psycopg2
import numpy as np
from tqdm import tqdm
import logging
import fastf1
import pandas as pd
import signal
import sys
from datetime import datetime
import tempfile
import shutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Predefined F1 schedules to avoid API calls for schedule data
F1_SCHEDULES = {
    2022: [
        "Bahrain Grand Prix", "Saudi Arabian Grand Prix", "Australian Grand Prix",
        "Emilia Romagna Grand Prix", "Miami Grand Prix", "Spanish Grand Prix", 
        "Monaco Grand Prix", "Azerbaijan Grand Prix", "Canadian Grand Prix",
        "British Grand Prix", "Austrian Grand Prix", "French Grand Prix",
        "Hungarian Grand Prix", "Belgian Grand Prix", "Dutch Grand Prix",
        "Italian Grand Prix", "Singapore Grand Prix", "Japanese Grand Prix",
        "United States Grand Prix", "Mexico City Grand Prix", "São Paulo Grand Prix",
        "Abu Dhabi Grand Prix"
    ],
    2023: [
        "Bahrain Grand Prix", "Saudi Arabian Grand Prix", "Australian Grand Prix",
        "Azerbaijan Grand Prix", "Miami Grand Prix", "Monaco Grand Prix",
        "Spanish Grand Prix", "Canadian Grand Prix", "Austrian Grand Prix",
        "British Grand Prix", "Hungarian Grand Prix", "Belgian Grand Prix",
        "Dutch Grand Prix", "Italian Grand Prix", "Singapore Grand Prix",
        "Japanese Grand Prix", "Qatar Grand Prix", "United States Grand Prix", 
        "Mexico City Grand Prix", "São Paulo Grand Prix", "Las Vegas Grand Prix",
        "Abu Dhabi Grand Prix"
    ],
    2024: [
        "Bahrain Grand Prix", "Saudi Arabian Grand Prix", "Australian Grand Prix",
        "Japanese Grand Prix", "Chinese Grand Prix", "Miami Grand Prix",
        "Emilia Romagna Grand Prix", "Monaco Grand Prix", "Canadian Grand Prix",
        "Spanish Grand Prix", "Austrian Grand Prix", "British Grand Prix",
        "Hungarian Grand Prix", "Belgian Grand Prix", "Dutch Grand Prix",
        "Italian Grand Prix", "Azerbaijan Grand Prix", "Singapore Grand Prix",
        "United States Grand Prix", "Mexico City Grand Prix", "São Paulo Grand Prix",
        "Las Vegas Grand Prix", "Qatar Grand Prix", "Abu Dhabi Grand Prix"
    ]
}

SEASONS = [2022, 2023, 2024]
SESSION_TYPES = ['R']
WINDOW_SIZE = 30

class SmartIncrementalETL:
    def __init__(self):
        self.conn = None
        signal.signal(signal.SIGINT, self.signal_handler)
        
    def signal_handler(self, sig, frame):
        logger.info("🛑 Shutdown signal received - cleaning up...")
        if self.conn:
            self.conn.close()
        sys.exit(0)
    
    def get_connection(self):
        self.conn = psycopg2.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            port=os.getenv('DB_PORT', '5432'),
            database=os.getenv('DB_NAME', 'racing_telemetry'),
            user=os.getenv('DB_USER', 'racing_user'),
            password=os.getenv('DB_PASSWORD', 'racing_password')
        )
        return self.conn
    
    def normalize_race_name(self, name):
        """Normalize race names for comparison"""
        if not name:
            return ""
        normalized = name.lower().strip()
        # Handle common variations
        normalized = normalized.replace('grand prix', 'gp')
        normalized = normalized.replace('  ', ' ')
        # Remove extra words that might cause mismatches
        normalized = normalized.replace(' gp', '')
        return normalized
    
    def get_existing_races_smart(self):
        """Get existing races from database with smart analysis"""
        logger.info("🔍 Analyzing existing races in database...")
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT DISTINCT 
                EXTRACT(YEAR FROM s.start_time) as year,
                s.race_name,
                s.track_name,
                COUNT(t.driver_id) as telemetry_count,
                MIN(t.time) as first_record,
                MAX(t.time) as last_record
            FROM sessions s
            LEFT JOIN telemetry t ON s.session_id = t.session_id
            WHERE s.session_type = 'Race' 
              AND s.start_time IS NOT NULL
            GROUP BY EXTRACT(YEAR FROM s.start_time), s.race_name, s.track_name
            ORDER BY year, s.race_name
        """)
        
        existing_races = cursor.fetchall()
        cursor.close()
        conn.close()
        
        # Organize by year
        races_by_year = {}
        for year, race_name, track_name, tel_count, first_rec, last_rec in existing_races:
            year = int(year) if year else 0
            if year not in races_by_year:
                races_by_year[year] = []
            
            races_by_year[year].append({
                'race_name': race_name,
                'track_name': track_name,
                'telemetry_count': tel_count,
                'first_record': first_rec,
                'last_record': last_rec,
                'normalized_name': self.normalize_race_name(race_name)
            })
        
        return races_by_year
    
    def identify_missing_races_smart(self, existing_races):
        """Identify missing races using predefined schedules (no API calls)"""
        logger.info("🎯 Identifying missing races (using offline schedule data)...")
        
        missing_races = []
        
        for year in SEASONS:
            if year not in F1_SCHEDULES:
                logger.warning(f"⚠️  No schedule data for {year}")
                continue
                
            logger.info(f"\n📅 Analyzing {year} season:")
            existing_year = existing_races.get(year, [])
            existing_normalized = {race['normalized_name'] for race in existing_year}
            
            logger.info(f"   Database has {len(existing_year)} races:")
            for race in existing_year:
                status = "✅" if race['telemetry_count'] > 1000 else "⚠️" if race['telemetry_count'] > 0 else "❌"
                logger.info(f"      {status} {race['race_name']} ({race['telemetry_count']:,} records)")
            
            logger.info(f"   Expected {len(F1_SCHEDULES[year])} races from F1 calendar")
            
            year_missing = []
            for round_num, scheduled_race in enumerate(F1_SCHEDULES[year], 1):
                normalized_scheduled = self.normalize_race_name(scheduled_race)
                
                # Check if this race exists with good data
                found_with_good_data = False
                for existing_race in existing_year:
                    if existing_race['normalized_name'] == normalized_scheduled:
                        if existing_race['telemetry_count'] > 1000:  # Good data threshold
                            found_with_good_data = True
                            break
                
                if not found_with_good_data:
                    year_missing.append({
                        'year': year,
                        'round': round_num,
                        'race_name': scheduled_race,
                        'normalized_name': normalized_scheduled
                    })
            
            if year_missing:
                logger.info(f"   🚫 Missing {len(year_missing)} races:")
                for race in year_missing:
                    logger.info(f"      Round {race['round']:2d}: {race['race_name']}")
            else:
                logger.info(f"   ✅ All races present with good data!")
            
            missing_races.extend(year_missing)
        
        return missing_races
    
    def fetch_session_smart(self, year, race_name):
        """Fetch session with temporary cache and rate limit awareness"""
        logger.info(f"🏎️  Fetching {year} {race_name}...")
        
        # Create temporary cache directory
        temp_dir = tempfile.mkdtemp(prefix='f1_smart_')
        
        try:
            # Set up minimal cache
            fastf1.Cache.enable_cache(temp_dir)
            
            # Try different race name formats
            race_attempts = [
                race_name,
                race_name.replace('Grand Prix', 'GP'),
                race_name.replace('GP', 'Grand Prix'),
                race_name.split(' Grand Prix')[0] if 'Grand Prix' in race_name else race_name
            ]
            
            for i, attempt_name in enumerate(race_attempts, 1):
                logger.info(f"   Attempt {i}: '{attempt_name}'")
                try:
                    session = fastf1.get_session(year, attempt_name, 'R')
                    session.load()
                    logger.info(f"✅ Successfully loaded {year} {race_name}")
                    return session
                except Exception as e:
                    logger.warning(f"   Failed: {str(e)[:100]}...")
                    continue
            
            logger.error(f"❌ All attempts failed for {year} {race_name}")
            return None
            
        except Exception as e:
            logger.error(f"❌ Fatal error: {e}")
            return None
        finally:
            # Always clean up temp directory
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except:
                pass
    
    def extract_windows(self, session):
        """Extract telemetry windows from session"""
        logger.info("📊 Extracting telemetry windows...")
        session_id = f"{session.date.year}_{session.event.EventName.replace(' ', '_')}_R"
        
        windows = []
        driver_count = len(session.drivers)
        logger.info(f"   Processing {driver_count} drivers...")
        
        for i, driver_number in enumerate(session.drivers, 1):
            logger.info(f"   Driver {i}/{driver_count}: #{driver_number}")
            
            try:
                driver_data = session.laps.pick_drivers(driver_number)
                if driver_data.empty:
                    logger.warning(f"   No lap data for driver #{driver_number}")
                    continue
                
                telemetry = driver_data.get_telemetry()
                if telemetry.empty:
                    logger.warning(f"   No telemetry data for driver #{driver_number}")
                    continue
                
                start_time = telemetry['Time'].min()
                end_time = telemetry['Time'].max()
                current_time = start_time
                driver_windows = 0
                
                while current_time + pd.Timedelta(seconds=WINDOW_SIZE) <= end_time:
                    window_end = current_time + pd.Timedelta(seconds=WINDOW_SIZE)
                    mask = (telemetry['Time'] >= current_time) & (telemetry['Time'] <= window_end)
                    window_data = telemetry.loc[mask]
                    
                    if len(window_data) > 10:
                        features = {
                            'Speed': float(window_data['Speed'].mean()) if 'Speed' in window_data else 0.0,
                            'Throttle': float(window_data['Throttle'].mean()) if 'Throttle' in window_data else 0.0,
                            'Brake': bool(window_data['Brake'].any()) if 'Brake' in window_data else False,
                            'RPM': float(window_data['RPM'].mean()) if 'RPM' in window_data else 0.0,
                            'Gear': int(window_data['nGear'].mode().iloc[0]) if 'nGear' in window_data and not window_data['nGear'].empty else 0,
                        }
                        
                        coords = {
                            'X': float(window_data['X'].iloc[0]) if 'X' in window_data and not window_data['X'].empty else 0.0,
                            'Y': float(window_data['Y'].iloc[0]) if 'Y' in window_data and not window_data['Y'].empty else 0.0,
                            'Z': float(window_data['Z'].iloc[0]) if 'Z' in window_data and not window_data['Z'].empty else 0.0,
                        }
                        
                        windows.append({
                            'session_id': session_id,
                            'driver_number': int(driver_number),
                            'start_time': (session.date + current_time).isoformat(),
                            'features': features,
                            'coordinates': coords,
                            'y_true': 0
                        })
                        driver_windows += 1
                    
                    current_time += pd.Timedelta(seconds=WINDOW_SIZE * 0.5)
                
                logger.info(f"   Driver #{driver_number}: {driver_windows} windows extracted")
                
            except Exception as e:
                logger.error(f"   Error processing driver #{driver_number}: {e}")
                continue
        
        logger.info(f"✅ Total windows extracted: {len(windows)}")
        return windows
    
    def insert_data(self, windows):
        """Insert data into database"""
        if not windows:
            logger.warning("⚠️  No windows to insert")
            return
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Insert drivers
            drivers = set((w['driver_number'], f"Driver {w['driver_number']}") for w in windows)
            for driver_num, driver_name in drivers:
                cursor.execute("""
                    INSERT INTO drivers (driver_id, name, car_number)
                    VALUES (%s, %s, %s) 
                    ON CONFLICT (driver_id) DO UPDATE SET
                        name = EXCLUDED.name,
                        car_number = EXCLUDED.car_number
                """, (f"driver_{driver_num}", driver_name, driver_num))
            
            # Insert sessions
            sessions = {}
            for w in windows:
                sid = w['session_id']
                if sid not in sessions:
                    parts = sid.split('_')
                    sessions[sid] = {
                        'race_name': ' '.join(parts[1:-1]).replace('_', ' '),
                        'session_type': 'Race',
                        'track_name': parts[1].replace('_', ' ') if len(parts) > 1 else 'Unknown',
                        'start_time': w['start_time']
                    }
            
            for sid, data in sessions.items():
                cursor.execute("""
                    INSERT INTO sessions (session_id, race_name, session_type, track_name, start_time)
                    VALUES (%s, %s, %s, %s, %s) 
                    ON CONFLICT (session_id) DO UPDATE SET
                        race_name = EXCLUDED.race_name,
                        track_name = EXCLUDED.track_name,
                        start_time = EXCLUDED.start_time
                """, (sid, data['race_name'], data['session_type'], data['track_name'], data['start_time']))
            
            # Insert telemetry and predictions
            for i, w in enumerate(tqdm(windows, desc="Inserting telemetry"), 1):
                driver_id = f"driver_{w['driver_number']}"
                
                # Insert telemetry
                cursor.execute("""
                    INSERT INTO telemetry (
                        time, session_id, driver_id, speed, rpm, ngear, throttle, brake,
                        x, y, z, track_status
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    w['start_time'], w['session_id'], driver_id,
                    w['features']['Speed'], w['features']['RPM'], w['features']['Gear'],
                    w['features']['Throttle'], w['features']['Brake'],
                    w['coordinates']['X'], w['coordinates']['Y'], w['coordinates']['Z'],
                    'Normal'
                ))
                
                # Insert prediction
                cursor.execute("""
                    INSERT INTO predictions (
                        session_id, driver_id, model_version, predicted_track_status, 
                        confidence, prediction_start_time, prediction_end_time
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    w['session_id'], driver_id, 'safety_car_predictor_v1.0',
                    'Normal', 0.85, w['start_time'], w['start_time']
                ))
                
                if i % 1000 == 0:
                    conn.commit()
                    logger.info(f"   Committed {i}/{len(windows)} records")
            
            conn.commit()
            logger.info("✅ All data inserted successfully")
            
        except Exception as e:
            logger.error(f"❌ Error inserting data: {e}")
            conn.rollback()
            raise
        finally:
            cursor.close()
            conn.close()
    
    def run_smart_incremental(self, max_races=None, specific_year=None):
        """Run smart incremental ETL"""
        logger.info("🚀 Smart Incremental F1 ETL Starting")
        logger.info("🎯 Uses offline schedule data to avoid rate limits")
        
        # Test connection
        try:
            conn = self.get_connection()
            conn.close()
            logger.info("✅ Database connection successful")
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            return 1
        
        # Get existing races
        existing_races = self.get_existing_races_smart()
        
        # Identify missing races
        missing_races = self.identify_missing_races_smart(existing_races)
        
        if not missing_races:
            logger.info("🎉 No missing races found! Database is complete.")
            return 0
        
        # Filter by year if specified
        if specific_year:
            missing_races = [r for r in missing_races if r['year'] == specific_year]
            logger.info(f"🎯 Filtering to {specific_year}: {len(missing_races)} races")
        
        # Limit number of races if specified
        if max_races and len(missing_races) > max_races:
            missing_races = missing_races[:max_races]
            logger.info(f"🎯 Limiting to {max_races} races to avoid rate limits")
        
        logger.info(f"\n🏁 Will process {len(missing_races)} missing races:")
        for race in missing_races:
            logger.info(f"   {race['year']} Round {race['round']:2d}: {race['race_name']}")
        
        # Confirm before proceeding
        if len(missing_races) > 5:
            response = input(f"\nProcess {len(missing_races)} races? This may take time and use API quota. [y/N]: ")
            if response.lower() != 'y':
                logger.info("❌ Cancelled by user")
                return 0
        
        # Process missing races
        processed_count = 0
        for race in missing_races:
            logger.info(f"\n🏁 Processing: {race['year']} {race['race_name']} (Round {race['round']})")
            
            try:
                session = self.fetch_session_smart(race['year'], race['race_name'])
                
                if session:
                    windows = self.extract_windows(session)
                    if windows:
                        self.insert_data(windows)
                        processed_count += 1
                        logger.info(f"✅ {race['race_name']} completed: {len(windows)} windows added")
                    else:
                        logger.warning(f"⚠️  No telemetry extracted for {race['race_name']}")
                else:
                    logger.warning(f"⚠️  Could not fetch {race['race_name']}")
                    
            except Exception as e:
                logger.error(f"❌ Error processing {race['race_name']}: {e}")
                if "RateLimitExceededError" in str(e):
                    logger.error("🛑 Rate limit exceeded! Wait before continuing.")
                    break
                continue
        
        logger.info(f"\n🎉 Smart Incremental ETL Complete!")
        logger.info(f"📊 Successfully processed {processed_count}/{len(missing_races)} races")
        
        return 0

def main():
    """Main function with options"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Smart Incremental F1 ETL')
    parser.add_argument('--year', type=int, choices=[2022, 2023, 2024], 
                       help='Process only specific year')
    parser.add_argument('--max-races', type=int, default=5,
                       help='Maximum races to process (default: 5 to avoid rate limits)')
    parser.add_argument('--list-only', action='store_true',
                       help='Only list missing races, don\'t download')
    
    args = parser.parse_args()
    
    etl = SmartIncrementalETL()
    
    if args.list_only:
        logger.info("📋 LIST MODE: Identifying missing races only")
        existing = etl.get_existing_races_smart()
        missing = etl.identify_missing_races_smart(existing)
        
        if args.year:
            missing = [r for r in missing if r['year'] == args.year]
        
        logger.info(f"\n📋 SUMMARY: {len(missing)} missing races found")
        return 0
    else:
        return etl.run_smart_incremental(max_races=args.max_races, specific_year=args.year)

if __name__ == "__main__":
    exit(main())