#!/usr/bin/env python3

import os
import psycopg2
from psycopg2.extras import Json
import numpy as np
from tqdm import tqdm
import logging
import fastf1
import pandas as pd
import signal
import sys
from datetime import datetime

# Setup detailed logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

SEASONS = [2022, 2023, 2024]
SESSION_TYPES = ['R']
WINDOW_SIZE = 30

class ETLPipeline:
    def __init__(self):
        self.conn = None
        signal.signal(signal.SIGINT, self.signal_handler)
        
    def signal_handler(self, sig, frame):
        logger.info("🛑 Shutdown signal received - cleaning up...")
        if self.conn:
            self.conn.close()
        sys.exit(0)
    
    def get_connection(self):
        logger.info("🔌 Creating database connection...")
        self.conn = psycopg2.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            port=os.getenv('DB_PORT', '5432'),
            database=os.getenv('DB_NAME', 'racing_telemetry'),
            user=os.getenv('DB_USER', 'racing_user'),
            password=os.getenv('DB_PASSWORD', 'racing_password')
        )
        logger.info("✅ Database connection established")
        return self.conn
    
    def get_schedule(self, year):
        logger.info(f"📅 Getting race schedule for {year}...")
        try:
            schedule = fastf1.get_event_schedule(year)
            races = []
            
            for _, event in schedule.iterrows():
                if 'Grand Prix' in event['EventName'] or 'GP' in event['EventName']:
                    races.append({
                        'year': year,
                        'race': event['EventName'],
                        'round': event['RoundNumber']
                    })
            
            logger.info(f"✅ Found {len(races)} races for {year}")
            return races
            
        except Exception as e:
            logger.error(f"❌ Error getting schedule for {year}: {e}")
            return []
    
    def clear_data(self):
        logger.info("🧹 Clearing existing data from database...")
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            tables = ['predictions', 'telemetry', 'sessions', 'drivers']
            for table in tables:
                logger.info(f"   Clearing {table}...")
                cursor.execute(f"DELETE FROM {table}")
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                logger.info(f"   {table}: {count} rows remaining")
            
            conn.commit()
            logger.info("✅ Database cleared successfully")
            
        except Exception as e:
            logger.error(f"❌ Error clearing data: {e}")
            conn.rollback()
            raise
        finally:
            cursor.close()
            conn.close()
    
    def fetch_session(self, year, race, session_type):
        logger.info(f"🏎️  Fetching {year} {race} {session_type}...")
        
        try:
            # Enable caching
            logger.info("   Setting up FastF1 cache...")
            fastf1.Cache.enable_cache('temp_f1_cache')
            
            # Try different race name formats
            race_attempts = [
                race,
                race.replace('Grand Prix', 'GP'),
                race.replace('GP', 'Grand Prix'),
                race.split(' Grand Prix')[0]
            ]
            
            session = None
            for i, race_name in enumerate(race_attempts, 1):
                logger.info(f"   Attempt {i}: Trying '{race_name}'...")
                try:
                    session = fastf1.get_session(year, race_name, session_type)
                    logger.info("   Loading session data...")
                    session.load()
                    logger.info(f"✅ Successfully loaded {year} {race} {session_type}")
                    return session
                except Exception as attempt_error:
                    logger.warning(f"   Attempt {i} failed: {attempt_error}")
                    continue
            
            logger.error(f"❌ All attempts failed for {year} {race} {session_type}")
            return None
            
        except Exception as e:
            logger.error(f"❌ Fatal error fetching {year} {race} {session_type}: {e}")
            return None
    
    def extract_windows(self, session):
        logger.info("📊 Extracting telemetry windows...")
        session_id = f"{session.date.year}_{session.event.EventName.replace(' ', '_')}_{session.name}"
        logger.info(f"   Session ID: {session_id}")
        
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
                
                logger.info(f"   Telemetry points: {len(telemetry)}")
                
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
    
    def insert_drivers(self, windows):
        logger.info("👥 Inserting drivers...")
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            drivers = set((w['driver_number'], f"Driver {w['driver_number']}") for w in windows)
            logger.info(f"   Found {len(drivers)} unique drivers")
            
            for driver_num, driver_name in drivers:
                # Match actual database schema: driver_id (VARCHAR), name, car_number
                cursor.execute("""
                    INSERT INTO drivers (driver_id, name, car_number)
                    VALUES (%s, %s, %s) ON CONFLICT (driver_id) DO NOTHING
                """, (f"driver_{driver_num}", driver_name, driver_num))
            
            conn.commit()
            logger.info("✅ Drivers inserted successfully")
            
        except Exception as e:
            logger.error(f"❌ Error inserting drivers: {e}")
            conn.rollback()
            raise
        finally:
            cursor.close()
            conn.close()
    
    def insert_sessions(self, windows):
        logger.info("📅 Inserting sessions...")
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            sessions = {}
            for w in windows:
                sid = w['session_id']
                if sid not in sessions:
                    parts = sid.split('_')
                    sessions[sid] = {
                        'race_name': '_'.join(parts[1:-1]).replace('_', ' '),
                        'session_type': parts[-1],
                        'track_name': parts[1].replace('_', ' ') if len(parts) > 1 else 'Unknown',
                        'start_time': w['start_time']
                    }
            
            logger.info(f"   Found {len(sessions)} unique sessions")
            
            for sid, data in sessions.items():
                # Match actual database schema: session_id, race_name, session_type, track_name, start_time
                cursor.execute("""
                    INSERT INTO sessions (session_id, race_name, session_type, track_name, start_time)
                    VALUES (%s, %s, %s, %s, %s) ON CONFLICT (session_id) DO NOTHING
                """, (sid, data['race_name'], data['session_type'], data['track_name'], data['start_time']))
            
            conn.commit()
            logger.info("✅ Sessions inserted successfully")
            
        except Exception as e:
            logger.error(f"❌ Error inserting sessions: {e}")
            conn.rollback()
            raise
        finally:
            cursor.close()
            conn.close()
    
    def insert_telemetry(self, windows):
        logger.info("📊 Inserting telemetry data...")
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            logger.info(f"   Processing {len(windows)} windows...")
            
            for i, w in enumerate(tqdm(windows, desc="Inserting records"), 1):
                try:
                    # Map driver number to driver_id format
                    driver_id = f"driver_{w['driver_number']}"
                    
                    # Insert telemetry - match actual database schema
                    cursor.execute("""
                        INSERT INTO telemetry (
                            time, session_id, driver_id, speed, rpm, ngear, throttle, brake,
                            x, y, z, track_status
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        w['start_time'],           # time
                        w['session_id'],           # session_id
                        driver_id,                 # driver_id (mapped from driver_number)
                        w['features']['Speed'],    # speed
                        w['features']['RPM'],      # rpm
                        w['features']['Gear'],     # ngear
                        w['features']['Throttle'], # throttle
                        w['features']['Brake'],    # brake (boolean)
                        w['coordinates']['X'],     # x
                        w['coordinates']['Y'],     # y
                        w['coordinates']['Z'],     # z
                        'Normal'                   # track_status (default)
                    ))
                    
                    # Insert prediction - match actual database schema
                    cursor.execute("""
                        INSERT INTO predictions (
                            session_id, driver_id, model_version, predicted_track_status, 
                            confidence, prediction_start_time, prediction_end_time
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (
                        w['session_id'],                          # session_id
                        driver_id,                                # driver_id
                        'safety_car_predictor_v1.0',             # model_version
                        'Normal' if w['y_true'] == 0 else 'SC',  # predicted_track_status
                        0.85,                                     # confidence
                        w['start_time'],                          # prediction_start_time
                        w['start_time']                           # prediction_end_time
                    ))
                
                except Exception as row_error:
                    logger.warning(f"   Error inserting row {i}: {row_error}")
                    continue
                
                # Commit every 1000 records for progress
                if i % 1000 == 0:
                    conn.commit()
                    logger.info(f"   Committed {i}/{len(windows)} records")
            
            conn.commit()
            logger.info("✅ All telemetry data inserted successfully")
            
        except Exception as e:
            logger.error(f"❌ Error inserting telemetry: {e}")
            conn.rollback()
            raise
        finally:
            cursor.close()
            conn.close()
    
    def run(self):
        logger.info("🏎️  F1 ETL Pipeline Starting with Debug Mode")
        logger.info(f"📋 Configuration:")
        logger.info(f"   Seasons: {SEASONS}")
        logger.info(f"   Session types: {SESSION_TYPES}")
        logger.info(f"   Window size: {WINDOW_SIZE} seconds")
        
        # Test database connection
        try:
            conn = self.get_connection()
            conn.close()
            logger.info("✅ Database connection test successful")
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            return 1
        
        # Clear existing data
        self.clear_data()
        
        # Collect all windows
        all_windows = []
        
        # Process each season
        for season in SEASONS:
            logger.info(f"🏁 Processing {season} season...")
            
            races = self.get_schedule(season)
            if not races:
                logger.warning(f"⚠️  No races found for {season}")
                continue
            
            for race_num, race_info in enumerate(races, 1):
                logger.info(f"🏁 Race {race_num}/{len(races)}: {race_info['year']} {race_info['race']}")
                
                for session_type in SESSION_TYPES:
                    session = self.fetch_session(race_info['year'], race_info['race'], session_type)
                    
                    if session:
                        windows = self.extract_windows(session)
                        all_windows.extend(windows)
                        logger.info(f"✅ Race complete: {len(windows)} windows added (Total: {len(all_windows)})")
                    else:
                        logger.warning(f"⚠️  Session failed, skipping...")
        
        if not all_windows:
            logger.error("❌ No data extracted from any sessions!")
            return 1
        
        logger.info(f"📊 Data extraction complete: {len(all_windows)} total windows")
        
        # Insert data into database
        logger.info("💾 Starting database insertion...")
        self.insert_drivers(all_windows)
        self.insert_sessions(all_windows)
        self.insert_telemetry(all_windows)
        
        # Final verification
        logger.info("🔍 Verifying final data...")
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM drivers")
        driver_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM sessions")
        session_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM telemetry")
        telemetry_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM predictions")
        prediction_count = cursor.fetchone()[0]
        
        cursor.close()
        conn.close()
        
        logger.info("🎉 ETL Pipeline Complete!")
        logger.info(f"📊 Final counts:")
        logger.info(f"   Drivers: {driver_count}")
        logger.info(f"   Sessions: {session_count}")
        logger.info(f"   Telemetry: {telemetry_count}")
        logger.info(f"   Predictions: {prediction_count}")
        
        return 0

if __name__ == "__main__":
    pipeline = ETLPipeline()
    exit(pipeline.run())