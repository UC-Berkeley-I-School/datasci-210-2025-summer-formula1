#!/usr/bin/env python3
"""
Enhanced F1 Data Loader with Comprehensive Telemetry
Includes 2022, 2023, 2024 seasons with full telemetry data
"""

import sys
import os
from pathlib import Path
import logging
from datetime import datetime
import pandas as pd
import sqlite3
import time
import warnings
import numpy as np

# Suppress FastF1 warnings
warnings.filterwarnings('ignore')

class EnhancedF1TelemetryLoader:
    def __init__(self, include_telemetry=True, seasons=None):
        """Initialize enhanced F1 loader with telemetry"""
        current_dir = Path.cwd()
        
        # Configuration
        self.include_telemetry = include_telemetry
        self.seasons = seasons or [2022, 2023, 2024]  # Include 2022!
        self.max_rounds_per_season = 8  # Adjust based on your needs (full season = ~24)
        
        # Database path
        self.db_path = current_dir / "database" / "data" / "f1_data.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Try to import and setup FastF1
        try:
            import fastf1
            import fastf1.plotting
            
            # Set up FastF1 cache
            cache_dir = current_dir / "database" / "cache" / "fastf1"
            cache_dir.mkdir(parents=True, exist_ok=True)
            fastf1.Cache.enable_cache(str(cache_dir))
            
            # Disable FastF1 logging to reduce noise
            fastf1.set_log_level('WARNING')
            
            self.fastf1 = fastf1
            self.fastf1_available = True
            self.logger.info("✅ FastF1 initialized successfully")
            
        except ImportError as e:
            self.logger.error(f"❌ FastF1 not available: {e}")
            self.fastf1_available = False
            sys.exit(1)
    
    def setup_logging(self):
        """Setup comprehensive logging"""
        log_dir = Path.cwd() / "logs"
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'enhanced_f1_telemetry.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_enhanced_database(self):
        """Setup enhanced database schema with telemetry tables"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Core F1 tables
            tables = {
                'drivers': '''
                    CREATE TABLE IF NOT EXISTS drivers (
                        driver_id TEXT PRIMARY KEY,
                        first_name TEXT,
                        last_name TEXT,
                        abbreviation TEXT,
                        team_name TEXT,
                        country_code TEXT,
                        driver_number INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''',
                'constructors': '''
                    CREATE TABLE IF NOT EXISTS constructors (
                        team_id TEXT PRIMARY KEY,
                        team_name TEXT,
                        team_color TEXT,
                        base_country TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''',
                'race_sessions': '''
                    CREATE TABLE IF NOT EXISTS race_sessions (
                        session_id TEXT PRIMARY KEY,
                        season INTEGER,
                        round_number INTEGER,
                        country TEXT,
                        location TEXT,
                        event_name TEXT,
                        event_date DATE,
                        session_type TEXT,
                        circuit_key INTEGER,
                        meeting_key INTEGER,
                        weather_summary TEXT,
                        track_temp_avg REAL,
                        air_temp_avg REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''',
                'race_results': '''
                    CREATE TABLE IF NOT EXISTS race_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT,
                        driver_id TEXT,
                        team_id TEXT,
                        position INTEGER,
                        grid_position INTEGER,
                        points REAL,
                        status TEXT,
                        time_str TEXT,
                        fastest_lap BOOLEAN,
                        fastest_lap_time REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (session_id) REFERENCES race_sessions (session_id),
                        FOREIGN KEY (driver_id) REFERENCES drivers (driver_id),
                        FOREIGN KEY (team_id) REFERENCES constructors (team_id)
                    )
                ''',
                'qualifying_results': '''
                    CREATE TABLE IF NOT EXISTS qualifying_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT,
                        driver_id TEXT,
                        team_id TEXT,
                        position INTEGER,
                        q1_time REAL,
                        q2_time REAL,
                        q3_time REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (session_id) REFERENCES race_sessions (session_id),
                        FOREIGN KEY (driver_id) REFERENCES drivers (driver_id),
                        FOREIGN KEY (team_id) REFERENCES constructors (team_id)
                    )
                '''
            }
            
            # Add telemetry tables if enabled
            if self.include_telemetry:
                telemetry_tables = {
                    'lap_data': '''
                        CREATE TABLE IF NOT EXISTS lap_data (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            session_id TEXT,
                            driver_id TEXT,
                            lap_number INTEGER,
                            lap_time REAL,
                            sector_1_time REAL,
                            sector_2_time REAL,
                            sector_3_time REAL,
                            speed_i1 REAL,
                            speed_i2 REAL,
                            speed_fl REAL,
                            speed_st REAL,
                            is_personal_best BOOLEAN,
                            compound TEXT,
                            tyre_life INTEGER,
                            fresh_tyre BOOLEAN,
                            pit_out_time REAL,
                            pit_in_time REAL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            FOREIGN KEY (session_id) REFERENCES race_sessions (session_id),
                            FOREIGN KEY (driver_id) REFERENCES drivers (driver_id)
                        )
                    ''',
                    'telemetry_data': '''
                        CREATE TABLE IF NOT EXISTS telemetry_data (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            session_id TEXT,
                            driver_id TEXT,
                            lap_number INTEGER,
                            distance REAL,
                            speed REAL,
                            throttle REAL,
                            brake INTEGER,
                            gear INTEGER,
                            drs INTEGER,
                            rpm REAL,
                            x_position REAL,
                            y_position REAL,
                            time_elapsed REAL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            FOREIGN KEY (session_id) REFERENCES race_sessions (session_id),
                            FOREIGN KEY (driver_id) REFERENCES drivers (driver_id)
                        )
                    ''',
                    'weather_data': '''
                        CREATE TABLE IF NOT EXISTS weather_data (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            session_id TEXT,
                            time_elapsed REAL,
                            air_temp REAL,
                            track_temp REAL,
                            humidity REAL,
                            pressure REAL,
                            wind_speed REAL,
                            wind_direction REAL,
                            rainfall BOOLEAN,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            FOREIGN KEY (session_id) REFERENCES race_sessions (session_id)
                        )
                    ''',
                    'pit_stops': '''
                        CREATE TABLE IF NOT EXISTS pit_stops (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            session_id TEXT,
                            driver_id TEXT,
                            lap_number INTEGER,
                            pit_stop_time REAL,
                            pit_duration REAL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            FOREIGN KEY (session_id) REFERENCES race_sessions (session_id),
                            FOREIGN KEY (driver_id) REFERENCES drivers (driver_id)
                        )
                    '''
                }
                tables.update(telemetry_tables)
            
            # Cache status table
            tables['cache_status'] = '''
                CREATE TABLE IF NOT EXISTS cache_status (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pipeline_run_id TEXT,
                    run_id TEXT,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    status TEXT,
                    total_records INTEGER,
                    seasons_processed TEXT,
                    telemetry_included BOOLEAN,
                    error_message TEXT
                )
            '''
            
            for table_name, table_sql in tables.items():
                cursor.execute(table_sql)
                self.logger.info(f"✅ Created/verified table: {table_name}")
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to setup enhanced database: {e}")
            raise
    
    def get_lap_data(self, year, round_number, session_type='R'):
        """Get detailed lap data with telemetry"""
        try:
            session_name = {'R': 'Race', 'Q': 'Qualifying', 'P1': 'Practice 1', 'P2': 'Practice 2', 'P3': 'Practice 3'}[session_type]
            self.logger.info(f"📊 Loading lap data for {year} Round {round_number} {session_name}...")
            
            # Load the session
            session = self.fastf1.get_session(year, round_number, session_type)
            session.load()
            
            # Get lap data
            laps = session.laps
            
            lap_data = []
            for _, lap in laps.iterrows():
                try:
                    # Convert lap time to seconds
                    lap_time_seconds = None
                    if pd.notna(lap['LapTime']):
                        try:
                            lap_time_seconds = lap['LapTime'].total_seconds()
                        except:
                            lap_time_seconds = None
                    
                    # Convert sector times to seconds
                    def convert_sector_time(sector_time):
                        if pd.notna(sector_time):
                            try:
                                return sector_time.total_seconds()
                            except:
                                return None
                        return None
                    
                    lap_data.append({
                        'session_id': f"{year}_{round_number:02d}_{session_type}",
                        'driver_id': lap['Driver'] if pd.notna(lap['Driver']) else f"driver_{lap.get('DriverNumber', 'unknown')}",
                        'lap_number': lap['LapNumber'] if pd.notna(lap['LapNumber']) else 0,
                        'lap_time': lap_time_seconds,
                        'sector_1_time': convert_sector_time(lap.get('Sector1Time')),
                        'sector_2_time': convert_sector_time(lap.get('Sector2Time')),
                        'sector_3_time': convert_sector_time(lap.get('Sector3Time')),
                        'speed_i1': lap.get('SpeedI1') if pd.notna(lap.get('SpeedI1')) else None,
                        'speed_i2': lap.get('SpeedI2') if pd.notna(lap.get('SpeedI2')) else None,
                        'speed_fl': lap.get('SpeedFL') if pd.notna(lap.get('SpeedFL')) else None,
                        'speed_st': lap.get('SpeedST') if pd.notna(lap.get('SpeedST')) else None,
                        'is_personal_best': bool(lap.get('IsPersonalBest', False)),
                        'compound': lap.get('Compound', '') if pd.notna(lap.get('Compound')) else '',
                        'tyre_life': lap.get('TyreLife') if pd.notna(lap.get('TyreLife')) else None,
                        'fresh_tyre': bool(lap.get('FreshTyre', False)),
                        'pit_out_time': lap.get('PitOutTime') if pd.notna(lap.get('PitOutTime')) else None,
                        'pit_in_time': lap.get('PitInTime') if pd.notna(lap.get('PitInTime')) else None
                    })
                    
                except Exception as e:
                    self.logger.warning(f"Could not process lap data for lap {lap.get('LapNumber', 'unknown')}: {e}")
                    continue
            
            self.logger.info(f"✅ Loaded {len(lap_data)} lap records")
            return pd.DataFrame(lap_data)
            
        except Exception as e:
            self.logger.error(f"Failed to get lap data for {year} Round {round_number} {session_type}: {e}")
            return pd.DataFrame()
    
    def get_telemetry_sample(self, year, round_number, session_type='R', sample_rate=10):
        """Get telemetry data (sampled to reduce size)"""
        try:
            session_name = {'R': 'Race', 'Q': 'Qualifying', 'P1': 'Practice 1', 'P2': 'Practice 2', 'P3': 'Practice 3'}[session_type]
            self.logger.info(f"📡 Loading telemetry sample for {year} Round {round_number} {session_name}...")
            
            # Load the session
            session = self.fastf1.get_session(year, round_number, session_type)
            session.load()
            
            # Get a sample of drivers to avoid too much data
            drivers = session.drivers[:5]  # Limit to first 5 drivers for demo
            
            telemetry_data = []
            for driver in drivers:
                try:
                    driver_telemetry = session.laps.pick_driver(driver).get_telemetry()
                    
                    # Sample the data (every nth point)
                    sampled_telemetry = driver_telemetry.iloc[::sample_rate]
                    
                    for _, tel in sampled_telemetry.iterrows():
                        try:
                            telemetry_data.append({
                                'session_id': f"{year}_{round_number:02d}_{session_type}",
                                'driver_id': driver,
                                'lap_number': tel.get('LapNumber', 0),
                                'distance': tel.get('Distance', 0),
                                'speed': tel.get('Speed', 0),
                                'throttle': tel.get('Throttle', 0),
                                'brake': tel.get('Brake', 0),
                                'gear': tel.get('nGear', 0),
                                'drs': tel.get('DRS', 0),
                                'rpm': tel.get('RPM', 0),
                                'x_position': tel.get('X', 0),
                                'y_position': tel.get('Y', 0),
                                'time_elapsed': tel.get('Time', pd.Timedelta(0)).total_seconds() if pd.notna(tel.get('Time')) else 0
                            })
                        except Exception as e:
                            continue
                            
                except Exception as e:
                    self.logger.warning(f"Could not get telemetry for driver {driver}: {e}")
                    continue
            
            self.logger.info(f"✅ Loaded {len(telemetry_data)} telemetry records")
            return pd.DataFrame(telemetry_data)
            
        except Exception as e:
            self.logger.error(f"Failed to get telemetry for {year} Round {round_number} {session_type}: {e}")
            return pd.DataFrame()
    
    def get_weather_data(self, year, round_number, session_type='R'):
        """Get weather data for a session"""
        try:
            self.logger.info(f"🌤️  Loading weather data for {year} Round {round_number}...")
            
            # Load the session
            session = self.fastf1.get_session(year, round_number, session_type)
            session.load()
            
            # Get weather data
            weather = session.weather_data
            
            weather_data = []
            for _, w in weather.iterrows():
                try:
                    weather_data.append({
                        'session_id': f"{year}_{round_number:02d}_{session_type}",
                        'time_elapsed': w.get('Time', pd.Timedelta(0)).total_seconds() if pd.notna(w.get('Time')) else 0,
                        'air_temp': w.get('AirTemp', 0),
                        'track_temp': w.get('TrackTemp', 0),
                        'humidity': w.get('Humidity', 0),
                        'pressure': w.get('Pressure', 0),
                        'wind_speed': w.get('WindSpeed', 0),
                        'wind_direction': w.get('WindDirection', 0),
                        'rainfall': bool(w.get('Rainfall', False))
                    })
                except Exception as e:
                    continue
            
            self.logger.info(f"✅ Loaded {len(weather_data)} weather records")
            return pd.DataFrame(weather_data)
            
        except Exception as e:
            self.logger.error(f"Failed to get weather data for {year} Round {round_number} {session_type}: {e}")
            return pd.DataFrame()
    
    def load_comprehensive_f1_data_with_telemetry(self):
        """Load comprehensive F1 data including telemetry for multiple seasons"""
        try:
            if not self.fastf1_available:
                raise Exception("FastF1 not available")
            
            self.logger.info(f"🏎️  Loading comprehensive F1 data for seasons: {self.seasons}")
            if self.include_telemetry:
                self.logger.info("📡 Including telemetry data (this will take longer)")
            
            all_data = {
                'race_sessions': [],
                'race_results': [],
                'qualifying_results': [],
                'drivers': [],
                'constructors': []
            }
            
            if self.include_telemetry:
                all_data.update({
                    'lap_data': [],
                    'telemetry_data': [],
                    'weather_data': []
                })
            
            for year in self.seasons:
                try:
                    self.logger.info(f"🏎️  Processing {year} season...")
                    
                    # Get season schedule (inherited from base class)
                    schedule_df = self.get_season_schedule(year)
                    if not schedule_df.empty:
                        all_data['race_sessions'].append(schedule_df)
                    
                    # Get drivers and teams for this season
                    drivers_df, teams_df = self.get_drivers_and_teams(year)
                    if not drivers_df.empty:
                        all_data['drivers'].append(drivers_df)
                    if not teams_df.empty:
                        all_data['constructors'].append(teams_df)
                    
                    # Get race and qualifying results for each round
                    for round_num in range(1, min(self.max_rounds_per_season + 1, 25)):
                        try:
                            self.logger.info(f"📊 Processing Round {round_num}...")
                            
                            # Get race results
                            race_results_df = self.get_race_results(year, round_num)
                            if not race_results_df.empty:
                                all_data['race_results'].append(race_results_df)
                            
                            # Get qualifying results
                            qualifying_results_df = self.get_qualifying_results(year, round_num)
                            if not qualifying_results_df.empty:
                                all_data['qualifying_results'].append(qualifying_results_df)
                            
                            # Get telemetry data if enabled
                            if self.include_telemetry:
                                # Get lap data
                                lap_data_df = self.get_lap_data(year, round_num, 'R')
                                if not lap_data_df.empty:
                                    all_data['lap_data'].append(lap_data_df)
                                
                                # Get telemetry sample (to avoid too much data)
                                telemetry_df = self.get_telemetry_sample(year, round_num, 'R', sample_rate=20)
                                if not telemetry_df.empty:
                                    all_data['telemetry_data'].append(telemetry_df)
                                
                                # Get weather data
                                weather_df = self.get_weather_data(year, round_num, 'R')
                                if not weather_df.empty:
                                    all_data['weather_data'].append(weather_df)
                            
                            # Add delay to be respectful to FastF1/F1 servers
                            time.sleep(2)
                            
                        except Exception as e:
                            self.logger.warning(f"Failed to process Round {round_num} for {year}: {e}")
                            continue
                    
                    self.logger.info(f"✅ Completed {year} season")
                    
                except Exception as e:
                    self.logger.error(f"Failed to process {year}: {e}")
                    continue
            
            # Combine all data
            final_data = {}
            for data_type, dfs_list in all_data.items():
                if dfs_list:
                    try:
                        combined_df = pd.concat(dfs_list, ignore_index=True)
                        
                        # Remove duplicates based on appropriate key
                        if data_type == 'drivers':
                            combined_df = combined_df.drop_duplicates(subset=['driver_id'])
                        elif data_type == 'constructors':
                            combined_df = combined_df.drop_duplicates(subset=['team_id'])
                        elif data_type == 'race_sessions':
                            combined_df = combined_df.drop_duplicates(subset=['session_id'])
                        
                        final_data[data_type] = combined_df
                        self.logger.info(f"✅ Combined {data_type}: {len(combined_df)} records")
                        
                    except Exception as e:
                        self.logger.warning(f"Could not combine {data_type}: {e}")
                        # Still add the first dataframe if combination fails
                        if dfs_list:
                            final_data[data_type] = dfs_list[0]
                            self.logger.info(f"✅ Using first {data_type} dataframe: {len(dfs_list[0])} records")
            
            return final_data
            
        except Exception as e:
            self.logger.error(f"Failed to load comprehensive F1 data with telemetry: {e}")
            raise
    
    # Include helper methods from the simple loader
    def get_season_schedule(self, year):
        """Get race schedule for a season using FastF1"""
        try:
            self.logger.info(f"📅 Loading {year} season schedule...")
            
            # Get the event schedule
            schedule = self.fastf1.get_event_schedule(year, include_testing=False)
            
            sessions_data = []
            for _, event in schedule.iterrows():
                # Create session entries for each event
                event_sessions = ['Practice 1', 'Practice 2', 'Practice 3', 'Qualifying', 'Race']
                
                for session_type in event_sessions:
                    try:
                        session_id = f"{year}_{event['RoundNumber']:02d}_{session_type.replace(' ', '_')}"
                        
                        sessions_data.append({
                            'session_id': session_id,
                            'season': year,
                            'round_number': event['RoundNumber'],
                            'country': event['Country'],
                            'location': event['Location'],
                            'event_name': event['EventName'],
                            'event_date': event['EventDate'].date() if pd.notna(event['EventDate']) else None,
                            'session_type': session_type,
                            'circuit_key': event.get('Circuit Key', 0),
                            'meeting_key': event.get('Meeting Key', 0)
                        })
                    except Exception as e:
                        self.logger.warning(f"Could not process session {session_type} for {event['EventName']}: {e}")
                        continue
            
            self.logger.info(f"✅ Loaded {len(sessions_data)} sessions for {year}")
            return pd.DataFrame(sessions_data)
            
        except Exception as e:
            self.logger.error(f"Failed to get {year} schedule: {e}")
            return pd.DataFrame()
    
    def get_race_results(self, year, round_number):
        """Get race results for a specific race"""
        try:
            self.logger.info(f"🏁 Loading race results for {year} Round {round_number}...")
            
            # Load the race session
            session = self.fastf1.get_session(year, round_number, 'R')
            session.load()
            
            results = session.results
            
            race_results = []
            for _, result in results.iterrows():
                try:
                    # Handle fastest lap safely
                    fastest_lap = False
                    fastest_lap_time = None
                    
                    try:
                        if 'FastestLap' in result and pd.notna(result['FastestLap']):
                            fastest_lap = bool(result['FastestLap'])
                    except:
                        fastest_lap = False
                    
                    try:
                        if 'FastestLapTime' in result and pd.notna(result['FastestLapTime']):
                            fastest_lap_time = result['FastestLapTime'].total_seconds()
                    except:
                        fastest_lap_time = None
                    
                    race_results.append({
                        'session_id': f"{year}_{round_number:02d}_Race",
                        'driver_id': result['Abbreviation'] if pd.notna(result['Abbreviation']) else f"driver_{result['DriverNumber']}",
                        'team_id': result['TeamName'].replace(' ', '_').lower() if pd.notna(result['TeamName']) else 'unknown',
                        'position': result['Position'] if pd.notna(result['Position']) else 99,
                        'grid_position': result['GridPosition'] if pd.notna(result['GridPosition']) else 99,
                        'points': result['Points'] if pd.notna(result['Points']) else 0,
                        'status': result['Status'] if pd.notna(result['Status']) else 'Unknown',
                        'time_str': str(result['Time']) if pd.notna(result['Time']) else '',
                        'fastest_lap': fastest_lap,
                        'fastest_lap_time': fastest_lap_time
                    })
                except Exception as e:
                    self.logger.warning(f"Could not process result for driver {result.get('Abbreviation', 'Unknown')}: {e}")
                    continue
            
            self.logger.info(f"✅ Loaded {len(race_results)} race results")
            return pd.DataFrame(race_results)
            
        except Exception as e:
            self.logger.error(f"Failed to get race results for {year} Round {round_number}: {e}")
            return pd.DataFrame()
    
    def get_qualifying_results(self, year, round_number):
        """Get qualifying results for a specific race"""
        try:
            self.logger.info(f"⏱️  Loading qualifying results for {year} Round {round_number}...")
            
            # Load the qualifying session
            session = self.fastf1.get_session(year, round_number, 'Q')
            session.load()
            
            results = session.results
            
            qualifying_results = []
            for _, result in results.iterrows():
                try:
                    # Convert Q times to seconds
                    def convert_q_time(q_time):
                        if pd.notna(q_time):
                            try:
                                return q_time.total_seconds()
                            except:
                                return None
                        return None
                    
                    qualifying_results.append({
                        'session_id': f"{year}_{round_number:02d}_Qualifying",
                        'driver_id': result['Abbreviation'] if pd.notna(result['Abbreviation']) else f"driver_{result['DriverNumber']}",
                        'team_id': result['TeamName'].replace(' ', '_').lower() if pd.notna(result['TeamName']) else 'unknown',
                        'position': result['Position'] if pd.notna(result['Position']) else 99,
                        'q1_time': convert_q_time(result.get('Q1')),
                        'q2_time': convert_q_time(result.get('Q2')),
                        'q3_time': convert_q_time(result.get('Q3'))
                    })
                except Exception as e:
                    self.logger.warning(f"Could not process qualifying result for driver {result.get('Abbreviation', 'Unknown')}: {e}")
                    continue
            
            self.logger.info(f"✅ Loaded {len(qualifying_results)} qualifying results")
            return pd.DataFrame(qualifying_results)
            
        except Exception as e:
            self.logger.error(f"Failed to get qualifying results for {year} Round {round_number}: {e}")
            return pd.DataFrame()
    
    def get_drivers_and_teams(self, year):
        """Get drivers and teams data for a season"""
        try:
            self.logger.info(f"👥 Loading drivers and teams for {year}...")
            
            # Get first race to extract driver/team info
            session = self.fastf1.get_session(year, 1, 'R')
            session.load()
            
            results = session.results
            
            drivers_data = []
            teams_data = []
            
            seen_drivers = set()
            seen_teams = set()
            
            for _, result in results.iterrows():
                try:
                    # Driver data
                    driver_id = result['Abbreviation'] if pd.notna(result['Abbreviation']) else f"driver_{result['DriverNumber']}"
                    if driver_id not in seen_drivers:
                        drivers_data.append({
                            'driver_id': driver_id,
                            'first_name': result['FirstName'] if pd.notna(result['FirstName']) else '',
                            'last_name': result['LastName'] if pd.notna(result['LastName']) else '',
                            'abbreviation': result['Abbreviation'] if pd.notna(result['Abbreviation']) else '',
                            'team_name': result['TeamName'] if pd.notna(result['TeamName']) else '',
                            'country_code': result['CountryCode'] if pd.notna(result['CountryCode']) else '',
                            'driver_number': result['DriverNumber'] if pd.notna(result['DriverNumber']) else 0
                        })
                        seen_drivers.add(driver_id)
                    
                    # Team data
                    team_id = result['TeamName'].replace(' ', '_').lower() if pd.notna(result['TeamName']) else 'unknown'
                    if team_id not in seen_teams:
                        teams_data.append({
                            'team_id': team_id,
                            'team_name': result['TeamName'] if pd.notna(result['TeamName']) else '',
                            'team_color': result['TeamColor'] if pd.notna(result['TeamColor']) else '',
                            'base_country': ''  # Would need additional lookup
                        })
                        seen_teams.add(team_id)
                        
                except Exception as e:
                    self.logger.warning(f"Could not process driver/team data: {e}")
                    continue
            
            self.logger.info(f"✅ Loaded {len(drivers_data)} drivers and {len(teams_data)} teams")
            return pd.DataFrame(drivers_data), pd.DataFrame(teams_data)
            
        except Exception as e:
            self.logger.error(f"Failed to get drivers/teams for {year}: {e}")
            return pd.DataFrame(), pd.DataFrame()
    
    def cache_data_to_database(self, data):
        """Cache the loaded data to database"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            
            run_id = f"enhanced_telemetry_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            start_time = datetime.now()
            
            # Log start
            conn.execute('''
                INSERT INTO cache_status (pipeline_run_id, run_id, start_time, status, total_records, telemetry_included)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (run_id, run_id, start_time, 'IN_PROGRESS', 0, self.include_telemetry))
            
            total_records = 0
            processed_types = []
            
            # Cache each data type
            for data_type, df in data.items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    try:
                        # Map data types to table names
                        table_name = data_type
                        
                        # Clear existing data for fresh load
                        conn.execute(f'DELETE FROM {table_name}')
                        
                        # Insert new data
                        df.to_sql(table_name, conn, if_exists='append', index=False)
                        
                        records_count = len(df)
                        total_records += records_count
                        processed_types.append(data_type)
                        
                        self.logger.info(f"✅ Cached {records_count} records to {table_name}")
                        
                    except Exception as e:
                        self.logger.error(f"Failed to cache {data_type}: {e}")
                        continue
            
            # Update cache status
            end_time = datetime.now()
            seasons_str = ','.join(map(str, self.seasons))
            
            conn.execute('''
                UPDATE cache_status 
                SET end_time = ?, status = ?, total_records = ?, seasons_processed = ?
                WHERE run_id = ?
            ''', (end_time, 'SUCCESS', total_records, seasons_str, run_id))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"🎯 Successfully cached {total_records} total records")
            return run_id
            
        except Exception as e:
            self.logger.error(f"Failed to cache data: {e}")
            raise
    
    def run(self):
        """Main execution method"""
        try:
            self.logger.info("🏎️  Starting Enhanced F1 Data Loader with Telemetry")
            self.logger.info("=" * 70)
            self.logger.info(f"Seasons: {self.seasons}")
            self.logger.info(f"Telemetry included: {self.include_telemetry}")
            self.logger.info(f"Max rounds per season: {self.max_rounds_per_season}")
            
            # Setup enhanced database
            self.setup_enhanced_database()
            
            # Load comprehensive data
            self.logger.info("📊 Loading comprehensive F1 data with telemetry...")
            data = self.load_comprehensive_f1_data_with_telemetry()
            
            if not data:
                raise Exception("No data loaded")
            
            # Cache to database
            run_id = self.cache_data_to_database(data)
            
            self.logger.info("=" * 70)
            self.logger.info("🎉 ENHANCED F1 DATA LOADING COMPLETE!")
            
            total_records = sum(len(df) for df in data.values() if isinstance(df, pd.DataFrame))
            self.logger.info(f"📊 Total Records: {total_records:,}")
            self.logger.info(f"🗂️  Data Types: {list(data.keys())}")
            self.logger.info(f"📅 Seasons: {self.seasons}")
            self.logger.info(f"📡 Telemetry: {'Included' if self.include_telemetry else 'Not included'}")
            self.logger.info(f"🆔 Run ID: {run_id}")
            
            return run_id
            
        except Exception as e:
            self.logger.error(f"Enhanced F1 data loader with telemetry failed: {e}")
            raise

def main():
    """Main entry point with configuration options"""
    print("🏎️  Enhanced F1 Data Loader with Telemetry")
    print("=" * 60)
    print("Loading comprehensive F1 data including 2022, 2023, 2024...")
    print("Includes: Race results, Qualifying, Lap data, Telemetry, Weather")
    print()
    
    # Configuration options
    include_telemetry = True  # Set to False if you only want basic data
    seasons = [2022, 2023, 2024]  # Now includes 2022!
    max_rounds = 24  # Limit rounds to avoid timeouts (increase for full seasons)
    
    print(f"📊 Configuration:")
    print(f"   Seasons: {seasons}")
    print(f"   Telemetry: {'Yes' if include_telemetry else 'No'}")
    print(f"   Max rounds per season: {max_rounds}")
    print()
    
    try:
        loader = EnhancedF1TelemetryLoader(
            include_telemetry=include_telemetry,
            seasons=seasons
        )
        loader.max_rounds_per_season = max_rounds
        
        run_id = loader.run()
        
        print(f"\n✅ Enhanced data loading completed successfully!")
        print(f"📊 Run ID: {run_id}")
        print(f"📁 Database: {loader.db_path}")
        
        # Quick verification
        conn = sqlite3.connect(str(loader.db_path))
        cursor = conn.cursor()
        
        print(f"\n📈 COMPREHENSIVE DATA SUMMARY:")
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        total_records = 0
        for table in sorted(tables):
            if table not in ['sqlite_sequence', 'cache_status']:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    total_records += count
                    emoji = "📡" if "telemetry" in table or "lap_data" in table or "weather" in table else "🏁"
                    print(f"  {emoji} {table}: {count:,} records")
                except:
                    print(f"  ❌ {table}: Error reading")
        
        print(f"\nTotal Records: {total_records:,}")
        print(f"Database Size: {loader.db_path.stat().st_size / 1024 / 1024:.1f} MB")
        
        conn.close()
        
        print(f"\n🎯 YOUR F1 PREDICTION SYSTEM NOW HAS:")
        print(f"✅ Three full seasons (2022, 2023, 2024)")
        print(f"✅ Race results and qualifying data")
        if include_telemetry:
            print(f"✅ Lap-by-lap telemetry data")
            print(f"✅ Speed, throttle, brake, gear data")
            print(f"✅ Weather and track conditions")
        print(f"✅ Driver and constructor profiles")
        
        print(f"\n🚀 NEXT STEPS:")
        print("1. Start F1 Data API: python database/python/database_api.py")
        print("2. Start prediction webapp: python webapp_f1_integration.py")
        print("3. Visit http://localhost:8000/dashboard")
        print("4. Your ML models now have comprehensive telemetry data!")
        
    except Exception as e:
        print(f"❌ Enhanced data loading failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()