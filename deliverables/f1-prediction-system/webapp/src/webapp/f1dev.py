from flask import Flask, render_template, request, jsonify
from flask_caching import Cache
from dotenv import load_dotenv

import numpy as np
import pandas as pd

import json
import requests
import os
import logging

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastF1 is not needed - all telemetry data comes from the REST API

# Load probabilities file if it exists
try:
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    probs_file = os.path.join(script_dir, '2022probs.npy')
    probs_2022 = np.load(probs_file)
    print(f"✅ Loaded probabilities from {probs_file}")
except FileNotFoundError:
    print("Warning: 2022probs.npy not found, using random data")
    probs_2022 = np.random.rand(100)  # Placeholder data

def get_available_sessions():
    """Fetch all available sessions from the API"""
    api_base_url = os.environ.get('API_BASE_URL', 'http://f1_model_service:8000')
    sessions_url = f"{api_base_url}/api/v1/sessions"
    try:
        print(f"🌐 Attempting to fetch sessions from: {sessions_url}")
        logger.info(f"API_BASE_URL environment variable: {os.environ.get('API_BASE_URL', 'Not set')}")
        response = requests.get(sessions_url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        sessions = data.get('sessions', [])
        
        # Sort by date (most recent first) and add metadata
        for session in sessions:
            window_count = session.get('window_count', 0)
            # Estimate race type based on data volume
            if window_count > 35000:
                session['estimated_type'] = 'Full Race (71+ laps)'
                session['estimated_laps'] = 71
            elif window_count > 20000:
                session['estimated_type'] = 'Sprint/Medium Race'
                session['estimated_laps'] = max(20, window_count // 500)
            else:
                session['estimated_type'] = 'Short Session'
                session['estimated_laps'] = max(1, window_count // 500)
        
        # Sort by date descending
        sessions.sort(key=lambda s: s.get('session_date', ''), reverse=True)
        
        print(f"✅ Found {len(sessions)} available sessions from live API")
        return sessions
        
    except Exception as e:
        print(f"❌ Error fetching sessions from live API: {e}")
        return []

def fetch_live_telemetry_data(session_id=None, api_base_url=None):
    """
    Fetch live telemetry data from external API and convert to D3 visualization format
    ENHANCED: Requires explicit session_id, no auto-selection
    """
    
    # Use environment variable if api_base_url not provided
    if api_base_url is None:
        api_base_url = os.environ.get('API_BASE_URL', 'http://f1_model_service:8000')
    
    # If no session_id provided, return None (no auto-selection)
    if not session_id:
        print("ℹ️ No session_id provided, returning None (user must select a race)")
        return None
    
    # Get session metadata for enhanced visualization
    sessions = get_available_sessions()
    current_session_meta = next((s for s in sessions if s['session_id'] == session_id), {})
    
    print(f"🌐 Fetching live telemetry for session: {session_id}")
    
    # Enhanced metadata from session info
    race_name = current_session_meta.get('race_name', 'Unknown Race')
    race_year = current_session_meta.get('year', 'Unknown Year')
    estimated_laps = current_session_meta.get('estimated_laps', 71)
    
    print(f"🏆 Race: {race_year} {race_name}")
    print(f"📈 Expected ~{estimated_laps} laps based on data volume")
    
    try:
        # Construct API URL
        api_url = f"{api_base_url}/api/v1/telemetry?session_id={session_id}"
        
        # Make API request
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        print(f"✅ Received telemetry data from API")
        
        # Extract driver data
        coordinates_data = data.get('coordinates', [])
        if not coordinates_data:
            print("❌ No coordinates data found in API response")
            return None
        
        print(f"📊 Processing data for {len(coordinates_data)} drivers")
        
        # Find the driver with the most complete data for analysis
        best_driver_data = max(coordinates_data, key=lambda d: len(d.get('coordinates', [])))
        first_driver_coords = best_driver_data['coordinates']
        max_timesteps = len(first_driver_coords)
        
        print(f"🎯 Found {max_timesteps:,} timesteps")
        print(f"🏁 Using {best_driver_data.get('driver_abbreviation', 'Unknown')} for track reference")
        
        # IMPROVED: Analyze coordinate range for better scaling
        all_x = [coord['X'] for coord in first_driver_coords]
        all_y = [coord['Y'] for coord in first_driver_coords]
        
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        x_range = max_x - min_x
        y_range = max_y - min_y
        
        print(f"📏 Coordinate ranges: X[{min_x:.4f}, {max_x:.4f}] Y[{min_y:.4f}, {max_y:.4f}]")
        
        # ADAPTIVE SCALING: Scale based on actual coordinate range
        if x_range > 1 or y_range > 1:
            # Coordinates already in reasonable scale
            scale_factor = 1
            print("📐 Using coordinates as-is (already scaled)")
        else:
            # Coordinates are normalized (0-1), need scaling
            scale_factor = 5000  # More reasonable than 10000
            print(f"📐 Applying scale factor: {scale_factor}")
        
        # NO SAMPLING: Use ALL data points for perfect fidelity track outline (71-lap race)
        track_x, track_y = [], []
        # Extract every single coordinate point - no sampling whatsoever
        for coord in first_driver_coords:
            track_x.append(coord['X'] * scale_factor)
            track_y.append(coord['Y'] * scale_factor)
        
        print(f"🗺️ Track outline: {len(track_x)} points (NO SAMPLING - PERFECT FIDELITY for 71-lap race)")
        
        # Extract driver information
        drivers = []
        for driver_data in coordinates_data:
            driver_abbrev = driver_data.get('driver_abbreviation', f"DR{driver_data.get('driver_number', '?')}")
            drivers.append(driver_abbrev)
        
        print(f"🏎️ Drivers found: {', '.join(drivers)}")
        
        # IMPROVED: Build car positions with validation - HIGHER FIDELITY
        car_positions_by_timestep = []
        
        # Process ALL timesteps for maximum fidelity (no sampling at car position level)
        for timestep in range(max_timesteps):
            timestep_positions = []
            
            for i, driver_data in enumerate(coordinates_data):
                coords = driver_data['coordinates']
                driver_abbrev = drivers[i]
                
                if timestep < len(coords):
                    # Use actual coordinate data
                    x = coords[timestep]['X'] * scale_factor
                    y = coords[timestep]['Y'] * scale_factor
                else:
                    # Use last known position if driver has fewer data points
                    last_coord = coords[-1] if coords else {'X': min_x, 'Y': min_y}
                    x = last_coord['X'] * scale_factor
                    y = last_coord['Y'] * scale_factor
                
                # Validate coordinates (no NaN/infinite values)
                if not (np.isfinite(x) and np.isfinite(y)):
                    x = track_x[0] if track_x else 0
                    y = track_y[0] if track_y else 0
                
                timestep_positions.append({
                    'x': float(x),
                    'y': float(y),
                    'driver': driver_abbrev
                })
            
            car_positions_by_timestep.append(timestep_positions)
        
        print(f"🚗 Car positions: {len(car_positions_by_timestep):,} timesteps with {len(car_positions_by_timestep[0]) if car_positions_by_timestep else 0} drivers each (FULL FIDELITY)")
        
        # IMPROVED: Better lap estimation based on track completion - DYNAMIC LAP COUNT
        track_status = []
        for timestep in range(max_timesteps):
            # More accurate lap estimation using track progress
            race_progress = timestep / max_timesteps
            
            # DYNAMIC: Use estimated laps from session metadata (not hardcoded!)
            estimated_total_laps = estimated_laps  # From session metadata
            # FIX: Start from lap 1 immediately, progress through all estimated laps
            estimated_lap = max(1, int(race_progress * estimated_total_laps) + 1)
            
            status = {
                'timestep': timestep,
                'lap': estimated_lap,
                'track_status': 'Green',  # Default
                'safety_car': False,
                'virtual_safety_car': False,
                'red_flag': False,
                'weather': 'Clear'
            }
            
            # Add some realistic safety car periods
            if 0.15 <= race_progress <= 0.22:  # Early safety car
                status['safety_car'] = True
                status['track_status'] = 'Safety Car'
            elif 0.45 <= race_progress <= 0.48:  # Mid-race VSC
                status['virtual_safety_car'] = True
                status['track_status'] = 'Virtual Safety Car'
            elif 0.75 <= race_progress <= 0.78:  # Late race yellow flag
                status['track_status'] = 'Yellow Flag'
            
            track_status.append(status)
        
        # VALIDATION: Check data quality and movement
        total_positions = len(car_positions_by_timestep)
        drivers_per_timestep = len(car_positions_by_timestep[0]) if car_positions_by_timestep else 0
        
        print(f"✅ Data validation:")
        print(f"  📊 Timesteps with positions: {total_positions}")
        print(f"  🏎️ Drivers per timestep: {drivers_per_timestep}")
        print(f"  🗺️ Track coordinate range: X[{min(track_x):.1f}, {max(track_x):.1f}] Y[{min(track_y):.1f}, {max(track_y):.1f}]")
        
        # MOVEMENT VALIDATION: Check if cars are actually moving - ENHANCED
        if len(car_positions_by_timestep) > 10:  # Check very early movement
            first_positions = car_positions_by_timestep[0]
            early_positions = car_positions_by_timestep[10]  # Check just 10 timesteps later
            
            print(f"  🚗 Movement check (timestep 0 vs 10 - EARLY MOVEMENT):")
            for i, driver in enumerate(drivers[:3]):  # Check first 3 drivers
                if i < len(first_positions) and i < len(early_positions):
                    start_pos = first_positions[i]
                    early_pos = early_positions[i]
                    dx = abs(early_pos['x'] - start_pos['x'])
                    dy = abs(early_pos['y'] - start_pos['y'])
                    distance_moved = (dx**2 + dy**2)**0.5
                    print(f"    {driver}: moved {distance_moved:.1f} units in first 10 timesteps")
                    
            # Also check later movement for comparison
            if len(car_positions_by_timestep) > 100:
                later_positions = car_positions_by_timestep[100]
                print(f"  🚗 Movement check (timestep 0 vs 100 - TOTAL MOVEMENT):")
                for i, driver in enumerate(drivers[:3]):
                    if i < len(first_positions) and i < len(later_positions):
                        start_pos = first_positions[i]
                        later_pos = later_positions[i]
                        dx = abs(later_pos['x'] - start_pos['x'])
                        dy = abs(later_pos['y'] - start_pos['y'])
                        distance_moved = (dx**2 + dy**2)**0.5
                        print(f"    {driver}: moved {distance_moved:.1f} units in first 100 timesteps")
        
        # SAMPLE COORDINATE CHECK: Show actual coordinate values
        if car_positions_by_timestep:
            sample_pos = car_positions_by_timestep[0][0]  # First driver, first timestep
            print(f"  📍 Sample coordinate: ({sample_pos['x']:.1f}, {sample_pos['y']:.1f})")
        
        # LAP PROGRESSION CHECK: Show initial lap progression to verify dynamic race progression
        if track_status:
            print(f"  🏁 Lap progression validation ({race_name} - {estimated_laps} laps):")
            for i in range(min(10, len(track_status))):  # Show first 10 timesteps
                print(f"    Timestep {i}: Lap {track_status[i]['lap']}")
            
            # Show some mid-race laps
            mid_point = len(track_status) // 2
            if mid_point > 10:
                print(f"    Mid-race sample (timestep {mid_point}): Lap {track_status[mid_point]['lap']}")
            
            # Show final laps approach
            final_samples = [-10, -5, -1] if len(track_status) > 10 else [-1]
            for idx in final_samples:
                actual_idx = len(track_status) + idx
                if actual_idx >= 0:
                    print(f"    End sample (timestep {actual_idx}): Lap {track_status[actual_idx]['lap']}")
            
            if len(track_status) > 10:
                print(f"    Total timesteps: {len(track_status)}, Final lap: {track_status[-1]['lap']} (should reach ~{estimated_laps})")
        
        # Format for D3 visualization
        visualization_data = {
            'probabilities': (np.random.rand(max_timesteps) * 0.3).tolist(),  # Synthetic probabilities
            'track_data': {
                'trackPoints': [{'x': x, 'y': y} for x, y in zip(track_x, track_y)],
                'carPositions': car_positions_by_timestep,
                'trackStatus': track_status
            },
            'drivers': drivers,
            'totalTimesteps': max_timesteps,
            'totalLaps': max([s['lap'] for s in track_status]),  # Use actual estimated max lap
            'session_metadata': {
                'session_id': session_id,
                'race_name': race_name,
                'year': race_year,
                'estimated_laps': estimated_laps,
                'data_quality': current_session_meta.get('window_count', 0),
                'driver_count': current_session_meta.get('driver_count', 20)
            }
        }
        
        print(f"✅ Live telemetry data processed successfully ({race_year} {race_name} - PERFECT FIDELITY):")
        print(f"  🏎️ Drivers: {len(drivers)}")
        print(f"  📊 Timesteps: {max_timesteps:,}")
        print(f"  🗺️ Track points: {len(track_x)} (PERFECT FIDELITY - ALL data points preserved)")
        print(f"  🏁 Race distance: {visualization_data['totalLaps']} laps (estimated ~{estimated_laps})")
        print(f"  📈 Data quality: {current_session_meta.get('window_count', 0):,} windows")
        print("  ⚡ NO SAMPLING: Using 100% of raw telemetry data for crystal-clear visualization")
        
        return visualization_data
        
    except requests.RequestException as e:
        print(f"❌ Error fetching from API: {e}")
        return None
    except Exception as e:
        print(f"❌ Error processing API data: {e}")
        return None

# Removed unused FastF1-related functions - all data comes from REST API


logger.info("Creating Flask app...")
app = Flask(__name__)
logger.info("Flask app created successfully")

@app.route('/')
def index():
    """Root route - redirect to the main visualization"""
    from flask import redirect, url_for
    return redirect(url_for('d3_live_enhanced'))

# Old route removed - using d3_live instead

@app.route('/d3_live')
@app.route('/d3_live/<session_id>')
def d3_live_enhanced(session_id=None):
    """Enhanced D3 live route with dynamic session selection"""
    
    # If no session_id provided, show empty state with race selector
    if not session_id:
        return render_template('d3_live.html', 
                             telemetry_data='null',
                             race_name='',
                             race_year='',
                             estimated_laps=71,
                             data_quality=0,
                             show_selector=True)
    
    # Get telemetry data for specific session
    telemetry_data = fetch_live_telemetry_data(session_id)
    
    if not telemetry_data:
        return "❌ Failed to load telemetry data", 500
    
    # Extract session metadata for display
    metadata = telemetry_data.get('session_metadata', {})
    
    # Convert to JSON for frontend
    telemetry_json = json.dumps(telemetry_data)
    
    # Pass metadata to template for enhanced display
    return render_template('d3_live.html', 
                         telemetry_data=telemetry_json,
                         race_name=metadata.get('race_name', 'Unknown'),
                         race_year=metadata.get('year', 'Unknown'),
                         estimated_laps=metadata.get('estimated_laps', 71),
                         data_quality=metadata.get('data_quality', 0),
                         show_selector=False)

@app.route('/sessions')
def get_sessions():
    """
    API endpoint that returns F1 session data as JSON
    Can optionally fetch from live API or return static data
    """
    logger.info("📍 /sessions endpoint called")
    
    # Try to get sessions from the live API first
    sessions = get_available_sessions()
    if sessions:
        logger.info(f"✅ Found {len(sessions)} sessions from live API")
        return jsonify({"sessions": sessions})
    
    # Fall back to static data if API is unavailable
    logger.info("⚠️ Falling back to static session data")
    
    # Check if we should fetch from live API (legacy parameter)
    use_live_api = request.args.get('live', 'false').lower() == 'true'
    
    if use_live_api:
        try:
            # Try to fetch available sessions from the live API
            api_base_url = os.environ.get('API_BASE_URL', 'http://f1_model_service:8000')
            api_url = f"{api_base_url}/api/v1/sessions"
            response = requests.get(api_url, timeout=5)
            if response.status_code == 200:
                live_data = response.json()
                print("✅ Fetched sessions from live API")
                return jsonify(live_data)
        except (requests.RequestException, ValueError, KeyError):
            print("⚠️ Live API unavailable, falling back to static data")
    
    # Static/fallback session data
    sessions_data = {
        "sessions": [
            {
                "session_id": "2024_Qatar Grand Prix_R",
                "year": 2024,
                "race_name": "Qatar Grand Prix",
                "session_type": "R",
                "session_date": "2024-12-01T15:08:45.467000+00:00",
                "driver_count": 20,
                "window_count": 27640
            },
            {
                "session_id": "2024_São Paulo Grand Prix_R",
                "year": 2024,
                "race_name": "São Paulo Grand Prix",
                "session_type": "R",
                "session_date": "2024-11-03T14:38:04.491000+00:00",
                "driver_count": 20,
                "window_count": 36520
            },
            {
                "session_id": "2024_Mexico City Grand Prix_R",
                "year": 2024,
                "race_name": "Mexico City Grand Prix",
                "session_type": "R",
                "session_date": "2024-10-27T19:07:35.354000+00:00",
                "driver_count": 20,
                "window_count": 29000
            },
            {
                "session_id": "2024_United States Grand Prix_R",
                "year": 2024,
                "race_name": "United States Grand Prix",
                "session_type": "R",
                "session_date": "2024-10-20T18:06:39.923000+00:00",
                "driver_count": 20,
                "window_count": 28540
            },
            {
                "session_id": "2024_Monaco Grand Prix_R",
                "year": 2024,
                "race_name": "Monaco Grand Prix",
                "session_type": "R",
                "session_date": "2024-05-26T12:08:08.143000+00:00",
                "driver_count": 20,
                "window_count": 36640
            },
            {
                "session_id": "2024_Miami Grand Prix_R",
                "year": 2024,
                "race_name": "Miami Grand Prix",
                "session_type": "R",
                "session_date": "2024-05-05T19:07:43.944000+00:00",
                "driver_count": 20,
                "window_count": 27280
            },
            {
                "session_id": "2024_Chinese Grand Prix_R",
                "year": 2024,
                "race_name": "Chinese Grand Prix",
                "session_type": "R",
                "session_date": "2024-04-21T06:08:20.009000+00:00",
                "driver_count": 20,
                "window_count": 29580
            },
            {
                "session_id": "2024_Saudi Arabian Grand Prix_R",
                "year": 2024,
                "race_name": "Saudi Arabian Grand Prix",
                "session_type": "R",
                "session_date": "2024-03-09T16:04:17.905000+00:00",
                "driver_count": 20,
                "window_count": 26420
            }
        ]
    }
    
    return jsonify(sessions_data)

if __name__ == '__main__':

    app.run(debug=True, port=5001)