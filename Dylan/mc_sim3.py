#!/usr/bin/env python3

import os
import psycopg2
import numpy as np
import pandas as pd
from scipy import interpolate, stats
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import logging
from tqdm import tqdm
import json
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# FastF1 integration (optional - will fallback if not available)
try:
    import fastf1
    from fastf1 import plotting
    FASTF1_AVAILABLE = True
    logging.info("✅ FastF1 available - will use official circuit data")
except ImportError:
    FASTF1_AVAILABLE = False
    logging.info("⚠️ FastF1 not available - using database/mock data")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TrackSection:
    """A section of track with physics properties"""
    section_id: int
    center_coords: Tuple[float, float]
    avg_speed: float
    avg_throttle: float
    avg_brake_frequency: float
    typical_gear: int
    length_meters: float
    corner_type: str  # 'straight', 'slow_corner', 'fast_corner', 'chicane'
    elevation_change: float
    corner_number: Optional[int] = None  # Official corner number from FastF1

@dataclass
class DriverTrackModel:
    """Physics-based driver model for a specific track"""
    driver_id: str
    track_name: str
    
    # Section-specific performance
    section_speeds: Dict[int, float]
    section_consistency: Dict[int, float]
    
    # Physics parameters
    braking_efficiency: float
    acceleration_efficiency: float
    cornering_speed: float
    
    # Strategy patterns
    tire_management: float
    fuel_efficiency: float
    rain_performance: float
    
    # Reliability factors
    crash_probability: float
    mechanical_failure_rate: float

class FastF1EnhancedSimulator:
    def __init__(self):
        self.conn = None
        self.track_maps = {}
        self.driver_models = {}
        self.db_available = False
        self.fastf1_cache_enabled = False
        
        if FASTF1_AVAILABLE:
            try:
                # Set up FastF1 cache for better performance
                cache_dir = './fastf1_cache'
                if not os.path.exists(cache_dir):
                    os.makedirs(cache_dir)
                fastf1.Cache.enable_cache(cache_dir)
                self.fastf1_cache_enabled = True
                print("✅ FastF1 cache enabled")
            except Exception as e:
                print(f"⚠️ FastF1 cache setup failed: {e}")
    
    def get_fastf1_circuit_info(self, year=2024):
        """Get official circuit data from FastF1 - SIMPLIFIED VERSION"""
        if not FASTF1_AVAILABLE:
            return {}
        
        print("🏁 Creating circuits from F1 calendar...")
        circuits = {}
        
        try:
            # Just get the schedule, don't try to load session data
            schedule = fastf1.get_event_schedule(year)
            
            for _, event in schedule.iterrows():
                try:
                    if event['EventFormat'] != 'conventional':
                        continue
                    
                    circuit_name = event['Location']
                    country = event['Country']
                    
                    # Create basic circuit data without loading session
                    circuits[circuit_name] = self.create_basic_circuit_data(circuit_name, country)
                    print(f"   ✅ {circuit_name}: Created from calendar")
                    
                except Exception as e:
                    print(f"   ❌ Error processing {circuit_name}: {e}")
                    continue
            
            print(f"✅ Created {len(circuits)} circuits from F1 calendar")
            return circuits
            
        except Exception as e:
            print(f"❌ FastF1 schedule loading failed: {e}")
            # Return empty dict to fall back to standard tracks
            return {}

    def create_basic_circuit_data(self, circuit_name, country):
        """Create basic circuit data as fallback - ENHANCED VERSION"""
        # Real F1 track lengths (meters)
        track_lengths = {
            'Sakhir': 5412, 'Jeddah': 6174, 'Melbourne': 5278, 'Suzuka': 5807,
            'Imola': 4909, 'Monaco': 3337, 'Montréal': 4361, 'Barcelona': 4675,
            'Silverstone': 5891, 'Budapest': 4381, 'Spa-Francorchamps': 7004,
            'Zandvoort': 4259, 'Monza': 5793, 'Baku': 6003, 'Marina Bay': 5063,
            'Austin': 5513, 'Mexico City': 4304, 'Las Vegas': 6201, 'Yas Island': 5281
        }
        
        # Real corner counts for F1 circuits
        corner_counts = {
            'Sakhir': 15, 'Jeddah': 27, 'Melbourne': 16, 'Suzuka': 18,
            'Imola': 21, 'Monaco': 19, 'Montréal': 14, 'Barcelona': 16,
            'Silverstone': 18, 'Budapest': 14, 'Spa-Francorchamps': 20,
            'Zandvoort': 14, 'Monza': 11, 'Baku': 20, 'Marina Bay': 23,
            'Austin': 20, 'Mexico City': 17, 'Las Vegas': 17, 'Yas Island': 21
        }
        
        track_length = track_lengths.get(circuit_name, 5000)
        num_corners = corner_counts.get(circuit_name, 15)
        
        # Create realistic corners
        corners = []
        for i in range(num_corners):
            corners.append({
                'Number': i + 1,
                'Angle': np.random.uniform(30, 120),
                'Distance': (i + 1) * (track_length / num_corners),
                'Height': np.random.uniform(-5, 10)
            })
        
        return {
            'country': country,
            'corners': corners,
            'track_length': track_length,
            'sectors': [],
            'coordinate_system': 'realistic_generated'
        }
    
    def create_corners_from_track_name(self, circuit_name):
        """Create corner data based on known track characteristics"""
        # Famous F1 circuits with known corner counts and characteristics
        track_corners = {
            'Sakhir': [  # Bahrain - 15 corners
                {'Number': 1, 'Angle': 90, 'Distance': 700, 'Height': 0},
                {'Number': 2, 'Angle': 45, 'Distance': 1100, 'Height': 2},
                {'Number': 3, 'Angle': 45, 'Distance': 1300, 'Height': 2},
                {'Number': 4, 'Angle': 120, 'Distance': 1800, 'Height': 0},
                {'Number': 5, 'Angle': 60, 'Distance': 2200, 'Height': -3},
                {'Number': 6, 'Angle': 60, 'Distance': 2400, 'Height': -3},
                {'Number': 7, 'Angle': 45, 'Distance': 2800, 'Height': 0},
                {'Number': 8, 'Angle': 90, 'Distance': 3200, 'Height': 5},
                {'Number': 9, 'Angle': 75, 'Distance': 3600, 'Height': 3},
                {'Number': 10, 'Angle': 120, 'Distance': 4000, 'Height': 0},
                {'Number': 11, 'Angle': 45, 'Distance': 4400, 'Height': -2},
                {'Number': 12, 'Angle': 60, 'Distance': 4600, 'Height': -2},
                {'Number': 13, 'Angle': 90, 'Distance': 4900, 'Height': 0},
                {'Number': 14, 'Angle': 45, 'Distance': 5100, 'Height': 3},
                {'Number': 15, 'Angle': 30, 'Distance': 5300, 'Height': 0}
            ],
            'Monaco': [  # Monaco - 19 corners
                {'Number': 1, 'Angle': 30, 'Distance': 200, 'Height': 2},
                {'Number': 2, 'Angle': 45, 'Distance': 400, 'Height': 5},
                {'Number': 3, 'Angle': 90, 'Distance': 600, 'Height': 8},
                {'Number': 4, 'Angle': 60, 'Distance': 800, 'Height': 10},
                {'Number': 5, 'Angle': 120, 'Distance': 1000, 'Height': 8},
                {'Number': 6, 'Angle': 90, 'Distance': 1200, 'Height': 5},
                {'Number': 7, 'Angle': 45, 'Distance': 1400, 'Height': 3},
                {'Number': 8, 'Angle': 75, 'Distance': 1600, 'Height': 0},
                {'Number': 9, 'Angle': 30, 'Distance': 1800, 'Height': -2},
                {'Number': 10, 'Angle': 90, 'Distance': 2000, 'Height': -5},
                {'Number': 11, 'Angle': 45, 'Distance': 2200, 'Height': -3},
                {'Number': 12, 'Angle': 120, 'Distance': 2400, 'Height': 0},
                {'Number': 13, 'Angle': 60, 'Distance': 2600, 'Height': 2},
                {'Number': 14, 'Angle': 75, 'Distance': 2800, 'Height': 5},
                {'Number': 15, 'Angle': 90, 'Distance': 3000, 'Height': 3},
                {'Number': 16, 'Angle': 45, 'Distance': 3100, 'Height': 0},
                {'Number': 17, 'Angle': 30, 'Distance': 3150, 'Height': -2},
                {'Number': 18, 'Angle': 60, 'Distance': 3180, 'Height': 0},
                {'Number': 19, 'Angle': 45, 'Distance': 3200, 'Height': 2}
            ],
            'Silverstone': [  # 18 corners
                {'Number': 1, 'Angle': 45, 'Distance': 600, 'Height': 0},
                {'Number': 2, 'Angle': 30, 'Distance': 900, 'Height': 2},
                {'Number': 3, 'Angle': 90, 'Distance': 1300, 'Height': 5},
                {'Number': 4, 'Angle': 60, 'Distance': 1600, 'Height': 8},
                {'Number': 5, 'Angle': 45, 'Distance': 1900, 'Height': 6},
                {'Number': 6, 'Angle': 120, 'Distance': 2200, 'Height': 3},
                {'Number': 7, 'Angle': 90, 'Distance': 2500, 'Height': 0},
                {'Number': 8, 'Angle': 60, 'Distance': 2800, 'Height': -2},
                {'Number': 9, 'Angle': 75, 'Distance': 3100, 'Height': 0},
                {'Number': 10, 'Angle': 45, 'Distance': 3400, 'Height': 3},
                {'Number': 11, 'Angle': 30, 'Distance': 3700, 'Height': 5},
                {'Number': 12, 'Angle': 90, 'Distance': 4000, 'Height': 7},
                {'Number': 13, 'Angle': 60, 'Distance': 4300, 'Height': 4},
                {'Number': 14, 'Angle': 45, 'Distance': 4600, 'Height': 1},
                {'Number': 15, 'Angle': 75, 'Distance': 4900, 'Height': -1},
                {'Number': 16, 'Angle': 90, 'Distance': 5200, 'Height': 0},
                {'Number': 17, 'Angle': 45, 'Distance': 5500, 'Height': 2},
                {'Number': 18, 'Angle': 30, 'Distance': 5800, 'Height': 0}
            ]
        }
        
        # Return known corners or generate generic ones
        if circuit_name in track_corners:
            return track_corners[circuit_name]
        else:
            # Generate generic corners
            num_corners = np.random.randint(12, 20)  # 12-19 corners
            corners = []
            track_length = 5000  # Default track length
            
            for i in range(num_corners):
                corners.append({
                    'Number': i + 1,
                    'Angle': np.random.uniform(30, 120),
                    'Distance': (i + 1) * (track_length / num_corners),
                    'Height': np.random.uniform(-5, 10)
                })
            return corners
    
    def create_fastf1_track_sections(self, circuit_name, circuit_info):
        """Create track sections from FastF1 circuit data"""
        sections = []
        corners = circuit_info.get('corners', [])
        track_length = circuit_info.get('track_length', 5000)
        
        if not corners:
            # Create default sections if no corner data
            return self.create_default_sections(circuit_name, track_length)
        
        # Sort corners by position if available
        if corners and len(corners) > 0 and 'Distance' in corners[0]:
            corners = sorted(corners, key=lambda x: x.get('Distance', 0))
        
        # Create sections based on corners
        for i, corner in enumerate(corners):
            # Determine corner characteristics
            corner_num = corner.get('Number', i + 1)
            corner_angle = abs(corner.get('Angle', 45))  # Default moderate turn
            
            # Classify corner type based on angle and characteristics
            if corner_angle < 30:
                corner_type = 'fast_corner'
                base_speed = 220
                brake_freq = 0.3
            elif corner_angle < 90:
                corner_type = 'slow_corner'
                base_speed = 120
                brake_freq = 0.7
            else:
                corner_type = 'chicane'
                base_speed = 80
                brake_freq = 0.9
            
            # Estimate coordinates (FastF1 may not always have exact coordinates)
            distance = corner.get('Distance', i * (track_length / len(corners)))
            angle_rad = (distance / track_length) * 2 * np.pi
            x = 1000 * np.cos(angle_rad)
            y = 1000 * np.sin(angle_rad)
            
            section = TrackSection(
                section_id=i,
                center_coords=(x, y),
                avg_speed=base_speed + np.random.uniform(-20, 20),
                avg_throttle=0.3 if corner_type == 'chicane' else 0.6,
                avg_brake_frequency=brake_freq,
                typical_gear=2 if corner_type == 'chicane' else 4,
                length_meters=track_length / len(corners),
                corner_type=corner_type,
                elevation_change=corner.get('Height', 0),
                corner_number=corner_num
            )
            sections.append(section)
        
        # Add straight sections between corners
        straight_sections = []
        for i in range(len(sections)):
            if i < len(sections) - 1:
                # Straight between this corner and next
                straight = TrackSection(
                    section_id=len(corners) + i,
                    center_coords=((sections[i].center_coords[0] + sections[i+1].center_coords[0]) / 2,
                                 (sections[i].center_coords[1] + sections[i+1].center_coords[1]) / 2),
                    avg_speed=250 + np.random.uniform(-30, 30),
                    avg_throttle=0.95,
                    avg_brake_frequency=0.1,
                    typical_gear=7,
                    length_meters=track_length / (len(corners) * 2),
                    corner_type='straight',
                    elevation_change=0,
                    corner_number=None
                )
                straight_sections.append(straight)
        
        all_sections = sections + straight_sections
        print(f"   ✅ Created {len(all_sections)} sections ({len(corners)} corners + {len(straight_sections)} straights)")
        return all_sections
    
    def create_default_sections(self, circuit_name, track_length):
        """Create default sections when FastF1 data is unavailable"""
        # Create reasonable default based on track length
        num_sections = max(6, min(12, int(track_length / 500)))  # 6-12 sections
        sections = []
        
        for i in range(num_sections):
            # Alternate between corners and straights
            if i % 2 == 0:
                corner_type = np.random.choice(['slow_corner', 'fast_corner'], p=[0.6, 0.4])
                base_speed = 120 if corner_type == 'slow_corner' else 180
                brake_freq = 0.7 if corner_type == 'slow_corner' else 0.4
            else:
                corner_type = 'straight'
                base_speed = 240
                brake_freq = 0.1
            
            angle = (i / num_sections) * 2 * np.pi
            x = 1000 * np.cos(angle)
            y = 1000 * np.sin(angle)
            
            section = TrackSection(
                section_id=i,
                center_coords=(x, y),
                avg_speed=base_speed + np.random.uniform(-15, 15),
                avg_throttle=0.4 if corner_type != 'straight' else 0.9,
                avg_brake_frequency=brake_freq,
                typical_gear=3 if corner_type != 'straight' else 6,
                length_meters=track_length / num_sections,
                corner_type=corner_type,
                elevation_change=np.random.uniform(-5, 10),
                corner_number=i + 1 if corner_type != 'straight' else None
            )
            sections.append(section)
        
        return sections
    
    def build_fastf1_track_maps(self):
        """Build track maps using FastF1 data"""
        print("🗺️ Building track maps with FastF1 circuit data...")
        
        if FASTF1_AVAILABLE:
            # Get official circuit data
            circuits = self.get_fastf1_circuit_info()
            
            for circuit_name, circuit_info in circuits.items():
                try:
                    sections = self.create_fastf1_track_sections(circuit_name, circuit_info)
                    if sections:
                        self.track_maps[circuit_name] = sections
                except Exception as e:
                    print(f"❌ Error creating sections for {circuit_name}: {e}")
                    continue
        
        # Always supplement with known F1 tracks to ensure good coverage
        print("🎭 Adding standard F1 track layouts...")
        standard_tracks = {
            'Monaco': 3337,        # Monaco GP
            'Silverstone': 5891,   # British GP
            'Spa': 7004,          # Belgian GP
            'Monza': 5793,        # Italian GP
            'Suzuka': 5807,       # Japanese GP
            'Interlagos': 4309,   # Brazilian GP
            'Austin': 5513,       # US GP
            'Zandvoort': 4259,    # Dutch GP
            'Imola': 4909,        # Emilia Romagna GP
            'Barcelona': 4675,    # Spanish GP
            'Hungaroring': 4381   # Hungarian GP
        }
        
        for track_name, length in standard_tracks.items():
            if track_name not in self.track_maps:
                sections = self.create_default_sections(track_name, length)
                self.track_maps[track_name] = sections
        
        print(f"✅ Built maps for {len(self.track_maps)} tracks")
    
    def build_enhanced_driver_models(self):
        """Build driver models with both real data and FastF1 insights"""
        print("👥 Building enhanced driver models...")
        
        # Start with mock drivers to ensure we have a baseline
        f1_drivers = {
            'HAM': {'skill': 0.95, 'consistency': 0.90, 'rain': 1.15},
            'VER': {'skill': 0.98, 'consistency': 0.95, 'rain': 1.05},
            'LEC': {'skill': 0.92, 'consistency': 0.85, 'rain': 0.95},
            'RUS': {'skill': 0.88, 'consistency': 0.92, 'rain': 1.00},
            'SAI': {'skill': 0.86, 'consistency': 0.88, 'rain': 0.98},
            'NOR': {'skill': 0.84, 'consistency': 0.82, 'rain': 0.92},
            'PIA': {'skill': 0.82, 'consistency': 0.80, 'rain': 0.88},
            'ALO': {'skill': 0.90, 'consistency': 0.94, 'rain': 1.10},
            'STR': {'skill': 0.81, 'consistency': 0.83, 'rain': 0.95},
            'PER': {'skill': 0.79, 'consistency': 0.78, 'rain': 0.90}
        }
        
        for driver_name, stats in f1_drivers.items():
            self.driver_models[driver_name] = {}
            
            for track_name, track_sections in self.track_maps.items():
                section_speeds = {}
                section_consistency = {}
                
                for section in track_sections:
                    base_speed = section.avg_speed
                    
                    # Apply driver skill based on section type
                    if section.corner_type in ['slow_corner', 'fast_corner', 'chicane']:
                        skill_modifier = stats['skill'] * np.random.uniform(0.95, 1.05)
                    else:
                        skill_modifier = stats['skill'] * np.random.uniform(0.98, 1.02)
                    
                    section_speeds[section.section_id] = base_speed * skill_modifier
                    section_consistency[section.section_id] = 8.0 / stats['consistency']
                
                model = DriverTrackModel(
                    driver_id=driver_name,
                    track_name=track_name,
                    section_speeds=section_speeds,
                    section_consistency=section_consistency,
                    braking_efficiency=20.0 * stats['skill'],
                    acceleration_efficiency=5.0 * stats['skill'],
                    cornering_speed=stats['skill'],
                    tire_management=stats['consistency'],
                    fuel_efficiency=1.0,
                    rain_performance=stats['rain'],
                    crash_probability=0.005 / stats['skill'],
                    mechanical_failure_rate=0.002
                )
                self.driver_models[driver_name][track_name] = model
        
        print(f"✅ Built models for {len(self.driver_models)} drivers")
    
    def simulate_lap_physics(self, driver_model: DriverTrackModel, track_sections: List[TrackSection], 
                           weather: str, tire_condition: float, fuel_load: float) -> Dict:
        """Enhanced lap simulation with FastF1 insights"""
        
        total_time = 0.0
        incidents = []
        
        # Weather effects
        weather_effects = {'dry': 1.0, 'wet': 0.85, 'mixed': 0.92}
        weather_factor = weather_effects[weather]
        
        # Rain performance modifier
        if weather != 'dry':
            weather_factor *= driver_model.rain_performance
        
        for section in track_sections:
            base_speed = driver_model.section_speeds.get(section.section_id, section.avg_speed)
            
            # Apply physics effects
            section_speed = base_speed * weather_factor
            
            # Tire degradation effect (more important in corners)
            if section.corner_type in ['slow_corner', 'fast_corner', 'chicane']:
                tire_factor = 0.75 + (tire_condition * 0.25)
                section_speed *= tire_factor
            
            # Fuel load effect
            fuel_factor = 1.0 - ((fuel_load - 50) * 0.0003)
            section_speed *= fuel_factor
            
            # Add driver consistency
            consistency = driver_model.section_consistency.get(section.section_id, 5.0)
            speed_variation = np.random.normal(0, consistency * 0.05)
            section_speed = max(10.0, section_speed + speed_variation)
            
            # Calculate time
            section_time = (section.length_meters / 1000) / (section_speed / 3600) * 3600
            total_time += section_time
            
            # Incident probability
            incident_prob = driver_model.crash_probability
            if section.corner_type == 'chicane':
                incident_prob *= 2.5
            elif section.corner_type in ['slow_corner', 'fast_corner']:
                incident_prob *= 1.8
            
            if weather == 'wet':
                incident_prob *= 3.0
            elif weather == 'mixed':
                incident_prob *= 1.8
            
            if np.random.random() < incident_prob:
                incidents.append(f"{section.corner_type}_incident")
                total_time *= np.random.uniform(1.1, 1.4)
        
        return {
            'lap_time': total_time,
            'incidents': incidents,
            'avg_speed': sum(s.length_meters for s in track_sections) / total_time * 3.6 if total_time > 0 else 0
        }
    
    def run_enhanced_championship(self, num_simulations: int = 20):
        """Run championship with FastF1 enhanced data"""
        print(f"🏆 Running FastF1-Enhanced Championship ({num_simulations} simulations)")
        
        self.build_fastf1_track_maps()
        self.build_enhanced_driver_models()
        
        available_tracks = list(self.track_maps.keys())
        print(f"🏁 Simulating {len(available_tracks)} tracks")
        
        championship_results = {}
        
        for track_name in tqdm(available_tracks, desc="Simulating tracks"):
            track_sections = self.track_maps[track_name]
            track_results = []
            
            # Get all drivers
            drivers = list(self.driver_models.keys())
            
            for sim_num in range(num_simulations):
                weather = np.random.choice(['dry', 'wet', 'mixed'], p=[0.75, 0.15, 0.10])
                race_results = []
                
                for driver_name in drivers:
                    model = self.driver_models[driver_name][track_name]
                    
                    # Simulate race (5 representative laps)
                    lap_times = []
                    total_incidents = []
                    tire_condition = 1.0
                    fuel_load = 100.0
                    
                    for lap_num in range(5):
                        lap_result = self.simulate_lap_physics(
                            model, track_sections, weather, tire_condition, fuel_load
                        )
                        
                        lap_times.append(lap_result['lap_time'])
                        total_incidents.extend(lap_result['incidents'])
                        
                        # Degradation
                        tire_condition *= 0.97
                        fuel_load -= 2.0
                    
                    avg_lap_time = np.mean(lap_times)
                    race_results.append({
                        'driver_id': driver_name,
                        'avg_lap_time': avg_lap_time,
                        'best_lap_time': min(lap_times),
                        'incidents': len(total_incidents),
                        'weather': weather
                    })
                
                # Sort and assign points
                race_results.sort(key=lambda x: x['avg_lap_time'])
                points_map = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
                
                for pos, result in enumerate(race_results, 1):
                    result['position'] = pos
                    result['points'] = points_map.get(pos, 0)
                
                track_results.append({
                    'simulation': sim_num,
                    'results': race_results,
                    'weather': weather
                })
            
            championship_results[track_name] = track_results
        
        # Calculate standings
        standings = self.calculate_championship_standings(championship_results)
        
        return {
            'championship_results': championship_results,
            'standings': standings,
            'metadata': {
                'simulation_type': 'fastf1_enhanced',
                'tracks': available_tracks,
                'total_simulations': num_simulations * len(available_tracks),
                'fastf1_enabled': FASTF1_AVAILABLE
            }
        }
    
    def calculate_championship_standings(self, results: Dict) -> List[Dict]:
        """Calculate championship standings"""
        print("📊 Calculating championship standings...")
        
        driver_stats = {}
        
        for track_name, track_sims in results.items():
            for sim in track_sims:
                for result in sim['results']:
                    driver_id = result['driver_id']
                    
                    if driver_id not in driver_stats:
                        driver_stats[driver_id] = {
                            'points': 0, 'races': 0, 'wins': 0, 'podiums': 0,
                            'avg_position': [], 'total_incidents': 0
                        }
                    
                    stats = driver_stats[driver_id]
                    stats['points'] += result['points']
                    stats['races'] += 1
                    stats['avg_position'].append(result['position'])
                    stats['total_incidents'] += result['incidents']
                    
                    if result['position'] == 1:
                        stats['wins'] += 1
                    if result['position'] <= 3:
                        stats['podiums'] += 1
        
        # Create standings
        standings = []
        for driver_id, stats in driver_stats.items():
            standings.append({
                'position': 0,
                'driver_id': driver_id,
                'points': stats['points'],
                'races': stats['races'],
                'wins': stats['wins'],
                'podiums': stats['podiums'],
                'avg_position': np.mean(stats['avg_position']),
                'incident_rate': stats['total_incidents'] / stats['races']
            })
        
        standings.sort(key=lambda x: x['points'], reverse=True)
        
        for i, driver in enumerate(standings, 1):
            driver['position'] = i
        
        return standings
    
    def display_championship_results(self, results):
        """Display championship results with nice formatting"""
        print("\n" + "="*80)
        print("🏆 F1 PHYSICS-BASED CHAMPIONSHIP RESULTS")
        print("="*80)
        
        # Display metadata
        metadata = results.get('metadata', {})
        print(f"📊 Simulation Summary:")
        print(f"   • Simulation Type: {metadata.get('simulation_type', 'Unknown')}")
        print(f"   • Total Simulations: {metadata.get('total_simulations', 'Unknown')}")
        print(f"   • Tracks Simulated: {len(metadata.get('tracks', []))}")
        print(f"   • FastF1 Enabled: {metadata.get('fastf1_enabled', False)}")
        
        # Display track list
        if metadata.get('tracks'):
            print(f"\n🏁 Tracks Simulated:")
            tracks = metadata['tracks']
            for i, track in enumerate(tracks, 1):
                print(f"   {i:2d}. {track}")
        
        # Display championship standings
        standings = results.get('standings', [])
        if standings:
            print(f"\n🏆 FINAL CHAMPIONSHIP STANDINGS:")
            print("-"*80)
            print(f"{'Pos':<3} {'Driver':<8} {'Points':<7} {'Wins':<5} {'Podiums':<8} {'Avg Pos':<8} {'Races':<6} {'Incidents/Race':<12}")
            print("-"*80)
            
            for driver in standings:
                print(f"{driver['position']:<3} "
                     f"{driver['driver_id']:<8} "
                     f"{driver['points']:<7} "
                     f"{driver['wins']:<5} "
                     f"{driver['podiums']:<8} "
                     f"{driver['avg_position']:<8.1f} "
                     f"{driver['races']:<6} "
                     f"{driver['incident_rate']:<12.2f}")
            
            print("-"*80)
            
            # Championship summary
            winner = standings[0]
            runner_up = standings[1] if len(standings) > 1 else None
            third = standings[2] if len(standings) > 2 else None
            
            print(f"\n🏅 CHAMPIONSHIP SUMMARY:")
            print(f"🥇 Champion: {winner['driver_id']} with {winner['points']} points ({winner['wins']} wins)")
            if runner_up:
                points_gap = winner['points'] - runner_up['points']
                print(f"🥈 Runner-up: {runner_up['driver_id']} with {runner_up['points']} points (-{points_gap} points)")
            if third:
                points_gap = winner['points'] - third['points']
                print(f"🥉 Third: {third['driver_id']} with {third['points']} points (-{points_gap} points)")
            
            # Performance insights
            print(f"\n📈 PERFORMANCE INSIGHTS:")
            total_races = winner['races']
            win_percentage = (winner['wins'] / total_races) * 100
            podium_percentage = (winner['podiums'] / total_races) * 100
            
            print(f"   • {winner['driver_id']} dominated with {win_percentage:.1f}% win rate")
            print(f"   • {winner['driver_id']} achieved {podium_percentage:.1f}% podium rate")
            print(f"   • Average incidents per race: {winner['incident_rate']:.2f}")
            
            # Competition analysis
            if runner_up:
                runner_up_win_pct = (runner_up['wins'] / total_races) * 100
                print(f"   • {runner_up['driver_id']} won {runner_up_win_pct:.1f}% of races")
            
        else:
            print("\n❌ No championship standings found")
        
        print("\n" + "="*80)
    
    def run(self):
        """Main execution"""
        print("🚀 FastF1-Enhanced F1 Physics Simulator")
        print(f"📡 FastF1 Available: {FASTF1_AVAILABLE}")
        
        try:
            results = self.run_enhanced_championship(num_simulations=15)
            
            if not results['standings']:
                print("❌ No championship standings generated")
                return 1
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"f1_fastf1_championship_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"\n💾 Results saved to {filename}")
            
            # Display the championship results
            self.display_championship_results(results)
            
            print("🎉 Championship simulation complete!")
            
            return 0
            
        except Exception as e:
            print(f"❌ Simulation failed: {e}")
            import traceback
            traceback.print_exc()
            return 1

if __name__ == "__main__":
    simulator = FastF1EnhancedSimulator()
    exit(simulator.run())