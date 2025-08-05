#!/usr/bin/env python3

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class IncidentType(Enum):
    VSC = "Virtual Safety Car"
    SC = "Safety Car" 
    RED_FLAG = "Red Flag"
    NONE = "No Incident"

@dataclass
class RaceIncident:
    """Represents a race incident"""
    incident_type: IncidentType
    lap_number: int
    duration_laps: int
    cause: str
    track_section: str
    weather_factor: float
    probability: float

@dataclass
class TrackIncidentProfile:
    """Track-specific incident characteristics"""
    track_name: str
    base_vsc_probability: float      # Per lap base probability
    base_sc_probability: float       # Per lap base probability  
    base_red_flag_probability: float # Per lap base probability
    incident_prone_sections: List[str]  # High-risk track sections
    weather_sensitivity: float      # How much weather affects incidents
    track_type: str                 # 'street', 'permanent', 'hybrid'

class F1IncidentPredictor:
    def __init__(self):
        self.track_profiles = self._initialize_track_profiles()
        self.driver_risk_factors = self._initialize_driver_profiles()
        
    def _initialize_track_profiles(self) -> Dict[str, TrackIncidentProfile]:
        """Initialize incident probability profiles for F1 tracks"""
        profiles = {}
        
        # Street Circuits - Higher incident probability
        profiles['Monaco'] = TrackIncidentProfile(
            track_name='Monaco',
            base_vsc_probability=0.025,      # 2.5% per lap
            base_sc_probability=0.015,       # 1.5% per lap
            base_red_flag_probability=0.003, # 0.3% per lap
            incident_prone_sections=['Nouvelle Chicane', 'Rascasse', 'Sainte Devote'],
            weather_sensitivity=2.5,
            track_type='street'
        )
        
        profiles['Marina Bay'] = TrackIncidentProfile(
            track_name='Marina Bay',
            base_vsc_probability=0.020,
            base_sc_probability=0.012,
            base_red_flag_probability=0.002,
            incident_prone_sections=['Turn 10', 'Turn 14', 'Turn 18'],
            weather_sensitivity=1.8,
            track_type='street'
        )
        
        profiles['Baku'] = TrackIncidentProfile(
            track_name='Baku',
            base_vsc_probability=0.030,      # Very high due to walls
            base_sc_probability=0.018,
            base_red_flag_probability=0.004,
            incident_prone_sections=['Castle Section', 'Turn 15', 'Turn 20'],
            weather_sensitivity=2.0,
            track_type='street'
        )
        
        # Permanent Circuits - Lower incident probability
        profiles['Silverstone'] = TrackIncidentProfile(
            track_name='Silverstone',
            base_vsc_probability=0.008,
            base_sc_probability=0.005,
            base_red_flag_probability=0.001,
            incident_prone_sections=['Copse', 'Stowe', 'Club'],
            weather_sensitivity=1.5,
            track_type='permanent'
        )
        
        profiles['Spa'] = TrackIncidentProfile(
            track_name='Spa-Francorchamps',
            base_vsc_probability=0.012,
            base_sc_probability=0.008,
            base_red_flag_probability=0.002,
            incident_prone_sections=['Eau Rouge', 'Blanchimont', 'Bus Stop'],
            weather_sensitivity=3.0,  # Very weather sensitive
            track_type='permanent'
        )
        
        profiles['Monza'] = TrackIncidentProfile(
            track_name='Monza',
            base_vsc_probability=0.006,      # Low incident rate
            base_sc_probability=0.004,
            base_red_flag_probability=0.001,
            incident_prone_sections=['Chicane della Roggia', 'Ascari', 'Parabolica'],
            weather_sensitivity=1.2,
            track_type='permanent'
        )
        
        # Hybrid Circuits - Medium incident probability
        profiles['Suzuka'] = TrackIncidentProfile(
            track_name='Suzuka',
            base_vsc_probability=0.010,
            base_sc_probability=0.006,
            base_red_flag_probability=0.0015,
            incident_prone_sections=['Turn 1', 'Spoon Curve', '130R'],
            weather_sensitivity=2.2,
            track_type='permanent'
        )
        
        profiles['Interlagos'] = TrackIncidentProfile(
            track_name='Interlagos',
            base_vsc_probability=0.015,
            base_sc_probability=0.010,
            base_red_flag_probability=0.002,
            incident_prone_sections=['Turn 1', 'Senna S', 'Juncao'],
            weather_sensitivity=2.8,  # Very rainy
            track_type='permanent'
        )
        
        # Add more tracks with default values
        default_tracks = ['Sakhir', 'Jeddah', 'Melbourne', 'Imola', 'Montréal', 
                         'Barcelona', 'Budapest', 'Zandvoort', 'Mexico City', 
                         'Las Vegas', 'Yas Island', 'Austin']
        
        for track in default_tracks:
            profiles[track] = TrackIncidentProfile(
                track_name=track,
                base_vsc_probability=0.012,
                base_sc_probability=0.007,
                base_red_flag_probability=0.0015,
                incident_prone_sections=['Turn 1', 'Final Corner'],
                weather_sensitivity=1.5,
                track_type='permanent'
            )
        
        return profiles
    
    def _initialize_driver_profiles(self) -> Dict[str, float]:
        """Initialize driver risk factors (1.0 = average, >1.0 = higher incident risk)"""
        return {
            'VER': 0.8,   # Conservative, clean driver
            'HAM': 0.7,   # Very experienced, clean
            'LEC': 1.2,   # Aggressive, occasional mistakes
            'RUS': 0.9,   # Clean but still learning
            'SAI': 1.0,   # Average risk
            'NOR': 1.1,   # Sometimes pushes too hard
            'PIA': 1.3,   # Rookie, learning
            'ALO': 0.8,   # Experienced, clean
            'STR': 1.4,   # Aggressive, higher incident rate
            'PER': 1.2,   # Sometimes struggles under pressure
        }
    
    def calculate_lap_incident_probability(self, track_name: str, lap_number: int, 
                                         total_laps: int, weather: str, 
                                         championship_position: Dict[str, int]) -> Dict[IncidentType, float]:
        """Calculate incident probabilities for a specific lap"""
        
        if track_name not in self.track_profiles:
            track_name = 'Silverstone'  # Default fallback
        
        profile = self.track_profiles[track_name]
        
        # Base probabilities
        vsc_prob = profile.base_vsc_probability
        sc_prob = profile.base_sc_probability
        red_flag_prob = profile.base_red_flag_probability
        
        # Weather multiplier
        weather_multipliers = {
            'dry': 1.0,
            'mixed': 1.6,
            'wet': 2.2,
            'storm': 3.5
        }
        weather_factor = weather_multipliers.get(weather, 1.0)
        
        # Apply weather sensitivity
        vsc_prob *= (1 + (weather_factor - 1) * profile.weather_sensitivity)
        sc_prob *= (1 + (weather_factor - 1) * profile.weather_sensitivity)
        red_flag_prob *= (1 + (weather_factor - 1) * profile.weather_sensitivity)
        
        # Lap phase multipliers
        race_phase = lap_number / total_laps
        
        if race_phase <= 0.05:  # Opening 5% - start chaos
            phase_multiplier = 3.0
        elif race_phase <= 0.20:  # Early race - settling in
            phase_multiplier = 1.2
        elif race_phase <= 0.70:  # Mid race - stable
            phase_multiplier = 0.8
        elif race_phase <= 0.90:  # Late race - desperation
            phase_multiplier = 1.8
        else:  # Final laps - maximum desperation
            phase_multiplier = 2.5
        
        vsc_prob *= phase_multiplier
        sc_prob *= phase_multiplier
        red_flag_prob *= phase_multiplier
        
        # Championship desperation factor
        # Drivers further down the championship are more likely to take risks
        avg_championship_factor = 1.0
        if championship_position:
            total_desperation = sum(1.0 + (pos - 1) * 0.1 for pos in championship_position.values())
            avg_championship_factor = total_desperation / len(championship_position)
        
        vsc_prob *= avg_championship_factor
        sc_prob *= avg_championship_factor
        red_flag_prob *= avg_championship_factor
        
        # Ensure probabilities don't exceed reasonable limits
        vsc_prob = min(vsc_prob, 0.15)      # Max 15% per lap
        sc_prob = min(sc_prob, 0.08)        # Max 8% per lap
        red_flag_prob = min(red_flag_prob, 0.02)  # Max 2% per lap
        
        return {
            IncidentType.VSC: vsc_prob,
            IncidentType.SC: sc_prob,
            IncidentType.RED_FLAG: red_flag_prob,
            IncidentType.NONE: max(0, 1 - vsc_prob - sc_prob - red_flag_prob)
        }
    
    def determine_incident_duration(self, incident_type: IncidentType, track_name: str) -> int:
        """Determine how many laps an incident lasts"""
        
        if incident_type == IncidentType.VSC:
            # VSC typically 1-3 laps
            return np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
        
        elif incident_type == IncidentType.SC:
            # Safety Car typically 3-8 laps
            track_type = self.track_profiles.get(track_name, self.track_profiles['Silverstone']).track_type
            if track_type == 'street':
                # Street circuits take longer to clear
                return np.random.choice([4, 5, 6, 7, 8, 9], p=[0.1, 0.2, 0.3, 0.2, 0.15, 0.05])
            else:
                return np.random.choice([3, 4, 5, 6, 7], p=[0.2, 0.3, 0.3, 0.15, 0.05])
        
        elif incident_type == IncidentType.RED_FLAG:
            # Red flag can be 5-20 laps depending on severity
            severity = np.random.choice(['minor', 'major', 'severe'], p=[0.5, 0.3, 0.2])
            if severity == 'minor':
                return np.random.randint(5, 10)
            elif severity == 'major':
                return np.random.randint(8, 15)
            else:  # severe
                return np.random.randint(12, 25)
        
        return 0
    
    def determine_incident_cause(self, incident_type: IncidentType, track_name: str, 
                                weather: str, lap_number: int) -> str:
        """Determine the likely cause of an incident"""
        
        causes = {
            IncidentType.VSC: [
                "Debris on track", "Minor spin", "Car stopped on track", 
                "Small collision", "Mechanical failure", "Track limits violation cleanup"
            ],
            IncidentType.SC: [
                "Multi-car collision", "Car in dangerous position", "Barrier damage",
                "Large debris", "Oil spill", "Abandoned vehicle", "Track surface issue"
            ],
            IncidentType.RED_FLAG: [
                "Major crash with injuries", "Severe weather", "Barrier reconstruction needed",
                "Track surface damage", "Multiple car pile-up", "Fire incident", "Medical emergency"
            ]
        }
        
        # Weather-specific causes
        if weather in ['wet', 'storm', 'mixed']:
            if incident_type == IncidentType.VSC:
                causes[IncidentType.VSC].extend(["Aquaplaning incident", "Weather-related spin"])
            elif incident_type == IncidentType.SC:
                causes[IncidentType.SC].extend(["Weather-related crash", "Multiple weather spins"])
            elif incident_type == IncidentType.RED_FLAG:
                causes[IncidentType.RED_FLAG].extend(["Severe weather conditions", "Multiple weather crashes"])
        
        # Track-specific causes
        if track_name in ['Monaco', 'Marina Bay', 'Baku']:  # Street circuits
            if incident_type == IncidentType.SC:
                causes[IncidentType.SC].extend(["Crash into barriers", "Narrow track incident"])
        
        return np.random.choice(causes.get(incident_type, ["Unknown incident"]))
    
    def simulate_race_incidents(self, track_name: str, total_laps: int, weather: str,
                               championship_standings: Dict[str, int]) -> List[RaceIncident]:
        """Simulate all incidents for a complete race"""
        
        incidents = []
        current_lap = 1
        
        while current_lap <= total_laps:
            # Calculate probabilities for this lap
            lap_probs = self.calculate_lap_incident_probability(
                track_name, current_lap, total_laps, weather, championship_standings
            )
            
            # Randomly determine if an incident occurs
            incident_types = list(lap_probs.keys())
            probabilities = list(lap_probs.values())
            
            selected_incident = np.random.choice(incident_types, p=probabilities)
            
            if selected_incident != IncidentType.NONE:
                # Determine incident details
                duration = self.determine_incident_duration(selected_incident, track_name)
                cause = self.determine_incident_cause(selected_incident, track_name, weather, current_lap)
                
                # Select incident-prone section
                profile = self.track_profiles.get(track_name, self.track_profiles['Silverstone'])
                section = np.random.choice(profile.incident_prone_sections)
                
                incident = RaceIncident(
                    incident_type=selected_incident,
                    lap_number=current_lap,
                    duration_laps=duration,
                    cause=cause,
                    track_section=section,
                    weather_factor=lap_probs[selected_incident],
                    probability=lap_probs[selected_incident]
                )
                
                incidents.append(incident)
                
                # Skip laps during incident
                current_lap += duration
            else:
                current_lap += 1
        
        return incidents
    
    def run_monte_carlo_season(self, tracks: List[str], num_simulations: int = 10000) -> Dict:
        """Run Monte Carlo simulation for a full season"""
        
        print(f"🚨 Running F1 Incident Monte Carlo Simulation")
        print(f"📊 Simulating {len(tracks)} races × {num_simulations} iterations")
        
        season_results = {}
        
        for track in tracks:
            print(f"🏁 Simulating {track}...")
            
            track_incidents = []
            
            for sim in range(num_simulations):
                # Random race conditions
                weather = np.random.choice(['dry', 'mixed', 'wet'], p=[0.75, 0.15, 0.10])
                race_laps = np.random.randint(50, 70)  # Typical F1 race length
                
                # Mock championship standings (would use real data)
                championship_standings = {
                    f'Driver_{i}': i for i in range(1, 21)
                }
                
                # Simulate race incidents
                race_incidents = self.simulate_race_incidents(
                    track, race_laps, weather, championship_standings
                )
                
                track_incidents.extend([{
                    'simulation': sim,
                    'track': track,
                    'weather': weather,
                    'race_laps': race_laps,
                    'incident_type': incident.incident_type.value,
                    'lap_number': incident.lap_number,
                    'duration_laps': incident.duration_laps,
                    'cause': incident.cause,
                    'track_section': incident.track_section,
                    'probability': incident.probability
                } for incident in race_incidents])
            
            season_results[track] = track_incidents
        
        return season_results
    
    def analyze_incident_patterns(self, results: Dict) -> Dict:
        """Analyze patterns in the Monte Carlo results"""
        
        print("📈 Analyzing incident patterns...")
        
        analysis = {
            'track_incident_rates': {},
            'incident_type_distribution': {},
            'weather_impact': {},
            'lap_phase_patterns': {},
            'track_rankings': {}
        }
        
        all_incidents = []
        for track, incidents in results.items():
            all_incidents.extend(incidents)
        
        df = pd.DataFrame(all_incidents)
        
        if df.empty:
            return analysis
        
        # Track incident rates
        for track in df['track'].unique():
            track_data = df[df['track'] == track]
            simulations = track_data['simulation'].nunique()
            
            analysis['track_incident_rates'][track] = {
                'total_incidents': len(track_data),
                'incidents_per_race': len(track_data) / simulations,
                'vsc_rate': len(track_data[track_data['incident_type'] == 'Virtual Safety Car']) / simulations,
                'sc_rate': len(track_data[track_data['incident_type'] == 'Safety Car']) / simulations,
                'red_flag_rate': len(track_data[track_data['incident_type'] == 'Red Flag']) / simulations
            }
        
        # Incident type distribution
        incident_counts = df['incident_type'].value_counts()
        total_incidents = len(df)
        
        for incident_type, count in incident_counts.items():
            analysis['incident_type_distribution'][incident_type] = {
                'count': count,
                'percentage': (count / total_incidents) * 100
            }
        
        # Weather impact
        weather_groups = df.groupby('weather')['incident_type'].count()
        total_by_weather = df.groupby('weather')['simulation'].nunique()
        
        for weather in weather_groups.index:
            incidents_in_weather = weather_groups[weather]
            races_in_weather = total_by_weather[weather]
            analysis['weather_impact'][weather] = incidents_in_weather / races_in_weather
        
        # Lap phase patterns
        df['race_phase'] = df['lap_number'] / df['race_laps']
        phase_bins = pd.cut(df['race_phase'], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], 
                           labels=['Start', 'Early', 'Mid', 'Late', 'Final'])
        phase_incidents = phase_bins.value_counts()
        
        for phase, count in phase_incidents.items():
            analysis['lap_phase_patterns'][str(phase)] = count
        
        # Track safety rankings (lower incidents = safer)
        track_safety = []
        for track, stats in analysis['track_incident_rates'].items():
            track_safety.append((track, stats['incidents_per_race']))
        
        track_safety.sort(key=lambda x: x[1])
        safest = [track for track, _ in track_safety[:5]]
        
        # sort descending for most incident-prone
        most_prone = [track for track, _ in sorted(track_safety, key=lambda x: x[1], reverse=True)[:5]]
        
        analysis['track_rankings'] = {
           'safest_tracks': safest,
            'most_incident_prone': most_prone
        }
        
        return analysis
    
    def display_results(self, results: Dict, analysis: Dict):
        """Display comprehensive incident prediction results"""
        
        print("\n" + "="*80)
        print("🚨 F1 RACE INCIDENT PREDICTION RESULTS")
        print("="*80)
        
        # Overall statistics
        total_incidents = sum(len(incidents) for incidents in results.values())
        total_simulations = sum(max(inc.get('simulation', 0) for inc in incidents) + 1 
                              for incidents in results.values() if incidents)
        
        print(f"📊 SIMULATION SUMMARY:")
        print(f"   • Total Incidents Simulated: {total_incidents:,}")
        print(f"   • Total Race Simulations: {total_simulations:,}")
        print(f"   • Average Incidents per Race: {total_incidents/total_simulations:.2f}")
        
        # Incident type distribution
        print(f"\n🚨 INCIDENT TYPE DISTRIBUTION:")
        for incident_type, stats in analysis['incident_type_distribution'].items():
            print(f"   • {incident_type}: {stats['count']:,} ({stats['percentage']:.1f}%)")
        
        # Track safety rankings
        print(f"\n🏁 TRACK SAFETY RANKINGS:")
        print(f"\n✅ SAFEST TRACKS (Lowest Incident Rate):")
        for i, track in enumerate(analysis['track_rankings']['safest_tracks'], 1):
            rate = analysis['track_incident_rates'][track]['incidents_per_race']
            print(f"   {i}. {track}: {rate:.2f} incidents/race")
        
        print(f"\n⚠️ HIGHEST INCIDENT TRACKS:")
        for i, track in enumerate(analysis['track_rankings']['most_incident_prone'], 1):
            rate = analysis['track_incident_rates'][track]['incidents_per_race']
            print(f"   {i}. {track}: {rate:.2f} incidents/race")
        
        # Weather impact
        print(f"\n🌦️ WEATHER IMPACT ON INCIDENTS:")
        for weather, rate in analysis['weather_impact'].items():
            print(f"   • {weather.title()}: {rate:.2f} incidents/race")
        
        # Lap phase patterns
        print(f"\n⏱️ INCIDENT TIMING PATTERNS:")
        for phase, count in analysis['lap_phase_patterns'].items():
            percentage = (count / sum(analysis['lap_phase_patterns'].values())) * 100
            print(f"   • {phase} Race: {count:,} incidents ({percentage:.1f}%)")
        
        # Detailed track analysis
        print(f"\n🏁 DETAILED TRACK INCIDENT ANALYSIS:")
        print(f"{'Track':<20} {'VSC/Race':<10} {'SC/Race':<10} {'Red Flag/Race':<15} {'Total/Race':<12}")
        print("-" * 80)
        
        for track, stats in sorted(analysis['track_incident_rates'].items(), 
                                 key=lambda x: x[1]['incidents_per_race'], reverse=True):
            print(f"{track:<20} {stats['vsc_rate']:<10.3f} {stats['sc_rate']:<10.3f} "
                  f"{stats['red_flag_rate']:<15.3f} {stats['incidents_per_race']:<12.2f}")
        
        print("\n" + "="*80)
    
    def run_full_analysis(self, tracks: List[str] = None, num_simulations: int = 1000):
        """Run complete incident prediction analysis"""
        
        if tracks is None:
            tracks = ['Monaco', 'Silverstone', 'Spa', 'Monza', 'Suzuka', 'Interlagos', 
                     'Baku', 'Marina Bay', 'Sakhir', 'Melbourne']
        
        # Run Monte Carlo simulation
        results = self.run_monte_carlo_season(tracks, num_simulations)
        
        # Analyze patterns
        analysis = self.analyze_incident_patterns(results)
        
        # Display results
        self.display_results(results, analysis)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"f1_incident_prediction_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump({
                'results': results,
                'analysis': analysis,
                'metadata': {
                    'tracks': tracks,
                    'simulations': num_simulations,
                    'timestamp': timestamp
                }
            }, f, indent=2, default=str)
        
        print(f"💾 Results saved to {filename}")
        
        return results, analysis

if __name__ == "__main__":
    predictor = F1IncidentPredictor()
    
    # Run analysis for key F1 tracks
    results, analysis = predictor.run_full_analysis(num_simulations=1000)
    
    print("🎉 F1 Incident Prediction Analysis Complete!")