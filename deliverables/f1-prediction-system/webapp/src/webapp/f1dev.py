from flask import Flask, render_template, request, jsonify
from flask_caching import Cache

import plotly.graph_objects as go
import plotly.utils
from plotly.subplots import make_subplots

import numpy as np
import pandas as pd

import json 

import fastf1
from fastf1 import get_session
fastf1.Cache.enable_cache('E:\School Stuff\F1cache')

probs_2022 = np.load('2022probs.npy')


def extract_full_race_car_positions(session, drivers = ['VER', 'HAM', 'PER', 'ALO', 'SAI', 'RUS', 'PIA', 'STR', 'GAS', 'NOR', 'LEC', 'OCO', 'ALB', 'TSU', 'BOT', 'HUL', 'RIC', 'ZHO', 'MAG', 'DEV', 'SAR'], target_timesteps=None):
    """
    Extract car positions for THE ENTIRE RACE with proper synchronization
    
    Args:
        session: FastF1 session object
        drivers: List of driver codes
        target_timesteps: Target number of timesteps to generate (for sync with probability data)
    """
    
    try:
        # Get track outline from fastest lap
        fastest_lap = session.laps.pick_fastest()
        track_telemetry = fastest_lap.get_telemetry()
        
        track_x = track_telemetry['X'].dropna().values
        track_y = track_telemetry['Y'].dropna().values
        
        # Sample track for performance  
        track_sample_rate = max(1, len(track_x) // 200)
        track_x = track_x[::track_sample_rate]
        track_y = track_y[::track_sample_rate]
        
        print(f"Track outline: {len(track_x)} points")
        
        # Get ALL telemetry data for each driver (entire race)
        all_drivers_telemetry = {}

        available_drivers = session.laps['Driver'].unique()
        drivers = [d for d in drivers if d in available_drivers]
        
        if not drivers:
            raise ValueError("No valid drivers found in session")
        
        for driver in drivers:
            print(f"Processing {driver}...")
            
            try:
                # Get ALL laps for this driver
                driver_laps = session.laps.pick_driver(driver)
                print(f"  Found {len(driver_laps)} laps for {driver}")
                
                # Collect telemetry from ALL laps
                all_lap_telemetry = []
                
                for lap_number, lap in driver_laps.iterrows():
                    try:
                        lap_tel = lap.get_telemetry()
                        if len(lap_tel) > 0 and 'X' in lap_tel.columns:
                            # Add lap info for reference
                            lap_tel = lap_tel.copy()
                            lap_tel['LapNumber'] = lap['LapNumber']
                            lap_tel['Driver'] = driver
                            
                            # Remove invalid coordinates
                            lap_tel = lap_tel.dropna(subset=['X', 'Y'])
                            
                            if len(lap_tel) > 0:
                                all_lap_telemetry.append(lap_tel)
                                
                    except Exception as e:
                        print(f"    Skipping lap {lap['LapNumber']}: {e}")
                        continue
                
                if all_lap_telemetry:
                    # COMBINE ALL LAPS INTO ONE CONTINUOUS TELEMETRY
                    full_race_telemetry = pd.concat(all_lap_telemetry, ignore_index=True)
                    all_drivers_telemetry[driver] = full_race_telemetry
                    print(f"  {driver}: {len(full_race_telemetry)} total points (entire race)")
                else:
                    print(f"  No valid telemetry for {driver}")
                    
            except Exception as e:
                print(f"Error processing {driver}: {e}")
                continue
        
        if not all_drivers_telemetry:
            print("No telemetry data found for any driver!")
            return None
        
        # Find the longest telemetry sequence
        max_length = max(len(tel) for tel in all_drivers_telemetry.values())
        
        # If target timesteps specified, use that instead
        if target_timesteps:
            final_timesteps = target_timesteps
            print(f"Resampling to {final_timesteps} timesteps to match probability data")
        else:
            final_timesteps = max_length
            print(f"Using {final_timesteps} timesteps from telemetry data")
        
        # Get lap information from the longest telemetry sequence
        reference_driver = max(all_drivers_telemetry.keys(), 
                              key=lambda k: len(all_drivers_telemetry[k]))
        reference_telemetry = all_drivers_telemetry[reference_driver]
        
        print(f"Using {reference_driver} as reference for lap numbers")
        print(f"Reference driver lap range: {reference_telemetry['LapNumber'].min()} to {reference_telemetry['LapNumber'].max()}")
        
        # Create synchronized position arrays with interpolation
        car_x_by_timestep = []
        car_y_by_timestep = []
        lap_numbers = []
        track_status_data = []  # New: track status information
        
        # Get track status data if requested
        weather_data = None
        session_status = None

        try:
            # Get weather data
            weather_data = session.weather_data
            print(f"Weather data available: {len(weather_data) if weather_data is not None else 0} entries")
            
            # Get session status (race control messages)
            if hasattr(session, 'race_control_messages'):
                session_status = session.race_control_messages
                print(f"Race control messages: {len(session_status) if session_status is not None else 0} entries")
            
            # Get lap data for additional status info
            all_laps = session.laps
            print(f"Total laps in session: {len(all_laps)}")
            
        except Exception as e:
            print(f"Warning: Could not extract track status data: {e}")
            #include_track_status = False
        
        for timestep in range(final_timesteps):
            timestep_x = []
            timestep_y = []
            
            # Calculate lap number from reference driver
            if target_timesteps:
                # Map timestep to reference telemetry index
                ref_index = (timestep / target_timesteps) * (len(reference_telemetry) - 1)
                ref_index = int(ref_index)
                ref_index = min(ref_index, len(reference_telemetry) - 1)
                current_lap = reference_telemetry.iloc[ref_index]['LapNumber']
            else:
                if timestep < len(reference_telemetry):
                    current_lap = reference_telemetry.iloc[timestep]['LapNumber']
                else:
                    current_lap = reference_telemetry.iloc[-1]['LapNumber']
            
            for driver in drivers:
                if driver in all_drivers_telemetry:
                    driver_tel = all_drivers_telemetry[driver]
                    
                    # Calculate the corresponding index in driver's telemetry
                    if target_timesteps:
                        # Interpolate to match target timesteps
                        original_index = (timestep / target_timesteps) * (len(driver_tel) - 1)
                        index = int(original_index)
                        
                        # Linear interpolation between two points
                        if index < len(driver_tel) - 1:
                            fraction = original_index - index
                            x_val = (driver_tel.iloc[index]['X'] * (1 - fraction) + 
                                   driver_tel.iloc[index + 1]['X'] * fraction)
                            y_val = (driver_tel.iloc[index]['Y'] * (1 - fraction) + 
                                   driver_tel.iloc[index + 1]['Y'] * fraction)
                        else:
                            x_val = driver_tel.iloc[-1]['X']
                            y_val = driver_tel.iloc[-1]['Y']
                    else:
                        # Use direct indexing
                        if timestep < len(driver_tel):
                            x_val = driver_tel.iloc[timestep]['X']
                            y_val = driver_tel.iloc[timestep]['Y']
                        else:
                            # Repeat last position
                            x_val = driver_tel.iloc[-1]['X']
                            y_val = driver_tel.iloc[-1]['Y']
                    
                    timestep_x.append(x_val)
                    timestep_y.append(y_val)
                else:
                    # Driver not found, use track start position
                    timestep_x.append(track_x[0])
                    timestep_y.append(track_y[0])
            
            car_x_by_timestep.append(timestep_x)
            car_y_by_timestep.append(timestep_y)
            lap_numbers.append(int(current_lap))
        
            # Extract track status for this timestep
            status_info = extract_track_status_for_timestep(
                timestep, final_timesteps, current_lap, 
                weather_data, session_status, reference_telemetry
            )
            track_status_data.append(status_info)

        return {
            'track_x': track_x.tolist(),
            'track_y': track_y.tolist(),
            'x': car_x_by_timestep,
            'y': car_y_by_timestep,
            'lap_numbers': lap_numbers,
            'drivers': drivers,
            'timesteps': final_timesteps,
            'total_laps': max(lap_numbers) if lap_numbers else 71,
            'track_status': track_status_data
        }
        
    except Exception as e:
        print(f"Error extracting full race data: {e}")
        return None

def extract_track_status_for_timestep(timestep, total_timesteps, current_lap, 
                                     weather_data, session_status, reference_telemetry):
    """
    Extract track status information for a specific timestep
    """
    status_info = {
        'lap': current_lap,
        'session_time': None,
        'track_temp': None,
        'air_temp': None,
        'humidity': None,
        'pressure': None,
        'wind_speed': None,
        'wind_direction': None,
        'rainfall': None,
        'track_status': 'Green',  # Default
        'safety_car': False,
        'virtual_safety_car': False,
        'red_flag': False,
        'yellow_flag': False
    }
    
    try:
        # Calculate session time from reference telemetry
        if reference_telemetry is not None and len(reference_telemetry) > 0:
            if timestep < len(reference_telemetry):
                session_time = reference_telemetry.iloc[timestep].get('SessionTime', None)
                if session_time is not None:
                    # Convert to seconds (handle different timestamp types)
                    try:
                        if hasattr(session_time, 'total_seconds'):
                            status_info['session_time'] = session_time.total_seconds()
                        elif hasattr(session_time, 'timestamp'):
                            status_info['session_time'] = session_time.timestamp()
                        elif isinstance(session_time, (int, float)):
                            status_info['session_time'] = float(session_time)
                        else:
                            # Try pandas Timestamp conversion
                            import pandas as pd
                            if isinstance(session_time, pd.Timestamp):
                                # Convert relative to session start
                                status_info['session_time'] = (session_time - pd.Timestamp('1900-01-01')).total_seconds()
                            else:
                                status_info['session_time'] = None
                    except (ValueError, TypeError, AttributeError):
                        status_info['session_time'] = None
        
        # Get weather data for this timestep
        if weather_data is not None and len(weather_data) > 0:
            # Find closest weather entry by session time
            if status_info['session_time']:
                try:
                    # Convert session time to match weather data format
                    closest_weather = weather_data.iloc[0]  # Fallback
                    min_time_diff = float('inf')
                    
                    for _, weather_row in weather_data.iterrows():
                        weather_time = weather_row.get('Time', None)
                        if weather_time is not None:
                            # Convert weather time to seconds (handle different timestamp types)
                            try:
                                if hasattr(weather_time, 'total_seconds'):
                                    weather_time_seconds = weather_time.total_seconds()
                                elif hasattr(weather_time, 'timestamp'):
                                    weather_time_seconds = weather_time.timestamp()
                                elif isinstance(weather_time, (int, float)):
                                    weather_time_seconds = float(weather_time)
                                else:
                                    # Try pandas Timestamp conversion
                                    import pandas as pd
                                    if isinstance(weather_time, pd.Timestamp):
                                        weather_time_seconds = (weather_time - pd.Timestamp('1900-01-01')).total_seconds()
                                    else:
                                        continue
                                
                                session_time_seconds = float(status_info['session_time'])
                                time_diff = abs(weather_time_seconds - session_time_seconds)
                                
                                if time_diff < min_time_diff:
                                    min_time_diff = time_diff
                                    closest_weather = weather_row
                                    
                            except (ValueError, TypeError, AttributeError):
                                continue
                    
                    # Extract weather info (handle missing columns gracefully)
                    status_info['track_temp'] = closest_weather.get('TrackTemp', None)
                    status_info['air_temp'] = closest_weather.get('AirTemp', None)
                    status_info['humidity'] = closest_weather.get('Humidity', None)
                    status_info['pressure'] = closest_weather.get('Pressure', None)
                    status_info['wind_speed'] = closest_weather.get('WindSpeed', None)
                    status_info['wind_direction'] = closest_weather.get('WindDirection', None)
                    status_info['rainfall'] = closest_weather.get('Rainfall', False)
                    
                except Exception as e:
                    print(f"Weather data extraction error: {e}")
        
        # Check for race control messages (flags, safety car, etc.)
        if session_status is not None and len(session_status) > 0:
            try:
                for _, message in session_status.iterrows():
                    message_time = message.get('Time', None)
                    if message_time and status_info['session_time']:
                        # Convert message time to seconds (handle different timestamp types)
                        try:
                            if hasattr(message_time, 'total_seconds'):
                                message_time_seconds = message_time.total_seconds()
                            elif hasattr(message_time, 'timestamp'):
                                message_time_seconds = message_time.timestamp()
                            elif isinstance(message_time, (int, float)):
                                message_time_seconds = float(message_time)
                            else:
                                # Try to convert pandas Timestamp to seconds
                                import pandas as pd
                                if isinstance(message_time, pd.Timestamp):
                                    # Convert to timedelta from session start (assuming session starts at 0)
                                    message_time_seconds = (message_time - pd.Timestamp('1900-01-01')).total_seconds()
                                else:
                                    continue  # Skip if we can't convert
                            
                            # Ensure session_time is also a float
                            session_time_seconds = float(status_info['session_time'])
                            
                            # Check if this message is active for current timestep
                            if message_time_seconds <= session_time_seconds:
                                message_text = str(message.get('Message', '')).upper()
                                
                                # Parse message for track status
                                if 'SAFETY CAR' in message_text and 'VIRTUAL' not in message_text:
                                    status_info['safety_car'] = True
                                    status_info['track_status'] = 'Safety Car'
                                elif 'VIRTUAL SAFETY CAR' in message_text or 'VSC' in message_text:
                                    status_info['virtual_safety_car'] = True
                                    status_info['track_status'] = 'Virtual Safety Car'
                                elif 'RED FLAG' in message_text:
                                    status_info['red_flag'] = True
                                    status_info['track_status'] = 'Red Flag'
                                elif 'YELLOW FLAG' in message_text:
                                    status_info['yellow_flag'] = True
                                    status_info['track_status'] = 'Yellow Flag'
                                elif 'GREEN FLAG' in message_text or 'CLEAR' in message_text or 'ALL CLEAR' in message_text:
                                    # Reset flags
                                    status_info['safety_car'] = False
                                    status_info['virtual_safety_car'] = False
                                    status_info['red_flag'] = False
                                    status_info['yellow_flag'] = False
                                    status_info['track_status'] = 'Green'
                                    
                        except (ValueError, TypeError, AttributeError) as time_error:
                            print(f"Time conversion error for message: {time_error}")
                            continue
                                
            except Exception as e:
                print(f"Race control message parsing error: {e}")
    
    except Exception as e:
        print(f"Track status extraction error: {e}")
    
    return status_info


def create_smooth_animated_bar_with_track(probabilities, car_position_data=None, sample_every=50, animation_speed=150):
    """
    Create smooth animation with synchronized probability and track data
    
    Parameters:
    - probabilities: your probability data (should be list/array)
    - car_position_data: dict with car positions from extract_full_race_car_positions()
    - sample_every: sampling rate for animation smoothness
    - animation_speed: frame duration in milliseconds
    """
    
    # Convert probabilities to numpy array
    probs = np.array(probabilities)
    total_frames = len(probs)
    
    # If car position data is provided, ensure they have same length
    if car_position_data:
        car_frames = len(car_position_data['x'])
        if car_frames != total_frames:
            print(f"Warning: Probability frames ({total_frames}) != Car frames ({car_frames})")
            # Use the shorter sequence
            total_frames = min(total_frames, car_frames)
            probs = probs[:total_frames]
    
    # Sample the data for smooth animation
    sampled_indices = np.arange(0, total_frames, sample_every)
    sampled_probs = probs[sampled_indices] * 100  # Convert to percentage
    
    print(f"Creating animation with {len(sampled_probs)} frames from {total_frames} total timesteps")
    
    if car_position_data:
        sampled_lap_numbers = [car_position_data['lap_numbers'][idx] for idx in sampled_indices]
        print(f"Lap range in animation: {min(sampled_lap_numbers)} to {max(sampled_lap_numbers)}")
        print(f"Sample lap numbers: {sampled_lap_numbers[:10]}...")  # Show first 10
    
    drivers = ['VER', 'HAM', 'PER', 'ALO', 'SAI', 'RUS', 'PIA', 'STR', 'GAS', 'NOR', 'LEC', 'OCO', 'ALB', 'TSU', 'BOT', 'HUL', 'RIC', 'ZHO', 'MAG', 'DEV', 'SAR']
    #driver_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
    
    # Create different probability patterns for each driver (demo purposes)
    bar_probs = []
    for i in range(len(drivers)):
        # Create varied patterns for each driver
        pattern_offset = i * np.pi / 3
        pattern = sampled_probs + 10 * np.sin(np.linspace(0, 4*np.pi, len(sampled_probs)) + pattern_offset)
        pattern = np.clip(pattern, 0, 100)  # Ensure 0-100 range
        bar_probs.append(pattern)

    # CREATE SUBPLOTS
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Safety Car Probability', 'Live Track Position', 'Track Status'),
        specs=[[{"type": "bar"}, {"type": "scatter"}, {"type": "scatter"}]],
        column_widths=[0.4, 0.35, 0.25],
        horizontal_spacing=0.05
    )
    
    # ADD INITIAL BAR TRACE
    initial_values = [bar_probs[i][0] for i in range(len(drivers))]
    fig.add_trace(
        go.Bar(
            y=drivers,
            x=initial_values,
            marker=dict(
                #color=driver_colors,
                line=dict(color='white', width=1)
            ),
            orientation='h',
            name='Probability',
            text=[f'{val:.1f}%' for val in initial_values],
            textposition='outside'
        ),
        row=1, col=1
    )
    
    # ADD TRACK ELEMENTS (if car position data provided)
    if car_position_data:
        # Track outline
        fig.add_trace(
            go.Scatter(
                x=car_position_data['track_x'],
                y=car_position_data['track_y'],
                mode='lines',
                line=dict(color='white', width=4, dash='dot'),
                name='Track',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Initial car positions
        initial_car_x = car_position_data['x'][sampled_indices[0]]
        initial_car_y = car_position_data['y'][sampled_indices[0]]
        
        fig.add_trace(
            go.Scatter(
                x=initial_car_x,
                y=initial_car_y,
                mode='markers+text',
                marker=dict(
                    size=12,
                    #color=driver_colors[:len(initial_car_x)],
                    line=dict(color='white', width=2)
                ),
                text=drivers[:len(initial_car_x)],
                textposition='middle center',
                textfont=dict(color='white', size=8, family='Arial Black'),
                name='Cars',
                showlegend=False
            ),
            row=1, col=2
        )

    initial_status = car_position_data['track_status'][sampled_indices[0]]
            
    # Create status display (as scatter plot with text)
    fig.add_trace(
        go.Scatter(
            x=[0.5], y=[0.9],
            mode='text',
            text=[format_track_status_text(initial_status)],
            textfont=dict(size=10, color='white', family='monospace'),
            showlegend=False,
            name='Status'
        ),
        row=1, col=3
    )

    # CREATE ANIMATION FRAMES
    frames = []
    for i, frame_idx in enumerate(sampled_indices):
        
        # Bar chart data for this frame
        frame_bar_values = [bar_probs[j][i] for j in range(len(drivers))]
        
        frame_data = [
            go.Bar(
                y=drivers,
                x=frame_bar_values,
                orientation='h',
                marker=dict(
                    #color=driver_colors,
                    line=dict(color='white', width=1)
                ),
                text=[f'{val:.1f}%' for val in frame_bar_values],
                textposition='outside',
                showlegend=False
            )
        ]
        
        # Add track data if available
        if car_position_data and frame_idx < len(car_position_data['x']):
            # Track outline (static)
            frame_data.append(
                go.Scatter(
                    x=car_position_data['track_x'],
                    y=car_position_data['track_y'],
                    mode='lines',
                    line=dict(color='white', width=4, dash='dot'),
                    showlegend=False
                )
            )
            
            # Car positions for this frame
            frame_car_x = car_position_data['x'][frame_idx]
            frame_car_y = car_position_data['y'][frame_idx]
            current_lap = car_position_data['lap_numbers'][frame_idx]
            
            frame_data.append(
                go.Scatter(
                    x=frame_car_x,
                    y=frame_car_y,
                    mode='markers+text',
                    marker=dict(
                        size=12,
                        #color=driver_colors[:len(frame_car_x)],
                        line=dict(color='white', width=2)
                    ),
                    text=drivers[:len(frame_car_x)],
                    textposition='middle center',
                    textfont=dict(color='white', size=8, family='Arial Black'),
                    showlegend=False
                )
            )

        # Get current lap number for this frame
        current_lap = car_position_data["lap_numbers"][frame_idx] if car_position_data else 1
        
        frame = go.Frame(
            data=frame_data, 
            name=str(i),
            layout=go.Layout(
                title_text=f'F1 Safety Car Monitor - Lap {current_lap} | Timestep {frame_idx}'
            )
        )
        frames.append(frame)
    
    fig.frames = frames    

    # UPDATE LAYOUT
    fig.update_layout(
        title={
            'text': f'F1 Safety Car Probability & Live Track Position',
            'x': 0.5,
            'font': {'size': 20, 'color': 'white'}
        },
        paper_bgcolor='#1e1e1e',
        plot_bgcolor='#2d2d2d',
        font=dict(color='white', family='Arial'),
        height=650,
        width=1400,
        showlegend=False,
        
        # Animation controls
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [
                {
                    'label': '▶️ Play Race',
                    'method': 'animate',
                    'args': [None, {
                        'frame': {'duration': animation_speed, 'redraw': True},
                        'fromcurrent': True,
                        'transition': {'duration': 50, 'easing': 'linear'}
                    }]
                },
                {
                    'label': '⏸️ Pause', 
                    'method': 'animate',
                    'args': [[None], {
                        'frame': {'duration': 0, 'redraw': False},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                },
                {
                    'label': '⏮️ Restart', 
                    'method': 'animate',
                    'args': [[0], {
                        'frame': {'duration': 0, 'redraw': True},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                }
            ],
            'x': 1.0,
            'y': 1.05,
        }],
        
        # Timeline slider
        sliders=[{
            'steps': [
                {
                    'args': [[str(i)], {
                        'frame': {'duration': 0, 'redraw': True},
                        'mode': 'immediate',
                        'transition': {'duration': 100}
                    }],
                    'label': f'L{car_position_data["lap_numbers"][sampled_indices[i]] if car_position_data else i+1}',
                    'method': 'animate'
                }
                for i in range(len(sampled_indices))
            ],
            'active': 0,
            'currentvalue': {
                'prefix': 'Lap: ',
                'font': {'color': 'white'},
                'suffix': f' / {car_position_data["total_laps"] if car_position_data else "??"}'
            },
            'len': 0.9,
            'x': 0.05,
            'y': -0.05,
            'yanchor': 'top'
        }]
    )
    
    # SUBPLOT STYLING
    # Left subplot (bars)
    fig.update_xaxes(
        title_text='Probability (%)',
        range=[0, 110],
        showgrid=True,
        gridcolor='rgba(255,255,255,0.1)',
        tickfont=dict(color='white'),
        row=1, col=1
    )
    fig.update_yaxes(
        title_text='',
        showgrid=False,
        tickfont=dict(size=12, color='white'),
        row=1, col=1
    )
    
    # Right subplot (track)
    if car_position_data:
        fig.update_xaxes(
            title_text='Track Position',
            range=[-9000, 6500],
            showgrid=False,
            tickfont=dict(color='white'),
            zeroline=False,
            row=1, col=2
        )
        fig.update_yaxes(
            title_text='',
            range=[-10000, 2500],
            showgrid=False,
            tickfont=dict(color='white'),
            zeroline=False,
            row=1, col=2
        )
        
        # Make track subplot properly scaled
        fig.update_layout(
            xaxis2=dict(scaleanchor="y2", scaleratio=1),
        )

        # Third subplot (track status) styling
    
    fig.update_xaxes(
        title_text='',
        showgrid=False,
        showticklabels=False,
        zeroline=False,
        range=[0, 1],
        row=1, col=3
    )
    fig.update_yaxes(
        title_text='',
        showgrid=False,
        showticklabels=False,
        zeroline=False,
        range=[0, 1],
        row=1, col=3
    )
    
    return fig

def format_track_status_text(status_info):
    """
    Format track status information for display
    """
    if not status_info:
        return "No Data"
    
    # Main status with color coding
    track_status = status_info.get('track_status', 'Green')
    status_color = {
        'Green': '🟢',
        'Yellow Flag': '🟡',
        'Safety Car': '🟠',
        'Virtual Safety Car': '🔵',
        'Red Flag': '🔴'
    }.get(track_status, '⚪')
    
    # Weather info
    weather_icon = '🌧️' if status_info.get('rainfall', False) else '☀️'
    
    # Format text
    text_lines = [
        f"🏁 RACE CONTROL",
        f"{status_color} {track_status}",
        f"Lap: {status_info.get('lap', 'N/A')}",
        ""
    ]
    
    # Add specific flag information
    flags_active = []
    if status_info.get('safety_car', False):
        flags_active.append("🚗 Safety Car Deployed")
    if status_info.get('virtual_safety_car', False):
        flags_active.append("🔵 Virtual Safety Car")
    if status_info.get('red_flag', False):
        flags_active.append("🔴 RED FLAG - Session Stopped")
    if status_info.get('yellow_flag', False):
        flags_active.append("🟡 Yellow Flag - Caution")
    
    if flags_active:
        text_lines.extend(flags_active)
        text_lines.append("")
    
    # Session timing info
    if status_info.get('session_time'):
        session_mins = int(status_info['session_time'] // 60)
        session_secs = int(status_info['session_time'] % 60)
        text_lines.append(f"⏱️ Time: {session_mins:02d}:{session_secs:02d}")
        text_lines.append("")
    
    # Weather section
    text_lines.extend([
        f"{weather_icon} WEATHER:",
        f"Track: {status_info.get('track_temp', 'N/A'):.0f}°C" if status_info.get('track_temp') else "Track: N/A",
        f"Air: {status_info.get('air_temp', 'N/A'):.0f}°C" if status_info.get('air_temp') else "Air: N/A",
        f"Humidity: {status_info.get('humidity', 'N/A'):.0f}%" if status_info.get('humidity') else "Humidity: N/A"
    ])
    
    # Add pressure if available
    if status_info.get('pressure'):
        text_lines.append(f"Pressure: {status_info['pressure']:.0f} hPa")
    
    # Wind section
    text_lines.append("")
    text_lines.append("🌪️ WIND:")
    text_lines.append(f"Speed: {status_info.get('wind_speed', 'N/A'):.0f} km/h" if status_info.get('wind_speed') else "Speed: N/A")
    text_lines.append(f"Direction: {status_info.get('wind_direction', 'N/A'):.0f}°" if status_info.get('wind_direction') else "Direction: N/A")
    
    # Rain warning
    if status_info.get('rainfall', False):
        text_lines.extend(["", "⚠️ RAIN DETECTED", "Track conditions may be slippery"])
    
    # Join with line breaks
    return '<br>'.join(text_lines)






app = Flask(__name__)
app.config['CACHE_TYPE'] = 'filesystem'
app.config['CACHE_DIR'] = 'cache-directory'
app.config['CACHE_DEFAULT_TIMEOUT'] = 300

cache = Cache(app)

@cache.cached(timeout=3600, key_prefix='fastf1_session')

def get_session_data():
    print("Loading FastF1 data...")
    # Load FastF1 session
    session = fastf1.get_session(2023, 'São Paulo', 'R')
    session.load()

    # Extract car position data
    car_data = extract_full_race_car_positions(
        session, 
        drivers = ['VER', 'HAM', 'PER', 'ALO', 'SAI', 'RUS', 'PIA', 'STR', 'GAS', 'NOR', 'LEC', 'OCO', 'ALB', 'TSU', 'BOT', 'HUL', 'RIC', 'ZHO', 'MAG', 'DEV', 'SAR'],
        target_timesteps=len(probs_2022)
    )

    return car_data

@app.route('/')
def index():
    print(probs_2022.shape)

    car_data = get_session_data()

    fig = create_smooth_animated_bar_with_track(
        probabilities=probs_2022,
        car_position_data=car_data,
        sample_every=20,      # Adjust for performance
        animation_speed=50,   # Faster = more frames per second
    )
    
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    test = "test var"
    return render_template('index.html', test=test, graphJSON=graphJSON)
