"""Raw data extraction from FastF1"""

import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import fastf1

from .config import SessionConfig


class RawDataExtractor:
    """Handles extraction of raw fastf1 data"""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(exist_ok=True)

    def extract_session(self, config: SessionConfig) -> Dict[str, Any]:
        """Extract all data for a single session"""
        print(f"Loading session: {config.year} {config.race} {config.session_type}")

        cache_key = f"{config.year}_{config.race}_{config.session_type}".replace(
            " ", "_"
        )
        cache_file = self.cache_dir / f"{cache_key}.pkl" if self.cache_dir else None

        if cache_file and cache_file.exists():
            print(f"Loading from cache: {cache_file}")
            with open(cache_file, "rb") as f:
                return pickle.load(f)

        session = fastf1.get_session(config.year, config.race, config.session_type)
        session.load()

        driver_mapping = {}
        for driver_number in session.drivers:
            driver_info = session.get_driver(driver_number)
            driver_mapping[driver_number] = driver_info["Abbreviation"]

        session_data = {
            "session_info": {
                "year": config.year,
                "race": config.race,
                "session_type": config.session_type,
                "event_name": session.event.EventName,
                "event_date": session.event.EventDate,
                "session_start": session.session_start_time,
                "t0_date": session.t0_date,
            },
            "laps": session.laps,
            "weather": session.weather_data,
            "track_status": session.track_status,
            "car_data": {},
            "pos_data": {},
            "drivers": list(session.drivers),
            "driver_mapping": driver_mapping,
        }

        for driver_number in session_data["drivers"]:
            try:
                session_data["car_data"][driver_number] = session.car_data[
                    driver_number
                ]
                session_data["pos_data"][driver_number] = session.pos_data[
                    driver_number
                ]
            except Exception as e:
                abbreviation = driver_mapping.get(driver_number, driver_number)
                print(
                    f"Warning: Could not extract telemetry for driver ({abbreviation}, {driver_number}): {e}"
                )

        if cache_file:
            with open(cache_file, "wb") as f:
                pickle.dump(session_data, f)

        return session_data
