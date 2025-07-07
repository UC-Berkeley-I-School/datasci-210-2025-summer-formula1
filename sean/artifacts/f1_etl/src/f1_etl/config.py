"""Configuration classes for F1 ETL pipeline"""

from dataclasses import dataclass
from typing import List, Optional, Union, Dict

import fastf1

from .logging import logger


@dataclass
class SessionConfig:
    """Configuration for a single F1 session"""

    year: int
    race: str
    session_type: str
    drivers: Optional[List[str]] = None  # NEW: Per-session driver filter


@dataclass
class DataConfig:
    """Configuration for data processing"""

    sessions: List[SessionConfig]
    drivers: Optional[List[str]] = None  # Global driver filter (backwards compatible)
    telemetry_frequency: Union[str, int] = "original"
    include_weather: bool = True
    cache_dir: Optional[str] = None

    def get_effective_drivers(self, session: SessionConfig) -> Optional[List[str]]:
        """
        Get effective driver list for a session.

        Priority:
        1. Session-specific drivers (if set)
        2. Global drivers (if set)
        3. None (all drivers)
        """
        return session.drivers or self.drivers


def create_season_configs(
    year: int,
    session_types: Optional[List[str]] = None,
    include_testing: bool = False,
    exclude_events: Optional[List[str]] = None,
    drivers: Optional[List[str]] = None,  # NEW: Default drivers for all sessions
    drivers_per_session: Optional[
        Dict[str, List[str]]
    ] = None,  # NEW: Per-session drivers
) -> List[SessionConfig]:
    """
    Generate SessionConfig objects for all races in a given season.

    Args:
        year: F1 season year
        session_types: List of session types to include (default: ['R'] for race only)
        include_testing: Whether to include testing sessions
        exclude_events: List of event names to exclude (e.g., ['Saudi Arabian Grand Prix'])
        drivers: Default driver list for all sessions
        drivers_per_session: Dict mapping event names to driver lists
                           e.g., {"Qatar Grand Prix": ["27", "31"], "Monaco Grand Prix": ["1"]}

    Returns:
        List of SessionConfig objects
    """
    if session_types is None:
        session_types = ["R"]  # Default to race only

    if exclude_events is None:
        exclude_events = []

    if drivers_per_session is None:
        drivers_per_session = {}

    # Get the event schedule
    schedule = fastf1.get_event_schedule(year, include_testing=include_testing)

    configs = []

    for _, event in schedule.iterrows():
        event_name = event["EventName"]

        # Skip excluded events
        if event_name in exclude_events:
            logger.info(f"Skipping excluded event: {event_name}")
            continue

        # Get drivers for this specific event
        event_drivers = drivers_per_session.get(event_name, drivers)

        # Generate configs for each requested session type
        for session_type in session_types:
            config = SessionConfig(
                year=year,
                race=event_name,
                session_type=session_type,
                drivers=event_drivers,
            )
            configs.append(config)
            logger.debug(
                f"Created config: {year} {event_name} {session_type} (drivers: {event_drivers})"
            )

    logger.info(f"Generated {len(configs)} SessionConfig objects for {year} season")
    return configs


def create_multi_session_configs(
    year: int,
    session_types: Optional[List[str]] = None,
    include_testing: bool = False,
    exclude_events: Optional[List[str]] = None,
    drivers: Optional[List[str]] = None,  # NEW
    drivers_per_session: Optional[Dict[str, List[str]]] = None,  # NEW
) -> List[SessionConfig]:
    """
    Convenience function to generate configs for multiple session types.

    Common session types:
    - 'FP1', 'FP2', 'FP3': Free Practice sessions
    - 'Q': Qualifying
    - 'R': Race
    - 'S': Sprint (if applicable)

    ``session_types`` defaults to ["FP1", "FP2", "FP3", "Q", "R"]
    """
    session_types = (
        ["FP1", "FP2", "FP3", "Q", "R"] if not session_types else session_types
    )
    return create_season_configs(
        year=year,
        session_types=session_types,
        include_testing=include_testing,
        exclude_events=exclude_events,
        drivers=drivers,
        drivers_per_session=drivers_per_session,
    )
