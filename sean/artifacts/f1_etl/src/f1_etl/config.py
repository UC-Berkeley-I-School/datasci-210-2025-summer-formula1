"""Configuration classes for F1 ETL pipeline"""

from dataclasses import dataclass
from typing import List, Optional, Union

import fastf1

from .logging import logger


@dataclass
class SessionConfig:
    """Configuration for a single F1 session"""

    year: int
    race: str
    session_type: str


@dataclass
class DataConfig:
    """Configuration for data processing"""

    sessions: List[SessionConfig]
    drivers: Optional[List[str]] = None
    telemetry_frequency: Union[str, int] = "original"
    include_weather: bool = True
    cache_dir: Optional[str] = None


def create_season_configs(
    year: int,
    session_types: Optional[List[str]] = None,
    include_testing: bool = False,
    exclude_events: Optional[List[str]] = None,
) -> List[SessionConfig]:
    """
    Generate SessionConfig objects for all races in a given season.

    Args:
        year: F1 season year
        session_types: List of session types to include (default: ['R'] for race only)
        include_testing: Whether to include testing sessions
        exclude_events: List of event names to exclude (e.g., ['Saudi Arabian Grand Prix'])

    Returns:
        List of SessionConfig objects
    """
    if session_types is None:
        session_types = ["R"]  # Default to race only

    if exclude_events is None:
        exclude_events = []

    # Get the event schedule
    schedule = fastf1.get_event_schedule(year, include_testing=include_testing)

    configs = []

    for _, event in schedule.iterrows():
        event_name = event["EventName"]

        # Skip excluded events
        if event_name in exclude_events:
            logger.info(f"Skipping excluded event: {event_name}")
            continue

        # Generate configs for each requested session type
        for session_type in session_types:
            config = SessionConfig(
                year=year, race=event_name, session_type=session_type
            )
            configs.append(config)
            logger.debug(f"Created config: {year} {event_name} {session_type}")

    logger.info(f"Generated {len(configs)} SessionConfig objects for {year} season")
    return configs


def create_multi_session_configs(
    year: int,
    session_types: Optional[List[str]] = None,
    include_testing: bool = False,
    exclude_events: Optional[List[str]] = None,
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
    )
