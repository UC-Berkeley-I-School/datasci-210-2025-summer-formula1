"""Configuration classes for F1 ETL pipeline"""

from dataclasses import dataclass
from typing import List, Optional, Union


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
