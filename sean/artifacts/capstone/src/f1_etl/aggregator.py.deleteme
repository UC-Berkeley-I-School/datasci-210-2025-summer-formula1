import fastf1
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass
import logging
from tqdm import tqdm
import time

from .preprocessor import F1DatasetPreprocessor, PreprocessorConfig


@dataclass
class AggregatorConfig:
    """Configuration for the F1 dataset aggregator"""
    
    max_workers: int = 4  # Parallel processing threads
    session_types: List[str] = None  # ['R', 'Q', 'FP1', etc.]
    skip_testing: bool = True
    retry_attempts: int = 2
    cache_sessions: bool = True
    timeout_seconds: int = 300  # 5 minute timeout per session
    
    def __post_init__(self):
        if self.session_types is None:
            self.session_types = ['R']  # Race only by default


class SessionProcessor:
    """Handles processing of individual F1 sessions"""
    
    def __init__(self, preprocessor: F1DatasetPreprocessor, config: AggregatorConfig):
        self.preprocessor = preprocessor
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def process_session(self, year: int, event_name: str, session_type: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Process a single session and return dataset with metadata"""
        try:
            # Load session
            session = fastf1.get_session(year, event_name, session_type)
            session.load()
            
            # Process with preprocessor
            dataset = self.preprocessor.process_dataset(session)
            
            if dataset.empty:
                raise ValueError("Preprocessor returned empty dataset")
            
            # Add event metadata
            dataset = self._add_event_metadata(dataset, session, year)
            
            # Create metadata summary
            metadata = self._extract_metadata(dataset, session, year, event_name, session_type)
            
            return dataset, metadata
            
        except Exception as e:
            self.logger.error(f"Failed to process {year} {event_name} {session_type}: {e}")
            raise
    
    def _add_event_metadata(self, dataset: pd.DataFrame, session: fastf1.core.Session, year: int) -> pd.DataFrame:
        """Add event information columns to dataset"""
        session_info = session.session_info
        
        # Generate race ID
        start_date = session_info.get('StartDate')
        date_str = start_date.strftime('%Y%m%d') if start_date else f"{year}0101"
        meeting_name = session_info.get('Meeting', {}).get('Name', 'Unknown')
        race_id = f"{year}_R{session_info.get('RoundNumber', 0):02d}_{meeting_name.replace(' ', '_')}"
        
        # Add metadata columns
        dataset['RaceID'] = race_id
        dataset['Year'] = year
        dataset['RoundNumber'] = session_info.get('RoundNumber', 0)
        dataset['SessionName'] = meeting_name
        dataset['Country'] = session_info.get('Meeting', {}).get('Country', {}).get('Name', 'Unknown')
        dataset['Location'] = session_info.get('Meeting', {}).get('Location', 'Unknown')
        dataset['SessionType'] = session_info.get('Type', 'Unknown')
        dataset['StartDate'] = start_date
        
        return dataset
    
    def _extract_metadata(self, dataset: pd.DataFrame, session: fastf1.core.Session, 
                         year: int, event_name: str, session_type: str) -> Dict[str, Any]:
        """Extract metadata for tracking and analysis"""
        return {
            'RaceID': dataset['RaceID'].iloc[0],
            'Year': year,
            'RoundNumber': dataset['RoundNumber'].iloc[0],
            'EventName': event_name,
            'SessionName': dataset['SessionName'].iloc[0],
            'Country': dataset['Country'].iloc[0],
            'Location': dataset['Location'].iloc[0],
            'SessionType': session_type,
            'StartDate': dataset['StartDate'].iloc[0],
            'DataPoints': len(dataset),
            'UniqueDrivers': dataset['Driver'].nunique(),
            'SessionDuration': dataset['SessionTimeSeconds'].max() - dataset['SessionTimeSeconds'].min(),
            'ProcessedAt': pd.Timestamp.now()
        }


class F1SeasonAggregator:
    """
    Efficiently aggregates F1 datasets across multiple races and seasons
    with parallel processing and robust error handling.
    """
    
    def __init__(self, preprocessor_config: PreprocessorConfig = None, 
                 aggregator_config: AggregatorConfig = None):
        self.preprocessor_config = preprocessor_config or PreprocessorConfig()
        self.aggregator_config = aggregator_config or AggregatorConfig()
        
        self.preprocessor = F1DatasetPreprocessor(self.preprocessor_config)
        self.processor = SessionProcessor(self.preprocessor, self.aggregator_config)
        
        self.datasets: List[pd.DataFrame] = []
        self.metadata: List[Dict[str, Any]] = []
        self.failed_sessions: List[Dict[str, Any]] = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def add_session(self, year: int, event_name: str, session_type: str, 
                   race_identifier: Optional[str] = None) -> None:
        """Add a single session to the aggregator"""
        try:
            dataset, metadata = self.processor.process_session(year, event_name, session_type)
            
            if race_identifier:
                dataset['RaceID'] = race_identifier
                metadata['RaceID'] = race_identifier
            
            self.datasets.append(dataset)
            self.metadata.append(metadata)
            
            self.logger.info(f"Added: {metadata['SessionName']} ({metadata['DataPoints']} points)")
            
        except Exception as e:
            self._record_failure(year, event_name, session_type, str(e))
    
    def add_season(self, year: int) -> Dict[str, Any]:
        """
        Add all races from a season using parallel processing
        
        Returns:
            Dict with processing summary
        """
        try:
            schedule = fastf1.get_event_schedule(year)
            if self.aggregator_config.skip_testing:
                schedule = schedule[schedule['EventFormat'] != 'testing']
        except Exception as e:
            self.logger.error(f"Failed to get schedule for {year}: {e}")
            return self._create_summary(year, 0, 0, 0)
        
        # Create list of sessions to process
        sessions_to_process = []
        for _, event in schedule.iterrows():
            for session_type in self.aggregator_config.session_types:
                sessions_to_process.append({
                    'year': year,
                    'event_name': event['EventName'],
                    'session_type': session_type,
                    'round_number': event['RoundNumber'],
                    'country': event['Country']
                })
        
        self.logger.info(f"Processing {len(sessions_to_process)} sessions from {year} season...")
        
        # Process sessions in parallel
        successful_count = 0
        failed_count = 0
        
        with ThreadPoolExecutor(max_workers=self.aggregator_config.max_workers) as executor:
            # Submit all tasks
            future_to_session = {
                executor.submit(self._process_session_with_timeout, session): session
                for session in sessions_to_process
            }
            
            # Process completed tasks with progress bar
            with tqdm(total=len(sessions_to_process), desc=f"Processing {year} season") as pbar:
                for future in as_completed(future_to_session):
                    session = future_to_session[future]
                    
                    try:
                        dataset, metadata = future.result()
                        self.datasets.append(dataset)
                        self.metadata.append(metadata)
                        successful_count += 1
                        
                    except Exception as e:
                        self._record_failure(
                            session['year'], session['event_name'], 
                            session['session_type'], str(e)
                        )
                        failed_count += 1
                    
                    pbar.update(1)
        
        return self._create_summary(year, len(sessions_to_process), successful_count, failed_count)
    
    def _process_session_with_timeout(self, session_info: Dict) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Process session with timeout handling"""
        return self.processor.process_session(
            session_info['year'], 
            session_info['event_name'], 
            session_info['session_type']
        )
    
    def add_multiple_seasons(self, years: List[int]) -> Dict[str, Dict[str, Any]]:
        """Add multiple seasons efficiently"""
        results = {}
        
        for year in years:
            self.logger.info(f"\n=== Starting Season {year} ===")
            start_time = time.time()
            
            summary = self.add_season(year)
            summary['processing_time'] = time.time() - start_time
            results[year] = summary
            
            self.logger.info(f"Season {year} complete in {summary['processing_time']:.1f}s")
        
        return results
    
    def retry_failed_sessions(self) -> int:
        """Retry all failed sessions"""
        if not self.failed_sessions:
            self.logger.info("No failed sessions to retry")
            return 0
        
        retry_count = 0
        remaining_failures = []
        
        self.logger.info(f"Retrying {len(self.failed_sessions)} failed sessions...")
        
        for failure in self.failed_sessions:
            try:
                dataset, metadata = self.processor.process_session(
                    failure['year'], failure['event_name'], failure['session_type']
                )
                
                self.datasets.append(dataset)
                self.metadata.append(metadata)
                retry_count += 1
                
                self.logger.info(f"Retry successful: {failure['event_name']}")
                
            except Exception as e:
                remaining_failures.append(failure)
                self.logger.warning(f"Retry failed: {failure['event_name']} - {e}")
        
        self.failed_sessions = remaining_failures
        self.logger.info(f"Retry complete: {retry_count} recovered, {len(remaining_failures)} still failed")
        
        return retry_count
    
    def get_combined_dataset(self) -> pd.DataFrame:
        """Combine all datasets into single DataFrame"""
        if not self.datasets:
            raise ValueError("No datasets available. Add sessions first.")
        
        combined = pd.concat(self.datasets, ignore_index=True)
        
        self.logger.info(f"Combined dataset: {len(self.datasets)} races, "
                        f"{len(combined)} data points, "
                        f"{combined['Country'].nunique()} countries")
        
        return combined
    
    def get_metadata_summary(self) -> pd.DataFrame:
        """Get DataFrame with metadata for all processed sessions"""
        return pd.DataFrame(self.metadata)
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics"""
        if not self.metadata:
            return {}
        
        meta_df = pd.DataFrame(self.metadata)
        
        return {
            'total_sessions': len(self.metadata),
            'total_data_points': meta_df['DataPoints'].sum(),
            'unique_countries': meta_df['Country'].nunique(),
            'unique_years': meta_df['Year'].nunique(),
            'session_types': meta_df['SessionType'].unique().tolist(),
            'failed_sessions': len(self.failed_sessions),
            'success_rate': len(self.metadata) / (len(self.metadata) + len(self.failed_sessions)) * 100,
            'avg_data_points_per_session': meta_df['DataPoints'].mean(),
            'processing_date_range': [meta_df['StartDate'].min(), meta_df['StartDate'].max()]
        }
    
    def filter_by_country(self, country: str) -> pd.DataFrame:
        """Get combined dataset filtered by country"""
        combined = self.get_combined_dataset()
        return combined[combined['Country'] == country].copy()
    
    def filter_by_year(self, year: int) -> pd.DataFrame:
        """Get combined dataset filtered by year"""
        combined = self.get_combined_dataset()
        return combined[combined['Year'] == year].copy()
    
    def filter_by_session_type(self, session_type: str) -> pd.DataFrame:
        """Get combined dataset filtered by session type"""
        combined = self.get_combined_dataset()
        return combined[combined['SessionType'] == session_type].copy()
    
    def analyze_safety_car_patterns(self) -> pd.DataFrame:
        """Analyze safety car deployment patterns across all races"""
        combined = self.get_combined_dataset()
        
        # Group by race and analyze safety car usage
        race_analysis = []
        
        for race_id in combined['RaceID'].unique():
            race_data = combined[combined['RaceID'] == race_id]
            safety_car_data = race_data[race_data['SafetyCar'] == 1]
            
            analysis = {
                'RaceID': race_id,
                'SessionName': race_data['SessionName'].iloc[0],
                'Country': race_data['Country'].iloc[0],
                'Year': race_data['Year'].iloc[0],
                'TotalDataPoints': len(race_data),
                'SafetyCarPoints': len(safety_car_data),
                'SafetyCarPercentage': len(safety_car_data) / len(race_data) * 100 if len(race_data) > 0 else 0,
                'UniqueDrivers': race_data['Driver'].nunique(),
                'SessionDuration': race_data['SessionTimeSeconds'].max() - race_data['SessionTimeSeconds'].min()
            }
            race_analysis.append(analysis)
        
        return pd.DataFrame(race_analysis).sort_values('SafetyCarPercentage', ascending=False)
    
    def clear_cache(self):
        """Clear all cached data and preprocessor cache"""
        self.datasets.clear()
        self.metadata.clear()
        self.failed_sessions.clear()
        self.preprocessor.clear_cache()
        self.logger.info("All caches cleared")
    
    def _record_failure(self, year: int, event_name: str, session_type: str, error: str):
        """Record a failed session for later retry"""
        failure = {
            'year': year,
            'event_name': event_name,
            'session_type': session_type,
            'error': error,
            'failed_at': pd.Timestamp.now()
        }
        self.failed_sessions.append(failure)
        self.logger.error(f"Session failed: {event_name} {session_type} - {error}")
    
    def _create_summary(self, year: int, total: int, successful: int, failed: int) -> Dict[str, Any]:
        """Create processing summary"""
        return {
            'year': year,
            'total_attempted': total,
            'successful': successful,
            'failed': failed,
            'success_rate': successful / total * 100 if total > 0 else 0,
            'failed_sessions': [f['event_name'] for f in self.failed_sessions if f['year'] == year]
        }


# Convenience functions
def create_aggregator(preprocessor_config: PreprocessorConfig = None,
                     max_workers: int = 4,
                     session_types: List[str] = None) -> F1SeasonAggregator:
    """Create a configured F1SeasonAggregator instance"""

    aggregator_config = AggregatorConfig(
        max_workers=max_workers,
        session_types=session_types or ['R']
    )

    return F1SeasonAggregator(preprocessor_config, aggregator_config)


def test():
    aggregator = create_aggregator(preprocessor_config=None, max_workers=4)

    # Add season
    summary = aggregator.add_season(2024)

    # Retry failures once
    if aggregator.failed_sessions:
        aggregator.retry_failed_sessions()

    print(f"\nProcessing Summary:")
    print(f"Success rate: {summary['success_rate']:.1f}%")
    print(f"Total data points: {sum(m['DataPoints'] for m in aggregator.metadata)}")

    combined_dataset = aggregator.get_combined_dataset()
    metadata_summary = aggregator.get_metadata_summary()


if __name__ == "__main__":
    test()