"""
Safety Car Prediction - Data Split Experiments

This module contains experiments for proper train/test splitting and validation
to avoid data leakage in the temporal safety car prediction system.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Import from the main POC module
from poc_3 import (
    TemporalSafetyCarSystem, 
    SafetyCarEvent,
    TemporalPrediction,
    analyze_session_temporal,
    load_training_dataset
)

class DataSplitExperiments:
    """
    Experimental class for testing different data splitting strategies
    to avoid data leakage in temporal safety car prediction.
    """
    
    def __init__(self, system: TemporalSafetyCarSystem):
        """
        Initialize with a temporal safety car system.
        
        Args:
            system: Trained or untrained TemporalSafetyCarSystem
        """
        self.system = system
        self.session_data = None
        self.all_features = None
        self.events = None
    
    def load_session_data(self, session):
        """
        Load and process session data for experimentation.
        
        Args:
            session: FastF1 Session object
        """
        print("Loading and processing session data...")
        
        # Find safety car events
        self.events = self.system.data_processor.find_safety_car_events(session)
        print(f"Found {len(self.events)} safety car events")
        
        # Create temporal windows
        windows = self.system.data_processor.create_temporal_windows(session, self.events)
        print(f"Created {len(windows)} temporal windows")
        
        # Extract features
        temporal_features = []
        for window in windows:
            features = self.system.feature_engineer.extract_features(window)
            if not features.empty:
                temporal_features.append(features)
        
        # Combine all features
        self.all_features = pd.concat(temporal_features, ignore_index=True)
        self.all_features = self.all_features.sort_values('prediction_time').reset_index(drop=True)
        
        print(f"Total feature samples: {len(self.all_features)}")
        print(f"Safety car events in features: {self.all_features['sc_in_prediction_window'].sum()}")
        
        self.session_data = {
            'session': session,
            'windows': windows,
            'temporal_features': temporal_features
        }
    
    def analyze_event_distribution(self):
        """
        Analyze the temporal distribution of safety car events.
        """
        if self.all_features is None:
            print("No data loaded. Call load_session_data() first.")
            return
        
        print("\n" + "="*60)
        print("📊 SAFETY CAR EVENT DISTRIBUTION ANALYSIS")
        print("="*60)
        
        sc_events = self.all_features[self.all_features['sc_in_prediction_window'] == True]
        total_samples = len(self.all_features)
        
        # Basic statistics
        print(f"\nBasic Statistics:")
        print(f"  Total samples: {total_samples}")
        print(f"  SC event samples: {len(sc_events)}")
        print(f"  Normal samples: {total_samples - len(sc_events)}")
        print(f"  Class imbalance ratio: {len(sc_events)/total_samples:.1%} positive class")
        
        if len(sc_events) == 0:
            print("❌ No safety car events found in features!")
            return
        
        # Temporal distribution
        session_start = self.all_features['prediction_time'].min()
        session_end = self.all_features['prediction_time'].max()
        session_duration = session_end - session_start
        
        print(f"\nTemporal Distribution:")
        print(f"  Session start: {self._format_time(session_start)}")
        print(f"  Session end: {self._format_time(session_end)}")
        print(f"  Session duration: {session_duration}")
        
        # Show where each SC event occurs
        sc_times = sc_events['prediction_time'].sort_values().unique()
        print(f"\nSafety Car Events Timeline:")
        for i, sc_time in enumerate(sc_times, 1):
            position_pct = ((sc_time - session_start) / session_duration) * 100
            print(f"  Event {i}: {self._format_time(sc_time)} ({position_pct:.1f}% through session)")
        
        # Driver distribution
        print(f"\nDriver Distribution of SC Events:")
        driver_sc_counts = sc_events['driver'].value_counts()
        for driver, count in driver_sc_counts.head(10).items():
            print(f"  {driver}: {count} predictions")
        
        return {
            'total_samples': total_samples,
            'sc_samples': len(sc_events),
            'sc_times': sc_times,
            'session_start': session_start,
            'session_end': session_end,
            'driver_distribution': driver_sc_counts
        }
    
    def visualize_event_distribution(self, figsize=(15, 10)):
        """
        Create visualizations of the event distribution.
        """
        if self.all_features is None:
            print("No data loaded. Call load_session_data() first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Safety Car Event Distribution Analysis', fontsize=16)
        
        # 1. Timeline of all predictions
        ax1 = axes[0, 0]
        self.all_features['minutes_into_session'] = (
            self.all_features['prediction_time'] - self.all_features['prediction_time'].min()
        ).dt.total_seconds() / 60
        
        # Plot all predictions
        normal_data = self.all_features[~self.all_features['sc_in_prediction_window']]
        sc_data = self.all_features[self.all_features['sc_in_prediction_window']]
        
        ax1.scatter(normal_data['minutes_into_session'], 
                   normal_data.index, 
                   alpha=0.3, s=1, label='Normal', color='blue')
        ax1.scatter(sc_data['minutes_into_session'], 
                   sc_data.index, 
                   alpha=0.8, s=20, label='Safety Car', color='red')
        ax1.set_xlabel('Minutes into Session')
        ax1.set_ylabel('Sample Index')
        ax1.set_title('Temporal Distribution of Predictions')
        ax1.legend()
        
        # 2. Class distribution pie chart
        ax2 = axes[0, 1]
        class_counts = self.all_features['sc_in_prediction_window'].value_counts()
        ax2.pie(class_counts.values, 
                labels=['Normal', 'Safety Car'], 
                autopct='%1.1f%%',
                colors=['lightblue', 'red'])
        ax2.set_title('Class Distribution')
        
        # 3. SC events by driver
        ax3 = axes[1, 0]
        sc_by_driver = self.all_features[self.all_features['sc_in_prediction_window']]['driver'].value_counts().head(10)
        if len(sc_by_driver) > 0:
            sc_by_driver.plot(kind='bar', ax=ax3)
            ax3.set_title('SC Event Predictions by Driver (Top 10)')
            ax3.set_xlabel('Driver')
            ax3.set_ylabel('Number of SC Predictions')
            ax3.tick_params(axis='x', rotation=45)
        else:
            ax3.text(0.5, 0.5, 'No SC events found', 
                    transform=ax3.transAxes, ha='center', va='center')
            ax3.set_title('SC Event Predictions by Driver')
        
        # 4. Session timeline with SC events
        ax4 = axes[1, 1]
        time_bins = pd.cut(self.all_features['minutes_into_session'], bins=20)
        sc_by_time = self.all_features.groupby(time_bins)['sc_in_prediction_window'].agg(['sum', 'count'])
        sc_by_time['sc_rate'] = sc_by_time['sum'] / sc_by_time['count']
        
        bin_centers = [interval.mid for interval in sc_by_time.index]
        ax4.bar(bin_centers, sc_by_time['sc_rate'], alpha=0.7, color='orange')
        ax4.set_xlabel('Minutes into Session')
        ax4.set_ylabel('SC Event Rate')
        ax4.set_title('SC Event Rate Throughout Session')
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def experiment_split_strategies(self) -> Dict:
        """
        Test multiple splitting strategies and compare their effectiveness.
        """
        if self.all_features is None:
            print("No data loaded. Call load_session_data() first.")
            return {}
        
        print("\n" + "="*60)
        print("🧪 TESTING DIFFERENT SPLIT STRATEGIES")
        print("="*60)
        
        results = {}
        
        # Strategy 1: Simple temporal split (70/30)
        print("\n1️⃣ Simple Temporal Split (70/30)")
        results['simple_temporal'] = self._test_simple_temporal_split()
        
        # Strategy 2: Event-aware temporal split
        print("\n2️⃣ Event-Aware Temporal Split")
        results['event_aware'] = self._test_event_aware_split()
        
        # Strategy 3: Stratified temporal split
        print("\n3️⃣ Stratified Temporal Split")
        results['stratified'] = self._test_stratified_split()
        
        # Strategy 4: Time series cross-validation
        print("\n4️⃣ Time Series Cross-Validation")
        results['time_series_cv'] = self._test_time_series_cv()
        
        # Strategy 5: Block-based split
        print("\n5️⃣ Block-Based Split")
        results['block_based'] = self._test_block_based_split()
        
        # Compare results
        print("\n" + "="*60)
        print("📊 STRATEGY COMPARISON")
        print("="*60)
        self._compare_strategies(results)
        
        return results
    
    def _test_simple_temporal_split(self, split_ratio=0.7):
        """Test simple temporal split."""
        try:
            split_idx = int(len(self.all_features) * split_ratio)
            train_data = self.all_features.iloc[:split_idx]
            test_data = self.all_features.iloc[split_idx:]
            
            return self._evaluate_split(train_data, test_data, "Simple Temporal")
        except Exception as e:
            return {'error': str(e)}
    
    def _test_event_aware_split(self):
        """Test event-aware temporal split."""
        try:
            sc_events = self.all_features[self.all_features['sc_in_prediction_window'] == True]
            
            if len(sc_events) < 2:
                return {'error': 'Insufficient SC events for split'}
            
            # Find unique SC times and split them
            sc_times = sc_events['prediction_time'].sort_values().unique()
            split_event_idx = len(sc_times) // 2
            if split_event_idx == 0:
                split_event_idx = 1
            
            split_time = sc_times[split_event_idx]
            
            train_data = self.all_features[self.all_features['prediction_time'] < split_time]
            test_data = self.all_features[self.all_features['prediction_time'] >= split_time]
            
            return self._evaluate_split(train_data, test_data, "Event-Aware")
        except Exception as e:
            return {'error': str(e)}
    
    def _test_stratified_split(self):
        """Test stratified temporal split."""
        try:
            # Create time blocks
            n_blocks = 5
            self.all_features['time_block'] = pd.cut(
                range(len(self.all_features)), 
                bins=n_blocks, 
                labels=False
            )
            
            # Find blocks with SC events
            sc_blocks = self.all_features[
                self.all_features['sc_in_prediction_window'] == True
            ]['time_block'].unique()
            
            if len(sc_blocks) < 2:
                return {'error': 'SC events only in one time block'}
            
            # Split blocks
            train_blocks = sc_blocks[:len(sc_blocks)//2 + 1]
            test_blocks = sc_blocks[len(sc_blocks)//2:]
            
            if len(test_blocks) == 0:
                test_blocks = [train_blocks[-1]]
                train_blocks = train_blocks[:-1]
            
            train_data = self.all_features[self.all_features['time_block'].isin(train_blocks)]
            test_data = self.all_features[self.all_features['time_block'].isin(test_blocks)]
            
            return self._evaluate_split(train_data, test_data, "Stratified")
        except Exception as e:
            return {'error': str(e)}
    
    def _test_time_series_cv(self, n_splits=3):
        """Test time series cross-validation."""
        try:
            # Prepare features for sklearn
            feature_cols = [col for col in self.all_features.columns 
                          if col not in ['driver', 'prediction_time', 'window_start', 
                                       'window_end', 'sc_in_prediction_window']]
            X = self.all_features[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
            y = self.all_features['sc_in_prediction_window'].astype(int)
            
            tscv = TimeSeriesSplit(n_splits=n_splits)
            cv_scores = []
            
            for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
                train_data = self.all_features.iloc[train_idx]
                test_data = self.all_features.iloc[test_idx]
                
                fold_result = self._evaluate_split(train_data, test_data, f"TS-CV Fold {fold+1}")
                if 'accuracy' in fold_result:
                    cv_scores.append(fold_result['accuracy'])
            
            return {
                'cv_scores': cv_scores,
                'mean_accuracy': np.mean(cv_scores) if cv_scores else 0,
                'std_accuracy': np.std(cv_scores) if cv_scores else 0,
                'n_folds': len(cv_scores)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _test_block_based_split(self, n_blocks=4):
        """Test block-based split."""
        try:
            # Create time blocks
            block_size = len(self.all_features) // n_blocks
            train_blocks = list(range(0, n_blocks-1))  # First n-1 blocks for training
            test_block = n_blocks - 1  # Last block for testing
            
            train_indices = []
            for block in train_blocks:
                start_idx = block * block_size
                end_idx = (block + 1) * block_size
                train_indices.extend(range(start_idx, end_idx))
            
            test_start = test_block * block_size
            test_indices = list(range(test_start, len(self.all_features)))
            
            train_data = self.all_features.iloc[train_indices]
            test_data = self.all_features.iloc[test_indices]
            
            return self._evaluate_split(train_data, test_data, "Block-Based")
        except Exception as e:
            return {'error': str(e)}
    
    def _evaluate_split(self, train_data: pd.DataFrame, test_data: pd.DataFrame, strategy_name: str) -> Dict:
        """
        Evaluate a train/test split.
        """
        # Check for empty splits
        if len(train_data) == 0 or len(test_data) == 0:
            return {'error': 'Empty train or test set'}
        
        # Check for SC events in both splits
        train_sc = train_data['sc_in_prediction_window'].sum()
        test_sc = test_data['sc_in_prediction_window'].sum()
        
        print(f"  📊 {strategy_name}:")
        print(f"    Train: {len(train_data)} samples, {train_sc} SC events")
        print(f"    Test:  {len(test_data)} samples, {test_sc} SC events")
        
        if train_sc == 0:
            print(f"    ❌ No SC events in training set")
            return {'error': 'No SC events in training set'}
        
        if test_sc == 0:
            print(f"    ❌ No SC events in test set")
            return {'error': 'No SC events in test set'}
        
        try:
            # Train model
            model = self.system.model_trainer.train([train_data])
            
            # Make predictions on test set
            predictions = model.predict(test_data, self.events)
            
            # Calculate metrics
            y_true = test_data['sc_in_prediction_window'].values
            y_pred = [p.prediction for p in predictions]
            
            # Basic metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            # Confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            print(f"    ✅ Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
            print(f"    📊 TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'confusion_matrix': {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn},
                'train_samples': len(train_data),
                'test_samples': len(test_data),
                'train_sc_events': train_sc,
                'test_sc_events': test_sc
            }
            
        except Exception as e:
            print(f"    ❌ Error during evaluation: {str(e)}")
            return {'error': str(e)}
    
    def _compare_strategies(self, results: Dict):
        """
        Compare the results of different splitting strategies.
        """
        comparison_data = []
        
        for strategy, result in results.items():
            if 'error' not in result:
                if strategy == 'time_series_cv':
                    comparison_data.append({
                        'Strategy': strategy.replace('_', ' ').title(),
                        'Accuracy': result.get('mean_accuracy', 0),
                        'Precision': '-',
                        'Recall': '-',
                        'F1': '-',
                        'Notes': f"CV: {result.get('mean_accuracy', 0):.3f} ± {result.get('std_accuracy', 0):.3f}"
                    })
                else:
                    comparison_data.append({
                        'Strategy': strategy.replace('_', ' ').title(),
                        'Accuracy': result.get('accuracy', 0),
                        'Precision': result.get('precision', 0),
                        'Recall': result.get('recall', 0),
                        'F1': result.get('f1', 0),
                        'Notes': f"Train: {result.get('train_samples', 0)}, Test: {result.get('test_samples', 0)}"
                    })
            else:
                comparison_data.append({
                    'Strategy': strategy.replace('_', ' ').title(),
                    'Accuracy': 'Error',
                    'Precision': 'Error', 
                    'Recall': 'Error',
                    'F1': 'Error',
                    'Notes': result['error']
                })
        
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            print(df.to_string(index=False))
            
            # Find best strategy
            valid_results = [r for r in comparison_data if r['Accuracy'] != 'Error']
            if valid_results:
                best_strategy = max(valid_results, key=lambda x: x['F1'] if isinstance(x['F1'], (int, float)) else 0)
                print(f"\n🏆 Best Strategy: {best_strategy['Strategy']} (F1: {best_strategy['F1']:.3f})")
    
    def _format_time(self, time_value):
        """Format time value for display."""
        if hasattr(time_value, 'total_seconds'):
            total_seconds = int(time_value.total_seconds())
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            seconds = total_seconds % 60
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        elif hasattr(time_value, 'strftime'):
            return time_value.strftime('%H:%M:%S')
        else:
            return str(time_value)

def run_split_experiments(session):
    """
    Convenience function to run all split experiments on a session.
    
    Args:
        session: FastF1 Session object
    """
    # Create system and experiments
    system = TemporalSafetyCarSystem()
    experiments = DataSplitExperiments(system)
    
    # Load data
    experiments.load_session_data(session)
    
    # Analyze distribution
    experiments.analyze_event_distribution()
    
    # Visualize (if matplotlib available)
    try:
        experiments.visualize_event_distribution()
    except Exception as e:
        print(f"Visualization failed: {e}")
    
    # Test splitting strategies
    results = experiments.experiment_split_strategies()
    
    return experiments, results

def test_with_sample_session():
    """Test with a sample F1 session."""
    try:
        import fastf1 as f1
        
        print("Loading F1 session...")
        session = f1.get_session(2024, 'Saudi Arabian Grand Prix', 'R')
        session.load()
        
        print("Running split experiments...")
        experiments, results = run_split_experiments(session)
        
        return experiments, results
        
    except ImportError:
        print("FastF1 not available. Please load session data manually.")
        return None, None
    except Exception as e:
        print(f"Error loading session: {e}")
        return None, None

if __name__ == "__main__":
    test_with_sample_session()