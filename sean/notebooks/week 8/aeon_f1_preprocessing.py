import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

import fastf1

from capstone import PreprocessorConfig, BaseFeatures, F1DatasetPreprocessor

from aeon.transformations.collection import (
    Normalizer, Centerer, MinMaxScaler as AeonMinMaxScaler,
    Padder, Truncator, Resizer, SimpleImputer
)
from aeon.classification.convolution_based import RocketClassifier
from aeon.pipeline import make_pipeline
from aeon.utils.validation import has_missing, is_equal_length, is_univariate

class F1AeonPreprocessor:
    """
    Preprocessor to transform F1 tabular time series data into aeon-compatible format
    and apply aeon preprocessing techniques.
    """
    
    def __init__(self, 
                 sequence_length=None,
                 scaling_method='normalize',  # 'normalize', 'center', 'minmax'
                 handle_unequal_length='pad',  # 'pad', 'truncate', 'resize'
                 target_length=None,
                 imputation_strategy='mean'):
        """
        Initialize the F1 Aeon Preprocessor
        
        Parameters:
        -----------
        sequence_length : int, optional
            Length of sequences to create. If None, uses natural lap boundaries
        scaling_method : str
            Method for scaling time series ('normalize', 'center', 'minmax')
        handle_unequal_length : str
            How to handle unequal length series ('pad', 'truncate', 'resize')
        target_length : int, optional
            Target length for resize operation
        imputation_strategy : str
            Strategy for handling missing values ('mean', 'median', 'constant')
        """
        self.sequence_length = sequence_length
        self.scaling_method = scaling_method
        self.handle_unequal_length = handle_unequal_length
        self.target_length = target_length
        self.imputation_strategy = imputation_strategy
        
        # Features to use as time series channels
        self.time_series_features = [
            'Speed', 'RPM', 'Throttle', 'TyreLife', 'AirTemp', 'TrackTemp', 'Humidity'
        ]
        
        # Categorical features that might be used for stratification
        self.categorical_features = ['Driver', 'Compound', 'TrackStatus']
        
        # Initialize transformers
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.fitted = False
        
    def _prepare_categorical_features(self, df):
        """Encode categorical features for use as targets or stratification"""
        categorical_data = {}
        
        for feature in self.categorical_features:
            if feature in df.columns:
                if feature not in self.label_encoders:
                    self.label_encoders[feature] = LabelEncoder()
                    categorical_data[feature] = self.label_encoders[feature].fit_transform(df[feature])
                else:
                    categorical_data[feature] = self.label_encoders[feature].transform(df[feature])
        
        return categorical_data
    
    def _create_sequences_by_driver_lap(self, df):
        """
        Create sequences grouped by driver and lap number.
        Each sequence represents one driver's performance during one lap.
        """
        sequences = []
        labels = []
        metadata = []
        
        # Group by driver and lap
        for (driver, lap), group in df.groupby(['Driver', 'LapNumber']):
            if len(group) < 5:  # Skip very short sequences
                continue
                
            # Sort by session time to ensure temporal order
            group = group.sort_values('SessionTimeSeconds')
            
            # Extract time series features
            sequence_data = group[self.time_series_features].values.T  # Shape: (n_channels, n_timepoints)
            
            # Handle missing values
            if np.isnan(sequence_data).any():
                # Simple forward fill for missing values
                sequence_data = pd.DataFrame(sequence_data.T).fillna(method='ffill').fillna(method='bfill').values.T
            
            sequences.append(sequence_data)
            
            # Use driver as label (could also use compound, track status, etc.)
            labels.append(driver)
            
            # Store metadata
            metadata.append({
                'driver': driver,
                'lap': lap,
                'compound': group['Compound'].iloc[0],
                'track_status': group['TrackStatus'].iloc[0],
                'session_name': group['session_name'].iloc[0],
                'sequence_length': len(group)
            })
        
        return sequences, labels, metadata
    
    def _create_fixed_length_sequences(self, df):
        """
        Create fixed-length sequences by sliding window approach.
        """
        sequences = []
        labels = []
        metadata = []
        
        # Group by driver to maintain driver consistency within sequences
        for driver, driver_group in df.groupby('Driver'):
            driver_group = driver_group.sort_values('SessionTimeSeconds')
            
            # Create sliding windows
            for i in range(0, len(driver_group) - self.sequence_length + 1, self.sequence_length // 2):
                window = driver_group.iloc[i:i + self.sequence_length]
                
                # Extract time series features
                sequence_data = window[self.time_series_features].values.T
                
                # Handle missing values
                if np.isnan(sequence_data).any():
                    sequence_data = pd.DataFrame(sequence_data.T).fillna(method='ffill').fillna(method='bfill').values.T
                
                sequences.append(sequence_data)
                labels.append(driver)
                
                metadata.append({
                    'driver': driver,
                    'start_time': window['SessionTimeSeconds'].iloc[0],
                    'end_time': window['SessionTimeSeconds'].iloc[-1],
                    'dominant_compound': window['Compound'].mode().iloc[0],
                    'session_name': window['session_name'].iloc[0]
                })
        
        return sequences, labels, metadata
    
    def _convert_to_aeon_format(self, sequences, labels):
        """
        Convert sequences to aeon-compatible format.
        """
        # Check if all sequences have the same length
        lengths = [seq.shape[1] for seq in sequences]
        
        if len(set(lengths)) == 1:
            # All sequences same length - create 3D numpy array
            X = np.array(sequences)  # Shape: (n_samples, n_channels, n_timepoints)
        else:
            # Unequal length sequences - keep as list of 2D arrays
            X = sequences
        
        # Encode labels
        if 'Driver' not in self.label_encoders:
            self.label_encoders['Driver'] = LabelEncoder()
            y = self.label_encoders['Driver'].fit_transform(labels)
        else:
            y = self.label_encoders['Driver'].transform(labels)
        
        return X, y
    
    def _apply_aeon_preprocessing(self, X, y):
        """
        Apply aeon preprocessing transformations.
        """
        print(f"Original data shape: {X.shape if isinstance(X, np.ndarray) else f'List of {len(X)} sequences'}")
        print(f"Has missing values: {has_missing(X)}")
        print(f"Is equal length: {is_equal_length(X)}")
        print(f"Is univariate: {is_univariate(X)}")
        
        # Handle missing values if present
        if has_missing(X):
            print("Applying missing value imputation...")
            imputer = SimpleImputer(strategy=self.imputation_strategy)
            X = imputer.fit_transform(X)
        
        # Handle unequal length sequences if needed
        if not is_equal_length(X):
            print(f"Handling unequal length sequences using: {self.handle_unequal_length}")
            
            if self.handle_unequal_length == 'pad':
                transformer = Padder()
            elif self.handle_unequal_length == 'truncate':
                transformer = Truncator()
            elif self.handle_unequal_length == 'resize':
                target_len = self.target_length or 100  # Default target length
                transformer = Resizer(length=target_len)
            else:
                raise ValueError(f"Unknown unequal length handling method: {self.handle_unequal_length}")
            
            X = transformer.fit_transform(X)
            print(f"After length handling: {X.shape}")
        
        # Apply scaling
        print(f"Applying scaling method: {self.scaling_method}")
        if self.scaling_method == 'normalize':
            scaler = Normalizer()
        elif self.scaling_method == 'center':
            scaler = Centerer()
        elif self.scaling_method == 'minmax':
            scaler = AeonMinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {self.scaling_method}")
        
        X_scaled = scaler.fit_transform(X)
        
        print(f"Final preprocessed data shape: {X_scaled.shape}")
        print(f"Sample mean after scaling: {np.mean(X_scaled, axis=-1)[0:3]}")
        print(f"Sample std after scaling: {np.std(X_scaled, axis=-1)[0:3]}")
        
        return X_scaled, y
    
    def fit_transform(self, f1_data):
        """
        Fit the preprocessor and transform F1 data into aeon-compatible format.
        
        Parameters:
        -----------
        f1_data : pd.DataFrame
            F1 telemetry data from F1DatasetPreprocessor
            
        Returns:
        --------
        X : np.ndarray or list
            Preprocessed time series data in aeon format
        y : np.ndarray
            Encoded labels
        metadata : list
            Metadata about each sequence
        """
        print("Starting F1 to Aeon preprocessing...")
        print(f"Input data shape: {f1_data.shape}")
        
        # Create sequences
        if self.sequence_length is None:
            sequences, labels, metadata = self._create_sequences_by_driver_lap(f1_data)
        else:
            sequences, labels, metadata = self._create_fixed_length_sequences(f1_data)
        
        print(f"Created {len(sequences)} sequences")
        print(f"Sequence length statistics: min={min(len(seq[0]) for seq in sequences)}, "
              f"max={max(len(seq[0]) for seq in sequences)}, "
              f"mean={np.mean([len(seq[0]) for seq in sequences]):.1f}")
        
        # Convert to aeon format
        X, y = self._convert_to_aeon_format(sequences, labels)
        
        # Apply aeon preprocessing
        X_processed, y_processed = self._apply_aeon_preprocessing(X, y)
        
        self.fitted = True
        
        return X_processed, y_processed, metadata
    
    def get_driver_names(self):
        """Get mapping of encoded labels back to driver names"""
        if 'Driver' in self.label_encoders:
            return dict(enumerate(self.label_encoders['Driver'].classes_))
        return {}


# Example usage and demonstration
def demonstrate_f1_aeon_preprocessing(f1_data):
    """
    Demonstrate the F1 Aeon preprocessing pipeline with different configurations.
    """
    print("="*80)
    print("DEMONSTRATING F1 AEON PREPROCESSING PIPELINE")
    print("="*80)
    
    # Configuration 1: Variable length sequences (by driver/lap) with normalization
    print("\n1. Variable length sequences with normalization:")
    print("-" * 50)
    
    preprocessor1 = F1AeonPreprocessor(
        sequence_length=None,  # Use natural lap boundaries
        scaling_method='normalize',
        handle_unequal_length='pad'
    )
    
    X1, y1, metadata1 = preprocessor1.fit_transform(f1_data)
    
    print(f"Driver mapping: {preprocessor1.get_driver_names()}")
    print(f"Unique drivers in data: {len(np.unique(y1))}")
    
    # Configuration 2: Fixed length sequences with min-max scaling
    print("\n2. Fixed length sequences (100 time points) with min-max scaling:")
    print("-" * 50)
    
    preprocessor2 = F1AeonPreprocessor(
        sequence_length=100,
        scaling_method='minmax',
        handle_unequal_length='resize',
        target_length=100
    )
    
    X2, y2, metadata2 = preprocessor2.fit_transform(f1_data)
    
    # Configuration 3: Centered sequences with truncation
    print("\n3. Variable length sequences with centering and truncation:")
    print("-" * 50)
    
    preprocessor3 = F1AeonPreprocessor(
        sequence_length=None,
        scaling_method='center',
        handle_unequal_length='truncate'
    )
    
    X3, y3, metadata3 = preprocessor3.fit_transform(f1_data)
    
    # Demonstrate creating a simple classification pipeline
    print("\n4. Creating classification pipeline with ROCKET:")
    print("-" * 50)
    
    # Split data for training
    X_train, X_test, y_train, y_test = train_test_split(
        X1, y1, test_size=0.2, random_state=42, stratify=y1
    )
    
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Create pipeline with additional preprocessing and ROCKET classifier
    pipeline = make_pipeline([
        Normalizer(),  # Additional normalization step
        RocketClassifier(n_kernels=1000, random_state=42)
    ])
    
    print("Fitting ROCKET classifier...")
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    train_accuracy = pipeline.score(X_train, y_train)
    test_accuracy = pipeline.score(X_test, y_test)
    
    print(f"Training accuracy: {train_accuracy:.3f}")
    print(f"Test accuracy: {test_accuracy:.3f}")
    
    # Show some sample predictions
    predictions = pipeline.predict(X_test[:5])
    driver_names = preprocessor1.get_driver_names()
    
    print("\nSample predictions:")
    for i, (pred, actual) in enumerate(zip(predictions[:5], y_test[:5])):
        print(f"Sample {i+1}: Predicted={driver_names.get(pred, pred)}, "
              f"Actual={driver_names.get(actual, actual)}")
    
    return {
        'preprocessor1': preprocessor1,
        'preprocessor2': preprocessor2, 
        'preprocessor3': preprocessor3,
        'pipeline': pipeline,
        'results': {
            'X1': X1, 'y1': y1, 'metadata1': metadata1,
            'X2': X2, 'y2': y2, 'metadata2': metadata2,
            'X3': X3, 'y3': y3, 'metadata3': metadata3,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy
        }
    }


# Additional utility functions for analysis
def analyze_sequence_characteristics(sequences, metadata):
    """Analyze characteristics of the created sequences"""
    lengths = [seq.shape[1] for seq in sequences]
    
    print("Sequence Analysis:")
    print(f"Number of sequences: {len(sequences)}")
    print(f"Number of channels per sequence: {sequences[0].shape[0]}")
    print(f"Length statistics:")
    print(f"  Min: {min(lengths)}")
    print(f"  Max: {max(lengths)}")
    print(f"  Mean: {np.mean(lengths):.1f}")
    print(f"  Std: {np.std(lengths):.1f}")
    
    # Analyze metadata
    if metadata:
        drivers = [m['driver'] for m in metadata]
        unique_drivers = set(drivers)
        print(f"Unique drivers: {len(unique_drivers)}")
        print(f"Sequences per driver: {len(sequences) / len(unique_drivers):.1f}")
        
        if 'compound' in metadata[0]:
            compounds = [m['compound'] for m in metadata]
            unique_compounds = set(compounds)
            print(f"Tire compounds: {unique_compounds}")


def create_custom_preprocessing_pipeline(scaling='normalize', length_handling='pad'):
    """
    Create a custom aeon preprocessing pipeline for F1 data.
    
    Parameters:
    -----------
    scaling : str
        Scaling method ('normalize', 'center', 'minmax')
    length_handling : str
        Method for handling unequal lengths ('pad', 'truncate', 'resize')
    
    Returns:
    --------
    pipeline : aeon pipeline
        Configured preprocessing pipeline
    """
    steps = []
    
    # Add missing value imputation
    steps.append(SimpleImputer(strategy='mean'))
    
    # Add length handling
    if length_handling == 'pad':
        steps.append(Padder())
    elif length_handling == 'truncate':
        steps.append(Truncator())
    elif length_handling == 'resize':
        steps.append(Resizer(length=100))
    
    # Add scaling
    if scaling == 'normalize':
        steps.append(Normalizer())
    elif scaling == 'center':
        steps.append(Centerer())
    elif scaling == 'minmax':
        steps.append(AeonMinMaxScaler())
    
    # Add classifier
    steps.append(RocketClassifier(n_kernels=1000, random_state=42))
    
    return make_pipeline(steps)


# Main execution example
if __name__ == "__main__":
    import fastf1
    from capstone import PreprocessorConfig, BaseFeatures, F1DatasetPreprocessor
    
    session = fastf1.get_session(2024, "São Paulo Grand Prix", "R")
    session.load()

    config = PreprocessorConfig(
        interval_seconds=1.0,  # 1-second intervals for temporal slicing
        balance_features=[BaseFeatures.DRIVER, BaseFeatures.COMPOUND],  # Balance by driver and tire compound
        balance_method="remove_insufficient",  # Remove underrepresented classes
        target_samples=2000,  # Target number of samples per class
        include_track_status=True,  # Include track status information
        include_event_info=True   # Include event metadata
    )

    # Process the F1 session data
    preprocessor = F1DatasetPreprocessor(config)
    f1_data = preprocessor.process_dataset(session)

    results = demonstrate_f1_aeon_preprocessing(f1_data)
    
    print("F1 Aeon Preprocessing Pipeline Complete!")
    print("Results stored in 'results' variable.")

# Output:
# ================================================================================
# DEMONSTRATING F1 AEON PREPROCESSING PIPELINE
# ================================================================================

# 1. Variable length sequences with normalization:
# --------------------------------------------------
# Starting F1 to Aeon preprocessing...
# Input data shape: (102531, 25)
# Created 1116 sequences
# Sequence length statistics: min=44, max=1392, mean=91.9
# Original data shape: List of 1116 sequences
# Has missing values: False
# Is equal length: False
# Is univariate: False
# Handling unequal length sequences using: pad
# After length handling: (1116, 7, 1392)
# Applying scaling method: normalize
# Final preprocessed data shape: (1116, 7, 1392)
# Sample mean after scaling: [[ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.02089474e-16
#    4.08357894e-17  4.08357894e-17  1.02089474e-16]
#  [-1.22507368e-16  8.16715788e-17  4.08357894e-17  1.02089474e-16
#    4.08357894e-17  1.02089474e-16  1.02089474e-16]
#  [ 4.08357894e-17 -8.16715788e-17  2.04178947e-17  4.08357894e-17
#    4.08357894e-17  1.02089474e-16  1.02089474e-16]]
# Sample std after scaling: [[1. 1. 1. 1. 1. 1. 1.]
#  [1. 1. 1. 1. 1. 1. 1.]
#  [1. 1. 1. 1. 1. 1. 1.]]
# Driver mapping: {0: np.str_('ALO'), 1: np.str_('BEA'), 2: np.str_('BOT'), 3: np.str_('COL'), 4: np.str_('GAS'), 5: np.str_('HAM'), 6: np.str_('HUL'), 7: np.str_('LAW'), 8: np.str_('LEC'), 9: np.str_('NOR'), 10: np.str_('OCO'), 11: np.str_('PER'), 12: np.str_('PIA'), 13: np.str_('RUS'), 14: np.str_('SAI'), 15: np.str_('TSU'), 16: np.str_('VER'), 17: np.str_('ZHO')}
# Unique drivers in data: 18

# 2. Fixed length sequences (100 time points) with min-max scaling:
# --------------------------------------------------
# Starting F1 to Aeon preprocessing...
# Input data shape: (102531, 25)
# Created 2024 sequences
# Sequence length statistics: min=100, max=100, mean=100.0
# Original data shape: (2024, 7, 100)
# Has missing values: False
# Is equal length: True
# Is univariate: False
# Applying scaling method: minmax
# Final preprocessed data shape: (2024, 7, 100)
# Sample mean after scaling: [[0.52186275 0.66500271 0.46656566 0.02       0.98       0.98
#   0.        ]
#  [0.42360996 0.61426868 0.44505051 0.52       0.48       0.48
#   0.        ]
#  [0.43206612 0.6252741  0.46515152 0.15       0.         0.
#   0.        ]]
# Sample std after scaling: [[0.23892608 0.24085324 0.39906199 0.14       0.14       0.14
#   0.        ]
#  [0.29963913 0.27448848 0.41569754 0.49959984 0.49959984 0.49959984
#   0.        ]
#  [0.29300249 0.27668294 0.41255022 0.35707142 0.         0.
#   0.        ]]

# 3. Variable length sequences with centering and truncation:
# --------------------------------------------------
# Starting F1 to Aeon preprocessing...
# Input data shape: (102531, 25)
# Created 1116 sequences
# Sequence length statistics: min=44, max=1392, mean=91.9
# Original data shape: List of 1116 sequences
# Has missing values: False
# Is equal length: False
# Is univariate: False
# Handling unequal length sequences using: truncate
# After length handling: (1116, 7, 44)
# Applying scaling method: center
# Final preprocessed data shape: (1116, 7, 44)
# Sample mean after scaling: [[-7.75137530e-15  3.30725346e-13  2.26081780e-15  0.00000000e+00
#    0.00000000e+00  1.06581410e-14  0.00000000e+00]
#  [-1.03351671e-14 -6.61450692e-13  4.19866162e-15  0.00000000e+00
#    0.00000000e+00  3.55271368e-15  0.00000000e+00]
#  [-7.75137530e-15 -6.61450692e-13 -2.58379177e-15  0.00000000e+00
#    0.00000000e+00  3.55271368e-15  0.00000000e+00]]
# Sample std after scaling: [[  70.75979329 1752.90113054   37.99926598    0.            0.
#      0.            0.        ]
#  [  65.12630767 1491.66493105   43.60943179    0.            0.
#      0.            0.        ]
#  [  66.04560705 1189.34975528   42.74764384    0.            0.
#      0.            0.        ]]

# 4. Creating classification pipeline with ROCKET:
# --------------------------------------------------
# Training set: (892, 7, 1392), Test set: (224, 7, 1392)
# Fitting ROCKET classifier...
# Training accuracy: 1.000
# Test accuracy: 0.848

# Sample predictions:
# Sample 1: Predicted=PER, Actual=PER
# Sample 2: Predicted=OCO, Actual=VER
# Sample 3: Predicted=LEC, Actual=LEC
# Sample 4: Predicted=BOT, Actual=ZHO
# Sample 5: Predicted=GAS, Actual=GAS
# F1 Aeon Preprocessing Pipeline Complete!
# Results stored in 'results' variable.