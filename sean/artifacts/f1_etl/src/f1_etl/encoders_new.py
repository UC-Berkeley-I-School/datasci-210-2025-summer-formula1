"""Simple Fixed Vocabulary Track Status Encoder"""

from typing import Dict

import numpy as np
import pandas as pd


class FixedVocabTrackStatusEncoder:
    """
    Simple track status encoder with fixed vocabulary.

    Always produces the same integer labels for track statuses regardless
    of which race it's trained on. Can optionally output one-hot vectors.
    """

    def __init__(self, use_onehot: bool = False):
        """
        Initialize with fixed track status vocabulary.

        Parameters:
        -----------
        use_onehot : bool, default=False
            If True, transform() returns one-hot encoded vectors
            If False, transform() returns integer labels
        """
        # Fixed mapping for all known track statuses (alphabetically sorted for consistency)
        self.track_status_mapping = {
            "1": "green",  # Track clear
            "2": "yellow",  # Yellow flag
            "4": "safety_car",  # Safety Car
            "5": "red",  # Red Flag
            "6": "vsc",  # Virtual Safety Car deployed
            "7": "vsc_ending",  # Virtual Safety Car ending
        }

        # Fixed class vocabulary (alphabetically sorted)
        self.classes_ = np.array(
            ["green", "red", "safety_car", "unknown", "vsc", "vsc_ending", "yellow"]
        )

        self.use_onehot = use_onehot
        self.n_classes = len(self.classes_)

        # Create fixed mapping: class_name -> index
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes_)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}

        # Track what we've seen (for analysis purposes)
        self.fitted_classes_seen = set()
        self.is_fitted = False

    def fit(self, track_status_data: pd.Series) -> "FixedVocabTrackStatusEncoder":
        """
        Fit encoder (records which classes were seen in training).

        The encoder uses fixed vocabulary, so fitting just tracks
        which classes were actually present.
        """
        mapped_labels = self._map_track_statuses(track_status_data)
        self.fitted_classes_seen = set(mapped_labels.unique())
        self.is_fitted = True

        print("✅ FixedVocabTrackStatusEncoder fitted")
        print(f"   Classes seen: {sorted(self.fitted_classes_seen)}")
        print(f"   Total classes: {len(self.classes_)}")
        print(
            f"   Output mode: {'one-hot vectors' if self.use_onehot else 'integer labels'}"
        )

        return self

    def transform(self, track_status_data: pd.Series) -> np.ndarray:
        """
        Transform track status data to consistent labels.

        Returns:
        --------
        np.ndarray : Either integer labels or one-hot vectors depending on use_onehot
        """
        mapped_labels = self._map_track_statuses(track_status_data)

        # Convert to integer indices using fixed mapping
        integer_labels = np.array([self.class_to_idx[label] for label in mapped_labels])

        if self.use_onehot:
            # Convert to one-hot vectors
            onehot = np.zeros((len(integer_labels), self.n_classes))
            onehot[np.arange(len(integer_labels)), integer_labels] = 1
            return onehot
        else:
            return integer_labels

    def fit_transform(self, track_status_data: pd.Series) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(track_status_data).transform(track_status_data)

    def inverse_transform(self, encoded_labels: np.ndarray) -> np.ndarray:
        """
        Convert labels back to class names.

        Parameters:
        -----------
        encoded_labels : np.ndarray
            Either integer labels or one-hot vectors
        """
        if encoded_labels.ndim == 2:
            # One-hot vectors - convert to integer labels first
            integer_labels = np.argmax(encoded_labels, axis=1)
        else:
            # Already integer labels
            integer_labels = encoded_labels

        return np.array([self.idx_to_class[idx] for idx in integer_labels])

    def get_classes(self) -> np.ndarray:
        """Return all possible classes in fixed vocabulary."""
        return self.classes_.copy()

    def get_n_classes(self) -> int:
        """Return number of classes."""
        return self.n_classes

    def get_class_distribution(self, track_status_data: pd.Series) -> Dict[str, int]:
        """Get distribution of classes in given data."""
        mapped_labels = self._map_track_statuses(track_status_data)
        value_counts = mapped_labels.value_counts()

        # Ensure all classes are represented (with 0 for unseen classes)
        distribution = {}
        for class_name in self.classes_:
            distribution[class_name] = value_counts.get(class_name, 0)

        return distribution

    def _map_track_statuses(self, track_status_data: pd.Series) -> pd.Series:
        """Map raw track status codes to class names."""
        mapped = track_status_data.map(self.track_status_mapping).fillna("unknown")
        return mapped

    def analyze_data(
        self, track_status_data: pd.Series, data_name: str = "data"
    ) -> Dict:
        """Analyze track status distribution in data."""
        distribution = self.get_class_distribution(track_status_data)
        present_classes = {k for k, v in distribution.items() if v > 0}

        print(f"\n📊 Track Status Analysis ({data_name}):")
        total_samples = len(track_status_data)

        for class_name in self.classes_:
            count = distribution[class_name]
            if count > 0:
                percentage = count / total_samples * 100
                print(f"   {class_name:12s}: {count:5d} samples ({percentage:5.1f}%)")

        missing_classes = set(self.classes_) - present_classes
        if missing_classes:
            print(f"   Missing classes: {sorted(missing_classes)}")

        return {
            "distribution": distribution,
            "present_classes": sorted(present_classes),
            "missing_classes": sorted(missing_classes),
            "total_samples": total_samples,
        }


def compare_race_distributions(
    encoder: FixedVocabTrackStatusEncoder,
    y1: pd.Series,
    name1: str,
    y2: pd.Series,
    name2: str,
) -> None:
    """Compare track status distributions between two races."""

    print(f"\n🏁 Comparing {name1} vs {name2}")
    print("=" * 60)

    dist1 = encoder.get_class_distribution(y1)
    dist2 = encoder.get_class_distribution(y2)

    present1 = {k for k, v in dist1.items() if v > 0}
    present2 = {k for k, v in dist2.items() if v > 0}

    common = present1 & present2
    only_1 = present1 - present2
    only_2 = present2 - present1

    print(f"Common classes: {sorted(common)}")
    if only_1:
        print(f"Only in {name1}: {sorted(only_1)}")
    if only_2:
        print(f"Only in {name2}: {sorted(only_2)}")

    print("\nDistribution comparison:")
    print(f"{'Class':<12} {name1:<10} {name2:<10} {'Ratio':<8}")
    print("-" * 50)

    for class_name in encoder.get_classes():
        count1 = dist1[class_name]
        count2 = dist2[class_name]

        if count1 > 0 or count2 > 0:
            ratio = count2 / count1 if count1 > 0 else float("inf")
            print(f"{class_name:<12} {count1:<10} {count2:<10} {ratio:<8.2f}")


# Example usage and testing
if __name__ == "__main__":
    print("Testing Fixed Vocabulary Track Status Encoder:")
    print("=" * 60)

    # Test data: different races with different track status distributions
    monaco_data = pd.Series(
        ["1", "1", "2", "1", "4", "1", "1"]
    )  # Green, Yellow, Safety Car
    spa_data = pd.Series(["1", "1", "6", "7", "1", "5", "1"])  # Green, VSC, Red Flag

    # Test integer labels
    print("\n1. Testing with integer labels:")
    encoder_int = FixedVocabTrackStatusEncoder(use_onehot=False)
    encoder_int.fit(monaco_data)

    monaco_encoded = encoder_int.transform(monaco_data)
    spa_encoded = encoder_int.transform(spa_data)

    print(f"Monaco encoded: {monaco_encoded}")
    print(f"Spa encoded: {spa_encoded}")
    print(f"All classes: {encoder_int.get_classes()}")

    # Test one-hot encoding
    print("\n2. Testing with one-hot vectors:")
    encoder_onehot = FixedVocabTrackStatusEncoder(use_onehot=True)
    encoder_onehot.fit(monaco_data)

    monaco_onehot = encoder_onehot.transform(monaco_data)
    spa_onehot = encoder_onehot.transform(spa_data)

    print(f"Monaco one-hot shape: {monaco_onehot.shape}")
    print(f"Spa one-hot shape: {spa_onehot.shape}")
    print(f"First Monaco sample: {monaco_onehot[0]}")

    # Test analysis
    print("\n3. Distribution analysis:")
    encoder_int.analyze_data(monaco_data, "Monaco")
    encoder_int.analyze_data(spa_data, "Spa")

    # Test comparison
    compare_race_distributions(encoder_int, monaco_data, "Monaco", spa_data, "Spa")
