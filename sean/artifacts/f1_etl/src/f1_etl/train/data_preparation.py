"""Data preparation utilities for time series classification"""

import numpy as np


def prepare_data_with_validation(
    dataset, val_size=0.15, test_size=0.15, lookback=100, random_state=42
):
    """
    Prepare train/val/test splits for time series data with proper temporal ordering

    Args:
        dataset: Dataset from create_safety_car_dataset
        val_size: Proportion of data for validation (default 0.15)
        test_size: Proportion of data for testing (default 0.15)
        lookback: Number of timesteps to remove from val/test to prevent data leakage (default 100)
        random_state: Random seed for reproducibility (only used for train shuffle)

    Returns:
        Dictionary with train/val/test splits
    """
    X = dataset["X"]  # Shape: (n_samples, n_timesteps, n_features)
    y = dataset["y"]  # Encoded labels

    # Convert to Aeon format: (n_samples, n_features, n_timesteps)
    X_aeon = X.transpose(0, 2, 1)

    n_samples = len(y)

    # Calculate split indices (no shuffling to preserve temporal order)
    train_end = int(n_samples * (1 - val_size - test_size))
    val_end = int(n_samples * (1 - test_size))

    # Split data temporally
    X_train = X_aeon[:train_end]
    y_train = y[:train_end]

    X_val = X_aeon[train_end:val_end]
    y_val = y[train_end:val_end]

    X_test = X_aeon[val_end:]
    y_test = y[val_end:]

    # Remove first `lookback` samples from val and test to prevent data leakage
    if len(X_val) > lookback:
        X_val = X_val[lookback:]
        y_val = y_val[lookback:]

    if len(X_test) > lookback:
        X_test = X_test[lookback:]
        y_test = y_test[lookback:]

    # Shuffle only the training data
    np.random.seed(random_state)
    train_indices = np.random.permutation(len(y_train))
    X_train = X_train[train_indices]
    y_train = y_train[train_indices]

    # Print split information
    print("\n=== DATA SPLIT SUMMARY ===")
    print(f"Total samples: {n_samples:,}")
    print(f"Train: {len(y_train):,} ({len(y_train) / n_samples:.1%})")
    print(
        f"Val:   {len(y_val):,} ({len(y_val) / n_samples:.1%}) - removed {lookback} samples"
    )
    print(
        f"Test:  {len(y_test):,} ({len(y_test) / n_samples:.1%}) - removed {lookback} samples"
    )

    # Analyze class distribution in each split
    splits_info = {}
    for split_name, y_split in [("train", y_train), ("val", y_val), ("test", y_test)]:
        unique, counts = np.unique(y_split, return_counts=True)
        splits_info[split_name] = dict(zip(unique, counts))
        print(f"\n{split_name.capitalize()} class distribution:")
        for class_idx, count in zip(unique, counts):
            print(f"  Class {class_idx}: {count:,} ({count / len(y_split):.1%})")

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "splits_info": splits_info,
    }


def analyze_class_distribution(y, class_names=None):
    """
    Analyze and print class distribution

    Args:
        y: Label array
        class_names: Optional list of class names

    Returns:
        Dictionary mapping class indices to counts
    """
    unique, counts = np.unique(y, return_counts=True)
    distribution = dict(zip(unique, counts))

    print("Class distribution:")
    for class_idx, count in distribution.items():
        class_name = (
            class_names[class_idx]
            if class_names and class_idx < len(class_names)
            else f"Class {class_idx}"
        )
        print(f"  {class_name}: {count:,} ({count / len(y):.1%})")

    return distribution
