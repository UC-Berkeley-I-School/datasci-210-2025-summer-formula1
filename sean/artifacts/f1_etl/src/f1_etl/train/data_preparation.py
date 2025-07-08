"""Data preparation utilities for time series classification"""

import numpy as np


def prepare_data_with_validation(
    dataset, val_size=0.15, test_size=0.15, lookback=100, random_state=42
):
    """
    Prepare train/val/test splits for time series data with proper temporal ordering

    Args:
        dataset: Dataset from create_safety_car_dataset
        val_size: Proportion of data for validation (default 0.15, set to 0.0 to skip)
        test_size: Proportion of data for testing (default 0.15, set to 0.0 to skip)
        lookback: Number of timesteps to remove from val/test to prevent data leakage (default 100)
        random_state: Random seed for reproducibility (only used for train shuffle)

    Returns:
        Dictionary with train/val/test splits (val/test may be None if size is 0.0)
    """
    X = dataset["X"]  # Shape: (n_samples, n_timesteps, n_features)
    y = dataset["y"]  # Encoded labels

    # Convert to Aeon format: (n_samples, n_features, n_timesteps)
    X_aeon = X.transpose(0, 2, 1)

    n_samples = len(y)

    # Validate split sizes
    total_split = val_size + test_size
    if total_split > 1.0:
        raise ValueError(f"val_size + test_size ({total_split}) cannot exceed 1.0")

    # Calculate split indices based on provided sizes
    if test_size > 0 and val_size > 0:
        # All three splits
        train_end = int(n_samples * (1 - val_size - test_size))
        val_end = int(n_samples * (1 - test_size))

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

    elif test_size > 0 and val_size == 0:
        # Train + Test only
        train_end = int(n_samples * (1 - test_size))

        X_train = X_aeon[:train_end]
        y_train = y[:train_end]

        X_val = None
        y_val = None

        X_test = X_aeon[train_end:]
        y_test = y[train_end:]

        # Remove first `lookback` samples from test to prevent data leakage
        if len(X_test) > lookback:
            X_test = X_test[lookback:]
            y_test = y_test[lookback:]

    elif val_size > 0 and test_size == 0:
        # Train + Val only
        train_end = int(n_samples * (1 - val_size))

        X_train = X_aeon[:train_end]
        y_train = y[:train_end]

        X_val = X_aeon[train_end:]
        y_val = y[train_end:]

        X_test = None
        y_test = None

        # Remove first `lookback` samples from val to prevent data leakage
        if len(X_val) > lookback:
            X_val = X_val[lookback:]
            y_val = y_val[lookback:]

    else:
        # Train only
        X_train = X_aeon
        y_train = y

        X_val = None
        y_val = None

        X_test = None
        y_test = None

    # Shuffle only the training data
    np.random.seed(random_state)
    train_indices = np.random.permutation(len(y_train))
    X_train = X_train[train_indices]
    y_train = y_train[train_indices]

    # Print split information
    print("\n=== DATA SPLIT SUMMARY ===")
    print(f"Total samples: {n_samples:,}")
    print(f"Train: {len(y_train):,} ({len(y_train) / n_samples:.1%})")

    if y_val is not None:
        print(
            f"Val:   {len(y_val):,} ({len(y_val) / n_samples:.1%}) - removed {lookback} samples"
        )
    else:
        print("Val:   None (skipped)")

    if y_test is not None:
        print(
            f"Test:  {len(y_test):,} ({len(y_test) / n_samples:.1%}) - removed {lookback} samples"
        )
    else:
        print("Test:  None (skipped)")

    # Analyze class distribution in each split
    splits_info = {}
    splits_to_analyze = [("train", y_train)]

    if y_val is not None:
        splits_to_analyze.append(("val", y_val))
    if y_test is not None:
        splits_to_analyze.append(("test", y_test))

    for split_name, y_split in splits_to_analyze:
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
