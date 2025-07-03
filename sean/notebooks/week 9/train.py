# %% [markdown]
# # USAGE EXAMPLES
#
# ### QUICK START - Run this in notebook cells:
#
# # 1. Test driver encoder for a session
# ```
# # 1. Quick dataset creation (no setup needed)
# session_config = SessionConfig(year=2024, race="Monaco Grand Prix", session_type="R")
# # Driver abbreviations are handled automatically by the ETL pipeline
# ```
#
# # 2. Single experiment
# ```
# result = run_quick_test()
# ```
#
# # 3. Custom experiment with different drivers
# ```
# # First check available drivers for the session
# DATA_SCOPES["custom"] = {
#     "sessions": [SessionConfig(year=2024, race="Monaco Grand Prix", session_type="R")],
#     "drivers": ["HAM", "LEC"]  # Will be auto-converted to numbers
# }
# custom_result = run_single_experiment(
#     scope_name="custom",
#     window_config={"window_size": 150, "prediction_horizon": 5}
# )
# ```
#
# # 4. Run all experiments (warning: takes time!)
# ```
# all_results = run_all_experiments()
# summary_df = create_summary_report(all_results)
# print(summary_df)
# ```
#
# # 5. Manual step-by-step for debugging
# ```
# dataset = create_dataset("one_session_all_drivers", WINDOW_CONFIGS[0])
# X_train, X_test, y_train, y_test = prepare_data(dataset)
# class_names, class_dist = analyze_class_distribution(dataset, y_train)
#
# models = create_models()
# rf_result = train_single_model(
#     models["random_forest"], "random_forest",
#     X_train, X_test, y_train, y_test
# )
# ```

# %%
# F1 Safety Car Prediction - Cross-Race Evaluation Extension

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    confusion_matrix,
)
from aeon.classification.feature_based import Catch22Classifier
from aeon.classification.dummy import DummyClassifier
from f1_etl import SessionConfig, DataConfig, create_safety_car_dataset

# %%
# ============================================================================
# STEP 1: CONFIGURATION (ORIGINAL + CROSS-RACE)
# ============================================================================


def setup_experiment_config():
    """Define experimental parameters including cross-race evaluation"""

    # Window configurations to test
    WINDOW_CONFIGS = [
        {"window_size": 200, "prediction_horizon": 10},
        {"window_size": 300, "prediction_horizon": 15},
        {"window_size": 250, "prediction_horizon": 20},
    ]

    # Original training data configurations
    TRAINING_SCOPES = {
        "one_session_one_driver": {
            "sessions": [
                SessionConfig(year=2024, race="Monaco Grand Prix", session_type="R")
            ],
            "drivers": ["VER"],
        },
        "one_session_all_drivers": {
            "sessions": [
                SessionConfig(year=2024, race="Monaco Grand Prix", session_type="R")
            ],
            "drivers": None,
        },
    }

    # Cross-race evaluation configurations
    CROSS_RACE_CONFIGS = {
        "silverstone_2024": {
            "session": SessionConfig(
                year=2024, race="British Grand Prix", session_type="R"
            ),
            "description": "Silverstone 2024 Race",
        },
        "spa_2024": {
            "session": SessionConfig(
                year=2024, race="Belgian Grand Prix", session_type="R"
            ),
            "description": "Spa 2024 Race",
        },
        "monza_2024": {
            "session": SessionConfig(
                year=2024, race="Italian Grand Prix", session_type="R"
            ),
            "description": "Monza 2024 Race",
        },
    }

    return WINDOW_CONFIGS, TRAINING_SCOPES, CROSS_RACE_CONFIGS


# Run this first
WINDOW_CONFIGS, TRAINING_SCOPES, CROSS_RACE_CONFIGS = setup_experiment_config()
print("✅ Configuration loaded")
print(f"Window configs: {len(WINDOW_CONFIGS)}")
print(f"Training scopes: {list(TRAINING_SCOPES.keys())}")
print(f"Cross-race configs: {list(CROSS_RACE_CONFIGS.keys())}")

# %%
# ============================================================================
# STEP 2: ORIGINAL DATA GENERATION AND TRAINING
# ============================================================================


def create_dataset(
    scope_name, window_config, cache_dir="./f1_cache", training_scopes=None
):
    """Create a single dataset for given scope and window configuration"""

    scopes = training_scopes or TRAINING_SCOPES
    scope = scopes[scope_name]

    # Use driver abbreviations directly (F1 ETL pipeline handles the conversion)
    drivers = scope["drivers"]  # Keep as abbreviations
    if drivers:
        print(f"Using drivers: {drivers}")

    # Create data configuration
    config = DataConfig(
        sessions=scope["sessions"], cache_dir=cache_dir, drivers=drivers
    )

    # Generate dataset
    print(
        f"Generating dataset: {scope_name}, window={window_config['window_size']}, horizon={window_config['prediction_horizon']}"
    )
    dataset = create_safety_car_dataset(
        config=config,
        window_size=window_config["window_size"],
        prediction_horizon=window_config["prediction_horizon"],
        handle_non_numeric="encode",
        handle_missing=False,
        missing_strategy="forward_fill",
        normalize=True,
        normalization_method="per_sequence",
        target_column="TrackStatus",
        enable_debug=False,
    )

    return dataset


def prepare_data(dataset, test_size=0.2):
    """Prepare train/test splits and convert to Aeon format"""

    X = dataset["X"]
    y = dataset["y"]

    # Convert to Aeon format: (n_samples, n_features, n_timesteps)
    X_aeon = X.transpose(0, 2, 1)

    # Use only Speed feature (index 0) for simplicity
    X_speed = X_aeon[:, 0:1, :]

    # Train/test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X_speed, y, test_size=test_size, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test


def create_models():
    """Create dictionary of models to test"""

    models = {
        "dummy_frequent": DummyClassifier(strategy="most_frequent"),
        "dummy_stratified": DummyClassifier(strategy="stratified"),
        "logistic_regression": Catch22Classifier(
            estimator=LogisticRegression(
                random_state=42,
                max_iter=3000,
                solver="liblinear",
                class_weight="balanced",
            ),
            outlier_norm=True,
            random_state=42,
        ),
        "random_forest": Catch22Classifier(
            estimator=RandomForestClassifier(
                n_estimators=100, random_state=42, class_weight="balanced", max_depth=10
            ),
            outlier_norm=True,
            random_state=42,
        ),
    }

    return models


def train_models(X_train, y_train):
    """Train all models and return fitted models dictionary"""

    models = create_models()
    fitted_models = {}

    print("Training models...")
    for model_name, model in models.items():
        print(f"  Training {model_name}...")
        try:
            model.fit(X_train, y_train)
            fitted_models[model_name] = model
            print(f"  ✅ {model_name} trained successfully")
        except Exception as e:
            print(f"  ❌ {model_name} failed: {str(e)}")

    return fitted_models


# %%
# ============================================================================
# STEP 3: CROSS-RACE DATASET GENERATION
# ============================================================================


def create_cross_race_dataset(cross_race_key, window_config, cache_dir="./f1_cache"):
    """Create dataset from a different race for evaluation"""

    cross_race_config = CROSS_RACE_CONFIGS[cross_race_key]
    session_config = cross_race_config["session"]

    print(f"\n🏁 Creating cross-race dataset: {cross_race_config['description']}")

    # Create data configuration (all drivers for broader evaluation)
    config = DataConfig(
        sessions=[session_config],
        cache_dir=cache_dir,
        drivers=None,  # Use all drivers
    )

    # Generate dataset with same window configuration as training
    dataset = create_safety_car_dataset(
        config=config,
        window_size=window_config["window_size"],
        prediction_horizon=window_config["prediction_horizon"],
        handle_non_numeric="encode",
        handle_missing=False,
        missing_strategy="forward_fill",
        normalize=True,
        normalization_method="per_sequence",
        target_column="TrackStatus",
        enable_debug=False,
    )

    return dataset


def prepare_cross_race_data(dataset):
    """Prepare cross-race data for evaluation (no train/test split needed)"""

    X = dataset["X"]
    y = dataset["y"]

    # Convert to Aeon format: (n_samples, n_features, n_timesteps)
    X_aeon = X.transpose(0, 2, 1)

    # Use only Speed feature (index 0) to match training data
    X_speed = X_aeon[:, 0:1, :]

    return X_speed, y


# %%
# ============================================================================
# STEP 4: CROSS-RACE EVALUATION
# ============================================================================


def evaluate_models_cross_race(
    fitted_models, X_cross_race, y_cross_race, cross_race_name
):
    """Evaluate fitted models on cross-race data"""

    print(f"\n🔍 Evaluating models on {cross_race_name}")
    print(f"Cross-race data shape: {X_cross_race.shape}")
    print(f"Cross-race labels: {len(y_cross_race)}")

    results = {}

    for model_name, model in fitted_models.items():
        print(f"\n  Evaluating {model_name}...")

        try:
            # Predict on cross-race data
            y_pred = model.predict(X_cross_race)

            # Calculate metrics
            results[model_name] = {
                "model_name": model_name,
                "cross_race": cross_race_name,
                "accuracy": accuracy_score(y_cross_race, y_pred),
                "f1_macro": f1_score(
                    y_cross_race, y_pred, average="macro", zero_division=0
                ),
                "f1_weighted": f1_score(
                    y_cross_race, y_pred, average="weighted", zero_division=0
                ),
                "predictions": y_pred,
                "true_labels": y_cross_race,
            }

            print(f"    Accuracy: {results[model_name]['accuracy']:.4f}")
            print(f"    F1-Macro: {results[model_name]['f1_macro']:.4f}")

        except Exception as e:
            print(f"    ERROR: {str(e)}")
            results[model_name] = {"model_name": model_name, "error": str(e)}

    return results


def analyze_cross_race_distribution(dataset, y_cross_race):
    """Analyze class distribution in cross-race data"""

    label_encoder = dataset["label_encoder"]
    class_names = label_encoder.get_classes()

    unique, counts = np.unique(y_cross_race, return_counts=True)

    print("\n=== CROSS-RACE CLASS DISTRIBUTION ===")
    for class_id, count in zip(unique, counts):
        class_name = (
            class_names[class_id]
            if class_id < len(class_names)
            else f"Class_{class_id}"
        )
        percentage = count / len(y_cross_race) * 100
        print(f"{class_name:12s}: {count:5d} samples ({percentage:5.1f}%)")

    return class_names, dict(zip(unique, counts))


# %%
# ============================================================================
# STEP 5: COMPARISON AND ANALYSIS
# ============================================================================


def compare_train_vs_cross_race(train_results, cross_race_results, comparison_title=""):
    """Compare training performance vs cross-race performance"""

    print(f"\n{'=' * 80}")
    print(
        f"TRAIN vs CROSS-RACE COMPARISON{' - ' + comparison_title if comparison_title else ''}"
    )
    print(f"{'=' * 80}")

    comparison_data = []

    for model_name in train_results.keys():
        if (
            model_name in cross_race_results
            and "error" not in train_results[model_name]
            and "error" not in cross_race_results[model_name]
        ):
            train_result = train_results[model_name]
            cross_result = cross_race_results[model_name]

            comparison_data.append(
                {
                    "Model": model_name,
                    "Train_Accuracy": train_result["accuracy"],
                    "Cross_Accuracy": cross_result["accuracy"],
                    "Accuracy_Drop": train_result["accuracy"]
                    - cross_result["accuracy"],
                    "Train_F1": train_result["f1_macro"],
                    "Cross_F1": cross_result["f1_macro"],
                    "F1_Drop": train_result["f1_macro"] - cross_result["f1_macro"],
                }
            )

    if not comparison_data:
        print("No comparable results found!")
        return None

    df = pd.DataFrame(comparison_data)
    df = df.sort_values("Cross_F1", ascending=False)

    print(df.to_string(index=False, float_format="%.4f"))

    # Highlight generalization performance
    print("\n📊 GENERALIZATION ANALYSIS:")
    for _, row in df.iterrows():
        model = row["Model"]
        acc_drop = row["Accuracy_Drop"]
        f1_drop = row["F1_Drop"]

        generalization = (
            "Good" if f1_drop < 0.1 else "Poor" if f1_drop > 0.3 else "Moderate"
        )
        print(
            f"  {model:20s}: F1 drop = {f1_drop:+.3f}, Generalization = {generalization}"
        )

    return df


def detailed_cross_race_analysis(
    cross_race_results, class_names, target_class="safety_car"
):
    """Detailed analysis of cross-race performance"""

    print("\n=== CROSS-RACE DETAILED ANALYSIS ===")

    # Find target class index
    target_idx = None
    if class_names is not None:
        try:
            target_idx = list(class_names).index(target_class)
        except ValueError:
            print(f"Class '{target_class}' not found in {list(class_names)}")

    for model_name, result in cross_race_results.items():
        if "error" in result:
            continue

        y_true = result["true_labels"]
        y_pred = result["predictions"]

        print(f"\n{'-' * 60}")
        print(f"CROSS-RACE MODEL: {model_name.upper()}")
        print(f"{'-' * 60}")

        # Classification report
        if class_names is not None:
            try:
                unique_classes = sorted(np.unique(np.concatenate([y_true, y_pred])))
                present_class_names = [
                    class_names[i] for i in unique_classes if i < len(class_names)
                ]

                report = classification_report(
                    y_true,
                    y_pred,
                    target_names=present_class_names,
                    zero_division=0,
                    digits=4,
                )
                print("📊 Classification Report:")
                print(report)
            except Exception as e:
                print(f"Error in classification report: {e}")

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        print("\n🔍 Confusion Matrix:")
        print(cm)

        # Target class analysis
        if target_idx is not None:
            unique_classes = sorted(np.unique(np.concatenate([y_true, y_pred])))
            if target_idx in unique_classes:
                target_in_unique = unique_classes.index(target_idx)

                if target_in_unique < cm.shape[0] and target_in_unique < cm.shape[1]:
                    tp = cm[target_in_unique, target_in_unique]
                    fn = cm[target_in_unique, :].sum() - tp
                    fp = cm[:, target_in_unique].sum() - tp

                    print(f"\n🚨 {target_class.title()} Class Performance:")
                    print(f"  True Positives:  {tp:4d}")
                    print(f"  False Negatives: {fn:4d}")
                    print(f"  False Positives: {fp:4d}")

                    if tp + fn > 0:
                        recall = tp / (tp + fn)
                        print(f"  Cross-Race Recall: {recall:.3f}")


# %%
# ============================================================================
# STEP 6: COMPLETE CROSS-RACE EXPERIMENT RUNNER
# ============================================================================


def run_cross_race_experiment(
    train_scope, cross_race_key, window_config, cache_dir="./f1_cache"
):
    """Run complete cross-race experiment: train on one race, test on another"""

    print(f"\n{'=' * 80}")
    print("CROSS-RACE EXPERIMENT")
    print(f"Training: {train_scope}")
    print(f"Testing: {CROSS_RACE_CONFIGS[cross_race_key]['description']}")
    print(
        f"Window: {window_config['window_size']}, Horizon: {window_config['prediction_horizon']}"
    )
    print(f"{'=' * 80}")

    try:
        # Step 1: Create and prepare training dataset
        print("\n📚 STEP 1: Creating training dataset...")
        train_dataset = create_dataset(train_scope, window_config, cache_dir)
        X_train, X_test, y_train, y_test = prepare_data(train_dataset)

        # Step 2: Train models
        print("\n🏋️ STEP 2: Training models...")
        fitted_models = train_models(X_train, y_train)

        # Step 3: Evaluate on original test set
        print("\n📊 STEP 3: Evaluating on original test set...")
        train_results = {}
        for model_name, model in fitted_models.items():
            try:
                y_pred = model.predict(X_test)
                train_results[model_name] = {
                    "model_name": model_name,
                    "accuracy": accuracy_score(y_test, y_pred),
                    "f1_macro": f1_score(
                        y_test, y_pred, average="macro", zero_division=0
                    ),
                    "f1_weighted": f1_score(
                        y_test, y_pred, average="weighted", zero_division=0
                    ),
                    "predictions": y_pred,
                    "true_labels": y_test,
                }
            except Exception as e:
                train_results[model_name] = {"model_name": model_name, "error": str(e)}

        # Step 4: Create cross-race dataset
        print("\n🏁 STEP 4: Creating cross-race dataset...")
        cross_race_dataset = create_cross_race_dataset(
            cross_race_key, window_config, cache_dir
        )
        X_cross_race, y_cross_race = prepare_cross_race_data(cross_race_dataset)

        # Step 5: Analyze cross-race data distribution
        cross_race_class_names, cross_race_dist = analyze_cross_race_distribution(
            cross_race_dataset, y_cross_race
        )

        # Step 6: Evaluate models on cross-race data
        print("\n🔍 STEP 5: Evaluating on cross-race data...")
        cross_race_results = evaluate_models_cross_race(
            fitted_models,
            X_cross_race,
            y_cross_race,
            CROSS_RACE_CONFIGS[cross_race_key]["description"],
        )

        # Step 7: Compare and analyze results
        print("\n📈 STEP 6: Comparing results...")
        comparison_title = f"{train_scope} → {cross_race_key}"
        comparison_df = compare_train_vs_cross_race(
            train_results, cross_race_results, comparison_title
        )

        # Step 8: Detailed analysis
        detailed_cross_race_analysis(cross_race_results, cross_race_class_names)

        return {
            "train_scope": train_scope,
            "cross_race_key": cross_race_key,
            "window_config": window_config,
            "fitted_models": fitted_models,
            "train_results": train_results,
            "cross_race_results": cross_race_results,
            "comparison": comparison_df,
            "train_class_names": train_dataset["label_encoder"].get_classes(),
            "cross_race_class_names": cross_race_class_names,
        }

    except Exception as e:
        print(f"❌ CROSS-RACE EXPERIMENT FAILED: {str(e)}")
        return None


def run_quick_cross_race_test():
    """Run a single quick cross-race test"""

    print("🚀 Running quick cross-race test...")
    result = run_cross_race_experiment(
        train_scope="one_session_all_drivers",
        cross_race_key="silverstone_2024",
        window_config=WINDOW_CONFIGS[0],
    )
    return result


def run_full_cross_race_matrix():
    """Run cross-race evaluation matrix: all training scopes vs all cross-race targets"""

    all_results = []

    for train_scope in TRAINING_SCOPES.keys():
        for cross_race_key in CROSS_RACE_CONFIGS.keys():
            for window_config in WINDOW_CONFIGS:
                print(f"\n🎯 Running: {train_scope} → {cross_race_key}")
                result = run_cross_race_experiment(
                    train_scope, cross_race_key, window_config
                )
                if result:
                    all_results.append(result)

    return all_results


def create_cross_race_summary(all_results):
    """Create comprehensive summary of all cross-race experiments"""

    summary_data = []

    for experiment in all_results:
        train_scope = experiment["train_scope"]
        cross_race = experiment["cross_race_key"]
        window_size = experiment["window_config"]["window_size"]
        horizon = experiment["window_config"]["prediction_horizon"]

        for model_name, train_result in experiment["train_results"].items():
            if (
                "error" not in train_result
                and model_name in experiment["cross_race_results"]
            ):
                cross_result = experiment["cross_race_results"][model_name]
                if "error" not in cross_result:
                    summary_data.append(
                        {
                            "Train_Scope": train_scope,
                            "Cross_Race": cross_race,
                            "Window": window_size,
                            "Horizon": horizon,
                            "Model": model_name,
                            "Train_F1": train_result["f1_macro"],
                            "Cross_F1": cross_result["f1_macro"],
                            "F1_Drop": train_result["f1_macro"]
                            - cross_result["f1_macro"],
                            "Generalization": train_result["f1_macro"]
                            - cross_result["f1_macro"],
                        }
                    )

    df = pd.DataFrame(summary_data)

    # Add generalization quality column
    df["Gen_Quality"] = df["F1_Drop"].apply(
        lambda x: "Good" if x < 0.1 else "Poor" if x > 0.3 else "Moderate"
    )

    return df.sort_values(["Cross_F1"], ascending=False)


# %%
# ============================================================================
# STEP 7: ISOLATED EXECUTION EXAMPLES
# ============================================================================

# Example 1: Quick single cross-race test
# result = run_quick_cross_race_test()

# Example 2: Specific cross-race experiment
result = run_cross_race_experiment(
    train_scope="one_session_all_drivers",
    cross_race_key="spa_2024",
    window_config=WINDOW_CONFIGS[1],
)

# Example 3: Full cross-race matrix (WARNING: Takes long time)
# all_results = run_full_cross_race_matrix()
# summary_df = create_cross_race_summary(all_results)
# print(summary_df.to_string(index=False))


# %%
result.keys()
# %%
result["cross_race_results"]
# %%
data
