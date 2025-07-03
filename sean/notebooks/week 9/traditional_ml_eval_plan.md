# Traditional ML Evaluation Plan - F1 Safety Car Prediction

## Objective
Evaluate traditional ML algorithms (Logistic Regression, Linear Regression, Random Forest) for safety car prediction using Catch22 temporal feature extraction across different data scope and temporal window configurations.

## Technical Approach

### Feature Extraction Strategy
- **Catch22**: Automated temporal feature extraction via Aeon's `Catch22Classifier`
- **Input**: Raw time series windows of telemetry data
- **Output**: 22 statistical features per window for classification

### Model Configuration
```python
# Example implementation
clf = Catch22Classifier(
    estimator=RandomForestClassifier(n_estimators=100, random_state=42),
    outlier_norm=True,
    random_state=42,
)
```

### Algorithms Under Test
1. **Logistic Regression** - Linear baseline with feature importance
2. **Linear Regression** - Continuous output baseline  
3. **Random Forest** - Non-linear ensemble method

## Experimental Design

### Data Scope Scenarios
1. **One Session, One Driver** - Single race, single driver (minimal data)
2. **Whole Season, One Driver** - All 2024 races, single driver (temporal generalization)
3. **One Session, All Drivers** - Single race, all drivers (driver generalization) 
4. **Whole Season, All Drivers** - All 2024 races, all drivers (maximum data)

### Window Size & Prediction Horizon Combinations
Given the need for sufficient samples with Catch22 feature extraction:

- **Window=200, Horizon=10** - Large context, short prediction
- **Window=300, Horizon=15** - Maximum context, medium prediction  
- **Window=250, Horizon=20** - Balanced context, longer prediction

*Rationale: Larger windows ensure adequate sample sizes after windowing and provide rich temporal context for Catch22 feature extraction.*

## Experiment Tracking

### One Session, One Driver

#### Window=200, Horizon=10
| Algorithm | F1 Score | Precision | Recall | Accuracy | Sample Count | Notes |
|-----------|----------|-----------|--------|----------|--------------|-------|
| Baseline (DummyClassifier) | | | | | | |
| Logistic Regression | | | | | | |
| Linear Regression | | | | | | |
| Random Forest | | | | | | |

#### Window=300, Horizon=15
| Algorithm | F1 Score | Precision | Recall | Accuracy | Sample Count | Notes |
|-----------|----------|-----------|--------|----------|--------------|-------|
| Baseline (DummyClassifier) | | | | | | |
| Logistic Regression | | | | | | |
| Linear Regression | | | | | | |
| Random Forest | | | | | | |

#### Window=250, Horizon=20
| Algorithm | F1 Score | Precision | Recall | Accuracy | Sample Count | Notes |
|-----------|----------|-----------|--------|----------|--------------|-------|
| Baseline (DummyClassifier) | | | | | | |
| Logistic Regression | | | | | | |
| Linear Regression | | | | | | |
| Random Forest | | | | | | |

### Whole Season, One Driver

#### Window=200, Horizon=10
| Algorithm | F1 Score | Precision | Recall | Accuracy | Sample Count | Notes |
|-----------|----------|-----------|--------|----------|--------------|-------|
| Baseline (DummyClassifier) | | | | | | |
| Logistic Regression | | | | | | |
| Linear Regression | | | | | | |
| Random Forest | | | | | | |

#### Window=300, Horizon=15
| Algorithm | F1 Score | Precision | Recall | Accuracy | Sample Count | Notes |
|-----------|----------|-----------|--------|----------|--------------|-------|
| Baseline (DummyClassifier) | | | | | | |
| Logistic Regression | | | | | | |
| Linear Regression | | | | | | |
| Random Forest | | | | | | |

#### Window=250, Horizon=20
| Algorithm | F1 Score | Precision | Recall | Accuracy | Sample Count | Notes |
|-----------|----------|-----------|--------|----------|--------------|-------|
| Baseline (DummyClassifier) | | | | | | |
| Logistic Regression | | | | | | |
| Linear Regression | | | | | | |
| Random Forest | | | | | | |

### One Session, All Drivers

#### Window=200, Horizon=10
| Algorithm | F1 Score | Precision | Recall | Accuracy | Sample Count | Notes |
|-----------|----------|-----------|--------|----------|--------------|-------|
| Baseline (DummyClassifier) | | | | | | |
| Logistic Regression | | | | | | |
| Linear Regression | | | | | | |
| Random Forest | | | | | | |

#### Window=300, Horizon=15
| Algorithm | F1 Score | Precision | Recall | Accuracy | Sample Count | Notes |
|-----------|----------|-----------|--------|----------|--------------|-------|
| Baseline (DummyClassifier) | | | | | | |
| Logistic Regression | | | | | | |
| Linear Regression | | | | | | |
| Random Forest | | | | | | |

#### Window=250, Horizon=20
| Algorithm | F1 Score | Precision | Recall | Accuracy | Sample Count | Notes |
|-----------|----------|-----------|--------|----------|--------------|-------|
| Baseline (DummyClassifier) | | | | | | |
| Logistic Regression | | | | | | |
| Linear Regression | | | | | | |
| Random Forest | | | | | | |

### Whole Season, All Drivers

#### Window=200, Horizon=10
| Algorithm | F1 Score | Precision | Recall | Accuracy | Sample Count | Notes |
|-----------|----------|-----------|--------|----------|--------------|-------|
| Baseline (DummyClassifier) | | | | | | |
| Logistic Regression | | | | | | |
| Linear Regression | | | | | | |
| Random Forest | | | | | | |

#### Window=300, Horizon=15
| Algorithm | F1 Score | Precision | Recall | Accuracy | Sample Count | Notes |
|-----------|----------|-----------|--------|----------|--------------|-------|
| Baseline (DummyClassifier) | | | | | | |
| Logistic Regression | | | | | | |
| Linear Regression | | | | | | |
| Random Forest | | | | | | |

#### Window=250, Horizon=20
| Algorithm | F1 Score | Precision | Recall | Accuracy | Sample Count | Notes |
|-----------|----------|-----------|--------|----------|--------------|-------|
| Baseline (DummyClassifier) | | | | | | |
| Logistic Regression | | | | | | |
| Linear Regression | | | | | | |
| Random Forest | | | | | | |

## Implementation Priority

### Phase 1: Proof of Concept
1. **One Session, One Driver** - Fastest to implement and debug
2. Focus on **Window=200, Horizon=10** first
3. Establish baseline performance and pipeline

### Phase 2: Scaling Analysis  
1. **Whole Season, One Driver** - Test temporal generalization
2. **One Session, All Drivers** - Test driver generalization
3. Compare against Phase 1 results

### Phase 3: Full Evaluation
1. **Whole Season, All Drivers** - Maximum data configuration
2. Complete window size/horizon grid
3. Final performance comparison

## Success Criteria

### Minimum Viable Performance
- **F1 Score > DummyClassifier baseline** across all scenarios
- **Sample counts sufficient** for reliable evaluation (>1000 samples minimum)

### Target Performance  
- **F1 > 0.3** in at least one configuration
- **Consistent performance** across window size variations (±0.05 F1)

### Analysis Deliverables
1. **Best performing configuration** (data scope + window/horizon combination)
2. **Generalization analysis** (single vs. multi driver/session performance)
3. **Temporal window sensitivity** analysis
4. **Recommendations** for production deployment

## Risk Mitigation

### Data Availability
- **Monitor sample counts** in each configuration
- **Adjust window sizes** if insufficient samples generated
- **Document data limitations** per scenario

### Computational Efficiency
- **Profile Catch22 extraction time** across window sizes
- **Implement parallel processing** for larger datasets
- **Set timeout limits** for long-running experiments

## Next Steps
1. Implement data preprocessing pipeline for windowing
2. Set up Catch22 feature extraction workflow  
3. Execute Phase 1 experiments
4. Analyze initial results and refine approach for subsequent phases