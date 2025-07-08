## v0.6.1 (2025-07-08)

### Fix

- pass resampling_config during x-evals

## v0.6.0 (2025-07-08)

### Feat

- capture resampling config in metadata
- enable optional val and test sets

## v0.5.0 (2025-07-06)

### Feat

- implement session-level filtering by driver

## v0.4.1 (2025-07-06)

### Fix

- add error handling for session loading

## v0.4.0 (2025-07-06)

### Feat

- implement resampling methods for combatting class imbalance

## v0.3.1 (2025-07-04)

### Fix

- **train**: add missing import to training.py

## v0.3.0 (2025-07-04)

### Feat

- bake training script into train subpackage

## v0.2.3 (2025-07-01)

### Fix

- use fixed track status label encoder for more robust cross evals

## v0.2.2 (2025-06-29)

### Fix

- add missing create_season_configs function

### Refactor

- remove convenience pipeline functions

## v0.2.1 (2025-06-28)

### Fix

- force bump

## v0.2.0 (2025-06-28)

### Feat

- update week 8 notebooks
- complete rewrite (unstable state)
- add script and notebook for working with aeon-toolkit preprocessing
- run week 8/toy_model.ipynb
- add dataset package for preprocessing and aggregating races across whole season
- working on data preprocessor

### Fix

- update __init__

### Refactor

- rename capstone to f1_etl
- break f1-etl main module into smaller files
