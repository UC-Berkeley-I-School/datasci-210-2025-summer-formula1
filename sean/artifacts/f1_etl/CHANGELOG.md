## v0.7.0 (2025-08-04)

### Feat

- **training**: extract validation probabilities (y_proba) if available
- **ssl**: add support for hosting via HTTPS with LetsEncrypt and Nginx
- **models**: add metadata for final models
- **models**: add final models with git lfs
- **database**: add utility scripts for db clients
- **database**: add support for writing raw features
- **database**: functioning but not saving raw telemetry yet
- **database**: push almost-working database project
- add support for training on PCA features
- **deliverables**: add boilerplate

### Fix

- **ssl**: add files for initial bootstrap for letsencrypt
- **webapp**: add better logging and error handling
- **model_service**: switch to nightly builds for aeon and imbalanced
- **model_service**: load correct model for each driver
- **model_service**: load models from correct path
- **model_service**: restore app export with uvicorn execution in script mode
- **compose**: add a new centrally located Docker Compose manifest
- **model_manager.py**: make the model manager portable
- **api_v1.py**: inify environment variables across services
- **load_data.py**: respect PGHOST in the database ETL script
- **data_config.py**: revert DB_CONFIG to original settings
- **db_client**: minor query bug
- **database**: small schema tweaks
- **database**: stop requiring requirements.txt

### Refactor

- **webapp**: get webapp working in docker
- **model_service**: overhaul fastapi app and container image to work with docker compose
- **99-load-data.sh**: add robustness for database connection

### Perf

- **model_service**: load models at startup rather than on-demand

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
