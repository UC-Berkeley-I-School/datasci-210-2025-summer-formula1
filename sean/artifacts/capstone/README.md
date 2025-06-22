
# The `capstone` package

This package contains bits and pieces from my notebook experiments that I thought might be helpful for others.

## Installation

The project is managed with `uv` but you can just use `pip` if that is preferable.

Install:

- From Source...
  ```bash
  uv pip install -e .
  ```
- From Wheel...
  ```bash
  uv build
  uv pip install dist/capstone-0.1.0-py3-none-any.whl
  ```

Verify:

```bash
uv pip list | grep capstone
```

## Usage

There are two sub-packages: `preprocessor` and `aggregator`.

The `preprocessor` package contains code for preprocessing FastF1 sessions into temporal datasets. It will aggregate all telemetry for a given race into one dataframe, split the samples into temporal intervals, and balance the dataset based on a balance method and set of features that you specify.

Here is the gist of how you use it:

```python
import fastf1
from capstone import PreprocessorConfig, BaseFeatures, F1DatasetPreprocessor

session = fastf1.get_session(2024, "São Paulo Grand Prix", "R")
session.load()

config = PreprocessorConfig(
    interval_seconds=1.0,
    balance_features=[BaseFeatures.DRIVER, BaseFeatures.COMPOUND],
    balance_method="remove_insufficient",
    target_samples=2000,
    include_track_status=True,
    include_event_info=True
)

preprocessor = F1DatasetPreprocessor(config)
result = preprocessor.process_dataset(session)
```

The `aggregator` package takes the preprocessor to the next level. It makes it possible to preprocess large quantities of race datasets (or FastF1 sessions) concurrently. For example, you can create a temporally aligned, balanced dataset containing telemetry from every race in a given season:

```python
from capstone import create_aggregator

aggregator = create_aggregator(preprocessor_config=None, max_workers=12)
summary = aggregator.add_season(2024)
```

Note that the aggregator can take a long to complete. For reference, it took me about 14 minutes to process all races from 2024 using 4 workers on a MacBook Pro 14" with an M1 Pro.