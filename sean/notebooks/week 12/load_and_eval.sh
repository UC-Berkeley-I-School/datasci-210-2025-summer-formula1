uv run python load_and_eval.py \
  --model-path /Users/seansica/Documents/Development/mids/capstone/datasci-210-2025-summer-formula1/deliverables/f1-prediction-system/models/20250722_222236_windows_100_horizon_10_rocket_ridge_weighted_driver1/my_model.pkl \
  --driver 1 \
  --window-size 100 \
  --prediction-horizon 10 \
  --model-name rocket_rf_driver1