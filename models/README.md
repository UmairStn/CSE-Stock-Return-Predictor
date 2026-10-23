# Model artifacts

Generated XGBoost models and their metadata are written here by `cse-analyzer train`.
They are ignored by Git by default because binary/generated artifacts are not source code.

Each training run creates one model per horizon `h`:

- `cse_next_day_regressor.json` + `.metadata.json` (h = 1, the default)
- `cse_next_day_regressor_h7.json` + `.metadata_h7.json` (weekly)
- `cse_next_day_regressor_h30.json` + `.metadata_h30.json` (monthly)

The metadata files record the horizon, target definition, features, data split
sizes, symbols, hyperparameters, and evaluation metrics — plus the serving-safety
fields:

- `min_bars_required: 5` — the live backend must refuse with fewer completed bars.
- `feature_ranges` — per-feature `p1`/`p50`/`p99` over the training rows; the
  backend alarms (in the API response's `warnings`, never silently) when a live
  feature falls outside its training band.
- `feature_importances` — per-horizon feature weights for the report.
- `feature_ablation_report` — path to `reports/generated/feature_ablation_h*.json`,
  which holds each feature's held-out MAE delta when dropped.

Models are trained on features computed from each row's trailing 5-bar slice via
`build_feature_row` — the same function the serving backend runs on the live CSE
payload (`docs/serving-contract.md`).

To reproduce them, fetch data and run the training commands described in the root README.
