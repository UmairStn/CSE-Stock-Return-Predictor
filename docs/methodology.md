# Methodology

## Research question

Can one model learn short-horizon technical patterns shared across a diverse set of Colombo Stock Exchange securities and estimate the cumulative close-to-close return over the next 1, 7, or 30 trading sessions?

The implementation uses gradient-boosted trees (`XGBRegressor`) because they capture nonlinear interactions, require little feature scaling, and work well on tabular data. One model is trained per horizon; the features are identical, only the label window differs. This choice does not establish that the market is predictably profitable.

## Data pipeline

1. Convert canonical CSE symbols such as `JKH.N0000` to Yahoo's `JKH-N0000.CM` format.
2. Download daily unadjusted High, Low, Close, and Volume bars.
3. Reject empty histories, invalid schemas, and histories shorter than 200 rows.
4. Sort by date and remove rows with non-positive volume. Yahoo can forward-fill inactive or unavailable CSE observations; retaining these would create artificial zero returns.
5. Synthesize the Open column: `Open(t) = Close(t-1)`. The live CSE feed used at prediction time has no open price (`o` is always `null`), so the model must never learn anything that serving cannot reproduce — training data is force-converted to the same convention, and Yahoo's real Open is discarded.
6. Featurize every completed bar from *exactly its trailing 5-bar slice* (see below), using the same `build_feature_row` the serving backend runs.
7. Compute the target as the cumulative close-to-close return over the next `h` trading rows: `Close(t+h) / Close(t) - 1`.
8. Remove rows affected by an absolute daily return above 40% anywhere in their own 5-bar slice, plus rows whose target window (the next `h` rows) contains a jump.
9. Drop warm-up rows with missing features (the first 4 rows per security).

## Feature definitions

**The 5-bar slice contract.** Every feature of row `t` is computed from only the trailing five completed bars `s0..s4` (s4 = current bar). Recursive indicators are seeded *inside* the slice, so values cannot depend on how much history the caller holds. The serving backend runs the identical module (`features.py`, copied byte-for-byte into `TradeVision-backend`), which makes train-time and live-time features equal by construction. This exists because the live CSE feed exposes only about 23 daily bars with no open price — the previous EMA-12/26 MACD, the 20-session volume mean, and full-history EMA warm-up were impossible at serving time.

Per bar `t` with slice closes `s0..s4` (C = Close(t) = s4):

- `Open_Pct = Close(s3) / C - 1` (Open is *defined* as the previous close)
- `High_Pct = High(s4) / C - 1`
- `Low_Pct = Low(s4) / C - 1`
- `Rel_Volume = Volume(s4) / mean(Volume(s2..s4))` — 3-bar window including the current bar; the ratio makes it invariant to quantity units
- `RSI_3`: three-session Cutler-style RSI (equal-weight SMA of gains/losses) over closes `s1..s4`, scale 0–100
- `Return_Lag1 = Close(s3) / Close(s2) - 1`
- `Return_Lag2 = Close(s2) / Close(s1) - 1`
- `SMA_3_Pct = mean(Close(s2..s4)) / C - 1`
- `EMA_3_Pct = EMA_3(closes s0..s4) / C - 1`
- `MACD_Pct = (EMA_3 - EMA_5) / C`, both EMAs seeded at `s0`
- `MACD_Hist_Pct = (MACD - EMA_3(MACD)) / C`, signal computed from the slice's MACD series

Normalization allows securities at different nominal price and volume levels to share one model. Company identity is intentionally absent.

## Target and contamination controls

`Target_Return(t, h) = Close(t + h) / Close(t) - 1` is the cumulative return over the next `h` trading rows. The final `h` rows per security have no complete target and are excluded.

Unadjusted corporate actions can resemble extraordinary market returns. A jump contaminates:

- every row whose own 5-bar slice contains the jump bar (features computed on those bars are skewed),
- rows whose target window (the next `h` rows) contains the jump day (labels are untrustworthy),

The pipeline removes these neighbourhoods with a configurable-in-code 40% threshold; for multi-day horizons the target-window check widens to the full horizon. A production system should prefer authoritative corporate-action adjustments rather than this heuristic.

## Validation

Each stock is prepared and split independently: its earliest 80% becomes training data and its latest 20% becomes test data. Assets are concatenated only after this split. This avoids a random row split and guarantees that a stock's own test observations follow its training observations.

**Label embargo.** The last `h` rows of each training slice are dropped because their target windows reach into the test period — without the embargo, test-period closes would appear in training labels.

**Overlapping labels.** For `h > 1`, consecutive targets share most of their window (two adjacent 7-day targets share 6 of 7 days). Test rows are therefore highly correlated with each other, which makes metrics look more stable than the effective sample size justifies. The zero-return baseline is reported per horizon so the model is always compared against a like-for-like naïve alternative; a purged walk-forward design would be stronger still.

There remains a cross-sectional calendar caveat: because assets have unequal histories, one asset's test dates can overlap another asset's training dates. The model contains no direct cross-asset features, but a strict global-date or walk-forward split would be stronger for production-grade research.

Reported metrics are:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- R²
- Directional accuracy
- Zero-return baseline MAE

Directional accuracy alone is not profitability. Evaluation does not currently model spreads, fees, liquidity, execution delay, or position sizing.

## Reproducibility

Use `--train-end YYYY-MM-DD` to freeze the information set and `--horizon h` to select the label window. Every trained model receives adjacent JSON metadata containing its horizon, target definition, features, hyperparameters, assets, row counts, cutoff, creation time, and metrics — plus the serving-safety additions: `min_bars_required` (5), `feature_engineering` (the 5-bar slice contract), per-feature `feature_ranges` (`p1`/`p50`/`p99` over the training rows, used as out-of-distribution tripwire bands when serving), and `feature_importances`. Each horizon also writes `reports/generated/feature_ablation_h{h}.json`, which refits the model once per dropped feature and records each feature's held-out MAE delta. The loader rejects a model whose embedded feature names differ from the application's feature schema.

## Serving contract

The prediction-side contract (CSE payload mapping, feature formulas, tripwire rules) lives in `docs/serving-contract.md` and is implemented by `TradeVision-backend` running a byte-identical copy of `features.py`.
