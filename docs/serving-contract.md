# Serving contract — live prediction against the CSE feed

This document is the contract `TradeVision-backend` implements for live
predictions. It is deliberately complete: the backend must be able to satisfy
every numbered rule below from this file alone.

**The one rule above all others:** the backend runs a *byte-for-byte copy* of
this repository's `src/cse_stock_analyzer/features.py` and never re-implements
feature math. The last drifting re-implementation divided RSI by 100, silently
serving values 100× off-distribution. Copy the file on every deploy; a startup
integrity check (module hash against the trainer repo at the same commit tag)
is the cheapest guard.

## 1. What the live feed provides

Observed CSE API price-history record (2026-10-23):

```json
{
  "h": 18.1, "l": 17.6, "o": null, "s": 108923744269, "q": 6308277,
  "p": 17.7, "c": null, "pc": null, "t": 1792348200000, "n": null, "id": 297
}
```

| Field | Meaning | Disposition |
|---|---|---|
| `p` | session last traded price | **Close** |
| `h` / `l` | session high / low | **High / Low** |
| `o` | open — **always null** | never read; `Open` is *defined* as `Close(t-1)` |
| `q` | session traded quantity | **Volume** (see §6 for the one pending check) |
| `t` | epoch timestamp (ms; seconds tolerated) | trade **Date** in Asia/Colombo (fixed UTC+05:30) |
| `s` | ambiguous (~10⁸ scale, possibly turnover/size) | **never read** |
| `c`, `pc`, `n`, `id` | null / record id / trade count | **never read** |

`features.cse_payload_to_ohlcv(payload)` implements this mapping — use it; do
not hand-roll the mapping in `prediction_engine.py`.

## 2. Ingestion rules (identical to training hygiene)

1. Reject (skip, never clamp) any bar where `p`, `h`, `l` are missing/non-finite
   or `not (l ≤ p ≤ h)`. A physically impossible bar is malformed data.
2. Collapse duplicate trade dates to the record with the latest `t`.
3. Sort ascending by trade date.
4. Drop today's **still-open session**: the model only ever predicts *from* a
   completed close. If the newest record's date equals "now" in Colombo, drop it
   (`drop_open_session=True` is the default; pass `False` only for offline
   parity/shadow tests).
5. Drop bars with `q ≤ 0` or missing `q` (same filter as training's
   `drop_non_trading_rows`).
6. If fewer than **5** completed bars survive, **refuse to predict** with
   `need ≥ 5 completed sessions`. Raise an error; never pad, guess, or use
   partial windows. There is no `MIN_USABLE_ROWS = 200` at serving — 200 is a
   training-only guard and must not appear in backend code paths.
7. Extra headroom: the feed currently returns ~23 records; the predictor uses
   the trailing 5. The remaining ~18 bars are headroom for §5's distribution
   checks — never an input to feature computation (the slice contract makes
   them mathematically irrelevant).

## 3. Prediction flow

```
CSE payload (~23 records/symbol)
  → cse_payload_to_ohlcv()          # §2 rules; refuses < 5 completed bars
  → featurize_history().dropna(FEATURE_COLUMNS)
  → last row = today's observation
  → model.predict(FEATURE_COLUMNS)  # per horizon 1, 7, 30
  → §5 tripwire per feature
  → API response
```

`prepare_ohlcv` inside the featurizer synthesizes `Open = Close.shift(1)` —
this is why training data was force-converted to the same convention (Yahoo's
real Open is discarded at training time). Do not "fix" the missing Open by
imputing intraday opens.

## 4. Feature math (reference — the code is authoritative)

Slice rows s0..s4 (oldest → newest), features describing s4. C = Close(s4):

| # | Feature | Formula |
|---|---|---|
| 1 | `Open_Pct` | `Close(s3)/C − 1` |
| 2 | `High_Pct` | `High(s4)/C − 1` |
| 3 | `Low_Pct` | `Low(s4)/C − 1` |
| 4 | `Rel_Volume` | `Volume(s4)/mean(Volume(s2..s4))` |
| 5 | `RSI_3` | Cutler equal-weight 3-day RSI over closes s1..s4, **scale 0–100** |
| 6 | `Return_Lag1` | `Close(s3)/Close(s2) − 1` |
| 7 | `Return_Lag2` | `Close(s2)/Close(s1) − 1` |
| 8 | `SMA_3_Pct` | `mean(Close(s2..s4))/C − 1` |
| 9 | `EMA_3_Pct` | `EMA_3(slice)/C − 1` |
| 10 | `MACD_Pct` | `(EMA_3 − EMA_5)/C`, both seeded at s0 |
| 11 | `MACD_Hist_Pct` | `(MACD − EMA_3(MACD))/C` |

All indicators recurse **only inside the slice**. Names and order are exactly
`FEATURE_COLUMNS`; the artifact schema guard in `load_model` rejects anything
else.

## 5. Tripwire — out-of-distribution alarms (never silent suppression)

Each horizon's metadata carries `"feature_ranges": {feature: {p1, p50, p99}}`
computed over that horizon's training rows. On every live prediction:

1. Compare each of the 11 feature values against its training band
   `[p1, p99]`.
2. Any breach appends to the response:

```json
"warnings": [
  {"feature": "Rel_Volume", "value": 7.4, "band": {"p1": 0.21, "p99": 4.9},
   "severity": "out_of_distribution"}
]
```

3. Also log the breach server-side. The prediction itself is **still returned**
   — the alarm adds context, it does not suppress. The alternative (quietly
   returning "garbage in, garbage out" numbers) is the failure mode that
   motivated this contract.
4. With ~18 headroom bars, features can additionally be computed for the last
   ~19 rows and the *share of recent breaches* monitored (single bars breach
   p1/p99 by definition 2% of the time; a rate > ~10% is signal).
5. `min_bars_required = 5` in metadata is validated on load; mismatched
   metadata ↔ artifact pairs fail fast.

## 6. One pending verification before go-live

Confirm `q` is **per-session** traded quantity, not cumulative or turnover
whenever a full multi-day payload is captured:

- If `q` ever *decreases* between consecutive records, it is per-session
  (cumulative counters can only rise) → contract confirmed.
- If `q` proves cumulative/turnover: `Rel_Volume` trains on ~1.0-valued stability
  and must be renegotiated — do not hide the finding.

Until then `Rel_Volume` is unit-invariant *by construction* (a within-slice
ratio of like quantities), so Yahoo-vs-CSE unit differences cannot poison the
model either way; only the cumulative-vs-per-day question genuinely matters.

## 7. Response schema

Unchanged keys plus optional `warnings`:

```json
{
  "symbol": "JKH.N0000",
  "observation_date": "2026-10-19",
  "horizons": {
    "1d":  {"predicted_percent": -0.05, "direction": "DOWN"},
    "7d":  {"predicted_percent": 1.2,   "direction": "UP"},
    "30d": {"predicted_percent": 4.7,   "direction": "UP"}
  },
  "warnings": []
}
```

`observation_date` is the date of the **last completed bar used** — if this is
stale (e.g. yesterday's session never arrived), the consumer sees it and the
staleness alarm fires when the gap exceeds one session.

## 8. Deployment checklist

- [ ] `services/features.py` replaced with this repo's file (byte-identical; hash-checked at startup)
- [ ] model + metadata file pairs loaded per horizon; schema guard active
- [ ] `cse_payload_to_ohlcv` used as the single mapping entry point
- [ ] tripwire reads `feature_ranges` from metadata; warnings surface in API + logs
- [ ] no `/100` scaling of RSI anywhere (search the backend for `100` in feature code)
- [ ] refusal path tested: 4-bar payload → clean error, not a fabricated prediction
- [ ] shadow run: backend features diffed against the training featurizer on an identical historical slice → exact match
- [ ] a deliberately fed >40% jump day triggers tripwire warnings and the API still responds
