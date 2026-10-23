# Data and limitations

## Source and coverage

The downloader uses Yahoo Finance through `yfinance`. CSE coverage is uneven: some listings may be absent, delayed, stale, short, or forward-filled. The software skips unavailable histories rather than treating missing data as valid zero returns.

Before publishing this repository, review Yahoo Finance's terms and the Colombo Stock Exchange's data policies. Downloaded market data is ignored for future commits because the project's MIT license applies to source code, not automatically to third-party datasets.

## Symbol convention

Canonical project symbols use `TICKER.N0000`; Yahoo symbols generally use `TICKER-N0000.CM`. `normalize_symbol` accepts short symbols and Yahoo-form symbols, but this convention may not cover every CSE security class.

## Corporate actions

Prices are fetched with `auto_adjust=False` so values represent the returned unadjusted series. Splits, scrip dividends, and other actions can create false extreme returns. The 40% jump filter is a defensive heuristic, not a replacement for verified corporate-action data.

## The 5-bar slice and live-feed constraints

The model's features are computed from exactly the trailing **5 completed daily bars** of each row (see [methodology](methodology.md)). This is dictated by the serving data source rather than preference: the live CSE API used at prediction time exposes only about **23 daily bars per symbol** with the fields `h`, `l`, `p` (last price), `q` (quantity), and `t` (epoch-milliseconds timestamp). There is **no open price** (`o` is always `null`), so `Open` is *defined* as the previous close on both the training and serving sides, and Yahoo's real open column is discarded at ingestion.

Two data quirks this makes harmless:

- Yahoo stores prices with float32 noise (`22.600000381469727`); features are computed in float64, and every feature is a ratio within one slice, so a stock's price level cancels out.
- The CSE payload field `s` (an ambiguous ~10⁸-scale number, possibly turnover) is never read; only `h/l/p/q/t` are mapped.

Consequence: any row's features depend on at most the five closes/highs/lows/volumes around it. Information the live feed cannot produce (longer lookbacks) is excluded from training *by construction*, not by convention.

## Modeling limitations

- The universe emphasizes relatively liquid counters and is not the entire exchange.
- Technical indicators are backward-looking and often weak predictors.
- Thin trading and zero-volume periods can make a "next day" target mean the next observed trading session, not necessarily the next calendar or exchange session.
- Survivorship bias and symbol-history changes are not explicitly corrected.
- No fundamentals, filings, dividends, index movements, FX, interest rates, or news are modeled.
- A model trained in one market regime can degrade without warning.
- Prediction uncertainty and calibration are not yet estimated.

## Evaluation limitations

The per-asset temporal holdout is more realistic than a random row split but is not a full trading backtest. It does not account for:

- bid/ask spreads and slippage,
- brokerage fees and taxes,
- CSE price bands,
- order size and market impact,
- trading halts or settlement constraints,
- look-ahead introduced by revised data from a provider,
- portfolio construction and risk limits.

Use a global-date walk-forward design and cost-aware simulation before considering any strategy experiment.

## Interpretation

The output is an estimated cumulative return over the requested horizon (1, 7, or 30 trading sessions). It is not a fair-value estimate, guaranteed movement, confidence score, or recommendation. A positive prediction may be economically insignificant after costs, even when its sign is correct. Longer horizons inherit every short-horizon caveat plus overlapping-label optimism in their metrics (see [methodology](methodology.md)).
