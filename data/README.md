# Data directory

## data/raw/ — price history (not committed)

This folder holds the project's daily OHLCV price history: one CSV per ticker
(e.g. `JKH-N0000.CM.csv`), roughly **66,500 trading rows across ~53 CSE
securities** from **2021-08-11 onward**, ~4 MB total. The CSV files are
intentionally **ignored by Git** (see `.gitignore`) — market data may carry
redistribution restrictions, and generated data does not belong in version
control.

Generate or refresh the dataset with:

```bash
cse-analyzer fetch --period 5y
```

Two properties of this dataset worth knowing:

1. **Dead feed.** Yahoo's CSE feed began forward-filling ghost rows
   (Volume=0, unchanged prices) after mid-February 2026. These rows are
   harmless to the pipeline — `drop_non_trading_rows` removes them — but
   training also applies a default cutoff of 2026-01-31 as a second guard
   (see `CSE_TRAIN_END` in `.env.example`).
2. **Unadjusted prices.** Prices are downloaded with `auto_adjust=False`.
   Corporate actions (splits, scrip issues) appear as extreme daily returns;
   the training pipeline drops neighborhoods around returns above 40%.
3. **Open is synthesized.** Although the raw CSVs carry Yahoo's `Open`, the
   pipeline overwrites it with `Open(t) = Close(t-1)` everywhere, because the
   live CSE feed has no open price (`o: null`). Do not delete the column in
   re-fetches, but never treat it as a model input.

## Attribution

Market data originates from Yahoo Finance and the Colombo Stock Exchange and
remains subject to their terms. It is used here for educational and research
purposes only.

## data/processed/

Reserved for derived datasets (features, splits). Contents of this folder are
generated and ignored by Git.
