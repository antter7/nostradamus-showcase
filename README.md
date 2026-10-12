# Nostradamus — ML and LLM forecasting for prediction markets (showcase)

Nostradamus is my personal autonomous trading system for [Polymarket](https://polymarket.com) prediction markets.
It runs in production with real money and is controlled through Telegram. This repository is a curated excerpt
of the research and modelling code. Order execution, wallet handling, deployment, live configs and data stay private.

## What the full system does

- Scans active markets, scores candidates with several independent strategies and sizes positions by risk.
- New strategies start in **shadow mode**: they log signals without trading. A strategy is promoted to live only after it
  passes forward-validation gates, and it falls back to shadow automatically when its record drifts.
- Every forecast is stored as an auditable snapshot: prompt, sources, answer, rejection reasons, and the trade it led to.
- A daily validation report tracks win rate with Wilson intervals, breakeven, ROI, live P&L and model drift.

## Modules in this repository

| File | What it shows |
| --- | --- |
| `grounded_forecast.py` | LLM probability forecasts grounded in fresh web search, market rules and exchange quotes |
| `hybrid_ml_strategy.py` | LightGBM ranking model on top of LLM forecasts, with drift monitoring |
| `category_classifier.py` | Rule-based market topic classifier used as a model feature |
| `post_count_strategy.py` | Count-forecast model for "how many posts" markets, Wilson-bound sizing, kill switch |
| `post_count_research.py` | Point-in-time backtest that replays the exact live rule on historical hourly prices |
| `test_smoke.py` | Runnable check that the modules import standalone and the core math holds |

### LLM forecasting with grounding (`grounded_forecast.py`)

- Builds context for the model: full market rules, current UTC time, hours left, dated web sources and, for crypto
  questions, a live exchange quote with 7-day volatility.
- The search widens from a week to a month to unlimited until it finds dated, relevant publications.
- **Answer validation against hallucinations:** the model must quote the resolution rule and its sources. The code checks
  every quote against the fetched text, and invented quotes are rejected rather than replaced with a number.
- Probability and confidence are separate: weak evidence forces low confidence and blocks entry, but keeps the estimate.
- Fetching is SSRF-safe: private and local addresses are refused.

### ML ranking on LLM forecasts (`hybrid_ml_strategy.py`)

- First stage: an LLM probability for each market. Second stage: a LightGBM model ranks candidates on the LLM edge,
  price, time to resolution, volume, market topic and TF-IDF text features of the question.
- Trains only on the first forecast per market, to avoid selection bias from re-asked forecasts.
- **Drift monitoring:** population stability index (PSI) on numeric and categorical features plus AUC of the old model on
  fresh resolved data against a price-only baseline, reported automatically after every retrain.

### Count model and risk control (`post_count_strategy.py`, `post_count_research.py`)

- Forecasts how many posts an account will publish in a window: the mean of recent exact counts, with an empirical
  log-ratio error distribution and a kernel over it to price every bucket.
- Position size: half Kelly on the **lower Wilson bound** of the historical win rate, net of fees.
- **Kill switch and drift test:** a two-proportion z-test compares the recent window with earlier history; a significant drop
  cuts the stake, and an upper bound below breakeven moves the series back to shadow.
- The backtest replays the same `top_bucket_signal()` the live scanner uses, so research and production cannot diverge.

## Results

Backtest of the count model, January–October 2026, 121 windows: 35 trades, win rate 77% against a 53% breakeven,
ROI +48%, positive in every quarter. Parameters were fitted on the same sample, so this is a candidate in shadow
validation, not a proven edge. Exact entry thresholds and live configs are intentionally left out.

## Stack

Python, pandas, scikit-learn, LightGBM, SQLite, requests, DeepSeek API, Polymarket Gamma/CLOB APIs, Telegram bot, systemd on a VPS.

## Run the check

```bash
pip install -r requirements.txt
pytest -q test_smoke.py
```

The modules need private data (`data/`) and config (`config/strategy.json`) to run end to end; they are not included.
