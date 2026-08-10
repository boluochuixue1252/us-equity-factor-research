# Large-Scale U.S. Equity Factor and Signal Research

A Python research pipeline for evaluating technical and market-regime features across a broad U.S. equity universe. The latest research output covered 4,372 unique tickers and approximately 18,800 signal observations.

> Historical research only. Results are not investment advice and do not imply future performance.

## Research question

Can trend-repair and momentum signals be described more reliably by combining stock-level price/volume features with the broader SPY market regime?

## Pipeline

```text
Daily OHLCV files (one ticker per file)
                |
                v
       Schema and history checks
                |
                v
      Signal detection and labelling
                |
        +-------+--------+
        v                v
 Stock-level factors   SPY regime factors
        |                |
        +-------+--------+
                v
  Forward-return and exit evaluation
                |
                v
      Research-ready merged dataset
```

## What the code does

- Scans a directory of per-ticker daily OHLCV files.
- Computes signal dates using predefined trend and momentum rules.
- Generates forward 10-day return, maximum-gain, and drawdown labels.
- Evaluates an MA14-buffer exit rule and reports return distributions and win rate.
- Engineers more than 70 stock and market-context fields across trend, momentum, volume, volatility, breakout, and repair-state families.
- Merges SPY market features by signal date, including moving-average state, recent returns, regime, and a rule-based risk score.
- Produces a single research-ready CSV for downstream attribution and modelling.

See [the factor dictionary](examples/factor_dictionary.md) for the factor families.

## Repository structure

```text
.
├── src/equity_factor_pipeline.py  # signal, factors, labels, SPY merge
├── src/update_universe.py         # optional Polygon universe metadata pull
├── examples/factor_dictionary.md
├── .env.example
└── requirements.txt
```

## Data contract

Daily input files are expected to provide:

```text
date, open, high, low, close, volume
```

Place one CSV per ticker in `data/daily/` and a compatible `SPY.csv` in `data/market/`. Raw licensed data and full research outputs are excluded from GitHub.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python src/equity_factor_pipeline.py
```

The universe updater requires a Polygon API key supplied through the environment:

```bash
export POLYGON_API_KEY="your_key_here"
python src/update_universe.py
```

## Research discipline

- Signal definitions are separated from forward labels to reduce accidental leakage.
- Stock-level features use only information available on or before the signal date.
- SPY features are merged by the corresponding signal date.
- Raw observations are retained so performance can be sliced by factor and market regime.

## Limitations and next steps

- The current pipeline is event-based research, not a portfolio backtester.
- Delisted-symbol coverage, survivorship bias, corporate actions, and point-in-time universe membership require additional controls.
- Transaction costs, liquidity constraints, exposure limits, and portfolio construction are not yet modelled.
- Next steps: add automated tests, point-in-time universe controls, cross-sectional factor attribution, and walk-forward portfolio evaluation.

## Skills demonstrated

Python, Pandas, large-universe data processing, feature engineering, forward labelling, regime analysis, data-quality controls, and transparent documentation of research limitations.
