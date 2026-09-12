# Project Sentinel — v2.0.1

Project Sentinel is a modular, AI-enhanced quantitative market analysis pipeline built in Python. The system retrieves historical market data, computes technical indicators, trains a Machine Learning model (`RandomForestClassifier`) to predict short-term price movements, and executes mark-to-market backtesting alongside a multi-panel visual dashboard.

---

## Technical Overview

* **Data Ingestion (`data_loader.py`):** Fetches real-time and historical market data using `yfinance`, handles multi-index ticker structures, and manages time-stamped CSV exports.
* **Machine Learning Engine (`predictor.py`):** Utilizes an 80/20 chronological train/test split on historical price, volatility, and indicator metrics to project 5-day directional probabilities (`AI_Probability`).
* **Signal Logic & Backtesting (`analysis.py`):** Integrates technical indicators (RSI, Moving Averages, Bollinger Bands) with AI probability thresholds ($P > 0.50$) to evaluate trading conditions. Calculates strategy PnL with commission deductions, auto-closes open positions, and tracks overall performance metrics (Equity Curve, Win Rate, Max Drawdown).
* **Visualization Matrix (`visualizer.py`):** Generates a unified 5-panel dashboard displaying price dynamics, AI probability metrics, RSI levels, volume metrics, and real-time equity curves.
* **Pipeline Orchestration (`main.py`):** Handles parameter input validation, time-series continuity, dynamic window adjustments, and execution control across all sub-modules.

---

## Architecture Flow

```text
Data Ingestion (data_loader.py)
       │
       ▼
Feature Engineering & Technical Analysis (analysis.py)
       │
       ▼
Machine Learning Training & Prediction (predictor.py)
       │
       ▼
Signal Generation & Backtest Execution (analysis.py)
       │
       ├───> CSV Export (Timestamped)
       └───> 5-Panel Dashboard (visualizer.py)
