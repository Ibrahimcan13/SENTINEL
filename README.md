# SENTINEL 

SENTINEL is a high-performance Python-based quantitative trading infrastructure designed for automated backtesting and market analysis. It leverages modular architecture, Numba vectorization, and Parquet data caching for maximum execution speed.

##  Key Features

* **WSL/Linux Optimized:** Native Linux environment execution for enhanced throughput and system efficiency.
* **Numba Acceleration:** Low-latency backtesting loops and technical indicator calculations via JIT compilation.
* **Parquet Data Engine:** Optimized high-volume data ingestion and caching routines.
* **Modular Architecture:** Fully decoupled components for data loading, signal prediction, execution backtesting, and visualization.

##  Repository Structure

* `main.py` - Core execution entry point.
* `data_loader.py` - Data retrieval and Parquet caching engine.
* `back_tester.py` - Vectorized and Numba-accelerated backtest engine.
* `predictor.py` - Strategy signal generation and algorithmic model layer.
* `analysis.py` - Portfolio metrics, drawdown, and risk calculation module.
* `visualizer.py` - Automated plotting and statistical report generator.
* `config.yaml` - System settings, data parameters, and strategy options.



