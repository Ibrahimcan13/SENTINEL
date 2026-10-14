import os

os.environ["SKLEARN_ASSUME_FINITE"] = "true"
os.environ["PYTHONWARNINGS"] = "ignore"

from datetime import datetime, timedelta
import pytz  
import logging
import sys
import traceback
import pandas as pd
import yaml

from analysis import extract_features
from back_tester import run_backtest
from data_loader import fetch_market_data, save_data_to_parquet
from predictor import train_and_predict
from strategy import generate_hybrid_signals  
from visualizer import plot_signals

logger = logging.getLogger("Sentinel")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    file_handler = logging.FileHandler("sentinel_system.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] | %(filename)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def load_config(config_path: str = "config.yaml") -> dict:
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.critical(f"Failed to load config file ({config_path}). System cannot start.", exc_info=True)
        raise e


def get_optional_date(prompt: str, default_dt: datetime) -> datetime:
    default_str = default_dt.strftime("%Y-%m-%d")
    while True:
        user_input = input(f"{prompt} [Default: {default_str}]: ").strip()
        if not user_input:
            return default_dt
        try:
            return datetime.strptime(user_input, "%Y-%m-%d")
        except ValueError:
            logger.error("Invalid format! Please use YYYY-MM-DD (e.g., 2026-01-01).")


def run_sentinel():
    print("                    PROJECT SENTINEL                    ")

    cfg = load_config("config.yaml")

    data_cfg = cfg.get("data", {})
    analysis_cfg = cfg.get("analysis", {})
    ml_cfg = cfg.get("machine_learning", {})
    bt_cfg = cfg.get("backtest", {})
    strategy_cfg = cfg.get("strategy", {})

    try:
        default_ticker = data_cfg.get("default_ticker", "RACE")
        user_ticker = input(f"Enter asset ticker [Default: {default_ticker}]: ").strip().upper()
        if not user_ticker:
            user_ticker = default_ticker

        initial_capital = bt_cfg.get("initial_capital", 1000.0)
        window_size = analysis_cfg.get("window_size", 20)
        rsi_window = analysis_cfg.get("rsi_window", 14)

        train_window = ml_cfg.get("train_window", 200)
        forecast_days = ml_cfg.get("forecast_days", 5)
        retrain_step = ml_cfg.get("retrain_step", 20)

        ml_hyperparameters = ml_cfg.get("hyperparameters", {})

        print("\n[Date Configuration] (Press Enter to use automatic UTC-safe window)")

        ny_tz = pytz.timezone("America/New_York")
        now_ny = datetime.now(ny_tz)

        today = now_ny.date()
        default_start = today - timedelta(days=365)

        while True:
            start_dt = get_optional_date("Enter Start Date (YYYY-MM-DD)", datetime.combine(default_start, datetime.min.time()))
            end_dt = get_optional_date("Enter End Date (YYYY-MM-DD)", datetime.combine(today, datetime.min.time()))

            if start_dt.date() > today or end_dt.date() > today:
                logger.warning(f"Dates cannot be in the future! NY Market Date: {today}\n")
                continue

            if start_dt >= end_dt:
                logger.warning("Start date must be BEFORE the end date!\n")
                continue

            days_difference = (end_dt - start_dt).days
            min_required = train_window + max(window_size, rsi_window) + 14
            if days_difference < min_required:
                logger.warning(
                    f"Date range is too short ({days_difference} days). Requires AT LEAST {min_required} days!\n"
                )
                continue

            break

        start_date = start_dt.strftime("%Y-%m-%d")
        end_date = end_dt.strftime("%Y-%m-%d")

        logger.info(f"Fetching {user_ticker} data from {start_date} to {end_date}...")
        df = fetch_market_data(user_ticker, start_date=start_date, end_date=end_date)
        
        if df.empty:
            logger.error(f"Failed to fetch market data for {user_ticker}. Terminating pipeline.")
            return

        logger.info("Extracting ML Features & Technical Indicators (Kalman, RSI, ATR)...")
        df = extract_features(df, window=window_size)

        logger.info("Executing Walk-Forward AI Training & Inference Pipeline...")
        df = train_and_predict(
            df,
            forecast_days=forecast_days,
            train_window=train_window,
            retrain_step=retrain_step,
            hyperparameters=ml_hyperparameters
        )

        logger.info("Generating Hybrid Trading Signals (AI Probability + Trend Filter)...")
        df = generate_hybrid_signals(
            df,
            ai_upper_threshold=strategy_cfg.get("ai_upper_threshold", 0.55),
            ai_lower_threshold=strategy_cfg.get("ai_lower_threshold", 0.45),
            allow_short=strategy_cfg.get("allow_short", True),
            confirmation_days=strategy_cfg.get("confirmation_days", 1)
        )

        logger.info("Executing High-Performance Numba Backtest Engine...")
        df, trade_log, metrics = run_backtest(
            df,
            initial_capital=initial_capital,
            risk_per_trade=bt_cfg.get("risk_per_trade", 0.02),
            commission_rate=bt_cfg.get("commission_rate", 0.001),
            slippage_rate=bt_cfg.get("slippage_rate", 0.0005),
            use_atr_stop=bt_cfg.get("use_atr_stop", True),
            atr_multiplier=bt_cfg.get("atr_multiplier", 2.0),
            ai_exit_threshold=bt_cfg.get("ai_exit_threshold", 0.50)
        )

        save_data_to_parquet(df, ticker=user_ticker)

        latest_prob = (
            df["AI_Probability"].iloc[-1]
            if "AI_Probability" in df.columns and not pd.isna(df["AI_Probability"].iloc[-1])
            else None
        )
        latest_price = df["Close"].iloc[-1]

        print("\n            SENTINEL STATUS REPORT             ")
        print(f"Target Asset       : {user_ticker}")
        print(f"Starting Capital   : ${initial_capital:.2f}")
        print(f"Latest Close Price : ${latest_price:.2f}")
        print(f"Analysis Pipeline  : Kalman Filter | SMA {window_size}d | Robust Features")
        print(f"Total Trades       : {metrics.get('total_trades', 0)}")
        print(f"Winning Trades     : {metrics.get('winning_trades', 0)}")
        print(f"Win Rate           : %{metrics.get('win_rate', 0.0):.1f}")
        print(f"Max Drawdown       : %{metrics.get('max_drawdown', 0.0):.2f}")
        print(f"Sharpe Ratio       : {metrics.get('sharpe_ratio', 0.0):.2f}")
        print(f"Avg Holding Period : {metrics.get('avg_duration_days', 0.0):.1f} days")
        print(f"Net Realized PnL   : ${metrics.get('total_pnl', 0.0):.2f}")

        if latest_prob is not None:
            bullish_pct = latest_prob * 100
            bearish_pct = (1 - latest_prob) * 100
            direction = "BULLISH" if bullish_pct >= 50 else "BEARISH"

            print(f"\nAI FORECAST ({forecast_days}-Day Horizon) : {direction}")
            print(f"Bullish Probability ({forecast_days}d ahead) : %{bullish_pct:.1f}")
            print(f"Bearish Probability ({forecast_days}d ahead) : %{bearish_pct:.1f}\n")
        else:
            print("\nAI FORECAST            : Insufficient data for prediction window.\n")
 
        logger.info("Rendering Plotly Interactive Dashboard...")
        save_filename = f"{user_ticker}_sentinel_report.html"
        plot_signals(
            df,
            ticker=user_ticker,
            window=window_size,
            metrics=metrics,
            trade_log=trade_log,
            train_window=train_window,
            initial_capital=initial_capital,
            save_path=save_filename
        )
        
        logger.info("Pipeline execution completed successfully.")

    except KeyboardInterrupt:
        logger.warning("Operation cancelled by user (KeyboardInterrupt). Exiting safely.")
    except Exception as e:
        logger.critical("A fatal error occurred during pipeline execution.", exc_info=True)


if __name__ == "__main__":
    run_sentinel()