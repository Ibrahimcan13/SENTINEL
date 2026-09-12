import os

os.environ["SKLEARN_ASSUME_FINITE"] = "true"
os.environ["PYTHONWARNINGS"] = "ignore"


import logging
import traceback
from datetime import datetime
import pandas as pd
import yaml

from analysis import (
    add_bollinger_bands,
    add_kalman_filter,
    add_rsi,
    calculate_average_true_range,
    calculate_moving_average,
    generate_signals,
)
from back_tester import run_backtest
from data_loader import fetch_market_data, save_data_to_parquet
from predictor import train_and_predict
from visualizer import plot_signals

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


def load_config(config_path: str = "config.yaml") -> dict:
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Failed to load config file ({config_path}): {e}")
        raise e


def get_valid_date(prompt: str) -> datetime:
    while True:
        date_str = input(prompt).strip()
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            return dt
        except ValueError:
            logging.error("Invalid format! Please use YYYY-MM-DD (e.g., 2026-01-01).")


def run_sentinel():
    print("           PROJECT SENTINEL             ")

    cfg = load_config("config.yaml")

    data_cfg = cfg.get("data", {})
    analysis_cfg = cfg.get("analysis", {})
    ml_cfg = cfg.get("machine_learning", {})
    bt_cfg = cfg.get("backtest", {})

    try:
        # Ticker Selection
        default_ticker = data_cfg.get("default_ticker", "RACE")
        user_ticker = input(f"Enter asset ticker [Default: {default_ticker}]: ").strip().upper()
        if not user_ticker:
            user_ticker = default_ticker

        initial_capital = bt_cfg.get("initial_capital", 1000.0)
        window_size = analysis_cfg.get("window_size", 20)
        rsi_window = analysis_cfg.get("rsi_window", 14)
        atr_window = analysis_cfg.get("atr_window", 14)
        bollinger_std = analysis_cfg.get("bollinger_std", 2.0)
        use_kalman = analysis_cfg.get("use_kalman", True)

        train_window = ml_cfg.get("train_window", 200)
        forecast_days = ml_cfg.get("forecast_days", 5)
        retrain_step = ml_cfg.get("retrain_step", 20)

        print("\n[Date Configuration]")
        today = datetime.now()

        while True:
            start_dt = get_valid_date("Enter Start Date (YYYY-MM-DD): ")
            end_dt = get_valid_date("Enter End Date (YYYY-MM-DD): ")

            if start_dt > today or end_dt > today:
                logging.warning(f"Dates cannot be in the future! Current System Date: {today.strftime('%Y-%m-%d')}\n")
                continue

            if start_dt >= end_dt:
                logging.warning("Start date must be BEFORE the end date!\n")
                continue

            days_difference = (end_dt - start_dt).days
            min_required = train_window + max(window_size, rsi_window) + 14
            if days_difference < min_required:
                logging.warning(
                    f"Date range is too short ({days_difference} days). Requires AT LEAST {min_required} days!\n")
                continue

            break

        start_date = start_dt.strftime("%Y-%m-%d")
        end_date = end_dt.strftime("%Y-%m-%d")

        logging.info(f"Fetching {user_ticker} data from {start_date} to {end_date}...")
        df = fetch_market_data(user_ticker, start_date, end_date)
        if df.empty:
            logging.error("Failed to fetch market data. Terminating.")
            return

        logging.info("Calculating technical indicators & Kalman Filter...")
        if use_kalman:
            df = add_kalman_filter(df)

        df = calculate_moving_average(df, window=window_size)
        df = add_rsi(df, window=rsi_window)
        df = calculate_average_true_range(df, window=atr_window)
        df = add_bollinger_bands(df, window=window_size, num_std=bollinger_std)

        logging.info("Generating Baseline Trading Signals...")
        try:
            df = generate_signals(df, window=window_size, use_kalman=use_kalman)
        except TypeError:
            df = generate_signals(df, window=window_size)

        logging.info("Executing Walk-Forward AI Training & Inference Pipeline...")
        df = train_and_predict(
            df,
            forecast_days=forecast_days,
            train_window=train_window,
            retrain_step=retrain_step
        )

        logging.info("Executing Backtest Engine with AI Probability Filtering...")
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

        print("\n          SENTINEL STATUS REPORT                  ")
        print(f"Target Asset       : {user_ticker}")
        print(f"Starting Capital   : ${initial_capital:.2f}")
        print(f"Latest Close Price : ${latest_price:.2f}")
        print(
            f"Analysis Pipeline  : Kalman ({use_kalman}) | SMA {window_size}d | RSI {rsi_window}d | ATR {atr_window}d")
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

            print(f"AI FORECAST ({forecast_days}-Day Horizon) : {direction}")
            print(f"Bullish Probability ({forecast_days}d ahead) : %{bullish_pct:.1f}")
            print(f"Bearish Probability ({forecast_days}d ahead) : %{bearish_pct:.1f}")
        else:
            print("AI FORECAST            : Insufficient data for prediction window.")

        logging.info("Rendering Plotly Interactive Dashboard...")
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

    except KeyboardInterrupt:
        logging.warning("Operation cancelled by user. Exiting safely.")
    except Exception as e:
        logging.error(f"Fatal error encountered: {e}")
        logging.debug(traceback.format_exc())


if __name__ == "__main__":
    run_sentinel()