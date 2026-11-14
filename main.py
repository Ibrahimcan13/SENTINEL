import os

os.environ["SKLEARN_ASSUME_FINITE"] = "true"
os.environ["PYTHONWARNINGS"] = "ignore"

from datetime import datetime, timedelta
import logging
import sys
import pandas as pd
import pytz
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


def parse_tickers(raw_input: str, default_ticker: str) -> list[str]:
    target = raw_input if raw_input else default_ticker
    seen = set()
    tickers = []
    for token in target.replace(",", " ").split():
        clean = token.strip().upper()
        if clean and clean not in seen:
            seen.add(clean)
            tickers.append(clean)
    return tickers


def process_single_asset(ticker: str, df: pd.DataFrame, cfg: dict) -> dict | None:
    data_cfg = cfg.get("data", {})
    analysis_cfg = cfg.get("analysis", {})
    ml_cfg = cfg.get("machine_learning", {})
    strategy_cfg = cfg.get("strategy", {})
    bt_cfg = cfg.get("backtest", {})

    interval = data_cfg.get("interval", "1d")
    window_size = analysis_cfg.get("window_size", 20)
    rsi_window = analysis_cfg.get("rsi_window", 14)
    momentum_window = analysis_cfg.get("momentum_window", 10)

    train_window = ml_cfg.get("train_window", 200)
    forecast_days = ml_cfg.get("forecast_days", 5)
    retrain_step = ml_cfg.get("retrain_step", 20)
    pt_multiplier = ml_cfg.get("pt_multiplier", 2.0)
    sl_multiplier = ml_cfg.get("sl_multiplier", 1.0)
    ml_hyperparameters = ml_cfg.get("hyperparameters", {})

    initial_capital = bt_cfg.get("initial_capital", 1000.0)

    min_bars = train_window + max(window_size, rsi_window, momentum_window) + forecast_days
    if len(df) < min_bars:
        logger.warning(
            f"[{ticker}] Insufficient bars ({len(df)} available, {min_bars} required). Skipping asset."
        )
        return None

    logger.info(f"[{ticker}] Extracting ML Features & Technical Indicators (Interval: {interval})...")
    df = extract_features(df, window=window_size, momentum_window=momentum_window)

    logger.info(f"[{ticker}] Executing Walk-Forward AI Training & Inference Pipeline...")
    df = train_and_predict(
        df,
        forecast_days=forecast_days,
        train_window=train_window,
        retrain_step=retrain_step,
        pt_multiplier=pt_multiplier,
        sl_multiplier=sl_multiplier,
        hyperparameters=ml_hyperparameters,
    )

    logger.info(f"[{ticker}] Generating Hybrid Trading Signals (Dynamic Volatility Thresholds)...")
    df = generate_hybrid_signals(
        df,
        ai_upper_threshold=strategy_cfg.get("ai_upper_threshold", 0.55),
        ai_lower_threshold=strategy_cfg.get("ai_lower_threshold", 0.45),
        allow_short=strategy_cfg.get("allow_short", True),
        confirmation_days=strategy_cfg.get("confirmation_days", 1),
        use_dynamic_thresholds=strategy_cfg.get("use_dynamic_thresholds", True),
        vol_sensitivity=strategy_cfg.get("vol_sensitivity", 0.05),
    )

    logger.info(f"[{ticker}] Executing High-Performance Numba Backtest Engine...")
    df, trade_log, metrics = run_backtest(
        df,
        initial_capital=initial_capital,
        risk_per_trade=bt_cfg.get("risk_per_trade", 0.02),
        commission_rate=bt_cfg.get("commission_rate", 0.001),
        slippage_rate=bt_cfg.get("slippage_rate", 0.0005),
        use_atr_stop=bt_cfg.get("use_atr_stop", True),
        atr_multiplier=bt_cfg.get("atr_multiplier", 2.0),
        ai_exit_threshold=bt_cfg.get("ai_exit_threshold", 0.50),
        max_vol_participation=bt_cfg.get("max_vol_participation", 0.10),
    )

    save_data_to_parquet(df, ticker=f"{ticker}_{interval}")

    latest_prob = (
        df["AI_Probability"].iloc[-1]
        if "AI_Probability" in df.columns and not pd.isna(df["AI_Probability"].iloc[-1])
        else None
    )
    latest_price = df["Close"].iloc[-1]

    print(f"\n{'=' * 18} SENTINEL REPORT: {ticker} ({interval}) {'=' * 18}")
    print(f"Starting Capital   : ${initial_capital:.2f}")
    print(f"Latest Close Price : ${latest_price:.2f}")
    print(f"Analysis Pipeline  : Kalman | SMA {window_size} | Momentum {momentum_window} | Robust Features")
    print(f"Total Trades       : {metrics.get('total_trades', 0)}")
    print(f"Winning Trades     : {metrics.get('winning_trades', 0)}")
    print(f"Win Rate           : %{metrics.get('win_rate', 0.0):.1f}")
    print(f"Max Drawdown       : %{metrics.get('max_drawdown', 0.0):.2f}")
    print(f"Sharpe Ratio       : {metrics.get('sharpe_ratio', 0.0):.2f}")
    print(f"Profit Factor      : {metrics.get('profit_factor', 0.0):.2f}")
    print(f"Avg Holding Period : {metrics.get('avg_duration_days', 0.0):.1f} days")
    print(f"Net Realized PnL   : ${metrics.get('total_pnl', 0.0):.2f}")

    forecast_str = "N/A"
    if latest_prob is not None:
        bullish_pct = latest_prob * 100
        bearish_pct = (1 - latest_prob) * 100
        direction = "BULLISH" if bullish_pct >= 50 else "BEARISH"
        forecast_str = f"{direction} (%{bullish_pct:.1f})"

        print(f"\nAI FORECAST ({forecast_days}-Bar Horizon)  : {direction}")
        print(f"Bullish Probability ({forecast_days} bars ahead) : %{bullish_pct:.1f}")
        print(f"Bearish Probability ({forecast_days} bars ahead) : %{bearish_pct:.1f}\n")
    else:
        print("\nAI FORECAST            : Insufficient data for prediction window.\n")

    logger.info(f"[{ticker}] Rendering Plotly Interactive Dashboard...")
    save_filename = f"{ticker}_{interval}_sentinel_report.html"
    plot_signals(
        df,
        ticker=ticker,
        window=window_size,
        metrics=metrics,
        trade_log=trade_log,
        train_window=train_window,
        initial_capital=initial_capital,
        save_path=save_filename,
    )

    return {
        "Ticker": ticker,
        "Trades": metrics.get("total_trades", 0),
        "Win_Rate": metrics.get("win_rate", 0.0),
        "Max_DD": metrics.get("max_drawdown", 0.0),
        "Sharpe": metrics.get("sharpe_ratio", 0.0),
        "Profit_Factor": metrics.get("profit_factor", 0.0),
        "Net_PnL": metrics.get("total_pnl", 0.0),
        "Forecast": forecast_str,
    }


def print_multi_ticker_summary(summaries: list[dict], interval: str):
    print(f"\n{'=' * 24} MULTI-ASSET PORTFOLIO SUMMARY ({interval}) {'=' * 24}")
    header = (
        f"{'Ticker':<8} | {'Trades':>6} | {'Win %':>6} | {'MaxDD %':>7} | "
        f"{'Sharpe':>6} | {'PF':>5} | {'Net PnL ($)':>11} | {'AI Forecast':<16}"
    )
    print(header)
    print("-" * len(header))

    total_portfolio_pnl = 0.0
    for s in summaries:
        total_portfolio_pnl += s["Net_PnL"]
        pf_display = f"{s['Profit_Factor']:.2f}" if s["Profit_Factor"] != float("inf") else "INF"
        print(
            f"{s['Ticker']:<8} | {s['Trades']:>6} | {s['Win_Rate']:>5.1f}% | {s['Max_DD']:>6.2f}% | "
            f"{s['Sharpe']:>6.2f} | {pf_display:>5} | {s['Net_PnL']:>11.2f} | {s['Forecast']:<16}"
        )

    print("-" * len(header))
    print(f"Combined Net Realized PnL Across {len(summaries)} Assets: ${total_portfolio_pnl:.2f}\n")


def run_sentinel():
    print("                    PROJECT SENTINEL                    ")

    cfg = load_config("config.yaml")

    data_cfg = cfg.get("data", {})
    analysis_cfg = cfg.get("analysis", {})
    ml_cfg = cfg.get("machine_learning", {})

    try:
        default_ticker = data_cfg.get("default_ticker", "RACE")
        interval = data_cfg.get("interval", "1d")

        raw_tickers = input(f"Enter asset ticker(s) (e.g., RACE, NVDA) [Default: {default_ticker}]: ").strip()
        tickers = parse_tickers(raw_tickers, default_ticker)

        window_size = analysis_cfg.get("window_size", 20)
        rsi_window = analysis_cfg.get("rsi_window", 14)
        train_window = ml_cfg.get("train_window", 200)

        print(f"\n[Configuration] Target(s): {', '.join(tickers)} | Interval: {interval}")
        print("[Date Configuration] (Press Enter to use automatic UTC-safe window)")

        ny_tz = pytz.timezone("America/New_York")
        now_ny = datetime.now(ny_tz)

        today = now_ny.date()
        default_lookback = 365 if interval in ("1d", "1wk", "1mo") else 59
        default_start = today - timedelta(days=default_lookback)

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
            if interval in ("1d", "1wk", "1mo"):
                min_required = train_window + max(window_size, rsi_window) + 14
                if days_difference < min_required:
                    logger.warning(
                        f"Date range is too short ({days_difference} days) for '{interval}' interval. "
                        f"Requires AT LEAST {min_required} calendar days!\n"
                    )
                    continue
            elif days_difference < 5:
                logger.warning("Intraday date range must be at least 5 calendar days.\n")
                continue

            break

        start_date = start_dt.strftime("%Y-%m-%d")
        end_date = end_dt.strftime("%Y-%m-%d")

        logger.info(f"Fetching market data for {len(tickers)} ticker(s) [{start_date} -> {end_date} | {interval}]...")
        raw_data = fetch_market_data(tickers, start_date=start_date, end_date=end_date, interval=interval)

        if isinstance(raw_data, pd.DataFrame):
            data_map = {tickers[0]: raw_data} if not raw_data.empty else {}
        elif isinstance(raw_data, dict):
            data_map = raw_data
        else:
            data_map = {}

        if not data_map:
            logger.error("Failed to fetch market data for all requested tickers. Terminating pipeline.")
            return

        summaries = []
        for ticker in tickers:
            df = data_map.get(ticker)
            if df is None or df.empty:
                logger.warning(f"[{ticker}] No valid market data returned. Skipping.")
                continue

            try:
                res = process_single_asset(ticker, df, cfg)
                if res is not None:
                    summaries.append(res)
            except Exception:
                logger.error(f"[{ticker}] Pipeline failed during execution. Continuing with next asset...", exc_info=True)

        if len(summaries) > 1:
            print_multi_ticker_summary(summaries, interval=interval)

        logger.info("Pipeline execution completed successfully.")

    except KeyboardInterrupt:
        logger.warning("Operation cancelled by user (KeyboardInterrupt). Exiting safely.")
    except Exception:
        logger.critical("A fatal error occurred during pipeline execution.", exc_info=True)


if __name__ == "__main__":
    run_sentinel()