import numpy as np
import pandas as pd
from numba import njit

@njit(fastmath=True)
def _kalman_loop(prices: np.ndarray, r_variances: np.ndarray, q_variances: np.ndarray) -> np.ndarray:
    n = len(prices)
    kalman_estimates = np.zeros(n, dtype=np.float64)
    if n == 0:
        return kalman_estimates

    post_estimate = prices[0]
    post_error = 1.0

    for i in range(n):
        prior_estimate = post_estimate
        prior_error = post_error + q_variances[i]

        kalman_gain = prior_error / (prior_error + r_variances[i] + 1e-9)
        post_estimate = prior_estimate + kalman_gain * (prices[i] - prior_estimate)
        post_error = (1.0 - kalman_gain) * prior_error

        kalman_estimates[i] = post_estimate

    return kalman_estimates


def calculate_average_true_range(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    required_cols = {"High", "Low", "Close"}
    if df.empty or not required_cols.issubset(df.columns):
        print(f"[Warning] DataFrame missing required columns {required_cols}. Skipping ATR calculation.")
        return df

    df = df.copy()
    high_low = df["High"] - df["Low"]
    high_prev_close = (df["High"] - df["Close"].shift(1)).abs()
    low_prev_close = (df["Low"] - df["Close"].shift(1)).abs()

    true_range = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    df["ATR"] = true_range.ewm(alpha=1 / window, adjust=False).mean()
    print(f"[Sentinel] Calculated {window}-period Average True Range (ATR).")
    return df


def add_kalman_filter(
        df: pd.DataFrame,
        base_process_variance: float = 1e-5,
        base_measurement_variance: float = 1e-3
) -> pd.DataFrame:
    if df.empty or "Close" not in df.columns:
        return df

    df = df.copy()

    if "ATR" not in df.columns:
        df = calculate_average_true_range(df)

    prices = df["Close"].to_numpy(dtype=np.float64)
    atr_values = df["ATR"].fillna(df["Close"] * 0.02).to_numpy(dtype=np.float64)
    atr_norm = atr_values / (prices + 1e-9)

    r_variances = base_measurement_variance / (1.0 + (atr_norm * 100.0))
    q_variances = base_process_variance * (1.0 + (atr_norm * 50.0))

    df["Kalman"] = _kalman_loop(prices, r_variances, q_variances)
    print("[Sentinel] Dynamic ATR-based Kalman Filter applied successfully.")
    return df


def calculate_moving_average(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    df = df.copy()
    df[f"SMA_{window}"] = df["Close"].rolling(window=window).mean()
    if "Volume" in df.columns:
        df[f"Vol_SMA_{window}"] = df["Volume"].rolling(window=window).mean()
    print(f"[Sentinel] Calculated {window}-day Moving Average and Volume SMA.")
    return df


def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    if df.empty or "Close" not in df.columns:
        print("[Warning] DataFrame is empty or missing 'Close' column. Skipping RSI calculation.")
        return df

    df = df.copy()
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()

    rs = avg_gain / (avg_loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))
    df["RSI"] = df["RSI"].fillna(50)

    print(f"[Sentinel] RSI indicator added using {window}-period EMA.")
    return df


def get_robust_zscore(series: pd.Series, window: int = 20) -> pd.Series:
    rolling_median = series.rolling(window=window).median()
    q75 = series.rolling(window=window).quantile(0.75)
    q25 = series.rolling(window=window).quantile(0.25)
    iqr = q75 - q25

    robust_zscore = (series - rolling_median) / (iqr / 1.349 + 1e-9)
    return robust_zscore.fillna(0.0)


def get_dynamic_margin(df: pd.DataFrame, window: int = 20) -> pd.Series:
    sma_col = f"SMA_{window}" if f"SMA_{window}" in df.columns else "Close"

    q75 = df["Close"].rolling(window=window).quantile(0.75)
    q25 = df["Close"].rolling(window=window).quantile(0.25)
    robust_std = (q75 - q25) / 1.349

    margin = robust_std / (df[sma_col] + 1e-9)
    return margin.clip(0.01, 0.05)


def add_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    df = df.copy()
    sma = df["Close"].rolling(window=window).mean()
    std_dev = df["Close"].rolling(window=window).std()

    df[f"BB_Upper_{window}"] = sma + (num_std * std_dev)
    df[f"BB_Lower_{window}"] = sma - (num_std * std_dev)
    print(f"[Sentinel] Added Bollinger Bands (window={window}, std={num_std}).")
    return df


def generate_signals(
        df: pd.DataFrame,
        window: int = 20,
        buy_threshold: float = 0.65,
        sell_threshold: float = -0.65,
        use_kalman: bool = True
) -> pd.DataFrame:
    df = df.copy()
    sma_col = f"SMA_{window}"

    if sma_col not in df.columns:
        df = calculate_moving_average(df, window=window)

    if "RSI" not in df.columns:
        df = add_rsi(df)

    if use_kalman and "Kalman" not in df.columns:
        df = add_kalman_filter(df)

    margins = get_dynamic_margin(df, window)

    dev = (df["Close"] - df[sma_col]) / (df[sma_col] * margins + 1e-9)
    price_score = np.clip(-dev, -1.0, 1.0)

    rsi_score = np.clip((50.0 - df["RSI"]) / 20.0, -1.0, 1.0)

    if use_kalman and "Kalman" in df.columns:
        kalman_dev = (df["Kalman"] - df["Close"]) / (df["Close"] + 1e-9)
        kalman_score = np.clip(kalman_dev * 50.0, -1.0, 1.0)
    else:
        kalman_score = 0.0

    vol_robust_zscore = get_robust_zscore(df["Volume"], window=window)
    volume_score = np.clip(vol_robust_zscore / 3.0, 0.0, 1.0)

    if "AI_Probability" in df.columns:
        ai_score = (df["AI_Probability"] - 0.5) * 2.0
    else:
        ai_score = 0.0

    weights = {
        "price": 0.25,
        "rsi": 0.25,
        "kalman": 0.20 if use_kalman else 0.0,
        "volume": 0.15,
        "ai": 0.15
    }

    total_weight = sum(weights.values())

    composite_score = (
        weights["price"] * price_score +
        weights["rsi"] * rsi_score +
        weights["kalman"] * kalman_score +
        weights["volume"] * volume_score +
        weights["ai"] * ai_score
    ) / total_weight

    df["Signal_Score"] = composite_score

    df["Signal"] = 0
    df.loc[df["Signal_Score"] >= buy_threshold, "Signal"] = 1
    df.loc[df["Signal_Score"] <= sell_threshold, "Signal"] = -1

    return df