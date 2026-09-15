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
        return df

    df = df.copy()
    high_low = df["High"] - df["Low"]
    high_prev_close = (df["High"] - df["Close"].shift(1)).abs()
    low_prev_close = (df["Low"] - df["Close"].shift(1)).abs()

    true_range = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    df["ATR"] = true_range.ewm(alpha=1 / window, adjust=False).mean()
    df["ATR_Norm"] = df["ATR"] / (df["Close"] + 1e-9)
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
    df["feat_kalman_dev"] = (df["Kalman"] - df["Close"]) / (df["Close"] + 1e-9)
    return df


def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    if df.empty or "Close" not in df.columns:
        return df

    df = df.copy()
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()

    rs = avg_gain / (avg_loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))
    df["feat_rsi_scaled"] = (df["RSI"] - 50.0) / 50.0  # Normalized between -1 and 1
    return df


def get_robust_zscore(series: pd.Series, window: int = 20) -> pd.Series:
    rolling_median = series.rolling(window=window).median()
    q75 = series.rolling(window=window).quantile(0.75)
    q25 = series.rolling(window=window).quantile(0.25)
    iqr = q75 - q25

    robust_zscore = (series - rolling_median) / (iqr / 1.349 + 1e-9)
    return robust_zscore.fillna(0.0)


def extract_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:

    if df.empty:
        return df

    df = df.copy()

    df = calculate_average_true_range(df, window=14)
    df = add_rsi(df, window=14)
    df = add_kalman_filter(df)

    sma_col = f"SMA_{window}"
    df[sma_col] = df["Close"].rolling(window=window).mean()

    q75 = df["Close"].rolling(window=window).quantile(0.75)
    q25 = df["Close"].rolling(window=window).quantile(0.25)
    robust_std = (q75 - q25) / 1.349
    df["feat_volatility"] = robust_std / (df[sma_col] + 1e-9)

    df["feat_price_dev"] = (df["Close"] - df[sma_col]) / (df[sma_col] * df["feat_volatility"] + 1e-9)
    df["feat_vol_zscore"] = get_robust_zscore(df["Volume"], window=window) if "Volume" in df.columns else 0.0

    print(f"[Sentinel] Successfully generated ML features for dataset ({len(df)} rows).")
    return df