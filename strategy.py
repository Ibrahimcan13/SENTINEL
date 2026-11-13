import pandas as pd
import numpy as np


def generate_hybrid_signals(
    df: pd.DataFrame, 
    ai_upper_threshold: float = 0.55,
    ai_lower_threshold: float = 0.45,
    allow_short: bool = True,
    confirmation_days: int = 1,
    use_dynamic_thresholds: bool = True,
    vol_sensitivity: float = 0.5
) -> pd.DataFrame:
    if df.empty or "AI_Probability" not in df.columns:
        print("[Sentinel Warning] 'AI_Probability' column is missing for Strategy Engine.")
        return df

    df = df.copy()
    df["Signal"] = 0
    if use_dynamic_thresholds and  "feat_volatility" in df.columns:
        baseline_vol = df["feat_volatility"].rolling(window=50, min_periods=10).median()
        vol_ratio = df["feat_volatility"] / (baseline_vol + 1e-9)
       
        vol_excess = (vol_ratio - 1.0).clip(lower=0.0, upper=2.0).fillna(0.0)
        threshold_shift = vol_excess * vol_sensitivity

        upper_thresh = (ai_upper_threshold + threshold_shift).clip(upper=0.85)
        lower_thresh = (ai_lower_threshold - threshold_shift).clip(lower=0.15)
    else:
        upper_thresh = ai_upper_threshold
        lower_thresh = ai_lower_threshold

    df["Dynamic_Upper_Thresh"] = upper_thresh
    df["Dynamic_Lower_Thresh"] = lower_thresh

    if "Kalman" in df.columns:
        trend_bullish = df["Close"] > df["Kalman"]
        trend_bearish = df["Close"] < df["Kalman"]
    else:
        trend_bullish = True
        trend_bearish = True

    ai_bullish = df["AI_Probability"] > ai_upper_threshold
    ai_bearish = df["AI_Probability"] < ai_lower_threshold

    long_raw = ai_bullish & trend_bullish
    short_raw = ai_bearish & trend_bearish

    if confirmation_days > 1:
        long_condition = long_raw.rolling(window=confirmation_days).sum() == confirmation_days
        short_condition = short_raw.rolling(window=confirmation_days).sum() == confirmation_days
    else:
        long_condition = long_raw
        short_condition = short_raw

    df.loc[long_condition, "Signal"] = 1

    if allow_short:
        df.loc[short_condition, "Signal"] = -1
    
    return df