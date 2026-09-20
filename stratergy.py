import pandas as pd
import numpy as np


def generate_hybrid_signals(
    df: pd.DataFrame, 
    ai_upper_threshold: float = 0.55,
    ai_lower_threshold: float = 0.45
) -> pd.DataFrame:
    if df.empty or "AI_Probability" not in df.columns:
        print("[Sentinel Warning] 'AI_Probability' column is missing for Strategy Engine.")
        return df

    df = df.copy()
    df["Signal"] = 0

    if "Kalman" in df.columns:
        trend_bullish = df["Close"] > df["Kalman"]
        trend_bearish = df["Close"] < df["Kalman"]
    else:
        trend_bullish = True
        trend_bearish = True

    ai_bullish = df["AI_Probability"] > ai_upper_threshold
    ai_bearish = df["AI_Probability"] < ai_lower_threshold

    df.loc[ai_bullish & trend_bullish, "Signal"] = 1

    df.loc[ai_bearish | trend_bearish, "Signal"] = -1

    return df