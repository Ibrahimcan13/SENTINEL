import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler


def apply_triple_barrier_labels(
    df: pd.DataFrame,
    pt_multiplier: float = 2.0,
    sl_multiplier: float = 1.0,
    max_holding_days: int = 5
) -> pd.Series:

    if "ATR" not in df.columns or df.empty:
        simple_target = (df["Close"].shift(-max_holding_days) > df["Close"]).astype(float)
        simple_target.iloc[-max_holding_days:] = np.nan
        return simple_target

    close_prices = df["Close"].to_numpy()
    atr_values = df["ATR"].to_numpy()
    n = len(df)

    labels = np.full(n, np.nan, dtype=float)

    for i in range(n - max_holding_days):
        entry_price = close_prices[i]
        atr = atr_values[i]

        if np.isnan(entry_price) or np.isnan(atr) or atr == 0:
            continue

        upper_barrier = entry_price + (atr * pt_multiplier)
        lower_barrier = entry_price - (atr * sl_multiplier)

        path = close_prices[i + 1: i + 1 + max_holding_days]

        touch_tp = np.where(path >= upper_barrier)[0]
        touch_sl = np.where(path <= lower_barrier)[0]

        first_tp = touch_tp[0] if len(touch_tp) > 0 else max_holding_days + 1
        first_sl = touch_sl[0] if len(touch_sl) > 0 else max_holding_days + 1

        if first_tp < first_sl:
            labels[i] = 1.0
        elif first_sl < first_tp:
            labels[i] = 0.0
        else:
            labels[i] = 1.0 if path[-1] > entry_price else 0.0

    return pd.Series(labels, index=df.index)


def create_features_and_targets(df: pd.DataFrame, forecast_days: int = 5) -> pd.DataFrame:
    if df.empty:
        return df

    data = df.copy()

    data["feat_return"] = data["Close"].pct_change().fillna(0.0)

    if "feat_rsi_scaled" not in data.columns:
        data["feat_rsi_scaled"] = ((data["RSI"] - 50.0) / 50.0).fillna(0.0) if "RSI" in data.columns else 0.0
        
    if "feat_kalman_dev" not in data.columns and "Kalman" in data.columns:
        data["feat_kalman_dev"] = ((data["Close"] - data["Kalman"]) / (data["Kalman"] + 1e-9)).fillna(0.0)

    data["Target_Direction"] = apply_triple_barrier_labels(
        data, pt_multiplier=2.0, sl_multiplier=1.0, max_holding_days=forecast_days
    )

    return data


def train_and_predict(
    df: pd.DataFrame,
    forecast_days: int = 5,
    train_window: int = 200,
    retrain_step: int = 20
) -> pd.DataFrame:
    processed_df = create_features_and_targets(df, forecast_days=forecast_days)

    if processed_df.empty:
        print("[Sentinel] Predictor error: Insufficient data to train the model.")
        return df

    feature_cols = [col for col in processed_df.columns if col.startswith("feat_")]

    if len(processed_df) < train_window + forecast_days + 10:
        print(f"[Sentinel] Predictor warning: Too few rows ({len(processed_df)}) for rolling window.")
        return df

    probabilities = [0.50] * len(processed_df)
    current_model = None
    scaler = None

    for i in range(train_window + forecast_days, len(processed_df)):
        if (i - (train_window + forecast_days)) % retrain_step == 0 or current_model is None:
            train_end_idx = i - forecast_days
            train_start_idx = train_end_idx - train_window

            train_chunk = processed_df.iloc[train_start_idx:train_end_idx]
            train_chunk_clean = train_chunk.dropna(subset=["Target_Direction"])

            if len(train_chunk_clean) < 30:
                continue

            X_train = train_chunk_clean[feature_cols]
            y_train = train_chunk_clean["Target_Direction"].astype(int)

            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)

            current_model = RandomForestClassifier(
                n_estimators=50,
                max_depth=5,
                min_samples_leaf=5,
                class_weight="balanced",
                n_jobs=-1,
                random_state=42
            )
            current_model.fit(X_train_scaled, y_train)

        X_test = processed_df.iloc[[i]][feature_cols]
        X_test_scaled = scaler.transform(X_test)
        probabilities[i] = current_model.predict_proba(X_test_scaled)[0, 1]

    processed_df["AI_Probability"] = probabilities
    return processed_df