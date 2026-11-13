import unittest
import numpy as np
import pandas as pd
from data_loader import clean_market_data
from analysis import add_rsi, add_price_momentum, extract_features
from strategy import generate_hybrid_signals

class TestSentinelPipeline(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        dates = pd.date_range(start="2026-01-01", periods=100, freq="B")

        returns = np.random.normal(0.0, 0.02, size=100)
        close_prices = 100.0 * np.exp(np.cumsum(returns))
        
        self.mock_df = pd.DataFrame({
            "Open": close_prices * 0.99,
            "High": close_prices * 1.02,
            "Low": close_prices * 0.98,
            "Close": close_prices,
            "Volume": np.random.randint(1000, 50000, size=100)
        }, index=dates)

    def test_clean_market_data_drops_invalid_ohlc(self):
        bad_df = self.mock_df.copy()
        bad_df.loc[bad_df.index[10], "High"] = 50.0
        bad_df.loc[bad_df.index[10], "Low"] = 100.0 
        
        cleaned_df = clean_market_data(bad_df, keep_weekends=True)
        self.assertEqual(len(cleaned_df), len(bad_df) - 1, "Failed to drop invalid OHLC row.")

    def test_clean_market_data_fixes_negative_volume(self):
        bad_df = self.mock_df.copy()
        bad_df.loc[bad_df.index[5], "Volume"] = -5000
        
        cleaned_df = clean_market_data(bad_df, keep_weekends=True)
        self.assertEqual(cleaned_df.loc[cleaned_df.index[5], "Volume"], 0.0, "Failed to fix negative volume.")

    def test_rsi_mathematical_bounds(self):
        df_with_rsi = add_rsi(self.mock_df.copy(), window=14)
        rsi_series = df_with_rsi["RSI"].dropna()
        
        self.assertTrue((rsi_series >= 0).all(), "RSI value dropped below 0.")
        self.assertTrue((rsi_series <= 100).all(), "RSI value exceeded 100.")

    def test_momentum_slope_returns_valid_shape(self):
        df_with_momentum = add_price_momentum(self.mock_df.copy(), window=10)
        
        self.assertIn("feat_trend_slope_10", df_with_momentum.columns, "Momentum column missing.")
        self.assertEqual(len(df_with_momentum), len(self.mock_df), "DataFrame length altered during calculation.")

    def test_strategy_generates_valid_signals(self):
        df = self.mock_df.copy()
        df["AI_Probability"] = np.random.uniform(0.2, 0.8, size=len(df))
        df["Kalman"] = df["Close"].rolling(5).mean()  # Mock Kalman trend
        
        signal_df = generate_hybrid_signals(df, ai_upper_threshold=0.55, ai_lower_threshold=0.45)
        
        unique_signals = set(signal_df["Signal"].dropna().unique())
        valid_signals = {-1, 0, 1}
        self.assertTrue(unique_signals.issubset(valid_signals), f"Generated invalid signals: {unique_signals}")

    def test_dynamic_threshold_activation(self):
        df = self.mock_df.copy()
        df["AI_Probability"] = np.random.uniform(0.4, 0.6, size=len(df))
        df["Kalman"] = df["Close"] * 0.9

        df["feat_volatility"] = 1.0  
        df.loc[df.index[-10:], "feat_volatility"] = 5.0  
        
        signal_df = generate_hybrid_signals(
            df, 
            ai_upper_threshold=0.55, 
            ai_lower_threshold=0.45, 
            use_dynamic_threshold=True, 
            vol_sensitivity=0.1
        )

        dynamic_upper = signal_df["Dynamic_Upper_Thresh"].iloc[-1]
        self.assertGreater(dynamic_upper, 0.55, "Dynamic threshold failed to expand during high volatility.")

if __name__ == "__main__":
    print("[Sentinel Testbench] Initializing automated module verification...\n")
    unittest.main()