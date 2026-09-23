import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_signals(
    df: pd.DataFrame,
    ticker: str,
    window: int,
    metrics: dict,
    trade_log: list = None,
    train_window: int = 200,
    initial_capital: float = 1000.0,
    save_path: str = None
) -> None:
    if df.empty:
        print("[Sentinel] Visualizer Error: DataFrame is empty.")
        return

    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(
            f"Sentinel: {ticker} Price Analysis & Trading Signals",
            "AI Bullish Probability (%)",
            "Relative Strength Index (RSI)",
            "Volume & ATR Volatility",
            "Portfolio Equity Curve ($)"
        ),
        row_heights=[0.35, 0.15, 0.15, 0.15, 0.20]
    )

    if all(col in df.columns for col in ["Open", "High", "Low", "Close"]):
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="Price (OHLC)",
                increasing_line_color="#26a69a",
                decreasing_line_color="#ef5350"
            ),
            row=1, col=1
        )
    elif "Close" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close Price", line=dict(color="#cccccc", width=1.5)),
            row=1, col=1
        )

    if "Kalman" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["Kalman"], mode="lines", name="Kalman Filter", line=dict(color="#00e5ff", width=1.5)),
            row=1, col=1
        )

    sma_col = f"SMA_{window}"
    if sma_col in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df[sma_col], mode="lines", name=f"SMA {window}", line=dict(color="#ffd700", width=1.2, dash="dash")),
            row=1, col=1
        )

    bb_upper_col = f"BB_Upper_{window}"
    bb_lower_col = f"BB_Lower_{window}"
    if bb_upper_col in df.columns and bb_lower_col in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df[bb_upper_col], mode="lines", name="BB Upper", line=dict(color="#ab47bc", width=1, dash="dot")),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df[bb_lower_col], mode="lines", name="BB Lower",
                line=dict(color="#ab47bc", width=1, dash="dot"),
                fill="tonexty", fillcolor="rgba(0, 229, 255, 0.03)"
            ),
            row=1, col=1
        )

    if "Signal" in df.columns:
        buy_signals = df[df["Signal"] == 1]
        sell_signals = df[df["Signal"] == -1]

        if not buy_signals.empty and "Close" in buy_signals.columns:
            hover_buy = [
                f"BUY Signal<br>RSI: {row.get('RSI', 0):.1f}<br>AI Prob: {row.get('AI_Probability', 0)*100:.1f}%"
                for _, row in buy_signals.iterrows()
            ]
            fig.add_trace(
                go.Scatter(
                    x=buy_signals.index, y=buy_signals["Close"], mode="markers", name="BUY",
                    marker=dict(symbol="triangle-up", size=12, color="#00ff00"),
                    hovertext=hover_buy, hoverinfo="text+x+y"
                ),
                row=1, col=1
            )

        if not sell_signals.empty and "Close" in sell_signals.columns:
            fig.add_trace(
                go.Scatter(
                    x=sell_signals.index, y=sell_signals["Close"], mode="markers", name="SELL",
                    marker=dict(symbol="triangle-down", size=12, color="#ff0000"),
                    hovertext="SELL Signal", hoverinfo="text+x+y"
                ),
                row=1, col=1
            )

    if trade_log:
        stop_events = []
        for trade in trade_log:
            exit_reason = str(trade.get("Reason", trade.get("exit_reason", ""))).upper()
            if "STOP" in exit_reason:
                date = trade.get("Exit_Date", trade.get("exit_date"))
                price = trade.get("Exit_Price", trade.get("exit_price"))
                if date is not None and price is not None:
                    stop_events.append((date, price))

        if stop_events:
            stop_dates, stop_prices = zip(*stop_events)
            hover_stops = [f"Stop-Loss Hit<br>Price: ${p:.2f}" for p in stop_prices]

            fig.add_trace(
                go.Scatter(
                    x=list(stop_dates),
                    y=list(stop_prices),
                    mode="markers",
                    name="Stop-Loss Hit",
                    marker=dict(symbol="x", size=12, color="#ff9800", line=dict(width=2)),
                    hovertext=hover_stops,
                    hoverinfo="text+x"
                ),
                row=1, col=1
            )

    if len(df) > train_window:
        split_date = df.index[train_window]
        fig.add_vline(x=split_date, line_width=1.5, line_dash="dash", line_color="#ffff00", row="all", col=1)

    if "AI_Probability" in df.columns:
        ai_prob = df["AI_Probability"] * 100
        fig.add_trace(
            go.Scatter(x=df.index, y=ai_prob, mode="lines", name="AI Prob (%)", line=dict(color="#00e5ff", width=1.5)),
            row=2, col=1
        )
        fig.add_hline(y=50, line_dash="dash", line_color="gray", row=2, col=1)
        fig.add_hline(y=55, line_dash="dot", line_color="#00ff00", row=2, col=1)

    if "RSI" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["RSI"], mode="lines", name="RSI", line=dict(color="#e040fb", width=1.2)),
            row=3, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="#ff1744", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#00e645", row=3, col=1)

    if "Volume" in df.columns:
        fig.add_trace(
            go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color="#29b6f6", opacity=0.4),
            row=4, col=1
        )
        vol_sma_col = f"Vol_SMA_{window}"
        if vol_sma_col in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df[vol_sma_col], mode="lines", name=f"Vol SMA {window}", line=dict(color="#ff9800", width=1.2)),
                row=4, col=1
            )

    if "equity_curve" in metrics and isinstance(metrics["equity_curve"], pd.Series):
        equity = metrics["equity_curve"]
        fig.add_trace(
            go.Scatter(x=equity.index, y=equity.values, mode="lines", name="Portfolio Equity", line=dict(color="#ffd700", width=2)),
            row=5, col=1
        )
        fig.add_hline(y=initial_capital, line_dash="dash", line_color="gray", row=5, col=1)

        stats_text = (
            f"<b>Net PnL:</b> ${metrics.get('total_pnl', 0.0):.2f}<br>"
            f"<b>Win Rate:</b> %{metrics.get('win_rate', 0.0):.1f}<br>"
            f"<b>Max DD:</b> %{metrics.get('max_drawdown', 0.0):.2f}<br>"
            f"<b>Sharpe:</b> {metrics.get('sharpe_ratio', 0.0):.2f}"
        )
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.99, y=0.02,
            text=stats_text,
            showarrow=False,
            align="right",
            bordercolor="#ffd700",
            borderwidth=1,
            borderpad=6,
            bgcolor="rgba(0, 0, 0, 0.8)",
            font=dict(color="#ffffff", size=11)
        )

    fig.update_layout(
        template="plotly_dark",
        title=dict(text=f"PROJECT SENTINEL — {ticker} Interactive Analysis", font=dict(size=18)),
        height=1000,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        margin=dict(l=50, r=50, t=60, b=40)
    )

    if save_path:
        path = save_path if save_path.endswith(".html") else f"{save_path}.html"
        fig.write_html(path)
        print(f"[Sentinel] Interactive Plotly dashboard saved to {path}")
    else:
        fig.show()