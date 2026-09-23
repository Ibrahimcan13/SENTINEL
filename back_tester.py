import numpy as np
import pandas as pd


def calculate_backtest_metrics(
    df: pd.DataFrame,
    trade_log: list[dict],
    initial_capital: float,
    equity_series: pd.Series,
) -> dict:
    total_trades = len(trade_log)
    winning_trades = sum(1 for t in trade_log if t["PnL"] > 0)

    win_rate = (winning_trades / total_trades * 100.0) if total_trades > 0 else 0.0
    total_pnl = equity_series.iloc[-1] - initial_capital if not equity_series.empty else 0.0

    equity_arr = equity_series.to_numpy(dtype=np.float64)

    if len(equity_arr) > 0:
        peak = np.maximum.accumulate(equity_arr)
        drawdown = (equity_arr - peak) / (np.where(peak == 0, 1e-9, peak))
        max_drawdown = np.min(drawdown) * 100.0
    else:
        max_drawdown = 0.0

    if len(equity_arr) > 1:
        prev_equity = equity_arr[:-1]
        daily_returns = np.diff(equity_arr) / (np.where(prev_equity == 0, 1e-9, prev_equity))
        mean_return = np.mean(daily_returns)
        std_return = np.std(daily_returns)
    else:
        mean_return = 0.0
        std_return = 0.0

    sharpe_ratio = (
        (mean_return / std_return) * np.sqrt(252) if std_return > 0.0 else 0.0
    )

    gross_profits = sum(t["PnL"] for t in trade_log if t["PnL"] > 0)
    gross_losses = abs(sum(t["PnL"] for t in trade_log if t["PnL"] < 0))

    if gross_losses > 0.0:
        profit_factor = gross_profits / gross_losses
    elif gross_profits > 0.0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    avg_duration = (
        sum(t["Duration_Days"] for t in trade_log) / total_trades
        if total_trades > 0 else 0.0
    )

    metrics = {
        "total_pnl": total_pnl,
        "win_rate": win_rate,
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "max_drawdown": abs(max_drawdown),
        "sharpe_ratio": sharpe_ratio,
        "profit_factor": profit_factor,
        "avg_duration_days": avg_duration,
        "equity_curve": equity_series,
    }

    print(
        f"[Sentinel] Backtest Completed -> Net PnL: ${total_pnl:.2f} | Win Rate: {win_rate:.1f}% | "
        f"Max DD: {abs(max_drawdown):.2f}% | Sharpe: {sharpe_ratio:.2f} | Avg Duration: {avg_duration:.1f} days"
    )

    return metrics


def run_backtest(
    df: pd.DataFrame,
    initial_capital: float = 1000.0,
    risk_per_trade: float = 0.02,
    commission_rate: float = 0.001,
    slippage_rate: float = 0.0005,
    use_atr_stop: bool = True,
    atr_multiplier: float = 2.0,
    ai_exit_threshold: float = 0.50,
) -> tuple[pd.DataFrame, list[dict], dict]:
    if df.empty or "Signal" not in df.columns:
        print("[Warning] DataFrame is empty or missing 'Signal' column.")
        empty_metrics = {
            "total_pnl": 0.0,
            "win_rate": 0.0,
            "total_trades": 0,
            "winning_trades": 0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "profit_factor": 0.0,
            "avg_duration_days": 0.0,
            "equity_curve": pd.Series([initial_capital]),
        }
        return df, [], empty_metrics

    if "Open" not in df.columns or df["Open"].isnull().all():
        print("[Warning] 'Open' column missing/NaN. Deriving from previous Close.")
        df["Open"] = df["Close"].shift(1).fillna(df["Close"])

    df = df.copy()

    dates = df.index.to_numpy()
    close_arr = df["Close"].to_numpy(dtype=np.float64)
    open_arr = df["Open"].to_numpy(dtype=np.float64) if "Open" in df.columns else close_arr
    high_arr = df["High"].to_numpy(dtype=np.float64)
    low_arr = df["Low"].to_numpy(dtype=np.float64)
    signal_arr = df["Signal"].to_numpy(dtype=np.int8)

    ai_prob_arr = (
        df["AI_Probability"].to_numpy(dtype=np.float64)
        if "AI_Probability" in df.columns
        else np.full(len(df), np.nan)
    )

    atr_arr = (
        df["ATR"].to_numpy(dtype=np.float64)
        if "ATR" in df.columns
        else np.zeros(len(df), dtype=np.float64)
    )

    cash = initial_capital
    position = 0.0
    entry_price = 0.0
    entry_cost = 0.0
    entry_date = None
    peak_price_since_entry = 0.0
    trough_price_since_entry = 0.0

    pending_action = 0  
    pending_reason = ""

    trade_log = []
    equity_list = np.zeros(len(df), dtype=np.float64)

    for i in range(len(df)):
        current_date = dates[i]
        current_open = open_arr[i]
        current_close = close_arr[i]
        current_high = high_arr[i]
        current_low = low_arr[i]
        atr_val = atr_arr[i]

        if pending_action != 0 and position == 0.0:
            if pending_action == 1:  
                buy_price = current_open * (1.0 + slippage_rate)
                if atr_val > 0.0 and use_atr_stop:
                    risk_amount = cash * risk_per_trade
                    stop_distance = atr_val * atr_multiplier
                    raw_shares = risk_amount / (stop_distance + 1e-9)
                    allocated_cash = min(cash * 0.95, raw_shares * buy_price)
                else:
                    allocated_cash = cash * 0.10

                commission = allocated_cash * commission_rate
                investable_cash = allocated_cash - commission

                position = investable_cash / buy_price
                entry_price = buy_price
                entry_cost = allocated_cash
                entry_date = current_date
                peak_price_since_entry = current_high
                cash -= allocated_cash

            elif pending_action == -1:  
                short_price = current_open * (1.0 - slippage_rate)
                if atr_val > 0.0 and use_atr_stop:
                    risk_amount = cash * risk_per_trade
                    stop_distance = atr_val * atr_multiplier
                    raw_shares = risk_amount / (stop_distance + 1e-9)
                    allocated_cash = min(cash * 0.95, raw_shares * short_price)
                else:
                    allocated_cash = cash * 0.10

                commission = allocated_cash * commission_rate
                investable_cash = allocated_cash - commission

                position = -(investable_cash / short_price)
                entry_price = short_price
                entry_cost = allocated_cash
                entry_date = current_date
                trough_price_since_entry = current_low
                cash -= allocated_cash

            pending_action = 0  

        if position > 0.0:  
            peak_price_since_entry = max(peak_price_since_entry, current_high)
            stop_loss_price = (
                peak_price_since_entry - (atr_val * atr_multiplier)
                if use_atr_stop and not np.isnan(atr_val)
                else 0.0
            )

            hit_stop_loss = use_atr_stop and (current_low <= stop_loss_price)
            ai_bearish_exit = not np.isnan(ai_prob_arr[i]) and (ai_prob_arr[i] < ai_exit_threshold)
            sell_signal = (signal_arr[i] == -1)

            if hit_stop_loss or ai_bearish_exit or sell_signal:
                exit_base_price = stop_loss_price if hit_stop_loss else current_close
                sell_price = exit_base_price * (1.0 - slippage_rate)
                gross_cash = position * sell_price
                commission = gross_cash * commission_rate
                net_returned_cash = gross_cash - commission

                pnl = net_returned_cash - entry_cost
                pnl_pct = (pnl / (entry_cost + 1e-9)) * 100.0
                cash += net_returned_cash

                duration_days = (
                    (pd.Timestamp(current_date) - pd.Timestamp(entry_date)).days
                    if entry_date is not None else 0
                )

                reason = "LONG Exit (SELL)"
                if hit_stop_loss:
                    reason = "LONG ATR Stop Loss"
                elif ai_bearish_exit:
                    reason = "LONG AI Bearish Exit"

                trade_log.append({
                    "Type": "LONG",
                    "Entry_Date": entry_date,
                    "Exit_Date": current_date,
                    "Entry_Price": entry_price,
                    "Exit_Price": sell_price,
                    "PnL": pnl,
                    "PnL_Pct": pnl_pct,
                    "Duration_Days": duration_days,
                    "Reason": reason,
                })

                position = 0.0
                entry_price = 0.0
                entry_cost = 0.0
                entry_date = None

        elif position < 0.0: 
            trough_price_since_entry = min(trough_price_since_entry, current_low)
            stop_loss_price = (
                trough_price_since_entry + (atr_val * atr_multiplier)
                if use_atr_stop and not np.isnan(atr_val)
                else float("inf")
            )

            hit_stop_loss = use_atr_stop and (current_high >= stop_loss_price)
            ai_bullish_exit = not np.isnan(ai_prob_arr[i]) and (ai_prob_arr[i] > (1.0 - ai_exit_threshold))
            buy_signal = (signal_arr[i] == 1)

            if hit_stop_loss or ai_bullish_exit or buy_signal:
                exit_base_price = stop_loss_price if hit_stop_loss else current_close
                cover_price = exit_base_price * (1.0 + slippage_rate)
                
                num_shares = abs(position)
                gross_pnl = (entry_price - cover_price) * num_shares
                
                buyback_value = num_shares * cover_price
                commission = buyback_value * commission_rate
                net_pnl = gross_pnl - commission
                
                pnl_pct = (net_pnl / (entry_cost + 1e-9)) * 100.0
                cash += (entry_cost + net_pnl)

                duration_days = (
                    (pd.Timestamp(current_date) - pd.Timestamp(entry_date)).days
                    if entry_date is not None else 0
                )

                reason = "SHORT Exit (COVER)"
                if hit_stop_loss:
                    reason = "SHORT ATR Stop Loss"
                elif ai_bullish_exit:
                    reason = "SHORT AI Bullish Exit"

                trade_log.append({
                    "Type": "SHORT",
                    "Entry_Date": entry_date,
                    "Exit_Date": current_date,
                    "Entry_Price": entry_price,
                    "Exit_Price": cover_price,
                    "PnL": net_pnl,
                    "PnL_Pct": pnl_pct,
                    "Duration_Days": duration_days,
                    "Reason": reason,
                })

                position = 0.0
                entry_price = 0.0
                entry_cost = 0.0
                entry_date = None

        if position == 0.0 and pending_action == 0:
            if signal_arr[i] == 1:
                pending_action = 1
            elif signal_arr[i] == -1:
                pending_action = -1

        if position > 0.0:
            current_equity = cash + (position * current_close)
        elif position < 0.0:
            unrealized_pnl = (entry_price - current_close) * abs(position)
            current_equity = cash + entry_cost + unrealized_pnl
        else:
            current_equity = cash

        equity_list[i] = current_equity

    if position != 0.0:
        last_close = close_arr[-1]
        if position > 0.0:
            sell_price = last_close * (1.0 - slippage_rate)
            gross_cash = position * sell_price
            commission = gross_cash * commission_rate
            pnl = (gross_cash - commission) - entry_cost
            pnl_pct = (pnl / (entry_cost + 1e-9)) * 100.0
            cash += (gross_cash - commission)
            p_type = "LONG"
            exit_p = sell_price
        else:
            cover_price = last_close * (1.0 + slippage_rate)
            num_shares = abs(position)
            gross_pnl = (entry_price - cover_price) * num_shares
            commission = (num_shares * cover_price) * commission_rate
            pnl = gross_pnl - commission
            pnl_pct = (pnl / (entry_cost + 1e-9)) * 100.0
            cash += (entry_cost + pnl)
            p_type = "SHORT"
            exit_p = cover_price

        trade_log.append({
            "Type": p_type,
            "Entry_Date": entry_date,
            "Exit_Date": dates[-1],
            "Entry_Price": entry_price,
            "Exit_Price": exit_p,
            "PnL": pnl,
            "PnL_Pct": pnl_pct,
            "Duration_Days": (pd.Timestamp(dates[-1]) - pd.Timestamp(entry_date)).days if entry_date is not None else 0,
            "Reason": "End of Data (Auto Close)",
        })
        equity_list[-1] = cash

    equity_series = pd.Series(equity_list, index=df.index)
    df["Portfolio_Equity"] = equity_series

    metrics = calculate_backtest_metrics(
        df, trade_log, initial_capital, equity_series
    )

    return df, trade_log, metrics