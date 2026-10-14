import numpy as np
import pandas as pd
from numba import njit


@njit(fastmath=True)
def _run_backtest_kernel(
    close_arr: np.ndarray,
    open_arr: np.ndarray,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    signal_arr: np.ndarray,
    ai_prob_arr: np.ndarray,
    atr_arr: np.ndarray,
    initial_capital: float,
    risk_per_trade: float,
    commission_rate: float,
    slippage_rate: float,
    use_atr_stop: bool,
    atr_multiplier: float,
    ai_exit_threshold: float,
):
    n = len(close_arr)
    equity_list = np.zeros(n, dtype=np.float64)

    records = np.zeros((n, 8), dtype=np.float64) 
    trade_count = 0

    cash = initial_capital
    position = 0.0
    entry_price = 0.0
    entry_cost = 0.0
    entry_idx = 0
    peak_price = 0.0
    trough_price = 0.0
    pending_action = 0

    for i in range(n):
        c_open = open_arr[i]
        c_close = close_arr[i]
        c_high = high_arr[i]
        c_low = low_arr[i]
        atr_val = atr_arr[i]
        ai_prob = ai_prob_arr[i]

        if pending_action != 0 and position == 0.0:
            if pending_action == 1:  # LONG
                buy_price = c_open * (1.0 + slippage_rate)
                if atr_val > 0.0 and use_atr_stop:
                    risk_amt = cash * risk_per_trade
                    stop_dist = atr_val * atr_multiplier
                    raw_shares = risk_amt / (stop_dist + 1e-9)
                    allocated = min(cash * 0.95, raw_shares * buy_price)
                else:
                    allocated = cash * 0.10

                comm = allocated * commission_rate
                position = (allocated - comm) / buy_price
                entry_price = buy_price
                entry_cost = allocated
                entry_idx = i
                peak_price = c_high
                cash -= allocated

            elif pending_action == -1:  
                short_price = c_open * (1.0 - slippage_rate)
                if atr_val > 0.0 and use_atr_stop:
                    risk_amt = cash * risk_per_trade
                    stop_dist = atr_val * atr_multiplier
                    raw_shares = risk_amt / (stop_dist + 1e-9)
                    allocated = min(cash * 0.95, raw_shares * short_price)
                else:
                    allocated = cash * 0.10

                comm = allocated * commission_rate
                position = -((allocated - comm) / short_price)
                entry_price = short_price
                entry_cost = allocated
                entry_idx = i
                trough_price = c_low
                cash -= allocated

            pending_action = 0

        if position > 0.0:  
            peak_price = max(peak_price, c_high)
            stop_price = peak_price - (atr_val * atr_multiplier) if use_atr_stop and not np.isnan(atr_val) else 0.0

            hit_stop = use_atr_stop and (c_low <= stop_price)
            ai_exit = not np.isnan(ai_prob) and (ai_prob < ai_exit_threshold)
            sell_sig = (signal_arr[i] == -1)

            if hit_stop or ai_exit or sell_sig:
                exit_base = stop_price if hit_stop else c_close
                sell_p = exit_base * (1.0 - slippage_rate)
                gross = position * sell_p
                comm = gross * commission_rate
                net_returned = gross - comm
                pnl = net_returned - entry_cost
                pnl_pct = (pnl / (entry_cost + 1e-9)) * 100.0
                cash += net_returned

                reason_code = 1.0 if hit_stop else (2.0 if ai_exit else 3.0)
                records[trade_count] = np.array([1.0, entry_idx, i, entry_price, sell_p, pnl, pnl_pct, reason_code])
                trade_count += 1

                position = 0.0

        elif position < 0.0:  
            trough_price = min(trough_price, c_low)
            stop_price = trough_price + (atr_val * atr_multiplier) if use_atr_stop and not np.isnan(atr_val) else 1e9

            hit_stop = use_atr_stop and (c_high >= stop_price)
            ai_exit = not np.isnan(ai_prob) and (ai_prob > (1.0 - ai_exit_threshold))
            buy_sig = (signal_arr[i] == 1)

            if hit_stop or ai_exit or buy_sig:
                exit_base = stop_price if hit_stop else c_close
                cover_p = exit_base * (1.0 + slippage_rate)
                num_shares = abs(position)
                gross_pnl = (entry_price - cover_p) * num_shares
                comm = (num_shares * cover_p) * commission_rate
                net_pnl = gross_pnl - comm
                pnl_pct = (net_pnl / (entry_cost + 1e-9)) * 100.0
                cash += (entry_cost + net_pnl)

                reason_code = 1.0 if hit_stop else (2.0 if ai_exit else 3.0)
                records[trade_count] = np.array([-1.0, entry_idx, i, entry_price, cover_p, net_pnl, pnl_pct, reason_code])
                trade_count += 1

                position = 0.0

        if position == 0.0 and pending_action == 0:
            if signal_arr[i] == 1:
                pending_action = 1
            elif signal_arr[i] == -1:
                pending_action = -1

        if position > 0.0:
            equity_list[i] = cash + (position * c_close)
        elif position < 0.0:
            unrealized = (entry_price - c_close) * abs(position)
            equity_list[i] = cash + entry_cost + unrealized
        else:
            equity_list[i] = cash

    return equity_list, records[:trade_count]


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

    sharpe_ratio = ((mean_return / std_return) * np.sqrt(252)) if std_return > 0.0 else 0.0

    gross_profits = sum(t["PnL"] for t in trade_log if t["PnL"] > 0)
    gross_losses = abs(sum(t["PnL"] for t in trade_log if t["PnL"] < 0))

    profit_factor = (gross_profits / gross_losses) if gross_losses > 0.0 else (float("inf") if gross_profits > 0.0 else 0.0)

    avg_duration = sum(t["Duration_Days"] for t in trade_log) / total_trades if total_trades > 0 else 0.0

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
        return df, [], {"total_pnl": 0.0, "equity_curve": pd.Series([initial_capital])}

    df = df.copy()

    close_arr = np.ascontiguousarray(df["Close"].to_numpy(dtype=np.float64))
    open_arr = np.ascontiguousarray(df["Open"].to_numpy(dtype=np.float64) if "Open" in df.columns else close_arr)
    high_arr = np.ascontiguousarray(df["High"].to_numpy(dtype=np.float64))
    low_arr = np.ascontiguousarray(df["Low"].to_numpy(dtype=np.float64))
    signal_arr = np.ascontiguousarray(df["Signal"].to_numpy(dtype=np.int8))
    ai_prob_arr = np.ascontiguousarray(df["AI_Probability"].to_numpy(dtype=np.float64) if "AI_Probability" in df.columns else np.full(len(df), np.nan))
    atr_arr = np.ascontiguousarray(df["ATR"].to_numpy(dtype=np.float64) if "ATR" in df.columns else np.zeros(len(df), dtype=np.float64))

    equity_arr, raw_records = _run_backtest_kernel(
        close_arr, open_arr, high_arr, low_arr, signal_arr, ai_prob_arr, atr_arr,
        initial_capital, risk_per_trade, commission_rate, slippage_rate,
        use_atr_stop, atr_multiplier, ai_exit_threshold
    )

    dates = df.index
    trade_log = []
    reason_map = {1.0: "ATR Stop Loss", 2.0: "AI Exit", 3.0: "Signal Exit"}

    for rec in raw_records:
        t_type = "LONG" if rec[0] == 1.0 else "SHORT"
        entry_idx = int(rec[1])
        exit_idx = int(rec[2])
        dur_days = (pd.Timestamp(dates[exit_idx]) - pd.Timestamp(dates[entry_idx])).days

        trade_log.append({
            "Type": t_type,
            "Entry_Date": dates[entry_idx],
            "Exit_Date": dates[exit_idx],
            "Entry_Price": rec[3],
            "Exit_Price": rec[4],
            "PnL": rec[5],
            "PnL_Pct": rec[6],
            "Duration_Days": dur_days,
            "Reason": f"{t_type} {reason_map.get(rec[7], 'Exit')}",
        })

    equity_series = pd.Series(equity_arr, index=df.index)
    df["Portfolio_Equity"] = equity_series

    metrics = calculate_backtest_metrics(df, trade_log, initial_capital, equity_series)

    return df, trade_log, metrics