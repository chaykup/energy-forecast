import pandas as pd
import numpy as np

def backtest_strategy(
        actual_lmps: np.ndarray,
        predicted_lmps: np.ndarray,
        model_name: str,
        battery_capacity_mwh: float = 100,
        max_charge_rate_mw: float = 25,     # MW per hour
        efficiency: float = 0.90,           # Round-trip efficiency
        min_price_spread: float = 5.0,      # Minimum $/Mwh spread to act
) -> dict:
    cash = 0.0
    stored_mwh = 0.0
    charge_cost_basis = 0.0
    trades = []
    hourly_pnl = []
    cumulative_pnl = []

    for i in range(len(actual_lmps) - 1):
        current_price = actual_lmps[i]
        predicted_next = predicted_lmps[i]  # Forecast for this hour
        actual_next = actual_lmps[i + 1] if i + 1 < len(actual_lmps) else current_price

        spread = predicted_next - current_price
        hour_pnl = 0.0

        # CHARGE: buy at lower prices
        if spread > min_price_spread and stored_mwh < battery_capacity_mwh:
            charge_mwh = min(max_charge_rate_mw, battery_capacity_mwh - stored_mwh)
            cost = charge_mwh * current_price
            cash -= cost
            stored_mwh += charge_mwh * (efficiency ** 0.5)
            hour_pnl = -cost
            trades.append({"hour": i, "action": "CHARGE", "mwh": charge_mwh, "price": current_price})

        # DISCHARGE: sell at higher prices
        elif spread < -min_price_spread and stored_mwh > 0:
            discharge_mwh = min(max_charge_rate_mw, stored_mwh)
            revenue = discharge_mwh * (efficiency ** 0.5) * current_price
            cash += revenue
            stored_mwh -= discharge_mwh
            hour_pnl = revenue
            trades.append({"hour": i, "action": "DISCHARGE", "mwh": discharge_mwh, "price": current_price})

        hourly_pnl.append(hour_pnl)
        cumulative_pnl.append(cash)

    # Value remaining stored energy at last known price
    terminal_value = stored_mwh * actual_lmps[-1] * (efficiency ** 0.5)
    total_pnl = cash + terminal_value

    return {
        "model": model_name,
        "total_pnl": float(total_pnl),
        "cash_pnl": float(cash),
        "terminal_value": float(terminal_value),
        "num_trades": len(trades),
        "num_charges": sum(1 for t in trades if t["action"] == "CHARGE"),
        "num_discharges": sum(1 for t in trades if t["action"] == "DISCHARGE"),
        "avg_charge_price": float(np.mean([t["price"] for t in trades if t["action"] == "CHARGE"]) if trades else 0),
        "avg_discharge_price": float(np.mean([t["price"] for t in trades if t["action"] == "DISCHARGE"]) if trades else 0),
        "cumulative_pnl": cumulative_pnl,
        "hourly_pnl": hourly_pnl,
    }

def naive_baseline_pnl(
        actual_lmps: np.ndarray,
        battery_capacity_mwh: float = 100,
        max_charge_rate_mw: float = 25,
        efficiency: float = 0.90,
) -> dict:
    cash = 0.0
    stored_mwh = 0.0
    trades = []
    cumulative_pnl = []

    for i, price in enumerate(actual_lmps):
        hour_of_day = i % 24
        if hour_of_day in range(0,7) and stored_mwh < battery_capacity_mwh:
            charge_mwh = min(max_charge_rate_mw, battery_capacity_mwh - stored_mwh)
            cash -= charge_mwh * price
            stored_mwh += charge_mwh * (efficiency ** 0.5)
            trades.append({"hour": i, "action": "CHARGE", "price": price})
        elif hour_of_day in range(16, 21) and stored_mwh > 0:
            discharge_mwh = min(max_charge_rate_mw, stored_mwh)
            cash += discharge_mwh * (efficiency ** 0.5) * price
            stored_mwh -= discharge_mwh
            trades.append({"hour": i, "action": "DISCHARGE", "price": price})
        cumulative_pnl.append(cash)

    terminal_value = stored_mwh * actual_lmps[-1] * (efficiency ** 0.5)
    return {
        "model": "naive_baseline",
        "total_pnl": float(cash + terminal_value),
        "num_trades": len(trades),
        "cumulative_pnl": cumulative_pnl,
    }