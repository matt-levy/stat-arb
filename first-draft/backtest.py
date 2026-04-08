import numpy as np
import pandas as pd

from config import (
    COMMISSION_BPS_PER_LEG,
    EVALUATION_MODE,
    INITIAL_CAPITAL,
    MAX_GROSS_EXPOSURE_FRACTION,
    SLIPPAGE_BPS_PER_LEG,
    TRAIN_TEST_SPLIT_FRACTION,
    WALK_FORWARD_STEP_WINDOW,
    WALK_FORWARD_TEST_WINDOW,
    WALK_FORWARD_TRAIN_WINDOW,
)
from signals import build_signal_frame, generate_strategy_state, signal_mult, vol_mult


def bps_to_decimal(bps: float) -> float:
    return bps / 10_000.0


def total_trading_cost_rate() -> float:
    return bps_to_decimal(COMMISSION_BPS_PER_LEG + SLIPPAGE_BPS_PER_LEG)


def compute_trading_cost(traded_notional: float) -> float:
    return traded_notional * total_trading_cost_rate()


def compute_metrics(df: pd.DataFrame, initial_capital: float = INITIAL_CAPITAL):
    if df.empty:
        return np.nan, np.nan, np.nan, np.nan

    total_return = df["equity"].iloc[-1] / initial_capital - 1
    std_dev = df["ret"].std()
    sharpe = (df["ret"].mean() / std_dev * np.sqrt(252)) if std_dev > 0 else np.nan
    drawdown = (df["equity"] / df["equity"].cummax() - 1).min()
    trade_count = int(df["action_type"].isin(["ENTER_LONG", "ENTER_SHORT", "EXIT", "FLIP_TO_LONG", "FLIP_TO_SHORT"]).sum())

    return total_return, sharpe, drawdown, trade_count


def aggregate_backtest_frames(frames: list[pd.DataFrame], initial_capital: float = INITIAL_CAPITAL) -> pd.DataFrame:
    if not frames:
        return _combine_with_empty_columns(pd.DataFrame())

    combined_df = pd.concat(frames).sort_index().copy()
    if combined_df.empty:
        return _combine_with_empty_columns(combined_df)

    equity = initial_capital
    equity_series = []

    for daily_return in combined_df["ret"].fillna(0.0):
        equity *= 1 + daily_return
        equity_series.append(equity)

    combined_df["equity"] = equity_series
    return combined_df


def _leg_notionals(equity: float, gross_multiplier: float, beta: float, position: float):
    if position == 0 or pd.isna(beta) or pd.isna(gross_multiplier) or gross_multiplier <= 0:
        return 0.0, 0.0

    gross_notional = equity * MAX_GROSS_EXPOSURE_FRACTION * gross_multiplier
    y_weight = 1 / (1 + abs(beta))
    x_weight = abs(beta) / (1 + abs(beta))

    y_notional = position * gross_notional * y_weight
    x_notional = -position * gross_notional * x_weight
    return y_notional, x_notional


def _combine_with_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    df["equity"] = pd.Series(dtype=float)
    df["ret"] = pd.Series(dtype=float)
    df["gross_multiplier"] = pd.Series(dtype=float)
    df["gross_multiplier_shifted"] = pd.Series(dtype=float)
    df["y_notional"] = pd.Series(dtype=float)
    df["x_notional"] = pd.Series(dtype=float)
    df["trade_notional"] = pd.Series(dtype=float)
    df["trading_cost"] = pd.Series(dtype=float)
    df["pnl_before_cost"] = pd.Series(dtype=float)
    df["pnl_after_cost"] = pd.Series(dtype=float)
    return df


def run_backtest(
    df: pd.DataFrame,
    y_ticker: str,
    x_ticker: str,
    initial_capital: float = INITIAL_CAPITAL,
):
    backtest_df = df.copy()
    backtest_df[f"{y_ticker}_ret"] = backtest_df[y_ticker].pct_change()
    backtest_df[f"{x_ticker}_ret"] = backtest_df[x_ticker].pct_change()

    backtest_df = generate_strategy_state(build_signal_frame(backtest_df, y_ticker, x_ticker))

    if backtest_df.empty:
        return _combine_with_empty_columns(backtest_df)

    backtest_df["gross_multiplier"] = (
        backtest_df["z"].abs().apply(signal_mult) * backtest_df["sig_vol"].apply(vol_mult)
    )
    backtest_df.loc[backtest_df["pos"] == 0, "gross_multiplier"] = 0.0
    backtest_df["gross_multiplier_shifted"] = backtest_df["gross_multiplier"].shift(1).fillna(0.0)
    backtest_df["pos_shift"] = backtest_df["pos"].shift(1).fillna(0).astype(int)

    equity = initial_capital
    equity_series = []
    returns = []
    y_notionals = []
    x_notionals = []
    trade_notionals = []
    trading_costs = []
    pnl_before_costs = []
    pnl_after_costs = []

    for _, row in backtest_df.iterrows():
        prev_position = int(row["pos_shift"])
        current_position = int(row["pos"])

        prev_y_notional, prev_x_notional = _leg_notionals(
            equity,
            row["gross_multiplier_shifted"],
            row["beta_shifted"],
            prev_position,
        )

        pnl_before_cost = (
            prev_y_notional * row[f"{y_ticker}_ret"]
            + prev_x_notional * row[f"{x_ticker}_ret"]
        )
        equity_after_pnl = equity + pnl_before_cost

        current_y_notional, current_x_notional = _leg_notionals(
            equity_after_pnl,
            row["gross_multiplier"],
            row["beta_shifted"],
            current_position,
        )

        traded_notional = abs(current_y_notional - prev_y_notional) + abs(current_x_notional - prev_x_notional)
        trading_cost = compute_trading_cost(traded_notional)

        pnl_after_cost = pnl_before_cost - trading_cost
        equity = equity_after_pnl - trading_cost

        base_equity = equity - pnl_after_cost
        returns.append(pnl_after_cost / base_equity if base_equity > 0 else 0.0)
        equity_series.append(equity)
        y_notionals.append(current_y_notional)
        x_notionals.append(current_x_notional)
        trade_notionals.append(traded_notional)
        trading_costs.append(trading_cost)
        pnl_before_costs.append(pnl_before_cost)
        pnl_after_costs.append(pnl_after_cost)

    backtest_df["equity"] = equity_series
    backtest_df["ret"] = pd.Series(returns, index=backtest_df.index)
    backtest_df["y_notional"] = pd.Series(y_notionals, index=backtest_df.index)
    backtest_df["x_notional"] = pd.Series(x_notionals, index=backtest_df.index)
    backtest_df["trade_notional"] = pd.Series(trade_notionals, index=backtest_df.index)
    backtest_df["trading_cost"] = pd.Series(trading_costs, index=backtest_df.index)
    backtest_df["pnl_before_cost"] = pd.Series(pnl_before_costs, index=backtest_df.index)
    backtest_df["pnl_after_cost"] = pd.Series(pnl_after_costs, index=backtest_df.index)

    return backtest_df


def evaluate_split(
    df: pd.DataFrame,
    y_ticker: str,
    x_ticker: str,
    initial_capital: float = INITIAL_CAPITAL,
):
    split = int(len(df) * TRAIN_TEST_SPLIT_FRACTION)
    train_df = df.iloc[:split]
    test_df = df.iloc[split:]

    train_bt = run_backtest(train_df, y_ticker, x_ticker, initial_capital=initial_capital)
    test_bt = run_backtest(test_df, y_ticker, x_ticker, initial_capital=initial_capital)

    train_ret, train_sharpe, train_dd, train_trades = compute_metrics(train_bt, initial_capital=initial_capital)
    test_ret, test_sharpe, test_dd, test_trades = compute_metrics(test_bt, initial_capital=initial_capital)

    return {
        "mode": "split",
        "train_backtest": train_bt,
        "test_backtest": test_bt,
        "train_return": train_ret,
        "train_sharpe": train_sharpe,
        "train_dd": train_dd,
        "train_trades": train_trades,
        "test_return": test_ret,
        "test_sharpe": test_sharpe,
        "test_dd": test_dd,
        "test_trades": test_trades,
        "walk_forward_folds": 0,
    }


def evaluate_walk_forward(
    df: pd.DataFrame,
    y_ticker: str,
    x_ticker: str,
    train_window: int = WALK_FORWARD_TRAIN_WINDOW,
    test_window: int = WALK_FORWARD_TEST_WINDOW,
    step_window: int = WALK_FORWARD_STEP_WINDOW,
    initial_capital: float = INITIAL_CAPITAL,
):
    train_metrics = []
    test_folds = []
    fold_count = 0

    for start in range(0, len(df) - train_window - test_window + 1, step_window):
        train_slice = df.iloc[start : start + train_window]
        test_slice = df.iloc[start + train_window : start + train_window + test_window]

        train_bt = run_backtest(train_slice, y_ticker, x_ticker, initial_capital=initial_capital)
        test_bt = run_backtest(test_slice, y_ticker, x_ticker, initial_capital=initial_capital)

        train_ret, train_sharpe, train_dd, train_trades = compute_metrics(train_bt, initial_capital=initial_capital)
        test_ret, test_sharpe, test_dd, test_trades = compute_metrics(test_bt, initial_capital=initial_capital)

        train_metrics.append(
            {
                "train_return": train_ret,
                "train_sharpe": train_sharpe,
                "train_dd": train_dd,
                "train_trades": train_trades,
            }
        )

        fold_test = test_bt.copy()
        fold_test["fold"] = fold_count
        test_folds.append(fold_test)
        fold_count += 1

    if not test_folds:
        empty_bt = _combine_with_empty_columns(pd.DataFrame())
        return {
            "mode": "walk_forward",
            "train_backtest": empty_bt,
            "test_backtest": empty_bt,
            "train_return": np.nan,
            "train_sharpe": np.nan,
            "train_dd": np.nan,
            "train_trades": np.nan,
            "test_return": np.nan,
            "test_sharpe": np.nan,
            "test_dd": np.nan,
            "test_trades": np.nan,
            "walk_forward_folds": 0,
        }

    combined_test = aggregate_backtest_frames(test_folds, initial_capital=initial_capital)
    test_ret, test_sharpe, test_dd, test_trades = compute_metrics(combined_test, initial_capital=initial_capital)
    train_metrics_df = pd.DataFrame(train_metrics)

    return {
        "mode": "walk_forward",
        "train_backtest": pd.DataFrame(train_metrics),
        "test_backtest": combined_test,
        "train_return": train_metrics_df["train_return"].mean(),
        "train_sharpe": train_metrics_df["train_sharpe"].mean(),
        "train_dd": train_metrics_df["train_dd"].mean(),
        "train_trades": train_metrics_df["train_trades"].mean(),
        "test_return": test_ret,
        "test_sharpe": test_sharpe,
        "test_dd": test_dd,
        "test_trades": test_trades,
        "walk_forward_folds": fold_count,
    }


def evaluate_pair(
    df: pd.DataFrame,
    y_ticker: str,
    x_ticker: str,
    mode: str = EVALUATION_MODE,
    initial_capital: float = INITIAL_CAPITAL,
):
    if mode == "walk_forward":
        return evaluate_walk_forward(df, y_ticker, x_ticker, initial_capital=initial_capital)
    return evaluate_split(df, y_ticker, x_ticker, initial_capital=initial_capital)
