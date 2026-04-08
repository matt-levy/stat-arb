import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

from config import (
    ENABLE_RECENT_COINTEGRATION_CHECK,
    ENTRY_THRESHOLD,
    EXIT_THRESHOLD,
    HALF_LIFE_WINDOW,
    MAX_ABS_BETA,
    MAX_ENTRY_ZSCORE,
    MAX_HALF_LIFE,
    MAX_HOLD_DAYS,
    MAX_RECENT_BETA_STD,
    MAX_RECENT_COINTEGRATION_PVALUE,
    MAX_RECENT_SPREAD_VOL_RATIO,
    MAX_SIGNAL_MULTIPLIER,
    MIN_ABS_BETA,
    MIN_CORRELATION,
    MIN_RECENT_CORRELATION_MEAN,
    MIN_STABILITY_OBS,
    MIN_TRADES,
    RECENT_COINTEGRATION_WINDOW,
    RECENT_SPREAD_VOL_WINDOW,
    REGRESSION_WINDOW,
    ROLLING_CORRELATION_WINDOW,
    SOFT_PVALUE_THRESHOLD,
    STABILITY_LOOKBACK,
    TARGET_DAILY_VOL,
    VOL_TARGET_WINDOW,
    ZSCORE_WINDOW,
    MAX_RECENT_CORRELATION_STD,
)


def rolling_ols_parameters(y: pd.Series, x: pd.Series, window: int):
    alphas = []
    betas = []

    for i in range(len(y)):
        if i < window - 1:
            alphas.append(np.nan)
            betas.append(np.nan)
            continue

        y_window = y.iloc[i - window + 1 : i + 1]
        x_window = x.iloc[i - window + 1 : i + 1]

        model = sm.OLS(y_window, sm.add_constant(x_window)).fit()
        alphas.append(model.params.iloc[0])
        betas.append(model.params.iloc[1])

    return pd.Series(alphas, index=y.index), pd.Series(betas, index=y.index)


def clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def estimate_half_life(spread_window: pd.Series):
    spread_lag = spread_window.shift(1)
    delta = spread_window - spread_lag

    reg_df = pd.DataFrame({"lag": spread_lag, "delta": delta}).dropna()
    if len(reg_df) < 10:
        return np.nan

    model = sm.OLS(reg_df["delta"], sm.add_constant(reg_df["lag"])).fit()
    beta = model.params["lag"]

    if pd.isna(beta) or beta >= 0:
        return np.nan

    half_life = -np.log(2) / beta
    return half_life if half_life > 0 else np.nan


def rolling_half_life(spread: pd.Series, window: int):
    values = []
    for i in range(len(spread)):
        if i < window - 1:
            values.append(np.nan)
            continue
        values.append(estimate_half_life(spread.iloc[i - window + 1 : i + 1]))
    return pd.Series(values, index=spread.index)


def signal_mult(zscore: float) -> float:
    return clip(abs(zscore) / ENTRY_THRESHOLD, 1.0, MAX_SIGNAL_MULTIPLIER)


def vol_mult(volatility: float) -> float:
    if pd.isna(volatility) or volatility <= 0:
        return 1.0
    return clip(TARGET_DAILY_VOL / volatility, 0.25, 3.0)


def _safe_cointegration_pvalue(y: pd.Series, x: pd.Series):
    try:
        _, pvalue, _ = coint(y, x)
        return pvalue
    except Exception:
        return np.nan


def _stability_failure_reasons(metrics: dict) -> list[str]:
    reasons = []

    if pd.isna(metrics["recent_corr_mean"]) or metrics["recent_corr_mean"] < MIN_RECENT_CORRELATION_MEAN:
        reasons.append("LOW_RECENT_CORRELATION")

    if pd.isna(metrics["recent_corr_std"]) or metrics["recent_corr_std"] > MAX_RECENT_CORRELATION_STD:
        reasons.append("UNSTABLE_CORRELATION")

    if pd.isna(metrics["recent_beta_std"]) or metrics["recent_beta_std"] > MAX_RECENT_BETA_STD:
        reasons.append("UNSTABLE_BETA")

    if pd.isna(metrics["recent_spread_vol_ratio"]) or metrics["recent_spread_vol_ratio"] > MAX_RECENT_SPREAD_VOL_RATIO:
        reasons.append("ELEVATED_SPREAD_VOL")

    if ENABLE_RECENT_COINTEGRATION_CHECK:
        recent_coint_pvalue = metrics["recent_coint_pvalue"]
        if pd.isna(recent_coint_pvalue) or recent_coint_pvalue > MAX_RECENT_COINTEGRATION_PVALUE:
            reasons.append("RECENT_COINTEGRATION_FAIL")

    return reasons


def compute_pair_stability_metrics(df: pd.DataFrame, y_ticker: str, x_ticker: str):
    y_prices = df[y_ticker]
    x_prices = df[x_ticker]
    y_returns = y_prices.pct_change()
    x_returns = x_prices.pct_change()

    rolling_corr = y_returns.rolling(ROLLING_CORRELATION_WINDOW).corr(x_returns)
    recent_corr = rolling_corr.tail(STABILITY_LOOKBACK).dropna()

    alphas, betas = rolling_ols_parameters(y_prices, x_prices, REGRESSION_WINDOW)
    spread = y_prices - (alphas + betas * x_prices)

    recent_beta = betas.tail(STABILITY_LOOKBACK).dropna()
    recent_spread_vol = spread.tail(RECENT_SPREAD_VOL_WINDOW).std()
    historical_spread_vol = spread.dropna().std()

    spread_vol_ratio = np.nan
    if pd.notna(recent_spread_vol) and pd.notna(historical_spread_vol) and historical_spread_vol > 0:
        spread_vol_ratio = recent_spread_vol / historical_spread_vol

    recent_coint_pvalue = np.nan
    if ENABLE_RECENT_COINTEGRATION_CHECK:
        recent_window = df.tail(RECENT_COINTEGRATION_WINDOW)
        if len(recent_window) >= MIN_STABILITY_OBS:
            recent_coint_pvalue = _safe_cointegration_pvalue(
                recent_window[y_ticker],
                recent_window[x_ticker],
            )

    metrics = {
        "recent_corr_mean": recent_corr.mean() if len(recent_corr) >= MIN_STABILITY_OBS else np.nan,
        "recent_corr_std": recent_corr.std() if len(recent_corr) >= MIN_STABILITY_OBS else np.nan,
        "recent_beta_std": recent_beta.std() if len(recent_beta) >= MIN_STABILITY_OBS else np.nan,
        "recent_spread_vol": recent_spread_vol,
        "historical_spread_vol": historical_spread_vol,
        "recent_spread_vol_ratio": spread_vol_ratio,
        "recent_coint_pvalue": recent_coint_pvalue,
    }

    failure_reasons = _stability_failure_reasons(metrics)
    metrics["passes_stability"] = len(failure_reasons) == 0
    metrics["stability_rejection_reason"] = "|".join(failure_reasons) if failure_reasons else ""
    return metrics


def compute_pair_filters(df: pd.DataFrame, y_ticker: str, x_ticker: str):
    y_prices = df[y_ticker]
    x_prices = df[x_ticker]

    y_returns = y_prices.pct_change()
    x_returns = x_prices.pct_change()

    pvalue = _safe_cointegration_pvalue(y_prices, x_prices)

    try:
        model = sm.OLS(y_prices, sm.add_constant(x_prices)).fit()
        beta = model.params.iloc[1]
    except Exception:
        beta = np.nan

    correlation = y_returns.corr(x_returns)
    stability_metrics = compute_pair_stability_metrics(df, y_ticker, x_ticker)

    passes_beta = pd.notna(beta) and (MIN_ABS_BETA <= abs(beta) <= MAX_ABS_BETA)
    passes_corr = pd.notna(correlation) and correlation >= MIN_CORRELATION

    rejection_reasons = []
    if not passes_beta:
        rejection_reasons.append("BETA_FILTER")
    if not passes_corr:
        rejection_reasons.append("CORRELATION_FILTER")
    if not stability_metrics["passes_stability"]:
        rejection_reasons.append(stability_metrics["stability_rejection_reason"])

    return {
        "pvalue": pvalue,
        "beta": beta,
        "correlation": correlation,
        "passes_beta": passes_beta,
        "passes_corr": passes_corr,
        "passes_stability": stability_metrics["passes_stability"],
        "rejection_reason": "|".join(filter(None, rejection_reasons)),
        **stability_metrics,
    }


def score_pair(pvalue, beta, correlation, train_sharpe, test_sharpe, trade_count):
    score = 0

    if pd.notna(test_sharpe):
        if test_sharpe > 1.0:
            score += 4
        elif test_sharpe > 0.75:
            score += 3
        elif test_sharpe > 0.5:
            score += 2

    if pd.notna(train_sharpe):
        if train_sharpe > 1.0:
            score += 2
        elif train_sharpe > 0.5:
            score += 1

    if pd.notna(beta) and MIN_ABS_BETA <= abs(beta) <= MAX_ABS_BETA:
        score += 1

    if pd.notna(correlation):
        if correlation > 0.80:
            score += 1
        elif correlation > 0.65:
            score += 0.5

    if pd.notna(pvalue):
        if pvalue <= 0.05:
            score += 1
        elif pvalue <= SOFT_PVALUE_THRESHOLD:
            score += 0.5

    if pd.notna(trade_count):
        if trade_count >= 60:
            score += 1
        elif trade_count >= MIN_TRADES:
            score += 0.5

    return score


def build_signal_frame(df: pd.DataFrame, y_ticker: str, x_ticker: str) -> pd.DataFrame:
    signal_df = df.copy()

    signal_df["alpha"], signal_df["beta"] = rolling_ols_parameters(
        signal_df[y_ticker],
        signal_df[x_ticker],
        REGRESSION_WINDOW,
    )
    signal_df["alpha_shifted"] = signal_df["alpha"].shift(1)
    signal_df["beta_shifted"] = signal_df["beta"].shift(1)

    signal_df["spread"] = signal_df[y_ticker] - (
        signal_df["alpha_shifted"] + signal_df["beta_shifted"] * signal_df[x_ticker]
    )
    signal_df["mean"] = signal_df["spread"].rolling(ZSCORE_WINDOW).mean()
    signal_df["std"] = signal_df["spread"].rolling(ZSCORE_WINDOW).std()
    signal_df["z"] = (signal_df["spread"] - signal_df["mean"]) / signal_df["std"]

    signal_df["hl"] = rolling_half_life(signal_df["spread"], HALF_LIFE_WINDOW)
    signal_df["hl_ok"] = signal_df["hl"] <= MAX_HALF_LIFE

    signal_df["norm"] = signal_df["spread"] / signal_df["std"]
    signal_df["norm_chg"] = signal_df["norm"].diff()
    signal_df["sig_vol"] = signal_df["norm_chg"].rolling(VOL_TARGET_WINDOW).std().shift(1)
    signal_df["spread_vol"] = signal_df["spread"].rolling(RECENT_SPREAD_VOL_WINDOW).std().shift(1)

    return signal_df.dropna().copy()


def get_entry_decision(current_z: float, previous_z: float, hl_ok: bool):
    if pd.isna(current_z) or pd.isna(previous_z):
        return 0, "INSUFFICIENT_SIGNAL_CONTEXT"

    if not hl_ok:
        return 0, "HALF_LIFE_FILTER"

    if abs(current_z) > MAX_ENTRY_ZSCORE:
        return 0, "EXTREME_Z_REJECT"

    if current_z <= -ENTRY_THRESHOLD:
        if current_z > previous_z:
            return 1, ""
        return 0, "LONG_NO_REVERSION_CONFIRMATION"

    if current_z >= ENTRY_THRESHOLD:
        if current_z < previous_z:
            return -1, ""
        return 0, "SHORT_NO_REVERSION_CONFIRMATION"

    return 0, ""


def get_exit_decision(position: int, current_z: float, hold_days: int):
    if position == 0 or pd.isna(current_z):
        return False, ""

    if hold_days >= MAX_HOLD_DAYS:
        return True, "TIME_STOP"

    if position == 1 and current_z >= -EXIT_THRESHOLD:
        return True, "MEAN_REVERT"

    if position == -1 and current_z <= EXIT_THRESHOLD:
        return True, "MEAN_REVERT"

    return False, ""


def generate_strategy_state(signal_df: pd.DataFrame) -> pd.DataFrame:
    state_df = signal_df.copy()
    if state_df.empty:
        return state_df

    position = 0
    hold_days = 0

    positions = []
    hold_days_list = []
    action_types = []
    exit_reasons = []
    rejection_reasons = []

    prior_zscores = state_df["z"].shift(1)

    for i in range(len(state_df)):
        current_z = state_df["z"].iloc[i]
        previous_z = prior_zscores.iloc[i]
        hl_ok = bool(state_df["hl_ok"].iloc[i])

        candidate_hold_days = hold_days + 1 if position != 0 else 0
        should_exit, exit_reason = get_exit_decision(position, current_z, candidate_hold_days)
        entry_signal, rejection_reason = get_entry_decision(current_z, previous_z, hl_ok)

        action_type = "FLAT"

        if position == 0:
            if entry_signal == 1:
                position = 1
                hold_days = 1
                action_type = "ENTER_LONG"
                rejection_reason = ""
            elif entry_signal == -1:
                position = -1
                hold_days = 1
                action_type = "ENTER_SHORT"
                rejection_reason = ""
            else:
                hold_days = 0
                action_type = "NO_TRADE"
        else:
            if should_exit:
                if entry_signal != 0 and entry_signal != position:
                    position = entry_signal
                    hold_days = 1
                    action_type = "FLIP_TO_LONG" if entry_signal == 1 else "FLIP_TO_SHORT"
                    rejection_reason = ""
                else:
                    position = 0
                    hold_days = 0
                    action_type = "EXIT"
            elif entry_signal != 0 and entry_signal != position:
                position = entry_signal
                hold_days = 1
                exit_reason = "SIGNAL_FLIP"
                action_type = "FLIP_TO_LONG" if entry_signal == 1 else "FLIP_TO_SHORT"
                rejection_reason = ""
            else:
                hold_days = candidate_hold_days
                action_type = "HOLD_LONG" if position == 1 else "HOLD_SHORT"
                exit_reason = ""
                rejection_reason = ""

        positions.append(position)
        hold_days_list.append(hold_days)
        action_types.append(action_type)
        exit_reasons.append(exit_reason)
        rejection_reasons.append(rejection_reason)

    state_df["pos"] = pd.Series(positions, index=state_df.index)
    state_df["hold_days"] = pd.Series(hold_days_list, index=state_df.index)
    state_df["action_type"] = pd.Series(action_types, index=state_df.index)
    state_df["exit_reason"] = pd.Series(exit_reasons, index=state_df.index)
    state_df["rejection_reason"] = pd.Series(rejection_reasons, index=state_df.index)

    return state_df


def get_latest_signal(df: pd.DataFrame, y_ticker: str, x_ticker: str):
    state_df = generate_strategy_state(build_signal_frame(df, y_ticker, x_ticker))
    if state_df.empty:
        return None

    last_row = state_df.iloc[-1]

    if last_row["pos"] == 1:
        action = f"LONG {y_ticker} / SHORT {x_ticker}"
    elif last_row["pos"] == -1:
        action = f"SHORT {y_ticker} / LONG {x_ticker}"
    else:
        action = "FLAT"

    return {
        "date": str(state_df.index[-1].date()),
        "y_price": float(last_row[y_ticker]),
        "x_price": float(last_row[x_ticker]),
        "zscore": float(last_row["z"]),
        "beta": float(last_row["beta_shifted"]),
        "half_life": float(last_row["hl"]) if pd.notna(last_row["hl"]) else np.nan,
        "signal_vol": float(last_row["sig_vol"]) if pd.notna(last_row["sig_vol"]) else np.nan,
        "spread_vol": float(last_row["spread_vol"]) if pd.notna(last_row["spread_vol"]) else np.nan,
        "hold_days": int(last_row["hold_days"]),
        "position": int(last_row["pos"]),
        "action": action,
        "action_type": last_row["action_type"],
        "exit_reason": last_row["exit_reason"],
        "rejection_reason": last_row["rejection_reason"],
    }
