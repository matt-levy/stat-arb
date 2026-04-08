import itertools
from pathlib import Path
from datetime import datetime

import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

# ----------------------------
# Settings
# ----------------------------
LOOKBACK_PERIOD = "2y"
INITIAL_CAPITAL = 100_000.0
MAX_GROSS_EXPOSURE_FRACTION = 2.0
COST_BPS_PER_LEG = 0.00050

REGRESSION_WINDOW = 60
ZSCORE_WINDOW = 30
VOL_TARGET_WINDOW = 20
HALF_LIFE_WINDOW = 60

ENTRY_THRESHOLD = 2.0
EXIT_THRESHOLD = 0.5
MAX_HALF_LIFE = 20

MAX_SIGNAL_MULTIPLIER = 3.0
TARGET_DAILY_VOL = 0.01

TOP_N_PAIRS = 3
MAX_PAIRS_PER_BUCKET = 1
MAX_PAIR_WEIGHT = 0.45

SOFT_PVALUE_THRESHOLD = 0.10
MIN_ABS_BETA = 0.20
MAX_ABS_BETA = 2.50
MIN_LIVE_ABS_BETA = 0.10

MIN_TEST_SHARPE = 0.25
MIN_TRAIN_SHARPE = -0.10
MIN_CORRELATION = 0.50
MIN_TRADES = 8
MIN_ROWS = 300

VERBOSE = False

LOG_PATH = Path("paper_trading_log.csv")

UNIVERSE = {
    "semis": ["NVDA", "AMD", "AVGO", "QCOM", "TXN", "MU"],
    "payments": ["V", "MA", "AXP", "PYPL"],
    "oil_majors": ["XOM", "CVX", "COP", "EOG", "OXY", "SLB"],
    "banks": ["GS", "MS", "JPM", "BAC", "C", "WFC"],
    "retail": ["WMT", "TGT", "COST", "DG", "DLTR"],
    "home": ["HD", "LOW"],
    "beverages_food": ["KO", "PEP", "MNST", "KDP", "MCD", "SBUX"],
    "software": ["MSFT", "ADBE", "CRM", "ORCL", "INTU"],
}


# ----------------------------
# Helpers
# ----------------------------
def log(msg: str) -> None:
    if VERBOSE:
        print(msg)


def download_and_prepare(ticker: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        period=LOOKBACK_PERIOD,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    if df.empty:
        return pd.DataFrame(columns=[ticker])

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Close"]].copy()
    df = df.rename(columns={"Close": ticker})
    return df


def rolling_ols_parameters(y: pd.Series, x: pd.Series, window: int):
    alphas = []
    betas = []

    for i in range(len(y)):
        if i < window - 1:
            alphas.append(np.nan)
            betas.append(np.nan)
            continue

        y_window = y.iloc[i - window + 1:i + 1]
        x_window = x.iloc[i - window + 1:i + 1]

        model = sm.OLS(y_window, sm.add_constant(x_window)).fit()
        alphas.append(model.params.iloc[0])
        betas.append(model.params.iloc[1])

    return pd.Series(alphas, index=y.index), pd.Series(betas, index=y.index)


def clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def estimate_half_life(spread_window: pd.Series):
    spread_lag = spread_window.shift(1)
    delta = spread_window - spread_lag

    reg_df = pd.DataFrame({
        "lag": spread_lag,
        "delta": delta
    }).dropna()

    if len(reg_df) < 10:
        return np.nan

    model = sm.OLS(reg_df["delta"], sm.add_constant(reg_df["lag"])).fit()
    beta = model.params["lag"]

    if pd.isna(beta) or beta >= 0:
        return np.nan

    hl = -np.log(2) / beta
    return hl if hl > 0 else np.nan


def rolling_half_life(spread: pd.Series, window: int):
    out = []
    for i in range(len(spread)):
        if i < window - 1:
            out.append(np.nan)
            continue
        out.append(estimate_half_life(spread.iloc[i - window + 1:i + 1]))
    return pd.Series(out, index=spread.index)


def signal_mult(z: float) -> float:
    return clip(abs(z) / ENTRY_THRESHOLD, 1.0, MAX_SIGNAL_MULTIPLIER)


def vol_mult(vol: float) -> float:
    if pd.isna(vol) or vol <= 0:
        return 1.0
    return clip(TARGET_DAILY_VOL / vol, 0.25, 3.0)


def compute_pair_filters(df: pd.DataFrame, y_ticker: str, x_ticker: str):
    y = df[y_ticker]
    x = df[x_ticker]

    y_ret = y.pct_change()
    x_ret = x.pct_change()

    try:
        _, pvalue, _ = coint(y, x)
    except Exception:
        pvalue = np.nan

    try:
        model = sm.OLS(y, sm.add_constant(x)).fit()
        beta = model.params.iloc[1]
    except Exception:
        beta = np.nan

    correlation = y_ret.corr(x_ret)

    return {
        "pvalue": pvalue,
        "beta": beta,
        "correlation": correlation,
        "passes_beta": pd.notna(beta) and (MIN_ABS_BETA <= abs(beta) <= MAX_ABS_BETA),
        "passes_corr": pd.notna(correlation) and correlation >= MIN_CORRELATION,
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


def weight_from_row(row):
    test_sharpe_component = max(row["test_sharpe"], 0) if pd.notna(row["test_sharpe"]) else 0
    train_sharpe_component = max(row["train_sharpe"], 0) if pd.notna(row["train_sharpe"]) else 0
    test_return_component = max(row["test_return"], 0) if pd.notna(row["test_return"]) else 0
    corr_component = max(row["correlation"], 0) if pd.notna(row["correlation"]) else 0

    test_dd_penalty = abs(row["test_dd"]) if pd.notna(row["test_dd"]) else 0
    dd_multiplier = 1 / (1 + test_dd_penalty * 10)

    return (
        1.0
        + 2.5 * test_sharpe_component
        + 0.75 * train_sharpe_component
        + 10.0 * test_return_component
        + 0.5 * corr_component
    ) * dd_multiplier


def generate_pair_candidates(universe_dict):
    pairs = []
    for bucket, tickers in universe_dict.items():
        for y, x in itertools.combinations(tickers, 2):
            pairs.append((bucket, y, x))
    return pairs


def normalize_capped_weights(raw_weights: pd.Series, cap: float) -> pd.Series:
    weights = raw_weights.copy().astype(float)

    if weights.empty:
        return weights

    weights = weights / weights.sum()

    for _ in range(20):
        over_mask = weights > cap
        if not over_mask.any():
            break

        excess = (weights[over_mask] - cap).sum()
        weights[over_mask] = cap

        under_mask = weights < cap
        under_sum = weights[under_mask].sum()

        if under_sum <= 0 or excess <= 0:
            break

        weights[under_mask] += weights[under_mask] / under_sum * excess
        weights = weights / weights.sum()

    return weights / weights.sum()


def compute_metrics(df: pd.DataFrame, initial_capital: float = INITIAL_CAPITAL):
    if df.empty:
        return np.nan, np.nan, np.nan, np.nan

    total_return = df["equity"].iloc[-1] / initial_capital - 1
    std = df["ret"].std()
    sharpe = (df["ret"].mean() / std * np.sqrt(252)) if std > 0 else np.nan
    dd = (df["equity"] / df["equity"].cummax() - 1).min()
    trade_count = int((df["pos"].diff().fillna(0) != 0).sum())

    return total_return, sharpe, dd, trade_count


def run_backtest(df: pd.DataFrame, y_ticker: str, x_ticker: str, initial_capital: float = INITIAL_CAPITAL):
    df = df.copy()

    df[f"{y_ticker}_ret"] = df[y_ticker].pct_change()
    df[f"{x_ticker}_ret"] = df[x_ticker].pct_change()

    df["alpha"], df["beta"] = rolling_ols_parameters(df[y_ticker], df[x_ticker], REGRESSION_WINDOW)
    df["alpha_shifted"] = df["alpha"].shift(1)
    df["beta_shifted"] = df["beta"].shift(1)

    df["spread"] = df[y_ticker] - (
        df["alpha_shifted"] + df["beta_shifted"] * df[x_ticker]
    )

    df["mean"] = df["spread"].rolling(ZSCORE_WINDOW).mean()
    df["std"] = df["spread"].rolling(ZSCORE_WINDOW).std()
    df["z"] = (df["spread"] - df["mean"]) / df["std"]

    df["hl"] = rolling_half_life(df["spread"], HALF_LIFE_WINDOW)
    df["hl_ok"] = df["hl"] <= MAX_HALF_LIFE

    df["norm"] = df["spread"] / df["std"]
    df["norm_chg"] = df["norm"].diff()
    df["sig_vol"] = df["norm_chg"].rolling(VOL_TARGET_WINDOW).std().shift(1)

    df = df.dropna().copy()

    if df.empty:
        df["equity"] = pd.Series(dtype=float)
        df["ret"] = pd.Series(dtype=float)
        df["pos"] = pd.Series(dtype=float)
        return df

    pos = 0
    positions = []

    long_arm = False
    short_arm = False

    z = df["z"]
    z_prev = z.shift(1)

    for i in range(len(df)):
        cur = z.iloc[i]
        prev = z_prev.iloc[i]

        if pd.isna(cur) or pd.isna(prev):
            positions.append(pos)
            continue

        if cur < -ENTRY_THRESHOLD:
            long_arm = True
        if cur > ENTRY_THRESHOLD:
            short_arm = True

        if pos == 0:
            if long_arm and df["hl_ok"].iloc[i] and cur > prev:
                pos = 1
                long_arm = False
                short_arm = False
            elif short_arm and df["hl_ok"].iloc[i] and cur < prev:
                pos = -1
                long_arm = False
                short_arm = False

        elif pos == 1 and cur >= -EXIT_THRESHOLD:
            pos = 0
            long_arm = False
            short_arm = False

        elif pos == -1 and cur <= EXIT_THRESHOLD:
            pos = 0
            long_arm = False
            short_arm = False

        positions.append(pos)

    df["pos"] = pd.Series(positions, index=df.index)
    df["pos_shift"] = df["pos"].shift(1).fillna(0)

    equity = initial_capital
    equity_series = []
    returns = []

    prev_pos = 0

    for _, row in df.iterrows():
        pos = row["pos_shift"]
        beta = row["beta_shifted"]

        if pd.isna(beta):
            equity_series.append(equity)
            returns.append(0.0)
            continue

        mult = signal_mult(row["z"]) * vol_mult(row["sig_vol"]) if pos != 0 else 0.0
        gross = equity * MAX_GROSS_EXPOSURE_FRACTION * mult

        y_w = 1 / (1 + abs(beta))
        x_w = abs(beta) / (1 + abs(beta))

        y_notional = gross * y_w
        x_notional = gross * x_w

        pnl = (
            pos * y_notional * row[f"{y_ticker}_ret"]
            - pos * x_notional * row[f"{x_ticker}_ret"]
        )

        trade_size = abs(row["pos"] - prev_pos)
        cost = trade_size * (y_notional + x_notional) * COST_BPS_PER_LEG

        pnl -= cost
        equity += pnl

        returns.append(pnl / equity if equity > 0 else 0.0)
        equity_series.append(equity)

        prev_pos = row["pos"]

    df["equity"] = equity_series
    df["ret"] = pd.Series(returns, index=df.index)

    return df


def get_latest_signal(df: pd.DataFrame, y_ticker: str, x_ticker: str):
    df = df.copy()

    df["alpha"], df["beta"] = rolling_ols_parameters(df[y_ticker], df[x_ticker], REGRESSION_WINDOW)
    df["alpha_shifted"] = df["alpha"].shift(1)
    df["beta_shifted"] = df["beta"].shift(1)

    df["spread"] = df[y_ticker] - (
        df["alpha_shifted"] + df["beta_shifted"] * df[x_ticker]
    )
    df["mean"] = df["spread"].rolling(ZSCORE_WINDOW).mean()
    df["std"] = df["spread"].rolling(ZSCORE_WINDOW).std()
    df["z"] = (df["spread"] - df["mean"]) / df["std"]

    df["hl"] = rolling_half_life(df["spread"], HALF_LIFE_WINDOW)
    df["hl_ok"] = df["hl"] <= MAX_HALF_LIFE

    df["norm"] = df["spread"] / df["std"]
    df["norm_chg"] = df["norm"].diff()
    df["sig_vol"] = df["norm_chg"].rolling(VOL_TARGET_WINDOW).std().shift(1)

    df = df.dropna().copy()
    if df.empty:
        return None

    pos = 0
    positions = []

    long_arm = False
    short_arm = False

    z = df["z"]
    z_prev = z.shift(1)

    for i in range(len(df)):
        cur = z.iloc[i]
        prev = z_prev.iloc[i]

        if pd.isna(cur) or pd.isna(prev):
            positions.append(pos)
            continue

        if cur < -ENTRY_THRESHOLD:
            long_arm = True
        if cur > ENTRY_THRESHOLD:
            short_arm = True

        if pos == 0:
            if long_arm and df["hl_ok"].iloc[i] and cur > prev:
                pos = 1
                long_arm = False
                short_arm = False
            elif short_arm and df["hl_ok"].iloc[i] and cur < prev:
                pos = -1
                long_arm = False
                short_arm = False

        elif pos == 1 and cur >= -EXIT_THRESHOLD:
            pos = 0
            long_arm = False
            short_arm = False

        elif pos == -1 and cur <= EXIT_THRESHOLD:
            pos = 0
            long_arm = False
            short_arm = False

        positions.append(pos)

    df["pos"] = pd.Series(positions, index=df.index)
    last = df.iloc[-1]

    if last["pos"] == 1:
        action = f"LONG {y_ticker} / SHORT {x_ticker}"
    elif last["pos"] == -1:
        action = f"SHORT {y_ticker} / LONG {x_ticker}"
    else:
        action = "FLAT"

    return {
        "date": str(df.index[-1].date()),
        "y_price": float(last[y_ticker]),
        "x_price": float(last[x_ticker]),
        "zscore": float(last["z"]),
        "beta": float(last["beta_shifted"]),
        "half_life": float(last["hl"]) if pd.notna(last["hl"]) else np.nan,
        "signal_vol": float(last["sig_vol"]) if pd.notna(last["sig_vol"]) else np.nan,
        "position": int(last["pos"]),
        "action": action,
    }


def select_top_pairs(ranked_df: pd.DataFrame) -> pd.DataFrame:
    strict_df = ranked_df[
        ranked_df["passes_beta"]
        & ranked_df["passes_corr"]
        & ranked_df["beta"].abs().between(MIN_ABS_BETA, MAX_ABS_BETA)
        & (ranked_df["test_sharpe"] > MIN_TEST_SHARPE)
        & (ranked_df["train_sharpe"] > MIN_TRAIN_SHARPE)
        & (ranked_df["test_trades"] >= MIN_TRADES)
    ].copy()

    selected_rows = []
    bucket_counts = {}

    for _, row in strict_df.iterrows():
        bucket = row["bucket"]

        if bucket_counts.get(bucket, 0) >= MAX_PAIRS_PER_BUCKET:
            continue

        selected_rows.append(row)
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

        if len(selected_rows) >= TOP_N_PAIRS:
            break

    top_pairs_df = pd.DataFrame(selected_rows)

    if not top_pairs_df.empty:
        return top_pairs_df

    fallback_df = ranked_df[
        ranked_df["passes_beta"]
        & ranked_df["passes_corr"]
        & ranked_df["beta"].abs().between(MIN_ABS_BETA, MAX_ABS_BETA)
    ].copy()

    selected_rows = []
    bucket_counts = {}

    for _, row in fallback_df.iterrows():
        bucket = row["bucket"]

        if bucket_counts.get(bucket, 0) >= MAX_PAIRS_PER_BUCKET:
            continue

        selected_rows.append(row)
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

        if len(selected_rows) >= TOP_N_PAIRS:
            break

    return pd.DataFrame(selected_rows)


def upsert_log(new_rows: pd.DataFrame, path: Path) -> None:
    if new_rows.empty:
        return

    key_cols = ["signal_date", "pair"]

    if path.exists():
        existing = pd.read_csv(path)
        combined = pd.concat([existing, new_rows], ignore_index=True)
        combined = combined.drop_duplicates(subset=key_cols, keep="last")
        combined.to_csv(path, index=False)
    else:
        new_rows.to_csv(path, index=False)


# ----------------------------
# Main daily run
# ----------------------------
def main():
    pair_candidates = generate_pair_candidates(UNIVERSE)

    all_tickers = sorted({ticker for _, y, x in pair_candidates for ticker in (y, x)})
    all_price_data = {}

    print("Downloading price data...")
    for ticker in all_tickers:
        all_price_data[ticker] = download_and_prepare(ticker)

    pair_rows = []

    for bucket, y, x in pair_candidates:
        raw_df = all_price_data[y].join(all_price_data[x]).dropna()

        if len(raw_df) < MIN_ROWS:
            continue

        log(f"Scoring {bucket}: {y} vs {x}")

        filt = compute_pair_filters(raw_df, y, x)

        split = int(len(raw_df) * 0.7)
        train_df = raw_df.iloc[:split]
        test_df = raw_df.iloc[split:]

        train_bt = run_backtest(train_df, y, x)
        test_bt = run_backtest(test_df, y, x)

        train_ret, train_sharpe, train_dd, train_trades = compute_metrics(train_bt)
        test_ret, test_sharpe, test_dd, test_trades = compute_metrics(test_bt)

        score = score_pair(
            filt["pvalue"],
            filt["beta"],
            filt["correlation"],
            train_sharpe,
            test_sharpe,
            test_trades,
        )

        pair_rows.append({
            "bucket": bucket,
            "pair": f"{y} vs {x}",
            "y_ticker": y,
            "x_ticker": x,
            "pvalue": filt["pvalue"],
            "beta": filt["beta"],
            "correlation": filt["correlation"],
            "passes_beta": filt["passes_beta"],
            "passes_corr": filt["passes_corr"],
            "train_sharpe": train_sharpe,
            "test_sharpe": test_sharpe,
            "train_return": train_ret,
            "test_return": test_ret,
            "train_dd": train_dd,
            "test_dd": test_dd,
            "train_trades": train_trades,
            "test_trades": test_trades,
            "score": score,
        })

    if not pair_rows:
        print("No pairs had enough data.")
        return

    ranked_df = pd.DataFrame(pair_rows).sort_values(
        by=["score", "test_sharpe", "train_sharpe"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    top_pairs_df = select_top_pairs(ranked_df)

    if top_pairs_df.empty:
        print("No valid pairs selected today.")
        return

    top_pairs_df["raw_weight"] = top_pairs_df.apply(weight_from_row, axis=1)
    top_pairs_df["portfolio_weight"] = normalize_capped_weights(
        top_pairs_df["raw_weight"],
        MAX_PAIR_WEIGHT,
    )

    signal_rows = []

    for _, row in top_pairs_df.iterrows():
        y = row["y_ticker"]
        x = row["x_ticker"]

        raw_df = all_price_data[y].join(all_price_data[x]).dropna()
        latest_signal = get_latest_signal(raw_df, y, x)

        if latest_signal is None:
            continue

        if pd.isna(latest_signal["beta"]) or abs(latest_signal["beta"]) < MIN_LIVE_ABS_BETA:
            continue

        signal_rows.append({
            "run_timestamp": datetime.now().isoformat(timespec="seconds"),
            "signal_date": latest_signal["date"],
            "bucket": row["bucket"],
            "pair": row["pair"],
            "weight": row["portfolio_weight"],
            "y_price": latest_signal["y_price"],
            "x_price": latest_signal["x_price"],
            "zscore": latest_signal["zscore"],
            "beta": latest_signal["beta"],
            "half_life": latest_signal["half_life"],
            "signal_vol": latest_signal["signal_vol"],
            "position": latest_signal["position"],
            "action": latest_signal["action"],
            "correlation": row["correlation"],
            "score": row["score"],
            "train_sharpe": row["train_sharpe"],
            "test_sharpe": row["test_sharpe"],
        })

    signal_df = pd.DataFrame(signal_rows)

    if signal_df.empty:
        print("No signals generated today after live beta guardrail.")
        return

    print("\nSelected Pairs For Today")
    print("------------------------")
    print(top_pairs_df[[
        "bucket", "pair", "correlation", "score",
        "train_sharpe", "test_sharpe", "portfolio_weight"
    ]])
    print()

    print("Daily Paper Trading Signals")
    print("---------------------------")
    print(signal_df[[
        "signal_date",
        "bucket",
        "pair",
        "weight",
        "y_price",
        "x_price",
        "zscore",
        "beta",
        "position",
        "action",
    ]])
    print()

    upsert_log(signal_df, LOG_PATH)
    print(f"Log updated: {LOG_PATH.resolve()}")


if __name__ == "__main__":
    main()