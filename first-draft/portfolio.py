from pathlib import Path

import pandas as pd

from config import (
    MAX_ABS_BETA,
    MAX_PAIRS_PER_BUCKET,
    MAX_PAIR_WEIGHT,
    MIN_ABS_BETA,
    MIN_TEST_SHARPE,
    MIN_TRAIN_SHARPE,
    MIN_TRADES,
    SPREAD_VOL_WEIGHT_EXPONENT,
    SPREAD_VOL_WEIGHT_FLOOR,
    TOP_N_PAIRS,
)


def weight_from_row(row):
    test_sharpe_component = max(row["test_sharpe"], 0) if pd.notna(row["test_sharpe"]) else 0
    train_sharpe_component = max(row["train_sharpe"], 0) if pd.notna(row["train_sharpe"]) else 0
    test_return_component = max(row["test_return"], 0) if pd.notna(row["test_return"]) else 0
    corr_component = max(row["correlation"], 0) if pd.notna(row["correlation"]) else 0
    spread_vol_ratio = row.get("recent_spread_vol_ratio", 1.0)
    if pd.isna(spread_vol_ratio):
        spread_vol_ratio = 1.0

    test_drawdown_penalty = abs(row["test_dd"]) if pd.notna(row["test_dd"]) else 0
    drawdown_multiplier = 1 / (1 + test_drawdown_penalty * 10)
    spread_vol_penalty = max(spread_vol_ratio, SPREAD_VOL_WEIGHT_FLOOR) ** SPREAD_VOL_WEIGHT_EXPONENT

    base_weight = (
        1.0
        + 2.5 * test_sharpe_component
        + 0.75 * train_sharpe_component
        + 10.0 * test_return_component
        + 0.5 * corr_component
    ) * drawdown_multiplier

    return base_weight / spread_vol_penalty


def normalize_capped_weights(raw_weights: pd.Series, cap: float = MAX_PAIR_WEIGHT) -> pd.Series:
    weights = raw_weights.copy().astype(float)

    if weights.empty:
        return weights

    weights = weights / weights.sum()

    for _ in range(20):
        over_cap = weights > cap
        if not over_cap.any():
            break

        excess = (weights[over_cap] - cap).sum()
        weights[over_cap] = cap

        under_cap = weights < cap
        under_sum = weights[under_cap].sum()

        if under_sum <= 0 or excess <= 0:
            break

        weights[under_cap] += weights[under_cap] / under_sum * excess
        weights = weights / weights.sum()

    return weights / weights.sum()


def select_top_pairs(ranked_df: pd.DataFrame) -> pd.DataFrame:
    strict_df = ranked_df[
        ranked_df["passes_beta"]
        & ranked_df["passes_corr"]
        & ranked_df["passes_stability"]
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
        & ranked_df["passes_stability"]
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

    key_cols = [col for col in ["signal_date", "pair", "record_type", "action_type"] if col in new_rows.columns]

    if path.exists():
        existing = pd.read_csv(path)
        combined = pd.concat([existing, new_rows], ignore_index=True)
        combined = combined.drop_duplicates(subset=key_cols, keep="last")
        combined.to_csv(path, index=False)
    else:
        new_rows.to_csv(path, index=False)
