from datetime import datetime
from pathlib import Path

from project_paths import OUTPUTS_DIR, LOGS_DIR, ensure_project_directories
from typing import Dict, List

import numpy as np
import pandas as pd


# =========================
# Configuration
# =========================

LIVE_SIGNALS_INPUT = OUTPUTS_DIR / "live_pair_signals.csv"
RANKED_PAIRS_INPUT = OUTPUTS_DIR / "ranked_pairs_walk_forward.csv"

READY_SIGNALS_OUTPUT = OUTPUTS_DIR / "paper_trade_ready_signals.csv"
READY_LOG_OUTPUT = LOGS_DIR / "paper_trade_ready_log.csv"

MAX_ACTIVE_PAIRS = 3
MAX_PAIR_WEIGHT = 0.50
READY_OUTPUT_COLUMNS = [
    "run_timestamp",
    "latest_date",
    "sector",
    "pair",
    "live_recommendation",
    "current_action",
    "current_position",
    "portfolio_weight",
    "live_zscore",
    "live_beta",
    "live_half_life",
    "passes_live_stability",
    "live_stability_reason",
    "passes_leg_contribution",
    "leg_contribution_reason",
    "recent_x_contribution",
    "recent_y_contribution",
    "dominant_leg",
    "dominant_leg_share",
    "score",
    "confidence_score",
    "confidence_rank",
    "robustness_score",
    "robustness_pass_rate",
    "oos_sharpe",
    "oos_return",
    "oos_annualized_return",
    "oos_max_drawdown",
    "oos_trades",
    "oos_unique_test_days",
    "avg_coint_pvalue_passed",
    "avg_adf_pvalue_passed",
    "avg_half_life_passed",
    "latest_price_x",
    "latest_price_y",
]


# =========================
# Helpers
# =========================

def safe_float(value: object) -> float:
    """Convert values to float while preserving invalid values as NaN."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV if it exists, otherwise return an empty frame."""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def normalize_capped_weights(raw_weights: pd.Series, cap: float) -> pd.Series:
    """Normalize weights while enforcing a simple per-pair cap."""
    weights = raw_weights.astype(float).copy()
    if weights.empty or weights.sum() <= 0:
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


def weight_from_row(row: pd.Series) -> float:
    """Convert research quality into a simple portfolio weight score."""
    score_component = max(safe_float(row.get("score")), 0.0)
    confidence_component = max(safe_float(row.get("confidence_score")), 0.0)
    robustness_component = max(safe_float(row.get("robustness_score")), 0.0)
    sharpe_component = max(safe_float(row.get("oos_sharpe")), 0.0)
    annualized_return_component = max(safe_float(row.get("oos_annualized_return")), 0.0)

    return (
        1.0
        + 0.60 * score_component
        + 0.50 * confidence_component
        + 0.40 * robustness_component
        + 0.75 * sharpe_component
        + 4.00 * annualized_return_component
    )


def build_ready_pairs(live_signals: pd.DataFrame, ranked_pairs: pd.DataFrame) -> pd.DataFrame:
    """Merge research and live outputs, keeping only currently approved pairs."""
    if live_signals.empty or ranked_pairs.empty:
        return pd.DataFrame()

    ready_live = live_signals[live_signals["live_recommendation"].astype(str) == "PAPER_TRADE_READY"].copy()
    if ready_live.empty:
        return pd.DataFrame(columns=READY_OUTPUT_COLUMNS)

    ranked_subset = ranked_pairs[
        [
            "sector",
            "pair",
            "research_verdict",
            "research_recommendation",
            "score",
            "confidence_score",
            "confidence_rank",
            "robustness_score",
            "robustness_pass_rate",
            "oos_sharpe",
            "oos_return",
            "oos_annualized_return",
            "oos_max_drawdown",
            "oos_trades",
            "oos_unique_test_days",
            "avg_coint_pvalue_passed",
            "avg_adf_pvalue_passed",
            "avg_half_life_passed",
        ]
    ].copy()

    merged = ready_live.merge(ranked_subset, on=["sector", "pair"], how="left")
    if merged.empty:
        return merged

    for column, default in {
        "passes_leg_contribution": True,
        "leg_contribution_reason": "",
        "recent_x_contribution": np.nan,
        "recent_y_contribution": np.nan,
        "dominant_leg": "",
        "dominant_leg_share": np.nan,
    }.items():
        if column not in merged.columns:
            merged[column] = default

    merged["raw_weight"] = merged.apply(weight_from_row, axis=1)
    merged = merged.sort_values(
        by=["score", "confidence_score", "robustness_score", "oos_sharpe"],
        ascending=[False, False, False, False],
    ).head(MAX_ACTIVE_PAIRS).reset_index(drop=True)
    merged["portfolio_weight"] = normalize_capped_weights(merged["raw_weight"], MAX_PAIR_WEIGHT)

    merged["run_timestamp"] = datetime.now().isoformat(timespec="seconds")

    return merged[READY_OUTPUT_COLUMNS]


def upsert_log(new_rows: pd.DataFrame, path: Path) -> None:
    """Append daily paper-trade rows while replacing duplicate pair/date entries."""
    if new_rows.empty:
        return

    key_cols = ["latest_date", "pair"]
    if path.exists():
        existing = pd.read_csv(path)
        combined = pd.concat([existing, new_rows], ignore_index=True)
        combined = combined.drop_duplicates(subset=key_cols, keep="last")
        combined.to_csv(path, index=False)
    else:
        new_rows.to_csv(path, index=False)


def print_ready_summary(ready_pairs: pd.DataFrame) -> None:
    """Print the current paper-trade-ready set."""
    if ready_pairs.empty:
        print("No PAPER_TRADE_READY pairs found in live signals.")
        return

    display_df = ready_pairs[
        [
            "latest_date",
            "sector",
            "pair",
            "current_action",
            "portfolio_weight",
            "live_zscore",
            "live_beta",
            "score",
            "confidence_score",
            "robustness_score",
        ]
    ].copy()

    for column in ["portfolio_weight", "live_zscore", "live_beta", "dominant_leg_share", "score", "confidence_score", "robustness_score"]:
        if column not in display_df.columns:
            continue
        display_df[column] = display_df[column].astype(float).round(4)

    print("\nPaper Trade Ready Pairs")
    print("-----------------------")
    print(display_df.to_string(index=False))


# =========================
# Main
# =========================

def main() -> None:
    ensure_project_directories()
    live_signals = load_csv(LIVE_SIGNALS_INPUT)
    ranked_pairs = load_csv(RANKED_PAIRS_INPUT)

    if live_signals.empty:
        print(f"Missing or empty live signal file: {LIVE_SIGNALS_INPUT.resolve()}")
        return

    if ranked_pairs.empty:
        print(f"Missing or empty ranked pair file: {RANKED_PAIRS_INPUT.resolve()}")
        return

    ready_pairs = build_ready_pairs(live_signals, ranked_pairs)
    ready_pairs.to_csv(READY_SIGNALS_OUTPUT, index=False)
    if ready_pairs.empty:
        print("No PAPER_TRADE_READY pairs found in live signals.")
        print(f"Cleared ready signals at: {READY_SIGNALS_OUTPUT.resolve()}")
        return

    upsert_log(ready_pairs, READY_LOG_OUTPUT)
    print_ready_summary(ready_pairs)
    print(f"\nSaved ready signals to: {READY_SIGNALS_OUTPUT.resolve()}")
    print(f"Updated ready log: {READY_LOG_OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
