import argparse
from datetime import datetime

import pandas as pd

from backtest import evaluate_pair, total_trading_cost_rate
from config import (
    EVALUATION_MODE,
    INITIAL_CAPITAL,
    LOG_PATH,
    MAX_GROSS_EXPOSURE_FRACTION,
    MIN_LIVE_ABS_BETA,
    MIN_ROWS,
    RUN_JOURNAL_PATH,
    UNIVERSE,
    VERBOSE,
)
from data import download_and_prepare, generate_pair_candidates
from portfolio import normalize_capped_weights, select_top_pairs, upsert_log, weight_from_row
from signals import compute_pair_filters, get_latest_signal, score_pair


def log(message: str) -> None:
    if VERBOSE:
        print(message)


def format_percent(value: float) -> str:
    return f"{value:.1%}"


def estimate_live_trade_cost(weight: float, action_type: str) -> float:
    gross_allocation = INITIAL_CAPITAL * MAX_GROSS_EXPOSURE_FRACTION * weight

    if action_type in {"ENTER_LONG", "ENTER_SHORT", "EXIT"}:
        turnover_multiplier = 1.0
    elif action_type in {"FLIP_TO_LONG", "FLIP_TO_SHORT"}:
        turnover_multiplier = 2.0
    else:
        turnover_multiplier = 0.0

    return gross_allocation * total_trading_cost_rate() * turnover_multiplier


def format_trade_summary(signal_df: pd.DataFrame) -> str:
    if signal_df.empty:
        return "Trade Summary\n-------------\nNo live trades generated."

    signal_date = signal_df["signal_date"].iloc[0]
    lines = [
        "Trade Summary",
        "-------------",
        f"Signal date: {signal_date}",
        f"Active ideas: {len(signal_df)}",
        "",
    ]

    for index, row in enumerate(
        signal_df.sort_values(by=["weight", "score"], ascending=[False, False]).itertuples(index=False),
        start=1,
    ):
        lines.append(
            (
                f"{index}. {row.pair} [{row.bucket}]"
                f" | {row.action}"
                f" | Weight {format_percent(row.weight)}"
                f" | Z {row.zscore:.2f}"
                f" | Beta {row.beta:.2f}"
                f" | {row.action_type}"
            )
        )
        lines.append(
            (
                f"   Prices: {row.y_price:.2f} vs {row.x_price:.2f}"
                f" | Corr {row.correlation:.2f}"
                f" | Spread vol ratio {row.recent_spread_vol_ratio:.2f}"
                f" | Est cost ${row.estimated_trade_cost:.2f}"
            )
        )

    return "\n".join(lines)


def frame_to_markdown(df: pd.DataFrame, columns: list[str] | None = None, max_rows: int = 10) -> str:
    if df.empty:
        return "_None_"

    output_df = df.copy()
    if columns is not None:
        available_columns = [column for column in columns if column in output_df.columns]
        output_df = output_df[available_columns]

    output_df = output_df.head(max_rows).copy()
    output_df = output_df.fillna("")

    headers = [str(column) for column in output_df.columns]
    separator = ["---"] * len(headers)
    rows = [
        [str(value) for value in row]
        for row in output_df.astype(object).itertuples(index=False, name=None)
    ]

    table_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(separator) + " |",
    ]

    for row in rows:
        table_lines.append("| " + " | ".join(row) + " |")

    return "\n".join(table_lines)


def build_journal_trade_lines(signal_df: pd.DataFrame) -> list[str]:
    if signal_df.empty:
        return ["- No trades done."]

    trade_actions = {"ENTER_LONG", "ENTER_SHORT", "EXIT", "FLIP_TO_LONG", "FLIP_TO_SHORT"}
    trade_df = signal_df[signal_df["action_type"].isin(trade_actions)].copy()

    if trade_df.empty:
        return ["- No trades done."]

    lines = []
    for row in trade_df.sort_values(by=["weight", "score"], ascending=[False, False]).itertuples(index=False):
        exit_reason = f" ({row.exit_reason})" if getattr(row, "exit_reason", "") else ""
        lines.append(
            (
                f"- {row.pair}: {row.action_type}"
                f" | {row.action}"
                f" | wgt {format_percent(row.weight)}"
                f" | z {row.zscore:.2f}"
                f" | est cost ${row.estimated_trade_cost:.2f}"
                f"{exit_reason}"
            )
        )

    return lines


def append_run_journal(
    run_timestamp: str,
    evaluation_mode: str,
    ranked_df: pd.DataFrame,
    top_pairs_df: pd.DataFrame,
    signal_df: pd.DataFrame,
    live_rejection_df: pd.DataFrame,
) -> None:
    signal_date = signal_df["signal_date"].iloc[0] if not signal_df.empty else "N/A"
    trade_lines = build_journal_trade_lines(signal_df)

    sections = [
        f"## Run {run_timestamp} ({evaluation_mode})",
        "",
        f"- Signal date: {signal_date}",
        f"- Trades recorded: {0 if trade_lines == ['- No trades done.'] else len(trade_lines)}",
        "",
        "### Trades",
        *trade_lines,
        "",
        "---",
        "",
    ]

    if not RUN_JOURNAL_PATH.exists():
        header = [
            "# Daily Run Journal",
            "",
            "Readable history of the daily paper-trading run and walk-forward evaluation.",
            "",
        ]
        RUN_JOURNAL_PATH.write_text("\n".join(header), encoding="utf-8")

    with RUN_JOURNAL_PATH.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(sections))


def build_pair_audit_log(ranked_df: pd.DataFrame, run_timestamp: str) -> pd.DataFrame:
    if ranked_df.empty:
        return pd.DataFrame()

    audit_df = ranked_df.copy()
    audit_df["run_timestamp"] = run_timestamp
    audit_df["record_type"] = "pair_selection"
    audit_df["action_type"] = audit_df["selection_status"]
    audit_df["signal_date"] = audit_df["latest_price_date"]
    if "portfolio_weight" in audit_df.columns:
        audit_df["weight"] = audit_df["portfolio_weight"].fillna(0.0)
    else:
        audit_df["weight"] = 0.0
    audit_df["action"] = audit_df["selection_status"]
    audit_df["estimated_trade_cost"] = 0.0
    audit_df["exit_reason"] = ""
    audit_df["position"] = 0
    return audit_df


def main(evaluation_mode: str = EVALUATION_MODE):
    pair_candidates = generate_pair_candidates(UNIVERSE)
    all_tickers = sorted({ticker for _, y_ticker, x_ticker in pair_candidates for ticker in (y_ticker, x_ticker)})
    all_price_data = {}

    print("Downloading price data...")
    for ticker in all_tickers:
        all_price_data[ticker] = download_and_prepare(ticker)

    pair_rows = []

    for bucket, y_ticker, x_ticker in pair_candidates:
        raw_df = all_price_data[y_ticker].join(all_price_data[x_ticker]).dropna()
        latest_price_date = str(raw_df.index[-1].date()) if not raw_df.empty else ""

        if len(raw_df) < MIN_ROWS:
            pair_rows.append(
                {
                    "bucket": bucket,
                    "pair": f"{y_ticker} vs {x_ticker}",
                    "y_ticker": y_ticker,
                    "x_ticker": x_ticker,
                    "latest_price_date": latest_price_date,
                    "passes_beta": False,
                    "passes_corr": False,
                    "passes_stability": False,
                    "rejection_reason": "INSUFFICIENT_ROWS",
                    "score": -1.0,
                    "walk_forward_folds": 0,
                    "evaluation_mode": evaluation_mode,
                }
            )
            continue

        log(f"Scoring {bucket}: {y_ticker} vs {x_ticker}")

        filters = compute_pair_filters(raw_df, y_ticker, x_ticker)
        evaluation = evaluate_pair(raw_df, y_ticker, x_ticker, mode=evaluation_mode)

        score = score_pair(
            filters["pvalue"],
            filters["beta"],
            filters["correlation"],
            evaluation["train_sharpe"],
            evaluation["test_sharpe"],
            evaluation["test_trades"],
        )

        pair_rows.append(
            {
                "bucket": bucket,
                "pair": f"{y_ticker} vs {x_ticker}",
                "y_ticker": y_ticker,
                "x_ticker": x_ticker,
                "latest_price_date": latest_price_date,
                "pvalue": filters["pvalue"],
                "beta": filters["beta"],
                "correlation": filters["correlation"],
                "passes_beta": filters["passes_beta"],
                "passes_corr": filters["passes_corr"],
                "passes_stability": filters["passes_stability"],
                "recent_corr_mean": filters["recent_corr_mean"],
                "recent_corr_std": filters["recent_corr_std"],
                "recent_beta_std": filters["recent_beta_std"],
                "recent_spread_vol": filters["recent_spread_vol"],
                "historical_spread_vol": filters["historical_spread_vol"],
                "recent_spread_vol_ratio": filters["recent_spread_vol_ratio"],
                "recent_coint_pvalue": filters["recent_coint_pvalue"],
                "rejection_reason": filters["rejection_reason"],
                "train_sharpe": evaluation["train_sharpe"],
                "test_sharpe": evaluation["test_sharpe"],
                "train_return": evaluation["train_return"],
                "test_return": evaluation["test_return"],
                "train_dd": evaluation["train_dd"],
                "test_dd": evaluation["test_dd"],
                "train_trades": evaluation["train_trades"],
                "test_trades": evaluation["test_trades"],
                "walk_forward_folds": evaluation["walk_forward_folds"],
                "evaluation_mode": evaluation["mode"],
                "score": score,
            }
        )

    if not pair_rows:
        print("No pairs had enough data.")
        return

    ranked_df = pd.DataFrame(pair_rows).sort_values(
        by=["score", "test_sharpe", "train_sharpe"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    top_pairs_df = select_top_pairs(ranked_df).copy()
    ranked_df["selection_status"] = "FILTERED"

    if top_pairs_df.empty:
        run_timestamp = datetime.now().isoformat(timespec="seconds")
        audit_log_df = build_pair_audit_log(ranked_df, run_timestamp)
        upsert_log(audit_log_df, LOG_PATH)
        append_run_journal(run_timestamp, evaluation_mode, ranked_df, top_pairs_df, pd.DataFrame(), pd.DataFrame())
        print("No valid pairs selected today.")
        print(f"Journal updated: {RUN_JOURNAL_PATH.resolve()}")
        print(f"Log updated: {LOG_PATH.resolve()}")
        return

    top_pairs_df["raw_weight"] = top_pairs_df.apply(weight_from_row, axis=1)
    top_pairs_df["portfolio_weight"] = normalize_capped_weights(top_pairs_df["raw_weight"])

    selected_pairs = set(top_pairs_df["pair"])
    ranked_df.loc[ranked_df["pair"].isin(selected_pairs), "selection_status"] = "SELECTED"
    ranked_df["portfolio_weight"] = ranked_df["pair"].map(top_pairs_df.set_index("pair")["portfolio_weight"]).fillna(0.0)

    ranked_df.loc[
        ranked_df["selection_status"].eq("FILTERED") & ranked_df["rejection_reason"].fillna("").eq(""),
        "rejection_reason",
    ] = "PORTFOLIO_RANK_CUTOFF"

    signal_rows = []
    live_rejections = []
    run_timestamp = datetime.now().isoformat(timespec="seconds")

    for _, row in top_pairs_df.iterrows():
        y_ticker = row["y_ticker"]
        x_ticker = row["x_ticker"]
        raw_df = all_price_data[y_ticker].join(all_price_data[x_ticker]).dropna()
        latest_signal = get_latest_signal(raw_df, y_ticker, x_ticker)

        if latest_signal is None:
            live_rejections.append(
                {
                    "run_timestamp": run_timestamp,
                    "record_type": "signal",
                    "signal_date": row["latest_price_date"],
                    "bucket": row["bucket"],
                    "pair": row["pair"],
                    "weight": row["portfolio_weight"],
                    "action_type": "LIVE_REJECTED",
                    "action": "NO_SIGNAL",
                    "rejection_reason": "NO_SIGNAL_FRAME",
                    "estimated_trade_cost": 0.0,
                }
            )
            continue

        if pd.isna(latest_signal["beta"]) or abs(latest_signal["beta"]) < MIN_LIVE_ABS_BETA:
            live_rejections.append(
                {
                    "run_timestamp": run_timestamp,
                    "record_type": "signal",
                    "signal_date": latest_signal["date"],
                    "bucket": row["bucket"],
                    "pair": row["pair"],
                    "weight": row["portfolio_weight"],
                    "action_type": "LIVE_REJECTED",
                    "action": latest_signal["action"],
                    "rejection_reason": "LIVE_BETA_FILTER",
                    "estimated_trade_cost": 0.0,
                    "zscore": latest_signal["zscore"],
                    "beta": latest_signal["beta"],
                }
            )
            continue

        signal_rows.append(
            {
                "run_timestamp": run_timestamp,
                "record_type": "signal",
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
                "spread_vol": latest_signal["spread_vol"],
                "position": latest_signal["position"],
                "hold_days": latest_signal["hold_days"],
                "action": latest_signal["action"],
                "action_type": latest_signal["action_type"],
                "exit_reason": latest_signal["exit_reason"],
                "rejection_reason": latest_signal["rejection_reason"],
                "correlation": row["correlation"],
                "score": row["score"],
                "train_sharpe": row["train_sharpe"],
                "test_sharpe": row["test_sharpe"],
                "recent_corr_mean": row["recent_corr_mean"],
                "recent_beta_std": row["recent_beta_std"],
                "recent_spread_vol_ratio": row["recent_spread_vol_ratio"],
                "recent_coint_pvalue": row["recent_coint_pvalue"],
                "estimated_trade_cost": estimate_live_trade_cost(row["portfolio_weight"], latest_signal["action_type"]),
                "evaluation_mode": row["evaluation_mode"],
                "walk_forward_folds": row["walk_forward_folds"],
            }
        )

    signal_df = pd.DataFrame(signal_rows)
    live_rejection_df = pd.DataFrame(live_rejections)
    audit_log_df = build_pair_audit_log(ranked_df, run_timestamp)
    combined_log_df = pd.concat([audit_log_df, live_rejection_df, signal_df], ignore_index=True, sort=False)

    print("\nSelected Pairs For Today")
    print("------------------------")
    print(
        top_pairs_df[
            [
                "bucket",
                "pair",
                "correlation",
                "recent_corr_mean",
                "recent_beta_std",
                "recent_spread_vol_ratio",
                "score",
                "train_sharpe",
                "test_sharpe",
                "portfolio_weight",
                "evaluation_mode",
                "walk_forward_folds",
            ]
        ]
    )
    print()

    filtered_pairs_df = ranked_df[ranked_df["selection_status"] == "FILTERED"]
    if not filtered_pairs_df.empty:
        print("Filtered Pairs")
        print("--------------")
        print(
            filtered_pairs_df[
                [
                    "bucket",
                    "pair",
                    "score",
                    "correlation",
                    "recent_corr_mean",
                    "recent_beta_std",
                    "recent_spread_vol_ratio",
                    "rejection_reason",
                ]
            ].head(10)
        )
        print()

    if signal_df.empty:
        print("No signals generated today after live guards.")
    else:
        print("Daily Paper Trading Signals")
        print("---------------------------")
        print(
            signal_df[
                [
                    "signal_date",
                    "bucket",
                    "pair",
                    "weight",
                    "y_price",
                    "x_price",
                    "zscore",
                    "beta",
                    "position",
                    "hold_days",
                    "action_type",
                    "exit_reason",
                    "estimated_trade_cost",
                    "action",
                ]
            ]
        )
        print()
        print(format_trade_summary(signal_df))
        print()

    if not live_rejection_df.empty:
        print("Live Rejections")
        print("---------------")
        print(live_rejection_df[["signal_date", "bucket", "pair", "action_type", "rejection_reason", "weight"]])
        print()

    upsert_log(combined_log_df, LOG_PATH)
    append_run_journal(run_timestamp, evaluation_mode, ranked_df, top_pairs_df, signal_df, live_rejection_df)
    print(f"Journal updated: {RUN_JOURNAL_PATH.resolve()}")
    print(f"Log updated: {LOG_PATH.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the pairs trading daily workflow.")
    parser.add_argument(
        "--evaluation-mode",
        choices=["split", "walk_forward"],
        default=EVALUATION_MODE,
        help="Evaluation mode used during pair scoring.",
    )
    args = parser.parse_args()
    main(evaluation_mode=args.evaluation_mode)
