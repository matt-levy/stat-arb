import argparse
import os
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests


READY_SIGNALS_INPUT = Path("paper_trade_ready_signals.csv")
RANKED_PAIRS_INPUT = Path("ranked_pairs_walk_forward.csv")
ENV_FILE = Path(".env")

ORDER_PREVIEW_OUTPUT = Path("alpaca_order_preview.csv")
EXECUTION_LOG_OUTPUT = Path("alpaca_execution_log.csv")
TRADE_LOG_OUTPUT = Path("alpaca_trade_log.csv")

DEFAULT_BASE_URL = "https://paper-api.alpaca.markets"
DEFAULT_DRY_RUN = True
DEFAULT_GROSS_EXPOSURE_FRACTION = 0.90
DEFAULT_MIN_LEG_NOTIONAL = 100.0
DEFAULT_MAX_SIGNAL_STALENESS_DAYS = 3
REQUEST_TIMEOUT_SECONDS = 20


@dataclass
class AlpacaConfig:
    """Runtime configuration loaded from environment variables."""

    api_key: str
    secret_key: str
    base_url: str
    dry_run: bool
    gross_exposure_fraction: float
    min_leg_notional: float
    max_signal_staleness_days: int


def load_env_file(path: Path) -> None:
    """Load KEY=VALUE pairs from a local .env file if present."""
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def parse_bool_env(name: str, default: bool) -> bool:
    """Read a boolean environment variable."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def parse_float_env(name: str, default: float) -> float:
    """Read a float environment variable with a fallback."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def parse_int_env(name: str, default: int) -> int:
    """Read an integer environment variable with a fallback."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def load_config() -> AlpacaConfig:
    """Load Alpaca settings from environment variables."""
    load_env_file(ENV_FILE)

    api_key = os.getenv("ALPACA_API_KEY", "").strip()
    secret_key = os.getenv("ALPACA_SECRET_KEY", "").strip()
    base_url = os.getenv("ALPACA_BASE_URL", DEFAULT_BASE_URL).strip().rstrip("/")

    if not api_key or not secret_key:
        raise ValueError(
            "Missing Alpaca credentials. Set ALPACA_API_KEY and ALPACA_SECRET_KEY in your environment or .env file."
        )

    return AlpacaConfig(
        api_key=api_key,
        secret_key=secret_key,
        base_url=base_url,
        dry_run=parse_bool_env("ALPACA_DRY_RUN", DEFAULT_DRY_RUN),
        gross_exposure_fraction=parse_float_env(
            "ALPACA_GROSS_EXPOSURE_FRACTION",
            DEFAULT_GROSS_EXPOSURE_FRACTION,
        ),
        min_leg_notional=parse_float_env(
            "ALPACA_MIN_LEG_NOTIONAL",
            DEFAULT_MIN_LEG_NOTIONAL,
        ),
        max_signal_staleness_days=parse_int_env(
            "ALPACA_MAX_SIGNAL_STALENESS_DAYS",
            DEFAULT_MAX_SIGNAL_STALENESS_DAYS,
        ),
    )


def safe_float(value: object) -> float:
    """Convert a value to float, returning NaN on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV file if it exists."""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


class AlpacaClient:
    """Minimal Alpaca Trading API client."""

    def __init__(self, config: AlpacaConfig) -> None:
        self.base_url = config.base_url
        self.session = requests.Session()
        self.session.headers.update(
            {
                "APCA-API-KEY-ID": config.api_key,
                "APCA-API-SECRET-KEY": config.secret_key,
                "Content-Type": "application/json",
            }
        )

    def request(self, method: str, path: str, payload: Optional[Dict[str, object]] = None) -> object:
        """Send a request to the Alpaca Trading API."""
        response = self.session.request(
            method=method,
            url=f"{self.base_url}{path}",
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.json()

    def get_account(self) -> Dict[str, object]:
        """Fetch account details."""
        return self.request("GET", "/v2/account")

    def get_positions(self) -> List[Dict[str, object]]:
        """Fetch all open positions."""
        return self.request("GET", "/v2/positions")

    def submit_order(self, symbol: str, qty: int, side: str, client_order_id: str) -> Dict[str, object]:
        """Submit a simple market order."""
        payload = {
            "symbol": symbol,
            "qty": str(qty),
            "side": side,
            "type": "market",
            "time_in_force": "day",
            "client_order_id": client_order_id,
        }
        return self.request("POST", "/v2/orders", payload)


def get_signal_staleness_days(ready_signals: pd.DataFrame) -> int:
    """Return how many calendar days old the latest ready signal is."""
    if ready_signals.empty or "latest_date" not in ready_signals.columns:
        return 0

    latest_signal_date = pd.to_datetime(ready_signals["latest_date"]).max().date()
    return (date.today() - latest_signal_date).days


def build_ready_universe(ready_signals: pd.DataFrame, ranked_pairs: pd.DataFrame) -> pd.DataFrame:
    """Join ready signals to ticker legs from the ranked-pairs output."""
    if ready_signals.empty or ranked_pairs.empty:
        return pd.DataFrame()

    leg_columns = ["sector", "pair", "stock_x", "stock_y"]
    merged = ready_signals.merge(ranked_pairs[leg_columns], on=["sector", "pair"], how="left")
    merged = merged.dropna(subset=["stock_x", "stock_y"]).copy()
    return merged


def build_leg_targets(ready_universe: pd.DataFrame, account_equity: float, config: AlpacaConfig) -> pd.DataFrame:
    """Convert pair allocations into per-symbol target share counts."""
    if ready_universe.empty:
        return pd.DataFrame()

    leg_rows: List[Dict[str, object]] = []

    for _, row in ready_universe.iterrows():
        action = str(row.get("current_action", "")).strip()
        if action not in {"LONG_SPREAD", "SHORT_SPREAD"}:
            continue

        beta = abs(safe_float(row.get("live_beta")))
        price_x = safe_float(row.get("latest_price_x"))
        price_y = safe_float(row.get("latest_price_y"))
        weight = max(safe_float(row.get("portfolio_weight")), 0.0)

        if not np.isfinite(beta) or not np.isfinite(price_x) or not np.isfinite(price_y):
            continue
        if beta <= 0 or price_x <= 0 or price_y <= 0 or weight <= 0:
            continue

        gross_pair_notional = account_equity * config.gross_exposure_fraction * weight
        x_weight = 1.0 / (1.0 + beta)
        y_weight = beta / (1.0 + beta)

        x_notional = gross_pair_notional * x_weight
        y_notional = gross_pair_notional * y_weight

        if x_notional < config.min_leg_notional or y_notional < config.min_leg_notional:
            continue

        x_qty = int(np.floor(x_notional / price_x))
        y_qty = int(np.floor(y_notional / price_y))
        if x_qty <= 0 or y_qty <= 0:
            continue

        x_sign = 1 if action == "LONG_SPREAD" else -1
        y_sign = -1 if action == "LONG_SPREAD" else 1

        leg_rows.extend(
            [
                {
                    "pair": row["pair"],
                    "symbol": row["stock_x"],
                    "target_qty": x_sign * x_qty,
                    "reference_price": price_x,
                    "target_notional": x_sign * x_qty * price_x,
                },
                {
                    "pair": row["pair"],
                    "symbol": row["stock_y"],
                    "target_qty": y_sign * y_qty,
                    "reference_price": price_y,
                    "target_notional": y_sign * y_qty * price_y,
                },
            ]
        )

    if not leg_rows:
        return pd.DataFrame()

    leg_df = pd.DataFrame(leg_rows)
    aggregated = (
        leg_df.groupby("symbol", as_index=False)
        .agg(
            target_qty=("target_qty", "sum"),
            reference_price=("reference_price", "last"),
            target_notional=("target_notional", "sum"),
            source_pairs=("pair", lambda values: " | ".join(sorted(set(values)))),
        )
    )
    return aggregated


def build_pair_trade_plan(ready_universe: pd.DataFrame, account_equity: float, config: AlpacaConfig) -> pd.DataFrame:
    """Build a clear per-pair trade plan for actionable live signals."""
    if ready_universe.empty:
        return pd.DataFrame()

    plan_rows: List[Dict[str, object]] = []

    for _, row in ready_universe.iterrows():
        action = str(row.get("current_action", "")).strip()
        if action not in {"LONG_SPREAD", "SHORT_SPREAD"}:
            continue

        beta = abs(safe_float(row.get("live_beta")))
        price_x = safe_float(row.get("latest_price_x"))
        price_y = safe_float(row.get("latest_price_y"))
        weight = max(safe_float(row.get("portfolio_weight")), 0.0)

        if not np.isfinite(beta) or not np.isfinite(price_x) or not np.isfinite(price_y):
            continue
        if beta <= 0 or price_x <= 0 or price_y <= 0 or weight <= 0:
            continue

        gross_pair_notional = account_equity * config.gross_exposure_fraction * weight
        x_weight = 1.0 / (1.0 + beta)
        y_weight = beta / (1.0 + beta)

        x_notional = gross_pair_notional * x_weight
        y_notional = gross_pair_notional * y_weight

        if x_notional < config.min_leg_notional or y_notional < config.min_leg_notional:
            continue

        x_qty = int(np.floor(x_notional / price_x))
        y_qty = int(np.floor(y_notional / price_y))
        if x_qty <= 0 or y_qty <= 0:
            continue

        if action == "LONG_SPREAD":
            long_symbol = str(row["stock_x"]).upper()
            long_qty = x_qty
            short_symbol = str(row["stock_y"]).upper()
            short_qty = y_qty
            exit_rule = "Exit when z-score >= 0.0"
        else:
            long_symbol = str(row["stock_y"]).upper()
            long_qty = y_qty
            short_symbol = str(row["stock_x"]).upper()
            short_qty = x_qty
            exit_rule = "Exit when z-score <= 0.0"

        plan_rows.append(
            {
                "signal_date": row["latest_date"],
                "pair": row["pair"],
                "action": action,
                "long_symbol": long_symbol,
                "long_qty": long_qty,
                "short_symbol": short_symbol,
                "short_qty": short_qty,
                "portfolio_weight": safe_float(row["portfolio_weight"]),
                "live_zscore": safe_float(row["live_zscore"]),
                "live_beta": safe_float(row["live_beta"]),
                "live_half_life": safe_float(row["live_half_life"]),
                "exit_rule": exit_rule,
            }
        )

    return pd.DataFrame(plan_rows)


def build_current_position_map(positions: List[Dict[str, object]]) -> Dict[str, int]:
    """Normalize Alpaca position quantities into signed integer shares."""
    current_positions: Dict[str, int] = {}
    for position in positions:
        symbol = str(position.get("symbol", "")).upper()
        qty = int(round(safe_float(position.get("qty", 0.0))))
        side = str(position.get("side", "")).lower()
        if side == "short":
            qty *= -1
        current_positions[symbol] = qty
    return current_positions


def build_order_preview(
    leg_targets: pd.DataFrame,
    current_positions: Dict[str, int],
    managed_symbols: List[str],
) -> pd.DataFrame:
    """Build rebalance orders from desired and current shares."""
    target_map = {
        str(row["symbol"]).upper(): {
            "target_qty": int(row["target_qty"]),
            "reference_price": safe_float(row["reference_price"]),
            "target_notional": safe_float(row["target_notional"]),
            "source_pairs": row["source_pairs"],
        }
        for _, row in leg_targets.iterrows()
    }

    preview_rows: List[Dict[str, object]] = []
    all_symbols = sorted(set(managed_symbols) | set(target_map.keys()))

    for symbol in all_symbols:
        target_qty = int(target_map.get(symbol, {}).get("target_qty", 0))
        current_qty = int(current_positions.get(symbol, 0))
        delta_qty = target_qty - current_qty
        if delta_qty == 0:
            continue

        side = "buy" if delta_qty > 0 else "sell"
        preview_rows.append(
            {
                "symbol": symbol,
                "side": side,
                "order_qty": abs(delta_qty),
                "current_qty": current_qty,
                "target_qty": target_qty,
                "delta_qty": delta_qty,
                "reference_price": safe_float(target_map.get(symbol, {}).get("reference_price")),
                "target_notional": safe_float(target_map.get(symbol, {}).get("target_notional")),
                "source_pairs": target_map.get(symbol, {}).get("source_pairs", ""),
            }
        )

    return pd.DataFrame(preview_rows)


def print_order_preview(preview_df: pd.DataFrame) -> None:
    """Print the proposed paper-trade rebalance."""
    if preview_df.empty:
        print("No Alpaca rebalance orders are needed.")
        return

    display_df = preview_df.copy()
    for column in ["reference_price", "target_notional"]:
        display_df[column] = display_df[column].astype(float).round(2)

    print("\nAlpaca Paper Trade Preview")
    print("--------------------------")
    print(
        display_df[
            ["symbol", "side", "order_qty", "current_qty", "target_qty", "reference_price", "source_pairs"]
        ].to_string(index=False)
    )


def execute_orders(client: AlpacaClient, preview_df: pd.DataFrame) -> pd.DataFrame:
    """Submit preview orders to Alpaca and capture responses."""
    if preview_df.empty:
        return pd.DataFrame()

    execution_rows: List[Dict[str, object]] = []
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    for index, row in preview_df.reset_index(drop=True).iterrows():
        symbol = str(row["symbol"]).upper()
        response = client.submit_order(
            symbol=symbol,
            qty=int(row["order_qty"]),
            side=str(row["side"]),
            client_order_id=f"pairs-{timestamp}-{index + 1}-{symbol}".lower(),
        )
        execution_rows.append(
            {
                "submitted_at": datetime.now().isoformat(timespec="seconds"),
                "symbol": symbol,
                "side": row["side"],
                "order_qty": int(row["order_qty"]),
                "target_qty": int(row["target_qty"]),
                "current_qty": int(row["current_qty"]),
                "alpaca_order_id": response.get("id", ""),
                "alpaca_status": response.get("status", ""),
                "source_pairs": row["source_pairs"],
            }
        )

    return pd.DataFrame(execution_rows)


def build_trade_log_rows(pair_trade_plan: pd.DataFrame, execution_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize each executed pair trade in one clear log row."""
    if pair_trade_plan.empty or execution_df.empty:
        return pd.DataFrame()

    execution_map = (
        execution_df.groupby("source_pairs", as_index=False)
        .agg(
            submitted_at=("submitted_at", "first"),
            order_count=("alpaca_order_id", "count"),
            alpaca_status=("alpaca_status", lambda values: "|".join(sorted(set(map(str, values))))),
            order_ids=("alpaca_order_id", lambda values: "|".join(map(str, values))),
        )
        .rename(columns={"source_pairs": "pair"})
    )

    trade_log = pair_trade_plan.merge(execution_map, on="pair", how="inner")
    if trade_log.empty:
        return trade_log

    ordered_columns = [
        "submitted_at",
        "signal_date",
        "pair",
        "action",
        "long_symbol",
        "long_qty",
        "short_symbol",
        "short_qty",
        "portfolio_weight",
        "live_zscore",
        "live_beta",
        "live_half_life",
        "exit_rule",
        "order_count",
        "alpaca_status",
        "order_ids",
    ]
    return trade_log[ordered_columns]


def append_csv_rows(path: Path, rows: pd.DataFrame) -> None:
    """Append rows to a CSV log, creating it if needed."""
    if rows.empty:
        return

    if path.exists():
        existing = pd.read_csv(path)
        rows = pd.concat([existing, rows], ignore_index=True)

    rows.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit PAPER_TRADE_READY pair trades to Alpaca paper trading.")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually submit orders. Without this flag the script only previews the rebalance.",
    )
    parser.add_argument(
        "--allow-stale",
        action="store_true",
        help="Allow execution even if the signal file is older than ALPACA_MAX_SIGNAL_STALENESS_DAYS.",
    )
    args = parser.parse_args()

    config = load_config()
    ready_signals = load_csv(READY_SIGNALS_INPUT)
    ranked_pairs = load_csv(RANKED_PAIRS_INPUT)

    if ready_signals.empty:
        raise ValueError(f"Missing or empty ready signal file: {READY_SIGNALS_INPUT.resolve()}")
    if ranked_pairs.empty:
        raise ValueError(f"Missing or empty ranked pair file: {RANKED_PAIRS_INPUT.resolve()}")

    staleness_days = get_signal_staleness_days(ready_signals)

    ready_universe = build_ready_universe(ready_signals, ranked_pairs)
    if ready_universe.empty:
        raise ValueError("No ticker legs could be resolved from the ready signal and ranked pair files.")

    client = AlpacaClient(config)
    account = client.get_account()
    positions = client.get_positions()

    account_equity = safe_float(account.get("equity"))
    if not np.isfinite(account_equity) or account_equity <= 0:
        raise ValueError("Alpaca account equity is unavailable or invalid.")

    leg_targets = build_leg_targets(ready_universe, account_equity, config)
    pair_trade_plan = build_pair_trade_plan(ready_universe, account_equity, config)
    if leg_targets.empty or pair_trade_plan.empty:
        raise ValueError("No executable target legs were produced from the ready pairs.")

    managed_symbols = sorted(
        set(ready_universe["stock_x"].astype(str).str.upper()) | set(ready_universe["stock_y"].astype(str).str.upper())
    )
    current_positions = build_current_position_map(positions)
    preview_df = build_order_preview(leg_targets, current_positions, managed_symbols)
    preview_df.to_csv(ORDER_PREVIEW_OUTPUT, index=False)

    print(f"Account equity: ${account_equity:,.2f}")
    print(f"Preview saved to: {ORDER_PREVIEW_OUTPUT.resolve()}")
    if staleness_days > config.max_signal_staleness_days:
        latest_signal_date = pd.to_datetime(ready_signals["latest_date"]).max().date().isoformat()
        print(
            f"Warning: ready signals are stale by {staleness_days} days "
            f"(latest signal date {latest_signal_date})."
        )
    print_order_preview(preview_df)

    if preview_df.empty:
        return

    should_execute = args.execute and not config.dry_run
    if not should_execute:
        print(
            "\nDry run only. Set ALPACA_DRY_RUN=false in .env and run with --execute when you are ready to submit paper orders."
        )
        return

    if staleness_days > config.max_signal_staleness_days and not args.allow_stale:
        raise ValueError(
            f"Ready signals are stale by {staleness_days} days. "
            f"Rerun pair-checker.py and paper-trading-ready.py first, or pass --allow-stale to override."
        )

    execution_df = execute_orders(client, preview_df)
    if execution_df.empty:
        print("No orders were submitted.")
        return

    trade_log_df = build_trade_log_rows(pair_trade_plan, execution_df)

    append_csv_rows(EXECUTION_LOG_OUTPUT, execution_df)
    append_csv_rows(TRADE_LOG_OUTPUT, trade_log_df)

    print(f"\nExecution log updated: {EXECUTION_LOG_OUTPUT.resolve()}")
    print(f"Trade log updated: {TRADE_LOG_OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
