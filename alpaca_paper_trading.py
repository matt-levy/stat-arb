import argparse
import hashlib
import os
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from project_paths import LOGS_DIR, OUTPUTS_DIR, ensure_project_directories
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
import requests


READY_SIGNALS_INPUT = OUTPUTS_DIR / "paper_trade_ready_signals.csv"
RANKED_PAIRS_INPUT = OUTPUTS_DIR / "ranked_pairs_walk_forward.csv"
ENV_FILE = Path(".env")

ORDER_PREVIEW_OUTPUT = OUTPUTS_DIR / "alpaca_order_preview.csv"
EXECUTION_LOG_OUTPUT = LOGS_DIR / "alpaca_execution_log.csv"
TRADE_LOG_OUTPUT = LOGS_DIR / "alpaca_trade_log.csv"
ACCOUNT_SNAPSHOT_LOG_OUTPUT = LOGS_DIR / "alpaca_account_snapshots.csv"
POSITIONS_SNAPSHOT_LOG_OUTPUT = LOGS_DIR / "alpaca_positions_snapshots.csv"
ORDER_FILL_LOG_OUTPUT = LOGS_DIR / "alpaca_order_fills.csv"
PAIR_LIFECYCLE_LOG_OUTPUT = LOGS_DIR / "alpaca_pair_lifecycle_log.csv"
PAIR_RISK_EVENTS_LOG_OUTPUT = LOGS_DIR / "alpaca_pair_risk_events.csv"

DEFAULT_BASE_URL = "https://paper-api.alpaca.markets"
DEFAULT_DRY_RUN = True
DEFAULT_GROSS_EXPOSURE_FRACTION = 0.90
DEFAULT_BUYING_POWER_USAGE_FRACTION = 0.50
DEFAULT_MIN_LEG_NOTIONAL = 100.0
DEFAULT_MAX_SIGNAL_STALENESS_DAYS = 3
DEFAULT_PAIR_STOP_LOSS_FRACTION = 0.0
REQUEST_TIMEOUT_SECONDS = 20
ACTIVE_EXECUTION_STATUSES = {
    "accepted",
    "calculated",
    "done_for_day",
    "filled",
    "held",
    "new",
    "partially_filled",
    "pending_new",
}


@dataclass
class AlpacaConfig:
    """Runtime configuration loaded from environment variables."""

    api_key: str
    secret_key: str
    base_url: str
    dry_run: bool
    gross_exposure_fraction: float
    buying_power_usage_fraction: float
    min_leg_notional: float
    max_signal_staleness_days: int
    flatten_on_no_targets: bool
    pair_stop_loss_fraction: float
    pair_denylist: List[str]


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


def parse_list_env(name: str) -> List[str]:
    """Read a comma-separated environment variable into a normalized list."""
    raw = os.getenv(name, "")
    if not raw.strip():
        return []
    values = [value.strip() for value in raw.replace(";", ",").split(",")]
    return [value for value in values if value]


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
        buying_power_usage_fraction=parse_float_env(
            "ALPACA_BUYING_POWER_USAGE_FRACTION",
            DEFAULT_BUYING_POWER_USAGE_FRACTION,
        ),
        min_leg_notional=parse_float_env(
            "ALPACA_MIN_LEG_NOTIONAL",
            DEFAULT_MIN_LEG_NOTIONAL,
        ),
        max_signal_staleness_days=parse_int_env(
            "ALPACA_MAX_SIGNAL_STALENESS_DAYS",
            DEFAULT_MAX_SIGNAL_STALENESS_DAYS,
        ),
        flatten_on_no_targets=parse_bool_env("ALPACA_FLATTEN_ON_NO_TARGETS", False),
        pair_stop_loss_fraction=max(parse_float_env("ALPACA_PAIR_STOP_LOSS_FRACTION", DEFAULT_PAIR_STOP_LOSS_FRACTION), 0.0),
        pair_denylist=parse_list_env("ALPACA_PAIR_DENYLIST"),
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
        if not response.ok:
            detail = response.text.strip()
            raise requests.HTTPError(
                f"{response.status_code} {response.reason} for {method} {path}: {detail}",
                response=response,
            )
        if not response.content or not response.text.strip():
            return {}
        return response.json()

    def get_account(self) -> Dict[str, object]:
        """Fetch account details."""
        return self.request("GET", "/v2/account")

    def get_positions(self) -> List[Dict[str, object]]:
        """Fetch all open positions."""
        return self.request("GET", "/v2/positions")

    def get_order(self, order_id: str) -> Dict[str, object]:
        """Fetch one order by id."""
        return self.request("GET", f"/v2/orders/{order_id}")

    def list_open_orders(self) -> List[Dict[str, object]]:
        """Fetch currently open orders."""
        return self.request("GET", "/v2/orders", {"status": "open", "direction": "desc", "limit": 500})

    def cancel_order(self, order_id: str) -> None:
        """Cancel one open order."""
        self.request("DELETE", f"/v2/orders/{order_id}")

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


def get_pairs_already_submitted_this_cycle(ready_universe: pd.DataFrame, trade_log: pd.DataFrame) -> Set[str]:
    """Return pairs that were already submitted for the same signal date."""
    if ready_universe.empty or trade_log.empty:
        return set()
    if "pair" not in trade_log.columns or "signal_date" not in trade_log.columns or "alpaca_status" not in trade_log.columns:
        return set()

    normalized_log = trade_log.copy()
    normalized_log["pair"] = normalized_log["pair"].astype(str).str.strip()
    normalized_log["signal_date"] = normalized_log["signal_date"].astype(str).str.strip()
    normalized_log["alpaca_status"] = normalized_log["alpaca_status"].fillna("").astype(str)

    executed_pairs: Set[str] = set()
    for _, row in ready_universe.iterrows():
        pair = str(row.get("pair", "")).strip()
        latest_date = str(row.get("latest_date", "")).strip()
        if not pair or not latest_date:
            continue

        matches = normalized_log[
            (normalized_log["pair"] == pair)
            & (normalized_log["signal_date"] == latest_date)
        ]
        if matches.empty:
            continue

        for raw_status in matches["alpaca_status"]:
            statuses = {status.strip().lower() for status in str(raw_status).split("|") if status.strip()}
            if statuses & ACTIVE_EXECUTION_STATUSES:
                executed_pairs.add(pair)
                break

    return executed_pairs


def build_client_order_id(
    signal_date: str,
    symbol: str,
    side: str,
    target_qty: int,
    source_pairs: str,
) -> str:
    """Build a deterministic client order id for one target state."""
    normalized_date = str(signal_date).replace("-", "")[:8] or "nodate"
    normalized_symbol = str(symbol).upper().strip()[:8] or "nosym"
    normalized_side = str(side).lower().strip()[:4] or "side"
    payload = "|".join(
        [
            str(signal_date).strip(),
            normalized_symbol,
            normalized_side,
            str(int(target_qty)),
            str(source_pairs).strip(),
        ]
    )
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return f"pairs-{normalized_date}-{normalized_symbol.lower()}-{normalized_side}-{digest}"


def determine_deployable_capital(account: Dict[str, object], config: AlpacaConfig) -> float:
    """Cap deployment by both equity budget and currently available buying power."""
    equity = safe_float(account.get("equity"))
    buying_power = safe_float(account.get("buying_power"))

    if not np.isfinite(equity) or equity <= 0:
        return float("nan")

    equity_budget = equity * config.gross_exposure_fraction
    if not np.isfinite(buying_power) or buying_power <= 0:
        return equity_budget

    buying_power_budget = buying_power * config.buying_power_usage_fraction
    return min(equity_budget, buying_power_budget)


def build_leg_targets(ready_universe: pd.DataFrame, deployable_capital: float, config: AlpacaConfig) -> pd.DataFrame:
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

        gross_pair_notional = deployable_capital * weight
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


def build_pair_trade_plan(ready_universe: pd.DataFrame, deployable_capital: float, config: AlpacaConfig) -> pd.DataFrame:
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

        gross_pair_notional = deployable_capital * weight
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
            qty = -abs(qty)
        elif side == "long":
            qty = abs(qty)
        current_positions[symbol] = qty
    return current_positions


def build_order_preview(
    leg_targets: pd.DataFrame,
    current_positions: Dict[str, int],
    managed_symbols: List[str],
    flatten_symbols: Optional[Set[str]] = None,
) -> pd.DataFrame:
    """Build rebalance orders from desired and current shares."""
    flatten_symbols = set(flatten_symbols or set())
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
        if symbol not in target_map and symbol not in flatten_symbols:
            continue
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


def execute_orders(client: AlpacaClient, preview_df: pd.DataFrame, signal_date: str) -> pd.DataFrame:
    """Submit preview orders to Alpaca and capture responses."""
    if preview_df.empty:
        return pd.DataFrame()

    execution_rows: List[Dict[str, object]] = []
    for index, row in preview_df.reset_index(drop=True).iterrows():
        symbol = str(row["symbol"]).upper()
        side = str(row["side"])
        qty = int(row["order_qty"])
        target_qty = int(row["target_qty"])
        source_pairs = str(row.get("source_pairs", ""))
        client_order_id = build_client_order_id(
            signal_date=signal_date,
            symbol=symbol,
            side=side,
            target_qty=target_qty,
            source_pairs=source_pairs,
        )
        try:
            response = client.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                client_order_id=client_order_id,
            )
            execution_rows.append(
                {
                    "submitted_at": datetime.now().isoformat(timespec="seconds"),
                    "symbol": symbol,
                    "side": side,
                    "order_qty": qty,
                    "target_qty": target_qty,
                    "current_qty": int(row["current_qty"]),
                    "alpaca_order_id": response.get("id", ""),
                    "client_order_id": client_order_id,
                    "alpaca_status": response.get("status", ""),
                    "filled_qty": safe_float(response.get("filled_qty")),
                    "filled_avg_price": safe_float(response.get("filled_avg_price")),
                    "created_at": response.get("created_at", ""),
                    "updated_at": response.get("updated_at", ""),
                    "source_pairs": source_pairs,
                    "error": "",
                }
            )
        except requests.HTTPError as exc:
            execution_rows.append(
                {
                    "submitted_at": datetime.now().isoformat(timespec="seconds"),
                    "symbol": symbol,
                    "side": side,
                    "order_qty": qty,
                    "target_qty": target_qty,
                    "current_qty": int(row["current_qty"]),
                    "alpaca_order_id": "",
                    "client_order_id": client_order_id,
                    "alpaca_status": "rejected",
                    "filled_qty": float("nan"),
                    "filled_avg_price": float("nan"),
                    "created_at": "",
                    "updated_at": "",
                    "source_pairs": source_pairs,
                    "error": str(exc),
                }
            )
            print(f"Order rejected for {symbol} {side} {qty}: {exc}")

    return pd.DataFrame(execution_rows)


def cancel_conflicting_open_orders(client: AlpacaClient, preview_df: pd.DataFrame) -> pd.DataFrame:
    """Cancel open orders for symbols about to be rebalanced."""
    if preview_df.empty:
        return pd.DataFrame()

    symbols_to_rebalance = set(preview_df["symbol"].astype(str).str.upper())
    if not symbols_to_rebalance:
        return pd.DataFrame()

    cancelled_rows: List[Dict[str, object]] = []
    for order in client.list_open_orders():
        symbol = str(order.get("symbol", "")).upper()
        if symbol not in symbols_to_rebalance:
            continue

        order_id = str(order.get("id", ""))
        if not order_id:
            continue

        client.cancel_order(order_id)
        cancelled_rows.append(
            {
                "cancelled_at": datetime.now().isoformat(timespec="seconds"),
                "alpaca_order_id": order_id,
                "symbol": symbol,
                "side": str(order.get("side", "")),
                "qty": safe_float(order.get("qty")),
                "alpaca_status": str(order.get("status", "")),
            }
        )

    return pd.DataFrame(cancelled_rows)


def build_position_details_map(positions: List[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    """Normalize current Alpaca positions for risk checks."""
    position_details: Dict[str, Dict[str, float]] = {}
    for position in positions:
        symbol = str(position.get("symbol", "")).upper()
        side = str(position.get("side", "")).lower()
        qty = int(round(safe_float(position.get("qty", 0.0))))
        if side == "short":
            qty = -abs(qty)
        elif side == "long":
            qty = abs(qty)
        position_details[symbol] = {
            "qty": qty,
            "avg_entry_price": safe_float(position.get("avg_entry_price")),
            "current_price": safe_float(position.get("current_price")),
        }
    return position_details


def estimate_pair_unrealized_pnl(pair_row: pd.Series, position_details: Dict[str, Dict[str, float]]) -> float:
    """Estimate pair unrealized PnL from current Alpaca positions."""
    long_symbol = str(pair_row.get("long_symbol", "")).upper()
    short_symbol = str(pair_row.get("short_symbol", "")).upper()
    long_target_qty = int(safe_float(pair_row.get("long_qty", 0)))
    short_target_qty = int(safe_float(pair_row.get("short_qty", 0)))

    long_position = position_details.get(long_symbol, {})
    short_position = position_details.get(short_symbol, {})

    long_current_qty = max(int(long_position.get("qty", 0)), 0)
    short_current_qty = max(-int(short_position.get("qty", 0)), 0)

    matched_long_qty = min(long_current_qty, long_target_qty)
    matched_short_qty = min(short_current_qty, short_target_qty)
    if matched_long_qty <= 0 and matched_short_qty <= 0:
        return float("nan")

    pnl = 0.0
    if matched_long_qty > 0:
        pnl += (safe_float(long_position.get("current_price")) - safe_float(long_position.get("avg_entry_price"))) * matched_long_qty
    if matched_short_qty > 0:
        pnl += (safe_float(short_position.get("avg_entry_price")) - safe_float(short_position.get("current_price"))) * matched_short_qty
    return pnl


def build_pair_risk_rows(
    pair_trade_plan: pd.DataFrame,
    position_details: Dict[str, Dict[str, float]],
    account_equity: float,
    config: AlpacaConfig,
) -> pd.DataFrame:
    """Detect pair stop-loss breaches from current unrealized PnL."""
    if pair_trade_plan.empty or config.pair_stop_loss_fraction <= 0 or not np.isfinite(account_equity) or account_equity <= 0:
        return pd.DataFrame()

    stop_threshold = -account_equity * config.pair_stop_loss_fraction
    risk_rows: List[Dict[str, object]] = []

    for _, row in pair_trade_plan.iterrows():
        estimated_pnl = estimate_pair_unrealized_pnl(row, position_details)
        if not np.isfinite(estimated_pnl) or estimated_pnl > stop_threshold:
            continue

        risk_rows.append(
            {
                "event_at": datetime.now().isoformat(timespec="seconds"),
                "latest_date": row["signal_date"],
                "pair": row["pair"],
                "event_type": "STOP_LOSS",
                "event_value": estimated_pnl,
                "threshold_value": stop_threshold,
                "notes": f"Estimated unrealized PnL {estimated_pnl:.2f} breached threshold {stop_threshold:.2f}.",
            }
        )

    return pd.DataFrame(risk_rows)


def upsert_risk_rows(path: Path, new_rows: pd.DataFrame) -> None:
    """Append risk rows while replacing duplicate pair/date/event entries."""
    if new_rows.empty:
        return

    key_cols = ["latest_date", "pair", "event_type"]
    if path.exists():
        existing = pd.read_csv(path)
        new_rows = pd.concat([existing, new_rows], ignore_index=True)
        new_rows = new_rows.drop_duplicates(subset=key_cols, keep="last")

    new_rows.to_csv(path, index=False)


def load_pair_risk_rows(path: Path) -> pd.DataFrame:
    """Load prior pair risk events if present."""
    return load_csv(path)


def get_pairs_in_cooldown(ready_universe: pd.DataFrame, risk_rows: pd.DataFrame) -> Set[str]:
    """Block re-entry for pairs stopped earlier in the same signal cycle."""
    if ready_universe.empty or risk_rows.empty:
        return set()

    cooldown_pairs: Set[str] = set()
    for _, row in ready_universe.iterrows():
        pair = str(row.get("pair", ""))
        latest_date = str(row.get("latest_date", ""))
        matches = risk_rows[
            (risk_rows["pair"].astype(str) == pair)
            & (risk_rows["latest_date"].astype(str) == latest_date)
            & (risk_rows["event_type"].astype(str) == "STOP_LOSS")
        ]
        if not matches.empty:
            cooldown_pairs.add(pair)

    return cooldown_pairs


def filter_blocked_pairs(ready_universe: pd.DataFrame, blocked_pairs: Set[str]) -> pd.DataFrame:
    """Remove blocked pairs from the active trading universe."""
    if ready_universe.empty or not blocked_pairs:
        return ready_universe
    return ready_universe[~ready_universe["pair"].astype(str).isin(blocked_pairs)].copy()


def build_trade_log_rows(pair_trade_plan: pd.DataFrame, execution_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize each executed pair trade in one clear log row."""
    if pair_trade_plan.empty or execution_df.empty:
        return pd.DataFrame()

    expanded_execution = execution_df.copy()
    expanded_execution["pair"] = expanded_execution["source_pairs"].fillna("").astype(str).str.split(
        " | ",
        regex=False,
    )
    expanded_execution = expanded_execution.explode("pair")
    expanded_execution["pair"] = expanded_execution["pair"].astype(str).str.strip()
    expanded_execution = expanded_execution[expanded_execution["pair"] != ""]
    if expanded_execution.empty:
        return pd.DataFrame()

    execution_map = (
        expanded_execution.groupby("pair", as_index=False)
        .agg(
            submitted_at=("submitted_at", "first"),
            order_count=("alpaca_order_id", "count"),
            alpaca_status=("alpaca_status", lambda values: "|".join(sorted(set(map(str, values))))),
            order_ids=("alpaca_order_id", lambda values: "|".join(map(str, values))),
        )
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


def build_account_snapshot_rows(account: Dict[str, object], deployable_capital: float) -> pd.DataFrame:
    """Build one-row account snapshot for the current run."""
    snapshot = {
        "captured_at": datetime.now().isoformat(timespec="seconds"),
        "account_number": str(account.get("account_number", "")),
        "status": str(account.get("status", "")),
        "equity": safe_float(account.get("equity")),
        "cash": safe_float(account.get("cash")),
        "buying_power": safe_float(account.get("buying_power")),
        "long_market_value": safe_float(account.get("long_market_value")),
        "short_market_value": safe_float(account.get("short_market_value")),
        "portfolio_value": safe_float(account.get("portfolio_value")),
        "regt_buying_power": safe_float(account.get("regt_buying_power")),
        "daytrading_buying_power": safe_float(account.get("daytrading_buying_power")),
        "deployable_capital": safe_float(deployable_capital),
    }
    return pd.DataFrame([snapshot])


def build_positions_snapshot_rows(positions: List[Dict[str, object]]) -> pd.DataFrame:
    """Build a positions snapshot for the current run."""
    captured_at = datetime.now().isoformat(timespec="seconds")
    rows: List[Dict[str, object]] = []

    for position in positions:
        rows.append(
            {
                "captured_at": captured_at,
                "symbol": str(position.get("symbol", "")),
                "side": str(position.get("side", "")),
                "qty": safe_float(position.get("qty")),
                "market_value": safe_float(position.get("market_value")),
                "avg_entry_price": safe_float(position.get("avg_entry_price")),
                "current_price": safe_float(position.get("current_price")),
                "unrealized_pl": safe_float(position.get("unrealized_pl")),
                "unrealized_plpc": safe_float(position.get("unrealized_plpc")),
                "cost_basis": safe_float(position.get("cost_basis")),
            }
        )

    return pd.DataFrame(rows)


def build_pair_lifecycle_rows(ready_universe: pd.DataFrame, pair_trade_plan: pd.DataFrame) -> pd.DataFrame:
    """Record the daily intended lifecycle state for each candidate pair."""
    if ready_universe.empty:
        return pd.DataFrame()

    lifecycle = ready_universe.copy()
    lifecycle["captured_at"] = datetime.now().isoformat(timespec="seconds")
    executable_pairs = set(pair_trade_plan["pair"].astype(str)) if not pair_trade_plan.empty else set()
    lifecycle["execution_state"] = lifecycle["pair"].astype(str).apply(
        lambda pair: "EXECUTABLE" if pair in executable_pairs else "NON_EXECUTABLE"
    )

    ordered_columns = [
        "captured_at",
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
        "execution_state",
    ]
    return lifecycle[ordered_columns]


def build_order_fill_rows(execution_df: pd.DataFrame) -> pd.DataFrame:
    """Record submitted order status details in a compact fill/status log."""
    if execution_df.empty:
        return pd.DataFrame()

    keep_columns = [
        "submitted_at",
        "symbol",
        "side",
        "order_qty",
        "target_qty",
        "client_order_id",
        "alpaca_order_id",
        "alpaca_status",
        "filled_qty",
        "filled_avg_price",
        "created_at",
        "updated_at",
        "source_pairs",
        "error",
    ]
    available = [column for column in keep_columns if column in execution_df.columns]
    return execution_df[available].copy()


def append_csv_rows(path: Path, rows: pd.DataFrame) -> None:
    """Append rows to a CSV log, creating it if needed."""
    if rows.empty:
        return

    if path.exists():
        existing = pd.read_csv(path)
        rows = pd.concat([existing, rows], ignore_index=True)

    rows.to_csv(path, index=False)


def main() -> None:
    ensure_project_directories()
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
    account_buying_power = safe_float(account.get("buying_power"))
    if not np.isfinite(account_equity) or account_equity <= 0:
        raise ValueError("Alpaca account equity is unavailable or invalid.")

    deployable_capital = determine_deployable_capital(account, config)
    if not np.isfinite(deployable_capital) or deployable_capital <= 0:
        raise ValueError("No deployable capital is available from equity/buying power constraints.")

    initial_pair_trade_plan = build_pair_trade_plan(ready_universe, deployable_capital, config)
    position_details = build_position_details_map(positions)
    new_risk_rows = build_pair_risk_rows(initial_pair_trade_plan, position_details, account_equity, config)
    if not new_risk_rows.empty:
        upsert_risk_rows(PAIR_RISK_EVENTS_LOG_OUTPUT, new_risk_rows)
    risk_rows = load_pair_risk_rows(PAIR_RISK_EVENTS_LOG_OUTPUT)
    prior_trade_log = load_csv(TRADE_LOG_OUTPUT)

    blocked_pairs = set(config.pair_denylist)
    blocked_pairs.update(get_pairs_in_cooldown(ready_universe, risk_rows))
    already_submitted_pairs = get_pairs_already_submitted_this_cycle(ready_universe, prior_trade_log)
    ready_universe = filter_blocked_pairs(ready_universe, blocked_pairs | already_submitted_pairs)

    leg_targets = build_leg_targets(ready_universe, deployable_capital, config)
    pair_trade_plan = build_pair_trade_plan(ready_universe, deployable_capital, config)
    pair_lifecycle_df = build_pair_lifecycle_rows(ready_universe, pair_trade_plan)
    account_snapshot_df = build_account_snapshot_rows(account, deployable_capital)
    positions_snapshot_df = build_positions_snapshot_rows(positions)

    append_csv_rows(ACCOUNT_SNAPSHOT_LOG_OUTPUT, account_snapshot_df)
    append_csv_rows(POSITIONS_SNAPSHOT_LOG_OUTPUT, positions_snapshot_df)
    append_csv_rows(PAIR_LIFECYCLE_LOG_OUTPUT, pair_lifecycle_df)

    all_managed_symbols = sorted(
        set(load_csv(READY_SIGNALS_INPUT).merge(load_csv(RANKED_PAIRS_INPUT)[["sector", "pair", "stock_x", "stock_y"]], on=["sector", "pair"], how="left")["stock_x"].astype(str).str.upper())
        | set(load_csv(READY_SIGNALS_INPUT).merge(load_csv(RANKED_PAIRS_INPUT)[["sector", "pair", "stock_x", "stock_y"]], on=["sector", "pair"], how="left")["stock_y"].astype(str).str.upper())
    )
    flatten_symbols: Set[str] = set()
    if blocked_pairs:
        blocked_rows = load_csv(READY_SIGNALS_INPUT).merge(
            load_csv(RANKED_PAIRS_INPUT)[["sector", "pair", "stock_x", "stock_y"]],
            on=["sector", "pair"],
            how="left",
        )
        blocked_rows = blocked_rows[blocked_rows["pair"].astype(str).isin(blocked_pairs)]
        flatten_symbols.update(blocked_rows["stock_x"].astype(str).str.upper())
        flatten_symbols.update(blocked_rows["stock_y"].astype(str).str.upper())
    if config.flatten_on_no_targets and leg_targets.empty:
        flatten_symbols.update(all_managed_symbols)

    current_positions = build_current_position_map(positions)
    preview_df = build_order_preview(
        leg_targets,
        current_positions,
        all_managed_symbols,
        flatten_symbols=flatten_symbols,
    )
    preview_df.to_csv(ORDER_PREVIEW_OUTPUT, index=False)

    print(f"Account equity: ${account_equity:,.2f}")
    if np.isfinite(account_buying_power):
        print(f"Account buying power: ${account_buying_power:,.2f}")
    print(f"Deployable capital: ${deployable_capital:,.2f}")
    print(f"Preview saved to: {ORDER_PREVIEW_OUTPUT.resolve()}")
    if blocked_pairs:
        print(f"Blocked pairs this cycle: {', '.join(sorted(blocked_pairs))}")
    if already_submitted_pairs:
        print(f"Already submitted this signal cycle: {', '.join(sorted(already_submitted_pairs))}")
    if leg_targets.empty or pair_trade_plan.empty:
        if config.flatten_on_no_targets:
            print("No executable target legs were produced from the ready pairs. Managed symbols will be flattened.")
        else:
            print("No executable target legs were produced from the ready pairs. Existing positions will be left unchanged.")
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
            f"Rerun pair_checker.py and paper_trading_ready.py first, or pass --allow-stale to override."
        )

    cancelled_orders_df = cancel_conflicting_open_orders(client, preview_df)
    if not cancelled_orders_df.empty:
        print(f"Cancelled {len(cancelled_orders_df)} open order(s) that conflicted with this rebalance.")

    latest_signal_date = pd.to_datetime(ready_signals["latest_date"]).max().date().isoformat()
    execution_df = execute_orders(client, preview_df, signal_date=latest_signal_date)
    if execution_df.empty:
        print("No orders were submitted.")
        return

    trade_log_df = build_trade_log_rows(pair_trade_plan, execution_df)
    order_fill_log_df = build_order_fill_rows(execution_df)

    append_csv_rows(EXECUTION_LOG_OUTPUT, execution_df)
    append_csv_rows(TRADE_LOG_OUTPUT, trade_log_df)
    append_csv_rows(ORDER_FILL_LOG_OUTPUT, order_fill_log_df)

    print(f"\nExecution log updated: {EXECUTION_LOG_OUTPUT.resolve()}")
    print(f"Trade log updated: {TRADE_LOG_OUTPUT.resolve()}")
    print(f"Order fill log updated: {ORDER_FILL_LOG_OUTPUT.resolve()}")
    print(f"Account snapshot log updated: {ACCOUNT_SNAPSHOT_LOG_OUTPUT.resolve()}")
    print(f"Positions snapshot log updated: {POSITIONS_SNAPSHOT_LOG_OUTPUT.resolve()}")
    print(f"Pair lifecycle log updated: {PAIR_LIFECYCLE_LOG_OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
