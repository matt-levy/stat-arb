import argparse
import hashlib
import os
from dataclasses import dataclass
from datetime import date, datetime
from itertools import combinations
from pathlib import Path

from project_paths import LOGS_DIR, OUTPUTS_DIR, ensure_project_directories
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
import requests


READY_SIGNALS_INPUT = OUTPUTS_DIR / "paper_trade_ready_signals.csv"
LIVE_SIGNALS_INPUT = OUTPUTS_DIR / "live_pair_signals.csv"
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
PAIR_ATTRIBUTION_OUTPUT = OUTPUTS_DIR / "alpaca_pair_attribution.csv"
PAIR_ATTRIBUTION_LOG_OUTPUT = LOGS_DIR / "alpaca_pair_attribution_log.csv"
PAIR_ROUNDTRIP_LOG_OUTPUT = LOGS_DIR / "alpaca_pair_roundtrip_log.csv"

DEFAULT_BASE_URL = "https://paper-api.alpaca.markets"
DEFAULT_DRY_RUN = True
DEFAULT_GROSS_EXPOSURE_FRACTION = 0.90
DEFAULT_BUYING_POWER_USAGE_FRACTION = 0.50
DEFAULT_MIN_LEG_NOTIONAL = 100.0
DEFAULT_MAX_SIGNAL_STALENESS_DAYS = 3
DEFAULT_PAIR_STOP_LOSS_FRACTION = 0.0075
DEFAULT_MAX_PAIR_NOTIONAL_IMBALANCE_PCT = 0.20
DEFAULT_NEAR_EXIT_NO_ADD_Z = 0.75
DEFAULT_TIME_STOP_HALF_LIVES = 3.0
DEFAULT_TIME_STOP_MIN_DAYS = 5
DEFAULT_REENTRY_COOLDOWN_DAYS = 3
DEFAULT_MIN_EXPECTED_EDGE = 0.0
DEFAULT_MIN_REBALANCE_SHARES = 0
DEFAULT_MIN_REBALANCE_NOTIONAL = 0.0
DEFAULT_FAIL_ON_RECONCILE_MISMATCH = True
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
    max_pair_notional_imbalance_pct: float
    near_exit_no_add_z: float
    time_stop_half_lives: float
    time_stop_min_days: int
    reentry_cooldown_days: int
    min_expected_edge: float
    min_rebalance_shares: int
    min_rebalance_notional: float
    fail_on_reconcile_mismatch: bool


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
        max_pair_notional_imbalance_pct=max(
            parse_float_env("ALPACA_MAX_PAIR_NOTIONAL_IMBALANCE_PCT", DEFAULT_MAX_PAIR_NOTIONAL_IMBALANCE_PCT),
            0.0,
        ),
        near_exit_no_add_z=max(parse_float_env("ALPACA_NEAR_EXIT_NO_ADD_Z", DEFAULT_NEAR_EXIT_NO_ADD_Z), 0.0),
        time_stop_half_lives=max(parse_float_env("ALPACA_TIME_STOP_HALF_LIVES", DEFAULT_TIME_STOP_HALF_LIVES), 0.0),
        time_stop_min_days=max(parse_int_env("ALPACA_TIME_STOP_MIN_DAYS", DEFAULT_TIME_STOP_MIN_DAYS), 0),
        reentry_cooldown_days=max(parse_int_env("ALPACA_REENTRY_COOLDOWN_DAYS", DEFAULT_REENTRY_COOLDOWN_DAYS), 0),
        min_expected_edge=max(parse_float_env("ALPACA_MIN_EXPECTED_EDGE", DEFAULT_MIN_EXPECTED_EDGE), 0.0),
        min_rebalance_shares=max(parse_int_env("ALPACA_MIN_REBALANCE_SHARES", DEFAULT_MIN_REBALANCE_SHARES), 0),
        min_rebalance_notional=max(parse_float_env("ALPACA_MIN_REBALANCE_NOTIONAL", DEFAULT_MIN_REBALANCE_NOTIONAL), 0.0),
        fail_on_reconcile_mismatch=parse_bool_env("ALPACA_FAIL_ON_RECONCILE_MISMATCH", DEFAULT_FAIL_ON_RECONCILE_MISMATCH),
    )


def safe_float(value: object) -> float:
    """Convert a value to float, returning NaN on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def parse_timestamp(value: object) -> pd.Timestamp:
    """Parse a timestamp-like value into a pandas timestamp."""
    return pd.to_datetime(value, errors="coerce")


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


def build_retry_client_order_id(base_client_order_id: str) -> str:
    """Build a unique retry client order id when the deterministic id was already used."""
    normalized_base = str(base_client_order_id).strip() or "pairs-retry"
    retry_suffix = datetime.now().strftime("%H%M%S")
    retry_digest = hashlib.sha1(f"{normalized_base}|{datetime.now().isoformat(timespec='microseconds')}".encode("utf-8")).hexdigest()[:6]
    return f"{normalized_base}-r{retry_suffix}{retry_digest}"[:48]


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


def is_near_exit(row: pd.Series, config: AlpacaConfig) -> bool:
    """Return whether an active spread is close enough to exit that exposure should not increase."""
    action = str(row.get("current_action", "")).strip()
    zscore = safe_float(row.get("live_zscore"))
    if not np.isfinite(zscore) or config.near_exit_no_add_z <= 0:
        return False
    if action == "SHORT_SPREAD":
        return zscore <= config.near_exit_no_add_z
    if action == "LONG_SPREAD":
        return zscore >= -config.near_exit_no_add_z
    return False


def has_event_window(row: pd.Series) -> bool:
    """Return whether a pair is in a manually curated public event window."""
    raw_value = row.get("has_event_window", False)
    if isinstance(raw_value, bool):
        return raw_value
    return str(raw_value).strip().lower() in {"1", "true", "yes", "y"}


def effective_expected_edge(row: pd.Series) -> float:
    """Estimate the usable per-trade edge after live sizing penalties."""
    net_current_edge = safe_float(row.get("current_net_expected_edge"))
    if np.isfinite(net_current_edge):
        return net_current_edge

    current_edge = safe_float(row.get("current_expected_edge"))
    if np.isfinite(current_edge):
        return max(current_edge, 0.0)

    base_edge = safe_float(row.get("oos_return_per_trade"))
    if not np.isfinite(base_edge):
        return float("nan")

    size_multiplier = safe_float(row.get("live_size_multiplier", 1.0))
    if not np.isfinite(size_multiplier):
        size_multiplier = 1.0
    size_multiplier = max(size_multiplier, 0.0)
    return base_edge * size_multiplier


def calculate_pair_sizing(row: pd.Series, deployable_capital: float, config: AlpacaConfig) -> Optional[Dict[str, object]]:
    """Convert one pair row into whole-share targets and reject poor hedge geometry."""
    action = str(row.get("current_action", "")).strip()
    if action not in {"LONG_SPREAD", "SHORT_SPREAD"}:
        return None

    beta = abs(safe_float(row.get("live_beta")))
    price_x = safe_float(row.get("latest_price_x"))
    price_y = safe_float(row.get("latest_price_y"))
    weight = max(safe_float(row.get("portfolio_weight")), 0.0)

    if not np.isfinite(beta) or not np.isfinite(price_x) or not np.isfinite(price_y):
        return None
    if beta <= 0 or price_x <= 0 or price_y <= 0 or weight <= 0:
        return None

    gross_pair_notional = deployable_capital * weight
    x_weight = 1.0 / (1.0 + beta)
    y_weight = beta / (1.0 + beta)

    intended_x_notional = gross_pair_notional * x_weight
    intended_y_notional = gross_pair_notional * y_weight

    if intended_x_notional < config.min_leg_notional or intended_y_notional < config.min_leg_notional:
        return None

    x_qty = int(np.floor(intended_x_notional / price_x))
    y_qty = int(np.floor(intended_y_notional / price_y))
    if x_qty <= 0 or y_qty <= 0:
        return None

    actual_x_notional = x_qty * price_x
    actual_y_notional = y_qty * price_y
    actual_gross_notional = actual_x_notional + actual_y_notional
    notional_imbalance = abs(actual_x_notional - actual_y_notional)
    imbalance_pct = notional_imbalance / actual_gross_notional if actual_gross_notional > 0 else float("inf")
    hedge_quality_pass = (
        np.isfinite(imbalance_pct)
        and imbalance_pct <= config.max_pair_notional_imbalance_pct
    )
    if not hedge_quality_pass:
        return None

    return {
        "beta": beta,
        "price_x": price_x,
        "price_y": price_y,
        "weight": weight,
        "x_qty": x_qty,
        "y_qty": y_qty,
        "intended_x_notional": intended_x_notional,
        "intended_y_notional": intended_y_notional,
        "actual_x_notional": actual_x_notional,
        "actual_y_notional": actual_y_notional,
        "actual_gross_notional": actual_gross_notional,
        "notional_imbalance": notional_imbalance,
        "notional_imbalance_pct": imbalance_pct,
        "near_exit_no_add": is_near_exit(row, config),
        "event_no_add": has_event_window(row),
    }


def build_executable_ready_universe(
    ready_universe: pd.DataFrame,
    deployable_capital: float,
    config: AlpacaConfig,
) -> pd.DataFrame:
    """Redistribute ready-pair weights across the best fully executable subset."""
    if ready_universe.empty:
        return pd.DataFrame()

    candidates = ready_universe.copy().reset_index(drop=True)
    candidates["_effective_expected_edge"] = candidates.apply(effective_expected_edge, axis=1)
    candidates["portfolio_weight"] = pd.to_numeric(candidates.get("portfolio_weight"), errors="coerce").fillna(0.0)
    edge_columns = {"current_net_expected_edge", "current_expected_edge", "oos_return_per_trade"}
    if edge_columns & set(candidates.columns):
        candidates = candidates[
            pd.to_numeric(candidates["_effective_expected_edge"], errors="coerce").fillna(float("-inf")) > 0.0
        ].reset_index(drop=True)
    if config.min_expected_edge > 0:
        candidates = candidates[
            pd.to_numeric(candidates["_effective_expected_edge"], errors="coerce").fillna(float("-inf"))
            >= config.min_expected_edge
        ].reset_index(drop=True)
    candidates = candidates[candidates["portfolio_weight"] > 0].reset_index(drop=True)
    if candidates.empty:
        return pd.DataFrame(columns=ready_universe.columns)
    candidates["_base_portfolio_weight"] = candidates["portfolio_weight"]

    target_weight_budget = min(float(candidates["portfolio_weight"].sum()), 1.0)
    if target_weight_budget <= 0:
        return pd.DataFrame(columns=ready_universe.columns)

    best_subset: Optional[pd.DataFrame] = None
    best_gross_notional = -1.0
    best_score = -1.0
    best_pair_count = -1

    def allocate_subset(subset: pd.DataFrame) -> Optional[pd.DataFrame]:
        base_weights = pd.to_numeric(subset["_base_portfolio_weight"], errors="coerce").fillna(0.0)
        base_weight_sum = float(base_weights.sum())
        if base_weight_sum <= 0:
            return None

        allocated = subset.copy()
        allocated["portfolio_weight"] = base_weights / base_weight_sum * target_weight_budget
        sizing_rows: List[Dict[str, object]] = []
        total_gross_notional = 0.0

        for _, pair_row in allocated.iterrows():
            sizing = calculate_pair_sizing(pair_row, deployable_capital, config)
            if sizing is None:
                return None
            sizing_rows.append(sizing)
            total_gross_notional += safe_float(sizing.get("actual_gross_notional"))

        allocated["_actual_gross_notional"] = [safe_float(row.get("actual_gross_notional")) for row in sizing_rows]
        return allocated

    candidate_indexes = list(candidates.index)
    for subset_size in range(len(candidate_indexes), 0, -1):
        for subset_indexes in combinations(candidate_indexes, subset_size):
            subset = candidates.loc[list(subset_indexes)].reset_index(drop=True)
            allocated_subset = allocate_subset(subset)
            if allocated_subset is None:
                continue

            gross_notional = float(pd.to_numeric(allocated_subset["_actual_gross_notional"], errors="coerce").fillna(0.0).sum())
            score = float(pd.to_numeric(allocated_subset["_base_portfolio_weight"], errors="coerce").fillna(0.0).sum())
            pair_count = len(allocated_subset)
            if (
                score > best_score + 1e-9
                or (
                    abs(score - best_score) <= 1e-9
                    and (
                        gross_notional > best_gross_notional + 1e-9
                        or (abs(gross_notional - best_gross_notional) <= 1e-9 and pair_count > best_pair_count)
                    )
                )
            ):
                best_subset = allocated_subset
                best_gross_notional = gross_notional
                best_score = score
                best_pair_count = pair_count

    if best_subset is None:
        return pd.DataFrame(columns=ready_universe.columns)

    return best_subset.drop(
        columns=["_actual_gross_notional", "_base_portfolio_weight"],
        errors="ignore",
    ).reset_index(drop=True)


def build_leg_targets(ready_universe: pd.DataFrame, deployable_capital: float, config: AlpacaConfig) -> pd.DataFrame:
    """Convert pair allocations into per-symbol target share counts."""
    executable_universe = build_executable_ready_universe(ready_universe, deployable_capital, config)
    if executable_universe.empty:
        return pd.DataFrame()

    leg_rows: List[Dict[str, object]] = []

    for _, row in executable_universe.iterrows():
        action = str(row.get("current_action", "")).strip()
        sizing = calculate_pair_sizing(row, deployable_capital, config)
        if sizing is None:
            continue

        x_sign = 1 if action == "LONG_SPREAD" else -1
        y_sign = -1 if action == "LONG_SPREAD" else 1
        x_qty = int(sizing["x_qty"])
        y_qty = int(sizing["y_qty"])
        price_x = safe_float(sizing["price_x"])
        price_y = safe_float(sizing["price_y"])

        leg_rows.extend(
            [
                {
                    "pair": row["pair"],
                    "symbol": row["stock_x"],
                    "target_qty": x_sign * x_qty,
                    "reference_price": price_x,
                    "target_notional": x_sign * x_qty * price_x,
                    "notional_imbalance_pct": safe_float(sizing["notional_imbalance_pct"]),
                    "near_exit_no_add": bool(sizing["near_exit_no_add"]),
                    "event_no_add": bool(sizing["event_no_add"]),
                    "event_reason": str(row.get("event_reason", "")),
                },
                {
                    "pair": row["pair"],
                    "symbol": row["stock_y"],
                    "target_qty": y_sign * y_qty,
                    "reference_price": price_y,
                    "target_notional": y_sign * y_qty * price_y,
                    "notional_imbalance_pct": safe_float(sizing["notional_imbalance_pct"]),
                    "near_exit_no_add": bool(sizing["near_exit_no_add"]),
                    "event_no_add": bool(sizing["event_no_add"]),
                    "event_reason": str(row.get("event_reason", "")),
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
            notional_imbalance_pct=("notional_imbalance_pct", "max"),
            near_exit_no_add=("near_exit_no_add", "max"),
            event_no_add=("event_no_add", "max"),
            event_reason=("event_reason", lambda values: " | ".join(sorted({str(value) for value in values if str(value)}))),
            source_pairs=("pair", lambda values: " | ".join(sorted(set(values)))),
        )
    )
    return aggregated


def build_pair_trade_plan(ready_universe: pd.DataFrame, deployable_capital: float, config: AlpacaConfig) -> pd.DataFrame:
    """Build a clear per-pair trade plan for actionable live signals."""
    executable_universe = build_executable_ready_universe(ready_universe, deployable_capital, config)
    if executable_universe.empty:
        return pd.DataFrame()

    plan_rows: List[Dict[str, object]] = []

    for _, row in executable_universe.iterrows():
        action = str(row.get("current_action", "")).strip()
        sizing = calculate_pair_sizing(row, deployable_capital, config)
        if sizing is None:
            continue
        x_qty = int(sizing["x_qty"])
        y_qty = int(sizing["y_qty"])

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
                "target_long_notional": y_qty * safe_float(sizing["price_y"]) if action == "SHORT_SPREAD" else x_qty * safe_float(sizing["price_x"]),
                "target_short_notional": x_qty * safe_float(sizing["price_x"]) if action == "SHORT_SPREAD" else y_qty * safe_float(sizing["price_y"]),
                "net_notional": (
                    y_qty * safe_float(sizing["price_y"]) - x_qty * safe_float(sizing["price_x"])
                    if action == "SHORT_SPREAD"
                    else x_qty * safe_float(sizing["price_x"]) - y_qty * safe_float(sizing["price_y"])
                ),
                "notional_imbalance_pct": safe_float(sizing["notional_imbalance_pct"]),
                "near_exit_no_add": bool(sizing["near_exit_no_add"]),
                "event_no_add": bool(sizing["event_no_add"]),
                "event_reason": str(row.get("event_reason", "")),
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
    config: Optional[AlpacaConfig] = None,
) -> pd.DataFrame:
    """Build rebalance orders from desired and current shares."""
    flatten_symbols = set(flatten_symbols or set())
    target_map = {
        str(row["symbol"]).upper(): {
            "target_qty": int(row["target_qty"]),
            "reference_price": safe_float(row["reference_price"]),
            "target_notional": safe_float(row["target_notional"]),
            "source_pairs": row["source_pairs"],
            "notional_imbalance_pct": safe_float(row.get("notional_imbalance_pct")),
            "near_exit_no_add": bool(row.get("near_exit_no_add", False)),
            "event_no_add": bool(row.get("event_no_add", False)),
            "event_reason": str(row.get("event_reason", "")),
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
        if target_map.get(symbol, {}).get("near_exit_no_add", False) and abs(target_qty) > abs(current_qty):
            continue
        if target_map.get(symbol, {}).get("event_no_add", False) and abs(target_qty) > abs(current_qty):
            continue
        if (
            config is not None
            and should_skip_small_rebalance(
                current_qty=current_qty,
                target_qty=target_qty,
                reference_price=safe_float(target_map.get(symbol, {}).get("reference_price")),
                config=config,
            )
        ):
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
                "notional_imbalance_pct": safe_float(target_map.get(symbol, {}).get("notional_imbalance_pct")),
                "near_exit_no_add": bool(target_map.get(symbol, {}).get("near_exit_no_add", False)),
                "event_no_add": bool(target_map.get(symbol, {}).get("event_no_add", False)),
                "event_reason": str(target_map.get(symbol, {}).get("event_reason", "")),
                "source_pairs": target_map.get(symbol, {}).get("source_pairs", ""),
            }
        )

    return pd.DataFrame(preview_rows)


def should_skip_small_rebalance(
    *,
    current_qty: int,
    target_qty: int,
    reference_price: float,
    config: AlpacaConfig,
) -> bool:
    """Skip tiny same-direction top-ups that are likely to cost more than they add."""
    if config.min_rebalance_shares <= 0 and config.min_rebalance_notional <= 0:
        return False
    if current_qty == 0 or target_qty == 0:
        return False
    if (current_qty > 0 > target_qty) or (current_qty < 0 < target_qty):
        return False

    delta_qty = abs(target_qty - current_qty)
    delta_notional = delta_qty * reference_price if np.isfinite(reference_price) and reference_price > 0 else float("nan")
    below_share_floor = config.min_rebalance_shares > 0 and delta_qty < config.min_rebalance_shares
    below_notional_floor = (
        config.min_rebalance_notional > 0
        and np.isfinite(delta_notional)
        and delta_notional < config.min_rebalance_notional
    )
    return below_share_floor or below_notional_floor


def print_order_preview(preview_df: pd.DataFrame) -> None:
    """Print the proposed paper-trade rebalance."""
    if preview_df.empty:
        print("No Alpaca rebalance orders are needed.")
        return

    display_df = preview_df.copy()
    for column in ["reference_price", "target_notional", "notional_imbalance_pct"]:
        if column not in display_df.columns:
            continue
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
            retried_client_order_id = ""
            response = client.submit_order(symbol=symbol, qty=qty, side=side, client_order_id=client_order_id)
        except requests.HTTPError as exc:
            error_text = str(exc)
            if "client_order_id must be unique" not in error_text:
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
                        "retried_client_order_id": "",
                        "alpaca_status": "rejected",
                        "filled_qty": float("nan"),
                        "filled_avg_price": float("nan"),
                        "created_at": "",
                        "updated_at": "",
                        "source_pairs": source_pairs,
                        "error": error_text,
                    }
                )
                print(f"Order rejected for {symbol} {side} {qty}: {exc}")
                continue

            retry_client_order_id = build_retry_client_order_id(client_order_id)
            try:
                response = client.submit_order(symbol=symbol, qty=qty, side=side, client_order_id=retry_client_order_id)
                client_order_id = retry_client_order_id
                retried_client_order_id = retry_client_order_id
            except requests.HTTPError as retry_exc:
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
                        "retried_client_order_id": retry_client_order_id,
                        "alpaca_status": "rejected",
                        "filled_qty": float("nan"),
                        "filled_avg_price": float("nan"),
                        "created_at": "",
                        "updated_at": "",
                        "source_pairs": source_pairs,
                        "error": str(retry_exc),
                    }
                )
                print(f"Order rejected for {symbol} {side} {qty}: {retry_exc}")
                continue

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
                "retried_client_order_id": retried_client_order_id,
                "alpaca_status": response.get("status", ""),
                "filled_qty": safe_float(response.get("filled_qty")),
                "filled_avg_price": safe_float(response.get("filled_avg_price")),
                "created_at": response.get("created_at", ""),
                "updated_at": response.get("updated_at", ""),
                "source_pairs": source_pairs,
                "error": "",
            }
        )

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


def build_latest_snapshot_position_map(snapshot_df: pd.DataFrame) -> Dict[str, int]:
    """Collapse the most recent local position snapshot into a signed quantity map."""
    if snapshot_df.empty or "captured_at" not in snapshot_df.columns or "symbol" not in snapshot_df.columns:
        return {}

    normalized = snapshot_df.copy()
    normalized["captured_at"] = pd.to_datetime(normalized["captured_at"], errors="coerce")
    normalized = normalized.dropna(subset=["captured_at"])
    if normalized.empty:
        return {}

    latest_capture = normalized["captured_at"].max()
    latest_rows = normalized[normalized["captured_at"] == latest_capture]
    snapshot_positions: Dict[str, int] = {}
    for _, row in latest_rows.iterrows():
        symbol = str(row.get("symbol", "")).upper()
        qty = int(round(safe_float(row.get("qty", 0.0))))
        side = str(row.get("side", "")).lower()
        if side == "short":
            qty = -abs(qty)
        elif side == "long":
            qty = abs(qty)
        if symbol:
            snapshot_positions[symbol] = qty
    return snapshot_positions


def find_position_reconciliation_mismatches(
    current_positions: Dict[str, int],
    latest_snapshot_positions: Dict[str, int],
) -> List[str]:
    """Report symbols whose live quantities differ from the most recent local snapshot."""
    mismatches: List[str] = []
    for symbol in sorted(set(current_positions) | set(latest_snapshot_positions)):
        live_qty = int(current_positions.get(symbol, 0))
        local_qty = int(latest_snapshot_positions.get(symbol, 0))
        if live_qty != local_qty:
            mismatches.append(f"{symbol}: live={live_qty}, local={local_qty}")
    return mismatches


def get_orphan_position_symbols(current_positions: Dict[str, int], managed_symbols: Set[str]) -> Set[str]:
    """Return live positions that no longer belong to the current signal universe."""
    return {
        symbol
        for symbol, qty in current_positions.items()
        if int(qty) != 0 and symbol not in managed_symbols
    }


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


def estimate_active_cycle_start(
    pair: str,
    action: str,
    current_position: float,
    lifecycle_log: pd.DataFrame,
    as_of: pd.Timestamp,
) -> pd.Timestamp:
    """Estimate when the currently active pair cycle began from lifecycle history."""
    if not np.isfinite(current_position) or int(current_position) == 0:
        return pd.NaT
    if action not in {"LONG_SPREAD", "SHORT_SPREAD"}:
        return pd.NaT
    if lifecycle_log.empty or "pair" not in lifecycle_log.columns:
        return as_of

    history = lifecycle_log[lifecycle_log["pair"].astype(str) == str(pair)].copy()
    if history.empty:
        return as_of
    if "captured_at" not in history.columns or "current_action" not in history.columns:
        return as_of

    history["captured_at"] = pd.to_datetime(history["captured_at"], errors="coerce")
    history["current_position"] = pd.to_numeric(history.get("current_position"), errors="coerce")
    history = history.dropna(subset=["captured_at"]).sort_values("captured_at")

    cycle_start = as_of
    for _, prior in history.iloc[::-1].iterrows():
        prior_position = safe_float(prior.get("current_position"))
        prior_action = str(prior.get("current_action", "")).strip()
        if not np.isfinite(prior_position) or int(prior_position) == 0:
            break
        if prior_action != action:
            break
        cycle_start = prior["captured_at"]

    return cycle_start


def build_live_pair_risk_rows(
    live_universe: pd.DataFrame,
    attribution_df: pd.DataFrame,
    lifecycle_log: pd.DataFrame,
    account_equity: float,
    config: AlpacaConfig,
) -> pd.DataFrame:
    """Detect stop-loss and stale-trade time-stop breaches for currently held pairs."""
    if live_universe.empty or attribution_df.empty or not np.isfinite(account_equity) or account_equity <= 0:
        return pd.DataFrame()

    live_map = live_universe.set_index("pair", drop=False).to_dict("index")
    captured_at = pd.Timestamp.now()
    stop_threshold = -account_equity * config.pair_stop_loss_fraction
    risk_rows: List[Dict[str, object]] = []

    for _, row in attribution_df.iterrows():
        pair = str(row.get("pair", "")).strip()
        if not pair or pair not in live_map:
            continue

        live_row = live_map[pair]
        latest_date = str(live_row.get("latest_date", row.get("latest_date", "")))
        pair_pnl = safe_float(row.get("pair_unrealized_pl"))
        if np.isfinite(pair_pnl) and config.pair_stop_loss_fraction > 0 and pair_pnl <= stop_threshold:
            risk_rows.append(
                {
                    "event_at": captured_at.isoformat(timespec="seconds"),
                    "latest_date": latest_date,
                    "pair": pair,
                    "event_type": "STOP_LOSS",
                    "event_value": pair_pnl,
                    "threshold_value": stop_threshold,
                    "notes": f"Live pair PnL {pair_pnl:.2f} breached stop-loss threshold {stop_threshold:.2f}.",
                }
            )

        if config.time_stop_half_lives <= 0 and config.time_stop_min_days <= 0:
            continue

        action = str(live_row.get("current_action", "")).strip()
        current_position = safe_float(live_row.get("current_position"))
        cycle_start = estimate_active_cycle_start(
            pair=pair,
            action=action,
            current_position=current_position,
            lifecycle_log=lifecycle_log,
            as_of=captured_at,
        )
        if pd.isna(cycle_start):
            continue

        hold_days = max((captured_at.date() - cycle_start.date()).days, 0)
        live_half_life = safe_float(live_row.get("live_half_life"))
        time_stop_days = config.time_stop_min_days
        if np.isfinite(live_half_life) and live_half_life > 0 and config.time_stop_half_lives > 0:
            time_stop_days = max(time_stop_days, int(np.ceil(live_half_life * config.time_stop_half_lives)))
        if time_stop_days <= 0:
            continue
        if hold_days < time_stop_days:
            continue
        if not np.isfinite(pair_pnl) or pair_pnl > 0:
            continue

        risk_rows.append(
            {
                "event_at": captured_at.isoformat(timespec="seconds"),
                "latest_date": latest_date,
                "pair": pair,
                "event_type": "TIME_STOP",
                "event_value": hold_days,
                "threshold_value": time_stop_days,
                "notes": (
                    f"Held {hold_days} days with pair PnL {pair_pnl:.2f}; "
                    f"time-stop threshold is {time_stop_days} days."
                ),
            }
        )

    if not risk_rows:
        return pd.DataFrame()

    risk_df = pd.DataFrame(risk_rows)
    risk_df = risk_df.drop_duplicates(subset=["latest_date", "pair", "event_type"], keep="last")
    return risk_df


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


def get_pairs_in_cooldown(ready_universe: pd.DataFrame, risk_rows: pd.DataFrame, cooldown_days: int = 0) -> Set[str]:
    """Block re-entry for pairs stopped in the same or a recent signal cycle."""
    if ready_universe.empty or risk_rows.empty:
        return set()

    normalized_risk = risk_rows.copy()
    normalized_risk["pair"] = normalized_risk.get("pair", "").astype(str)
    normalized_risk["latest_date"] = pd.to_datetime(normalized_risk.get("latest_date"), errors="coerce")
    normalized_risk["event_type"] = normalized_risk.get("event_type", "").astype(str)
    normalized_risk = normalized_risk[normalized_risk["event_type"].isin({"STOP_LOSS", "TIME_STOP"})]

    cooldown_pairs: Set[str] = set()
    for _, row in ready_universe.iterrows():
        pair = str(row.get("pair", ""))
        latest_date = pd.to_datetime(row.get("latest_date"), errors="coerce")
        if not pair or pd.isna(latest_date):
            continue

        matches = normalized_risk[normalized_risk["pair"] == pair].dropna(subset=["latest_date"])
        if not matches.empty:
            same_cycle = matches["latest_date"].dt.date == latest_date.date()
            if bool(same_cycle.any()):
                cooldown_pairs.add(pair)
                continue

            if cooldown_days > 0:
                recent_days = (latest_date.normalize() - matches["latest_date"].dt.normalize()).dt.days
                if bool(((recent_days >= 0) & (recent_days <= cooldown_days)).any()):
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
        "target_long_notional",
        "target_short_notional",
        "net_notional",
        "notional_imbalance_pct",
        "near_exit_no_add",
        "event_no_add",
        "event_reason",
        "order_count",
        "alpaca_status",
        "order_ids",
    ]
    available = [column for column in ordered_columns if column in trade_log.columns]
    return trade_log[available]


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


def extract_pair_symbols(pair_universe: pd.DataFrame) -> Set[str]:
    """Return valid ticker symbols represented in a pair universe."""
    if pair_universe.empty:
        return set()

    symbols: Set[str] = set()
    for column in ["stock_x", "stock_y"]:
        if column not in pair_universe.columns:
            continue
        symbols.update(
            pair_universe[column]
            .dropna()
            .astype(str)
            .str.upper()
            .loc[lambda values: (values != "") & (values != "NAN")]
        )
    return symbols


def get_flatten_symbols_from_live_universe(live_universe: pd.DataFrame, blocked_pairs: Set[str]) -> Set[str]:
    """Select symbols that should be deliberately flattened this cycle."""
    if live_universe.empty:
        return set()

    current_position = pd.Series(dtype=float)
    if "current_position" in live_universe.columns:
        current_position = pd.to_numeric(live_universe["current_position"], errors="coerce").fillna(0)

    flatten_row_indexes: Set[int] = set()
    if "current_position" in live_universe.columns:
        flatten_row_indexes.update(live_universe.index[current_position == 0].tolist())

    if blocked_pairs:
        flatten_row_indexes.update(
            live_universe.index[live_universe["pair"].astype(str).isin(blocked_pairs)].tolist()
        )

    if "live_recommendation" in live_universe.columns:
        flatten_row_indexes.update(
            live_universe.index[
                live_universe["live_recommendation"].astype(str) == "QUALIFIED_BUT_BLOCKED"
            ].tolist()
        )

    if not flatten_row_indexes:
        return set()

    flatten_rows = live_universe.loc[sorted(flatten_row_indexes)].drop_duplicates(subset=["sector", "pair"], keep="last")
    flatten_symbols = extract_pair_symbols(flatten_rows)

    active_protected_symbols = set()
    if "current_position" in live_universe.columns:
        active_mask = current_position != 0
        if "current_action" in live_universe.columns:
            active_mask = active_mask & live_universe["current_action"].astype(str).isin({"LONG_SPREAD", "SHORT_SPREAD"})
        active_rows = live_universe.loc[active_mask].drop_duplicates(subset=["sector", "pair"], keep="last")
        if not active_rows.empty:
            active_pairs = set(active_rows["pair"].astype(str))
            flatten_pairs = set(flatten_rows["pair"].astype(str))
            protected_rows = active_rows[~active_rows["pair"].astype(str).isin(flatten_pairs)]
            active_protected_symbols = extract_pair_symbols(protected_rows)

    return flatten_symbols - active_protected_symbols


def build_pair_lifecycle_rows(pair_universe: pd.DataFrame, pair_trade_plan: pd.DataFrame) -> pd.DataFrame:
    """Record the daily intended lifecycle state for each candidate pair."""
    if pair_universe.empty:
        return pd.DataFrame()

    lifecycle = pair_universe.copy()
    lifecycle["captured_at"] = datetime.now().isoformat(timespec="seconds")
    executable_pairs = set(pair_trade_plan["pair"].astype(str)) if not pair_trade_plan.empty else set()
    if "portfolio_weight" not in lifecycle.columns:
        lifecycle["portfolio_weight"] = 0.0

    if not pair_trade_plan.empty and "portfolio_weight" in pair_trade_plan.columns:
        weight_map = pair_trade_plan.set_index("pair")["portfolio_weight"].to_dict()
        lifecycle["portfolio_weight"] = lifecycle.apply(
            lambda row: safe_float(weight_map.get(row.get("pair"), row.get("portfolio_weight", 0.0))),
            axis=1,
        )

    def execution_state(row: pd.Series) -> str:
        pair = str(row.get("pair", ""))
        if pair in executable_pairs:
            return "EXECUTABLE"
        if str(row.get("live_recommendation", "")) == "QUALIFIED_BUT_BLOCKED":
            return "QUALIFIED_BUT_BLOCKED"
        current_position = safe_float(row.get("current_position"))
        if np.isfinite(current_position) and int(current_position) == 0:
            return "FLATTEN_IF_HELD"
        return "NON_EXECUTABLE"

    lifecycle["execution_state"] = lifecycle.apply(execution_state, axis=1)

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


def reconcile_order_fill_log(client: AlpacaClient, path: Path) -> pd.DataFrame:
    """Refresh known Alpaca order statuses so local fill logs do not stay stale."""
    fill_log = load_csv(path)
    if fill_log.empty or "alpaca_order_id" not in fill_log.columns:
        return fill_log

    reconciled = fill_log.copy()
    for column in ["filled_at", "expired_at", "canceled_at"]:
        if column not in reconciled.columns:
            reconciled[column] = ""

    for index, row in reconciled.iterrows():
        order_id = str(row.get("alpaca_order_id", "")).strip()
        if not order_id or order_id.lower() == "nan":
            continue

        try:
            order = client.get_order(order_id)
        except requests.HTTPError as exc:
            reconciled.at[index, "error"] = str(exc)
            continue

        reconciled.at[index, "alpaca_status"] = str(order.get("status", ""))
        reconciled.at[index, "filled_qty"] = safe_float(order.get("filled_qty"))
        reconciled.at[index, "filled_avg_price"] = safe_float(order.get("filled_avg_price"))
        reconciled.at[index, "created_at"] = order.get("created_at", "")
        reconciled.at[index, "updated_at"] = order.get("updated_at", "")
        reconciled.at[index, "filled_at"] = order.get("filled_at", "")
        reconciled.at[index, "expired_at"] = order.get("expired_at", "")
        reconciled.at[index, "canceled_at"] = order.get("canceled_at", "")

    reconciled.to_csv(path, index=False)
    return reconciled


def build_pair_attribution_rows(pair_trade_plan: pd.DataFrame, positions: List[Dict[str, object]]) -> pd.DataFrame:
    """Build current share-level attribution for active pair targets."""
    if pair_trade_plan.empty:
        return pd.DataFrame()

    position_details = build_position_details_map(positions)
    captured_at = datetime.now().isoformat(timespec="seconds")
    rows: List[Dict[str, object]] = []

    for _, row in pair_trade_plan.iterrows():
        long_symbol = str(row.get("long_symbol", "")).upper()
        short_symbol = str(row.get("short_symbol", "")).upper()
        long_qty = int(safe_float(row.get("long_qty", 0)))
        short_qty = int(safe_float(row.get("short_qty", 0)))
        long_position = position_details.get(long_symbol, {})
        short_position = position_details.get(short_symbol, {})

        current_long_qty = max(int(long_position.get("qty", 0)), 0)
        current_short_qty = max(-int(short_position.get("qty", 0)), 0)
        matched_long_qty = min(current_long_qty, long_qty)
        matched_short_qty = min(current_short_qty, short_qty)

        long_entry = safe_float(long_position.get("avg_entry_price"))
        long_price = safe_float(long_position.get("current_price"))
        short_entry = safe_float(short_position.get("avg_entry_price"))
        short_price = safe_float(short_position.get("current_price"))

        long_pnl = (
            (long_price - long_entry) * matched_long_qty
            if matched_long_qty > 0 and np.isfinite(long_entry) and np.isfinite(long_price)
            else float("nan")
        )
        short_pnl = (
            (short_entry - short_price) * matched_short_qty
            if matched_short_qty > 0 and np.isfinite(short_entry) and np.isfinite(short_price)
            else float("nan")
        )
        actual_long_notional = long_price * matched_long_qty if np.isfinite(long_price) else float("nan")
        actual_short_notional = short_price * matched_short_qty if np.isfinite(short_price) else float("nan")
        actual_gross_notional = actual_long_notional + actual_short_notional
        actual_net_notional = actual_long_notional - actual_short_notional
        actual_imbalance_pct = (
            abs(actual_net_notional) / actual_gross_notional
            if np.isfinite(actual_gross_notional) and actual_gross_notional > 0
            else float("nan")
        )

        rows.append(
            {
                "captured_at": captured_at,
                "signal_date": row.get("signal_date", ""),
                "pair": row.get("pair", ""),
                "action": row.get("action", ""),
                "live_zscore": safe_float(row.get("live_zscore")),
                "long_symbol": long_symbol,
                "long_target_qty": long_qty,
                "long_current_qty": current_long_qty,
                "long_avg_entry_price": long_entry,
                "long_current_price": long_price,
                "long_unrealized_pl": long_pnl,
                "short_symbol": short_symbol,
                "short_target_qty": short_qty,
                "short_current_qty": current_short_qty,
                "short_avg_entry_price": short_entry,
                "short_current_price": short_price,
                "short_unrealized_pl": short_pnl,
                "pair_unrealized_pl": np.nansum([long_pnl, short_pnl]),
                "actual_long_notional": actual_long_notional,
                "actual_short_notional": actual_short_notional,
                "actual_net_notional": actual_net_notional,
                "actual_imbalance_pct": actual_imbalance_pct,
                "target_long_notional": safe_float(row.get("target_long_notional")),
                "target_short_notional": safe_float(row.get("target_short_notional")),
                "target_net_notional": safe_float(row.get("net_notional")),
                "target_imbalance_pct": safe_float(row.get("notional_imbalance_pct")),
                "near_exit_no_add": bool(row.get("near_exit_no_add", False)),
            }
        )

    return pd.DataFrame(rows)


def build_live_pair_attribution_rows(live_universe: pd.DataFrame, positions: List[Dict[str, object]]) -> pd.DataFrame:
    """Build attribution for currently held pairs, even when they are not executable today."""
    if live_universe.empty:
        return pd.DataFrame()

    position_details = build_position_details_map(positions)
    captured_at = datetime.now().isoformat(timespec="seconds")
    rows: List[Dict[str, object]] = []

    for _, row in live_universe.iterrows():
        action = str(row.get("current_action", "")).strip()
        if action == "LONG_SPREAD":
            long_symbol = str(row.get("stock_x", "")).upper()
            short_symbol = str(row.get("stock_y", "")).upper()
        elif action == "SHORT_SPREAD":
            long_symbol = str(row.get("stock_y", "")).upper()
            short_symbol = str(row.get("stock_x", "")).upper()
        else:
            continue

        long_position = position_details.get(long_symbol, {})
        short_position = position_details.get(short_symbol, {})
        long_qty = max(int(long_position.get("qty", 0)), 0)
        short_qty = max(-int(short_position.get("qty", 0)), 0)
        if long_qty <= 0 or short_qty <= 0:
            continue

        long_entry = safe_float(long_position.get("avg_entry_price"))
        long_price = safe_float(long_position.get("current_price"))
        short_entry = safe_float(short_position.get("avg_entry_price"))
        short_price = safe_float(short_position.get("current_price"))
        long_pnl = (
            (long_price - long_entry) * long_qty
            if long_qty > 0 and np.isfinite(long_entry) and np.isfinite(long_price)
            else float("nan")
        )
        short_pnl = (
            (short_entry - short_price) * short_qty
            if short_qty > 0 and np.isfinite(short_entry) and np.isfinite(short_price)
            else float("nan")
        )
        actual_long_notional = long_price * long_qty if np.isfinite(long_price) else float("nan")
        actual_short_notional = short_price * short_qty if np.isfinite(short_price) else float("nan")
        actual_gross_notional = actual_long_notional + actual_short_notional
        actual_net_notional = actual_long_notional - actual_short_notional
        actual_imbalance_pct = (
            abs(actual_net_notional) / actual_gross_notional
            if np.isfinite(actual_gross_notional) and actual_gross_notional > 0
            else float("nan")
        )

        rows.append(
            {
                "captured_at": captured_at,
                "latest_date": row.get("latest_date", ""),
                "pair": row.get("pair", ""),
                "action": action,
                "live_recommendation": row.get("live_recommendation", ""),
                "live_zscore": safe_float(row.get("live_zscore")),
                "live_beta": safe_float(row.get("live_beta")),
                "long_symbol": long_symbol,
                "long_current_qty": long_qty,
                "long_avg_entry_price": long_entry,
                "long_current_price": long_price,
                "long_unrealized_pl": long_pnl,
                "short_symbol": short_symbol,
                "short_current_qty": short_qty,
                "short_avg_entry_price": short_entry,
                "short_current_price": short_price,
                "short_unrealized_pl": short_pnl,
                "pair_unrealized_pl": np.nansum([long_pnl, short_pnl]),
                "actual_long_notional": actual_long_notional,
                "actual_short_notional": actual_short_notional,
                "actual_net_notional": actual_net_notional,
                "actual_imbalance_pct": actual_imbalance_pct,
            }
        )

    return pd.DataFrame(rows)


def build_pair_roundtrip_rows(
    lifecycle_log: pd.DataFrame,
    attribution_log: pd.DataFrame,
    risk_rows: pd.DataFrame,
    existing_roundtrips: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Summarize closed pair cycles into one row per round-trip trade."""
    if lifecycle_log.empty or attribution_log.empty:
        return pd.DataFrame()

    lifecycle = lifecycle_log.copy()
    attribution = attribution_log.copy()
    lifecycle["captured_at"] = pd.to_datetime(lifecycle["captured_at"], errors="coerce")
    attribution["captured_at"] = pd.to_datetime(attribution["captured_at"], errors="coerce")
    lifecycle["current_position"] = pd.to_numeric(lifecycle.get("current_position"), errors="coerce")
    lifecycle = lifecycle.dropna(subset=["captured_at"]).sort_values(["pair", "captured_at"])
    attribution = attribution.dropna(subset=["captured_at"]).sort_values(["pair", "captured_at"])

    existing_ids: Set[str] = set()
    if existing_roundtrips is not None and not existing_roundtrips.empty and "cycle_id" in existing_roundtrips.columns:
        existing_ids = set(existing_roundtrips["cycle_id"].astype(str))

    risk_lookup: Dict[tuple, str] = {}
    if not risk_rows.empty and {"pair", "latest_date", "event_type"}.issubset(risk_rows.columns):
        for _, risk in risk_rows.iterrows():
            risk_lookup[(str(risk.get("pair", "")), str(risk.get("latest_date", "")))] = str(risk.get("event_type", ""))

    roundtrip_rows: List[Dict[str, object]] = []
    for pair, pair_lifecycle in lifecycle.groupby("pair", sort=True):
        active_cycle: Optional[Dict[str, object]] = None
        for _, row in pair_lifecycle.iterrows():
            action = str(row.get("current_action", "")).strip()
            current_position = safe_float(row.get("current_position"))
            is_active = np.isfinite(current_position) and int(current_position) != 0 and action in {"LONG_SPREAD", "SHORT_SPREAD"}

            if active_cycle is None:
                if is_active:
                    active_cycle = row.to_dict()
                continue

            same_cycle = is_active and str(active_cycle.get("current_action", "")) == action
            if same_cycle:
                continue

            entry_at = parse_timestamp(active_cycle.get("captured_at"))
            exit_at = parse_timestamp(row.get("captured_at"))
            cycle_id = hashlib.sha1(f"{pair}|{entry_at.isoformat()}|{active_cycle.get('current_action','')}".encode("utf-8")).hexdigest()[:16]
            if cycle_id not in existing_ids and pd.notna(entry_at) and pd.notna(exit_at) and exit_at > entry_at:
                cycle_attr = attribution[
                    (attribution["pair"].astype(str) == str(pair))
                    & (attribution["captured_at"] >= entry_at)
                    & (attribution["captured_at"] <= exit_at)
                ].copy()
                if not cycle_attr.empty:
                    first_attr = cycle_attr.iloc[0]
                    last_attr = cycle_attr.iloc[-1]
                    long_exit_pl = safe_float(last_attr.get("long_unrealized_pl"))
                    short_exit_pl = safe_float(last_attr.get("short_unrealized_pl"))
                    pair_exit_pl = safe_float(last_attr.get("pair_unrealized_pl"))
                    both_positive_exit = np.isfinite(long_exit_pl) and np.isfinite(short_exit_pl) and long_exit_pl > 0 and short_exit_pl > 0
                    both_negative_exit = np.isfinite(long_exit_pl) and np.isfinite(short_exit_pl) and long_exit_pl < 0 and short_exit_pl < 0
                    split_sign_exit = np.isfinite(long_exit_pl) and np.isfinite(short_exit_pl) and ((long_exit_pl > 0 > short_exit_pl) or (short_exit_pl > 0 > long_exit_pl))
                    exit_signal_date = str(row.get("latest_date", ""))
                    roundtrip_rows.append(
                        {
                            "cycle_id": cycle_id,
                            "pair": pair,
                            "action": str(active_cycle.get("current_action", "")),
                            "entry_at": entry_at.isoformat(),
                            "exit_at": exit_at.isoformat(),
                            "entry_signal_date": str(active_cycle.get("latest_date", "")),
                            "exit_signal_date": exit_signal_date,
                            "hold_days": max((exit_at.date() - entry_at.date()).days, 0),
                            "entry_live_zscore": safe_float(active_cycle.get("live_zscore")),
                            "exit_live_zscore": safe_float(row.get("live_zscore")),
                            "entry_live_beta": safe_float(active_cycle.get("live_beta")),
                            "entry_live_half_life": safe_float(active_cycle.get("live_half_life")),
                            "entry_portfolio_weight": safe_float(active_cycle.get("portfolio_weight")),
                            "long_symbol": str(first_attr.get("long_symbol", "")),
                            "short_symbol": str(first_attr.get("short_symbol", "")),
                            "max_favorable_excursion": pd.to_numeric(cycle_attr["pair_unrealized_pl"], errors="coerce").max(),
                            "max_adverse_excursion": pd.to_numeric(cycle_attr["pair_unrealized_pl"], errors="coerce").min(),
                            "exit_long_unrealized_pl": long_exit_pl,
                            "exit_short_unrealized_pl": short_exit_pl,
                            "exit_pair_unrealized_pl": pair_exit_pl,
                            "both_legs_positive_at_exit": both_positive_exit,
                            "both_legs_negative_at_exit": both_negative_exit,
                            "split_sign_at_exit": split_sign_exit,
                            "both_legs_positive_rate": float(((pd.to_numeric(cycle_attr["long_unrealized_pl"], errors="coerce") > 0) & (pd.to_numeric(cycle_attr["short_unrealized_pl"], errors="coerce") > 0)).mean()),
                            "exit_reason": risk_lookup.get((pair, exit_signal_date), str(row.get("execution_state", ""))),
                        }
                    )

            active_cycle = row.to_dict() if is_active else None

    return pd.DataFrame(roundtrip_rows)


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
    parser = argparse.ArgumentParser(description="Submit ELIGIBLE pair trades to Alpaca paper trading.")
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
    live_signals = load_csv(LIVE_SIGNALS_INPUT)
    ranked_pairs = load_csv(RANKED_PAIRS_INPUT)

    if ready_signals.empty and live_signals.empty:
        raise ValueError(
            f"Missing or empty signal files: {READY_SIGNALS_INPUT.resolve()} and {LIVE_SIGNALS_INPUT.resolve()}"
        )
    if ranked_pairs.empty:
        raise ValueError(f"Missing or empty ranked pair file: {RANKED_PAIRS_INPUT.resolve()}")

    signal_universe_source = live_signals if not live_signals.empty else ready_signals
    staleness_days = get_signal_staleness_days(signal_universe_source)

    ready_universe = build_ready_universe(ready_signals, ranked_pairs)
    live_universe = build_ready_universe(signal_universe_source, ranked_pairs)
    if live_universe.empty:
        raise ValueError("No ticker legs could be resolved from the live signal and ranked pair files.")

    client = AlpacaClient(config)
    account = client.get_account()
    positions = client.get_positions()
    reconciled_fills = reconcile_order_fill_log(client, ORDER_FILL_LOG_OUTPUT)
    if not reconciled_fills.empty:
        print(f"Reconciled {len(reconciled_fills)} local order fill row(s) with Alpaca.")

    account_equity = safe_float(account.get("equity"))
    account_buying_power = safe_float(account.get("buying_power"))
    if not np.isfinite(account_equity) or account_equity <= 0:
        raise ValueError("Alpaca account equity is unavailable or invalid.")

    deployable_capital = determine_deployable_capital(account, config)
    if not np.isfinite(deployable_capital) or deployable_capital <= 0:
        raise ValueError("No deployable capital is available from equity/buying power constraints.")

    attribution_df = build_live_pair_attribution_rows(live_universe, positions)
    prior_lifecycle_log = load_csv(PAIR_LIFECYCLE_LOG_OUTPUT)
    initial_pair_trade_plan = build_pair_trade_plan(ready_universe, deployable_capital, config)
    position_details = build_position_details_map(positions)
    stop_loss_rows = build_pair_risk_rows(initial_pair_trade_plan, position_details, account_equity, config)
    live_risk_rows = build_live_pair_risk_rows(live_universe, attribution_df, prior_lifecycle_log, account_equity, config)
    new_risk_rows = pd.concat([stop_loss_rows, live_risk_rows], ignore_index=True) if not stop_loss_rows.empty or not live_risk_rows.empty else pd.DataFrame()
    if not new_risk_rows.empty:
        upsert_risk_rows(PAIR_RISK_EVENTS_LOG_OUTPUT, new_risk_rows)
    risk_rows = load_pair_risk_rows(PAIR_RISK_EVENTS_LOG_OUTPUT)
    prior_trade_log = load_csv(TRADE_LOG_OUTPUT)

    blocked_pairs = set(config.pair_denylist)
    if not ready_universe.empty:
        blocked_pairs.update(get_pairs_in_cooldown(ready_universe, risk_rows, cooldown_days=config.reentry_cooldown_days))
    already_submitted_pairs = get_pairs_already_submitted_this_cycle(ready_universe, prior_trade_log)
    ready_universe = filter_blocked_pairs(ready_universe, blocked_pairs | already_submitted_pairs)

    leg_targets = build_leg_targets(ready_universe, deployable_capital, config)
    pair_trade_plan = build_pair_trade_plan(ready_universe, deployable_capital, config)
    pair_lifecycle_df = build_pair_lifecycle_rows(live_universe, pair_trade_plan)
    account_snapshot_df = build_account_snapshot_rows(account, deployable_capital)
    positions_snapshot_df = build_positions_snapshot_rows(positions)

    current_positions = build_current_position_map(positions)
    latest_snapshot_positions = build_latest_snapshot_position_map(load_csv(POSITIONS_SNAPSHOT_LOG_OUTPUT))
    reconciliation_mismatches = find_position_reconciliation_mismatches(current_positions, latest_snapshot_positions)

    append_csv_rows(ACCOUNT_SNAPSHOT_LOG_OUTPUT, account_snapshot_df)
    append_csv_rows(POSITIONS_SNAPSHOT_LOG_OUTPUT, positions_snapshot_df)
    append_csv_rows(PAIR_LIFECYCLE_LOG_OUTPUT, pair_lifecycle_df)
    append_csv_rows(PAIR_ATTRIBUTION_LOG_OUTPUT, attribution_df)
    attribution_df.to_csv(PAIR_ATTRIBUTION_OUTPUT, index=False)
    roundtrip_rows = build_pair_roundtrip_rows(
        lifecycle_log=load_csv(PAIR_LIFECYCLE_LOG_OUTPUT),
        attribution_log=load_csv(PAIR_ATTRIBUTION_LOG_OUTPUT),
        risk_rows=risk_rows,
        existing_roundtrips=load_csv(PAIR_ROUNDTRIP_LOG_OUTPUT),
    )
    append_csv_rows(PAIR_ROUNDTRIP_LOG_OUTPUT, roundtrip_rows)

    all_managed_symbols = sorted(extract_pair_symbols(live_universe))
    flatten_symbols = get_flatten_symbols_from_live_universe(live_universe, blocked_pairs)
    orphan_symbols = get_orphan_position_symbols(current_positions, set(all_managed_symbols))
    flatten_symbols.update(orphan_symbols)
    if config.flatten_on_no_targets and leg_targets.empty:
        flatten_symbols.update(all_managed_symbols)

    preview_df = build_order_preview(
        leg_targets,
        current_positions,
        all_managed_symbols,
        flatten_symbols=flatten_symbols,
        config=config,
    )
    preview_df.to_csv(ORDER_PREVIEW_OUTPUT, index=False)

    print(f"Account equity: ${account_equity:,.2f}")
    if np.isfinite(account_buying_power):
        print(f"Account buying power: ${account_buying_power:,.2f}")
    print(f"Deployable capital: ${deployable_capital:,.2f}")
    print(f"Preview saved to: {ORDER_PREVIEW_OUTPUT.resolve()}")
    print(f"Pair attribution saved to: {PAIR_ATTRIBUTION_OUTPUT.resolve()}")
    if not roundtrip_rows.empty:
        print(f"Pair round-trip log updated: {PAIR_ROUNDTRIP_LOG_OUTPUT.resolve()}")
    if flatten_symbols:
        print(f"Symbols marked for flatten-if-held: {', '.join(sorted(flatten_symbols))}")
    if orphan_symbols:
        print(f"Orphan symbols marked for flatten: {', '.join(sorted(orphan_symbols))}")
    if blocked_pairs:
        print(f"Blocked pairs this cycle: {', '.join(sorted(blocked_pairs))}")
    if already_submitted_pairs:
        print(f"Already submitted this signal cycle: {', '.join(sorted(already_submitted_pairs))}")
    if config.min_expected_edge > 0:
        print(f"Minimum expected edge per trade: {config.min_expected_edge:.4f}")
    if reconciliation_mismatches:
        mismatch_summary = "; ".join(reconciliation_mismatches)
        print(f"Position reconciliation mismatch detected: {mismatch_summary}")
    if leg_targets.empty or pair_trade_plan.empty:
        if config.flatten_on_no_targets:
            print("No executable target legs were produced from the ready pairs. Managed symbols will be flattened.")
        else:
            print("No executable target legs were produced from the ready pairs. Existing positions will be left unchanged.")
    if staleness_days > config.max_signal_staleness_days:
        latest_signal_date = pd.to_datetime(signal_universe_source["latest_date"]).max().date().isoformat()
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

    if config.fail_on_reconcile_mismatch and reconciliation_mismatches:
        raise ValueError(
            "Live Alpaca positions do not match the most recent local snapshot. "
            f"Resolve or reconcile before trading: {'; '.join(reconciliation_mismatches)}"
        )

    if staleness_days > config.max_signal_staleness_days and not args.allow_stale:
        raise ValueError(
            f"Ready signals are stale by {staleness_days} days. "
            f"Rerun pair_checker.py and paper_trading_ready.py first, or pass --allow-stale to override."
        )

    cancelled_orders_df = cancel_conflicting_open_orders(client, preview_df)
    if not cancelled_orders_df.empty:
        print(f"Cancelled {len(cancelled_orders_df)} open order(s) that conflicted with this rebalance.")

    latest_signal_date = pd.to_datetime(signal_universe_source["latest_date"]).max().date().isoformat()
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
