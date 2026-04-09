import itertools
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import adfuller, coint


# =========================
# Configuration
# =========================

START_DATE = "2022-01-01"
END_DATE = None

Z_WINDOW = 20
ENTRY_Z = 1.75
EXIT_Z = 0.0

MIN_CORR = 0.65
MAX_COINTEGRATION_P = 0.10
MAX_ADF_P = 0.10
MIN_HALFLIFE = 2.0
MAX_HALFLIFE = 60.0

COST_PER_TURN = 0.0005
SLIPPAGE_PER_TURN = 0.0005

TRAINING_WINDOW_DAYS = 252
TEST_WINDOW_DAYS = 63
WALK_FORWARD_STEP_DAYS = 21
WALK_FORWARD_VALIDATION_MODE = "overlapping"  # "overlapping" or "non_overlapping"

MIN_PASSING_WINDOWS = 1
MIN_TOTAL_OOS_TRADES = 5
QUALIFYING_MIN_WINDOWS_PASSED = 2
QUALIFYING_MIN_OOS_TRADES = 5
QUALIFYING_MIN_OOS_SHARPE = 0.0
QUALIFYING_MIN_OOS_RETURN = 0.0

USE_STRUCTURAL_PASSED_WINDOW_FILTER = True
MAX_AVG_COINT_PVALUE_PASSED = 0.10
MAX_AVG_ADF_PVALUE_PASSED = 0.10
MIN_AVG_HALFLIFE_PASSED = 3.0
MAX_AVG_HALFLIFE_PASSED = 40.0

MAX_MISSING_RATIO = 0.10
MIN_ROWS_PER_PAIR = TRAINING_WINDOW_DAYS + TEST_WINDOW_DAYS

TOP_N_PRINT = 20
TOP_N_PLOTS = 5
LIVE_SIGNAL_TOP_N = 10

RANKED_OUTPUT_CSV = "ranked_pairs_walk_forward.csv"
FAILED_OUTPUT_CSV = "failed_pairs_diagnostics.csv"
NEAR_MISS_OUTPUT_CSV = "near_miss_pairs.csv"
WINDOW_OUTPUT_CSV = "walk_forward_window_metrics.csv"
LIVE_SIGNALS_OUTPUT_CSV = "live_pair_signals.csv"
SUMMARY_REPORT_MD = "pair_research_summary.md"
PLOT_DIR = Path("pair_plots")

ROBUSTNESS_ENTRY_Z_DELTA = 0.25
ROBUSTNESS_Z_WINDOW_DELTA = 5
ROBUSTNESS_MIN_PASS_RATE = 0.60

LIVE_STABILITY_LOOKBACK = 60
LIVE_ROLLING_CORR_WINDOW = 20
LIVE_BETA_WINDOW = 60
LIVE_SPREAD_VOL_WINDOW = 20
LIVE_MIN_RECENT_CORR_MEAN = 0.55
LIVE_MAX_RECENT_CORR_STD = 0.20
LIVE_MAX_RECENT_BETA_STD = 0.20
LIVE_MAX_RECENT_SPREAD_VOL_RATIO = 1.75

SCORE_SHARPE_WEIGHT = 4.0
SCORE_RETURN_WEIGHT = 1.5
SCORE_DRAWDOWN_PENALTY = 2.0
SCORE_PASSING_WINDOW_WEIGHT = 0.25
SCORE_TRADE_WEIGHT = 0.05


UNIVERSE: Dict[str, List[str]] = {
    "banks": ["JPM", "BAC", "WFC", "C", "MS", "GS", "PNC", "USB", "BK", "SCHW"],
    "payments": ["V", "MA", "AXP", "PYPL", "COF"],
    "semis": ["NVDA", "AMD", "AVGO", "QCOM", "MU", "INTC", "AMAT", "LRCX", "KLAC", "TXN", "ADI"],
    "oil_majors": ["XOM", "CVX"],
    "energy_e&p": ["COP", "EOG", "OXY", "DVN"],
    "energy_services": ["SLB", "HAL", "BKR"],
    "home_improvement": ["HD", "LOW"],
    "railroads": ["UNP", "CSX", "NSC"],
    "exchanges": ["ICE", "CME", "NDAQ"],
    "managed_care": ["UNH", "ELV", "CI", "HUM"],
    "defense": ["LMT", "RTX", "NOC", "GD"],
}


def resolve_end_date() -> str:
    """Return the configured end date, defaulting to tomorrow for current data pulls."""
    if END_DATE:
        return END_DATE
    return (date.today() + timedelta(days=1)).isoformat()


@dataclass
class PairMetrics:
    sharpe: float
    total_return: float
    annualized_return: float
    max_drawdown: float
    trades: int
    return_per_trade: float


@dataclass
class BacktestResult:
    metrics: PairMetrics
    net_pnl: pd.Series
    equity_curve: pd.Series
    spread: pd.Series
    zscore: pd.Series
    position: pd.Series
    turns: pd.Series


@dataclass
class PairResult:
    sector: str
    stock_x: str
    stock_y: str
    pair: str
    windows_tested: int
    windows_passed: int
    avg_train_corr_passed: float
    avg_coint_pvalue_passed: float
    avg_adf_pvalue_passed: float
    avg_half_life_passed: float
    latest_train_corr: float
    latest_beta: float
    latest_coint_pvalue: float
    latest_adf_pvalue: float
    latest_half_life: float
    oos_sharpe: float
    oos_return: float
    oos_annualized_return: float
    oos_max_drawdown: float
    oos_trades: int
    oos_return_per_trade: float
    oos_accumulated_test_days: int
    oos_unique_test_days: int
    research_verdict: str
    research_recommendation: str
    walk_forward_mode: str
    robustness_score: float
    robustness_pass_rate: float
    robustness_pass_count: int
    robustness_scenarios_tested: int
    confidence_score: float
    confidence_rank: int
    score_sharpe_component: float
    score_return_component: float
    score_drawdown_penalty: float
    score_windows_component: float
    score_trades_component: float
    score: float


def flatten_universe(universe: Dict[str, List[str]]) -> List[str]:
    """Return unique tickers from the sector universe."""
    tickers: List[str] = []
    for names in universe.values():
        tickers.extend(names)
    return sorted(set(tickers))


def download_prices(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """Download adjusted close prices for the requested tickers."""
    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    if data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        data = data["Close"]

    if isinstance(data, pd.Series):
        data = data.to_frame()

    data = data.sort_index()
    return data.astype(float)


def safe_float(value: object) -> float:
    """Convert values to float while preserving missing values as NaN."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def fit_beta(log_x: pd.Series, log_y: pd.Series) -> float:
    """Estimate OLS beta for log_x ~ beta * log_y."""
    aligned = pd.concat([log_x, log_y], axis=1).dropna()
    if len(aligned) < 2:
        return np.nan

    x_values = aligned.iloc[:, 0].values
    y_values = aligned.iloc[:, 1].values
    y_var = np.var(y_values, ddof=1)
    if not np.isfinite(y_var) or y_var <= 0:
        return np.nan

    beta = np.cov(x_values, y_values, ddof=1)[0, 1] / y_var
    return safe_float(beta) if np.isfinite(beta) else np.nan


def compute_spread(log_x: pd.Series, log_y: pd.Series, beta: float) -> pd.Series:
    """Compute the pair spread from log prices and a fixed beta."""
    return log_x - beta * log_y


def compute_zscore(series: pd.Series, window: int) -> pd.Series:
    """Compute a rolling z-score with divide-by-zero protection."""
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std().replace(0.0, np.nan)
    zscore = (series - rolling_mean) / rolling_std
    return zscore.replace([np.inf, -np.inf], np.nan)


def estimate_half_life(spread: pd.Series) -> float:
    """Estimate mean-reversion half-life from a lagged spread regression."""
    spread = spread.dropna()
    if len(spread) < 20:
        return np.nan

    lagged = spread.shift(1)
    delta = spread.diff()
    aligned = pd.concat([lagged, delta], axis=1).dropna()
    if len(aligned) < 10:
        return np.nan

    x = aligned.iloc[:, 0].values
    y = aligned.iloc[:, 1].values
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    denom = np.sum((x - x_mean) ** 2)
    if not np.isfinite(denom) or denom <= 0:
        return np.nan

    beta = np.sum((x - x_mean) * (y - y_mean)) / denom
    if not np.isfinite(beta) or beta >= 0:
        return np.inf

    half_life = -np.log(2) / beta
    return safe_float(half_life) if np.isfinite(half_life) else np.nan


def calc_adf_pvalue(series: pd.Series) -> float:
    """Return the ADF p-value, or NaN if the test fails."""
    series = series.dropna()
    if len(series) < 30 or series.std() == 0:
        return np.nan

    try:
        return safe_float(adfuller(series)[1])
    except Exception:
        return np.nan


def calc_coint_pvalue(log_x: pd.Series, log_y: pd.Series) -> float:
    """Return the cointegration p-value, or NaN if the test fails."""
    aligned = pd.concat([log_x, log_y], axis=1).dropna()
    if len(aligned) < 30:
        return np.nan

    try:
        return safe_float(coint(aligned.iloc[:, 0], aligned.iloc[:, 1])[1])
    except Exception:
        return np.nan


def max_drawdown(equity_curve: pd.Series) -> float:
    """Compute max drawdown from an equity curve."""
    equity_curve = equity_curve.replace([np.inf, -np.inf], np.nan).dropna()
    if equity_curve.empty:
        return np.nan

    running_max = equity_curve.cummax().replace(0.0, np.nan)
    drawdown = equity_curve / running_max - 1.0
    return safe_float(drawdown.min())


def annualize_return(total_return: float, periods: int, periods_per_year: int = 252) -> float:
    """Annualize cumulative return over a given number of trading periods."""
    if pd.isna(total_return) or periods <= 0:
        return np.nan

    terminal_value = 1.0 + total_return
    if terminal_value <= 0:
        return np.nan

    return safe_float(terminal_value ** (periods_per_year / periods) - 1.0)


def compute_pair_metrics(net_pnl: pd.Series, turns: pd.Series) -> Optional[PairMetrics]:
    """Compute summary metrics from a net pnl series."""
    net_pnl = net_pnl.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=-0.99, upper=0.99)
    if net_pnl.empty:
        return None

    equity_curve = (1.0 + net_pnl).cumprod()
    if equity_curve.empty or (equity_curve <= 0).any():
        return None

    pnl_std = net_pnl.std()
    sharpe = np.nan
    if np.isfinite(pnl_std) and pnl_std > 0:
        sharpe = np.sqrt(252) * net_pnl.mean() / pnl_std

    trades = int((turns.fillna(0.0) > 0).sum() / 2)
    total_return = safe_float(equity_curve.iloc[-1] - 1.0)
    return PairMetrics(
        sharpe=safe_float(sharpe),
        total_return=total_return,
        annualized_return=annualize_return(total_return=total_return, periods=len(net_pnl)),
        max_drawdown=max_drawdown(equity_curve),
        trades=trades,
        return_per_trade=safe_float(total_return / trades) if trades > 0 else np.nan,
    )


def build_positions(zscore: pd.Series, entry_z: float, exit_z: float) -> pd.Series:
    """Build spread positions from z-score entry and exit thresholds."""
    position = pd.Series(index=zscore.index, dtype=float)
    current = 0.0

    for i, z_value in enumerate(zscore):
        if pd.isna(z_value):
            position.iloc[i] = current
            continue

        if current == 0.0:
            if z_value <= -entry_z:
                current = 1.0
            elif z_value >= entry_z:
                current = -1.0
        elif current == 1.0 and z_value >= exit_z:
            current = 0.0
        elif current == -1.0 and z_value <= -exit_z:
            current = 0.0

        position.iloc[i] = current

    return position.fillna(0.0)


def backtest_pair_from_spread(
    spread: pd.Series,
    z_window: int,
    entry_z: float,
    exit_z: float,
    cost_per_turn: float,
    slippage_per_turn: float,
) -> Optional[BacktestResult]:
    """Run the spread backtest using yesterday's position for today's pnl."""
    spread = spread.replace([np.inf, -np.inf], np.nan).dropna()
    if len(spread) < z_window + 5 or spread.std() == 0:
        return None

    zscore = compute_zscore(spread, z_window)
    position = build_positions(zscore, entry_z=entry_z, exit_z=exit_z)
    spread_change = spread.diff().replace([np.inf, -np.inf], np.nan).fillna(0.0)

    gross_pnl = position.shift(1).fillna(0.0) * spread_change
    turns = position.diff().abs().fillna(position.abs())
    total_costs = turns * (cost_per_turn + slippage_per_turn)
    net_pnl = (gross_pnl - total_costs).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    net_pnl = net_pnl.clip(lower=-0.99, upper=0.99)

    metrics = compute_pair_metrics(net_pnl=net_pnl, turns=turns)
    if metrics is None:
        return None

    return BacktestResult(
        metrics=metrics,
        net_pnl=net_pnl,
        equity_curve=(1.0 + net_pnl).cumprod(),
        spread=spread,
        zscore=zscore,
        position=position,
        turns=turns,
    )


def generate_walk_forward_windows(
    dates: pd.Index,
    training_window_days: int,
    test_window_days: int,
    step_days: int,
    mode: str = WALK_FORWARD_VALIDATION_MODE,
) -> Iterable[Tuple[int, int, int, int]]:
    """Yield rolling walk-forward index windows."""
    total_required = training_window_days + test_window_days
    if len(dates) < total_required:
        return

    effective_step_days = test_window_days if mode == "non_overlapping" else step_days
    start_idx = 0
    while start_idx + total_required <= len(dates):
        train_end_idx = start_idx + training_window_days
        test_end_idx = train_end_idx + test_window_days
        yield start_idx, train_end_idx, train_end_idx, test_end_idx
        start_idx += effective_step_days


def evaluate_training_slice(
    train_df: pd.DataFrame,
    stock_x: str,
    stock_y: str,
) -> Tuple[bool, str, Dict[str, float]]:
    """Run training-slice statistical checks and return diagnostics."""
    train_log_x = np.log(train_df[stock_x])
    train_log_y = np.log(train_df[stock_y])
    train_returns = pd.concat([train_log_x.diff(), train_log_y.diff()], axis=1).dropna()
    train_returns.columns = ["x_ret", "y_ret"]

    if len(train_returns) < 30:
        diagnostics = {
            "train_corr": np.nan,
            "beta": np.nan,
            "coint_pvalue": np.nan,
            "adf_pvalue": np.nan,
            "half_life": np.nan,
        }
        return False, "insufficient training data", diagnostics

    train_corr = safe_float(train_returns["x_ret"].corr(train_returns["y_ret"]))
    if pd.isna(train_corr) or train_corr < MIN_CORR:
        diagnostics = {
            "train_corr": train_corr,
            "beta": np.nan,
            "coint_pvalue": np.nan,
            "adf_pvalue": np.nan,
            "half_life": np.nan,
        }
        return False, "failed correlation", diagnostics

    beta = fit_beta(train_log_x, train_log_y)
    if pd.isna(beta):
        diagnostics = {
            "train_corr": train_corr,
            "beta": np.nan,
            "coint_pvalue": np.nan,
            "adf_pvalue": np.nan,
            "half_life": np.nan,
        }
        return False, "failed beta fit", diagnostics

    train_spread = compute_spread(train_log_x, train_log_y, beta)
    coint_pvalue = calc_coint_pvalue(train_log_x, train_log_y)
    if pd.isna(coint_pvalue) or coint_pvalue > MAX_COINTEGRATION_P:
        diagnostics = {
            "train_corr": train_corr,
            "beta": beta,
            "coint_pvalue": coint_pvalue,
            "adf_pvalue": np.nan,
            "half_life": np.nan,
        }
        return False, "failed cointegration", diagnostics

    adf_pvalue = calc_adf_pvalue(train_spread)
    if pd.isna(adf_pvalue) or adf_pvalue > MAX_ADF_P:
        diagnostics = {
            "train_corr": train_corr,
            "beta": beta,
            "coint_pvalue": coint_pvalue,
            "adf_pvalue": adf_pvalue,
            "half_life": np.nan,
        }
        return False, "failed ADF", diagnostics

    half_life = estimate_half_life(train_spread)
    if pd.isna(half_life) or np.isinf(half_life) or half_life < MIN_HALFLIFE or half_life > MAX_HALFLIFE:
        diagnostics = {
            "train_corr": train_corr,
            "beta": beta,
            "coint_pvalue": coint_pvalue,
            "adf_pvalue": adf_pvalue,
            "half_life": half_life,
        }
        return False, "failed half-life", diagnostics

    diagnostics = {
        "train_corr": train_corr,
        "beta": beta,
        "coint_pvalue": coint_pvalue,
        "adf_pvalue": adf_pvalue,
        "half_life": half_life,
    }
    return True, "", diagnostics


def aggregate_out_of_sample_results(backtests: List[BacktestResult]) -> Optional[PairMetrics]:
    """Combine walk-forward test-window pnl series into one OOS result."""
    if not backtests:
        return None

    combined_pnl = pd.concat([bt.net_pnl for bt in backtests], axis=0, ignore_index=True)
    combined_turns = pd.concat([bt.turns for bt in backtests], axis=0, ignore_index=True)
    return compute_pair_metrics(net_pnl=combined_pnl, turns=combined_turns)


def compute_pair_score(
    oos_sharpe: float,
    oos_return: float,
    oos_max_drawdown: float,
    windows_passed: int,
    oos_trades: int,
) -> float:
    """Compute the explicit ranking score used for pair ordering."""
    capped_sharpe = -2.0 if pd.isna(oos_sharpe) else max(oos_sharpe, -2.0)
    sharpe_component = SCORE_SHARPE_WEIGHT * capped_sharpe
    return_component = SCORE_RETURN_WEIGHT * (0.0 if pd.isna(oos_return) else oos_return)
    drawdown_penalty = SCORE_DRAWDOWN_PENALTY * (0.0 if pd.isna(oos_max_drawdown) else abs(oos_max_drawdown))
    passing_windows_component = SCORE_PASSING_WINDOW_WEIGHT * windows_passed
    trade_component = SCORE_TRADE_WEIGHT * oos_trades
    return safe_float(
        sharpe_component
        + return_component
        - drawdown_penalty
        + passing_windows_component
        + trade_component
    )


def compute_score_components(
    oos_sharpe: float,
    oos_return: float,
    oos_max_drawdown: float,
    windows_passed: int,
    oos_trades: int,
) -> Dict[str, float]:
    """Return the explicit score components for easier inspection."""
    capped_sharpe = -2.0 if pd.isna(oos_sharpe) else max(oos_sharpe, -2.0)
    return {
        "score_sharpe_component": safe_float(SCORE_SHARPE_WEIGHT * capped_sharpe),
        "score_return_component": safe_float(
            SCORE_RETURN_WEIGHT * (0.0 if pd.isna(oos_return) else oos_return)
        ),
        "score_drawdown_penalty": safe_float(
            SCORE_DRAWDOWN_PENALTY * (0.0 if pd.isna(oos_max_drawdown) else abs(oos_max_drawdown))
        ),
        "score_windows_component": safe_float(SCORE_PASSING_WINDOW_WEIGHT * windows_passed),
        "score_trades_component": safe_float(SCORE_TRADE_WEIGHT * oos_trades),
    }


def compute_average_from_passed_windows(window_records: List[Dict[str, object]], column: str) -> float:
    """Compute the average of a metric across passed walk-forward windows only."""
    passed_values: List[float] = []
    for record in window_records:
        if not record.get("passed", False):
            continue
        value = safe_float(record.get(column))
        if np.isfinite(value):
            passed_values.append(value)

    if not passed_values:
        return np.nan

    return safe_float(np.mean(passed_values))


def compute_oos_day_counts(backtests: List[BacktestResult]) -> Tuple[int, int]:
    """Return accumulated and unique OOS test-day counts across passed windows."""
    if not backtests:
        return 0, 0

    accumulated_days = int(sum(len(bt.net_pnl) for bt in backtests))
    unique_days = int(len(pd.Index(np.concatenate([bt.net_pnl.index.values for bt in backtests])).unique()))
    return accumulated_days, unique_days


def determine_research_verdict(
    oos_sharpe: float,
    oos_return: float,
    oos_trades: int,
    windows_passed: int,
) -> str:
    """Assign a research verdict from aggregated out-of-sample results."""
    if (
        pd.notna(oos_sharpe)
        and oos_sharpe > 0.5
        and pd.notna(oos_return)
        and oos_return > 0
        and oos_trades >= QUALIFYING_MIN_OOS_TRADES
        and windows_passed >= QUALIFYING_MIN_WINDOWS_PASSED
    ):
        return "STRONG_CANDIDATE"

    if pd.notna(oos_sharpe) and oos_sharpe > 0 and pd.notna(oos_return) and oos_return > 0:
        return "WEAK_CANDIDATE"

    return "REJECT"


def verdict_sort_key(verdict: str) -> int:
    """Map research verdicts to a stable sort order."""
    order = {
        "STRONG_CANDIDATE": 0,
        "WEAK_CANDIDATE": 1,
        "REJECT": 2,
    }
    return order.get(verdict, 3)


def compute_confidence_score(
    windows_passed: int,
    oos_trades: int,
    oos_unique_test_days: int,
    avg_train_corr_passed: float,
    avg_coint_pvalue_passed: float,
    avg_adf_pvalue_passed: float,
    avg_half_life_passed: float,
) -> float:
    """Estimate how believable a pair is based on sample depth and structural stability."""
    windows_component = min(windows_passed / 4.0, 1.0) * 3.0
    trades_component = min(oos_trades / 8.0, 1.0) * 2.5
    unique_days_component = min(oos_unique_test_days / 252.0, 1.0) * 2.0

    corr_component = 0.0
    if pd.notna(avg_train_corr_passed):
        corr_component = max(min((avg_train_corr_passed - 0.65) / 0.20, 1.0), 0.0) * 1.0

    coint_component = 0.0
    if pd.notna(avg_coint_pvalue_passed):
        coint_component = max(min((0.10 - avg_coint_pvalue_passed) / 0.10, 1.0), 0.0) * 0.75

    adf_component = 0.0
    if pd.notna(avg_adf_pvalue_passed):
        adf_component = max(min((0.10 - avg_adf_pvalue_passed) / 0.10, 1.0), 0.0) * 0.75

    half_life_component = 0.0
    if pd.notna(avg_half_life_passed) and 3.0 <= avg_half_life_passed <= 40.0:
        center_distance = abs(avg_half_life_passed - 12.0)
        half_life_component = max(1.0 - center_distance / 28.0, 0.0) * 1.0

    return safe_float(
        windows_component
        + trades_component
        + unique_days_component
        + corr_component
        + coint_component
        + adf_component
        + half_life_component
    )


def confidence_rank_from_score(confidence_score: float) -> int:
    """Convert confidence score into an ordinal rank where 1 is best."""
    if confidence_score >= 8.0:
        return 1
    if confidence_score >= 6.5:
        return 2
    if confidence_score >= 5.0:
        return 3
    if confidence_score >= 3.5:
        return 4
    return 5


def robustness_scenarios(
    base_z_window: int,
    base_entry_z: float,
    base_exit_z: float,
    base_mode: str,
) -> List[Dict[str, object]]:
    """Return nearby parameter scenarios for robustness testing."""
    lower_z_window = max(10, base_z_window - ROBUSTNESS_Z_WINDOW_DELTA)
    return [
        {"name": "base", "z_window": base_z_window, "entry_z": base_entry_z, "exit_z": base_exit_z, "walk_forward_mode": base_mode},
        {"name": "entry_lo", "z_window": base_z_window, "entry_z": max(0.5, base_entry_z - ROBUSTNESS_ENTRY_Z_DELTA), "exit_z": base_exit_z, "walk_forward_mode": base_mode},
        {"name": "entry_hi", "z_window": base_z_window, "entry_z": base_entry_z + ROBUSTNESS_ENTRY_Z_DELTA, "exit_z": base_exit_z, "walk_forward_mode": base_mode},
        {"name": "z_window_lo", "z_window": lower_z_window, "entry_z": base_entry_z, "exit_z": base_exit_z, "walk_forward_mode": base_mode},
        {"name": "z_window_hi", "z_window": base_z_window + ROBUSTNESS_Z_WINDOW_DELTA, "entry_z": base_entry_z, "exit_z": base_exit_z, "walk_forward_mode": base_mode},
        {"name": "non_overlap", "z_window": base_z_window, "entry_z": base_entry_z, "exit_z": base_exit_z, "walk_forward_mode": "non_overlapping"},
    ]


def evaluate_robustness(
    sector: str,
    stock_x: str,
    stock_y: str,
    prices: pd.DataFrame,
    base_z_window: int,
    base_entry_z: float,
    base_exit_z: float,
    base_mode: str,
) -> Dict[str, float]:
    """Evaluate whether a pair survives nearby settings and non-overlapping validation."""
    scenarios = robustness_scenarios(
        base_z_window=base_z_window,
        base_entry_z=base_entry_z,
        base_exit_z=base_exit_z,
        base_mode=base_mode,
    )
    pass_count = 0

    for scenario in scenarios:
        scenario_result, _, _ = analyze_pair(
            sector=sector,
            stock_x=stock_x,
            stock_y=stock_y,
            prices=prices,
            z_window=int(scenario["z_window"]),
            entry_z=float(scenario["entry_z"]),
            exit_z=float(scenario["exit_z"]),
            walk_forward_mode=str(scenario["walk_forward_mode"]),
            compute_robustness=False,
        )
        if scenario_result is not None:
            pass_count += 1

    scenarios_tested = len(scenarios)
    pass_rate = safe_float(pass_count / scenarios_tested) if scenarios_tested > 0 else np.nan
    return {
        "robustness_score": safe_float(pass_rate * 10.0) if pd.notna(pass_rate) else np.nan,
        "robustness_pass_rate": pass_rate,
        "robustness_pass_count": int(pass_count),
        "robustness_scenarios_tested": int(scenarios_tested),
    }


def analyze_pair(
    sector: str,
    stock_x: str,
    stock_y: str,
    prices: pd.DataFrame,
    z_window: int = Z_WINDOW,
    entry_z: float = ENTRY_Z,
    exit_z: float = EXIT_Z,
    walk_forward_mode: str = WALK_FORWARD_VALIDATION_MODE,
    compute_robustness: bool = True,
) -> Tuple[Optional[PairResult], Dict[str, object], List[Dict[str, object]]]:
    """Analyze a pair with rolling walk-forward evaluation."""
    pair_name = f"{stock_x} vs {stock_y}"
    pair_df = prices[[stock_x, stock_y]].copy()

    missing_ratio_x = pair_df[stock_x].isna().mean()
    missing_ratio_y = pair_df[stock_y].isna().mean()
    latest_diag = {
        "train_corr": np.nan,
        "beta": np.nan,
        "coint_pvalue": np.nan,
        "adf_pvalue": np.nan,
        "half_life": np.nan,
    }
    window_records: List[Dict[str, object]] = []

    if max(missing_ratio_x, missing_ratio_y) > MAX_MISSING_RATIO:
        diagnostics = {
            "sector": sector,
            "stock_x": stock_x,
            "stock_y": stock_y,
            "pair": pair_name,
            "fail_reason": "too much missing data",
            "avg_train_corr_passed": np.nan,
            "avg_coint_pvalue_passed": np.nan,
            "avg_adf_pvalue_passed": np.nan,
            "avg_half_life_passed": np.nan,
            "latest_train_corr": np.nan,
            "latest_coint_pvalue": np.nan,
            "latest_adf_pvalue": np.nan,
            "latest_half_life": np.nan,
            "windows_tested": 0,
            "windows_passed": 0,
        }
        return None, diagnostics, window_records

    pair_df = pair_df.dropna()
    if len(pair_df) < MIN_ROWS_PER_PAIR:
        diagnostics = {
            "sector": sector,
            "stock_x": stock_x,
            "stock_y": stock_y,
            "pair": pair_name,
            "fail_reason": "insufficient training data",
            "avg_train_corr_passed": np.nan,
            "avg_coint_pvalue_passed": np.nan,
            "avg_adf_pvalue_passed": np.nan,
            "avg_half_life_passed": np.nan,
            "latest_train_corr": np.nan,
            "latest_coint_pvalue": np.nan,
            "latest_adf_pvalue": np.nan,
            "latest_half_life": np.nan,
            "windows_tested": 0,
            "windows_passed": 0,
        }
        return None, diagnostics, window_records

    oos_backtests: List[BacktestResult] = []
    windows_tested = 0
    windows_passed = 0
    last_fail_reason = "no valid walk-forward windows"

    for train_start, train_end, test_start, test_end in generate_walk_forward_windows(
        dates=pair_df.index,
        training_window_days=TRAINING_WINDOW_DAYS,
        test_window_days=TEST_WINDOW_DAYS,
        step_days=WALK_FORWARD_STEP_DAYS,
        mode=walk_forward_mode,
    ):
        train_df = pair_df.iloc[train_start:train_end].copy()
        test_df = pair_df.iloc[test_start:test_end].copy()

        if len(train_df) < TRAINING_WINDOW_DAYS:
            last_fail_reason = "insufficient training data"
            continue
        if len(test_df) < TEST_WINDOW_DAYS:
            last_fail_reason = "insufficient test data"
            continue

        windows_tested += 1
        passed, fail_reason, latest_diag = evaluate_training_slice(train_df, stock_x, stock_y)

        window_record = {
            "sector": sector,
            "stock_x": stock_x,
            "stock_y": stock_y,
            "pair": pair_name,
            "train_start": train_df.index[0].date().isoformat(),
            "train_end": train_df.index[-1].date().isoformat(),
            "test_start": test_df.index[0].date().isoformat(),
            "test_end": test_df.index[-1].date().isoformat(),
            "train_corr": safe_float(latest_diag["train_corr"]),
            "beta": safe_float(latest_diag["beta"]),
            "coint_pvalue": safe_float(latest_diag["coint_pvalue"]),
            "adf_pvalue": safe_float(latest_diag["adf_pvalue"]),
            "half_life": safe_float(latest_diag["half_life"]),
            "test_sharpe": np.nan,
            "test_return": np.nan,
            "test_max_drawdown": np.nan,
            "test_trades": 0,
            "passed": passed,
            "fail_reason": fail_reason,
        }

        if not passed:
            last_fail_reason = fail_reason
            window_records.append(window_record)
            continue

        beta = safe_float(latest_diag["beta"])
        test_log_x = np.log(test_df[stock_x])
        test_log_y = np.log(test_df[stock_y])
        test_spread = compute_spread(test_log_x, test_log_y, beta)

        backtest = backtest_pair_from_spread(
            spread=test_spread,
            z_window=z_window,
            entry_z=entry_z,
            exit_z=exit_z,
            cost_per_turn=COST_PER_TURN,
            slippage_per_turn=SLIPPAGE_PER_TURN,
        )

        if backtest is None:
            window_record["passed"] = False
            window_record["fail_reason"] = "insufficient test trades"
            last_fail_reason = "insufficient test trades"
            window_records.append(window_record)
            continue

        windows_passed += 1
        window_record["test_sharpe"] = backtest.metrics.sharpe
        window_record["test_return"] = backtest.metrics.total_return
        window_record["test_max_drawdown"] = backtest.metrics.max_drawdown
        window_record["test_trades"] = backtest.metrics.trades
        window_record["fail_reason"] = ""
        window_records.append(window_record)
        oos_backtests.append(backtest)

    avg_train_corr_passed = compute_average_from_passed_windows(window_records, "train_corr")
    avg_coint_pvalue_passed = compute_average_from_passed_windows(window_records, "coint_pvalue")
    avg_adf_pvalue_passed = compute_average_from_passed_windows(window_records, "adf_pvalue")
    avg_half_life_passed = compute_average_from_passed_windows(window_records, "half_life")

    if windows_tested == 0:
        diagnostics = {
            "sector": sector,
            "stock_x": stock_x,
            "stock_y": stock_y,
            "pair": pair_name,
            "fail_reason": "no valid walk-forward windows",
            "avg_train_corr_passed": np.nan,
            "avg_coint_pvalue_passed": np.nan,
            "avg_adf_pvalue_passed": np.nan,
            "avg_half_life_passed": np.nan,
            "latest_train_corr": np.nan,
            "latest_coint_pvalue": np.nan,
            "latest_adf_pvalue": np.nan,
            "latest_half_life": np.nan,
            "windows_tested": 0,
            "windows_passed": 0,
        }
        return None, diagnostics, window_records

    aggregate_metrics = aggregate_out_of_sample_results(oos_backtests)
    oos_accumulated_test_days, oos_unique_test_days = compute_oos_day_counts(oos_backtests)
    if aggregate_metrics is None:
        diagnostics = {
            "sector": sector,
            "stock_x": stock_x,
            "stock_y": stock_y,
            "pair": pair_name,
            "fail_reason": last_fail_reason,
            "avg_train_corr_passed": avg_train_corr_passed,
            "avg_coint_pvalue_passed": avg_coint_pvalue_passed,
            "avg_adf_pvalue_passed": avg_adf_pvalue_passed,
            "avg_half_life_passed": avg_half_life_passed,
            "latest_train_corr": safe_float(latest_diag["train_corr"]),
            "latest_coint_pvalue": safe_float(latest_diag["coint_pvalue"]),
            "latest_adf_pvalue": safe_float(latest_diag["adf_pvalue"]),
            "latest_half_life": safe_float(latest_diag["half_life"]),
            "windows_tested": windows_tested,
            "windows_passed": windows_passed,
        }
        return None, diagnostics, window_records

    if windows_passed < MIN_PASSING_WINDOWS:
        diagnostics = {
            "sector": sector,
            "stock_x": stock_x,
            "stock_y": stock_y,
            "pair": pair_name,
            "fail_reason": "not enough passing windows",
            "avg_train_corr_passed": avg_train_corr_passed,
            "avg_coint_pvalue_passed": avg_coint_pvalue_passed,
            "avg_adf_pvalue_passed": avg_adf_pvalue_passed,
            "avg_half_life_passed": avg_half_life_passed,
            "latest_train_corr": safe_float(latest_diag["train_corr"]),
            "latest_coint_pvalue": safe_float(latest_diag["coint_pvalue"]),
            "latest_adf_pvalue": safe_float(latest_diag["adf_pvalue"]),
            "latest_half_life": safe_float(latest_diag["half_life"]),
            "windows_tested": windows_tested,
            "windows_passed": windows_passed,
        }
        return None, diagnostics, window_records

    if aggregate_metrics.trades < MIN_TOTAL_OOS_TRADES:
        diagnostics = {
            "sector": sector,
            "stock_x": stock_x,
            "stock_y": stock_y,
            "pair": pair_name,
            "fail_reason": "insufficient test trades",
            "avg_train_corr_passed": avg_train_corr_passed,
            "avg_coint_pvalue_passed": avg_coint_pvalue_passed,
            "avg_adf_pvalue_passed": avg_adf_pvalue_passed,
            "avg_half_life_passed": avg_half_life_passed,
            "latest_train_corr": safe_float(latest_diag["train_corr"]),
            "latest_coint_pvalue": safe_float(latest_diag["coint_pvalue"]),
            "latest_adf_pvalue": safe_float(latest_diag["adf_pvalue"]),
            "latest_half_life": safe_float(latest_diag["half_life"]),
            "windows_tested": windows_tested,
            "windows_passed": windows_passed,
        }
        return None, diagnostics, window_records

    if USE_STRUCTURAL_PASSED_WINDOW_FILTER:
        if pd.isna(avg_coint_pvalue_passed) or avg_coint_pvalue_passed >= MAX_AVG_COINT_PVALUE_PASSED:
            diagnostics = {
                "sector": sector,
                "stock_x": stock_x,
                "stock_y": stock_y,
                "pair": pair_name,
                "fail_reason": "failed avg cointegration",
                "avg_train_corr_passed": avg_train_corr_passed,
                "avg_coint_pvalue_passed": avg_coint_pvalue_passed,
                "avg_adf_pvalue_passed": avg_adf_pvalue_passed,
                "avg_half_life_passed": avg_half_life_passed,
                "latest_train_corr": safe_float(latest_diag["train_corr"]),
                "latest_coint_pvalue": safe_float(latest_diag["coint_pvalue"]),
                "latest_adf_pvalue": safe_float(latest_diag["adf_pvalue"]),
                "latest_half_life": safe_float(latest_diag["half_life"]),
                "windows_tested": windows_tested,
                "windows_passed": windows_passed,
            }
            return None, diagnostics, window_records

        if pd.isna(avg_adf_pvalue_passed) or avg_adf_pvalue_passed >= MAX_AVG_ADF_PVALUE_PASSED:
            diagnostics = {
                "sector": sector,
                "stock_x": stock_x,
                "stock_y": stock_y,
                "pair": pair_name,
                "fail_reason": "failed avg ADF",
                "avg_train_corr_passed": avg_train_corr_passed,
                "avg_coint_pvalue_passed": avg_coint_pvalue_passed,
                "avg_adf_pvalue_passed": avg_adf_pvalue_passed,
                "avg_half_life_passed": avg_half_life_passed,
                "latest_train_corr": safe_float(latest_diag["train_corr"]),
                "latest_coint_pvalue": safe_float(latest_diag["coint_pvalue"]),
                "latest_adf_pvalue": safe_float(latest_diag["adf_pvalue"]),
                "latest_half_life": safe_float(latest_diag["half_life"]),
                "windows_tested": windows_tested,
                "windows_passed": windows_passed,
            }
            return None, diagnostics, window_records

        if (
            pd.isna(avg_half_life_passed)
            or avg_half_life_passed < MIN_AVG_HALFLIFE_PASSED
            or avg_half_life_passed > MAX_AVG_HALFLIFE_PASSED
        ):
            diagnostics = {
                "sector": sector,
                "stock_x": stock_x,
                "stock_y": stock_y,
                "pair": pair_name,
                "fail_reason": "failed avg half-life",
                "avg_train_corr_passed": avg_train_corr_passed,
                "avg_coint_pvalue_passed": avg_coint_pvalue_passed,
                "avg_adf_pvalue_passed": avg_adf_pvalue_passed,
                "avg_half_life_passed": avg_half_life_passed,
                "latest_train_corr": safe_float(latest_diag["train_corr"]),
                "latest_coint_pvalue": safe_float(latest_diag["coint_pvalue"]),
                "latest_adf_pvalue": safe_float(latest_diag["adf_pvalue"]),
                "latest_half_life": safe_float(latest_diag["half_life"]),
                "windows_tested": windows_tested,
                "windows_passed": windows_passed,
            }
            return None, diagnostics, window_records

    research_verdict = determine_research_verdict(
        oos_sharpe=aggregate_metrics.sharpe,
        oos_return=aggregate_metrics.total_return,
        oos_trades=aggregate_metrics.trades,
        windows_passed=windows_passed,
    )
    confidence_score = compute_confidence_score(
        windows_passed=windows_passed,
        oos_trades=aggregate_metrics.trades,
        oos_unique_test_days=oos_unique_test_days,
        avg_train_corr_passed=avg_train_corr_passed,
        avg_coint_pvalue_passed=avg_coint_pvalue_passed,
        avg_adf_pvalue_passed=avg_adf_pvalue_passed,
        avg_half_life_passed=avg_half_life_passed,
    )
    confidence_rank = confidence_rank_from_score(confidence_score)
    score_components = compute_score_components(
        oos_sharpe=aggregate_metrics.sharpe,
        oos_return=aggregate_metrics.total_return,
        oos_max_drawdown=aggregate_metrics.max_drawdown,
        windows_passed=windows_passed,
        oos_trades=aggregate_metrics.trades,
    )
    robustness_metrics = {
        "robustness_score": np.nan,
        "robustness_pass_rate": np.nan,
        "robustness_pass_count": 0,
        "robustness_scenarios_tested": 0,
    }
    if compute_robustness:
        robustness_metrics = evaluate_robustness(
            sector=sector,
            stock_x=stock_x,
            stock_y=stock_y,
            prices=prices,
            base_z_window=z_window,
            base_entry_z=entry_z,
            base_exit_z=exit_z,
            base_mode=walk_forward_mode,
        )

    if (
        aggregate_metrics.trades < QUALIFYING_MIN_OOS_TRADES
        or windows_passed < QUALIFYING_MIN_WINDOWS_PASSED
        or pd.isna(aggregate_metrics.sharpe)
        or aggregate_metrics.sharpe <= QUALIFYING_MIN_OOS_SHARPE
        or pd.isna(aggregate_metrics.total_return)
        or aggregate_metrics.total_return <= QUALIFYING_MIN_OOS_RETURN
    ):
        diagnostics = {
            "sector": sector,
            "stock_x": stock_x,
            "stock_y": stock_y,
            "pair": pair_name,
            "fail_reason": "failed qualifying filter",
            "avg_train_corr_passed": avg_train_corr_passed,
            "avg_coint_pvalue_passed": avg_coint_pvalue_passed,
            "avg_adf_pvalue_passed": avg_adf_pvalue_passed,
            "avg_half_life_passed": avg_half_life_passed,
            "latest_train_corr": safe_float(latest_diag["train_corr"]),
            "latest_coint_pvalue": safe_float(latest_diag["coint_pvalue"]),
            "latest_adf_pvalue": safe_float(latest_diag["adf_pvalue"]),
            "latest_half_life": safe_float(latest_diag["half_life"]),
            "windows_tested": windows_tested,
            "windows_passed": windows_passed,
            "research_verdict": research_verdict,
            "oos_sharpe": aggregate_metrics.sharpe,
            "oos_return": aggregate_metrics.total_return,
            "oos_annualized_return": aggregate_metrics.annualized_return,
            "oos_max_drawdown": aggregate_metrics.max_drawdown,
            "oos_trades": aggregate_metrics.trades,
            "oos_return_per_trade": aggregate_metrics.return_per_trade,
            "oos_accumulated_test_days": oos_accumulated_test_days,
            "oos_unique_test_days": oos_unique_test_days,
        }
        return None, diagnostics, window_records

    result = PairResult(
        sector=sector,
        stock_x=stock_x,
        stock_y=stock_y,
        pair=pair_name,
        windows_tested=windows_tested,
        windows_passed=windows_passed,
        avg_train_corr_passed=avg_train_corr_passed,
        avg_coint_pvalue_passed=avg_coint_pvalue_passed,
        avg_adf_pvalue_passed=avg_adf_pvalue_passed,
        avg_half_life_passed=avg_half_life_passed,
        latest_train_corr=safe_float(latest_diag["train_corr"]),
        latest_beta=safe_float(latest_diag["beta"]),
        latest_coint_pvalue=safe_float(latest_diag["coint_pvalue"]),
        latest_adf_pvalue=safe_float(latest_diag["adf_pvalue"]),
        latest_half_life=safe_float(latest_diag["half_life"]),
        oos_sharpe=aggregate_metrics.sharpe,
        oos_return=aggregate_metrics.total_return,
        oos_annualized_return=annualize_return(
            total_return=aggregate_metrics.total_return,
            periods=oos_unique_test_days,
        ),
        oos_max_drawdown=aggregate_metrics.max_drawdown,
        oos_trades=aggregate_metrics.trades,
        oos_return_per_trade=aggregate_metrics.return_per_trade,
        oos_accumulated_test_days=oos_accumulated_test_days,
        oos_unique_test_days=oos_unique_test_days,
        research_verdict=research_verdict,
        research_recommendation=determine_research_recommendation(
            research_verdict=research_verdict,
            confidence_score=confidence_score,
            robustness_score=safe_float(robustness_metrics["robustness_score"]),
        ),
        walk_forward_mode=walk_forward_mode,
        robustness_score=safe_float(robustness_metrics["robustness_score"]),
        robustness_pass_rate=safe_float(robustness_metrics["robustness_pass_rate"]),
        robustness_pass_count=int(robustness_metrics["robustness_pass_count"]),
        robustness_scenarios_tested=int(robustness_metrics["robustness_scenarios_tested"]),
        confidence_score=confidence_score,
        confidence_rank=confidence_rank,
        score_sharpe_component=score_components["score_sharpe_component"],
        score_return_component=score_components["score_return_component"],
        score_drawdown_penalty=score_components["score_drawdown_penalty"],
        score_windows_component=score_components["score_windows_component"],
        score_trades_component=score_components["score_trades_component"],
        score=compute_pair_score(
            oos_sharpe=aggregate_metrics.sharpe,
            oos_return=aggregate_metrics.total_return,
            oos_max_drawdown=aggregate_metrics.max_drawdown,
            windows_passed=windows_passed,
            oos_trades=aggregate_metrics.trades,
        ),
    )

    diagnostics = {
        "sector": sector,
        "stock_x": stock_x,
        "stock_y": stock_y,
        "pair": pair_name,
        "fail_reason": "",
        "avg_train_corr_passed": avg_train_corr_passed,
        "avg_coint_pvalue_passed": avg_coint_pvalue_passed,
        "avg_adf_pvalue_passed": avg_adf_pvalue_passed,
        "avg_half_life_passed": avg_half_life_passed,
        "latest_train_corr": safe_float(latest_diag["train_corr"]),
        "latest_coint_pvalue": safe_float(latest_diag["coint_pvalue"]),
        "latest_adf_pvalue": safe_float(latest_diag["adf_pvalue"]),
        "latest_half_life": safe_float(latest_diag["half_life"]),
        "windows_tested": windows_tested,
        "windows_passed": windows_passed,
        "research_verdict": research_verdict,
        "oos_sharpe": aggregate_metrics.sharpe,
        "oos_return": aggregate_metrics.total_return,
        "oos_annualized_return": annualize_return(
            total_return=aggregate_metrics.total_return,
            periods=oos_unique_test_days,
        ),
        "oos_max_drawdown": aggregate_metrics.max_drawdown,
        "oos_trades": aggregate_metrics.trades,
        "oos_return_per_trade": aggregate_metrics.return_per_trade,
        "oos_accumulated_test_days": oos_accumulated_test_days,
        "oos_unique_test_days": oos_unique_test_days,
        "confidence_score": confidence_score,
        "confidence_rank": confidence_rank,
    }
    return result, diagnostics, window_records


def analyze_universe(
    universe: Dict[str, List[str]],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Analyze every sector pair and return ranked results plus diagnostics."""
    all_tickers = flatten_universe(universe)
    prices = download_prices(all_tickers, START_DATE, resolve_end_date())

    if prices.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), prices

    results: List[Dict[str, object]] = []
    diagnostics: List[Dict[str, object]] = []
    window_records: List[Dict[str, object]] = []

    for sector, tickers in universe.items():
        available = [ticker for ticker in tickers if ticker in prices.columns]
        if len(available) < 2:
            continue

        for stock_x, stock_y in itertools.combinations(available, 2):
            result, pair_diag, pair_windows = analyze_pair(
                sector=sector,
                stock_x=stock_x,
                stock_y=stock_y,
                prices=prices,
            )
            diagnostics.append(pair_diag)
            window_records.extend(pair_windows)
            if result is not None:
                results.append(asdict(result))

    ranked_df = pd.DataFrame(results)
    diagnostics_df = pd.DataFrame(diagnostics)
    window_df = pd.DataFrame(window_records)

    if not ranked_df.empty:
        ranked_df["verdict_rank"] = ranked_df["research_verdict"].map(verdict_sort_key)
        ranked_df = ranked_df.sort_values(
            by=["verdict_rank", "score", "oos_sharpe", "oos_return"],
            ascending=[True, False, False, False],
        ).reset_index(drop=True)
        ranked_df = ranked_df.drop(columns=["verdict_rank"])

    return ranked_df, diagnostics_df, window_df, prices


def format_float_columns(df: pd.DataFrame, columns: List[str], decimals: int = 4) -> pd.DataFrame:
    """Round selected float columns for console display."""
    display_df = df.copy()
    for column in columns:
        if column in display_df.columns:
            display_df[column] = display_df[column].astype(float).round(decimals)
    return display_df


def print_top_ranked_pairs(ranked_df: pd.DataFrame, top_n: int) -> None:
    """Print the top qualifying pairs."""
    if ranked_df.empty:
        print("No qualifying pairs found.")
        return

    display_columns = [
        "sector",
        "pair",
        "research_verdict",
        "research_recommendation",
        "walk_forward_mode",
        "robustness_score",
        "robustness_pass_rate",
        "confidence_rank",
        "confidence_score",
        "windows_passed",
        "windows_tested",
        "oos_accumulated_test_days",
        "oos_unique_test_days",
        "avg_train_corr_passed",
        "avg_coint_pvalue_passed",
        "avg_adf_pvalue_passed",
        "avg_half_life_passed",
        "oos_sharpe",
        "oos_return",
        "oos_annualized_return",
        "oos_max_drawdown",
        "oos_trades",
        "oos_return_per_trade",
        "score",
    ]
    display_df = format_float_columns(
        ranked_df[display_columns].copy(),
        columns=[
            "avg_train_corr_passed",
            "avg_coint_pvalue_passed",
            "avg_adf_pvalue_passed",
            "avg_half_life_passed",
            "oos_sharpe",
            "oos_return",
            "oos_annualized_return",
            "oos_max_drawdown",
            "oos_return_per_trade",
            "robustness_score",
            "robustness_pass_rate",
            "confidence_score",
            "score",
        ],
    )

    print("\nTop ranked pairs:\n")
    print(display_df.head(top_n).to_string(index=False))


def print_sector_summary(ranked_df: pd.DataFrame) -> None:
    """Print a compact sector-level summary."""
    if ranked_df.empty:
        print("\nSector summary:\n")
        print("No qualifying pairs found.")
        return

    summary_rows: List[Dict[str, object]] = []
    for sector, group in ranked_df.groupby("sector", sort=True):
        best_row = group.iloc[0]
        summary_rows.append(
            {
                "sector": sector,
                "qualifying_pairs": int(len(group)),
                "best_score": safe_float(best_row["score"]),
                "best_pair": best_row["pair"],
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df = format_float_columns(summary_df, columns=["best_score"])

    print("\nSector summary:\n")
    print(summary_df.to_string(index=False))


def sanitize_filename(value: str) -> str:
    """Make a filesystem-safe filename from a pair name."""
    safe = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in value)
    return safe.strip("_")


def clear_plot_dir(plot_dir: Path) -> None:
    """Remove stale plot files so the folder reflects the current run only."""
    plot_dir.mkdir(parents=True, exist_ok=True)
    for png_file in plot_dir.glob("*.png"):
        png_file.unlink(missing_ok=True)


def plot_pair_diagnostics(
    ranked_df: pd.DataFrame,
    prices: pd.DataFrame,
    top_n: int,
    plot_dir: Path,
) -> None:
    """Save spread, z-score, and price plots for the top-ranked pairs."""
    if ranked_df.empty or prices.empty:
        return

    plot_dir.mkdir(parents=True, exist_ok=True)

    for _, row in ranked_df.head(top_n).iterrows():
        stock_x = row["stock_x"]
        stock_y = row["stock_y"]
        beta = row["latest_beta"]
        pair_prices = prices[[stock_x, stock_y]].dropna()

        if pair_prices.empty or pd.isna(beta):
            continue

        log_x = np.log(pair_prices[stock_x])
        log_y = np.log(pair_prices[stock_y])
        spread = compute_spread(log_x, log_y, beta)
        zscore = compute_zscore(spread, Z_WINDOW)

        figure, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        axes[0].plot(spread.index, spread.values, color="navy", linewidth=1.2)
        axes[0].set_title(f"{row['pair']} Spread")
        axes[0].set_ylabel("Spread")
        axes[0].grid(alpha=0.3)

        axes[1].plot(zscore.index, zscore.values, color="darkgreen", linewidth=1.2)
        axes[1].axhline(ENTRY_Z, color="firebrick", linestyle="--", linewidth=1.0, label="Entry")
        axes[1].axhline(-ENTRY_Z, color="firebrick", linestyle="--", linewidth=1.0)
        axes[1].axhline(EXIT_Z, color="gray", linestyle=":", linewidth=1.0, label="Exit")
        axes[1].axhline(-EXIT_Z, color="gray", linestyle=":", linewidth=1.0)
        axes[1].set_title(f"{row['pair']} Z-Score")
        axes[1].set_ylabel("Z-Score")
        axes[1].grid(alpha=0.3)
        axes[1].legend(loc="upper left")

        normalized_prices = pair_prices / pair_prices.iloc[0]
        axes[2].plot(normalized_prices.index, normalized_prices[stock_x], label=stock_x, linewidth=1.2)
        axes[2].plot(normalized_prices.index, normalized_prices[stock_y], label=stock_y, linewidth=1.2)
        axes[2].set_title(f"{row['pair']} Normalized Prices")
        axes[2].set_ylabel("Normalized")
        axes[2].grid(alpha=0.3)
        axes[2].legend(loc="upper left")

        figure.tight_layout()
        output_path = plot_dir / f"{sanitize_filename(row['pair'])}.png"
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(figure)


def latest_live_signal_action(current_position: float, latest_zscore: float, entry_z: float) -> str:
    """Map the latest z-score and position state into a paper-trading action label."""
    if current_position > 0:
        return "LONG_SPREAD"
    if current_position < 0:
        return "SHORT_SPREAD"
    if pd.notna(latest_zscore) and latest_zscore <= -0.8 * entry_z:
        return "WATCH_LONG"
    if pd.notna(latest_zscore) and latest_zscore >= 0.8 * entry_z:
        return "WATCH_SHORT"
    return "FLAT"


def determine_operational_action(
    research_verdict: str,
    confidence_score: float,
    robustness_score: float,
    live_action: str,
    passes_live_stability: bool,
) -> str:
    """Convert research quality and current live context into an operational label."""
    actionable_signal = live_action in {"LONG_SPREAD", "SHORT_SPREAD"}
    watch_signal = live_action in {"WATCH_LONG", "WATCH_SHORT", "FLAT"}

    if (
        research_verdict == "STRONG_CANDIDATE"
        and pd.notna(confidence_score)
        and confidence_score >= 6.5
        and pd.notna(robustness_score)
        and robustness_score >= 6.0
        and actionable_signal
        and passes_live_stability
    ):
        return "PAPER_TRADE_READY"

    if (
        research_verdict in {"STRONG_CANDIDATE", "WEAK_CANDIDATE"}
        and pd.notna(confidence_score)
        and confidence_score >= 5.0
        and passes_live_stability
        and watch_signal
    ):
        return "WATCHLIST"

    return "AVOID"


def determine_research_recommendation(
    research_verdict: str,
    confidence_score: float,
    robustness_score: float,
) -> str:
    """Assign a research-layer recommendation before considering live signal state."""
    if (
        research_verdict == "STRONG_CANDIDATE"
        and pd.notna(confidence_score)
        and confidence_score >= 6.5
        and pd.notna(robustness_score)
        and robustness_score >= 6.0
    ):
        return "PAPER_TRADE_READY"

    if research_verdict in {"STRONG_CANDIDATE", "WEAK_CANDIDATE"}:
        return "WATCHLIST"

    return "AVOID"


def compute_live_stability_metrics(
    prices: pd.DataFrame,
    stock_x: str,
    stock_y: str,
) -> Dict[str, object]:
    """Compute recent beta/correlation/volatility stability checks for live approval."""
    live_df = prices[[stock_x, stock_y]].dropna().copy()
    if len(live_df) < max(LIVE_STABILITY_LOOKBACK, LIVE_BETA_WINDOW, LIVE_ROLLING_CORR_WINDOW) + 5:
        return {
            "recent_corr_mean": np.nan,
            "recent_corr_std": np.nan,
            "recent_beta_std": np.nan,
            "recent_spread_vol_ratio": np.nan,
            "passes_live_stability": False,
            "live_stability_reason": "INSUFFICIENT_LIVE_HISTORY",
        }

    log_x = np.log(live_df[stock_x])
    log_y = np.log(live_df[stock_y])
    ret_x = log_x.diff()
    ret_y = log_y.diff()
    rolling_corr = ret_x.rolling(LIVE_ROLLING_CORR_WINDOW).corr(ret_y)
    recent_corr = rolling_corr.tail(LIVE_STABILITY_LOOKBACK).dropna()

    beta_values: List[float] = []
    beta_index: List[pd.Timestamp] = []
    for end_idx in range(LIVE_BETA_WINDOW, len(live_df) + 1):
        window_log_x = log_x.iloc[end_idx - LIVE_BETA_WINDOW:end_idx]
        window_log_y = log_y.iloc[end_idx - LIVE_BETA_WINDOW:end_idx]
        beta_values.append(fit_beta(window_log_x, window_log_y))
        beta_index.append(live_df.index[end_idx - 1])
    rolling_beta = pd.Series(beta_values, index=beta_index)
    recent_beta_std = safe_float(rolling_beta.tail(LIVE_STABILITY_LOOKBACK).std())

    full_beta = fit_beta(log_x.tail(LIVE_BETA_WINDOW), log_y.tail(LIVE_BETA_WINDOW))
    spread = compute_spread(log_x, log_y, full_beta) if pd.notna(full_beta) else pd.Series(dtype=float)
    recent_spread_vol = safe_float(spread.tail(LIVE_SPREAD_VOL_WINDOW).std()) if not spread.empty else np.nan
    historical_spread_vol = safe_float(spread.std()) if not spread.empty else np.nan
    spread_vol_ratio = np.nan
    if pd.notna(recent_spread_vol) and pd.notna(historical_spread_vol) and historical_spread_vol > 0:
        spread_vol_ratio = safe_float(recent_spread_vol / historical_spread_vol)

    corr_mean = safe_float(recent_corr.mean()) if not recent_corr.empty else np.nan
    corr_std = safe_float(recent_corr.std()) if not recent_corr.empty else np.nan

    reasons: List[str] = []
    if pd.isna(corr_mean) or corr_mean < LIVE_MIN_RECENT_CORR_MEAN:
        reasons.append("LOW_RECENT_CORR")
    if pd.isna(corr_std) or corr_std > LIVE_MAX_RECENT_CORR_STD:
        reasons.append("UNSTABLE_CORR")
    if pd.isna(recent_beta_std) or recent_beta_std > LIVE_MAX_RECENT_BETA_STD:
        reasons.append("UNSTABLE_BETA")
    if pd.isna(spread_vol_ratio) or spread_vol_ratio > LIVE_MAX_RECENT_SPREAD_VOL_RATIO:
        reasons.append("ELEVATED_SPREAD_VOL")

    return {
        "recent_corr_mean": corr_mean,
        "recent_corr_std": corr_std,
        "recent_beta_std": recent_beta_std,
        "recent_spread_vol_ratio": spread_vol_ratio,
        "passes_live_stability": len(reasons) == 0,
        "live_stability_reason": "|".join(reasons) if reasons else "",
    }


def build_live_signal_row(
    pair_row: pd.Series,
    prices: pd.DataFrame,
    z_window: int = Z_WINDOW,
    entry_z: float = ENTRY_Z,
    exit_z: float = EXIT_Z,
) -> Optional[Dict[str, object]]:
    """Build a latest live-signal snapshot for a ranked pair."""
    stock_x = str(pair_row["stock_x"])
    stock_y = str(pair_row["stock_y"])
    pair_prices = prices[[stock_x, stock_y]].dropna()
    if len(pair_prices) < TRAINING_WINDOW_DAYS + z_window:
        return None
    live_stability = compute_live_stability_metrics(pair_prices, stock_x, stock_y)

    train_df = pair_prices.iloc[-TRAINING_WINDOW_DAYS:].copy()
    live_df = pair_prices.iloc[-(TRAINING_WINDOW_DAYS + z_window + 10):].copy()

    train_log_x = np.log(train_df[stock_x])
    train_log_y = np.log(train_df[stock_y])
    beta = fit_beta(train_log_x, train_log_y)
    if pd.isna(beta):
        return None

    live_log_x = np.log(live_df[stock_x])
    live_log_y = np.log(live_df[stock_y])
    spread = compute_spread(live_log_x, live_log_y, beta)
    zscore = compute_zscore(spread, z_window)
    position = build_positions(zscore, entry_z=entry_z, exit_z=exit_z)
    half_life = estimate_half_life(compute_spread(train_log_x, train_log_y, beta))

    latest_date = live_df.index[-1]
    latest_zscore = safe_float(zscore.iloc[-1]) if not zscore.empty else np.nan
    current_position = safe_float(position.iloc[-1]) if not position.empty else 0.0
    current_action = latest_live_signal_action(current_position, latest_zscore, entry_z)

    return {
        "sector": str(pair_row["sector"]),
        "pair": str(pair_row["pair"]),
        "latest_date": latest_date.date().isoformat(),
        "live_beta": safe_float(beta),
        "live_half_life": safe_float(half_life),
        "live_zscore": latest_zscore,
        "live_spread": safe_float(spread.iloc[-1]),
        "current_position": int(current_position),
        "current_action": current_action,
        "live_recommendation": determine_operational_action(
            research_verdict=str(pair_row["research_verdict"]),
            confidence_score=safe_float(pair_row["confidence_score"]),
            robustness_score=safe_float(pair_row["robustness_score"]),
            live_action=current_action,
            passes_live_stability=bool(live_stability["passes_live_stability"]),
        ),
        "passes_live_stability": bool(live_stability["passes_live_stability"]),
        "live_stability_reason": str(live_stability["live_stability_reason"]),
        "recent_corr_mean": safe_float(live_stability["recent_corr_mean"]),
        "recent_corr_std": safe_float(live_stability["recent_corr_std"]),
        "recent_beta_std": safe_float(live_stability["recent_beta_std"]),
        "recent_spread_vol_ratio": safe_float(live_stability["recent_spread_vol_ratio"]),
        "latest_price_x": safe_float(live_df[stock_x].iloc[-1]),
        "latest_price_y": safe_float(live_df[stock_y].iloc[-1]),
    }


def build_live_signals(ranked_df: pd.DataFrame, prices: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Build live-signal rows for the top ranked pairs."""
    if ranked_df.empty or prices.empty:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    for _, row in ranked_df.head(top_n).iterrows():
        live_row = build_live_signal_row(row, prices)
        if live_row is not None:
            rows.append(live_row)

    return pd.DataFrame(rows)


def markdown_table(headers: List[str], rows: List[List[str]]) -> List[str]:
    """Build a simple markdown table."""
    if not rows:
        return []

    widths: List[int] = []
    for col_idx, header in enumerate(headers):
        max_row_width = max(len(str(row[col_idx])) for row in rows)
        widths.append(max(len(header), max_row_width))

    padded_headers = [str(header).ljust(widths[idx]) for idx, header in enumerate(headers)]
    header_line = "| " + " | ".join(padded_headers) + " |"
    separator_line = "| " + " | ".join("-" * widths[idx] for idx in range(len(widths))) + " |"

    body_lines: List[str] = []
    for row in rows:
        padded_row = [str(cell).ljust(widths[idx]) for idx, cell in enumerate(row)]
        body_lines.append("| " + " | ".join(padded_row) + " |")

    return [header_line, separator_line, *body_lines]


def build_summary_report(ranked_df: pd.DataFrame, near_miss_df: pd.DataFrame, live_signals_df: pd.DataFrame) -> str:
    """Build a readable markdown summary of the current research output."""
    lines: List[str] = ["# Pair Research Summary", ""]
    lines.extend(
        [
            "## How To Read This",
            "",
            "- `Score` is a ranking convenience, not a probability of success. Higher is better, but positive OOS Sharpe and positive OOS return matter more than score alone.",
            "- `OOS return` is cumulative across passed walk-forward test windows concatenated in order.",
            "- `OOS annualized return` converts that cumulative return into a yearly rate using `OOS unique days`.",
            "- `OOS return per trade` is cumulative OOS return divided by total OOS trades. It is a rough efficiency measure, not a standalone quality metric.",
            "- `OOS accumulated days` is the total number of test-slice trading days included in that stitched return.",
            "- `OOS unique days` is the number of distinct calendar trading days covered by those passed windows. Because walk-forward test windows overlap, this can be lower than accumulated days.",
            "- `Avg Coint`, `Avg ADF`, and `Avg Half-Life` are averaged only across passed windows and are better structural diagnostics than the latest window alone.",
            "",
            "## Practical Criteria",
            "",
            "- Strong structure: avg cointegration p-value below `0.10`, avg ADF p-value below `0.10`, avg half-life between `3` and `40`, and avg train correlation preferably above `0.65`.",
            "- Strong performance: OOS Sharpe above `0.5`, OOS return above `0`, and low drawdown relative to return.",
            "- Better confidence: more passed windows and more OOS trades. A pair with only 2 windows and 5 trades can still rank well, but it is less proven than one with 4 windows and 8+ trades.",
            "- Weak sign: negative OOS Sharpe or negative OOS return, even if the pair has many passed windows.",
            "- `Confidence score` emphasizes depth and stability: passed windows, OOS trades, unique OOS days, correlation, and structural averages.",
            "- `Robustness score` measures how often the pair still qualifies across nearby parameter settings and a non-overlapping validation variant.",
            "",
            "## Score Heuristics",
            "",
            "- Roughly `7+`: very strong by this model, usually driven by strong OOS Sharpe plus positive return.",
            "- Roughly `5` to `7`: good research candidate.",
            "- Roughly `3` to `5`: usable but weaker or less proven.",
            "- Below `3`: marginal unless there is a specific reason to keep it.",
            "",
            "## Confidence Heuristics",
            "",
            "- Roughly `8+`: strong confidence for this script.",
            "- Roughly `6.5` to `8`: decent confidence.",
            "- Roughly `5` to `6.5`: moderate confidence.",
            "- Below `5`: thin evidence even if performance looks good.",
            "",
            "## Robustness Heuristics",
            "",
            "- Roughly `8+`: strong robustness across nearby settings.",
            "- Roughly `6` to `8`: decent robustness.",
            "- Below `6`: likely more parameter-sensitive.",
            "",
        ]
    )

    if ranked_df.empty:
        lines.extend(["No qualifying pairs found.", ""])
    else:
        lines.extend(["## Ranked Pairs", ""])
        ranked_rows: List[List[str]] = []
        for _, row in ranked_df.iterrows():
            ranked_rows.append(
                [
                    str(row["sector"]),
                    str(row["pair"]),
                    str(row["research_verdict"]),
                    str(row["research_recommendation"]),
                    str(row["walk_forward_mode"]),
                    f"{safe_float(row['robustness_score']):.2f}",
                    f"{safe_float(row['robustness_pass_rate']) * 100:.0f}%",
                    str(int(safe_float(row["confidence_rank"]))),
                    f"{safe_float(row['confidence_score']):.2f}",
                    f"{safe_float(row['score']):.2f}",
                    f"{safe_float(row['oos_sharpe']):.2f}",
                    f"{safe_float(row['oos_return']) * 100:.1f}%",
                    f"{safe_float(row['oos_annualized_return']) * 100:.1f}%",
                    f"{safe_float(row['oos_max_drawdown']) * 100:.1f}%",
                    str(int(safe_float(row["oos_trades"]))),
                    f"{safe_float(row['oos_return_per_trade']) * 100:.1f}%",
                    str(int(safe_float(row["oos_accumulated_test_days"]))),
                    str(int(safe_float(row["oos_unique_test_days"]))),
                    f"{int(safe_float(row['windows_passed']))}/{int(safe_float(row['windows_tested']))}",
                    f"{safe_float(row['avg_coint_pvalue_passed']):.3f}",
                    f"{safe_float(row['avg_adf_pvalue_passed']):.3f}",
                    f"{safe_float(row['avg_half_life_passed']):.1f}",
                ]
            )
        lines.extend(
            markdown_table(
                [
                    "Sector",
                    "Pair",
                    "Verdict",
                    "Research Rec",
                    "WF Mode",
                    "Robust",
                    "Robust Pass",
                    "Conf Rank",
                    "Conf Score",
                    "Score",
                    "Sharpe",
                    "Return",
                    "Ann Return",
                    "Max DD",
                    "Trades",
                    "Return/Trade",
                    "OOS Accum Days",
                    "OOS Unique Days",
                    "Passed",
                    "Avg Coint",
                    "Avg ADF",
                    "Avg Half-Life",
                ],
                ranked_rows,
            )
        )
        lines.extend(["", "## Score Breakdown", ""])
        score_rows: List[List[str]] = []
        for _, row in ranked_df.iterrows():
            score_rows.append(
                [
                    str(row["pair"]),
                    f"{safe_float(row['score_sharpe_component']):.2f}",
                    f"{safe_float(row['score_return_component']):.2f}",
                    f"-{safe_float(row['score_drawdown_penalty']):.2f}",
                    f"{safe_float(row['score_windows_component']):.2f}",
                    f"{safe_float(row['score_trades_component']):.2f}",
                    f"{safe_float(row['score']):.2f}",
                ]
            )
        lines.extend(
            markdown_table(
                ["Pair", "Sharpe Part", "Return Part", "DD Penalty", "Windows Part", "Trades Part", "Score"],
                score_rows,
            )
        )
        lines.extend(["", "## Confidence Ranking", ""])
        confidence_df = ranked_df.sort_values(
            by=["confidence_rank", "confidence_score", "oos_unique_test_days", "windows_passed", "oos_trades"],
            ascending=[True, False, False, False, False],
        ).reset_index(drop=True)
        confidence_rows: List[List[str]] = []
        for _, row in confidence_df.iterrows():
            confidence_rows.append(
                [
                    str(row["pair"]),
                    str(int(safe_float(row["confidence_rank"]))),
                    f"{safe_float(row['confidence_score']):.2f}",
                    str(int(safe_float(row["oos_unique_test_days"]))),
                    f"{int(safe_float(row['windows_passed']))}/{int(safe_float(row['windows_tested']))}",
                    str(int(safe_float(row["oos_trades"]))),
                    f"{safe_float(row['avg_coint_pvalue_passed']):.3f}",
                    f"{safe_float(row['avg_adf_pvalue_passed']):.3f}",
                ]
            )
        lines.extend(
            markdown_table(
                ["Pair", "Conf Rank", "Conf Score", "Unique Days", "Passed", "Trades", "Avg Coint", "Avg ADF"],
                confidence_rows,
            )
        )
        lines.extend(["", "## Interpretation", ""])
        top_row = ranked_df.iloc[0]
        top_confidence_row = confidence_df.iloc[0]
        lines.append(
            f"Top pair is `{top_row['pair']}` with score `{safe_float(top_row['score']):.2f}`, OOS Sharpe `{safe_float(top_row['oos_sharpe']):.2f}`, cumulative return `{safe_float(top_row['oos_return']) * 100:.1f}%`, annualized return `{safe_float(top_row['oos_annualized_return']) * 100:.1f}%`, accumulated over `{int(safe_float(top_row['oos_accumulated_test_days']))}` stitched test days (`{int(safe_float(top_row['oos_unique_test_days']))}` unique days)."
        )
        lines.append(
            f"Highest-confidence pair is `{top_confidence_row['pair']}` with confidence score `{safe_float(top_confidence_row['confidence_score']):.2f}` and confidence rank `{int(safe_float(top_confidence_row['confidence_rank']))}`."
        )
        robust_df = ranked_df.sort_values(
            by=["robustness_score", "robustness_pass_rate", "confidence_score", "oos_sharpe"],
            ascending=[False, False, False, False],
        ).reset_index(drop=True)
        top_robust_row = robust_df.iloc[0]
        lines.append(
            f"Most robust pair is `{top_robust_row['pair']}` with robustness score `{safe_float(top_robust_row['robustness_score']):.2f}` and scenario pass rate `{safe_float(top_robust_row['robustness_pass_rate']) * 100:.0f}%`."
        )
        action_counts = ranked_df["research_recommendation"].value_counts().to_dict()
        lines.append(
            f"Research summary: {action_counts.get('PAPER_TRADE_READY', 0)} `PAPER_TRADE_READY`, {action_counts.get('WATCHLIST', 0)} `WATCHLIST`, {action_counts.get('AVOID', 0)} `AVOID`."
        )
        lines.append("Use `windows_passed` and `oos_trades` as confidence context, not as the main reason to prefer a pair.")
        lines.append("")

    lines.extend(["## Live Signals", ""])
    if live_signals_df.empty:
        lines.extend(["No live-signal rows available.", ""])
    else:
        live_rows: List[List[str]] = []
        for _, row in live_signals_df.iterrows():
            live_rows.append(
                [
                    str(row["sector"]),
                    str(row["pair"]),
                    str(row["latest_date"]),
                    f"{safe_float(row['live_zscore']):.2f}",
                    f"{safe_float(row['live_beta']):.3f}",
                    f"{safe_float(row['live_half_life']):.1f}",
                    str(int(safe_float(row["current_position"]))),
                    str(row["current_action"]),
                    str(row["live_recommendation"]),
                    "YES" if bool(row["passes_live_stability"]) else "NO",
                    str(row["live_stability_reason"]),
                ]
            )
        lines.extend(
            markdown_table(
                ["Sector", "Pair", "Date", "Live Z", "Live Beta", "Live Half-Life", "Position", "Action", "Live Rec", "Stable", "Stability Reason"],
                live_rows,
            )
        )
        lines.append("")

    if near_miss_df.empty:
        lines.extend(["## Near Misses", "", "No near-miss pairs.", ""])
    else:
        lines.extend(["## Near Misses", ""])
        near_miss_rows: List[List[str]] = []
        for _, row in near_miss_df.head(10).iterrows():
            near_miss_rows.append(
                [
                    str(row["sector"]),
                    str(row["pair"]),
                    str(row["research_verdict"]),
                    f"{safe_float(row['oos_sharpe']):.2f}",
                    f"{safe_float(row['oos_return']) * 100:.1f}%",
                    str(int(safe_float(row["oos_trades"]))),
                    f"{int(safe_float(row['windows_passed']))}/{int(safe_float(row['windows_tested']))}",
                    str(row["fail_reason"]),
                ]
            )
        lines.extend(
            markdown_table(
                ["Sector", "Pair", "Verdict", "Sharpe", "Return", "Trades", "Passed", "Reason"],
                near_miss_rows,
            )
        )
        lines.append("")

    lines.extend(
        [
            "## Score Formula",
            "",
            "`score = 4.0 * max(oos_sharpe, -2.0) + 1.5 * oos_return - 2.0 * abs(oos_max_drawdown) + 0.25 * windows_passed + 0.05 * oos_trades`",
            "",
        ]
    )
    return "\n".join(lines)


def save_outputs(
    ranked_df: pd.DataFrame,
    diagnostics_df: pd.DataFrame,
    window_df: pd.DataFrame,
    live_signals_df: pd.DataFrame,
) -> Dict[str, Path]:
    """Save CSV outputs and return their resolved paths."""
    ranked_path = Path(RANKED_OUTPUT_CSV)
    failed_path = Path(FAILED_OUTPUT_CSV)
    near_miss_path = Path(NEAR_MISS_OUTPUT_CSV)
    window_path = Path(WINDOW_OUTPUT_CSV)
    live_signals_path = Path(LIVE_SIGNALS_OUTPUT_CSV)
    summary_report_path = Path(SUMMARY_REPORT_MD)

    ranked_df.to_csv(ranked_path, index=False)
    failed_df = diagnostics_df[diagnostics_df["fail_reason"].astype(str) != ""].copy() if not diagnostics_df.empty else diagnostics_df
    failed_df.to_csv(failed_path, index=False)
    near_miss_df = pd.DataFrame()
    if not diagnostics_df.empty:
        near_miss_mask = (
            (diagnostics_df["fail_reason"].astype(str) == "failed qualifying filter")
            & (diagnostics_df["research_verdict"].astype(str) == "WEAK_CANDIDATE")
        )
        near_miss_df = diagnostics_df[near_miss_mask].copy()
        if not near_miss_df.empty:
            near_miss_df = near_miss_df.sort_values(
                by=["oos_sharpe", "oos_return", "windows_passed", "oos_trades"],
                ascending=[False, False, False, False],
            ).reset_index(drop=True)
    near_miss_df.to_csv(near_miss_path, index=False)
    window_df.to_csv(window_path, index=False)
    live_signals_df.to_csv(live_signals_path, index=False)
    summary_report_path.write_text(build_summary_report(ranked_df, near_miss_df, live_signals_df), encoding="utf-8")

    return {
        "ranked": ranked_path.resolve(),
        "failed": failed_path.resolve(),
        "near_miss": near_miss_path.resolve(),
        "window": window_path.resolve(),
        "live_signals": live_signals_path.resolve(),
        "summary": summary_report_path.resolve(),
        "plots": PLOT_DIR.resolve(),
    }


if __name__ == "__main__":
    clear_plot_dir(PLOT_DIR)
    ranked_pairs, diagnostics, window_metrics, downloaded_prices = analyze_universe(UNIVERSE)
    live_signals = build_live_signals(ranked_pairs, downloaded_prices, LIVE_SIGNAL_TOP_N)

    if ranked_pairs.empty:
        print("No qualifying pairs found.")
        output_paths = save_outputs(ranked_pairs, diagnostics, window_metrics, live_signals)
        print("\nSaved CSV files:")
        print(f"Ranked qualifying pairs: {output_paths['ranked']}")
        print(f"Failed pair diagnostics: {output_paths['failed']}")
        print(f"Near-miss pairs: {output_paths['near_miss']}")
        print(f"Window-level metrics: {output_paths['window']}")
        print(f"Live signals: {output_paths['live_signals']}")
        print(f"Summary report: {output_paths['summary']}")
        print(f"Plot folder: {output_paths['plots']}")
    else:
        plot_pair_diagnostics(
            ranked_df=ranked_pairs,
            prices=downloaded_prices,
            top_n=TOP_N_PLOTS,
            plot_dir=PLOT_DIR,
        )

        output_paths = save_outputs(ranked_pairs, diagnostics, window_metrics, live_signals)
        print_top_ranked_pairs(ranked_pairs, TOP_N_PRINT)
        print_sector_summary(ranked_pairs)

        print("\nSaved CSV files:")
        print(f"Ranked qualifying pairs: {output_paths['ranked']}")
        print(f"Failed pair diagnostics: {output_paths['failed']}")
        print(f"Near-miss pairs: {output_paths['near_miss']}")
        print(f"Window-level metrics: {output_paths['window']}")
        print(f"Live signals: {output_paths['live_signals']}")
        print(f"Summary report: {output_paths['summary']}")
        print(f"Plot folder: {output_paths['plots']}")
