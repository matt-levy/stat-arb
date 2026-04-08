import unittest

import numpy as np
import pandas as pd

from backtest import compute_trading_cost, run_backtest, total_trading_cost_rate
from config import MAX_HOLD_DAYS, MAX_PAIR_WEIGHT
from portfolio import normalize_capped_weights, weight_from_row
from signals import compute_pair_filters, generate_strategy_state, get_entry_decision, get_latest_signal


def make_price_df(length: int = 260, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=length, freq="B")
    x = 100 + np.cumsum(rng.normal(0.1, 0.8, length))
    spread = np.sin(np.linspace(0, 8, length)) * 1.5 + rng.normal(0, 0.25, length)
    y = 1.1 * x + spread
    return pd.DataFrame({"Y": y, "X": x}, index=dates)


class StrategyComponentTests(unittest.TestCase):
    def test_cost_model_charges_expected_entry_and_exit_costs(self):
        traded_notional = 100_000.0
        expected_cost = traded_notional * total_trading_cost_rate()

        entry_cost = compute_trading_cost(traded_notional)
        exit_cost = compute_trading_cost(traded_notional)

        self.assertAlmostEqual(entry_cost, expected_cost)
        self.assertAlmostEqual(exit_cost, expected_cost)
        self.assertAlmostEqual(entry_cost + exit_cost, expected_cost * 2)

    def test_volatility_aware_sizing_normalizes_and_respects_caps(self):
        low_vol_weight = weight_from_row(
            {
                "test_sharpe": 1.0,
                "train_sharpe": 0.5,
                "test_return": 0.10,
                "correlation": 0.7,
                "test_dd": -0.05,
                "recent_spread_vol_ratio": 1.0,
            }
        )
        high_vol_weight = weight_from_row(
            {
                "test_sharpe": 1.0,
                "train_sharpe": 0.5,
                "test_return": 0.10,
                "correlation": 0.7,
                "test_dd": -0.05,
                "recent_spread_vol_ratio": 2.0,
            }
        )

        normalized = normalize_capped_weights(pd.Series([10.0, 5.0, 3.0]), cap=MAX_PAIR_WEIGHT)

        self.assertGreater(low_vol_weight, high_vol_weight)
        self.assertAlmostEqual(normalized.sum(), 1.0)
        self.assertLessEqual(normalized.max(), MAX_PAIR_WEIGHT + 1e-9)

    def test_extreme_z_entries_are_rejected(self):
        signal, reason = get_entry_decision(current_z=-4.2, previous_z=-5.0, hl_ok=True)
        self.assertEqual(signal, 0)
        self.assertEqual(reason, "EXTREME_Z_REJECT")

    def test_time_based_exit_triggers_at_max_hold_limit(self):
        dates = pd.date_range("2024-01-01", periods=MAX_HOLD_DAYS + 3, freq="B")
        signal_df = pd.DataFrame(
            {
                "z": [-3.2, -2.6] + [-1.8] * (MAX_HOLD_DAYS + 1),
                "hl_ok": [True] * (MAX_HOLD_DAYS + 3),
            },
            index=dates,
        )

        state_df = generate_strategy_state(signal_df)
        time_stop_rows = state_df[state_df["exit_reason"] == "TIME_STOP"]

        self.assertFalse(time_stop_rows.empty)
        self.assertEqual(time_stop_rows.iloc[0]["action_type"], "EXIT")

    def test_shared_signal_logic_matches_backtest_and_latest_signal(self):
        price_df = make_price_df()

        latest_signal = get_latest_signal(price_df, "Y", "X")
        backtest_df = run_backtest(price_df, "Y", "X")

        self.assertIsNotNone(latest_signal)
        self.assertEqual(latest_signal["position"], int(backtest_df["pos"].iloc[-1]))
        self.assertEqual(latest_signal["action_type"], backtest_df["action_type"].iloc[-1])
        self.assertEqual(latest_signal["exit_reason"], backtest_df["exit_reason"].iloc[-1])

    def test_stability_filters_reject_unstable_input(self):
        rng = np.random.default_rng(11)
        dates = pd.date_range("2023-01-01", periods=280, freq="B")
        x = 100 + np.cumsum(rng.normal(0.0, 1.0, len(dates)))
        stable_y = x[:180] + rng.normal(0.0, 0.5, 180)
        unstable_tail = 140 + np.cumsum(rng.normal(0.0, 4.0, len(dates) - 180))
        y = np.concatenate([stable_y, unstable_tail])
        df = pd.DataFrame({"Y": y, "X": x}, index=dates)

        filters = compute_pair_filters(df, "Y", "X")

        self.assertFalse(filters["passes_stability"])
        self.assertTrue(filters["stability_rejection_reason"])


if __name__ == "__main__":
    unittest.main()
