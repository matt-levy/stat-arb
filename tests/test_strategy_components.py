import unittest
from unittest.mock import patch

import pandas as pd

from pair_checker import (
    compute_current_edge_to_exit_ratio,
    compute_current_expected_edge,
    estimate_round_trip_cost_buffer,
    compute_event_context,
    leg_contribution_size_multiplier,
    determine_plot_window,
    determine_live_stability_tier,
    determine_operational_action,
    live_stability_size_multiplier,
    select_pairs_for_plotting,
)
from paper_trading_ready import (
    build_ready_pairs,
    capped_absolute_weights,
    normalize_capped_weights,
    select_distinct_symbol_pairs,
    weight_from_row,
)
from run_pipeline import main as run_pipeline_main


class ReadySignalTests(unittest.TestCase):
    def test_current_edge_to_exit_ratio_scales_with_current_zscore(self):
        self.assertAlmostEqual(compute_current_edge_to_exit_ratio(1.75, 1.75, 0.0), 1.0)
        self.assertAlmostEqual(compute_current_edge_to_exit_ratio(2.625, 1.75, 0.0), 1.5)

    def test_current_expected_edge_applies_live_haircut(self):
        edge = compute_current_expected_edge(
            latest_zscore=2.625,
            entry_z=1.75,
            exit_z=0.0,
            oos_return_per_trade=0.04,
            size_multiplier=0.5,
        )

        self.assertAlmostEqual(edge, 0.03, places=6)

    def test_estimated_round_trip_cost_buffer_matches_backtest_costs(self):
        self.assertAlmostEqual(estimate_round_trip_cost_buffer(), 0.002, places=9)

    def test_unstable_active_research_ready_pair_is_qualified_but_blocked(self):
        recommendation = determine_operational_action(
            research_verdict="STRONG_CANDIDATE",
            confidence_score=9.0,
            robustness_score=8.0,
            live_action="LONG_SPREAD",
            passes_live_stability=False,
            live_stability_reason="LOW_RECENT_CORR|UNSTABLE_BETA",
            live_degradation_score=3.5,
        )

        self.assertEqual(recommendation, "QUALIFIED_BUT_BLOCKED")

    def test_single_mild_live_stability_failure_becomes_borderline_eligible(self):
        recommendation = determine_operational_action(
            research_verdict="STRONG_CANDIDATE",
            confidence_score=9.0,
            robustness_score=8.5,
            live_action="LONG_SPREAD",
            passes_live_stability=False,
            live_stability_reason="UNSTABLE_BETA",
            live_degradation_score=1.5,
        )

        self.assertEqual(recommendation, "ELIGIBLE")
        self.assertEqual(
            determine_live_stability_tier(
                confidence_score=9.0,
                robustness_score=8.5,
                passes_live_stability=False,
                live_stability_reason="UNSTABLE_BETA",
                live_degradation_score=1.5,
            ),
            "BORDERLINE",
        )
        self.assertEqual(live_stability_size_multiplier("BORDERLINE"), 0.5)

    def test_high_degradation_score_stays_blocked(self):
        self.assertEqual(
            determine_live_stability_tier(
                confidence_score=9.0,
                robustness_score=8.5,
                passes_live_stability=False,
                live_stability_reason="UNSTABLE_BETA",
                live_degradation_score=3.2,
            ),
            "FAIL",
        )

    def test_extreme_leg_contribution_failure_blocks_active_signal(self):
        recommendation = determine_operational_action(
            research_verdict="STRONG_CANDIDATE",
            confidence_score=9.0,
            robustness_score=8.0,
            live_action="SHORT_SPREAD",
            passes_live_stability=True,
            live_stability_reason="",
            live_degradation_score=0.0,
            passes_leg_contribution=False,
            leg_contribution_reason="EXTREME_ONE_LEG_DOMINANCE",
        )

        self.assertEqual(recommendation, "QUALIFIED_BUT_BLOCKED")

    def test_mild_leg_dominance_stays_eligible_but_sizes_down(self):
        recommendation = determine_operational_action(
            research_verdict="STRONG_CANDIDATE",
            confidence_score=9.0,
            robustness_score=8.0,
            live_action="SHORT_SPREAD",
            passes_live_stability=True,
            live_stability_reason="",
            live_degradation_score=0.0,
            passes_leg_contribution=False,
            leg_contribution_reason="ONE_LEG_DOMINANCE",
        )

        self.assertEqual(recommendation, "ELIGIBLE")
        self.assertEqual(leg_contribution_size_multiplier(False, "ONE_LEG_DOMINANCE"), 0.5)

    def test_event_context_detects_recent_public_event_window(self):
        events = pd.DataFrame(
            [
                {
                    "symbol": "C",
                    "event_date": pd.Timestamp("2026-04-14"),
                    "event_type": "earnings",
                    "source": "https://example.com/public-earnings",
                }
            ]
        )

        context = compute_event_context("C", "GS", pd.Timestamp("2026-04-16"), events)

        self.assertTrue(context["has_event_window"])
        self.assertEqual(context["event_symbols"], "C")
        self.assertIn("C:earnings:recent", context["event_reason"])

    def test_weight_from_row_rewards_stronger_research_inputs(self):
        weak_row = {
            "score": 1.0,
            "confidence_score": 2.0,
            "robustness_score": 3.0,
            "oos_sharpe": 0.5,
            "oos_annualized_return": 0.05,
        }
        strong_row = {
            "score": 3.0,
            "confidence_score": 4.0,
            "robustness_score": 5.0,
            "oos_sharpe": 1.5,
            "oos_annualized_return": 0.15,
        }

        self.assertGreater(weight_from_row(strong_row), weight_from_row(weak_row))

    def test_normalize_capped_weights_respects_cap_and_sums_to_one(self):
        weights = normalize_capped_weights(pd.Series([10.0, 5.0, 1.0]), cap=0.50)

        self.assertAlmostEqual(weights.sum(), 1.0)
        self.assertLessEqual(weights.max(), 0.50 + 1e-9)

    def test_capped_absolute_weights_preserve_unused_capacity(self):
        weights = capped_absolute_weights(pd.Series([10.0, 5.0, 1.0]), cap=0.50)

        self.assertLess(float(weights.sum()), 1.0)
        self.assertLessEqual(float(weights.max()), 0.50 + 1e-9)

    def test_select_distinct_symbol_pairs_skips_overlapping_symbols(self):
        candidates = pd.DataFrame(
            [
                {"pair": "C vs GS", "stock_x": "C", "stock_y": "GS", "score": 9.0},
                {"pair": "JPM vs GS", "stock_x": "JPM", "stock_y": "GS", "score": 8.0},
                {"pair": "MU vs LRCX", "stock_x": "MU", "stock_y": "LRCX", "score": 7.0},
                {"pair": "XOM vs CVX", "stock_x": "XOM", "stock_y": "CVX", "score": 6.0},
            ]
        )

        selected = select_distinct_symbol_pairs(candidates, max_pairs=3)

        self.assertEqual(list(selected["pair"]), ["C vs GS", "MU vs LRCX", "XOM vs CVX"])

    def test_build_ready_pairs_keeps_only_ready_pairs_and_limits_count(self):
        live_signals = pd.DataFrame(
            [
                {
                    "latest_date": "2026-04-15",
                    "sector": "banks",
                    "pair": "C vs GS",
                    "live_recommendation": "ELIGIBLE",
                    "current_action": "SHORT_SPREAD",
                    "current_position": -1,
                    "live_zscore": 2.8,
                    "live_beta": 1.0,
                    "live_half_life": 10.0,
                    "current_edge_to_exit_ratio": 1.6,
                    "current_expected_edge": 0.028,
                    "estimated_round_trip_cost": 0.002,
                    "current_net_expected_edge": 0.026,
                    "passes_live_stability": True,
                    "live_stability_reason": "",
                    "live_degradation_score": 0.0,
                    "live_stability_tier": "PASS",
                    "live_size_multiplier": 1.0,
                    "passes_leg_contribution": True,
                    "leg_contribution_reason": "",
                    "recent_x_contribution": 0.01,
                    "recent_y_contribution": 0.01,
                    "dominant_leg": "X",
                    "dominant_leg_share": 0.5,
                    "has_event_window": False,
                    "event_symbols": "",
                    "event_reason": "",
                    "latest_event_date": "",
                    "event_days_from_signal": "",
                    "latest_price_x": 130.0,
                    "latest_price_y": 900.0,
                },
                {
                    "latest_date": "2026-04-15",
                    "sector": "banks",
                    "pair": "JPM vs MS",
                    "live_recommendation": "ELIGIBLE",
                    "current_action": "SHORT_SPREAD",
                    "current_position": -1,
                    "live_zscore": 1.5,
                    "live_beta": 0.45,
                    "live_half_life": 14.0,
                    "current_edge_to_exit_ratio": 0.9,
                    "current_expected_edge": 0.010,
                    "estimated_round_trip_cost": 0.002,
                    "current_net_expected_edge": 0.008,
                    "passes_live_stability": False,
                    "live_stability_reason": "UNSTABLE_BETA",
                    "live_degradation_score": 1.5,
                    "live_stability_tier": "BORDERLINE",
                    "live_size_multiplier": 0.5,
                    "passes_leg_contribution": True,
                    "leg_contribution_reason": "",
                    "recent_x_contribution": 0.01,
                    "recent_y_contribution": 0.01,
                    "dominant_leg": "X",
                    "dominant_leg_share": 0.5,
                    "has_event_window": False,
                    "event_symbols": "",
                    "event_reason": "",
                    "latest_event_date": "",
                    "event_days_from_signal": "",
                    "latest_price_x": 300.0,
                    "latest_price_y": 120.0,
                },
                {
                    "latest_date": "2026-04-15",
                    "sector": "semis",
                    "pair": "MU vs LRCX",
                    "live_recommendation": "ELIGIBLE",
                    "current_action": "LONG_SPREAD",
                    "current_position": 1,
                    "live_zscore": -1.2,
                    "live_beta": 1.4,
                    "live_half_life": 7.0,
                    "current_edge_to_exit_ratio": 0.7,
                    "current_expected_edge": 0.012,
                    "estimated_round_trip_cost": 0.002,
                    "current_net_expected_edge": 0.010,
                    "passes_live_stability": True,
                    "live_stability_reason": "",
                    "live_degradation_score": 0.0,
                    "live_stability_tier": "PASS",
                    "live_size_multiplier": 1.0,
                    "passes_leg_contribution": True,
                    "leg_contribution_reason": "",
                    "recent_x_contribution": 0.01,
                    "recent_y_contribution": 0.01,
                    "dominant_leg": "X",
                    "dominant_leg_share": 0.5,
                    "has_event_window": False,
                    "event_symbols": "",
                    "event_reason": "",
                    "latest_event_date": "",
                    "event_days_from_signal": "",
                    "latest_price_x": 450.0,
                    "latest_price_y": 265.0,
                },
                {
                    "latest_date": "2026-04-15",
                    "sector": "oil",
                    "pair": "XOM vs CVX",
                    "live_recommendation": "MONITOR",
                    "current_action": "HOLD",
                    "current_position": 0,
                    "live_zscore": 0.2,
                    "live_beta": 1.1,
                    "live_half_life": 9.0,
                    "current_edge_to_exit_ratio": 0.1,
                    "current_expected_edge": 0.001,
                    "estimated_round_trip_cost": 0.002,
                    "current_net_expected_edge": -0.001,
                    "passes_live_stability": True,
                    "live_stability_reason": "",
                    "live_degradation_score": 0.0,
                    "live_stability_tier": "PASS",
                    "live_size_multiplier": 1.0,
                    "passes_leg_contribution": True,
                    "leg_contribution_reason": "",
                    "recent_x_contribution": 0.01,
                    "recent_y_contribution": 0.01,
                    "dominant_leg": "X",
                    "dominant_leg_share": 0.5,
                    "has_event_window": False,
                    "event_symbols": "",
                    "event_reason": "",
                    "latest_event_date": "",
                    "event_days_from_signal": "",
                    "latest_price_x": 100.0,
                    "latest_price_y": 150.0,
                },
            ]
        )
        ranked_pairs = pd.DataFrame(
            [
                {
                    "sector": "banks",
                    "pair": "C vs GS",
                    "research_verdict": "PASS",
                    "research_recommendation": "TRADE",
                    "score": 9.0,
                    "confidence_score": 10.0,
                    "confidence_rank": 1,
                    "robustness_score": 9.5,
                    "robustness_pass_rate": 1.0,
                    "oos_sharpe": 1.7,
                    "oos_return": 0.40,
                    "oos_annualized_return": 0.55,
                    "oos_max_drawdown": -0.06,
                    "oos_trades": 15,
                    "oos_unique_test_days": 200,
                    "avg_coint_pvalue_passed": 0.02,
                    "avg_adf_pvalue_passed": 0.01,
                    "avg_half_life_passed": 8.0,
                },
                {
                    "sector": "banks",
                    "pair": "JPM vs MS",
                    "research_verdict": "PASS",
                    "research_recommendation": "TRADE",
                    "score": 6.0,
                    "confidence_score": 7.0,
                    "confidence_rank": 2,
                    "robustness_score": 7.5,
                    "robustness_pass_rate": 0.9,
                    "oos_sharpe": 1.2,
                    "oos_return": 0.12,
                    "oos_annualized_return": 0.10,
                    "oos_max_drawdown": -0.04,
                    "oos_trades": 8,
                    "oos_unique_test_days": 150,
                    "avg_coint_pvalue_passed": 0.03,
                    "avg_adf_pvalue_passed": 0.02,
                    "avg_half_life_passed": 9.0,
                },
                {
                    "sector": "semis",
                    "pair": "MU vs LRCX",
                    "research_verdict": "PASS",
                    "research_recommendation": "TRADE",
                    "score": 5.0,
                    "confidence_score": 8.0,
                    "confidence_rank": 1,
                    "robustness_score": 8.5,
                    "robustness_pass_rate": 0.95,
                    "oos_sharpe": 0.8,
                    "oos_return": 0.18,
                    "oos_annualized_return": 0.14,
                    "oos_max_drawdown": -0.15,
                    "oos_trades": 17,
                    "oos_unique_test_days": 300,
                    "avg_coint_pvalue_passed": 0.04,
                    "avg_adf_pvalue_passed": 0.01,
                    "avg_half_life_passed": 7.0,
                },
                {
                    "sector": "tech",
                    "pair": "AAPL vs MSFT",
                    "research_verdict": "PASS",
                    "research_recommendation": "TRADE",
                    "score": 12.0,
                    "confidence_score": 12.0,
                    "confidence_rank": 1,
                    "robustness_score": 12.0,
                    "robustness_pass_rate": 1.0,
                    "oos_sharpe": 2.0,
                    "oos_return": 0.60,
                    "oos_annualized_return": 0.75,
                    "oos_max_drawdown": -0.03,
                    "oos_trades": 25,
                    "oos_unique_test_days": 240,
                    "avg_coint_pvalue_passed": 0.01,
                    "avg_adf_pvalue_passed": 0.01,
                    "avg_half_life_passed": 6.0,
                },
            ]
        )

        ready_pairs = build_ready_pairs(live_signals, ranked_pairs)

        self.assertEqual(list(ready_pairs["pair"]), ["C vs GS", "JPM vs MS", "MU vs LRCX"])
        self.assertLessEqual(float(ready_pairs["portfolio_weight"].sum()), 1.0 + 1e-9)
        self.assertLessEqual(float(ready_pairs["portfolio_weight"].max()), 0.50 + 1e-9)
        weights = ready_pairs.set_index("pair")["portfolio_weight"].to_dict()
        self.assertLess(float(weights["JPM vs MS"]), float(weights["MU vs LRCX"]))
        self.assertEqual(float(ready_pairs.set_index("pair").loc["JPM vs MS", "live_size_multiplier"]), 0.5)
        self.assertNotIn("XOM vs CVX", set(ready_pairs["pair"]))

    def test_build_ready_pairs_leaves_idle_capital_when_only_borderline_pairs_survive(self):
        live_signals = pd.DataFrame(
            [
                {
                    "latest_date": "2026-04-15",
                    "sector": "managed_care",
                    "pair": "ELV vs HUM",
                    "live_recommendation": "ELIGIBLE",
                    "current_action": "SHORT_SPREAD",
                    "current_position": -1,
                    "live_zscore": 2.0,
                    "live_beta": 0.2,
                    "live_half_life": 18.0,
                    "current_edge_to_exit_ratio": 1.1,
                    "current_expected_edge": 0.018,
                    "estimated_round_trip_cost": 0.002,
                    "current_net_expected_edge": 0.016,
                    "passes_live_stability": False,
                    "live_stability_reason": "LOW_RECENT_CORR|UNSTABLE_CORR",
                    "live_degradation_score": 1.4,
                    "live_stability_tier": "BORDERLINE",
                    "live_size_multiplier": 0.5,
                    "passes_leg_contribution": True,
                    "leg_contribution_reason": "",
                    "recent_x_contribution": 0.01,
                    "recent_y_contribution": 0.01,
                    "dominant_leg": "X",
                    "dominant_leg_share": 0.5,
                    "has_event_window": False,
                    "event_symbols": "",
                    "event_reason": "",
                    "latest_event_date": "",
                    "event_days_from_signal": "",
                    "latest_price_x": 340.0,
                    "latest_price_y": 210.0,
                },
                {
                    "latest_date": "2026-04-15",
                    "sector": "banks",
                    "pair": "BAC vs MS",
                    "live_recommendation": "ELIGIBLE",
                    "current_action": "LONG_SPREAD",
                    "current_position": 1,
                    "live_zscore": -1.8,
                    "live_beta": 0.62,
                    "live_half_life": 17.0,
                    "current_edge_to_exit_ratio": 1.03,
                    "current_expected_edge": 0.011,
                    "estimated_round_trip_cost": 0.002,
                    "current_net_expected_edge": 0.009,
                    "passes_live_stability": False,
                    "live_stability_reason": "UNSTABLE_BETA",
                    "live_degradation_score": 0.6,
                    "live_stability_tier": "BORDERLINE",
                    "live_size_multiplier": 0.5,
                    "passes_leg_contribution": True,
                    "leg_contribution_reason": "",
                    "recent_x_contribution": 0.01,
                    "recent_y_contribution": 0.01,
                    "dominant_leg": "X",
                    "dominant_leg_share": 0.5,
                    "has_event_window": False,
                    "event_symbols": "",
                    "event_reason": "",
                    "latest_event_date": "",
                    "event_days_from_signal": "",
                    "latest_price_x": 52.0,
                    "latest_price_y": 188.0,
                },
            ]
        )
        ranked_pairs = pd.DataFrame(
            [
                {
                    "sector": "managed_care",
                    "pair": "ELV vs HUM",
                    "research_verdict": "PASS",
                    "research_recommendation": "TRADE",
                    "score": 10.4,
                    "confidence_score": 9.1,
                    "confidence_rank": 1,
                    "robustness_score": 8.3,
                    "robustness_pass_rate": 0.83,
                    "oos_sharpe": 2.0,
                    "oos_return": 0.36,
                    "oos_annualized_return": 0.52,
                    "oos_max_drawdown": -0.04,
                    "oos_trades": 13,
                    "oos_unique_test_days": 189,
                    "avg_coint_pvalue_passed": 0.07,
                    "avg_adf_pvalue_passed": 0.02,
                    "avg_half_life_passed": 10.4,
                },
                {
                    "sector": "banks",
                    "pair": "BAC vs MS",
                    "research_verdict": "PASS",
                    "research_recommendation": "TRADE",
                    "score": 5.8,
                    "confidence_score": 10.1,
                    "confidence_rank": 1,
                    "robustness_score": 10.0,
                    "robustness_pass_rate": 1.0,
                    "oos_sharpe": 0.77,
                    "oos_return": 0.16,
                    "oos_annualized_return": 0.14,
                    "oos_max_drawdown": -0.10,
                    "oos_trades": 14,
                    "oos_unique_test_days": 294,
                    "avg_coint_pvalue_passed": 0.04,
                    "avg_adf_pvalue_passed": 0.01,
                    "avg_half_life_passed": 8.0,
                },
            ]
        )

        ready_pairs = build_ready_pairs(live_signals, ranked_pairs)

        self.assertEqual(set(ready_pairs["live_stability_tier"]), {"BORDERLINE"})
        self.assertLess(float(ready_pairs["portfolio_weight"].sum()), 1.0)

    def test_build_ready_pairs_returns_empty_without_matching_inputs(self):
        live_signals = pd.DataFrame(
            [
                {
                    "latest_date": "2026-04-15",
                    "sector": "banks",
                    "pair": "C vs GS",
                    "live_recommendation": "MONITOR",
                }
            ]
        )
        ranked_pairs = pd.DataFrame(
            [
                {
                    "sector": "banks",
                    "pair": "C vs GS",
                    "research_verdict": "PASS",
                    "research_recommendation": "TRADE",
                    "score": 9.0,
                    "confidence_score": 10.0,
                    "confidence_rank": 1,
                    "robustness_score": 9.5,
                    "robustness_pass_rate": 1.0,
                    "oos_sharpe": 1.7,
                    "oos_return": 0.40,
                    "oos_annualized_return": 0.55,
                    "oos_max_drawdown": -0.06,
                    "oos_trades": 15,
                    "oos_unique_test_days": 200,
                    "avg_coint_pvalue_passed": 0.02,
                    "avg_adf_pvalue_passed": 0.01,
                    "avg_half_life_passed": 8.0,
                }
            ]
        )

        ready_pairs = build_ready_pairs(live_signals, ranked_pairs)

        self.assertTrue(ready_pairs.empty)

    def test_select_pairs_for_plotting_keeps_only_active_pairs_in_live_order(self):
        ranked_pairs = pd.DataFrame(
            [
                {"pair": "C vs GS", "stock_x": "C", "stock_y": "GS", "latest_beta": 1.0},
                {"pair": "MU vs LRCX", "stock_x": "MU", "stock_y": "LRCX", "latest_beta": 1.3},
                {"pair": "BAC vs MS", "stock_x": "BAC", "stock_y": "MS", "latest_beta": 0.7},
            ]
        )
        live_signals = pd.DataFrame(
            [
                {"pair": "MU vs LRCX", "current_position": 1},
                {"pair": "BAC vs MS", "current_position": 0},
                {"pair": "C vs GS", "current_position": -1},
            ]
        )

        selected = select_pairs_for_plotting(ranked_pairs, live_signals)

        self.assertEqual(list(selected["pair"]), ["MU vs LRCX", "C vs GS"])


class RunPipelineTests(unittest.TestCase):
    def test_determine_plot_window_starts_before_current_trade_entry(self):
        index = pd.date_range("2026-05-01", periods=8, freq="B")
        position = pd.Series([0, 0, 1, 1, 1, 1, 1, 1], index=index)

        visible_index, entry_timestamp = determine_plot_window(position, pre_entry_bars=2, fallback_bars=4)

        self.assertEqual(entry_timestamp, index[2])
        self.assertEqual(list(visible_index), list(index[0:]))

    def test_determine_plot_window_falls_back_when_flat(self):
        index = pd.date_range("2026-05-01", periods=8, freq="B")
        position = pd.Series([0, 0, 0, 0, 0, 0, 0, 0], index=index)

        visible_index, entry_timestamp = determine_plot_window(position, pre_entry_bars=2, fallback_bars=3)

        self.assertIsNone(entry_timestamp)
        self.assertEqual(list(visible_index), list(index[-3:]))

    @patch("run_pipeline.run_step")
    @patch("sys.argv", ["run_pipeline.py", "--skip-research", "--skip-ready", "--execute", "--allow-stale"])
    def test_main_only_runs_alpaca_step_with_requested_flags(self, run_step_mock):
        run_pipeline_main()

        run_step_mock.assert_called_once()
        command, label = run_step_mock.call_args.args
        self.assertEqual(label, "Alpaca")
        self.assertEqual(command[1:], ["alpaca_paper_trading.py", "--execute", "--allow-stale"])

    @patch("run_pipeline.run_step")
    @patch("sys.argv", ["run_pipeline.py"])
    def test_main_runs_all_pipeline_steps_by_default(self, run_step_mock):
        run_pipeline_main()

        labels = [call.args[1] for call in run_step_mock.call_args_list]
        self.assertEqual(labels, ["Research", "Ready Signals", "Alpaca"])


if __name__ == "__main__":
    unittest.main()
