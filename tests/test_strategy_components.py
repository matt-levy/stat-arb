import unittest
from unittest.mock import patch

import pandas as pd

from pair_checker import compute_event_context, determine_operational_action
from paper_trading_ready import build_ready_pairs, normalize_capped_weights, weight_from_row
from run_pipeline import main as run_pipeline_main


class ReadySignalTests(unittest.TestCase):
    def test_unstable_active_research_ready_pair_is_hold_only(self):
        recommendation = determine_operational_action(
            research_verdict="STRONG_CANDIDATE",
            confidence_score=9.0,
            robustness_score=8.0,
            live_action="LONG_SPREAD",
            passes_live_stability=False,
        )

        self.assertEqual(recommendation, "HOLD_ONLY")

    def test_leg_contribution_failure_is_diagnostic_only(self):
        recommendation = determine_operational_action(
            research_verdict="STRONG_CANDIDATE",
            confidence_score=9.0,
            robustness_score=8.0,
            live_action="SHORT_SPREAD",
            passes_live_stability=True,
            passes_leg_contribution=False,
        )

        self.assertEqual(recommendation, "PAPER_TRADE_READY")

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

    def test_build_ready_pairs_keeps_only_ready_pairs_and_limits_count(self):
        live_signals = pd.DataFrame(
            [
                {
                    "latest_date": "2026-04-15",
                    "sector": "banks",
                    "pair": "C vs GS",
                    "live_recommendation": "PAPER_TRADE_READY",
                    "current_action": "SHORT_SPREAD",
                    "current_position": -1,
                    "live_zscore": 2.8,
                    "live_beta": 1.0,
                    "live_half_life": 10.0,
                    "passes_live_stability": True,
                    "live_stability_reason": "",
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
                    "pair": "JPM vs GS",
                    "live_recommendation": "PAPER_TRADE_READY",
                    "current_action": "SHORT_SPREAD",
                    "current_position": -1,
                    "live_zscore": 1.5,
                    "live_beta": 0.45,
                    "live_half_life": 14.0,
                    "passes_live_stability": True,
                    "live_stability_reason": "",
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
                    "latest_price_y": 900.0,
                },
                {
                    "latest_date": "2026-04-15",
                    "sector": "semis",
                    "pair": "MU vs LRCX",
                    "live_recommendation": "PAPER_TRADE_READY",
                    "current_action": "LONG_SPREAD",
                    "current_position": 1,
                    "live_zscore": -1.2,
                    "live_beta": 1.4,
                    "live_half_life": 7.0,
                    "passes_live_stability": True,
                    "live_stability_reason": "",
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
                    "live_recommendation": "WATCHLIST",
                    "current_action": "HOLD",
                    "current_position": 0,
                    "live_zscore": 0.2,
                    "live_beta": 1.1,
                    "live_half_life": 9.0,
                    "passes_live_stability": True,
                    "live_stability_reason": "",
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
                    "pair": "JPM vs GS",
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

        self.assertEqual(list(ready_pairs["pair"]), ["C vs GS", "JPM vs GS", "MU vs LRCX"])
        self.assertAlmostEqual(float(ready_pairs["portfolio_weight"].sum()), 1.0)
        self.assertLessEqual(float(ready_pairs["portfolio_weight"].max()), 0.50 + 1e-9)
        self.assertNotIn("XOM vs CVX", set(ready_pairs["pair"]))

    def test_build_ready_pairs_returns_empty_without_matching_inputs(self):
        live_signals = pd.DataFrame(
            [
                {
                    "latest_date": "2026-04-15",
                    "sector": "banks",
                    "pair": "C vs GS",
                    "live_recommendation": "WATCHLIST",
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


class RunPipelineTests(unittest.TestCase):
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
