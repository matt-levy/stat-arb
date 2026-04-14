import unittest
from unittest.mock import Mock

import pandas as pd

from alpaca_paper_trading import (
    AlpacaClient,
    AlpacaConfig,
    build_client_order_id,
    build_order_preview,
    build_pair_risk_rows,
    build_trade_log_rows,
    cancel_conflicting_open_orders,
    filter_blocked_pairs,
    get_pairs_already_submitted_this_cycle,
    get_pairs_in_cooldown,
)


def make_config() -> AlpacaConfig:
    return AlpacaConfig(
        api_key="k",
        secret_key="s",
        base_url="https://example.com",
        dry_run=True,
        gross_exposure_fraction=0.5,
        buying_power_usage_fraction=0.5,
        min_leg_notional=100.0,
        max_signal_staleness_days=3,
        flatten_on_no_targets=False,
        pair_stop_loss_fraction=0.01,
        pair_denylist=[],
    )


class AlpacaPaperTradingTests(unittest.TestCase):
    def test_build_order_preview_does_not_flatten_without_explicit_symbols(self):
        preview_df = build_order_preview(
            leg_targets=pd.DataFrame(),
            current_positions={"C": -22, "GS": 1},
            managed_symbols=["C", "GS"],
            flatten_symbols=set(),
        )

        self.assertTrue(preview_df.empty)

    def test_build_order_preview_can_flatten_selected_symbols(self):
        preview_df = build_order_preview(
            leg_targets=pd.DataFrame(),
            current_positions={"C": -22, "GS": 1},
            managed_symbols=["C", "GS"],
            flatten_symbols={"C", "GS"},
        )

        preview = preview_df.sort_values("symbol").reset_index(drop=True)
        self.assertEqual(list(preview["symbol"]), ["C", "GS"])
        self.assertEqual(list(preview["side"]), ["buy", "sell"])
        self.assertEqual(list(preview["order_qty"]), [22, 1])

    def test_build_trade_log_rows_expands_shared_symbol_orders_back_to_each_pair(self):
        pair_trade_plan = pd.DataFrame(
            [
                {"signal_date": "2026-04-13", "pair": "C vs GS", "action": "SHORT_SPREAD", "long_symbol": "GS", "long_qty": 1, "short_symbol": "C", "short_qty": 8, "portfolio_weight": 0.4, "live_zscore": 3.1, "live_beta": 1.0, "live_half_life": 9.6, "exit_rule": "Exit when z-score <= 0.0"},
                {"signal_date": "2026-04-13", "pair": "JPM vs GS", "action": "SHORT_SPREAD", "long_symbol": "GS", "long_qty": 1, "short_symbol": "JPM", "short_qty": 2, "portfolio_weight": 0.3, "live_zscore": 2.7, "live_beta": 0.45, "live_half_life": 13.2, "exit_rule": "Exit when z-score <= 0.0"},
            ]
        )
        execution_df = pd.DataFrame(
            [
                {
                    "submitted_at": "2026-04-13T16:48:45",
                    "symbol": "GS",
                    "side": "sell",
                    "order_qty": 1,
                    "target_qty": 0,
                    "current_qty": 1,
                    "alpaca_order_id": "abc",
                    "alpaca_status": "accepted",
                    "source_pairs": "C vs GS | JPM vs GS",
                }
            ]
        )

        trade_log = build_trade_log_rows(pair_trade_plan, execution_df).sort_values("pair").reset_index(drop=True)

        self.assertEqual(list(trade_log["pair"]), ["C vs GS", "JPM vs GS"])
        self.assertEqual(list(trade_log["order_ids"]), ["abc", "abc"])

    def test_cancel_conflicting_open_orders_only_cancels_symbols_in_preview(self):
        client = Mock()
        client.list_open_orders.return_value = [
            {"id": "1", "symbol": "C", "side": "sell", "qty": "30", "status": "accepted"},
            {"id": "2", "symbol": "AAPL", "side": "buy", "qty": "5", "status": "accepted"},
            {"id": "3", "symbol": "LRCX", "side": "sell", "qty": "12", "status": "accepted"},
        ]
        preview_df = pd.DataFrame(
            [
                {"symbol": "C", "side": "buy", "order_qty": 14},
                {"symbol": "LRCX", "side": "buy", "order_qty": 6},
            ]
        )

        cancelled_df = cancel_conflicting_open_orders(client, preview_df).sort_values("symbol").reset_index(drop=True)

        self.assertEqual(client.cancel_order.call_count, 2)
        client.cancel_order.assert_any_call("1")
        client.cancel_order.assert_any_call("3")
        self.assertEqual(list(cancelled_df["symbol"]), ["C", "LRCX"])

    def test_request_handles_no_content_responses(self):
        client = AlpacaClient(make_config())
        response = Mock()
        response.ok = True
        response.content = b""
        response.text = ""
        client.session.request = Mock(return_value=response)

        payload = client.request("DELETE", "/v2/orders/123")

        self.assertEqual(payload, {})

    def test_build_pair_risk_rows_triggers_stop_loss(self):
        pair_trade_plan = pd.DataFrame(
            [
                {
                    "signal_date": "2026-04-13",
                    "pair": "C vs GS",
                    "long_symbol": "GS",
                    "long_qty": 1,
                    "short_symbol": "C",
                    "short_qty": 8,
                }
            ]
        )
        position_details = {
            "GS": {"qty": 1, "avg_entry_price": 902.38, "current_price": 860.0},
            "C": {"qty": -8, "avg_entry_price": 123.275, "current_price": 136.0},
        }

        risk_rows = build_pair_risk_rows(pair_trade_plan, position_details, account_equity=10_000.0, config=make_config())

        self.assertEqual(list(risk_rows["pair"]), ["C vs GS"])
        self.assertEqual(list(risk_rows["event_type"]), ["STOP_LOSS"])

    def test_get_pairs_in_cooldown_blocks_same_signal_cycle(self):
        ready_universe = pd.DataFrame(
            [
                {"pair": "C vs GS", "latest_date": "2026-04-13"},
                {"pair": "MU vs LRCX", "latest_date": "2026-04-13"},
            ]
        )
        risk_rows = pd.DataFrame(
            [
                {"pair": "C vs GS", "latest_date": "2026-04-13", "event_type": "STOP_LOSS"},
                {"pair": "C vs GS", "latest_date": "2026-04-10", "event_type": "STOP_LOSS"},
            ]
        )

        cooldown_pairs = get_pairs_in_cooldown(ready_universe, risk_rows)

        self.assertEqual(cooldown_pairs, {"C vs GS"})

    def test_filter_blocked_pairs_removes_denylisted_pair(self):
        ready_universe = pd.DataFrame(
            [
                {"pair": "C vs GS", "latest_date": "2026-04-13"},
                {"pair": "MU vs LRCX", "latest_date": "2026-04-13"},
            ]
        )

        filtered = filter_blocked_pairs(ready_universe, {"C vs GS"})

        self.assertEqual(list(filtered["pair"]), ["MU vs LRCX"])

    def test_get_pairs_already_submitted_this_cycle_blocks_matching_signal_date(self):
        ready_universe = pd.DataFrame(
            [
                {"pair": "C vs GS", "latest_date": "2026-04-13"},
                {"pair": "MU vs LRCX", "latest_date": "2026-04-13"},
                {"pair": "JPM vs GS", "latest_date": "2026-04-14"},
            ]
        )
        trade_log = pd.DataFrame(
            [
                {"pair": "C vs GS", "signal_date": "2026-04-13", "alpaca_status": "accepted"},
                {"pair": "MU vs LRCX", "signal_date": "2026-04-12", "alpaca_status": "accepted"},
                {"pair": "JPM vs GS", "signal_date": "2026-04-14", "alpaca_status": "rejected"},
            ]
        )

        already_submitted = get_pairs_already_submitted_this_cycle(ready_universe, trade_log)

        self.assertEqual(already_submitted, {"C vs GS"})

    def test_build_client_order_id_is_stable_for_same_target(self):
        first = build_client_order_id(
            signal_date="2026-04-14",
            symbol="C",
            side="buy",
            target_qty=-7,
            source_pairs="C vs GS",
        )
        second = build_client_order_id(
            signal_date="2026-04-14",
            symbol="C",
            side="buy",
            target_qty=-7,
            source_pairs="C vs GS",
        )
        changed = build_client_order_id(
            signal_date="2026-04-14",
            symbol="C",
            side="buy",
            target_qty=-8,
            source_pairs="C vs GS",
        )

        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)


if __name__ == "__main__":
    unittest.main()
