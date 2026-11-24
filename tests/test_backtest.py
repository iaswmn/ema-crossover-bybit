"""Tests for long-only next-bar-open execution, fees, slippage and stats."""

import pandas as pd
import pytest

from solution import BAR_MS, backtest, format_trades, summarize, warmup_bars

T0 = 1_700_000_040_000


def make_bars(opens, closes):
    n = len(opens)
    return pd.DataFrame(
        {
            "start_ms": [T0 + i * BAR_MS for i in range(n)],
            "time": pd.to_datetime([T0 + i * BAR_MS for i in range(n)], unit="ms", utc=True),
            "open": opens,
            "high": [max(o, c) for o, c in zip(opens, closes)],
            "low": [min(o, c) for o, c in zip(opens, closes)],
            "close": closes,
        }
    )


def make_signals(n, buys=(), sells=()):
    return pd.DataFrame({"buy": [i in buys for i in range(n)], "sell": [i in sells for i in range(n)]})


class TestBacktest:
    def test_enters_and_exits_on_next_bar_open(self):
        bars = make_bars(opens=[100, 101, 102, 103, 104], closes=[100.5, 101.5, 102.5, 103.5, 104.5])
        result = backtest(bars, make_signals(5, buys={1}, sells={3}))
        (trade,) = result.trades
        assert trade.entry_price == 102  # open of bar 2
        assert trade.exit_price == 104  # open of bar 4
        assert trade.entry_time == bars["time"][2]
        assert trade.exit_time == bars["time"][4]
        assert trade.signal_entry_close == 101.5
        assert trade.signal_exit_close == 103.5
        assert not trade.forced_close

    def test_pnl_includes_entry_and_exit_fees(self):
        bars = make_bars(opens=[100, 100, 100, 110], closes=[100, 100, 100, 110])
        (trade,) = backtest(bars, make_signals(4, buys={0}, sells={2})).trades
        qty = 1000 / 100
        exit_value = qty * 110
        expected = exit_value - exit_value * 0.0018 - 1000 - 1000 * 0.001
        assert trade.net_pnl_usdt == pytest.approx(expected)
        assert trade.net_pnl_pct == pytest.approx(expected / 1000 * 100)

    def test_flat_trade_loses_exactly_the_fees(self):
        bars = make_bars(opens=[100] * 4, closes=[100] * 4)
        (trade,) = backtest(bars, make_signals(4, buys={0}, sells={1})).trades
        assert trade.net_pnl_usdt == pytest.approx(-(1.0 + 1.8))

    def test_force_closes_open_position_at_last_close(self):
        bars = make_bars(opens=[100, 100, 105, 106], closes=[100, 100, 106, 107])
        (trade,) = backtest(bars, make_signals(4, buys={1})).trades
        assert trade.forced_close
        assert trade.exit_price == 107
        assert trade.exit_time == bars["time"][3] + pd.Timedelta(minutes=1)

    def test_signal_on_last_bar_is_not_executed(self):
        bars = make_bars(opens=[100] * 3, closes=[100] * 3)
        result = backtest(bars, make_signals(3, buys={2}))
        assert result.trades == []
        assert result.unexecuted_signals == 1

    def test_ignores_sell_when_flat_and_buy_when_in_position(self):
        bars = make_bars(opens=[100] * 6, closes=[100] * 6)
        result = backtest(bars, make_signals(6, buys={1, 2}, sells={0, 3}))
        assert len(result.trades) == 1
        assert result.trades[0].entry_time == bars["time"][2]
        assert result.trades[0].exit_time == bars["time"][4]

    def test_flags_slippage_above_half_percent(self):
        bars = make_bars(opens=[100, 100, 101, 100, 100], closes=[100, 100, 100, 100, 100])
        (trade,) = backtest(bars, make_signals(5, buys={1}, sells={3})).trades
        assert trade.entry_slippage_pct == pytest.approx(1.0)
        assert trade.has_slippage
        assert trade.exit_slippage_pct == pytest.approx(0.0)

    def test_no_slippage_flag_within_threshold(self):
        bars = make_bars(opens=[100, 100, 100.4, 100, 100], closes=[100] * 5)
        (trade,) = backtest(bars, make_signals(5, buys={1}, sells={3})).trades
        assert not trade.has_slippage


class TestSummarize:
    def test_stats(self):
        bars = make_bars(opens=[100, 100, 110, 110, 100, 100], closes=[100, 100, 110, 110, 100, 100])
        result = backtest(bars, make_signals(6, buys={0, 3}, sells={1, 4}))
        stats = summarize(result.trades)
        assert stats.trades == 2
        assert stats.winrate_pct == pytest.approx(50.0)
        assert stats.total_pnl_usdt == pytest.approx(sum(t.net_pnl_usdt for t in result.trades))

    def test_empty(self):
        stats = summarize([])
        assert stats.trades == 0
        assert stats.winrate_pct == 0.0
        assert stats.total_pnl_usdt == 0.0


class TestWarmup:
    def test_warmup_bars_until_seed_error_below_threshold(self):
        # (1 - 2/22)^n <= 1e-3 -> n = 73 bars after the seed bar (index 20)
        assert warmup_bars(21) == 20 + 73

    def test_flags_trades_with_signal_inside_warmup(self):
        bars = make_bars(opens=[100] * 8, closes=[100] * 8)
        result = backtest(bars, make_signals(8, buys={1, 5}, sells={3, 6}), warmup=4)
        first, second = result.trades
        assert first.in_warmup  # entry signal on bar 1 < 4
        assert not second.in_warmup  # signals on bars 5 and 6

    def test_warmup_flag_shown_in_report(self):
        bars = make_bars(opens=[100] * 4, closes=[100] * 4)
        result = backtest(bars, make_signals(4, buys={0}, sells={1}), warmup=2)
        assert "EMA warm-up" in format_trades(result.trades)
