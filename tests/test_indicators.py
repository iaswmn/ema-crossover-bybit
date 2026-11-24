"""Tests for the Pine-compatible EMA and crossover signals."""

import numpy as np
import pandas as pd
import pytest

from solution import compute_signals, crossover, crossunder, pine_ema


class TestPineEma:
    def test_is_na_until_length_bars_then_seeded_with_sma(self):
        src = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ema = pine_ema(src, 3)
        assert np.isnan(ema[:2]).all()
        assert ema[2] == pytest.approx(2.0)  # SMA of 1, 2, 3

    def test_recursion_uses_alpha_two_over_length_plus_one(self):
        src = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ema = pine_ema(src, 3)
        alpha = 2 / 4
        expected_3 = alpha * 4.0 + (1 - alpha) * 2.0
        assert ema[3] == pytest.approx(expected_3)
        assert ema[4] == pytest.approx(alpha * 5.0 + (1 - alpha) * expected_3)

    def test_all_na_when_not_enough_data(self):
        assert np.isnan(pine_ema(np.array([1.0, 2.0]), 3)).all()

    def test_constant_series_gives_constant_ema(self):
        ema = pine_ema(np.full(50, 7.0), 9)
        assert np.allclose(ema[8:], 7.0)


class TestCross:
    def test_crossover_fires_only_on_the_crossing_bar(self):
        a = np.array([1.0, 2.0, 4.0, 5.0])
        b = np.array([3.0, 3.0, 3.0, 3.0])
        assert list(crossover(a, b)) == [False, False, True, False]
        assert not crossunder(a, b).any()

    def test_crossunder_fires_only_on_the_crossing_bar(self):
        a = np.array([5.0, 4.0, 2.0, 1.0])
        b = np.array([3.0, 3.0, 3.0, 3.0])
        assert list(crossunder(a, b)) == [False, False, True, False]

    def test_touch_then_cross_counts_as_crossover(self):
        # previous bar equal, current above -> crossover (x[1] <= y[1])
        a = np.array([2.0, 3.0, 4.0])
        b = np.array([3.0, 3.0, 3.0])
        assert list(crossover(a, b)) == [False, False, True]

    def test_no_signal_when_previous_value_is_na(self):
        a = np.array([np.nan, 4.0])
        b = np.array([3.0, 3.0])
        assert not crossover(a, b).any()


class TestComputeSignals:
    def test_detects_up_and_down_cross_on_trend_reversal(self):
        close = np.concatenate([np.linspace(100, 90, 40), np.linspace(90, 110, 40), np.linspace(110, 95, 40)])
        df = pd.DataFrame({"close": close})
        signals = compute_signals(df, fast=9, slow=21)
        assert signals["buy"].sum() >= 1
        assert signals["sell"].sum() >= 1
        first_buy = signals.index[signals["buy"]][0]
        first_sell_after = signals.index[signals["sell"] & (signals.index > first_buy)][0]
        assert 40 <= first_buy < 80
        assert first_sell_after >= 80

    def test_does_not_mutate_input_frame(self):
        df = pd.DataFrame({"close": np.linspace(1, 2, 30)})
        columns = list(df.columns)
        compute_signals(df, fast=3, slow=5)
        assert list(df.columns) == columns
