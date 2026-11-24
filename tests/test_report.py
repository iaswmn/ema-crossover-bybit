"""End-to-end run on synthetic data and console formatting."""

import asyncio

import numpy as np

import solution
from solution import BAR_MS

T0 = 1_700_000_040_000


def synthetic_rows(n: int) -> list[list[str]]:
    """Newest-first rows with a down-up-down price path, one duplicate and one broken bar."""
    prices = np.concatenate([np.linspace(100, 90, n // 3), np.linspace(90, 110, n // 3), np.linspace(110, 95, n - 2 * (n // 3))])
    rows = [
        [str(T0 + i * BAR_MS), f"{p:.4f}", f"{p + 0.5:.4f}", f"{p - 0.5:.4f}", f"{p:.4f}", "1", "1"]
        for i, p in enumerate(prices)
    ]
    rows.append(list(rows[10]))  # duplicate
    rows[20] = [rows[20][0], "0", "0", "0", "0", "1", "1"]  # broken
    return list(reversed(rows))


def test_run_prints_all_sections(monkeypatch, capsys):
    n = 150
    server_ms = T0 + n * BAR_MS + 1

    async def fake_fetch(client, *args, **kwargs):
        return synthetic_rows(n), server_ms

    monkeypatch.setattr(solution, "fetch_klines", fake_fetch)
    assert asyncio.run(solution.run()) == 0

    out = capsys.readouterr().out
    assert "Duplicates removed:     1" in out
    assert "Broken bars removed:    1" in out
    assert "=== Closed trades ===" in out
    assert "Winrate:" in out
    assert "Total net PnL:" in out


def test_main_reports_api_failure(monkeypatch, capsys):
    async def failing_fetch(client, *args, **kwargs):
        raise solution.BybitApiError("Bybit error 10001: bad symbol")

    monkeypatch.setattr(solution, "fetch_klines", failing_fetch)
    assert solution.main() == 1
    assert "bad symbol" in capsys.readouterr().out


def test_slippage_warning_is_printed():
    trade = solution.Trade(
        entry_time=solution.pd.Timestamp("2026-01-01", tz="UTC"),
        entry_price=101.0,
        signal_entry_close=100.0,
        exit_time=solution.pd.Timestamp("2026-01-01 00:05", tz="UTC"),
        exit_price=100.0,
        signal_exit_close=100.0,
        net_pnl_usdt=-1.0,
        net_pnl_pct=-0.1,
        entry_slippage_pct=1.0,
        exit_slippage_pct=0.0,
    )
    warnings = solution.format_slippage_warnings([trade])
    assert len(warnings) == 1
    assert "entry slippage +1.00%" in warnings[0]
    assert "SLIPPAGE" in solution.format_trades([trade])
