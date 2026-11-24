"""Compare solution.py against a TradingView "Export chart data" CSV.

Usage: python tools/compare_tradingview.py path/to/export.csv

The CSV must come from a BYBIT:BTCUSDT 1m chart with the strategy script that
additionally plots ``emaFast``, ``emaSlow``, ``buy`` and ``sell`` (see README).
Time format: UNIX timestamp (ISO also works).

Three checks:
1. EMA implementation: our ``pine_ema`` on TradingView's own closes vs the
   exported EMA columns (also confirms the SMA seed and the na warm-up).
2. Cross definition: our ``crossover`` / ``crossunder`` on TradingView's EMAs
   vs the exported buy / sell flags.
3. Full pipeline: the same 1000-bar window fetched from Bybit REST, cleaned and
   processed exactly like solution.py, vs the chart, bar by bar.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import solution as sol

EPS = 1e-6


def load_tradingview(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    # Case-insensitive lookup; exact lowercase names win, so the "buy"/"sell"
    # plots take precedence over the "BUY"/"SELL" plotshape columns.
    by_lower: dict[str, str] = {}
    for col in raw.columns:
        by_lower.setdefault(col.strip().lower(), col)
    for col in raw.columns:
        if col.strip() == col.strip().lower():
            by_lower[col.strip()] = col

    def pick(name: str) -> pd.Series:
        if name not in by_lower:
            raise SystemExit(f"Column '{name}' not found in CSV. Columns: {list(raw.columns)}")
        return raw[by_lower[name]]

    time = pick("time")
    if pd.api.types.is_numeric_dtype(time):
        start_ms = (time.astype("int64") * 1000).to_numpy()
    else:
        start_ms = (pd.to_datetime(time, utc=True).astype("int64") // 1_000_000).to_numpy()
    return pd.DataFrame(
        {
            "start_ms": start_ms,
            "open": pick("open").astype(float),
            "high": pick("high").astype(float),
            "low": pick("low").astype(float),
            "close": pick("close").astype(float),
            "ema_fast": pick("emafast").astype(float),
            "ema_slow": pick("emaslow").astype(float),
            "buy": pick("buy").fillna(0).astype(float) > 0,
            "sell": pick("sell").fillna(0).astype(float) > 0,
        }
    )


def _first_value_seed_ema(src: np.ndarray, length: int) -> np.ndarray:
    """Alternative hypothesis: seed with the first value, no na warm-up."""
    return pd.Series(src).ewm(alpha=2 / (length + 1), adjust=False).mean().to_numpy()


def check_ema(tv: pd.DataFrame) -> list[str]:
    close = tv["close"].to_numpy()
    lines = ["=== 1. EMA implementation on TradingView closes ==="]
    for col, length in (("ema_fast", sol.FAST_LENGTH), ("ema_slow", sol.SLOW_LENGTH)):
        tv_ema = tv[col].to_numpy()
        ours = sol.pine_ema(close, length)
        alt = _first_value_seed_ema(close, length)
        tv_na = int(np.isnan(tv_ema[: length * 2]).sum())
        both = ~np.isnan(tv_ema) & ~np.isnan(ours)
        diff = np.abs(ours[both] - tv_ema[both])
        alt_diff = np.abs(alt[both] - tv_ema[both])
        lines.append(
            f"EMA({length}): TradingView na at start = {tv_na} (SMA seed expects {length - 1}); "
            f"max |ours - TV| = {diff.max():.2e}; first-value seed would give {alt_diff.max():.2e}"
        )
    return lines


def check_cross(tv: pd.DataFrame) -> list[str]:
    fast, slow = tv["ema_fast"].to_numpy(), tv["ema_slow"].to_numpy()
    lines = ["=== 2. Cross definition on TradingView EMAs ==="]
    for name, func in (("buy", sol.crossover), ("sell", sol.crossunder)):
        ours = func(fast, slow)
        theirs = tv[name].to_numpy()
        lines.append(
            f"{name}: TradingView {int(theirs.sum())}, ours {int(ours.sum())}, "
            f"mismatched bars {int((ours != theirs).sum())}"
        )
    return lines


async def _fetch_window(end_ms: int) -> tuple[list, int]:
    async with httpx.AsyncClient() as client:
        return await sol.fetch_klines(client, end_ms=end_ms)


def check_pipeline(tv: pd.DataFrame) -> list[str]:
    # The last exported bar may still be forming, so the window ends one bar earlier.
    end_ms = int(tv["start_ms"].iloc[-1]) - sol.BAR_MS
    rows, server_ms = asyncio.run(_fetch_window(end_ms))
    bars, report = sol.clean_klines(rows, server_ms)
    signals = sol.compute_signals(bars)
    warmup = sol.warmup_bars(sol.SLOW_LENGTH)
    ours = pd.concat([bars[["start_ms", "open", "high", "low", "close"]], signals], axis=1)
    ours["idx"] = np.arange(len(ours))

    merged = ours.merge(tv, on="start_ms", suffixes=("", "_tv"))
    lines = [
        "=== 3. Full pipeline: Bybit REST window vs chart ===",
        f"Bybit bars after cleaning: {report.final} "
        f"(duplicates {report.duplicates}, broken {report.broken}, gaps {report.gaps})",
        f"Bars present on both sides: {len(merged)}; only in Bybit: {len(ours) - len(merged)}",
    ]
    ohlc_diff = sum(int((np.abs(merged[c] - merged[f"{c}_tv"]) > EPS).sum()) for c in sol.PRICE_COLUMNS)
    lines.append(f"OHLC values differing between Bybit REST and TradingView: {ohlc_diff}")

    settled = merged["idx"] >= warmup
    ema_diff = np.abs(merged["ema_slow"] - merged["ema_slow_tv"])
    lines.append(
        f"EMA({sol.SLOW_LENGTH}) max |ours - TV|: warm-up {ema_diff[~settled].max():.2e}, "
        f"after warm-up {ema_diff[settled].max():.2e}"
    )
    for name in ("buy", "sell"):
        a, b = merged[name].to_numpy(), merged[f"{name}_tv"].to_numpy()
        for zone, mask in (("warm-up", ~settled.to_numpy()), ("after warm-up", settled.to_numpy())):
            lines.append(
                f"{name} [{zone}]: matched {int((a & b & mask).sum())}, "
                f"only ours {int((a & ~b & mask).sum())}, only TV {int((~a & b & mask).sum())}"
            )
        for _, row in merged[(merged[name] != merged[f"{name}_tv"]) & settled].iterrows():
            ts = pd.to_datetime(row["start_ms"], unit="ms", utc=True)
            lines.append(f"  mismatch after warm-up: {name} at {ts:%Y-%m-%d %H:%M} UTC")
    return lines


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    tv = load_tradingview(Path(sys.argv[1]))
    print(f"TradingView rows: {len(tv)}\n")
    for block in (check_ema(tv), check_cross(tv), check_pipeline(tv)):
        print("\n".join(block), end="\n\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
