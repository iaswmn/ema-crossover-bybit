"""EMA Crossover strategy ported from Pine Script v5 to async Python.

Fetches the last 1000 one-minute BTCUSDT spot klines from Bybit, cleans them,
reproduces TradingView's ``ta.ema`` / ``ta.crossover`` / ``ta.crossunder`` and
backtests a long-only strategy with next-bar-open execution, fees and a
slippage check.

Run: ``python solution.py``
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import httpx
import numpy as np
import pandas as pd

# --- Exchange / data settings -------------------------------------------------
BYBIT_KLINE_URL = "https://api.bybit.com/v5/market/kline"
CATEGORY = "spot"
SYMBOL = "BTCUSDT"
INTERVAL = "1"  # minutes
LIMIT = 1000  # max allowed by Bybit for a single request
BAR_MS = 60_000
REQUEST_TIMEOUT_S = 10.0
RETRIES = 3
BACKOFF_S = 1.0

KLINE_COLUMNS = ["start_ms", "open", "high", "low", "close", "volume", "turnover"]
PRICE_COLUMNS = ["open", "high", "low", "close"]


class BybitApiError(RuntimeError):
    """Bybit answered with a non-zero ``retCode`` or an unexpected payload."""


# --- Data fetching ------------------------------------------------------------
async def fetch_klines(
    client: httpx.AsyncClient,
    symbol: str = SYMBOL,
    retries: int = RETRIES,
    backoff_s: float = BACKOFF_S,
) -> tuple[list[list[str]], int]:
    """Fetch raw klines and the exchange server time (ms) in one request.

    Transient failures (network errors, 5xx, 429) are retried with exponential
    backoff; an API-level error (``retCode != 0``) fails immediately.
    """
    params = {"category": CATEGORY, "symbol": symbol, "interval": INTERVAL, "limit": LIMIT}
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = await client.get(BYBIT_KLINE_URL, params=params, timeout=REQUEST_TIMEOUT_S)
            response.raise_for_status()
            return _parse_kline_payload(response.json())
        except httpx.HTTPStatusError as exc:
            if not _is_retryable_status(exc.response.status_code):
                raise
            last_error = exc
        except httpx.TransportError as exc:
            last_error = exc
        if attempt < retries - 1:
            await asyncio.sleep(backoff_s * 2**attempt)
    raise BybitApiError(f"Bybit request failed after {retries} attempts: {last_error}")


def _is_retryable_status(status: int) -> bool:
    return status == 429 or status >= 500


def _parse_kline_payload(payload: dict) -> tuple[list[list[str]], int]:
    if payload.get("retCode") != 0:
        raise BybitApiError(f"Bybit error {payload.get('retCode')}: {payload.get('retMsg')}")
    try:
        rows = payload["result"]["list"]
        server_ms = int(payload["time"])
    except (KeyError, TypeError, ValueError) as exc:
        raise BybitApiError(f"Unexpected Bybit payload: {exc!r}") from exc
    return rows, server_ms


# --- Data cleaning ------------------------------------------------------------
@dataclass(frozen=True)
class CleaningReport:
    raw: int
    unparseable: int
    duplicates: int
    unclosed: int
    broken: int
    gaps: int
    missing_minutes: int
    final: int


def clean_klines(rows: list[list[str]], server_ms: int) -> tuple[pd.DataFrame, CleaningReport]:
    """Turn raw Bybit rows into a clean, ascending OHLC frame.

    What is done and why:
    * Unparseable rows (wrong length, non-numeric fields) are dropped: there is
      no way to trust or repair them.
    * Duplicates by start time are dropped keeping the FIRST occurrence. Bybit
      lists newest first, so the first copy is the latest snapshot of the bar.
    * The still-forming bar (start + 1 min > server time) is dropped: its close
      is not final, and a signal on it would repaint (look-ahead).
    * Broken bars are dropped: non-positive prices, high < low, open/close
      outside [low, high], or a start time not on a minute boundary.
    * Rows are sorted ascending: Bybit returns them newest first, while EMA and
      crossover logic needs chronological order.
    * Missing minutes are reported but NOT filled. TradingView does not draw
      bars for minutes without trades, so synthetic bars would make our EMA
      diverge from the chart. Execution simply happens on the next real bar.
    """
    raw = len(rows)
    well_formed = [list(r[: len(KLINE_COLUMNS)]) for r in rows if len(r) >= len(KLINE_COLUMNS)]
    df = pd.DataFrame(well_formed, columns=KLINE_COLUMNS)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["start_ms", *PRICE_COLUMNS])
    unparseable = raw - len(df)
    df = df.astype({"start_ms": "int64"})

    before = len(df)
    df = df.drop_duplicates(subset="start_ms", keep="first")
    duplicates = before - len(df)

    before = len(df)
    df = df[df["start_ms"] + BAR_MS <= server_ms]
    unclosed = before - len(df)

    is_broken = (
        (df[PRICE_COLUMNS] <= 0).any(axis=1)
        | (df["high"] < df["low"])
        | ~df["open"].between(df["low"], df["high"])
        | ~df["close"].between(df["low"], df["high"])
        | (df["start_ms"] % BAR_MS != 0)
    )
    broken = int(is_broken.sum())
    df = df[~is_broken]

    df = df.sort_values("start_ms").reset_index(drop=True)
    df["time"] = pd.to_datetime(df["start_ms"], unit="ms", utc=True)

    step = df["start_ms"].diff().dropna()
    gap_steps = step[step > BAR_MS]
    gaps = len(gap_steps)
    missing_minutes = int(((gap_steps // BAR_MS) - 1).sum())

    report = CleaningReport(
        raw=raw,
        unparseable=unparseable,
        duplicates=duplicates,
        unclosed=unclosed,
        broken=broken,
        gaps=gaps,
        missing_minutes=missing_minutes,
        final=len(df),
    )
    return df, report


# --- Indicators (Pine Script v5 semantics) -----------------------------------
FAST_LENGTH = 9
SLOW_LENGTH = 21


def pine_ema(src: np.ndarray, length: int) -> np.ndarray:
    """Reproduce Pine's ``ta.ema(src, length)``.

    * alpha = 2 / (length + 1);
    * the first ``length - 1`` values are ``na``;
    * the first value is seeded with the SMA of the first ``length`` sources,
      then ``ema = alpha * src + (1 - alpha) * ema[1]``.

    ``pandas.ewm(adjust=False)`` is NOT equivalent: it seeds with the first
    source value and has no ``na`` warm-up, which shifts early crossovers.
    """
    out = np.full(len(src), np.nan)
    if length <= 0 or len(src) < length:
        return out
    alpha = 2.0 / (length + 1)
    out[length - 1] = src[:length].mean()
    for i in range(length, len(src)):
        out[i] = alpha * src[i] + (1.0 - alpha) * out[i - 1]
    return out


def _previous(values: np.ndarray) -> np.ndarray:
    """Pine's ``x[1]``: the series shifted by one bar, ``na`` on the first bar."""
    return np.concatenate([[np.nan], values[:-1]])


def crossover(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pine's ``ta.crossover``: ``a > b`` now and ``a <= b`` on the previous bar.

    Comparisons with ``na`` are false, so there is no signal during warm-up.
    """
    with np.errstate(invalid="ignore"):
        return (a > b) & (_previous(a) <= _previous(b))


def crossunder(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pine's ``ta.crossunder``: ``a < b`` now and ``a >= b`` on the previous bar."""
    with np.errstate(invalid="ignore"):
        return (a < b) & (_previous(a) >= _previous(b))


def compute_signals(df: pd.DataFrame, fast: int = FAST_LENGTH, slow: int = SLOW_LENGTH) -> pd.DataFrame:
    """Return a new frame with both EMAs and buy / sell signals per bar close."""
    close = df["close"].to_numpy(dtype=float)
    ema_fast = pine_ema(close, fast)
    ema_slow = pine_ema(close, slow)
    return pd.DataFrame(
        {
            "ema_fast": ema_fast,
            "ema_slow": ema_slow,
            "buy": crossover(ema_fast, ema_slow),
            "sell": crossunder(ema_fast, ema_slow),
        },
        index=df.index,
    )
