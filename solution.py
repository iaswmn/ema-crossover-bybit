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
            return _parse_kline_payload(_decode_json(response))
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


def _decode_json(response: httpx.Response) -> object:
    """Decode the body; a 200 with an HTML page (e.g. a WAF block) is an API error."""
    try:
        return response.json()
    except ValueError as exc:
        snippet = response.text[:80].replace("\n", " ")
        raise BybitApiError(f"Bybit response is not valid JSON: {snippet!r}") from exc


def _parse_kline_payload(payload: object) -> tuple[list, int]:
    """Validate the envelope; individual rows are validated by ``clean_klines``."""
    if not isinstance(payload, dict):
        raise BybitApiError(f"Unexpected Bybit payload type: {type(payload).__name__}")
    if payload.get("retCode") != 0:
        raise BybitApiError(f"Bybit error {payload.get('retCode')}: {payload.get('retMsg')}")
    try:
        rows = payload["result"]["list"]
        server_ms = int(payload["time"])
    except (KeyError, TypeError, ValueError) as exc:
        raise BybitApiError(f"Unexpected Bybit payload: {exc!r}") from exc
    if not isinstance(rows, list):
        raise BybitApiError(f"Unexpected Bybit payload: result.list is {type(rows).__name__}")
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
    * Unparseable rows (not a list, wrong length, non-numeric fields) are
      dropped: there is no way to trust or repair them.
    * Broken bars are dropped: non-positive prices, high < low, open/close
      outside [low, high], or a start time not on a minute boundary. This runs
      before de-duplication so a broken copy cannot replace a valid one.
    * Duplicates by start time are dropped keeping the first occurrence. Copies
      of a closed minute within one response are expected to be identical; if
      they are not, there is no reliable way to choose, so the order is kept.
    * The still-forming bar (start + 1 min > server time) is dropped: its close
      is not final, and a signal on it would repaint (look-ahead).
    * Rows are sorted ascending: Bybit returns them newest first, while EMA and
      crossover logic needs chronological order.
    * Missing minutes are reported but NOT filled. TradingView does not draw
      bars for minutes without trades, so synthetic bars would make our EMA
      diverge from the chart. Execution simply happens on the next real bar.
    """
    raw = len(rows)
    well_formed = [
        list(r[: len(KLINE_COLUMNS)])
        for r in rows
        if isinstance(r, (list, tuple)) and len(r) >= len(KLINE_COLUMNS)
    ]
    df = pd.DataFrame(well_formed, columns=KLINE_COLUMNS)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["start_ms", *PRICE_COLUMNS])
    unparseable = raw - len(df)
    df = df.astype({"start_ms": "int64"})

    # Broken bars go first, so a broken copy of a minute can never win the
    # de-duplication against a valid copy of the same minute.
    is_broken = (
        (df[PRICE_COLUMNS] <= 0).any(axis=1)
        | (df["high"] < df["low"])
        | ~df["open"].between(df["low"], df["high"])
        | ~df["close"].between(df["low"], df["high"])
        | (df["start_ms"] % BAR_MS != 0)
    )
    broken = int(is_broken.sum())
    df = df[~is_broken]

    before = len(df)
    df = df.drop_duplicates(subset="start_ms", keep="first")
    duplicates = before - len(df)

    before = len(df)
    df = df[df["start_ms"] + BAR_MS <= server_ms]
    unclosed = before - len(df)

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


# --- Backtest -----------------------------------------------------------------
POSITION_USDT = 1000.0
ENTRY_FEE_RATE = 0.001  # 0.1% of the entry notional
EXIT_FEE_RATE = 0.0018  # 0.18% of the exit notional
SLIPPAGE_WARN_PCT = 0.5


@dataclass(frozen=True)
class Trade:
    entry_time: pd.Timestamp
    entry_price: float
    signal_entry_close: float
    exit_time: pd.Timestamp
    exit_price: float
    signal_exit_close: float | None  # None when force-closed at the end of data
    net_pnl_usdt: float
    net_pnl_pct: float
    entry_slippage_pct: float
    exit_slippage_pct: float

    @property
    def forced_close(self) -> bool:
        return self.signal_exit_close is None

    @property
    def has_slippage(self) -> bool:
        return max(abs(self.entry_slippage_pct), abs(self.exit_slippage_pct)) > SLIPPAGE_WARN_PCT


@dataclass(frozen=True)
class BacktestResult:
    trades: list[Trade]
    unexecuted_signals: int  # signals on the last bar: there is no next open


@dataclass(frozen=True)
class _OpenPosition:
    entry_time: pd.Timestamp
    entry_price: float
    signal_close: float


def _slippage_pct(executed: float, signal_close: float) -> float:
    return (executed / signal_close - 1.0) * 100.0


def _net_pnl(entry_price: float, exit_price: float) -> tuple[float, float]:
    """Net PnL in USDT and % of the position size, after both fees.

    The position size (1000 USDT) buys ``1000 / entry_price`` coins; the entry
    fee is charged on top of it, the exit fee on the actual exit notional.
    """
    qty = POSITION_USDT / entry_price
    exit_value = qty * exit_price
    fees = POSITION_USDT * ENTRY_FEE_RATE + exit_value * EXIT_FEE_RATE
    pnl = exit_value - POSITION_USDT - fees
    return pnl, pnl / POSITION_USDT * 100.0


def _close(position: _OpenPosition, exit_time, exit_price: float, signal_close: float | None) -> Trade:
    pnl_usdt, pnl_pct = _net_pnl(position.entry_price, exit_price)
    return Trade(
        entry_time=position.entry_time,
        entry_price=position.entry_price,
        signal_entry_close=position.signal_close,
        exit_time=exit_time,
        exit_price=exit_price,
        signal_exit_close=signal_close,
        net_pnl_usdt=pnl_usdt,
        net_pnl_pct=pnl_pct,
        entry_slippage_pct=_slippage_pct(position.entry_price, position.signal_close),
        exit_slippage_pct=0.0 if signal_close is None else _slippage_pct(exit_price, signal_close),
    )


def backtest(bars: pd.DataFrame, signals: pd.DataFrame) -> BacktestResult:
    """Long-only spot backtest without look-ahead.

    A signal is known only after bar T closes, so the order fills at the OPEN
    of bar T+1 (the next real bar if there is a gap). A signal on the very last
    bar cannot be executed and is counted separately. A position still open at
    the end of data is force-closed at the last close.
    """
    times = bars["time"].tolist()
    opens = bars["open"].to_numpy(dtype=float)
    closes = bars["close"].to_numpy(dtype=float)
    buys = signals["buy"].to_numpy(dtype=bool)
    sells = signals["sell"].to_numpy(dtype=bool)
    last = len(bars) - 1

    trades: list[Trade] = []
    position: _OpenPosition | None = None
    unexecuted = 0
    for i in range(len(bars)):
        wants_entry = position is None and buys[i]
        wants_exit = position is not None and sells[i]
        if not (wants_entry or wants_exit):
            continue
        if i == last:
            unexecuted += 1
            continue
        if wants_entry:
            position = _OpenPosition(times[i + 1], opens[i + 1], closes[i])
        else:
            trades = [*trades, _close(position, times[i + 1], opens[i + 1], closes[i])]
            position = None

    if position is not None:
        close_time = times[last] + pd.Timedelta(milliseconds=BAR_MS)
        trades = [*trades, _close(position, close_time, closes[last], None)]
    return BacktestResult(trades=trades, unexecuted_signals=unexecuted)


@dataclass(frozen=True)
class Stats:
    trades: int
    wins: int
    winrate_pct: float
    total_pnl_usdt: float


def summarize(trades: list[Trade]) -> Stats:
    wins = sum(1 for t in trades if t.net_pnl_usdt > 0)
    return Stats(
        trades=len(trades),
        wins=wins,
        winrate_pct=wins / len(trades) * 100.0 if trades else 0.0,
        total_pnl_usdt=sum(t.net_pnl_usdt for t in trades),
    )


# --- Console report -----------------------------------------------------------
def _fmt_time(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m-%d %H:%M UTC")


def format_cleaning(report: CleaningReport) -> str:
    return "\n".join(
        [
            "=== Data cleaning ===",
            f"Received bars:          {report.raw}",
            f"Unparseable removed:    {report.unparseable}",
            f"Duplicates removed:     {report.duplicates}",
            f"Broken bars removed:    {report.broken}",
            f"Unclosed bar removed:   {report.unclosed}",
            f"Gaps (not filled):      {report.gaps} ({report.missing_minutes} missing minutes)",
            f"Bars used:              {report.final}",
        ]
    )


def format_slippage_warnings(trades: list[Trade]) -> list[str]:
    warnings = []
    for t in trades:
        if abs(t.entry_slippage_pct) > SLIPPAGE_WARN_PCT:
            warnings.append(
                f"WARNING: entry slippage {t.entry_slippage_pct:+.2f}% at {_fmt_time(t.entry_time)} "
                f"(signal close {t.signal_entry_close:.2f}, filled {t.entry_price:.2f})"
            )
        if abs(t.exit_slippage_pct) > SLIPPAGE_WARN_PCT:
            warnings.append(
                f"WARNING: exit slippage {t.exit_slippage_pct:+.2f}% at {_fmt_time(t.exit_time)} "
                f"(signal close {t.signal_exit_close:.2f}, filled {t.exit_price:.2f})"
            )
    return warnings


def format_trades(trades: list[Trade]) -> str:
    lines = ["=== Closed trades ==="]
    if not trades:
        return "\n".join([*lines, "No trades."])
    header = f"{'#':>3}  {'Entry time':<20} {'Entry':>10}  {'Exit time':<20} {'Exit':>10} {'PnL %':>8} {'PnL USDT':>9}  Notes"
    lines.append(header)
    for n, t in enumerate(trades, start=1):
        notes = []
        if t.has_slippage:
            notes.append("SLIPPAGE")
        if t.forced_close:
            notes.append("forced close at last price")
        lines.append(
            f"{n:>3}  {_fmt_time(t.entry_time):<20} {t.entry_price:>10.2f}  "
            f"{_fmt_time(t.exit_time):<20} {t.exit_price:>10.2f} "
            f"{t.net_pnl_pct:>+7.3f}% {t.net_pnl_usdt:>+9.2f}  {', '.join(notes)}"
        )
    return "\n".join(lines)


def format_stats(stats: Stats, unexecuted_signals: int) -> str:
    lines = [
        "=== Summary ===",
        f"Trades:        {stats.trades}",
        f"Winrate:       {stats.winrate_pct:.1f}% ({stats.wins}/{stats.trades})",
        f"Total net PnL: {stats.total_pnl_usdt:+.2f} USDT",
    ]
    if unexecuted_signals:
        lines.append(f"Signals on the last bar not executed (no next open): {unexecuted_signals}")
    return "\n".join(lines)


# --- Entry point --------------------------------------------------------------
async def run() -> int:
    async with httpx.AsyncClient() as client:
        rows, server_ms = await fetch_klines(client)

    bars, cleaning = clean_klines(rows, server_ms)
    print(format_cleaning(cleaning))
    if len(bars) <= SLOW_LENGTH:
        print(f"\nNot enough clean bars for EMA({SLOW_LENGTH}): {len(bars)}")
        return 1

    signals = compute_signals(bars)
    result = backtest(bars, signals)

    warnings = format_slippage_warnings(result.trades)
    if warnings:
        print()
        print("\n".join(warnings))
    print()
    print(format_trades(result.trades))
    print()
    print(format_stats(summarize(result.trades), result.unexecuted_signals))
    return 0


def main() -> int:
    try:
        return asyncio.run(run())
    except (httpx.HTTPError, BybitApiError) as exc:
        print(f"Failed to load data from Bybit: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
