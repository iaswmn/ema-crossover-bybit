"""Tests for Bybit kline fetching and data cleaning."""

import asyncio

import httpx
import pytest

from solution import BAR_MS, BybitApiError, clean_klines, fetch_klines

T0 = 1_700_000_040_000  # aligned to a minute boundary


def kline(i: int, o=100.0, h=101.0, low=99.0, c=100.5) -> list[str]:
    """Build one raw Bybit kline row for minute ``i`` after T0."""
    return [str(T0 + i * BAR_MS), str(o), str(h), str(low), str(c), "1.0", "100.0"]


def closed_server_time(last_index: int) -> int:
    """Server time at which bar ``last_index`` is already closed."""
    return T0 + (last_index + 1) * BAR_MS + 1


class TestCleanKlines:
    def test_sorts_newest_first_response_ascending(self):
        rows = [kline(2), kline(1), kline(0)]  # Bybit returns newest first
        df, _ = clean_klines(rows, closed_server_time(2))
        assert list(df["start_ms"]) == [T0, T0 + BAR_MS, T0 + 2 * BAR_MS]

    def test_removes_duplicates_keeping_first_occurrence(self):
        rows = [kline(1, c=100.9), kline(1, c=100.1), kline(0)]
        df, report = clean_klines(rows, closed_server_time(1))
        assert report.duplicates == 1
        assert len(df) == 2
        assert df.iloc[-1]["close"] == pytest.approx(100.9)

    def test_drops_rows_with_infinite_fields_as_unparseable(self):
        inf_time = ["inf", "100", "101", "99", "100.5", "1.0", "100.0"]
        inf_price = kline(1, h=float("inf"))
        df, report = clean_klines([inf_time, inf_price, kline(0)], closed_server_time(1))
        assert report.unparseable == 2
        assert list(df["start_ms"]) == [T0]

    def test_broken_copy_does_not_shadow_valid_duplicate(self):
        broken_copy = kline(1, o=0.0, h=0.0, low=0.0, c=0.0)
        df, report = clean_klines([broken_copy, kline(1), kline(0)], closed_server_time(1))
        assert len(df) == 2
        assert report.broken == 1
        assert report.duplicates == 0

    @pytest.mark.parametrize(
        "bad",
        [
            kline(1, o=0.0),
            kline(1, c=0.0),
            kline(1, h=98.0, low=99.0),  # high < low
            kline(1, o=105.0),  # open above high
            kline(1, c=95.0),  # close below low
            kline(1, low=-1.0),
        ],
    )
    def test_drops_broken_bars(self, bad):
        df, report = clean_klines([kline(2), bad, kline(0)], closed_server_time(2))
        assert report.broken == 1
        assert len(df) == 2

    def test_drops_unparseable_rows(self):
        rows = [kline(2), ["oops"], [str(T0 + BAR_MS), "abc", "1", "1", "1", "1", "1"], kline(0)]
        df, report = clean_klines(rows, closed_server_time(2))
        assert report.unparseable == 2
        assert len(df) == 2

    def test_drops_rows_that_are_not_lists(self):
        rows = [kline(2), None, "1700000100000,1,1,1,1,1,1", {"a": 1}, kline(0)]
        df, report = clean_klines(rows, closed_server_time(2))
        assert report.unparseable == 3
        assert len(df) == 2

    def test_drops_unclosed_current_bar(self):
        rows = [kline(2), kline(1), kline(0)]
        server_ms = T0 + 2 * BAR_MS + 30_000  # bar 2 is still forming
        df, report = clean_klines(rows, server_ms)
        assert report.unclosed == 1
        assert df["start_ms"].max() == T0 + BAR_MS

    def test_reports_gaps_without_filling_them(self):
        rows = [kline(5), kline(4), kline(1), kline(0)]  # minutes 2-3 missing
        df, report = clean_klines(rows, closed_server_time(5))
        assert report.gaps == 1
        assert report.missing_minutes == 2
        assert len(df) == 4

    def test_report_counts_are_consistent(self):
        rows = [kline(3), kline(3), kline(2, o=0.0), kline(1), kline(0)]
        df, report = clean_klines(rows, closed_server_time(3))
        assert report.raw == 5
        assert report.final == len(df) == 3

    def test_does_not_mutate_input(self):
        rows = [kline(1), kline(0)]
        snapshot = [list(r) for r in rows]
        clean_klines(rows, closed_server_time(1))
        assert rows == snapshot


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


class TestFetchKlines:
    def test_returns_rows_and_server_time(self):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.params["category"] == "spot"
            assert request.url.params["symbol"] == "BTCUSDT"
            assert request.url.params["interval"] == "1"
            assert request.url.params["limit"] == "1000"
            body = {"retCode": 0, "retMsg": "OK", "result": {"list": [kline(0)]}, "time": 123}
            return httpx.Response(200, json=body)

        async def go():
            async with _client(handler) as client:
                return await fetch_klines(client)

        rows, server_ms = asyncio.run(go())
        assert rows == [kline(0)]
        assert server_ms == 123

    def test_raises_on_api_error_code(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"retCode": 10001, "retMsg": "bad symbol"})

        async def go():
            async with _client(handler) as client:
                await fetch_klines(client, retries=1)

        with pytest.raises(BybitApiError, match="bad symbol"):
            asyncio.run(go())

    def test_non_json_body_raises_api_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="<html>blocked</html>")

        async def go():
            async with _client(handler) as client:
                await fetch_klines(client, retries=1)

        with pytest.raises(BybitApiError, match="not valid JSON"):
            asyncio.run(go())

    @pytest.mark.parametrize(
        "body",
        [
            [1, 2, 3],
            {"retCode": 0, "result": {"list": None}, "time": 1},
            {"retCode": 0, "result": {"list": "oops"}, "time": 1},
            {"retCode": 0, "result": None, "time": 1},
            {"retCode": 0, "result": {"list": []}},
        ],
    )
    def test_malformed_payload_raises_api_error(self, body):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=body)

        async def go():
            async with _client(handler) as client:
                await fetch_klines(client, retries=1)

        with pytest.raises(BybitApiError):
            asyncio.run(go())

    def test_retries_transient_http_errors(self):
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] < 2:
                return httpx.Response(503)
            return httpx.Response(200, json={"retCode": 0, "result": {"list": []}, "time": 1})

        async def go():
            async with _client(handler) as client:
                return await fetch_klines(client, retries=3, backoff_s=0)

        rows, _ = asyncio.run(go())
        assert rows == []
        assert calls["n"] == 2
