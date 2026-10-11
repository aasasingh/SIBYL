"""Ledger + backfill-parser + resolution-query tests.

Parser fixtures mirror the payload shapes probed live on 2026-07-21 (Kalshi
demo events/candlesticks with *_dollars strings; gamma markets with
JSON-encoded string lists; data-api tape rows).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from sibyl.semantic.classify import (
    Domain,
    domain_from_kalshi_category,
    domain_from_polymarket_tags,
)
from sibyl.store.backfill import (
    parse_gamma_event,
    parse_gamma_market,
    parse_kalshi_candles,
    parse_kalshi_event,
    parse_ts,
    tape_to_hourly_bars,
)
from sibyl.store.ledger import Ledger, MarketRecord, PriceRecord
from sibyl.store.resolutions import calibration_rows, horizon_bucket, resolved_market_counts
from sibyl.venues.normalise import Venue

NOW = datetime(2026, 7, 21, 12, 0, tzinfo=UTC)
OLD = datetime(2020, 1, 1, tzinfo=UTC)


def _mk(
    market_id: str, *, resolution: str | None = "yes", domain: Domain = Domain.POLITICS
) -> MarketRecord:
    return MarketRecord(
        venue=Venue.KALSHI,
        market_id=market_id,
        title=f"market {market_id}",
        domain=domain,
        close_time=NOW,
        resolution=resolution,
    )


def _px(market_id: str, hours_before_close: int, price: str) -> PriceRecord:
    return PriceRecord(
        venue=Venue.KALSHI,
        market_id=market_id,
        ts=NOW - timedelta(hours=hours_before_close),
        price=Decimal(price),
        source="kalshi_candle",
    )


class TestLedger:
    def test_market_write_dedupes(self, tmp_path: Path) -> None:
        ledger = Ledger(tmp_path)
        assert ledger.write_markets([_mk("A"), _mk("B"), _mk("A")]) == 2
        assert ledger.write_markets([_mk("A"), _mk("C")]) == 1
        assert len(ledger.market_ids()) == 3

    def test_summary_joins_prices(self, tmp_path: Path) -> None:
        ledger = Ledger(tmp_path)
        ledger.write_markets([_mk("A")])
        ledger.write_prices([_px("A", 5, "0.60"), _px("A", 4, "0.65")])
        rows = ledger.summary()
        assert rows == [("kalshi", "politics", "yes", 1, 2)]

    def test_decimal_prices_survive_round_trip(self, tmp_path: Path) -> None:
        ledger = Ledger(tmp_path)
        ledger.write_markets([_mk("A")])
        ledger.write_prices([_px("A", 5, "0.6153")])
        with ledger.connect() as con:
            (value,) = con.execute("SELECT price FROM prices").fetchone()  # type: ignore[misc]
        assert value == Decimal("0.6153")

    def test_naive_datetime_rejected(self, tmp_path: Path) -> None:
        ledger = Ledger(tmp_path)
        record = MarketRecord(
            venue=Venue.KALSHI,
            market_id="X",
            title="x",
            domain=Domain.OTHER,
            close_time=datetime(2026, 7, 21),  # noqa: DTZ001 — the point of the test
        )
        with pytest.raises(ValueError, match="naive"):
            ledger.write_markets([record])


KALSHI_EVENT: dict[str, Any] = {
    "event_ticker": "KXTEST-26",
    "series_ticker": "KXTEST",
    "category": "Elections",
    "mutually_exclusive": True,
    "markets": [
        {
            "ticker": "KXTEST-26-A",
            "title": "Candidate A wins",
            "result": "yes",
            "open_time": "2026-05-01T00:00:00Z",
            "close_time": "2026-07-01T00:00:00Z",
            "settlement_ts": "2026-07-01T02:00:00Z",
        },
        {
            "ticker": "KXTEST-26-SCALAR",
            "title": "scalar leg",
            "result": "scalar",
            "close_time": "2026-07-01T00:00:00Z",
        },
        {
            "ticker": "KXTEST-22-OLD",
            "title": "too old",
            "result": "no",
            "close_time": "2022-01-01T00:00:00Z",
        },
    ],
}


class TestKalshiParsers:
    def test_event_parsing_filters_and_labels(self) -> None:
        records = parse_kalshi_event(KALSHI_EVENT, min_close=OLD.replace(year=2026))
        assert [r.market_id for r in records] == ["KXTEST-26-A"]
        rec = records[0]
        assert rec.domain is Domain.POLITICS
        assert rec.resolution == "yes"
        assert rec.series_id == "KXTEST"
        assert rec.neg_risk is True
        assert rec.settled_time == datetime(2026, 7, 1, 2, tzinfo=UTC)

    def test_candle_parsing_dollars_strings(self) -> None:
        payload = {
            "candlesticks": [
                {
                    "end_period_ts": 1773540000,
                    "price": {"close_dollars": "0.7000", "open_dollars": "0.6800"},
                    "yes_bid": {"close_dollars": "0.4600"},
                    "yes_ask": {"close_dollars": "0.9800"},
                    "volume_fp": "12.00",
                },
                {  # no trades this hour: price absent, bid/ask retained
                    "end_period_ts": 1773543600,
                    "price": {"previous_dollars": "0.7000"},
                    "yes_bid": {"close_dollars": "0.4500"},
                    "yes_ask": {"close_dollars": "0.9700"},
                    "volume_fp": "0.00",
                },
            ]
        }
        rows = parse_kalshi_candles(payload, market_id="M")
        assert rows[0].price == Decimal("0.7000")
        assert rows[0].volume == Decimal("12.00")
        assert rows[1].price is None
        assert rows[1].bid == Decimal("0.4500")
        assert rows[1].ts == datetime.fromtimestamp(1773543600, tz=UTC)


GAMMA_MARKET: dict[str, Any] = {
    "id": "517427",
    "question": "Will X happen by July?",
    "conditionId": "0xabc",
    "endDate": "2026-06-30T12:00:00Z",
    "closedTime": "2026-07-04 00:49:16+00",
    "startDate": "2026-01-01T00:00:00Z",
    "outcomes": '["Yes", "No"]',
    "outcomePrices": '["0", "1"]',
    "clobTokenIds": '["111", "222"]',
    "negRisk": False,
    "events": [{"id": "16390", "tags": [{"label": "Geopolitics"}, {"label": "Ukraine"}]}],
}


class TestGammaParser:
    def test_resolved_binary_market(self) -> None:
        rec = parse_gamma_market(GAMMA_MARKET)
        assert rec is not None
        assert rec.venue is Venue.POLYMARKET_INTL
        assert rec.resolution == "no"
        assert rec.domain is Domain.POLITICS
        assert rec.yes_token == "111" and rec.no_token == "222"
        assert rec.settled_time == datetime(2026, 7, 4, 0, 49, 16, tzinfo=UTC)

    def test_event_context_supplies_tags_and_augmented_flag(self) -> None:
        event = {
            "id": "E1",
            "tags": [{"label": "Crypto"}],
            "negRiskAugmented": True,
            "negRisk": True,
            "markets": [dict(GAMMA_MARKET)],
        }
        records = parse_gamma_event(event)
        assert len(records) == 1
        assert records[0].domain is Domain.CRYPTO  # event tags win over embedded ones
        assert records[0].neg_risk is True
        assert records[0].neg_risk_augmented is True
        assert records[0].event_id == "E1"

    def test_unresolved_or_odd_markets_skipped(self) -> None:
        half = dict(GAMMA_MARKET, outcomePrices='["0.5", "0.5"]')
        assert parse_gamma_market(half) is None
        scalar = dict(GAMMA_MARKET, outcomes='["Up", "Down"]')
        assert parse_gamma_market(scalar) is None

    def test_parse_ts_tolerates_venue_formats(self) -> None:
        assert parse_ts("2026-06-30T12:00:00Z") == datetime(2026, 6, 30, 12, tzinfo=UTC)
        assert parse_ts("2026-07-04 00:49:16+00") == datetime(2026, 7, 4, 0, 49, 16, tzinfo=UTC)
        assert parse_ts(None) is None
        assert parse_ts("garbage") is None


class TestTapeBars:
    def test_no_trades_converted_and_bucketed(self) -> None:
        base = 1780530000 - (1780530000 % 3600)
        trades = [
            {"timestamp": base + 100, "price": "0.40", "size": "10", "outcome": "Yes"},
            {"timestamp": base + 200, "price": "0.55", "size": "5", "outcome": "No"},
            {"timestamp": base + 3700, "price": "0.50", "size": "1", "outcome": "Yes"},
        ]
        bars = tape_to_hourly_bars(trades, venue=Venue.POLYMARKET_INTL, market_id="M")
        assert len(bars) == 2
        first, second = bars
        # last trade in hour 1 is the NO trade at 0.55 -> yes price 0.45
        assert first.price == Decimal("0.45")
        assert first.volume == Decimal("15")
        assert second.price == Decimal("0.50")

    def test_float_price_in_tape_raises(self) -> None:
        with pytest.raises(TypeError, match="float"):
            tape_to_hourly_bars(
                [{"timestamp": 1780530000, "price": 0.4, "size": "1", "outcome": "Yes"}],
                venue=Venue.POLYMARKET_INTL,
                market_id="M",
            )


class TestClassify:
    def test_kalshi_categories(self) -> None:
        assert domain_from_kalshi_category("Elections") is Domain.POLITICS
        assert domain_from_kalshi_category("Climate and Weather") is Domain.WEATHER
        assert domain_from_kalshi_category("Never Heard Of It") is Domain.OTHER
        assert domain_from_kalshi_category(None) is Domain.OTHER

    def test_polymarket_tag_priority(self) -> None:
        # sports beats politics when both present (NFL in Washington ≠ politics)
        assert domain_from_polymarket_tags(["Politics", "NFL"]) is Domain.SPORTS
        assert domain_from_polymarket_tags(["Economy", "Business"]) is Domain.FINANCE
        assert domain_from_polymarket_tags(["Obscure"]) is Domain.OTHER


class TestResolutionQueries:
    def test_calibration_rows_and_counts(self, tmp_path: Path) -> None:
        ledger = Ledger(tmp_path)
        ledger.write_markets(
            [
                _mk("A", resolution="yes"),
                _mk("B", resolution="no"),
                _mk("C", resolution=None),  # unresolved: excluded from labels
            ]
        )
        ledger.write_prices(
            [
                _px("A", 30, "0.70"),
                _px("A", 2, "0.90"),
                _px("B", 5, "0.20"),
                # price after close must be excluded
                PriceRecord(
                    venue=Venue.KALSHI,
                    market_id="A",
                    ts=NOW + timedelta(hours=1),
                    price=Decimal("1.00"),
                    source="kalshi_candle",
                ),
            ]
        )
        with ledger.connect() as con:
            rel = calibration_rows(con)
            cols = rel.columns
            rows = [dict(zip(cols, r, strict=True)) for r in rel.fetchall()]
            counts = resolved_market_counts(con)
        assert len(rows) == 3  # post-close row excluded
        by_market = {(r["market_id"], r["y"]) for r in rows}
        assert ("A", 1) in by_market and ("B", 0) in by_market
        assert counts == [("kalshi", "politics", 2, 2)]

    def test_horizon_anchors_on_actual_resolution_not_placeholder_close(
        self, tmp_path: Path
    ) -> None:
        """Regression guard for the worst bug found in Phase 3.

        Gamma's `endDate` is usually a far-future placeholder (88% of real
        markets settle before it, median 112 days early). Anchoring the
        horizon on it inflated every horizon by months and swept all rows
        into `1mo+`, silently destroying the per-horizon model. The horizon
        must be measured to LEAST(close_time, settled_time).
        """
        ledger = Ledger(tmp_path)
        settled = NOW  # actually resolved here
        placeholder_close = NOW + timedelta(days=200)  # bogus far-future endDate
        ledger.write_markets(
            [
                MarketRecord(
                    venue=Venue.KALSHI,
                    market_id="EARLY",
                    title="resolved long before its nominal close",
                    domain=Domain.POLITICS,
                    close_time=placeholder_close,
                    settled_time=settled,
                    resolution="yes",
                )
            ]
        )
        # A quote 3 days before actual resolution.
        ledger.write_prices(
            [
                PriceRecord(
                    venue=Venue.KALSHI,
                    market_id="EARLY",
                    ts=settled - timedelta(days=3),
                    price=Decimal("0.80"),
                    source="kalshi_candle",
                )
            ]
        )
        with ledger.connect() as con:
            rel = calibration_rows(con)
            cols = rel.columns
            rows = [dict(zip(cols, r, strict=True)) for r in rel.fetchall()]
        assert len(rows) == 1
        horizon_days = float(rows[0]["horizon_days"])
        assert horizon_days == pytest.approx(3.0, abs=0.01), (
            f"horizon {horizon_days:.1f}d — anchored on the placeholder close "
            "instead of actual resolution"
        )
        assert horizon_bucket(horizon_days) == "2d-1w"

    def test_rows_after_actual_resolution_are_excluded(self, tmp_path: Path) -> None:
        """Post-resolution prices converge to 0/1 and would leak the label."""
        ledger = Ledger(tmp_path)
        ledger.write_markets(
            [
                MarketRecord(
                    venue=Venue.KALSHI,
                    market_id="M",
                    title="m",
                    domain=Domain.POLITICS,
                    close_time=NOW + timedelta(days=200),
                    settled_time=NOW,
                    resolution="yes",
                )
            ]
        )
        ledger.write_prices(
            [
                _px("M", 2, "0.70"),  # before resolution: keep
                PriceRecord(
                    venue=Venue.KALSHI,
                    market_id="M",
                    ts=NOW + timedelta(hours=5),  # after resolution: drop
                    price=Decimal("1.00"),
                    source="kalshi_candle",
                ),
            ]
        )
        with ledger.connect() as con:
            rel = calibration_rows(con)
            cols = rel.columns
            rows = [dict(zip(cols, r, strict=True)) for r in rel.fetchall()]
        assert len(rows) == 1
        assert float(rows[0]["yes_price"]) == pytest.approx(0.70)

    def test_horizon_buckets(self) -> None:
        assert horizon_bucket(0.02) == "0-1h"
        assert horizon_bucket(0.5) == "12-24h"
        assert horizon_bucket(3.0) == "2d-1w"
        assert horizon_bucket(400.0) == "1mo+"
