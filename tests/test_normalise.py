"""Book normaliser tests — including the Phase 1 acceptance property:
for every Kalshi book, yes_ask == 1 - best_no_bid and the best bid is the
LAST element of the ascending ladder."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from sibyl.venues.normalise import (
    BinaryBook,
    Level,
    Quote,
    Venue,
    normalise_kalshi_book,
    parse_price,
)

NOW = datetime(2026, 7, 21, 12, 0, 0, tzinfo=UTC)


def _ladder(cents: list[int], size: str = "100") -> list[list[str]]:
    return [[str(Decimal(c) / 100), size] for c in cents]


def cents_ladder(min_size: int = 0) -> st.SearchStrategy[list[int]]:
    return st.lists(
        st.integers(min_value=1, max_value=99), min_size=min_size, max_size=10, unique=True
    ).map(sorted)


class TestKalshiNormaliser:
    def test_best_bid_is_last_element(self) -> None:
        book = normalise_kalshi_book(
            {"yes_dollars": _ladder([30, 32]), "no_dollars": _ladder([60, 65])},
            market_id="TEST-MKT",
            ts=NOW,
        )
        assert book.best_bid is not None and book.best_bid.price == Decimal("0.32")

    def test_yes_ask_derived_from_best_no_bid(self) -> None:
        book = normalise_kalshi_book(
            {"yes_dollars": _ladder([30, 32]), "no_dollars": _ladder([60, 65])},
            market_id="TEST-MKT",
            ts=NOW,
        )
        # best no bid = 0.65 (last) -> yes ask = 1 - 0.65 = 0.35
        assert book.best_ask is not None and book.best_ask.price == Decimal("0.35")
        assert book.spread == Decimal("0.03")
        assert book.mid == Decimal("0.335")
        assert not book.crossed

    @given(yes_cents=cents_ladder(min_size=1), no_cents=cents_ladder(min_size=1))
    def test_property_phase1_acceptance(self, yes_cents: list[int], no_cents: list[int]) -> None:
        book = normalise_kalshi_book(
            {"yes_dollars": _ladder(yes_cents), "no_dollars": _ladder(no_cents)},
            market_id="PROP",
            ts=NOW,
        )
        assert book.best_bid is not None and book.best_ask is not None
        # best bid is the LAST element of the ascending yes ladder
        assert book.best_bid.price == Decimal(yes_cents[-1]) / 100
        # yes_ask == 1 - best_no_bid, best_no_bid also the LAST element
        assert book.best_ask.price == 1 - Decimal(no_cents[-1]) / 100
        # crossed flag consistent with the raw ladders
        assert book.crossed == (yes_cents[-1] + no_cents[-1] >= 100)
        # every derived ask level preserves the NO ladder's sizes and count
        assert len(book.yes_asks) == len(no_cents)

    def test_empty_sides_allowed(self) -> None:
        book = normalise_kalshi_book(
            {"yes_dollars": [], "no_dollars": _ladder([40])}, market_id="M", ts=NOW
        )
        assert book.best_bid is None
        assert book.best_ask is not None and book.best_ask.price == Decimal("0.60")
        assert book.mid is None

    def test_numeric_prices_rejected(self) -> None:
        with pytest.raises(TypeError, match="strings"):
            normalise_kalshi_book(
                {"yes_dollars": [[0.30, 100]], "no_dollars": []}, market_id="M", ts=NOW
            )

    def test_descending_ladder_rejected(self) -> None:
        with pytest.raises(ValueError, match="ascending"):
            normalise_kalshi_book(
                {"yes_dollars": _ladder([32, 30]), "no_dollars": []}, market_id="M", ts=NOW
            )


class TestCanonicalModels:
    def test_book_requires_aware_timestamp(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            BinaryBook(
                venue=Venue.KALSHI,
                market_id="M",
                ts=datetime(2026, 7, 21),  # noqa: DTZ001 — the point of the test
                yes_bids=(),
                yes_asks=(),
            )

    def test_book_rejects_misordered_bids(self) -> None:
        with pytest.raises(ValueError, match="best-first"):
            BinaryBook(
                venue=Venue.KALSHI,
                market_id="M",
                ts=NOW,
                yes_bids=(
                    Level(price=Decimal("0.30"), size=Decimal("1")),
                    Level(price=Decimal("0.40"), size=Decimal("1")),
                ),
                yes_asks=(),
            )

    def test_level_rejects_float(self) -> None:
        with pytest.raises((TypeError, ValueError)):
            Level(price=0.30, size="1")  # type: ignore[arg-type]

    def test_parse_price_rejects_bool_and_float(self) -> None:
        for bad in (0.5, True, None, [0.5]):
            with pytest.raises(TypeError):
                parse_price(bad)
        assert parse_price("0.50") == Decimal("0.50")

    def test_quote_staleness(self) -> None:
        quote = Quote(
            venue=Venue.POLYMARKET_US,
            market_id="T",
            ts=NOW,
            bid=Decimal("0.40"),
            ask=Decimal("0.42"),
        )
        assert quote.staleness(now=NOW + timedelta(seconds=10)) == 10.0
        assert quote.mid == Decimal("0.41")

    def test_polymarket_venues_are_distinct_types(self) -> None:
        us = Quote(venue=Venue.POLYMARKET_US, market_id="T", ts=NOW, bid=Decimal("0.40"), ask=None)
        intl = us.model_copy(update={"venue": Venue.POLYMARKET_INTL})
        assert us.venue != intl.venue


def test_kalshi_book_payload_shape_guard() -> None:
    bad_payloads: list[dict[str, Any]] = [
        {"yes_dollars": "not-a-list", "no_dollars": []},
        {"yes_dollars": [["0.30"]], "no_dollars": []},  # missing size
    ]
    for payload in bad_payloads:
        with pytest.raises(TypeError):
            normalise_kalshi_book(payload, market_id="M", ts=NOW)
