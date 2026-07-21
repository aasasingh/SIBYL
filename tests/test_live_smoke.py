"""Live connectivity smoke tests — hit real endpoints, run with `-m live`."""

from __future__ import annotations

import pytest

from sibyl.venues.kalshi import KalshiClient
from sibyl.venues.normalise import Venue
from sibyl.venues.polymarket import PolymarketReadClient

pytestmark = pytest.mark.live


async def test_kalshi_demo_exchange_status() -> None:
    async with KalshiClient(demo=True) as kalshi:
        status = await kalshi.exchange_status()
    assert "exchange_active" in status or "trading_active" in status


async def test_kalshi_demo_markets_and_book() -> None:
    async with KalshiClient(demo=True) as kalshi:
        payload = await kalshi.markets(limit=5, status="open")
        markets = payload.get("markets", [])
        assert markets, "demo venue returned no open markets"
        book = await kalshi.book(markets[0]["ticker"])
    assert book.venue is Venue.KALSHI
    if book.best_bid and book.best_ask:
        assert not book.crossed, "kalshi book crossed — normaliser is misreading sides"


async def test_polymarket_gamma_markets() -> None:
    async with PolymarketReadClient(Venue.POLYMARKET_US) as poly:
        markets = await poly.markets(limit=2, active=True, closed=False)
    assert markets, "gamma returned no active markets"
    assert "question" in markets[0] or "title" in markets[0]
