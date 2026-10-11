"""Polymarket read clients: Gamma (discovery) + CLOB (books/prices).

Read paths only for now — order placement arrives with the paper→live gate.
Prices come back as JSON strings and stay strings until parsed to Decimal.

Polymarket US and International are distinct venues (different books, fees,
eligibility). The client is constructed for exactly one of them and stamps
that venue on everything it returns, so books can never be joined by accident.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from types import TracebackType
from typing import Any, Self

import httpx
from aiolimiter import AsyncLimiter
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from sibyl.venues.normalise import BinaryBook, Level, Venue

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"
DATA_BASE = "https://data-api.polymarket.com"


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    return isinstance(exc, httpx.TransportError)


class PolymarketReadClient:
    """Unauthenticated reads across gamma / clob / data APIs for ONE venue."""

    def __init__(self, venue: Venue = Venue.POLYMARKET_US) -> None:
        if venue not in (Venue.POLYMARKET_US, Venue.POLYMARKET_INTL):
            raise ValueError(f"not a Polymarket venue: {venue}")
        self.venue = venue
        # Spec rate limits: ~4000/10s general, 500/10s /events, 300/10s /markets.
        # We stay far under: these are research reads, not the hot path.
        self._gamma_limiter = AsyncLimiter(max_rate=25, time_period=1)
        self._clob_limiter = AsyncLimiter(max_rate=50, time_period=1)
        self._data_limiter = AsyncLimiter(max_rate=20, time_period=1)
        self._gamma = httpx.AsyncClient(base_url=GAMMA_BASE, http2=True, timeout=15.0)
        self._clob = httpx.AsyncClient(base_url=CLOB_BASE, http2=True, timeout=15.0)
        self._data = httpx.AsyncClient(base_url=DATA_BASE, http2=True, timeout=15.0)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self._gamma.aclose()
        await self._clob.aclose()
        await self._data.aclose()

    @retry(
        retry=retry_if_exception(_is_retryable),
        wait=wait_exponential_jitter(initial=1, max=30),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    async def _get(
        self,
        client: httpx.AsyncClient,
        limiter: AsyncLimiter,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> Any:
        async with limiter:
            response = await client.get(path, params=params)
        response.raise_for_status()
        return response.json(parse_float=str)  # prices must never become floats

    async def events(self, **params: Any) -> list[dict[str, Any]]:
        payload = await self._get(self._gamma, self._gamma_limiter, "/events", params or None)
        if not isinstance(payload, list):
            raise TypeError(f"expected list from /events, got {type(payload).__name__}")
        return payload

    async def gamma_events(self, **params: Any) -> list[dict[str, Any]]:
        """gamma /events — tags (domain source), negRiskAugmented flag, and
        fully-populated nested markets.

        Offset paging dies with 422 ("offset too large") past ~2100 rows; use
        `gamma_events_keyset` for anything deeper.
        """
        payload = await self._get(self._gamma, self._gamma_limiter, "/events", params or None)
        if not isinstance(payload, list):
            raise TypeError(f"expected list from /events, got {type(payload).__name__}")
        return payload

    async def gamma_events_keyset(
        self, *, cursor: str | None = None, **params: Any
    ) -> tuple[list[dict[str, Any]], str | None]:
        """gamma /events/keyset — cursor paging with no depth ceiling.

        Returns (events, next_cursor). Same nested shape as /events.
        """
        query: dict[str, Any] = dict(params)
        if cursor:
            query["cursor"] = cursor
        payload = await self._get(self._gamma, self._gamma_limiter, "/events/keyset", query)
        if not isinstance(payload, dict):
            raise TypeError(f"expected object from /events/keyset, got {type(payload).__name__}")
        events = payload.get("events") or []
        if not isinstance(events, list):
            raise TypeError("keyset 'events' is not a list")
        next_cursor = payload.get("next_cursor")
        return events, str(next_cursor) if next_cursor else None

    async def markets(self, **params: Any) -> list[dict[str, Any]]:
        payload = await self._get(self._gamma, self._gamma_limiter, "/markets", params or None)
        if not isinstance(payload, list):
            raise TypeError(f"expected list from /markets, got {type(payload).__name__}")
        return payload

    async def clob_book(self, token_id: str) -> BinaryBook:
        """Orderbook for one outcome token, normalised to the canonical book.

        CLOB levels arrive as {"price": str, "size": str}; ordering from the
        API is not guaranteed to be best-first (spec gotcha), so we sort
        explicitly rather than trusting array order.
        """
        payload = await self._get(self._clob, self._clob_limiter, "/book", {"token_id": token_id})
        if not isinstance(payload, dict):
            raise TypeError(f"expected object from /book, got {type(payload).__name__}")
        bids = self._parse_levels(payload.get("bids", []), "bids")
        asks = self._parse_levels(payload.get("asks", []), "asks")
        bids.sort(key=lambda level: level.price, reverse=True)
        asks.sort(key=lambda level: level.price)
        return BinaryBook(
            venue=self.venue,
            market_id=token_id,
            ts=datetime.now(UTC),
            yes_bids=tuple(bids),
            yes_asks=tuple(asks),
        )

    @staticmethod
    def _parse_levels(raw: object, field: str) -> list[Level]:
        if not isinstance(raw, list):
            raise TypeError(f"{field}: expected list, got {type(raw).__name__}")
        levels: list[Level] = []
        for entry in raw:
            if not isinstance(entry, dict):
                raise TypeError(f"{field}: malformed level {entry!r}")
            price, size = entry.get("price"), entry.get("size")
            if not isinstance(price, str) or not isinstance(size, str):
                raise TypeError(f"{field}: price/size must be strings, got {entry!r}")
            levels.append(Level(price=Decimal(price), size=Decimal(size)))
        return levels

    async def prices_history(
        self, token_id: str, *, start_ts: int, end_ts: int, fidelity: int = 60
    ) -> list[dict[str, Any]]:
        """CLOB price history for one token. Empty for resolved markets — the
        CLOB prunes history at resolution; use the data-api tape for those."""
        payload = await self._get(
            self._clob,
            self._clob_limiter,
            "/prices-history",
            {"market": token_id, "startTs": start_ts, "endTs": end_ts, "fidelity": fidelity},
        )
        if not isinstance(payload, dict):
            raise TypeError(f"expected object from /prices-history, got {type(payload).__name__}")
        history = payload.get("history", [])
        if not isinstance(history, list):
            raise TypeError("prices-history 'history' is not a list")
        return history

    async def trades(
        self, condition_id: str, *, limit: int = 500, offset: int = 0
    ) -> list[dict[str, Any]]:
        """data-api trade tape for a market (survives resolution). Trades
        carry price/size as strings plus the traded outcome label."""
        payload = await self._get(
            self._data,
            self._data_limiter,
            "/trades",
            {"market": condition_id, "limit": limit, "offset": offset},
        )
        if not isinstance(payload, list):
            raise TypeError(f"expected list from /trades, got {type(payload).__name__}")
        return payload

    async def midpoint(self, token_id: str) -> dict[str, Any]:
        payload = await self._get(
            self._clob, self._clob_limiter, "/midpoint", {"token_id": token_id}
        )
        if not isinstance(payload, dict):
            raise TypeError(f"expected object from /midpoint, got {type(payload).__name__}")
        return payload
