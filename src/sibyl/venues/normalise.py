"""Canonical market-data model shared by every venue.

Every price in this module is a `decimal.Decimal` parsed from the venue's
string representation. Floats are rejected at the boundary: a float that has
sneaked into the money path is a bug, not a value.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

ONE = Decimal("1")


class Venue(StrEnum):
    """Polymarket US and International are legally and economically distinct
    venues with separate books. They are separate enum members so that no
    code path can join them by accident."""

    KALSHI = "kalshi"
    POLYMARKET_US = "polymarket_us"
    POLYMARKET_INTL = "polymarket_intl"


class Side(StrEnum):
    YES = "yes"
    NO = "no"


def parse_price(raw: object) -> Decimal:
    """Parse a venue price into Decimal. Only str, int, or Decimal accepted.

    Venues serialise prices as JSON strings; anything arriving as float means
    someone decoded the payload with the wrong parser upstream.
    """
    if isinstance(raw, bool | float):
        raise TypeError(f"price must be str/int/Decimal, got {type(raw).__name__}: {raw!r}")
    if isinstance(raw, Decimal):
        return raw
    if isinstance(raw, int | str):
        return Decimal(raw)
    raise TypeError(f"price must be str/int/Decimal, got {type(raw).__name__}: {raw!r}")


class Level(BaseModel):
    """One price level: (price, size) in dollars/contracts."""

    model_config = ConfigDict(frozen=True)

    price: Decimal
    size: Decimal

    @field_validator("price", "size", mode="before")
    @classmethod
    def _no_floats(cls, v: object) -> Decimal:
        return parse_price(v)


class BinaryBook(BaseModel):
    """A binary market's book, normalised to the YES side.

    `yes_bids`/`yes_asks` are sorted best-first (highest bid first, lowest ask
    first). NO-side quotes are derived, never stored: no_bid = 1 - best yes_ask,
    no_ask = 1 - best yes_bid.
    """

    model_config = ConfigDict(frozen=True)

    venue: Venue
    market_id: str
    ts: datetime
    yes_bids: tuple[Level, ...]
    yes_asks: tuple[Level, ...]

    @field_validator("ts")
    @classmethod
    def _aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("book timestamp must be timezone-aware")
        return v

    @model_validator(mode="after")
    def _sorted_best_first(self) -> BinaryBook:
        bid_prices = [level.price for level in self.yes_bids]
        ask_prices = [level.price for level in self.yes_asks]
        if bid_prices != sorted(bid_prices, reverse=True):
            raise ValueError("yes_bids must be sorted best-first (descending price)")
        if ask_prices != sorted(ask_prices):
            raise ValueError("yes_asks must be sorted best-first (ascending price)")
        return self

    @property
    def best_bid(self) -> Level | None:
        return self.yes_bids[0] if self.yes_bids else None

    @property
    def best_ask(self) -> Level | None:
        return self.yes_asks[0] if self.yes_asks else None

    @property
    def mid(self) -> Decimal | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return (self.best_bid.price + self.best_ask.price) / 2

    @property
    def spread(self) -> Decimal | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return self.best_ask.price - self.best_bid.price

    @property
    def crossed(self) -> bool:
        """A crossed book from one venue is a data error, not an arb."""
        if self.best_bid is None or self.best_ask is None:
            return False
        return self.best_bid.price >= self.best_ask.price

    def staleness(self, now: datetime | None = None) -> float:
        """Seconds since this book was observed."""
        now = now or datetime.now(UTC)
        return (now - self.ts).total_seconds()


class Quote(BaseModel):
    """Top-of-book snapshot. Never exists without venue + timestamp (§3.3)."""

    model_config = ConfigDict(frozen=True)

    venue: Venue
    market_id: str
    ts: datetime
    bid: Decimal | None
    ask: Decimal | None

    @field_validator("bid", "ask", mode="before")
    @classmethod
    def _no_floats(cls, v: object) -> Decimal | None:
        return None if v is None else parse_price(v)

    @field_validator("ts")
    @classmethod
    def _aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("quote timestamp must be timezone-aware")
        return v

    @property
    def mid(self) -> Decimal | None:
        if self.bid is None or self.ask is None:
            return None
        return (self.bid + self.ask) / 2

    def staleness(self, now: datetime | None = None) -> float:
        now = now or datetime.now(UTC)
        return (now - self.ts).total_seconds()


class Market(BaseModel):
    model_config = ConfigDict(frozen=True)

    venue: Venue
    market_id: str
    event_id: str | None = None
    title: str
    close_time: datetime | None = None
    neg_risk: bool = False
    augmented_neg_risk: bool = False
    domain: str | None = None
    raw: dict[str, Any] | None = None


class Fill(BaseModel):
    model_config = ConfigDict(frozen=True)

    venue: Venue
    market_id: str
    side: Side
    price: Decimal
    size: Decimal
    fee: Decimal
    ts: datetime
    order_id: str | None = None

    @field_validator("price", "size", "fee", mode="before")
    @classmethod
    def _no_floats(cls, v: object) -> Decimal:
        return parse_price(v)


def _parse_kalshi_ladder(raw: object, field: str) -> list[Level]:
    if not isinstance(raw, list):
        raise TypeError(f"{field}: expected list of [price, size] pairs, got {type(raw).__name__}")
    levels: list[Level] = []
    for entry in raw:
        if not isinstance(entry, list | tuple) or len(entry) != 2:
            raise TypeError(f"{field}: malformed level {entry!r}")
        price, size = entry
        if not isinstance(price, str) or not isinstance(size, str):
            # Kalshi serialises dollars as strings; numbers mean a wrong decode upstream.
            raise TypeError(f"{field}: prices/sizes must be strings, got {entry!r}")
        levels.append(Level(price=Decimal(price), size=Decimal(size)))
    prices = [level.price for level in levels]
    if prices != sorted(prices):
        raise ValueError(f"{field}: expected ascending price order, got {prices}")
    return levels


def normalise_kalshi_book(
    orderbook_fp: dict[str, Any],
    *,
    market_id: str,
    ts: datetime,
) -> BinaryBook:
    """Normalise Kalshi's bids-only `orderbook_fp` into a canonical YES book.

    Kalshi returns two BID ladders — `yes_dollars` and `no_dollars` — each
    ascending, so the BEST bid is the LAST element. There are no resting asks:
    the YES ask is a derived view of the NO bid (yes_ask = 1 - best_no_bid).
    Treating the two ladders as independent books is the classic integration
    bug this function exists to prevent.
    """
    yes_bids_asc = _parse_kalshi_ladder(orderbook_fp.get("yes_dollars", []), "yes_dollars")
    no_bids_asc = _parse_kalshi_ladder(orderbook_fp.get("no_dollars", []), "no_dollars")

    # Best-first: reverse the ascending ladder.
    yes_bids = tuple(reversed(yes_bids_asc))
    # YES asks derived from NO bids: price 1 - no_bid, size unchanged. The best
    # (lowest) yes_ask comes from the best (highest, i.e. last) no_bid, so the
    # reversed NO ladder maps to an ascending ask ladder.
    yes_asks = tuple(
        Level(price=ONE - level.price, size=level.size) for level in reversed(no_bids_asc)
    )
    return BinaryBook(
        venue=Venue.KALSHI,
        market_id=market_id,
        ts=ts,
        yes_bids=yes_bids,
        yes_asks=yes_asks,
    )
