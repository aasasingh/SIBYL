"""Execution cost model: depth-walked fills, never top-of-book pricing (§5.1).

`walk_levels` integrates an order across book levels; the effective price of
size is the size-weighted average of the levels it consumes plus the venue fee
on the filled quantity. A partial fill is reported as such — sizing against
displayed depth that isn't there is how paper edge becomes live loss.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from decimal import Decimal

from pydantic import BaseModel, ConfigDict

from sibyl.venues.normalise import BinaryBook, Level

FeeFn = Callable[[Decimal, Decimal], Decimal]  # (price, contracts) -> total fee


class ExecutionEstimate(BaseModel):
    """Cost of taking `filled` contracts through the book, fees included."""

    model_config = ConfigDict(frozen=True)

    requested: Decimal
    filled: Decimal
    notional: Decimal  # sum(price * qty) across consumed levels
    fee: Decimal
    levels_consumed: int

    @property
    def complete(self) -> bool:
        return self.filled == self.requested

    @property
    def total_cost(self) -> Decimal:
        return self.notional + self.fee

    @property
    def avg_price(self) -> Decimal | None:
        """Size-weighted average price, excluding fees."""
        if self.filled == 0:
            return None
        return self.notional / self.filled

    @property
    def effective_price(self) -> Decimal | None:
        """Per-contract all-in cost — the number edge must beat."""
        if self.filled == 0:
            return None
        return self.total_cost / self.filled


def walk_levels(
    levels: Sequence[Level],
    size: Decimal,
    fee_fn: FeeFn,
) -> ExecutionEstimate:
    """Fill `size` contracts against best-first `levels`, accruing fees per level.

    Fees are computed per consumed level at that level's price (the venue
    charges on each fill's price, not on the average), then summed.
    """
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    remaining = size
    notional = Decimal("0")
    fee = Decimal("0")
    consumed = 0
    for level in levels:
        if remaining == 0:
            break
        qty = min(remaining, level.size)
        notional += level.price * qty
        fee += fee_fn(level.price, qty)
        remaining -= qty
        consumed += 1
    return ExecutionEstimate(
        requested=size,
        filled=size - remaining,
        notional=notional,
        fee=fee,
        levels_consumed=consumed,
    )


def buy_yes(book: BinaryBook, size: Decimal, fee_fn: FeeFn) -> ExecutionEstimate:
    """Take YES: walk the ask side."""
    return walk_levels(book.yes_asks, size, fee_fn)


def buy_no(book: BinaryBook, size: Decimal, fee_fn: FeeFn) -> ExecutionEstimate:
    """Take NO: buying NO at price q consumes the YES bid at 1-q. Walk the YES
    bid side, converting each level to its NO price."""
    no_asks = tuple(
        Level(price=Decimal("1") - level.price, size=level.size) for level in book.yes_bids
    )
    return walk_levels(no_asks, size, fee_fn)
