"""Capital lockup: APY against the pessimistic settlement date (§2.3.5, §5.3).

Every hold-to-resolution trade locks capital until the venue actually pays,
which is later than the nominal close: UMA's optimistic oracle has a challenge
window and dispute path on Polymarket; Kalshi's Outcome Review Committee adds
lag. Ranking on nominal close time systematically overstates APY on exactly
the trades most likely to be disputed.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

from sibyl.venues.normalise import Venue

# Conservative settlement buffers beyond nominal close. These are deliberate
# overestimates pending measured resolution-lag distributions from Phase 2
# (spec gives the mechanism but no numbers; tune from the resolution ledger).
SETTLEMENT_BUFFER: dict[Venue, timedelta] = {
    Venue.POLYMARKET_US: timedelta(days=2),  # UMA challenge window + dispute slack
    Venue.POLYMARKET_INTL: timedelta(days=2),
    Venue.KALSHI: timedelta(days=1),  # Outcome Review Committee lag
}

# Idle-cash yield by venue: Kalshi pays ~4% APY on cash, Polymarket 0%.
# The cash-like leg of any cross-venue structure belongs where cash earns.
IDLE_CASH_APY: dict[Venue, Decimal] = {
    Venue.KALSHI: Decimal("0.04"),
    Venue.POLYMARKET_US: Decimal("0"),
    Venue.POLYMARKET_INTL: Decimal("0"),
}

_DAYS_PER_YEAR = Decimal("365")
_MIN_LOCKUP_DAYS = Decimal("1")  # floor: intraday lockups still tie capital for the day


def pessimistic_settlement(nominal_close: datetime, venue: Venue) -> datetime:
    return nominal_close + SETTLEMENT_BUFFER[venue]


def lockup_days(now: datetime, nominal_close: datetime, venue: Venue) -> Decimal:
    """Days of capital lockup to pessimistic settlement, floored at one day."""
    seconds = (pessimistic_settlement(nominal_close, venue) - now).total_seconds()
    days = Decimal(str(seconds)) / Decimal("86400")
    return max(days, _MIN_LOCKUP_DAYS)


def lockup_apy(payoff: Decimal, total_cost: Decimal, days: Decimal) -> Decimal:
    """APY = (payoff / total_cost) ^ (365 / days) - 1  (§5.3).

    `total_cost` is all-in (notional + fees + any gas). Returns can be
    negative when total_cost exceeds payoff.
    """
    if total_cost <= 0:
        raise ValueError(f"total_cost must be positive, got {total_cost}")
    if payoff < 0:
        raise ValueError(f"payoff must be non-negative, got {payoff}")
    if days < _MIN_LOCKUP_DAYS:
        raise ValueError(f"days must be >= {_MIN_LOCKUP_DAYS} (use lockup_days), got {days}")
    ratio = payoff / total_cost
    return ratio ** (_DAYS_PER_YEAR / days) - 1


def excess_apy(payoff: Decimal, total_cost: Decimal, days: Decimal, venue: Venue) -> Decimal:
    """APY net of the idle-cash yield the same capital would earn at `venue`.

    On Kalshi, capital locked in a position forgoes ~4% cash APY; the
    opportunity is only the APY above that hurdle.
    """
    return lockup_apy(payoff, total_cost, days) - IDLE_CASH_APY[venue]
