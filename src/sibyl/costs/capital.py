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

# Settlement buffers beyond nominal close, as a PESSIMISTIC (p90) estimate of
# how long capital stays locked after a market's outcome window ends.
#
# These were guesses (2 days / 1 day) until measured against the resolution
# ledger on 2026-10-11 (`sibyl lag`). The guess was badly optimistic:
#
#   Polymarket Intl, 1,217 markets settling at or after close:
#       median 1.25d   p90 13.9d   p99 14.1d   max 60.3d
#
# p90 is ~7x the old 2-day assumption. That matters more than it looks: APY
# is annualised over the lockup, so understating the buffer inflates the
# ranking of exactly the long-settling markets most likely to be disputed.
# p90 is chosen over p99 as "pessimistic but not paranoid"; raise it with
# `buffer_percentile` reasoning if a strategy is sensitive to the tail.
SETTLEMENT_BUFFER: dict[Venue, timedelta] = {
    # Measured (p90). INTL is the venue we actually have history for.
    Venue.POLYMARKET_INTL: timedelta(days=14),
    # No US history is reachable from this jurisdiction. The US exchange runs
    # the same UMA optimistic oracle, so the INTL figure is the best
    # available proxy — carried over rather than invented.
    Venue.POLYMARKET_US: timedelta(days=14),
    # UNMEASURED. Kalshi demo reports median 0.02d / p90 0.05d / p99 18.8d,
    # but demo is a paper market and its settlement timing is not evidence
    # about production. Left as the original conservative guess; replace only
    # with figures from a venue we can actually observe.
    Venue.KALSHI: timedelta(days=1),
}

# Which venues' buffers rest on measured data vs. a documented assumption.
# Anything False should be treated as a modelling input, not a fact.
SETTLEMENT_BUFFER_MEASURED: dict[Venue, bool] = {
    Venue.POLYMARKET_INTL: True,
    Venue.POLYMARKET_US: False,
    Venue.KALSHI: False,
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
