"""Trade attribution: was a size decrease a trade or a cancel?

Step 2 of the fill-model build order. A resting order moves up the queue
when size ahead of it leaves the level, but *why* it left decides what the
simulation means: consumption by a taker is the event that precedes adverse
price movement; a cancel is not. The trade stream is the only evidence of
consumption, so every decrease is matched against it.

Measured facts the rules below rest on (20h of recordings, 36.8k trades):

  * **Every trade has a matching decrease** at its asset and price, 99.4%
    with exactly equal size. Unmatched trade volume is reported, not hidden:
    if it stops being ~0 the replay or the matching has broken.
  * **`side` is the taker's side.** BUY consumed the ask, SELL the bid, in
    every case.
  * **The book delta arrives first**: decrease minus trade timestamp is
    median -20 ms, 98% within [-387, -16] ms, extremes about +/-1.1 s. The
    matching window is symmetric and a tunable; report sensitivity to it.
  * **Each order sits on both legs.** A YES order at p also appears as a NO
    order at 1-p on the opposite side, but the venue reports each trade on
    one leg only. Without mirroring, every trade would read as a cancel on
    the other book. Mirrored decreases were found for all 36.8k trades.
  * **Cancels dominate.** ~99.9% of the volume leaving books is cancelled,
    not traded. Queue advance is overwhelmingly by cancellation.

Ambiguity is labelled, not silently resolved. Which interpretation is
"pessimistic" depends on where a simulated order sits relative to the
volume, so the queue simulator (step 3) resolves it; this step reports how
much there is. Two kinds:

  * **mixed** -- a decrease part-explained by trades. The remainder is
    booked as cancelled, but it could be a trade on a level we did not see
    swept, or a cancel racing a fill.
  * **contested** -- a trade that could have explained more than one
    decrease at its level within the window; which one it consumed is a
    guess (nearest). A *unique* exact-size match is not contested: on a busy
    level there are almost always other cancels within a second, and
    flagging those made the ambiguity rate a function of the window (1.2k
    contested at 250 ms, 74k at 3 s) rather than of the data.

Unmatched trades are reported by count as well as volume: one 90,000-share
trade (part of it matched against an order that never rested visibly)
accounts for 99% of the unmatched volume in the first 20h.

Counts are also misleading in the other direction: 97.8% of recorded trades
are one bot buying 5 shares at 0.002 roughly once a second. Ambiguity is
therefore also reported weighted by traded notional.

The window is bounded from both sides by measurement: above the trade-vs-
delta latency (98% within 387 ms) and below the spacing of repeated
identical trades (~1 s for that bot). At 1 s the bot's consecutive trades
become indistinguishable and "contested" jumps from ~1k to ~67k. Hence a
500 ms primary window, with a wider fallback for leftovers only.
"""

from __future__ import annotations

import bisect
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Literal

from sibyl.fill.ladder import Side

Kind = Literal["cancel", "trade", "mixed", "contested"]
ONE = Decimal(1)
ZERO = Decimal(0)


@dataclass(frozen=True)
class Trade:
    asset_id: str
    price: Decimal
    size: Decimal
    taker_side: Literal["BUY", "SELL"]
    venue_ts: int
    tx_hash: str

    @property
    def consumed(self) -> Side:
        """The book side this trade took liquidity from."""
        return "ask" if self.taker_side == "BUY" else "bid"


@dataclass(frozen=True)
class Decrease:
    asset_id: str
    side: Side
    price: Decimal
    amount: Decimal
    venue_ts: int
    recv_ts: datetime


@dataclass
class Attributed:
    decrease: Decrease
    traded: Decimal = ZERO
    contested: bool = False

    @property
    def cancelled(self) -> Decimal:
        return self.decrease.amount - self.traded

    @property
    def kind(self) -> Kind:
        if self.contested:
            return "contested"
        if self.traded == 0:
            return "cancel"
        if self.traded == self.decrease.amount:
            return "trade"
        return "mixed"


@dataclass
class AttributionStats:
    trades: int = 0
    trade_volume: Decimal = ZERO
    # Trades with any size left unexplained on their own (direct) leg.
    unmatched_trades: int = 0
    # Per leg: a trade is matched on its own book and on the mirror.
    unmatched_direct: Decimal = ZERO
    unmatched_mirror: Decimal = ZERO
    decreases: int = 0
    decrease_volume: Decimal = ZERO
    by_kind: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    traded_notional: Decimal = ZERO
    ambiguous_notional: Decimal = ZERO
    volume_by_kind: dict[str, Decimal] = field(default_factory=lambda: defaultdict(Decimal))

    def summary(self) -> dict[str, float | int]:
        dv = self.decrease_volume or ONE
        n = self.decreases or 1
        tv = self.trade_volume or ONE
        ambiguous = self.by_kind["mixed"] + self.by_kind["contested"]
        return {
            "trades": self.trades,
            "decreases": self.decreases,
            "unmatched_trades": self.unmatched_trades,
            "unmatched_trade_share_direct": round(float(self.unmatched_direct / tv), 6),
            "unmatched_trade_share_mirror": round(float(self.unmatched_mirror / tv), 6),
            "traded_volume_share": round(
                float(
                    (
                        self.volume_by_kind["trade"]
                        + self.volume_by_kind["mixed_traded"]
                        + self.volume_by_kind["contested_traded"]
                    )
                    / dv
                ),
                6,
            ),
            "decreases_cancel": self.by_kind["cancel"],
            "decreases_trade": self.by_kind["trade"],
            "decreases_mixed": self.by_kind["mixed"],
            "decreases_contested": self.by_kind["contested"],
            # Of decreases that involve a trade at all, the share whose
            # attribution is not clean. This is the number that says whether
            # a maker result rests on data or on the window parameter.
            "ambiguous_share_of_traded": round(
                ambiguous / max(1, ambiguous + self.by_kind["trade"]), 6
            ),
            "ambiguous_share_of_all": round(ambiguous / n, 6),
            # Same, weighted by dollars traded rather than decrease count, so
            # a cent-sized bot cannot dominate it.
            "ambiguous_share_of_traded_notional": round(
                float(self.ambiguous_notional / (self.traded_notional or ONE)), 6
            ),
        }


def parse_trade(payload: str) -> Trade | None:
    d = json.loads(payload)
    side = d.get("side")
    if side not in ("BUY", "SELL"):
        return None
    try:
        return Trade(
            asset_id=str(d["asset_id"]),
            price=Decimal(str(d["price"])),
            size=Decimal(str(d["size"])),
            taker_side=side,
            venue_ts=int(d["timestamp"]),
            tx_hash=str(d.get("transaction_hash") or ""),
        )
    except (KeyError, ValueError, ArithmeticError):
        return None


def _legs(trade: Trade, mirror: Mapping[str, str]) -> list[tuple[str, Side, Decimal, bool]]:
    """(asset, consumed side, price, is_mirror) for each book the trade hit."""
    legs: list[tuple[str, Side, Decimal, bool]] = [
        (trade.asset_id, trade.consumed, trade.price, False)
    ]
    other = mirror.get(trade.asset_id)
    if other is not None:
        flipped: Side = "bid" if trade.consumed == "ask" else "ask"
        legs.append((other, flipped, ONE - trade.price, True))
    return legs


def attribute(
    decreases: Iterable[Decrease],
    trades: Iterable[Trade],
    *,
    mirror: Mapping[str, str],
    window_ms: int = 500,
    fallback_ms: int = 1500,
) -> tuple[list[Attributed], AttributionStats]:
    """Match trades to the decreases they caused, per (asset, side, price).

    Two passes. The first matches within `window_ms`, which holds the bulk
    of the measured trade-vs-delta latency; a trade there is contested only
    if it lacks a unique exact-size match. The second gives trade volume
    still unexplained a `fallback_ms` window, and every decrease it touches
    is marked contested: a late match is a low-confidence match.
    """
    stats = AttributionStats()
    by_level: dict[tuple[str, Side, Decimal], list[Attributed]] = defaultdict(list)
    out: list[Attributed] = []
    for d in decreases:
        a = Attributed(d)
        out.append(a)
        by_level[(d.asset_id, d.side, d.price)].append(a)
    for level in by_level.values():
        level.sort(key=lambda a: a.decrease.venue_ts)
    times = {k: [a.decrease.venue_ts for a in v] for k, v in by_level.items()}

    # One entry per (trade, leg) with its still-unexplained size.
    legs: list[tuple[Trade, str, Side, Decimal, bool]] = []
    for trade in sorted(trades, key=lambda t: t.venue_ts):
        stats.trades += 1
        stats.trade_volume += trade.size
        legs.extend((trade, *leg) for leg in _legs(trade, mirror))
    remaining = [trade.size for trade, *_ in legs]

    for window, fallback in ((window_ms, False), (fallback_ms, True)):
        for i, (trade, asset, side, price, _) in enumerate(legs):
            if remaining[i] <= 0:
                continue
            key = (asset, side, price)
            level = by_level.get(key, [])
            ts = times.get(key, [])
            lo = bisect.bisect_left(ts, trade.venue_ts - window)
            hi = bisect.bisect_right(ts, trade.venue_ts + window)
            open_ = [a for a in level[lo:hi] if a.traded < a.decrease.amount]
            exact = [a for a in open_ if a.decrease.amount - a.traded == remaining[i]]
            if not fallback and len(exact) != 1 and len(open_) > 1:
                for a in open_:
                    a.contested = True
            # Exact size first, then nearest in time.
            open_.sort(
                key=lambda a: (
                    a.decrease.amount - a.traded != remaining[i],
                    abs(a.decrease.venue_ts - trade.venue_ts),
                )
            )
            for a in open_:
                if remaining[i] <= 0:
                    break
                take = min(remaining[i], a.decrease.amount - a.traded)
                a.traded += take
                remaining[i] -= take
                if fallback:
                    a.contested = True

    for i, (_, _, _, _, is_mirror) in enumerate(legs):
        if remaining[i] > 0:
            if is_mirror:
                stats.unmatched_mirror += remaining[i]
            else:
                stats.unmatched_direct += remaining[i]
                stats.unmatched_trades += 1

    for a in out:
        stats.decreases += 1
        stats.decrease_volume += a.decrease.amount
        kind = a.kind
        stats.by_kind[kind] += 1
        notional = a.traded * a.decrease.price
        stats.traded_notional += notional
        if kind in ("mixed", "contested"):
            stats.ambiguous_notional += notional
        if kind in ("trade", "cancel"):
            stats.volume_by_kind[kind] += a.decrease.amount
        else:
            stats.volume_by_kind[f"{kind}_traded"] += a.traded
            stats.volume_by_kind[f"{kind}_cancelled"] += a.cancelled
    return out, stats
