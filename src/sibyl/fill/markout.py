"""Markout harness: after a simulated fill, did the price move against us?

Step 4 of the fill-model build order, and the decisive number for the
maker-rebate question (spec §5.6). For each simulated fill at price p:

    markout(h) = mid(t + h) - p     for a bid (we bought)
               = p - mid(t + h)     for an ask (we sold)

in cents per share; positive is good for the maker. It splits exactly into

    spread  = +/-(mid_at_fill - p)        what the fill looked worth at once
    drift   = +/-(mid(t+h) - mid_at_fill) the adverse move, usually negative

and is set against the per-fill maker rebate, 0.0125 * p * (1 - p) per share
(Polymarket US schedule; ~0.31c at p = 0.5). On INTL the rebate is a daily
pool share that cannot be known at fill time (see `costs/fees.py`), so the
US figure is an upper reference for the credit, not a cost reduction.

A markout is only taken when the mid at t + h is *known*: a coverage gap
between the fill and the horizon, the end of the recording, or a one-sided
book at the horizon all make it unknown, and such fills are counted as
`unmarked`, never filled in.

Fills on one market are not independent (both legs of a market, and probes
a minute apart, see the same price path), so the standard error is
clustered by market. With few markets it is the honest, wide one.
"""

from __future__ import annotations

import math
from array import array
from bisect import bisect_left, bisect_right
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from sibyl.fill.ladder import BookSnapshot, ChangeFrame, LevelChange, Replayer, Reset
from sibyl.fill.queue import Fill, OrderState
from sibyl.fill.replay import mirror_map, probe_orders, simulate_recorded, venue_time_range
from sibyl.log import get_logger

log = get_logger(__name__)

HORIZONS_S: tuple[int, ...] = (60, 300, 3600)
REBATE_COEF = 0.0125


class MidTracker:
    """Record every change of mid per asset, and every coverage gap.

    Attach before replaying. Stores only *changes* of mid, as compact arrays;
    NaN marks "unknown" (one side empty, or state discarded by a reset).
    """

    def __init__(self, replayer: Replayer) -> None:
        self.replayer = replayer
        self._ts: dict[str, array[int]] = defaultdict(lambda: array("q"))
        self._mid: dict[str, array[float]] = defaultdict(lambda: array("d"))
        # Last venue time seen before each reset: a markout window that
        # contains one crosses a span whose prices are unknown.
        self.gaps: list[int] = []
        self.last_ts: int | None = None
        replayer.listeners.append(self._on_change)
        replayer.snapshot_listeners.append(self._on_snapshot)
        replayer.reset_listeners.append(self._on_reset)

    def _note(self, asset: str, ts: int) -> None:
        ladder = self.replayer.ladders.get(asset)
        mid = math.nan
        if ladder is not None:
            bid, ask = ladder.best_bid, ladder.best_ask
            if bid is not None and ask is not None:
                mid = float(bid + ask) / 2
        mids, times = self._mid[asset], self._ts[asset]
        if mids and (mids[-1] == mid or (math.isnan(mid) and math.isnan(mids[-1]))):
            return
        # Replay runs in receive order; venue times can step back a little.
        times.append(max(ts, times[-1]) if times else ts)
        mids.append(mid)

    def _on_change(self, change: LevelChange, before: Decimal, frame: ChangeFrame) -> None:
        self.last_ts = frame.venue_ts
        self._note(change.asset_id, frame.venue_ts)

    def _on_snapshot(self, snap: BookSnapshot) -> None:
        if snap.venue_ts > 0:
            self.last_ts = max(self.last_ts or 0, snap.venue_ts)
            self._note(snap.asset_id, snap.venue_ts)

    def _on_reset(self, reset: Reset) -> None:
        if self.last_ts is not None:
            self.gaps.append(self.last_ts)
        for asset, mids in self._mid.items():
            if mids and not math.isnan(mids[-1]):
                self._ts[asset].append(self.last_ts or 0)
                mids.append(math.nan)

    def mid_at(self, asset: str, ts: int) -> float | None:
        """Mid in force at venue time `ts` (after every update at `ts`)."""
        times = self._ts.get(asset)
        if not times:
            return None
        i = bisect_right(times, ts) - 1
        if i < 0:
            return None
        mid = self._mid[asset][i]
        return None if math.isnan(mid) else mid

    def gap_between(self, start: int, end: int) -> bool:
        """Is there a reset at or after `start` and before `end`?"""
        i = bisect_left(self.gaps, start)
        return i < len(self.gaps) and self.gaps[i] < end


@dataclass(frozen=True)
class Markout:
    fill: Fill
    side: str
    market: str
    horizon_s: int
    mid_at_fill: float
    mid_after: float

    @property
    def sign(self) -> int:
        return 1 if self.side == "bid" else -1

    @property
    def spread_c(self) -> float:
        return self.sign * (self.mid_at_fill - float(self.fill.price)) * 100

    @property
    def drift_c(self) -> float:
        return self.sign * (self.mid_after - self.mid_at_fill) * 100

    @property
    def markout_c(self) -> float:
        return self.sign * (self.mid_after - float(self.fill.price)) * 100

    @property
    def rebate_c(self) -> float:
        p = float(self.fill.price)
        return REBATE_COEF * p * (1 - p) * 100


def mark_out(
    states: Iterable[OrderState],
    tracker: MidTracker,
    *,
    mirror: Mapping[str, str] | None = None,
    horizons_s: Iterable[int] = HORIZONS_S,
) -> tuple[list[Markout], dict[int, int]]:
    """Markouts for every fill whose horizon mid is known, and per-horizon unmarked counts."""
    mirror = mirror or {}
    marks: list[Markout] = []
    unmarked: dict[int, int] = defaultdict(int)
    horizons = tuple(horizons_s)
    end = tracker.last_ts or 0
    for s in states:
        asset = s.order.asset_id
        market = min(asset, mirror.get(asset, asset))
        for f in s.fills:
            at_fill = float(f.mid) if f.mid is not None else None
            for h in horizons:
                target = f.venue_ts + h * 1000
                after = None
                if (
                    at_fill is not None
                    and target <= end
                    and not tracker.gap_between(f.venue_ts, target)
                ):
                    after = tracker.mid_at(asset, target)
                if at_fill is None or after is None:
                    unmarked[h] += 1
                    continue
                marks.append(Markout(f, s.order.side, market, h, at_fill, after))
    return marks, dict(unmarked)


@dataclass(frozen=True)
class MarkoutSummary:
    horizon_s: int
    fills: int
    markets: int
    shares: float
    spread_c: float  # all means are size-weighted, cents per share
    drift_c: float
    markout_c: float
    markout_se_c: float  # clustered by market
    rebate_c: float
    net_c: float  # markout + rebate (US per-fill schedule, the generous credit)
    net_ci95_c: tuple[float, float]

    def as_dict(self) -> dict[str, object]:
        return dict(self.__dict__)


def summarise_markouts(marks: Iterable[Markout], horizon_s: int) -> MarkoutSummary | None:
    rows = [m for m in marks if m.horizon_s == horizon_s]
    if not rows:
        return None
    w = [float(m.fill.size) for m in rows]
    total = sum(w)

    def mean(xs: list[float]) -> float:
        return sum(x * wi for x, wi in zip(xs, w, strict=True)) / total

    markout = [m.markout_c for m in rows]
    mu = mean(markout)
    # Cluster-robust SE of a weighted mean: residual sums per market.
    by_market: dict[str, float] = defaultdict(float)
    for m, x, wi in zip(rows, markout, w, strict=True):
        by_market[m.market] += wi * (x - mu)
    g = len(by_market)
    se = math.sqrt(sum(v * v for v in by_market.values()) * g / max(g - 1, 1)) / total
    rebate = mean([m.rebate_c for m in rows])
    net = mu + rebate
    return MarkoutSummary(
        horizon_s=horizon_s,
        fills=len(rows),
        markets=g,
        shares=round(total, 2),
        spread_c=round(mean([m.spread_c for m in rows]), 4),
        drift_c=round(mean([m.drift_c for m in rows]), 4),
        markout_c=round(mu, 4),
        markout_se_c=round(se, 4),
        rebate_c=round(rebate, 4),
        net_c=round(net, 4),
        net_ci95_c=(round(net - 1.96 * se, 4), round(net + 1.96 * se, 4)),
    )


@dataclass
class VariantResult:
    states: list[OrderState]
    marks: list[Markout]
    unmarked: dict[int, int]


def markout_recorded(
    root: Path,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    every_s: float = 300.0,
    life_s: float = 300.0,
    latency_ms: int = 250,
    chunk_h: float = 3.0,
    warmup_s: float = 600.0,
    horizons_s: tuple[int, ...] = HORIZONS_S,
) -> dict[str, VariantResult]:
    """Probe, simulate and mark out a long recording one chunk at a time.

    Replaying days at once holds every decrease and mid in memory (~1 GB per
    recorded hour on the current market set), so each chunk is replayed on
    its own: from `warmup_s` before its first probe, so books have resynced
    from snapshots (median ~13 s apart per asset), to the last probe's
    lifetime plus the longest horizon after it. Probes are placed only
    inside the chunk, so every probe is simulated exactly once.
    """
    span = venue_time_range(root, start=start, end=end)
    if span is None:
        return {}
    mirror = mirror_map(root)
    every, chunk = int(every_s * 1000), int(chunk_h * 3600_000)
    chunk -= chunk % every  # keep the probe grid aligned across chunks
    tail = int((life_s + max(horizons_s)) * 1000) + 60_000
    results: dict[str, VariantResult] = defaultdict(lambda: VariantResult([], [], defaultdict(int)))
    ws = span[0] + int(warmup_s * 1000)
    while ws < span[1]:
        we = min(ws + chunk, span[1])
        orders = [
            o
            for o in probe_orders(
                root,
                start_ms=ws - every,
                end_ms=we + int(life_s * 1000),
                every_s=every_s,
                life_s=life_s,
            )
            if o.place_ts < we
        ]
        lo, hi = _utc(ws - int(warmup_s * 1000)), _utc(we + tail)
        if start is not None:
            lo = max(lo, start)
        if end is not None:
            hi = min(hi, end)
        replayer = Replayer()
        tracker = MidTracker(replayer)
        sims = simulate_recorded(
            root, orders, latency_ms=latency_ms, start=lo, end=hi, replayer=replayer
        )
        for name, states in sims.items():
            marks, unmarked = mark_out(states, tracker, mirror=mirror, horizons_s=horizons_s)
            r = results[name]
            r.states.extend(states)
            r.marks.extend(marks)
            for h, n in unmarked.items():
                r.unmarked[h] += n
        log.info(
            "markout.chunk",
            start=_utc(ws).isoformat(),
            end=_utc(we).isoformat(),
            orders=len(orders),
        )
        ws = we
    return {name: VariantResult(r.states, r.marks, dict(r.unmarked)) for name, r in results.items()}


def _utc(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, UTC)
