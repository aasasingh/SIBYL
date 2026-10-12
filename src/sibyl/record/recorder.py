"""Record live Polymarket order books and trades over WebSocket.

Why this exists, and why it runs before anything else in Phase 5:

  * Every result in Phase 3 rests on TRADE TAPE prices -- prices somebody
    traded at, not prices we could have hit. That is unfixable
    retrospectively (the CLOB prunes price history at resolution), so the
    only route to a tape-free dataset is to record books from now on.
  * A maker strategy cannot be paper-traded honestly without recorded books
    PLUS the tape. Filling whenever the price touches your level is the most
    optimistic assumption available and would make any maker result fiction.
    Queue position needs every book update, which is why this subscribes to
    deltas rather than polling snapshots.
  * The data is time-locked. An hour not recorded is an hour lost.

Three durability properties, each of which matters more than any statistic
computed downstream:

  1. **Coverage is recorded, not assumed.** Every connection writes a segment
     to the coverage ledger as soon as it is subscribed, extends it with a
     heartbeat every `heartbeat_seconds`, and finalises it on close -- so a
     hard crash or sleep loses at most the last heartbeat interval, not the
     segment. The gaps between segments are where book state was unknown,
     and the fill model must refuse to simulate across them rather than
     interpolate. See `record/coverage.py`.
  2. **The subscription refreshes.** Markets are picked by *current*
     (24h) volume with a tradeable mid price -- see `choose_markets` for why
     not lifetime volume. Active markets resolve fast (sports within hours),
     so the asset set is re-selected every `refresh_hours` (default 6) and a
     new manifest written.
  3. **Nothing is buffered across a crash for longer than `flush_seconds`.**

The market channel delivers `book` (full ladder snapshot), `price_change`
(incremental level updates) and `last_trade_price` (executions). All are
stored verbatim.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import structlog
import websockets

from sibyl.record.coverage import CoverageWriter
from sibyl.record.writer import EventRow, EventWriter
from sibyl.venues.polymarket import PolymarketReadClient

log = structlog.get_logger(__name__)

WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

# Idle gaps are normal: a quiet market sends nothing for minutes. Only treat
# silence as a dead connection well past the library's own ping interval.
RECV_TIMEOUT_S = 90.0
PING_INTERVAL_S = 20.0
HEARTBEAT_S = 60.0


@dataclass
class RecorderStats:
    events: int = 0
    reconnects: int = 0
    refreshes: int = 0
    started: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_event: datetime | None = None
    by_type: dict[str, int] = field(default_factory=dict)


# Selection: where a maker could actually quote, and where trading happens now.
#
# The first recorder ranked by LIFETIME volume. Measured after 20h: 97.8% of
# recorded trades were one bot buying 5 @ 0.002, only 6 of 60 markets had a
# mid between 0.03 and 0.97, and most were long-dated longshots (2028
# nomination markets). A fill model needs fills; that set produced almost
# none. See research/fill-model-design.md.
SELECTION_ORDER = "volume24hr"
MID_RANGE = (Decimal("0.05"), Decimal("0.95"))
MAX_PER_EVENT = 3
CANDIDATE_PAGES = 5  # x100 candidates, ranked, before filtering


def _decimal(raw: Any) -> Decimal | None:
    try:
        return Decimal(str(raw)) if raw not in (None, "") else None
    except InvalidOperation:
        return None


def choose_markets(
    candidates: list[dict[str, Any]],
    *,
    n_markets: int = 60,
    max_assets: int = 150,
    mid_range: tuple[Decimal, Decimal] = MID_RANGE,
    max_per_event: int = MAX_PER_EVENT,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Filter ranked candidates down to a recordable set, preserving rank.

    Kept: order book enabled and accepting orders, a two-sided quote with
    mid inside `mid_range`, and at most `max_per_event` markets from any one
    event, so a single NegRisk set (30 candidates of one election) cannot
    fill every slot. Both legs of each market are subscribed: the YES and NO
    books are two views of one instrument and a fill model needs both.
    """
    assets: list[str] = []
    manifest: list[dict[str, Any]] = []
    per_event: dict[str, int] = {}
    for m in candidates:
        if len(manifest) >= n_markets or len(assets) + 2 > max_assets:
            break
        if not (m.get("enableOrderBook") and m.get("acceptingOrders")):
            continue
        bid, ask = _decimal(m.get("bestBid")), _decimal(m.get("bestAsk"))
        if bid is None or ask is None or bid <= 0 or ask >= 1 or ask <= bid:
            continue
        mid = (bid + ask) / 2
        if not mid_range[0] <= mid <= mid_range[1]:
            continue
        events = m.get("events") or []
        event_id = str(events[0].get("id")) if events else f"market:{m.get('id')}"
        if per_event.get(event_id, 0) >= max_per_event:
            continue
        raw = m.get("clobTokenIds")
        if not isinstance(raw, str):
            continue
        try:
            tokens = [str(t) for t in json.loads(raw)]
        except json.JSONDecodeError:
            continue
        if len(tokens) != 2:
            continue
        per_event[event_id] = per_event.get(event_id, 0) + 1
        assets.extend(tokens)
        manifest.append(
            {
                "market_id": str(m.get("id")),
                "condition_id": str(m.get("conditionId") or ""),
                "event_id": event_id,
                "question": str(m.get("question") or ""),
                "tokens": tokens,
                "volume": str(m.get("volumeNum") or ""),
                "volume24hr": str(m.get("volume24hr") or ""),
                "best_bid": str(bid),
                "best_ask": str(ask),
                "end_date": str(m.get("endDate") or ""),
            }
        )
    return assets, manifest


async def select_assets(
    client: PolymarketReadClient, *, n_markets: int = 60, max_assets: int = 150
) -> tuple[list[str], list[dict[str, Any]]]:
    """Pick currently-traded, mid-priced active markets; return their token ids."""
    candidates: list[dict[str, Any]] = []
    for page in range(CANDIDATE_PAGES):
        batch = await client.markets(
            active="true",
            closed="false",
            limit=100,
            offset=page * 100,
            order=SELECTION_ORDER,
            ascending="false",
        )
        candidates.extend(batch)
        if len(batch) < 100:
            break
    return choose_markets(candidates, n_markets=n_markets, max_assets=max_assets)


def _rows_from_message(raw: str, recv_ts: datetime) -> list[EventRow]:
    """Flatten one WS frame into event rows, preserving the payload verbatim."""
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        return [
            EventRow(
                recv_ts=recv_ts,
                venue_ts=0,
                event_type="unparseable",
                asset_id="",
                market="",
                payload=raw[:100_000],
            )
        ]
    events = decoded if isinstance(decoded, list) else [decoded]
    rows: list[EventRow] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        venue_ts_raw = event.get("timestamp")
        try:
            venue_ts = int(venue_ts_raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            venue_ts = 0
        rows.append(
            EventRow(
                recv_ts=recv_ts,
                venue_ts=venue_ts,
                event_type=str(event.get("event_type") or "unknown"),
                asset_id=str(event.get("asset_id") or ""),
                market=str(event.get("market") or ""),
                # Verbatim per-event JSON: lossless, and re-parseable alone.
                payload=json.dumps(event, separators=(",", ":")),
            )
        )
    return rows


def _write_manifest(
    data_dir: Path,
    assets: list[str],
    manifest: list[dict[str, Any]],
    version: int,
    session: str,
) -> None:
    """Record which assets were watched from when.

    Without this, an absence of events is ambiguous between "quiet market"
    and "never subscribed" -- and after a refresh, between "quiet" and "no
    longer in the subscription".
    """
    path = data_dir / "book_events" / f"manifest-{session}-v{version:03d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "session_id": session,
                "assets_version": version,
                "started": datetime.now(UTC).isoformat(),
                "selection": {
                    "order": SELECTION_ORDER,
                    "mid_range": [str(MID_RANGE[0]), str(MID_RANGE[1])],
                    "max_per_event": MAX_PER_EVENT,
                },
                "assets": assets,
                "markets": manifest,
            },
            indent=1,
        ),
        encoding="utf-8",
    )


async def record(
    *,
    data_dir: Path,
    hours: float | None = None,
    n_markets: int = 60,
    max_assets: int = 150,
    # Large enough that `flush_seconds` governs: at ~300 events/s on active
    # markets a 2000-row trigger wrote ~17k part files a day.
    flush_every: int = 50_000,
    flush_seconds: float = 60.0,
    log_every_s: float = 300.0,
    refresh_hours: float = 6.0,
    heartbeat_seconds: float = HEARTBEAT_S,
) -> RecorderStats:
    """Subscribe and record until `hours` elapses (or forever if None).

    Reconnects with backoff on any transport failure; a dropped connection
    costs a logged gap, not the session. The asset set is re-selected every
    `refresh_hours`.
    """
    writer = EventWriter(data_dir, flush_every=flush_every)
    coverage = CoverageWriter(data_dir)
    stats = RecorderStats()
    deadline = None if hours is None else time.monotonic() + hours * 3600

    log.info(
        "record.start",
        session=coverage.session_id,
        hours=hours,
        refresh_hours=refresh_hours,
        data_dir=str(data_dir),
    )
    assets_version = 0
    backoff = 1.0
    try:
        while deadline is None or time.monotonic() < deadline:
            # --- (re)select the asset set ---------------------------------
            async with PolymarketReadClient() as client:
                assets, manifest = await select_assets(
                    client, n_markets=n_markets, max_assets=max_assets
                )
            if not assets:
                raise RuntimeError("no active markets to subscribe to")
            assets_version += 1
            _write_manifest(data_dir, assets, manifest, assets_version, coverage.session_id)
            if assets_version > 1:
                stats.refreshes += 1
            log.info(
                "record.subscription",
                version=assets_version,
                assets=len(assets),
                markets=len(manifest),
            )

            refresh_at = time.monotonic() + refresh_hours * 3600
            cycle_end = refresh_at if deadline is None else min(refresh_at, deadline)

            # --- hold a connection until refresh or deadline --------------
            while time.monotonic() < cycle_end:
                segment = coverage.open_segment(assets_version=assets_version, n_assets=len(assets))
                events_at_open = stats.events
                reason, detail = "shutdown", ""
                last_flush = time.monotonic()
                last_log = time.monotonic()
                try:
                    async with websockets.connect(
                        WS_URL, open_timeout=20, ping_interval=PING_INTERVAL_S
                    ) as ws:
                        await ws.send(json.dumps({"assets_ids": assets, "type": "market"}))
                        coverage.mark_open(segment)
                        last_beat = time.monotonic()
                        backoff = 1.0
                        while time.monotonic() < cycle_end:
                            if time.monotonic() - last_beat >= heartbeat_seconds:
                                # Flush first: the heartbeat claims coverage
                                # up to now, so the data must already be on disk.
                                writer.flush()
                                last_flush = time.monotonic()
                                coverage.heartbeat(segment, n_events=stats.events - events_at_open)
                                last_beat = time.monotonic()
                            # Wake for the next heartbeat even if the feed is silent.
                            wait = max(
                                0.0,
                                min(
                                    RECV_TIMEOUT_S,
                                    heartbeat_seconds - (time.monotonic() - last_beat),
                                ),
                            )
                            try:
                                raw = await asyncio.wait_for(ws.recv(), timeout=wait)
                            except TimeoutError:
                                # Silence is normal; the library's pings
                                # police liveness. Flush and keep waiting.
                                writer.flush()
                                last_flush = time.monotonic()
                                continue
                            now = datetime.now(UTC)
                            text = raw if isinstance(raw, str) else raw.decode()
                            for row in _rows_from_message(text, now):
                                writer.add(row)
                                stats.events += 1
                                stats.by_type[row.event_type] = (
                                    stats.by_type.get(row.event_type, 0) + 1
                                )
                            stats.last_event = now
                            if time.monotonic() - last_flush > flush_seconds:
                                writer.flush()
                                last_flush = time.monotonic()
                            if time.monotonic() - last_log > log_every_s:
                                log.info(
                                    "record.progress",
                                    events=stats.events,
                                    rows_on_disk=writer.rows_written,
                                    buffered=writer.buffered,
                                    reconnects=stats.reconnects,
                                    refreshes=stats.refreshes,
                                    types=dict(sorted(stats.by_type.items())),
                                )
                                last_log = time.monotonic()
                        reason = "refresh" if time.monotonic() >= refresh_at else "deadline"
                except asyncio.CancelledError:
                    reason, detail = "shutdown", "cancelled"
                    raise
                except Exception as exc:
                    reason, detail = "error", f"{type(exc).__name__}: {exc}"
                    stats.reconnects += 1
                finally:
                    writer.flush()
                    coverage.close_segment(
                        segment,
                        reason=reason,
                        detail=detail,
                        n_events=stats.events - events_at_open,
                    )
                if reason == "error":
                    log.warning(
                        "record.reconnect",
                        error=detail,
                        attempt=stats.reconnects,
                        sleep=backoff,
                    )
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 60.0)
    finally:
        with contextlib.suppress(Exception):
            writer.flush()
        log.info(
            "record.stop",
            session=coverage.session_id,
            events=stats.events,
            rows_on_disk=writer.rows_written,
            parts=writer.parts_written,
            reconnects=stats.reconnects,
            refreshes=stats.refreshes,
            types=dict(sorted(stats.by_type.items())),
        )
    return stats
