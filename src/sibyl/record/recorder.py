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
     to the coverage ledger on close. The gaps between segments are where
     book state was unknown, and the fill model must refuse to simulate
     across them rather than interpolate. See `record/coverage.py`.
  2. **The subscription refreshes.** Markets are picked by liquidity at
     startup; within a week some have resolved and new liquid markets exist
     that a long-lived process would never watch. The asset set is
     re-selected every `refresh_hours` and a new manifest written.
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


@dataclass
class RecorderStats:
    events: int = 0
    reconnects: int = 0
    refreshes: int = 0
    started: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_event: datetime | None = None
    by_type: dict[str, int] = field(default_factory=dict)


async def select_assets(
    client: PolymarketReadClient, *, n_markets: int = 60, max_assets: int = 150
) -> tuple[list[str], list[dict[str, Any]]]:
    """Pick liquid active markets and return their outcome-token ids.

    Both legs of each market are subscribed: the YES and NO books are two
    views of one instrument and a fill model needs both sides.
    """
    markets = await client.markets(
        active="true",
        closed="false",
        limit=n_markets,
        order="volumeNum",
        ascending="false",
    )
    assets: list[str] = []
    manifest: list[dict[str, Any]] = []
    for m in markets:
        raw = m.get("clobTokenIds")
        if not isinstance(raw, str):
            continue
        try:
            tokens = [str(t) for t in json.loads(raw)]
        except json.JSONDecodeError:
            continue
        if len(tokens) != 2:
            continue
        if len(assets) + 2 > max_assets:
            break
        assets.extend(tokens)
        manifest.append(
            {
                "market_id": str(m.get("id")),
                "condition_id": str(m.get("conditionId") or ""),
                "question": str(m.get("question") or ""),
                "tokens": tokens,
                "volume": str(m.get("volumeNum") or ""),
                "end_date": str(m.get("endDate") or ""),
            }
        )
    return assets, manifest


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
    flush_every: int = 2000,
    flush_seconds: float = 60.0,
    log_every_s: float = 300.0,
    refresh_hours: float = 24.0,
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
                        backoff = 1.0
                        while time.monotonic() < cycle_end:
                            try:
                                raw = await asyncio.wait_for(ws.recv(), timeout=RECV_TIMEOUT_S)
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
