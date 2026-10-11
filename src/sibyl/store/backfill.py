"""Backfill resolved markets + price paths into the ledger (Phase 2).

Source realities discovered by probing (2026-07-21):
  - Kalshi: /events?status=settled carries category + series_ticker + nested
    markets with result. Candlesticks serve history for settled markets, in
    *_dollars string fields. Prod requires auth even for reads; demo is open
    but its prices are paper-market noise — fine for pipeline tests, useless
    for calibration.
  - Polymarket: gamma /markets?closed=true is open, but the CLOB prunes
    /prices-history at resolution. The data-api trade tape survives, so price
    paths for resolved markets are rebuilt from the tape as hourly last-trade
    bars (with §2.4's caveat: tape prices, not hittable quotes).

Parsers are pure functions over decoded JSON; orchestrators do bounded,
rate-limited IO and write through the deduping ledger.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any

import structlog

from sibyl.semantic.classify import domain_from_kalshi_category, domain_from_polymarket_tags
from sibyl.store.ledger import Ledger, MarketRecord, PriceRecord
from sibyl.venues.kalshi import KalshiClient
from sibyl.venues.normalise import Venue
from sibyl.venues.polymarket import PolymarketReadClient

log = structlog.get_logger(__name__)

BINARY_RESULTS = frozenset({"yes", "no"})


def parse_ts(raw: object) -> datetime | None:
    """Tolerant ISO parser for venue timestamps ('...Z', '... 00:49:16+00')."""
    if raw is None or not isinstance(raw, str) or not raw:
        return None
    text = raw.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    return dt.astimezone(UTC) if dt.tzinfo else dt.replace(tzinfo=UTC)


def _dec(raw: object) -> Decimal | None:
    if raw is None:
        return None
    if isinstance(raw, float):
        raise TypeError(f"float reached the money path: {raw!r}")
    try:
        return Decimal(str(raw))
    except InvalidOperation:
        return None


# -- Kalshi ---------------------------------------------------------------


def parse_kalshi_event(event: dict[str, Any], *, min_close: datetime) -> list[MarketRecord]:
    """Binary settled markets from one /events entry (category lives here)."""
    category = event.get("category")
    domain = domain_from_kalshi_category(category)
    records: list[MarketRecord] = []
    for market in event.get("markets") or []:
        result = market.get("result")
        if result not in BINARY_RESULTS:
            continue  # scalar / MVE / voided markets are not calibration labels
        close_time = parse_ts(market.get("close_time"))
        if close_time is None or close_time < min_close:
            continue
        records.append(
            MarketRecord(
                venue=Venue.KALSHI,
                market_id=str(market["ticker"]),
                event_id=str(event.get("event_ticker") or ""),
                series_id=str(event.get("series_ticker") or "") or None,
                title=str(market.get("title") or event.get("title") or ""),
                domain=domain,
                category_raw=category,
                open_time=parse_ts(market.get("open_time")),
                close_time=close_time,
                settled_time=parse_ts(market.get("settlement_ts"))
                or parse_ts(market.get("settled_time")),
                resolution=result,
                neg_risk=bool(event.get("mutually_exclusive", False)),
            )
        )
    return records


def parse_kalshi_candles(payload: dict[str, Any], *, market_id: str) -> list[PriceRecord]:
    """Hourly bars from a candlesticks response. Trade-price close is absent
    in hours without trades; bid/ask closes are kept so the row still carries
    information (mid is derivable)."""
    records: list[PriceRecord] = []
    for candle in payload.get("candlesticks") or []:
        end_ts = candle.get("end_period_ts")
        if not isinstance(end_ts, int):
            continue
        price = (candle.get("price") or {}).get("close_dollars")
        bid = (candle.get("yes_bid") or {}).get("close_dollars")
        ask = (candle.get("yes_ask") or {}).get("close_dollars")
        records.append(
            PriceRecord(
                venue=Venue.KALSHI,
                market_id=market_id,
                ts=datetime.fromtimestamp(end_ts, tz=UTC),
                price=_dec(price),
                bid=_dec(bid),
                ask=_dec(ask),
                volume=_dec(candle.get("volume_fp")),
                source="kalshi_candle",
            )
        )
    return records


async def backfill_kalshi(
    client: KalshiClient,
    ledger: Ledger,
    *,
    months: int = 6,
    max_pages: int = 50,
    page_size: int = 200,
    candle_markets: int = 0,
) -> dict[str, int]:
    """Pull settled binary markets (and optionally candles) into the ledger."""
    min_close = datetime.now(UTC) - timedelta(days=30 * months)
    markets_written = 0
    all_records: list[MarketRecord] = []
    cursor: str | None = None
    for page in range(max_pages):
        params: dict[str, Any] = {
            "status": "settled",
            "with_nested_markets": "true",
            "limit": page_size,
        }
        if cursor:
            params["cursor"] = cursor
        payload = await client.events(**params)
        events = payload.get("events") or []
        for event in events:
            all_records.extend(parse_kalshi_event(event, min_close=min_close))
        cursor = payload.get("cursor") or None
        log.info("kalshi.backfill.page", page=page, events=len(events), kept=len(all_records))
        if not cursor or not events:
            break
    markets_written = ledger.write_markets(all_records)

    prices_written = 0
    if candle_markets > 0:
        already = ledger.priced_market_ids(source="kalshi_candle")
        todo = [
            r for r in all_records if r.series_id and (r.venue.value, r.market_id) not in already
        ][:candle_markets]
        for record in todo:
            assert record.close_time is not None and record.series_id is not None
            start = record.open_time or (record.close_time - timedelta(days=90))
            payload = await client.candlesticks(
                record.series_id,
                record.market_id,
                start_ts=int(start.timestamp()),
                end_ts=int(record.close_time.timestamp()),
                period_interval=60,
            )
            rows = parse_kalshi_candles(payload, market_id=record.market_id)
            prices_written += ledger.write_prices(rows)
        log.info("kalshi.backfill.candles", markets=len(todo), rows=prices_written)
    return {"markets": markets_written, "price_rows": prices_written}


# -- Polymarket -----------------------------------------------------------


def _json_list(raw: object) -> list[Any]:
    """Gamma serialises list fields as JSON-encoded strings."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def parse_gamma_market(
    market: dict[str, Any], *, event: dict[str, Any] | None = None
) -> MarketRecord | None:
    """One resolved binary market from gamma. `event` supplies tags (domain)
    and the negRiskAugmented flag; markets fetched outside an event context
    fall back to whatever `events[0]` embeds (usually tagless)."""
    outcomes = [str(o) for o in _json_list(market.get("outcomes"))]
    prices = [str(p) for p in _json_list(market.get("outcomePrices"))]
    tokens = [str(t) for t in _json_list(market.get("clobTokenIds"))]
    if len(outcomes) != 2 or len(prices) != 2 or {o.casefold() for o in outcomes} != {"yes", "no"}:
        return None
    yes_idx = 0 if outcomes[0].casefold() == "yes" else 1
    yes_price, no_price = prices[yes_idx], prices[1 - yes_idx]
    if {yes_price, no_price} != {"1", "0"}:
        return None  # not cleanly resolved (disputed / 50-50'd / still open)
    resolution = "yes" if yes_price == "1" else "no"

    if event is None:
        embedded = market.get("events") or []
        event = embedded[0] if embedded else {}
    assert event is not None
    tags = [t.get("label", "") for t in (event.get("tags") or []) if isinstance(t, dict)]
    domain = domain_from_polymarket_tags(tags or None)

    return MarketRecord(
        venue=Venue.POLYMARKET_INTL,  # gamma serves the international book; see backfill notes
        market_id=str(market["id"]),
        event_id=str(event.get("id") or "") or None,
        series_id=str(market.get("conditionId") or "") or None,
        title=str(market.get("question") or ""),
        domain=domain,
        category_raw=",".join(tags) or None,
        open_time=parse_ts(market.get("startDate")) or parse_ts(market.get("createdAt")),
        close_time=parse_ts(market.get("endDate")),
        settled_time=parse_ts(market.get("closedTime")),
        resolution=resolution,
        neg_risk=bool(market.get("negRisk", False)) or bool(event.get("negRisk", False)),
        neg_risk_augmented=bool(event.get("negRiskAugmented", False)),
        yes_token=tokens[yes_idx] if len(tokens) == 2 else None,
        no_token=tokens[1 - yes_idx] if len(tokens) == 2 else None,
    )


def parse_gamma_event(event: dict[str, Any]) -> list[MarketRecord]:
    """All resolved binary markets nested in one gamma /events entry."""
    return [
        rec
        for market in event.get("markets") or []
        if (rec := parse_gamma_market(market, event=event)) is not None
    ]


def tape_to_hourly_bars(
    trades: Iterable[dict[str, Any]], *, venue: Venue, market_id: str
) -> list[PriceRecord]:
    """Collapse the trade tape into hourly last-trade YES-price bars.

    Trades on the NO outcome are converted (yes = 1 - price). Tape order is
    newest-first from the API; we bucket by hour and keep the latest trade in
    each bucket, summing volume.
    """
    buckets: dict[int, tuple[int, Decimal, Decimal]] = {}  # hour -> (ts, yes_price, vol)
    for trade in trades:
        ts_raw = trade.get("timestamp")
        price = _dec(trade.get("price"))
        size = _dec(trade.get("size")) or Decimal("0")
        outcome = str(trade.get("outcome") or "").casefold()
        if not isinstance(ts_raw, int) or price is None or outcome not in BINARY_RESULTS:
            continue
        yes_price = price if outcome == "yes" else Decimal("1") - price
        hour = ts_raw - (ts_raw % 3600)
        existing = buckets.get(hour)
        if existing is None:
            buckets[hour] = (ts_raw, yes_price, size)
        else:
            last_ts, last_price, vol = existing
            if ts_raw > last_ts:
                last_ts, last_price = ts_raw, yes_price
            buckets[hour] = (last_ts, last_price, vol + size)
    return [
        PriceRecord(
            venue=venue,
            market_id=market_id,
            ts=datetime.fromtimestamp(hour + 3600, tz=UTC),  # bar close time
            price=yes_price,
            volume=vol,
            source="poly_tape",
        )
        for hour, (_, yes_price, vol) in sorted(buckets.items())
    ]


async def backfill_polymarket_markets(
    client: PolymarketReadClient,
    ledger: Ledger,
    *,
    months: int = 6,
    max_pages: int = 20,
    page_size: int = 500,
) -> int:
    """Pull resolved binary markets from gamma into the ledger.

    Flushes each page as it lands: bounded memory and a killed run keeps
    everything it already fetched.
    """
    min_close = datetime.now(UTC) - timedelta(days=30 * months)
    written = 0
    cursor: str | None = None
    for page in range(max_pages):
        # Keyset paging: offset paging 422s past ~2100 rows, and gamma's own
        # error points at /events/keyset for depth. Page size is capped
        # server-side (observed 100 for limit=500), which is fine here.
        events, cursor = await client.gamma_events_keyset(
            cursor=cursor,
            closed="true",
            limit=page_size,
            end_date_min=min_close.strftime("%Y-%m-%dT%H:%M:%SZ"),
            order="endDate",
            ascending="false",
        )
        if not events:
            break
        page_records: list[MarketRecord] = []
        for event in events:
            page_records.extend(parse_gamma_event(event))
        written += ledger.write_markets(page_records)
        if page % 10 == 0 or not cursor:
            log.info(
                "poly.backfill.page",
                page=page,
                events=len(events),
                parsed=len(page_records),
                new_total=written,
            )
        if not cursor:
            break
    return written


async def backfill_polymarket_tape(
    client: PolymarketReadClient,
    ledger: Ledger,
    *,
    tape_markets: int,
    tape_pages_per_market: int = 6,
    flush_every: int = 25,
) -> dict[str, int]:
    """Rebuild price paths for resolved markets from the data-api trade tape.

    Candidates come from the LEDGER (not just this run's fetch) in a stable
    pseudo-random order, so the training sample spreads across time and
    domain instead of clustering on the most recently resolved events, and so
    a resumed run continues rather than re-walking the same head of the list.
    """
    candidates = ledger.tape_candidates(Venue.POLYMARKET_INTL, limit=tape_markets)
    buffer: list[PriceRecord] = []
    rows_written = 0
    done = 0
    failed = 0

    def flush() -> None:
        nonlocal buffer, rows_written
        if buffer:
            rows_written += ledger.write_prices(buffer)
            buffer = []

    for market_id, condition_id in candidates:
        tape: list[dict[str, Any]] = []
        try:
            for page in range(tape_pages_per_market):
                batch = await client.trades(condition_id, limit=500, offset=page * 500)
                tape.extend(batch)
                if len(batch) < 500:
                    break
        except Exception as exc:
            failed += 1
            log.warning("poly.tape.market_failed", market_id=market_id, error=str(exc))
            continue
        buffer.extend(tape_to_hourly_bars(tape, venue=Venue.POLYMARKET_INTL, market_id=market_id))
        done += 1
        if done % flush_every == 0:
            flush()
            log.info("poly.tape.progress", markets=done, rows=rows_written, failed=failed)
    flush()
    log.info("poly.tape.done", markets=done, rows=rows_written, failed=failed)
    return {"markets": done, "price_rows": rows_written, "failed": failed}


async def backfill_polymarket(
    client: PolymarketReadClient,
    ledger: Ledger,
    *,
    months: int = 6,
    max_pages: int = 20,
    page_size: int = 500,
    tape_markets: int = 0,
    tape_pages_per_market: int = 6,
) -> dict[str, int]:
    """Markets then price paths. See the two functions above."""
    markets_written = await backfill_polymarket_markets(
        client, ledger, months=months, max_pages=max_pages, page_size=page_size
    )
    result = {"markets": markets_written, "price_rows": 0, "failed": 0}
    if tape_markets > 0:
        tape_result = await backfill_polymarket_tape(
            client,
            ledger,
            tape_markets=tape_markets,
            tape_pages_per_market=tape_pages_per_market,
        )
        result["price_rows"] = tape_result["price_rows"]
        result["failed"] = tape_result["failed"]
    return result
