"""DuckDB + Parquet warehouse (spec §4: store/ledger.py).

Two tables, each a directory of immutable Parquet part-files:

  markets — one row per market with resolution + domain label (the training
            label source once resolution ∈ {yes, no})
  prices  — one row per (market, hour): the YES price path. `source` records
            provenance: kalshi_candle | poly_tape | snapshot. Tape-derived
            prices carry the §2.4 caveat — they are trades someone made, not
            quotes you could have hit.

Writes dedupe against what's already on disk; part-files are never rewritten,
so a crashed backfill never corrupts previous data (§3.6 spirit: append-only).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict

from sibyl.semantic.classify import Domain
from sibyl.venues.normalise import Venue

PRICE_TYPE = pa.decimal128(12, 6)
SIZE_TYPE = pa.decimal128(24, 6)
TS_TYPE = pa.timestamp("us", tz="UTC")

_PRICE_Q = Decimal("0.000001")

MARKETS_SCHEMA = pa.schema(
    [
        ("venue", pa.string()),
        ("market_id", pa.string()),
        ("event_id", pa.string()),
        ("series_id", pa.string()),
        ("title", pa.string()),
        ("domain", pa.string()),
        ("category_raw", pa.string()),
        ("open_time", TS_TYPE),
        ("close_time", TS_TYPE),
        ("settled_time", TS_TYPE),
        ("resolution", pa.string()),
        ("neg_risk", pa.bool_()),
        ("neg_risk_augmented", pa.bool_()),
        ("yes_token", pa.string()),
        ("no_token", pa.string()),
    ]
)

PRICES_SCHEMA = pa.schema(
    [
        ("venue", pa.string()),
        ("market_id", pa.string()),
        ("ts", TS_TYPE),
        ("price", PRICE_TYPE),  # YES trade price (close of the hour), if any
        ("bid", PRICE_TYPE),
        ("ask", PRICE_TYPE),
        ("volume", SIZE_TYPE),
        ("source", pa.string()),
    ]
)


class MarketRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    venue: Venue
    market_id: str
    event_id: str | None = None
    series_id: str | None = None
    title: str
    domain: Domain
    category_raw: str | None = None
    open_time: datetime | None = None
    close_time: datetime | None = None
    settled_time: datetime | None = None
    resolution: str | None = None  # 'yes' | 'no' | venue-specific otherwise
    neg_risk: bool = False
    # Augmented NegRisk sets have redefinable placeholder outcomes (§2.1) —
    # never treat one as a static partition.
    neg_risk_augmented: bool = False
    yes_token: str | None = None
    no_token: str | None = None


class PriceRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    venue: Venue
    market_id: str
    ts: datetime
    price: Decimal | None = None
    bid: Decimal | None = None
    ask: Decimal | None = None
    volume: Decimal | None = None
    source: str


def _q6(value: Decimal | None) -> Decimal | None:
    return None if value is None else value.quantize(_PRICE_Q)


def _utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        raise ValueError(f"naive datetime in ledger record: {dt}")
    return dt.astimezone(UTC)


class Ledger:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.markets_dir = root / "markets"
        self.prices_dir = root / "prices"
        self.markets_dir.mkdir(parents=True, exist_ok=True)
        self.prices_dir.mkdir(parents=True, exist_ok=True)

    # -- write paths ------------------------------------------------------

    def write_markets(self, records: list[MarketRecord]) -> int:
        """Append market rows, skipping (venue, market_id) already stored."""
        if not records:
            return 0
        seen = self.market_ids()
        fresh = [r for r in records if (r.venue.value, r.market_id) not in seen]
        # dedupe within the batch too
        unique: dict[tuple[str, str], MarketRecord] = {
            (r.venue.value, r.market_id): r for r in fresh
        }
        if not unique:
            return 0
        rows = [
            {
                **r.model_dump(mode="python"),
                "venue": r.venue.value,
                "domain": r.domain.value,
                "open_time": _utc(r.open_time),
                "close_time": _utc(r.close_time),
                "settled_time": _utc(r.settled_time),
            }
            for r in unique.values()
        ]
        self._write_part(self.markets_dir, rows, MARKETS_SCHEMA)
        return len(rows)

    def write_prices(self, records: list[PriceRecord]) -> int:
        """Append price rows. Caller is expected to backfill whole markets at
        a time and skip markets already priced (`priced_market_ids`).

        Prices/volumes are quantized to 6 dp for storage — research data, not
        the money path; fee math never reads these columns."""
        if not records:
            return 0
        rows = [
            {
                **r.model_dump(mode="python"),
                "venue": r.venue.value,
                "ts": _utc(r.ts),
                "price": _q6(r.price),
                "bid": _q6(r.bid),
                "ask": _q6(r.ask),
                "volume": _q6(r.volume),
            }
            for r in records
        ]
        self._write_part(self.prices_dir, rows, PRICES_SCHEMA)
        return len(rows)

    @staticmethod
    def _write_part(directory: Path, rows: list[dict[str, Any]], schema: pa.Schema) -> None:
        table = pa.Table.from_pylist(rows, schema=schema)
        name = f"part-{datetime.now(UTC):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:8]}.parquet"
        pq.write_table(table, directory / name)

    # -- read paths -------------------------------------------------------

    def connect(self) -> duckdb.DuckDBPyConnection:
        """In-memory DuckDB with `markets` and `prices` views over Parquet."""
        con = duckdb.connect()
        for view, directory, schema in (
            ("markets", self.markets_dir, MARKETS_SCHEMA),
            ("prices", self.prices_dir, PRICES_SCHEMA),
        ):
            if any(directory.glob("*.parquet")):
                pattern = (directory / "*.parquet").as_posix()
                con.execute(f"CREATE VIEW {view} AS SELECT * FROM read_parquet('{pattern}')")
            else:
                empty = pa.Table.from_pylist([], schema=schema)
                con.register(f"_empty_{view}", empty)
                con.execute(f"CREATE VIEW {view} AS SELECT * FROM _empty_{view}")
        return con

    def market_ids(self) -> set[tuple[str, str]]:
        with self.connect() as con:
            rows = con.execute("SELECT venue, market_id FROM markets").fetchall()
        return {(v, m) for v, m in rows}

    def priced_market_ids(self, source: str | None = None) -> set[tuple[str, str]]:
        query = "SELECT DISTINCT venue, market_id FROM prices"
        params: list[str] = []
        if source is not None:
            query += " WHERE source = ?"
            params.append(source)
        with self.connect() as con:
            rows = con.execute(query, params).fetchall()
        return {(v, m) for v, m in rows}

    def tape_candidates(self, venue: Venue, *, limit: int) -> list[tuple[str, str]]:
        """(market_id, condition_id) for resolved markets that still need a
        price path, in a stable pseudo-random order.

        `hash(market_id)` ordering is deterministic, so a resumed backfill
        continues through the same sequence, and the sample is spread across
        resolution dates and domains rather than being the most-recent head
        of the list — a time-clustered training sample would make the
        walk-forward split meaningless.
        """
        with self.connect() as con:
            rows = con.execute(
                """
                SELECT m.market_id, m.series_id
                FROM markets m
                LEFT JOIN (
                    SELECT DISTINCT venue, market_id FROM prices WHERE source = 'poly_tape'
                ) p USING (venue, market_id)
                WHERE m.venue = ?
                  AND m.resolution IN ('yes', 'no')
                  AND m.series_id IS NOT NULL
                  AND m.close_time IS NOT NULL
                  AND p.market_id IS NULL
                ORDER BY hash(m.market_id)
                LIMIT ?
                """,
                [venue.value, limit],
            ).fetchall()
        return [(str(r[0]), str(r[1])) for r in rows]

    def compact(self, min_parts: int = 50) -> dict[str, int]:
        """Rewrite each table's parts into one file. Many small Parquet files
        slow every later read; a long backfill produces hundreds."""
        merged: dict[str, int] = {}
        for name, directory, schema in (
            ("markets", self.markets_dir, MARKETS_SCHEMA),
            ("prices", self.prices_dir, PRICES_SCHEMA),
        ):
            parts = sorted(directory.glob("*.parquet"))
            if len(parts) < min_parts:
                merged[name] = 0
                continue
            table = pq.read_table(directory, schema=schema)
            tmp = directory / f"_compact-{uuid.uuid4().hex[:8]}.parquet.tmp"
            pq.write_table(table, tmp)
            for part in parts:
                part.unlink()
            tmp.rename(directory / f"part-compacted-{datetime.now(UTC):%Y%m%dT%H%M%S}.parquet")
            merged[name] = len(parts)
        return merged

    def summary(self) -> list[tuple[str, str, str | None, int, int]]:
        """(venue, domain, resolution, n_markets, n_price_rows) rollup."""
        with self.connect() as con:
            result = con.execute(
                """
                SELECT m.venue, m.domain, m.resolution,
                       count(DISTINCT m.market_id) AS n_markets,
                       count(p.ts) AS n_price_rows
                FROM markets m
                LEFT JOIN prices p USING (venue, market_id)
                GROUP BY 1, 2, 3
                ORDER BY 1, 2, 3
                """
            ).fetchall()
        return [(r[0], r[1], r[2], r[3], r[4]) for r in result]
