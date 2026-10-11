"""Load calibration rows from the ledger into NumPy arrays.

This is the one place where the warehouse's Decimal columns become floats.
Everything downstream is statistics; nothing downstream touches money.
"""

from __future__ import annotations

import numpy as np

from sibyl.calibration.walkforward import Rows
from sibyl.store.ledger import Ledger
from sibyl.store.resolutions import CALIBRATION_SQL, HORIZON_BUCKETS
from sibyl.venues.normalise import Venue

# SQL-side bucketing: the horizon label is derived in DuckDB so we never ship
# a Python loop over millions of rows.
_BUCKET_CASE = "\n".join(
    f"WHEN horizon_days >= {low} AND horizon_days < {high} THEN '{name}'"
    for name, low, high in HORIZON_BUCKETS
    if high != float("inf")
)
_LAST_BUCKET = HORIZON_BUCKETS[-1][0]


def load_rows(
    ledger: Ledger,
    *,
    venue: Venue | None = None,
    min_horizon_days: float = 0.0,
    max_horizon_days: float = 400.0,
    exclude_domains: tuple[str, ...] = (),
) -> Rows:
    """Pull (price, outcome, horizon bucket, domain, market) rows.

    `max_horizon_days` guards against the long tail of multi-year markets,
    whose price paths are dominated by a few stale quotes.
    """
    query = f"""
    WITH base AS ({CALIBRATION_SQL})
    SELECT
        domain,
        CASE {_BUCKET_CASE} ELSE '{_LAST_BUCKET}' END AS horizon,
        CAST(yes_price AS DOUBLE) AS price,
        CAST(y AS DOUBLE) AS y,
        market_id,
        cluster_id,
        epoch(close_time) AS close_epoch
    FROM base
    WHERE horizon_days >= {min_horizon_days}
      AND horizon_days <= {max_horizon_days}
    """
    params: list[str] = []
    if venue is not None:
        query += " AND venue = ?"
        params.append(venue.value)
    if exclude_domains:
        placeholders = ", ".join("?" for _ in exclude_domains)
        query += f" AND domain NOT IN ({placeholders})"
        params.extend(exclude_domains)

    with ledger.connect() as con:
        table = con.execute(query, params).fetch_arrow_table()

    return Rows(
        domain=np.asarray(table["domain"].to_pylist(), dtype=np.str_),
        horizon=np.asarray(table["horizon"].to_pylist(), dtype=np.str_),
        price=np.asarray(table["price"].to_pylist(), dtype=np.float64),
        y=np.asarray(table["y"].to_pylist(), dtype=np.float64),
        market_id=np.asarray(table["market_id"].to_pylist(), dtype=np.str_),
        cluster_id=np.asarray(table["cluster_id"].to_pylist(), dtype=np.str_),
        close_time=np.asarray(table["close_epoch"].to_pylist(), dtype=np.float64),
    )
