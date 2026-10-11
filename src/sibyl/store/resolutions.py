"""Resolution ledger queries — the training-label source for calibration.

`calibration_rows` is the Phase 3 input: one row per (market, hour) with the
observed YES price, the settled outcome, and the horizon to resolution. The
horizon uses close_time (when the question's outcome window ends), not
settled_time (when the venue got around to paying) — a trader at time t faced
the close, and payment lag is a capital cost handled in costs/capital.py.
"""

from __future__ import annotations

import duckdb

# The moment the outcome became known, which is NOT simply close_time.
#
# Polymarket's gamma `endDate` (-> close_time) is frequently a far-future
# placeholder: measured on real data, the median market settles 112 days
# BEFORE its nominal close, and 9347 of 10564 settle early. Anchoring the
# horizon on close_time therefore inflated time-to-resolution by months and
# swept almost every row into the `1mo+` bucket, destroying the per-horizon
# structure the whole model depends on.
#
# `settled_time` (gamma `closedTime`) is when the market actually closed and
# resolved. LEAST() handles both directions: early resolution (settled first)
# and settlement lag after a real close.
RESOLUTION_TIME_SQL = "LEAST(m.close_time, COALESCE(m.settled_time, m.close_time))"

CALIBRATION_SQL = f"""
SELECT
    p.venue,
    p.market_id,
    -- Clustering/splitting unit. Markets inside one event share an outcome:
    -- in a NegRisk set exactly one resolves YES, so the others are
    -- mechanically implied. Measured on real data: 10.66 markets per event,
    -- largest event 122 markets. Treating markets as independent overstated
    -- the evidence by roughly an order of magnitude.
    COALESCE(NULLIF(m.event_id, ''), m.market_id) AS cluster_id,
    m.domain,
    p.ts,
    COALESCE(p.price, (p.bid + p.ask) / 2) AS yes_price,
    (m.resolution = 'yes')::INTEGER AS y,
    epoch({RESOLUTION_TIME_SQL} - p.ts) / 86400.0 AS horizon_days,
    {RESOLUTION_TIME_SQL} AS close_time,
    p.source
FROM prices p
JOIN markets m USING (venue, market_id)
WHERE m.resolution IN ('yes', 'no')
  AND p.ts < {RESOLUTION_TIME_SQL}
  AND COALESCE(p.price, (p.bid + p.ask) / 2) IS NOT NULL
  AND COALESCE(p.price, (p.bid + p.ask) / 2) > 0
  AND COALESCE(p.price, (p.bid + p.ask) / 2) < 1
"""

# Spec §2.3.4 horizon buckets (days).
HORIZON_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("0-1h", 0.0, 1 / 24),
    ("1-3h", 1 / 24, 3 / 24),
    ("3-6h", 3 / 24, 6 / 24),
    ("6-12h", 6 / 24, 12 / 24),
    ("12-24h", 12 / 24, 1.0),
    ("24-48h", 1.0, 2.0),
    ("2d-1w", 2.0, 7.0),
    ("1w-1mo", 7.0, 30.0),
    ("1mo+", 30.0, float("inf")),
)


def calibration_rows(con: duckdb.DuckDBPyConnection) -> duckdb.DuckDBPyRelation:
    return con.sql(CALIBRATION_SQL)


def resolved_market_counts(
    con: duckdb.DuckDBPyConnection,
) -> list[tuple[str, str, int, int]]:
    """(venue, domain, n_resolved, n_with_prices) per cell."""
    rows = con.execute(
        """
        SELECT m.venue, m.domain,
               count(*) AS n_resolved,
               count(*) FILTER (WHERE p.market_id IS NOT NULL) AS n_with_prices
        FROM markets m
        LEFT JOIN (SELECT DISTINCT venue, market_id FROM prices) p
          USING (venue, market_id)
        WHERE m.resolution IN ('yes', 'no')
        GROUP BY 1, 2
        ORDER BY 1, 2
        """
    ).fetchall()
    return [(r[0], r[1], r[2], r[3]) for r in rows]


def resolution_lag(
    con: duckdb.DuckDBPyConnection,
) -> list[tuple[str, int, float, float, float, float]]:
    """Measured settlement lag per venue, in days.

    Returns (venue, n, median, p90, p99, max). This is the empirical basis for
    `costs.capital.SETTLEMENT_BUFFER` — which shipped as a guess. Ranking an
    APY on the nominal close time overstates returns on exactly the markets
    most likely to be disputed, so the buffer should be a high percentile of
    this distribution, not its mean.

    Negative lags are excluded: some venues stamp `closedTime` before the
    nominal `close_time` when a market resolves early, which is a different
    phenomenon (early resolution) and must not be averaged into payment lag.
    """
    rows = con.execute(
        """
        SELECT venue,
               count(*) AS n,
               median(lag_days) AS p50,
               quantile_cont(lag_days, 0.90) AS p90,
               quantile_cont(lag_days, 0.99) AS p99,
               max(lag_days) AS worst
        FROM (
            SELECT venue, epoch(settled_time - close_time) / 86400.0 AS lag_days
            FROM markets
            WHERE settled_time IS NOT NULL
              AND close_time IS NOT NULL
              AND resolution IN ('yes', 'no')
        )
        WHERE lag_days >= 0
        GROUP BY venue
        ORDER BY venue
        """
    ).fetchall()
    return [(r[0], int(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])) for r in rows]


def early_resolution_rate(con: duckdb.DuckDBPyConnection) -> list[tuple[str, int, float]]:
    """(venue, n, fraction settling BEFORE nominal close).

    Early resolution is capital returned sooner than modelled — good for APY,
    but it means `close_time` is not a reliable lockup end in either
    direction, so a hold-to-resolution APY needs both tails acknowledged.
    """
    rows = con.execute(
        """
        SELECT venue, count(*) AS n,
               avg(CASE WHEN settled_time < close_time THEN 1.0 ELSE 0.0 END) AS early_frac
        FROM markets
        WHERE settled_time IS NOT NULL AND close_time IS NOT NULL
          AND resolution IN ('yes', 'no')
        GROUP BY venue ORDER BY venue
        """
    ).fetchall()
    return [(r[0], int(r[1]), float(r[2])) for r in rows]


def horizon_bucket(days: float) -> str:
    for name, low, high in HORIZON_BUCKETS:
        if low <= days < high:
            return name
    return HORIZON_BUCKETS[-1][0]
