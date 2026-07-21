"""Resolution ledger queries — the training-label source for calibration.

`calibration_rows` is the Phase 3 input: one row per (market, hour) with the
observed YES price, the settled outcome, and the horizon to resolution. The
horizon uses close_time (when the question's outcome window ends), not
settled_time (when the venue got around to paying) — a trader at time t faced
the close, and payment lag is a capital cost handled in costs/capital.py.
"""

from __future__ import annotations

import duckdb

CALIBRATION_SQL = """
SELECT
    p.venue,
    p.market_id,
    m.domain,
    p.ts,
    COALESCE(p.price, (p.bid + p.ask) / 2) AS yes_price,
    (m.resolution = 'yes')::INTEGER AS y,
    epoch(m.close_time - p.ts) / 86400.0 AS horizon_days,
    p.source
FROM prices p
JOIN markets m USING (venue, market_id)
WHERE m.resolution IN ('yes', 'no')
  AND p.ts < m.close_time
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


def horizon_bucket(days: float) -> str:
    for name, low, high in HORIZON_BUCKETS:
        if low <= days < high:
            return name
    return HORIZON_BUCKETS[-1][0]
