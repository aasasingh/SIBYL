# Data limitations — measured, not assumed

Spec §2.4 is blunt that a backtest filling at historical trade prices is
fiction. This file records what is actually wrong with SIBYL's training data,
measured on the real ledger, so no later result gets over-read. Every number
here came from querying the warehouse, not from estimation.

## 1. Price paths are a trade tape, not a quote book

Polymarket's CLOB **deletes `/prices-history` at resolution** (verified: an
active market returns its path, a resolved one returns an empty array). The
only surviving record for a resolved market is the `data-api` trade tape, so
price paths are rebuilt as hourly last-trade bars.

Consequence, and it is the central one: **these are prices at which somebody
traded, not prices you could have hit.** A fitted edge against this series is
an upper bound on the real thing. The spec's own warning applies to our data
exactly as it applies to the published research.

## 2. Heavy markets are truncated to their final trades

The tape is returned **newest-first** (confirmed: descending timestamps), and
the backfill caps each market at 6 pages x 500 = 3,000 trades. A market with
more trades than that keeps only its **most recent** 3,000.

Measured over 1,289 markets with full span data:

| Metric | Value |
|---|---|
| Markets at/near the bar cap | 180 (14%) |
| Median gap between market open and first captured bar | 0.74 days |
| Markets missing their entire first half of life | 17.7% |

So the median market is captured nearly in full, but roughly one in six is
missing its early life. Because early life is the long-horizon part, the
`1w-1mo` and `1mo+` cells are **selectively drawn from less-traded markets**.
Treat long-horizon slopes as the least trustworthy in the report. Raising
`tape_pages_per_market` and re-running fixes this at the cost of API time.

## 3. `endDate` is not the resolution date

Gamma's `endDate` is frequently a far-future placeholder:

| Metric | Value |
|---|---|
| Markets with both timestamps | 10,564 |
| Median `settled_time - close_time` | **-112.6 days** |
| Markets settling BEFORE nominal close | 9,347 (88%) |

Anchoring time-to-resolution on `endDate` inflated every horizon by months
and swept nearly all rows into `1mo+`, silently destroying the per-horizon
structure the whole model is built on. Horizons are now anchored on
`LEAST(close_time, settled_time)` (`RESOLUTION_TIME_SQL`).

This also means **`costs.capital.SETTLEMENT_BUFFER` cannot be calibrated from
this field**: the timestamps measure "market closed early", not "venue paid
late". `sibyl lag` reports the distribution, but the positive tail is the only
part relevant to capital lockup, and the clean way to measure payment lag is
to observe our own settlements in paper trading. The 2-day UMA buffer stays a
documented guess rather than being replaced by a number that looks measured
but answers a different question.

## 4. Outcome base rate is not 50%, and rows over-weight YES markets

| Metric | Value |
|---|---|
| Markets resolving YES | 218 of 965 (22.6%) |
| Rows from YES-resolving markets | 32.0% |
| Mean rows per YES market | 117.5 |
| Mean rows per NO market | 72.9 |

YES-resolving markets carry ~1.6x more rows than NO-resolving ones, so the
row-level YES rate (32%) overstates the market-level rate (22.6%). The
logistic fit conditions on price so this is not fatal, but any unconditional
row-level statistic is biased and must not be quoted as a base rate.

This is also why the reliability table reports **market-level** frequencies
and distinct market counts per bin. A row-level read of the first run showed
observed outcomes running +15pp above quoted price in the mid-range — which
would be the largest inefficiency ever recorded in this asset class, and is
really a few dozen markets counted hundreds of times each.

## 5. One venue only

Kalshi is unavailable from Canada (see `jurisdiction.md`), so there is no
Kalshi history and the fit is **Polymarket International only**. The
published slopes in spec §2.3.4 are Kalshi figures, so they are a prior and a
sanity check, never a comparison baseline. Cross-venue divergence (§2.3.2)
cannot be measured here at all.

## 6. Domain labels are keyword-derived, not LLM-classified

`semantic/classify.py` maps gamma tags to the six spec domains by keyword.
Mislabels were found and fixed by auditing what landed in `other` (bare
"Culture" alone was 107 markets), but it remains a heuristic. The spec's
~20-category LLM classifier is Phase 4 work; until then a domain cell may
contain markets that do not belong to it.
