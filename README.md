# SIBYL

A prediction-market fair-value & edge engine for **Polymarket**.
Full design: [SIBYL-build-prompt.md](SIBYL-build-prompt.md).

> The Sibyl gave true prophecies in ambiguous form. Market prices are the same:
> informative, but not the probability they appear to be. This system's job is
> to translate.

## Status

| Phase | Scope | State |
|---|---|---|
| 0 | Skeleton: config, venue clients, kill switch, logging | ✅ |
| 1 | Ingestion & cost model (fees, depth walk, APY, parity) | ✅ |
| 2 | Historical store & resolution ledger | ✅ |
| 3 | Calibration engine | ✅ built, run, and **hypothesis killed** — [findings](research/phase3-findings.md) |
| 4 | Semantic identity graph | ⬜ needs an Anthropic API key |
| 5 | Alpha, sizing, paper trading | ⬜ **the finish line** |
| 6 | Live trading | 🚫 out of scope — see below |

### Phase 6 is out of scope, deliberately

The operator is in Canada. Kalshi does not serve Canadian residents, and
Polymarket is banned in Ontario (OSC enforcement, to at least 2027) and
geo-blocked in Alberta, BC and Quebec. Details and sources:
[research/jurisdiction.md](research/jurisdiction.md).

This costs the project less than it sounds. Spec §9 is explicit that the real
deliverable is Point 6 — an honest paper-trading write-up that names its own
dead hypotheses. **Phase 5 is the finish line**, and a system that reliably
tells you it has no edge is the thing worth building.

One consequence worth stating plainly: cross-venue arbitrage (§2.3.2), the
spec's best-documented inefficiency, is **unexecutable** here. The parity
conditions and semantic layer stay as tested research instruments.

### Phase 3 result: no tradeable recalibration edge

Fitted on 159,182 hourly rows across 1,977 resolved markets. The population
slope looks like the published research (mu = 1.179, CI [1.066, 1.292]), but
**walk-forward out-of-sample improvement is -0.0066, positive in only 1 of 4
folds** — the recalibration does not beat the raw price. Of 47 testable
cells, 5 clear the tradeability bar against ~2.4 expected by chance.

So the system declines to trade the signal, which is the designed behaviour.
Write-up, caveats and falsifiable next steps:
[research/phase3-findings.md](research/phase3-findings.md).

The most useful by-product: a mid-range "+15pp inefficiency" in the first run
turned out to be row-counting. Counted per market it collapses to ≈0 —
Polymarket is well calibrated between 0.35 and 0.65.

## What it does today

```bash
uv run sibyl status                     # config, venue eligibility, connectivity
uv run sibyl backfill --venue polymarket --max-pages 400 --price-markets 3000
uv run sibyl store                      # ledger rollup
uv run sibyl lag                        # measured settlement lag
uv run sibyl calibrate --out report.txt # fit slopes, walk forward, report honestly
uv run sibyl compact                    # merge Parquet parts (not while backfilling)

uv run pytest                           # offline suite
uv run pytest -m live                    # live connectivity checks
uv run mypy && uv run ruff check .
```

## Things that are true and easy to get wrong

Each of these is enforced by a test, and most were found by running the code
against real data rather than by reading the spec.

- **Two Polymarket venues, two incompatible fee schedules.** US: taker 0.06
  with a per-fill maker rebate, cent rounding. International: per-category
  taker coefficients (geopolitics is *free*, crypto 0.07, politics 0.04),
  5-decimal rounding, makers never charged. Crossing them invents edge.
- **INTL maker rebates are not a cost reduction.** They are a daily pro-rata
  share of the taker-fee pool, unknowable per fill — modelled as unmodelled
  upside, never subtracted from effective cost.
- **`endDate` is not the resolution date.** It is usually a far-future
  placeholder; 88% of markets settle before it, median 112 days early.
  Horizons anchor on `LEAST(close_time, settled_time)`.
- **Settlement lag was measured, and the guess was 7× too optimistic.**
  p90 is 13.9 days, not the 2 days originally assumed.
- **Resolved markets have no CLOB price history** — it is pruned at
  resolution. Paths are rebuilt from the trade tape, which means they are
  prices someone traded at, not prices you could have hit.
- **Rows are not independent observations.** Hourly rows within a market are
  one bet counted many times, so CIs come from a market-clustered bootstrap
  and reliability bins report market-level frequencies. A row-level read of
  the same data showed a +15pp "inefficiency" that was a few dozen markets.
- **A cell whose slope CI straddles b = 1 is not tradeable**, and the system
  says so rather than emitting a small positive edge.
- **gamma `/events` offset paging dies at ~2100 rows**; deep pagination needs
  the `/events/keyset` cursor endpoint.

Full, measured caveats: [research/data-limitations.md](research/data-limitations.md).

## Configuration

Environment variables, prefix `SIBYL_`, nested with `__`:

```
SIBYL_POLYMARKET__PLATFORM=intl     # us | intl — never both
SIBYL_KALSHI__ENABLED=false         # jurisdiction; demo reads still work
SIBYL_RISK__DAILY_LOSS_LIMIT_USD=50
```

No secrets in git, ever. The kill switch is the `KILL_SWITCH` file: `touch`
it to halt every trading path immediately.
