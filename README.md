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
| 4 | Semantic identity graph | 🔻 descoped — cross-venue execution is unavailable |
| 5 | Alpha, sizing, paper trading | 🔄 recorder running w/ coverage; [fill model](research/fill-model-design.md) steps 1-3 built (replay 99.98% validated, attribution, queue sim); markouts need days of data |
| 6 | Live trading | 🚫 closed by construction (Ontario) |

### Phase 6 is closed by construction

The operator is an Ontario resident. Kalshi does not serve Canadians, and
Polymarket operates under a permanent OSC undertaking not to serve Ontario
residents (geo-blocked in Alberta, BC and Quebec too). `Settings` defaults to
`operator_province="ON"` and **raises if `live=True`** — this is enforced,
not just documented. Sources:
[research/jurisdiction.md](research/jurisdiction.md).

This costs the project less than it sounds. Spec §9 is explicit that the real
deliverable is Point 6 — an honest paper-trading write-up that names its own
dead hypotheses. **Phase 5 is the finish line**, and a system that reliably
tells you it has no edge is the thing worth building.

One consequence worth stating plainly: cross-venue arbitrage (§2.3.2), the
spec's best-documented inefficiency, is **unexecutable** here. The parity
conditions and semantic layer stay as tested research instruments.

### Phase 3 result: no tradeable recalibration edge

Fitted on 159,182 hourly rows / 1,977 markets / 1,017 events. The population
slope resembles the published research (mu = 1.203, CI [1.082, 1.324]), but
**out-of-sample Brier improvement is -0.0030, positive in 2 of 4 folds** —
the recalibration does not beat the raw quoted price, stably across 3/4/6/8
folds. Of 45 testable cells, **4 clear the tradeability bar against ~2.2
expected by chance**, i.e. the multiple-testing null.

The claim is "no *usable* edge found", not "no edge exists". Measurement-error
attenuation was measured rather than assumed (noise share 1.8%, no cell
crosses b=1) and a power experiment shows a true slope >= 1.3 would still
have been detected. Clustering and fold splits are at the **event** level,
because NegRisk sets share an outcome (10.66 markets per event here).

Write-up, objections addressed, and falsifiable next steps:
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
uv run sibyl eiv                        # measurement-error attenuation check
uv run sibyl record                     # record live books+trades (run continuously)
uv run sibyl coverage                   # uptime + every gap where book state was unknown
uv run sibyl replay                     # rebuild recorded books, validate vs venue snapshots
uv run sibyl attribute                  # split book decreases into trades vs cancels
uv run sibyl probe --start <iso>        # simulated maker fills under each queue rule
uv run sibyl power --seeds 80           # what effect size could the test detect?
uv run sibyl tables                     # regenerate the write-up's figures
uv run python research/check_figures.py # verify every prose figure against artifacts
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

## Open item: the recorder is not yet durable

`sibyl record` currently runs as a local process. It survives dropped
connections (logged as coverage gaps) and refreshes its market set daily, but
it does **not** survive the laptop sleeping or restarting — and the data is
time-locked, so every hour it is down is unrecoverable. What *is* now
crash-safe is the record of the outage: each segment is written on connect
and heartbeated every minute, so a hard kill or sleep still shows up in
`sibyl coverage` as a gap starting at the last heartbeat.

It should run somewhere that stays on. Reading public market data is not
trading, so jurisdiction does not block hosting it; confirm the WebSocket is
reachable from wherever it runs. `uv run sibyl coverage` reports uptime and
every gap, which is the way to check it is actually working.

## Configuration

Environment variables, prefix `SIBYL_`, nested with `__`:

```
SIBYL_POLYMARKET__PLATFORM=intl     # us | intl — never both
SIBYL_KALSHI__ENABLED=false         # jurisdiction; demo reads still work
SIBYL_RISK__DAILY_LOSS_LIMIT_USD=50
```

No secrets in git, ever. The kill switch is the `KILL_SWITCH` file: `touch`
it to halt every trading path immediately.
