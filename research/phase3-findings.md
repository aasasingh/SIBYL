# Phase 3 findings — the recalibration hypothesis does not survive

Reproduce with:

```bash
uv run sibyl backfill --venue polymarket --months 24 --max-pages 400 --price-markets 3000
uv run sibyl calibrate --n-boot 400 --n-folds 4 --out research/calibration-report.txt
```

Data: **159,182 hourly rows across 1,977 resolved Polymarket International
markets**, 63 (domain x horizon) cells. Full output in
[calibration-report.txt](calibration-report.txt).

---

## Verdict

**Phase 3's acceptance criterion is NOT met.** The spec requires the fitted
recalibration to beat the raw-price baseline on out-of-sample Brier score.
It does not:

| fold | train markets | test markets | Brier (model) | Brier (raw price) | improvement |
|---|---|---|---|---|---|
| 0 | 790 | 296 | 0.090575 | 0.072392 | **-0.018183** |
| 1 | 1086 | 296 | 0.103149 | 0.086196 | **-0.016953** |
| 2 | 1382 | 296 | 0.118917 | 0.112881 | **-0.006035** |
| 3 | 1678 | 296 | 0.095546 | 0.110225 | +0.014678 |

Mean improvement **-0.0066**, positive in **1 of 4** folds. Recalibrating made
the forecast *worse* in three of four out-of-sample blocks.

Per spec §6 — "Cells whose CI straddles b = 1 are not tradeable. Say so." —
this is said. **No directional recalibration signal should be traded from
this fit.** The hypothesis of §2.3.4 is, on this data, dead.

---

## What the in-sample fit says, and why it is not enough

The global picture matches the published research suspiciously well. The
population slope is

> **mu = 1.179, 95% CI [1.066, 1.292]**, between-cell sd tau = 0.163

which excludes 1 and sits inside the spec's reported range (mean slope rising
0.99 -> 1.32 with horizon). Taken alone this says prices are mildly
compressed toward 0.5, i.e. favourites are underpriced — the headline result
of §2.3.4, apparently replicated on a different venue.

It does not survive contact with out-of-sample testing. Two reasons, both
visible in the report:

**1. Almost every cell's apparent signal is borrowed.** Of 47 cells with a
usable interval, 13 had a CI excluding 1 while shrinkage pulled the slope
they would actually trade back to ~mu. Example:

| cell | own b | slope actually traded | shrinkage |
|---|---|---|---|
| sports/6-12h | 2.252 | 1.184 | 0.99 |
| other/2d-1w | 2.719 | 1.211 | 0.98 |
| crypto/3-6h | 1.953 | 1.188 | 0.99 |

A shrinkage of 0.99 means the cell's own data contributed ~1% of the
estimate. These are not thirteen independent discoveries; they are thirteen
restatements of mu wearing cell-specific confidence intervals. This was a
genuine bug in the first run — they were all reported TRADEABLE — and fixing
the gate cut the count from 18 to 5.

**2. Five survivors is roughly chance.** 47 cells carry intervals, so at 95%
confidence about **2.4** clear the bar by luck. Five survived:

| cell | b traded | shrinkage | direction |
|---|---|---|---|
| other/3-6h | 1.363 | 0.53 | compressed |
| sports/1-3h | 1.334 | 0.45 | compressed |
| crypto/12-24h | 1.285 | 0.77 | compressed |
| politics/1mo+ | 1.253 | 0.80 | compressed |
| crypto/1w-1mo | 0.861 | 0.35 | over-extended |

Five versus an expected 2.4 is not nothing, but it is not a finding either —
especially as `other` is an acknowledged grab-bag, and the walk-forward says
whatever structure exists does not generalise forward.

Worth noting against the spec's prior: **politics/1mo+ at b = 1.253** is the
one survivor that matches a published expectation (politics compressed at
long horizons), though far weaker than the quoted 1.73-1.83. And
**crypto/1w-1mo at b = 0.861** points the *opposite* way to the spec's 1.09.

---

## The most useful thing we learned: the mid-range inefficiency was an artifact

The first run's reliability table showed observed outcomes running **+15
percentage points above** the quoted price in the mid-range. That would be the
largest inefficiency ever recorded in this asset class. It was an artifact of
counting rows instead of bets.

| price bin | mean price | gap (row-level) | gap (market-level) | rows | markets |
|---|---|---|---|---|---|
| 0.35 | 0.3474 | **+0.1046** | **-0.0067** | 6,283 | 411 |
| 0.45 | 0.4431 | **+0.1569** | **-0.0138** | 5,985 | 396 |
| 0.55 | 0.5501 | **+0.1559** | **+0.0010** | 6,797 | 401 |
| 0.65 | 0.6559 | +0.0862 | -0.0248 | 6,521 | 347 |

Count each market once and the mid-range gap collapses to roughly zero.
**Polymarket is well calibrated between 0.35 and 0.65.** The row-level gap
came from long-lived markets contributing hundreds of correlated rows each.

This is exactly the trap spec §8 lists ("trade-level bootstrap CIs on
serially correlated within-market trades") arriving through a side door — not
in the CI, but in a descriptive statistic. Every interval in this report is
therefore market-clustered, and the reliability table reports both views.

### A caveat on the extremes, stated rather than resolved

The market-level column shows a large *negative* gap at the top: markets
priced ~0.96 resolve YES only ~85% of the time at market level (gap -0.111).
Read naively that says favourites are badly *over*priced — the reverse of the
longshot bias.

Do not read it naively. Neither column is the right statistic:

- **Row-level** over-weights markets that sat at a price for a long time.
- **Market-level** (first row per market per bin) counts a market that merely
  *touched* 0.96 for an hour before collapsing the same as one that sat there
  for 500 hours and won.

The truth is between the two, and measuring it properly needs duration
weighting that neither column implements. The honest summary is: the
mid-range is well calibrated under any weighting, and the extremes are
*unresolved by this analysis*.

---

## Why this result is weaker evidence than it looks, in both directions

From [data-limitations.md](data-limitations.md), the things that could be
hiding a real effect or manufacturing a fake one:

1. **Prices are a trade tape, not a quote book.** The CLOB prunes price
   history at resolution, so paths are rebuilt from executed trades. These are
   prices someone traded at, not prices we could have hit — spec §2.4's exact
   warning. A fit against this is an upper bound on a tradeable edge.
2. **17.7% of markets are missing their first half of life** (newest-first
   tape, capped at 3,000 trades/market). Long-horizon cells are selectively
   drawn from less-traded markets, so `1w-1mo` and `1mo+` are the least
   trustworthy rows in the table — and `politics/1mo+` is one of five
   survivors.
3. **One venue.** The spec's slopes are Kalshi; this is Polymarket
   International. A disagreement is not necessarily a failed replication —
   it may be two different markets.
4. **Keyword domain labels.** A cell may contain markets that do not belong
   in it. `other` carrying two of five survivors is a warning sign.

Point 1 deserves emphasis: the measured out-of-sample result is *negative*
despite the data being biased in the optimistic direction.

---

## What this changes downstream

- **Phase 5 must not ship a directional recalibration strategy as its primary
  alpha.** `apply.recalibrate` already refuses to emit a signal for a
  non-tradeable cell, so the system fails safe: with this fit loaded, it
  declines almost everything. That is the correct behaviour, not a bug.
- **The parity/arbitrage path is unaffected** — it depends on prices and the
  cost model, not on a forecast. It is also unexecutable cross-venue from this
  jurisdiction (see [jurisdiction.md](jurisdiction.md)), leaving within-market
  and negRisk-basket conditions on a single venue.
- **The maker-rebate idea (§2.3.1) is untouched by this result** and is now
  the strongest remaining candidate, because it does not require being right
  about anything. On Polymarket International, though, the rebate is a daily
  pro-rata pool share rather than a per-fill credit, so its value cannot be
  known at decision time — which makes the §5.6 markout discipline the thing
  Phase 5 should actually build.

---

## Falsifiable next steps, in order of expected value

1. **Deepen the tape** (`tape_pages_per_market` up from 6) and re-run. If the
   long-horizon cells change materially, the current numbers were a sampling
   artifact. Cheap, purely API time.
2. **Duration-weighted reliability** to settle the extremes question above.
3. **Restrict to liquid markets** (volume floor) and re-fit. If an edge
   exists only in illiquid markets, it is not capacity-bearing anyway, and
   this would show that directly.
4. **Re-fit on quote mid-prices for *live* markets going forward.** The
   tape-vs-quote problem is unfixable retrospectively, but recording books
   from now on builds a clean dataset. This is the only route to a result
   that is not an upper bound.

A system that reliably tells you it has no edge is worth more than one that
tells you it has one. This is that, in writing.
