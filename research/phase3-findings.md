# Phase 3 findings — no usable recalibration edge found

Reproduce with:

```bash
uv run sibyl backfill --venue polymarket --months 24 --max-pages 400 --price-markets 3000
uv run sibyl calibrate --n-boot 400 --n-folds 4 --out research/calibration-report.txt
uv run sibyl eiv            # measurement-error attenuation
```

Data: **159,182 hourly rows / 1,977 resolved markets / 1,017 events** on
Polymarket International, 63 (domain x horizon) cells. Full output in
[calibration-report.txt](calibration-report.txt).

---

## Verdict

**Phase 3's acceptance criterion is not met.** The metric is the **Brier
score** (mean squared error of the probability forecast), compared against the
raw quoted price as baseline, out of sample.

| fold | train markets | test markets | Brier (model) | Brier (raw price) | improvement |
|---|---|---|---|---|---|
| 0 | 818 | 391 | 0.086083 | 0.068416 | **-0.017668** |
| 1 | 1209 | 258 | 0.120694 | 0.103355 | **-0.017339** |
| 2 | 1467 | 274 | 0.103078 | 0.109519 | +0.006441 |
| 3 | 1741 | 235 | 0.087649 | 0.104249 | +0.016600 |

Mean improvement **-0.0030**, positive in **2 of 4** folds. Stable under
re-splitting: -0.0050 (3 folds), -0.0053 (6), -0.0049 (8).

The defensible claim is **"no usable edge found"**, not "no edge exists" —
see the power section below.

Per spec §6, the cells are named: of 45 cells carrying an interval, **4 clear
the tradeability bar against ~2.2 expected by chance at 95% confidence.** That
is not a finding. No directional recalibration signal should be traded from
this fit, and `apply.recalibrate` accordingly declines almost everything.

---

## Three objections, addressed

### 1. "A negative result on optimistic data is a stronger negative" — wrong as stated

That argument holds for a P&L backtest: if measured profit on prices you
could not have hit is zero, real profit is zero. It does **not** hold for a
calibration fit, because noisy prices are an errors-in-variables problem.
Classical measurement error in the regressor attenuates its coefficient
*toward zero*:

> plim b_hat = lambda * b_true,  where lambda = Var(x_true) / Var(x_observed)

So tape noise pushes the slope toward "no compression". A real b > 1 could be
measured as b ~ 1, and the kill could be a false negative. The earlier
write-up asserted the bias cut one way; that was incorrect.

So it was measured rather than argued, two ways
([calibration/eiv.py](../src/sibyl/calibration/eiv.py)).

**(a) How big is the noise?** For a random walk plus independent noise,
`Var(x_{t+k} - x_t) = k*sigma2_signal + 2*sigma2_noise`, so regressing the
mean squared k-step difference on k identifies the noise from the intercept.
On the real hourly bars (1,475 series, regression r2 = 0.95):

| quantity | value |
|---|---|
| noise sd in logit(p) | **0.434** |
| Var(observed logit p) | **10.32** |
| reliability lambda | **0.982** |
| noise share of regressor variance | **1.8%** |

Attenuation is real but **small**, and the reason is structural: the
regressor's variance is dominated by genuine *cross-market* price dispersion
(logit spans about +/-4.6 from 1c to 99c), which dwarfs *within-market*
measurement noise. Per-cell lambda runs 0.90-1.00, so corrections move slopes
by 1-3%: politics/1mo+ 1.544 -> 1.567, finance/1w-1mo 1.836 -> 1.857.
**Zero cells cross b = 1 after correction** (`sibyl eiv`).

**(b) Could noise still have destroyed out-of-sample detectability?**
Attenuation also inflates prediction variance, which the lambda arithmetic
does not capture. So: generate synthetic markets with a *known* slope, add
logit-space noise at the measured 0.434, and run the same walk-forward.

| true b | noise sd | fitted b | OOS improvement | folds + | verdict |
|---|---|---|---|---|---|
| 1.0 | 0.000 | 1.072 | -0.00054 | 1/4 | kill |
| 1.0 | 0.434 | 0.813 | +0.00105 | 4/4 | *false positive* |
| 1.3 | 0.434 | 1.152 | +0.00073 | 3/4 | detected |
| 1.6 | 0.434 | 1.328 | +0.00218 | 3/4 | detected |
| 2.0 | 0.434 | 1.541 | +0.00362 | 4/4 | detected |

At the measured noise level a true slope of **1.3 or above would still have
been detected**, so the kill is not plausibly explained by attenuation. The
observed real result (-0.0030, 2/4) resembles the *b = 1* row, not any
genuine-signal row.

Two honest caveats on that table. The synthetic design is not the real
data's twin (different event structure, row counts, price distribution), so
it is a power check of the right order of magnitude, not a calibrated power
curve. And the `b = 1, noise 0.434` row produced a spurious positive — a
reminder that improvements of ~0.001 are inside the noise of this procedure.

**(c) The distinction that actually matters.** For *prediction from an
equally noisy input*, the attenuated coefficient is the **correct** one: if a
noisy price is what you observe at trade time, `b_hat` is what maps it to a
calibrated probability, and EIV correction would make the forecast worse.
Correction answers *"is the market really compressed?"* (inference), not
*"what should I multiply this quote by?"* (prediction).

The real problem is therefore not attenuation but a **mismatch**: trained on
tape noise, deployed against quote noise. Those have different variances, so
neither `b_hat` nor `b_corrected` is the right deployment coefficient, and no
arithmetic on tape data can produce it. Only recorded books can — which is
why `sibyl record` is now running.

### 2. Clustering was at the wrong level — fixed, and it changed the answer

Confidence intervals and fold splits were clustered on the **market**. They
should be clustered on the **event**, because markets inside an event share
an outcome: in a NegRisk set exactly one market resolves YES, so the rest are
mechanically implied. This is not an edge case here:

| quantity | value |
|---|---|
| resolved markets | 10,846 |
| distinct events | 1,017 |
| markets per event (mean) | **10.66** |
| largest single event | **122 markets** |
| markets flagged negRisk | 7,929 of 10,846 |

Both uses were corrected: the bootstrap resamples whole events, and the
walk-forward splits whole events. The split mattered independently of the
CIs — splitting a mutually-exclusive set across folds hands the test fold its
own answer ("train on A won" resolves "did B win?").

Measured effect on the intervals (36 cells, 300 bootstrap draws each):
**widened in 22 cells, narrowed in 14, median width ratio 1.041.** Verdicts
changed in three cells; `politics/3-6h` went from [1.02, 3.24] to
[0.95, 8.80] and lost tradeability.

Two things worth being precise about:

- The magnitude is modest (~4% median widening), not the order of magnitude
  a naive "10x fewer independent units" argument would predict.
- **Event clustering is not universally wider.** Synthetic designs showed
  both directions: when price variation lives *within* an event and the
  outcome is shared, each event is an informative within-event comparison and
  bundling can *tighten* the interval. The test suite deliberately asserts
  the counting and no-leakage invariants rather than a direction
  (`TestEventLevelClustering`).

Net effect on the headline: tradeable cells **5 -> 4**, and `politics/1mo+` —
the one survivor that matched a published prior — dropped out.

### 3. Four folds cannot detect a small effect

Named plainly: the metric is **Brier score**, and **four folds have low power
against a small effect.** From the table in section 1(b), genuine signals of
this size produce OOS improvements of roughly +0.001 to +0.004, while
fold-to-fold variation in the real data spans -0.018 to +0.017 — several
times the size of the effect being looked for. The procedure
can distinguish a large effect from none; it cannot resolve an effect worth
less than about a thousandth of a Brier point.

What survives that limitation: the result is **stable in sign** across 3, 4,
6 and 8 folds (-0.0050, -0.0030, -0.0053, -0.0049), and the tradeable-cell
count (4) is at the multiple-testing null (2.2). Those two together support
"no usable edge found." Neither supports "no edge exists," and the write-up
does not claim it.

---

## The most useful by-product: a mid-range inefficiency that was not real

The first run's reliability table showed observed outcomes running **+15
percentage points above** the quoted price mid-range — which would be the
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
came from long-lived markets contributing hundreds of correlated rows each —
spec §8's serial-correlation trap arriving through a descriptive statistic
rather than through a CI.

### The extremes remain unresolved, deliberately

The market-level column shows a large *negative* gap at the top: markets
priced ~0.96 resolve YES only ~85% of the time by that measure (-0.111).
Read naively that says favourites are badly *over*priced, reversing the
longshot bias. Do not read it naively — neither column is the right statistic:

- **Row-level** over-weights markets that sat at a price for a long time.
- **Market-level** counts a market that merely *touched* 0.96 for an hour
  before collapsing the same as one that sat there 500 hours and won.

The truth is between them and needs duration weighting that neither
implements. Honest summary: the mid-range is well calibrated under any
weighting; the extremes are not resolved by this analysis.

---

## Remaining weaknesses, in both directions

From [data-limitations.md](data-limitations.md):

1. **Prices are a trade tape, not a quote book** — unfixable retrospectively,
   and now quantified in section 1 rather than hand-waved.
2. **17.7% of markets are missing their first half of life** (newest-first
   tape, capped at 3,000 trades/market), so long-horizon cells are
   selectively drawn from less-traded markets.
3. **One venue.** The spec's slopes are Kalshi; this is Polymarket
   International. Disagreement may be two different markets, not a failed
   replication.
4. **Keyword domain labels**, so a cell may contain markets that do not
   belong in it. `other` holding two of four survivors is a warning sign.

---

## What this changes downstream

- **Phase 5 must not ship directional recalibration as its primary alpha.**
  The system already fails safe.
- **The maker-rebate idea (§2.3.1) is untouched by this result** and is the
  strongest remaining candidate, because it does not require being right
  about anything. But it is also the hardest thing to paper-trade honestly:
  the entire question is adverse selection, and a simulator that fills you
  whenever the price touches your level assumes the best possible case. A
  credible Phase 5 needs a fill model built from recorded books plus the
  trade tape, with queue position. See
  [fill-model-design.md](fill-model-design.md).
- **`sibyl record` is running** and is the prerequisite for both that fill
  model and any tape-free re-fit.
- **Phase 4 is descoped**: cross-venue identity is pointless when
  cross-venue execution is unavailable. A small subset-relation graph
  *within* Polymarket still feeds single-venue parity arbs, and is worth
  doing later, if at all.

## Falsifiable next steps

1. **Record books continuously** (running). The only route to a result that
   is not an upper bound, and the only fix for the noise mismatch in 1(c).
2. **Build the fill model** against recorded data before simulating any
   maker strategy.
3. **Deepen the tape** (`tape_pages_per_market` above 6) and re-run; if
   long-horizon cells move materially, those numbers were a sampling artifact.
4. **Duration-weighted reliability**, to settle the extremes question.
5. **Liquidity-floored re-fit.** An edge that exists only in illiquid markets
   is not capacity-bearing anyway.

A system that reliably tells you it has no usable edge is worth more than one
that tells you it has one. This is that, in writing.
