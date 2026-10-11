# BUILD SPEC — **SIBYL**
### A prediction-market fair-value & edge engine, in Python

> *The Sibyl gave true prophecies in ambiguous form. Market prices are the same: informative, but not the probability they appear to be. This system's job is to translate.*

---

## 0. Your role

You are a quantitative developer building a research-grade trading system for prediction markets (Polymarket + Kalshi). You have full autonomy over implementation details. You do **not** have autonomy over the domain facts in §2 — those were verified in July 2026 and supersede your training data, which is almost certainly stale on fees, SDKs, and auth.

Work in phases (§6). Each phase ships something testable. Do not write the trading loop before the cost model is unit-tested against the published fee tables.

---

## 1. What you are building

A system that answers three questions, in order:

1. **What is this market actually worth?** — convert a quoted price into a recalibrated probability with an uncertainty band.
2. **Are two markets the same market?** — build a cross-venue identity graph (equivalent / subset / mutually-exclusive-partition) so that prices can legally be compared at all.
3. **Is there a trade after costs?** — detect edge net of the fee curve, spread, slippage, and *capital lockup to resolution*, then size it.

Plus a paper-trading harness that proves or kills each hypothesis before a dollar moves.

This is **not** an "LLM reads the news and predicts the future" bot. Those lose to fees. See §2.4.

---

## 2. Domain briefing (verified July 2026 — treat as ground truth)

### 2.1 Venue mechanics

**Polymarket**

- CLOB **V2** went live 2026-04-28. V1-signed orders no longer work in production. Use `py-clob-client-v2` (installs as `py-clob-client-v2`, imports as `py_clob_client_v2`). A unified beta SDK exists — `pip install --pre polymarket-client`, imports as `from polymarket import PublicClient, AsyncPublicClient` — treat it as beta and pin it only for read paths.
- Four REST surfaces, each its own host and data model:
  - `gamma-api.polymarket.com` — events, markets, tags, series, search. Public, no auth. Primary discovery surface.
  - `clob.polymarket.com` — orderbook, midpoint, spread, `/prices-history`. Public reads; authenticated writes.
  - `data-api.polymarket.com` — positions, trades, holders, open interest, leaderboards. Public.
  - `bridge.polymarket.com` — deposits/withdrawals (third-party proxy).
  - WebSocket layered on CLOB for book + price deltas.
- Auth is two-level: **L1** = EIP-712 wallet signature → `create_or_derive_api_key()` (renamed from V1's `create_or_derive_api_creds`); **L2** = HMAC with `ApiCreds(api_key, api_secret, api_passphrase)`. Signature types: `EOA=0, POLY_PROXY=1, POLY_GNOSIS_SAFE=2, POLY_1271=3`. Chain 137 (Polygon); 80002 (Amoy) for testnet.
- CLOB V2 collateral is **pUSD**, not raw USDC. Do not hardcode USDC assumptions.
- Rate limits: ~4,000/10s general, 500/10s on `/events`, 300/10s on `/markets`, 350/10s on search.
- **Gotchas that silently invert your numbers:** prices are returned as JSON *strings*; orderbook array ordering is not what you'd guess; the `side` parameter's meaning differs by endpoint. Assert on parsed types everywhere.
- Positions are ERC-1155 tokens under the Gnosis Conditional Token Framework. Matching is off-chain, settlement on-chain.
- **Two separate platforms**: Polymarket US (CFTC-regulated) and Polymarket International. Different books, different fee schedules, different market sets. April 2026 volume: ~$1.3B US vs ~$9B International. Never join their books.

**Kalshi**

- Base `https://trading-api.kalshi.com/trade-api/v2`. Full demo environment at `demo-api.kalshi.co` — use it for all of Phases 0–5.
- Auth is RSA-PSS request signing, no bearer token. Message = `timestamp_ms + METHOD + path` where path **includes** the `/trade-api/v2` prefix and **excludes** the query string. Sign with SHA-256 for both hash and MGF1, salt length = digest length (32 bytes), base64-encode. Headers: `KALSHI-ACCESS-KEY`, `KALSHI-ACCESS-TIMESTAMP`, `KALSHI-ACCESS-SIGNATURE`. Official SDK is `kalshi-python`, but write the ~15-line signer yourself with `cryptography` — you want control over the timestamp and clock skew handling.
- The private key is **not recoverable**. Lose it and you regenerate the pair and re-upload. Secrets manager, not `.env`, in anything resembling production.
- **The orderbook returns bids only.** Response is `orderbook_fp` containing `yes_dollars` and `no_dollars`, each an array of `[price_string, size_string]` sorted *ascending* — so the best bid is the **last** element. Derive asks: `yes_ask = 1.00 − best_no_bid`, `no_ask = 1.00 − best_yes_bid`. Reading the two sides as independent books is the single most common integration bug. Prefer `orderbook_fp` / `yes_dollars` / `no_dollars` over older integer-cent fields.
- Rate limits ~10 rps authenticated, ~30 rps public, tiered; back off on 429.
- Idle cash earns ~4% APY on Kalshi; Polymarket pays 0% on stablecoin balances. This is not a footnote — see §2.3.5.
- FIX 4.4 exists for institutional low-latency. Out of scope, but know it's what you're racing.

**Market structure both venues share**

- Binary contracts settle at $1 or $0. Price ∈ [0.01, 0.99].
- Multi-outcome events decompose into sets of binaries. Polymarket links mutually-exclusive sets with **negative risk** (`negRisk: true` on Gamma events, `neg_risk: true` on CLOB). The `NegRiskAdapter` contract lets you **convert** 1 NO share in any member into 1 YES share in *every* other member plus collateral. Capital efficiency vs plain multi-outcome is reported up to ~9.5×.
- **Augmented negative risk**: some sets deploy named outcomes + placeholder unnamed outcomes + an "other." Unnamed slots can be *redefined later* via a bulletin board, which means the "other" contract's meaning can change under you mid-position. Flag these; do not treat them as a static partition.
- Resolution: Polymarket uses UMA's optimistic oracle with a challenge window and dispute path; Kalshi uses an internal Outcome Review Committee. Both are uninsurable tail risk on any hold-to-resolution trade.

### 2.2 The fee curve, and why it dictates every strategy

Both venues moved to symmetric, probability-weighted fees. Polymarket switched on taker fees 2026-03-30 and revised exchange-wide effective **2026-07-01**. The "Polymarket is free" era is over — anything you remember otherwise is wrong.

```
Polymarket (US), per contract:
    taker_fee      = 0.06   × C × p × (1 − p)      # you pay
    maker_rebate   = 0.0125 × C × p × (1 − p)      # you RECEIVE, credited at fill
    banker's rounding (half-to-even) to the cent

    Taker volume rebate, by prior calendar month notional taker volume:
        $250k–$1M   → 10%      $1M–$10M → 25%      $10M+ → 50%
    (Verifiable trailing-30d volume on another venue can accelerate tier placement.)

Kalshi, per contract:
    taker_fee = ceil(0.07 × C × p × (1 − p))
    maker     = 0 on most markets; a 0.0175 coefficient applies in some;
                a flat 0.25%/contract applies during designated special events
                (NFL championships, NBA Finals, presidential elections)
```

Three consequences you must internalise before writing any strategy:

1. **Fees peak at p = 0.50 and vanish at the extremes.** A Polymarket taker pays 1.50¢/contract at 50¢ but 0.06¢ at 99¢ — a **25× difference**. A taker/taker cross-venue arb at the money costs roughly 1.5¢ + 1.75¢ ≈ **3.25¢ of gross spread just to break even**. The same arb at p = 0.05 costs about 0.6¢. *Arbitrage in this asset class lives in the tails, not the middle.* Design your scanner's priority queue around that fact.
2. **Makers get paid.** +0.31¢/contract at p=0.50 on Polymarket US. That is a real, exchange-funded edge — the only one in this document that doesn't depend on you being right about anything. It is also the one most easily destroyed by adverse selection (§5.4).
3. **The Kalshi special-event maker fee is a trap at low prices.** 0.25% of $1 notional on a 3¢ contract is ~8% of the contract's price. Longshot market-making on Kalshi during major events is structurally unprofitable.

### 2.3 Where the edge actually is — ranked by durability × capacity

**2.3.1 — Maker rebate + spread capture.** Structural, paid by the venue, no forecasting required. Capacity is large. The entire difficulty is not getting picked off; the rebate is ~0.31¢ and a stale quote into a news event costs multiples of that. Requires a markout discipline (§5.4) and a fast cancel path off the WebSocket.

**2.3.2 — Cross-venue equivalence & subset divergence.** The best-documented persistent inefficiency in the asset class. A 2026 study aligning >100,000 events across ten venues found roughly **6% of events are concurrently cross-listed** (but ~10% of event-*days*, since cross-listed events are long-lived), and semantically equivalent markets show **execution-adjusted deviations of 2–4% that persist even in highly liquid, information-rich settings.** The cause is not disagreement — it is *enforceability*: you cannot net positions across venues, so the arb requires capital committed until resolution, which caps how much capital is willing to do it. The deviation is bimodal in persistence: either it's there nearly the whole time, or fees absorb it entirely.

The moat here is the semantic layer (§4.3). Anyone can diff two prices; almost nobody correctly determines whether the two contracts are actually the same contract.

**2.3.3 — Negative-risk basket dutch books.** Within a complete mutually-exclusive set, `Σ p_YES` should equal 1 and `Σ p_NO` should equal n−1. Violations are mechanically risk-free. They are also rare, fast, and usually smaller than the combined spread + taker fees — retail traders see `Σ = 1.04` and assume free money, then lose it to execution. Worth building because it's cheap to add once you have the book normaliser, and because NegRisk `convert` gives you a capital-efficiency lever others don't use. Do not expect it to carry the P&L.

**2.3.4 — Domain-conditional recalibration.** This is the "predictor," and it has the largest per-trade edge. The 2026 study of 292M trades across 327k contracts fits, per (domain × horizon) cell, `logit P(y=1) = a + b·logit(p)`. A slope `b > 1` means prices are compressed toward 50% — favourites underpriced, longshots overpriced.

Published slopes (Kalshi, cutoff 2025-12-31), by time-to-resolution:

| Domain | 0–1h | 1–3h | 3–6h | 6–12h | 12–24h | 24–48h | 2d–1w | 1w–1mo | 1mo+ |
|---|---|---|---|---|---|---|---|---|---|
| Politics | 1.34 | 0.93 | 1.32 | 1.55 | 1.48 | 1.52 | **1.83** | **1.83** | 1.73 |
| Sports | 1.10 | 0.96 | 0.90 | 1.01 | 1.05 | 1.08 | 1.04 | 1.24 | **1.74** |
| Crypto | 0.99 | 1.01 | 1.07 | 1.01 | 1.01 | 1.21 | 1.12 | 1.09 | 1.36 |
| Finance | 0.96 | 1.07 | 1.03 | 0.97 | 0.98 | 0.82 | 1.07 | 1.42 | 1.20 |
| Weather | **0.69** | 0.84 | **0.74** | 0.87 | 0.91 | 0.97 | 1.20 | 1.20 | 1.37 |
| Entertainment | 0.81 | 1.02 | 1.00 | 0.92 | 0.89 | 0.84 | 1.07 | 1.11 | 0.96 |

Domain intercepts: Politics **+0.151**, Weather −0.086, Entertainment −0.085; Sports/Crypto/Finance indistinguishable from zero. Universal horizon effect: mean slope rises 0.99 (<1h) → 1.32 (>1mo) — *everything* is underconfident about the distant future.

Two readable signals: **political markets at multi-day horizons are badly compressed** (a 70¢ political contract one week out implies a true probability near 83%), and **short-horizon weather markets are over-extended** (fade the extremes). A Kalshi-specific effect — large political trades are *more* compressed than small ones (Δ = +0.53) — does not replicate on Polymarket.

Treat this table as a **prior and a hypothesis generator, not as parameters**. Re-estimate on your own data (§6, Phase 3). See §2.4 for why.

**2.3.5 — Carry, and the near-certain contract as a bond.** A contract at 97¢ resolving in 30 days is a fixed-income instrument yielding ~37% annualized *if* you're right that it's certain. The whole strategy is a bet on resolution mechanics, not on the world. Two things make this tractable and mostly overlooked:
- Kalshi pays ~4% APY on idle cash; Polymarket pays 0%. For any hold-to-resolution structure, put the cash-like leg where cash earns.
- The correct objective for every locked-up trade is `APY`, not spread. Research on cross-venue arbs found that high headline APYs are driven mainly by *short time to resolution*, not by large mispricings. Your scanner should rank on annualized return computed against the **pessimistic** settlement date — including UMA dispute windows and committee lag — not the nominal close time.

**2.3.6 — Stale-book sweeps around scheduled information.** Economic releases, court calendars, scheduled announcements. Cheap to add on top of the ingestion layer. Highly competitive; assume you are last in the queue and size accordingly.

### 2.4 Where the edge is NOT — read this before you get excited

- **The published calibration slopes are measured on the trade tape, not on quotes you could have hit.** You cannot execute at a historical trade price. When the market is "wrong," the liquidity offered to you is offered by someone who may know more than you do. Every backtest that fills at the last trade price is fiction.
- **The measured longshot bias is smaller than the fee.** Realized return by price decile on Polymarket runs about −0.2 to −0.3 pp for p ≤ 0.30 and +0.5 to +0.8 pp for p ≥ 0.60. A Polymarket taker round trip at mid prices costs ~1.5¢/contract on $50 of notional. **The naive "buy underpriced favourites" trade is fee-negative.** It survives only (a) deep in the tails where `p(1−p)` collapses, (b) as a maker earning the rebate rather than paying the taker fee, or (c) where the mispricing is genuinely large — which per §2.3.4 means politics at multi-day horizons.
- **The political sample is dominated by one election cycle.** The 2024 US cycle drives much of the political underconfidence result, and the paper's author explicitly flags temporal stability as untested. Assume regime dependence until your own walk-forward says otherwise.
- **Sports is ~80% of Kalshi volume and ~39% of Polymarket's, is well calibrated at short horizons (b ≈ 0.90–1.10), and is priced by people with real models.** It is the deepest, most competitive, least mispriced corner. Do not start there.
- **Insider flow is real and you are not on the right side of it.** A January 2026 Polymarket position on Venezuela's leadership returned >$400k to a trader later identified as a participant in the operation's planning. Both venues announced integrity controls in March 2026. Price your unexplained-adverse-fill rate accordingly.
- **Do not build a browser extension.** Google's Chrome Web Store policy prohibiting extensions that facilitate real-money prediction-market trading takes effect **2026-08-01**, putting 50+ existing tools at risk.

### 2.5 Access preconditions (encode as config, not assumptions)

Availability is a moving state-by-state patchwork: 11+ US states have issued formal warnings, Minnesota's ban takes effect 2026-08-01, and litigation is active in a dozen more, with the CFTC suing states over preemption. Polymarket International and Polymarket US are legally distinct venues with distinct eligibility. Build a `VenueEligibility` config object resolved at startup, log which venues are enabled and why, and refuse to route to a disabled venue. Do not let jurisdiction be an implicit property of whichever API key happens to be loaded.

---

## 3. Non-negotiables

1. **Demo/testnet until Phase 6.** Kalshi demo (`demo-api.kalshi.co`) and Polygon Amoy (chain 80002). No production keys in the repo, ever.
2. **Decimal, never float, for money.** Use `decimal.Decimal` end to end. A float rounding error in a fee calculation will silently manufacture arbitrage that doesn't exist.
3. **Every price carries its venue, its timestamp, and its staleness.** A canonical `Quote` object with no source venue is a bug.
4. **Nothing is tradeable until the cost model is unit-tested to the cent.** Polymarket publishes exact taker-pays and maker-receives figures for a 100-lot at all 99 price points. Your fee function must reproduce every row, including banker's rounding.
5. **Kill switch, position cap, and daily loss limit exist from Phase 0.** Not added later.
6. **The paper-trading log is append-only and includes the full book snapshot at decision time.** You cannot audit fills you didn't record the context for.

---

## 4. Architecture

```
sibyl/
  venues/
    base.py            # Venue protocol: markets(), book(), quote(), place(), cancel(), ws()
    polymarket.py      # gamma + clob + data + ws; pUSD-aware; negRisk metadata
    kalshi.py          # RSA-PSS signer, bids-only book normaliser, demo/prod switch
    normalise.py       # → canonical Market / Book / Quote / Fill
  costs/
    fees.py            # per-venue fee curves; taker/maker; volume tiers; banker's rounding
    execution.py       # effective price incl. half-spread, depth-walked slippage, tick
    capital.py         # lockup, idle yield by venue, APY given pessimistic settlement
  store/
    ledger.py          # DuckDB + Parquet: markets, books, trades, resolutions, our fills
    resolutions.py     # settled-outcome ledger — the training label source
  calibration/
    fit.py             # per-(domain × horizon) logistic recalibration slope
    hierarchical.py    # NumPyro partial-pooling model for sparse cells
    apply.py           # p* = p^b / (p^b + (1-p)^b), with posterior interval
  semantic/
    classify.py        # domain taxonomy assignment
    embed.py           # embeddings + ANN candidate retrieval
    verify.py          # LLM logical verification → EQUIVALENT | SUBSET | INDEPENDENT
    graph.py           # identity graph; atomic outcome space Ω; partition validation
  alpha/
    parity.py          # within-market, cross-venue, negRisk, subset arb conditions
    recal.py           # recalibration-driven directional signals
    rank.py            # signals → APY-ranked, capacity-aware opportunity queue
  risk/
    sizing.py          # fractional Kelly on the posterior lower bound
    exposure.py        # aggregate at EVENT and ATOMIC-OUTCOME level, not market level
    limits.py          # caps, kill switch, drawdown governor
  backtest/
    replay.py          # event-driven; settles on resolution, not on price
    markout.py         # t+1m / t+5m / t+1h post-fill markout attribution
  runtime/
    paper.py
    live.py            # gated behind an explicit, loud config flag
```

**§4.3 The semantic layer, in detail** — this is the differentiator, so build it properly. Follow the three-stage pipeline validated at ecosystem scale:

1. **Structural filter** — cross-venue only, same domain category (LLM classifier into a fixed ~20-category taxonomy), non-empty overlap in validity windows. Conservative; preserves ~100% of true matches.
2. **Embedding retrieval** — embed title + description + outcome labels + resolution metadata. Take top *k* = 20 nearest neighbours by cosine. Published recall at k=20 is 99.9% of verified equivalent and subset relations, so 20 is a defensible cutoff.
3. **LLM logical verification, two passes** — a high-recall plausibility filter, then a structured comparison of implied YES-regions. Achieves <2% false positive rate. Output a typed relation, not a similarity score.

Model each market as a resolution function over an atomic outcome space Ω, with a YES-region `f(m) ⊆ Ω`:
- `f(m₁) = f(m₂)` → **equivalent**
- `f(m₁) ⊂ f(m₂)` → **subset** (enables `YES(superset) + NO(subset)` → guaranteed unit payoff)
- otherwise → **independent**

Cross-venue partitions are valid only under completeness (`⋃ f(m) = Ω`), mutual exclusivity, and unique resolution. Verify all three programmatically before pricing a basket — one venue listing an "other" bucket while another lists every candidate individually is exactly how a "risk-free" basket becomes an unhedged position.

---

## 5. The math, explicitly

**5.1 Effective per-contract cost** (Polymarket US taker buying YES at `p`):
```
c_eff = p + 0.06·p·(1−p)          # taker
c_eff = p − 0.0125·p·(1−p)        # maker fill (rebate reduces cost)
```
For depth-walked orders, integrate across levels; never price size at the top-of-book.

**5.2 No-arbitrage conditions.** Let `δ(m)` be total per-contract friction (fee + half-spread + slippage + gas).
```
Within-market:   p_Y + δ_Y + p_N + δ_N  <  1                → buy both, profit = 1 − cost
Cross-venue:     min[ p_Y(A)+δ + p_N(B)+δ , p_Y(B)+δ + p_N(A)+δ ]  <  1
NegRisk set:     Σ_m (p_Y(m) + δ(m))  <  1        → buy all YES,  payoff 1
                 Σ_m (p_N(m) + δ(m))  <  n − 1    → buy all NO,   payoff n−1
Subset f(s)⊂f(S): p_Y(S) + δ + p_N(s) + δ  <  1
```

**5.3 The objective is annualized, not absolute:**
```
APY = (payoff / total_cost) ^ (365 / days_to_pessimistic_settlement) − 1
```
Rank the opportunity queue on APY with a hard floor on absolute profit (so you don't chase 400% APY on $3 of edge, net of gas).

**5.4 Recalibration and edge:**
```
p* = p^b / ( p^b + (1−p)^b )          equivalently   logit(p*) = b · logit(p)
edge = p*_lower_bound − c_eff          # use the posterior lower bound, not the mean
```

**5.5 Sizing.** For a binary bought at effective cost `c` paying 1:
```
f* = (p* − c) / (1 − c)               # full Kelly, fraction of bankroll
size = 0.25 · f* · bankroll           # quarter-Kelly
```
Substitute the ~25th percentile of the posterior for `p*`. Cap per market, per event, and per correlated cluster. Positions across a NegRisk set are one position.

**5.6 Adverse-selection test for any passive strategy.** The maker rebate at p=0.50 is 0.31¢/contract. After every passive fill, record markout at t+1m, t+5m, t+1h. If mean adverse markout exceeds the rebate, the strategy is negative — regardless of how good the fill rate or the Sharpe on paper looks. This single metric decides whether §2.3.1 is real for you.

---

## 6. Build phases & acceptance criteria

**Phase 0 — Skeleton.** Config, secrets, both venue clients authenticating against demo/testnet, async rate limiters, structured logging, kill switch.
> ✅ Authenticated read from Kalshi demo and Polygon Amoy. Zero secrets in git. `mypy --strict` clean.

**Phase 1 — Ingestion & cost model.** Canonical schema, both venues normalised, fee/execution model.
> ✅ Property test: for every Kalshi book, `yes_ask == 1 − best_no_bid` and best bid is the *last* array element.
> ✅ Fee function reproduces all 99 rows of the published Polymarket taker/maker table exactly, including banker's rounding.
> ✅ Hypothesis-based property tests for the parity invariants in §5.2.

**Phase 2 — Historical store & resolution ledger.** Backfill markets, price history, and settled outcomes into DuckDB/Parquet.
> ✅ ≥6 months of resolved markets with price path, close time, resolution, domain label. Fully reproducible from a single command.

**Phase 3 — Calibration engine.** Fit per-cell slopes on *your* data; hierarchical partial pooling for thin cells.
> ✅ Walk-forward, out-of-sample, with **market-clustered** bootstrap CIs (trade-level bootstrap will overstate significance — trades within a market are serially correlated).
> ✅ Beats the raw-price baseline on Brier score out-of-sample, with a reliability diagram.
> ✅ Honest report on which cells have real signal and which are noise. Cells whose CI straddles b = 1 are not tradeable. Say so.

**Phase 4 — Semantic identity graph.** The three-stage pipeline from §4.3.
> ✅ ≥95% precision on a hand-labelled 200-pair validation set.
> ✅ **Must correctly classify the two canonical traps:** Kalshi's NYC high-temperature market (resolves on NOAA Central Park) vs Polymarket's (resolves on LaGuardia) → **not equivalent**. Polymarket's 2024 presidential market (resolved on network calls) vs Kalshi's (resolved on inauguration) → **subset, not equivalent**. If your pipeline calls either pair equivalent, it will lose money and you have not passed this phase.

**Phase 5 — Alpha, sizing, paper trading.** Opportunity queue, Kelly sizing, full paper harness.
> ✅ 30 days of paper trading with per-trade attribution: predicted edge, realized edge, decomposed into fee / slippage / adverse selection / model error.
> ✅ Markout report per §5.6.
> ✅ At least one hypothesis explicitly killed by the data, with the write-up.

**Phase 6 — Live (gated).** Only if Phase 5's realized edge is positive with a CI excluding zero. Start at 1% of intended size. Loud config flag, separate credentials, hard daily loss limit.

---

## 7. Library manifest

```toml
# Tooling
uv, ruff, mypy, pytest, hypothesis, pytest-asyncio, freezegun

# Transport & schema
httpx[http2]        # async REST, HTTP/2
websockets          # venue WS feeds
tenacity            # retry/backoff (429s are routine)
aiolimiter          # token-bucket rate limiting per host
pydantic>=2         # canonical Market/Book/Quote/Fill; validate at every boundary
pydantic-settings   # typed config
msgspec             # fast decode on the WS hot path
structlog

# Data
polars              # primary dataframe — lazy, fast, sane nulls
duckdb              # analytical store over Parquet; the whole warehouse for this scale
pyarrow
pandas              # interop only

# Venue SDKs
py-clob-client-v2   # Polymarket CLOB V2 (production path)
polymarket-client   # unified SDK, --pre, read paths only for now
kalshi-python       # official; but hand-roll the RSA-PSS signer
cryptography        # RSA-PSS signing
eth-account, web3   # EIP-712, NegRiskAdapter / CTF on-chain reads

# Modelling
numpy, scipy, statsmodels
scikit-learn        # LogisticRegression for slopes, isotonic, brier_score_loss
numpyro + jax       # hierarchical calibration model with partial pooling
arviz               # R-hat, ESS, posterior predictive checks
lightgbm            # residual fair-value models on engineered features
mapie               # conformal intervals on fair-value estimates
cvxpy               # multi-leg basket sizing under capital + correlation constraints

# Semantic layer
sentence-transformers   # local embeddings (BGE / E5 class)
usearch OR faiss-cpu    # ANN retrieval
rapidfuzz               # cheap lexical prefilter before embedding
anthropic               # LLM logical verification, structured output

# Ops
apscheduler, prometheus-client, rich
```

**Deliberately excluded: off-the-shelf backtesters** (vectorbt, backtrader, zipline). Write `backtest/replay.py` yourself. None of them model a per-contract quadratic fee curve, settlement on event resolution rather than price, capital locked until an uncertain settlement date, no-shorting, or NegRisk conversion. Bending an equity backtester to this shape costs more than writing 400 honest lines.

---

## 8. Known traps — a checklist to grep for before you ship

- [ ] Treating Kalshi's YES and NO books as independent markets
- [ ] Floats anywhere in the money path
- [ ] Assuming Polymarket is fee-free (changed 2026-03-30, revised 2026-07-01)
- [ ] Joining Polymarket US and Polymarket International books
- [ ] Parsing prices as numbers when the API returns strings
- [ ] Ranking opportunities on spread instead of APY
- [ ] Using nominal close time instead of pessimistic settlement (dispute windows)
- [ ] Filling backtests at the last trade price
- [ ] Matching markets on title similarity
- [ ] Treating an augmented-NegRisk "other" as a fixed-definition outcome
- [ ] Netting exposure at market level when positions share an atomic outcome
- [ ] Trade-level bootstrap CIs on serially correlated within-market trades
- [ ] Any code path that reaches production without the eligibility check from §2.5

---

## 9. Definition of done

The system can, on demand:

1. Ingest both venues into one canonical schema with correct, tested cost models.
2. Emit a recalibrated probability with an uncertainty interval for any live market, and honestly state when it has no signal for that cell.
3. Maintain a cross-venue identity graph that passes the NYC-temperature and 2024-election-call tests.
4. Produce an APY-ranked opportunity queue, net of all frictions, with capacity estimates.
5. Size positions on the posterior lower bound at fractional Kelly with cluster-level caps.
6. Show 30 days of paper trades decomposed into fee, slippage, adverse selection, and model error — and tell you, in writing, which of its own hypotheses are dead.

**Point 6 is the real deliverable.** A system that reliably tells you it has no edge is worth more than one that tells you it has one.

---

## Appendix — first commands

```bash
uv init sibyl && cd sibyl
uv add httpx[http2] websockets tenacity aiolimiter pydantic pydantic-settings \
       msgspec structlog polars duckdb pyarrow numpy scipy scikit-learn \
       statsmodels numpyro jax arviz lightgbm mapie cvxpy \
       sentence-transformers usearch rapidfuzz anthropic \
       cryptography eth-account web3
uv add py-clob-client-v2 kalshi-python
uv add --dev ruff mypy pytest pytest-asyncio hypothesis freezegun
```

Start with `costs/fees.py` and its test file. If the fee model is wrong, everything downstream is confidently wrong.
