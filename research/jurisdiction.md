# Venue eligibility — researched 2026-10-11, sources re-checked 2026-10-12

Spec §2.5 requires eligibility to be explicit config, never an implicit
property of whichever API key is loaded. This file records what the config
encodes and why.

**It is not a legal opinion.** Every claim below carries its verification
status, because an earlier version of this file stated two things as fact
that the sources do not support. Those corrections are marked ⚠️.

| status | meaning |
|---|---|
| ✅ | consistent across multiple independent secondary reports |
| ⚠️ | **corrected** — an earlier version of this file was wrong |
| ❓ | could not reach the primary source; treat as provisional |

## Operator: Ontario, Canada — settled as config

`Settings.operator_province` defaults to `"ON"`, `BLOCKED_PROVINCES` covers
ON/AB/BC/QC, and `live=True` raises for any of them. Phase 6 is closed by
construction. That config decision does not depend on the fine detail below;
the operative fact is simply that Polymarket does not accept new positions
from Ontario.

### The OSC settlement

✅ Respondents were **Blockratize Inc.** and **Adventure One QSS Inc.**
(operators and former operators of Polymarket).

✅ Approved by the **Capital Markets Tribunal on 2025-04-17**. (An earlier
version of this file said only "April 2025".)

✅ Financial terms: **CAD $200,000** administrative penalty, **CAD $25,000**
costs, and disgorgement of approximately **USD $22,966.75** of
Ontario-derived revenue — all reported as already paid. (The disgorgement was
missing from the earlier version.)

✅ Offending conduct ran **June 2020 to May 2023**: offering short-term binary
options to Ontario individual investors, and soliciting that trading via
social media, without an exemption.

⚠️ **The earlier claim of a "permanent" ban binding "to at least 2027" is not
supported.** OSC-derived reporting describes a **two-year trading and
registration ban** on the respondents, plus enforceable undertakings.
"Permanent" traces to a news headline (The Logic), and "until at least 2027"
to a secondary guide — neither is a settlement term. The repo should not have
asserted it.

❓ The OSC's own settlement document and news release were **not reachable**
(HTTP 403 from both `osc.ca` and the trade press mirror). Everything above is
from secondary reporting that agrees with itself; the primary document has not
been read.

**Why Ontario is still closed despite a two-year term.** The durable
restriction is not the settlement clock but the **standing Ontario prohibition
on offering short-term binary options to retail investors**, which is what the
settlement enforced rather than created. A two-year ban expiring in April 2027
would not reopen the underlying conduct. This is the structural reason, and it
is also why re-checking before any live work is still required.

### Kalshi — unavailable

✅ Kalshi does not serve Canadian residents. Consequences for this build:

- No prod API key, so **no Kalshi historical data** for calibration. The demo
  environment is open and the client stays fully tested, but demo prices are
  paper-market noise and fitting on them would produce a meaningless model.
- **Cross-venue arbitrage (§2.3.2) is unexecutable.** The parity conditions
  and semantic layer remain valid, tested research instruments.
- `kalshi.enabled` defaults to `False` with a recorded reason.

### Polymarket US — not applicable

A CFTC-regulated US exchange; the operator is not US-resident. The platform
default is `intl`. The US fee curve stays implemented and tested (it is the
better-documented of the two and the spec's fee acceptance test targets it),
but it must never price an INTL book.

### Polymarket International — restricted, province-dependent

| Province | Status |
|---|---|
| Ontario | Closed to new positions (see above). |
| Alberta, British Columbia, Quebec | ✅ Added to the restricted list in July 2026. |
| Other provinces | ✅ Reported as available. |

⚠️ **The restriction is "close-only", not a blanket block.** Reporting is
consistent that affected users may **close existing positions but not open new
ones**, on both the front end and the API. The earlier version said
"geo-blocked", which overstates it — though for a system that has never held a
position, close-only and blocked are operationally identical.

⚠️ **The date is not settled.** One report gives a terms-of-service update on
**July 6, 2026**; another gives Alberta geoblocked on **July 2, 2026**, days
before Alberta's regulated iGaming launch. The earlier version asserted
2026-07-06 as fact for all three provinces.

✅ Alberta's Minister (Dale Nally) said Polymarket made the decision **on its
own** rather than under an order; coverage links it to Polymarket lacking an
AGLC licence ahead of Alberta's regulated launch. For BC and Quebec, whether
the provinces demanded exit or Polymarket chose it is **unclear**.

❓ All of this is secondary reporting from gambling and crypto trade press,
some internally inconsistent on dates. **Polymarket's own geographic
restrictions page is the authoritative reference and was not reached.**

Sources consulted (secondary): [The Logic](https://thelogic.co/briefing/polymarket-pays-200000-penalty-permanently-bans-ontario-residents/),
[Globe and Mail / PR Newswire carrying the OSC release](https://www.theglobeandmail.com/investing/markets/markets-news/PR%20Newswire/31937259/osc-reaches-settlement-with-current-and-former-operators-of-polymarket-on-breach-of-binary-options-ban/),
[Investment Executive](https://www.investmentexecutive.com/from-the-regulators/crypto-based-prediction-market-settles-with-osc/),
[FinanceFeeds](https://financefeeds.com/polymarket-fined-243k-over-offering-binary-options-to-ontario-residents/),
[SiGMA World](https://sigma.world/news/polymarket-faces-wider-restrictions-across-canada/),
[Canadian Gaming Business](https://www.canadiangamingbusiness.com/2026/07/13/polymarket-blocks-provinces/) (403 on re-check).

## What this means for the build

**Phase 6 (live trading) is out of scope**, and the config enforces it rather
than documenting it. Reading public market data is not trading, so the
recorder and all research are unaffected — including running the recorder on a
hosted machine.

This is not a loss for the project. Spec §9 is explicit that the real
deliverable is Point 6: an honest paper-trading write-up that names its own
dead hypotheses. **Phase 5 is the finish line.**

One circularity worth stating: evading a geo-block to trade would not only be
the wrong call, it would invalidate the research. A system whose premise is
"report the truth about your edge" cannot begin by lying about where it is.

## Config encoding

```python
Settings(operator_province="ON", live=False)   # live=True raises in ON/AB/BC/QC
KalshiSettings(enabled=False, disabled_reason="operator jurisdiction (Canada): Kalshi unavailable")
PolymarketSettings(platform="intl", chain_id=80002)   # Amoy testnet
```

`resolve_eligibility()` logs each venue's state and reason at startup, and
`VenueEligibility.require()` raises `PermissionError` on any attempt to route
to a disabled venue.

## If this ever needs to be relied on

Read the primary sources, which this file has not: the OSC settlement document
on osc.ca or the Capital Markets Tribunal record, and Polymarket's own
geographic-restrictions page. Both were unreachable from here.
