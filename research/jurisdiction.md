# Venue eligibility — researched 2026-10-11

Spec §2.5 says eligibility must be explicit config, never an implicit property
of whichever API key is loaded. This file records what the config encodes and
why. **It is a snapshot, not a legal opinion, and the situation is actively
litigated — re-check before enabling anything live.**

## Operator: Canada

### Kalshi — unavailable

Kalshi does not serve Canadian residents. Consequences for this build:

- No prod API key is obtainable, so **no Kalshi historical data** for
  calibration. (The demo environment is open and the client stays fully
  tested, but demo prices are paper-market noise — fitting on them would
  produce a meaningless model.)
- **Cross-venue arbitrage (§2.3.2) is unexecutable.** The parity code and the
  semantic layer remain valid and tested as research instruments, but the
  trade itself requires funded accounts on both venues.
- `kalshi.enabled` defaults to `False` with a recorded reason.

### Polymarket US — not applicable

A CFTC-regulated US exchange; the operator is not US-resident. Config default
moved from `us` to `intl` accordingly. The US fee curve stays implemented and
tested (it is the better-documented of the two and the spec's fee acceptance
test targets it), but it must never price an INTL book.

### Polymarket International — restricted, province-dependent

As of 2026-10-11, per public reporting:

| Province | Status |
|---|---|
| Ontario | **Permanently banned.** April 2025 OSC enforcement against Polymarket's parent entities: CAD $200,000 penalty plus $25,000 costs, and a permanent undertaking not to serve Ontario residents — reported as binding until at least 2027. The admitted violation was offering/advertising short-term binary event contracts. |
| Alberta, British Columbia, Quebec | **Geo-blocked** as of a 2026-07-06 terms-of-service update. Alberta's Service minister indicated Polymarket made this decision itself rather than under a specific order, so commentators read it as caution, not enforcement. |
| Other provinces | Reported available. |

Enforcement is by IP geolocation applied at the order level; the wallet-based
global site performs no identity check.

Sources: [Canadian Gaming Business](https://www.canadiangamingbusiness.com/2026/07/13/polymarket-blocks-provinces/),
[NextPredict legal summary](https://nextpredict.io/legal/canada/),
[Polymarket101](https://www.polymarket101.com/en/docs/countries/is-polymarket-legal-in-canada/).

## What this means for the build

**Phase 6 (live trading) is presumed out of scope.** In four provinces —
including Ontario, where it is a standing regulatory ban rather than a
platform choice — there is no lawful path to it, and the remaining provinces
need confirmation the operator can and should verify personally.

This is not a loss for the project. Spec §9 is explicit that the real
deliverable is Point 6: an honest paper-trading write-up that names its own
dead hypotheses. **Phase 5 is the finish line**, and the system's value is
that it tells you truthfully whether an edge exists.

Note also the circularity worth being honest about: evading a geo-block to
trade would not only be the wrong call, it would invalidate the research. A
system whose premise is "report the truth about your edge" cannot start by
lying about where it is.

## Config encoding

```python
KalshiSettings(enabled=False, disabled_reason="operator jurisdiction (Canada): Kalshi unavailable")
PolymarketSettings(platform="intl", chain_id=80002)   # Amoy testnet until Phase 6
Settings(live=False)                                   # refuses to be True on a testnet chain
```

`resolve_eligibility()` logs each venue's enabled/disabled state and reason at
startup, and `VenueEligibility.require()` raises `PermissionError` on any
attempt to route to a disabled venue.
