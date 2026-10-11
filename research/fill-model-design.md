# Fill model design — the thing that decides whether Phase 5 is honest

## Why this document exists before any maker code

The maker rebate (spec §2.3.1) is the strongest remaining candidate after the
Phase 3 kill, because it is the one edge that does not require being right
about anything. It is also **the hardest strategy to paper-trade honestly**,
and for a specific reason:

> The entire question is adverse selection. Do you only get filled just
> before the price moves against you?

A naive simulator fills a resting order whenever the market price touches its
level. That is the **most optimistic assumption available**: it grants a fill
in exactly the cases where the counterparty was willing to cross to you, and
it grants it instantly, ahead of every order that was already queued there.
It inverts the thing being measured — adverse selection appears only as
upside. Any maker P&L produced that way is fiction, and would be fiction with
a convincing Sharpe ratio attached.

Spec §5.6 already names the metric that decides whether §2.3.1 is real: mean
adverse markout at t+1m / t+5m / t+1h versus the ~0.31c rebate. That test is
only meaningful if the fills being marked out are plausible fills.

Note a venue-specific twist: on Polymarket **International** the maker
"rebate" is a *daily pro-rata share of the taker-fee pool*, not a per-fill
credit (see `costs/fees.py`). So the credit side is not even knowable at
decision time, which makes the markout discipline more important, not less.

## What is being recorded

`sibyl record` is running and captures, per subscribed outcome token:

| event | content | role in the model |
|---|---|---|
| `book` | full bid/ask ladder, tick size | queue depth at our level on (re)sync |
| `price_change` | per-level price/size updates with side, plus best bid/ask | queue evolution |
| `last_trade_price` | executed trades | the only evidence of *consumption* |

Both legs (YES and NO) of each market are subscribed, since they are two
views of one instrument. A manifest per session records which assets were
watched from when, so an absence of events is unambiguous between "quiet
market" and "never subscribed".

Observed rate: roughly 860k events/day across 120 assets, no reconnects.

## The core difficulty: cancels look like trades

Queue position requires knowing *why* the size at a price level fell. A
decrease of 500 contracts at 0.42 is either:

- **a trade** — the queue ahead of us was consumed, and we moved up it; or
- **a cancel** — someone withdrew, and we moved up it *without* any trade
  occurring.

Both advance queue position, but they mean opposite things for adverse
selection: consumption by an aggressive counterparty is the event that
precedes price movement. Distinguishing them is exactly what the trade stream
is for. The reconstruction rule:

```
for each size decrease at a level:
    matched = trades at that price within the event window
    traded_volume   = min(decrease, matched)
    cancelled_volume = decrease - traded_volume
```

This is imperfect — trades and book deltas arrive as separate events with
their own timestamps, so the window is a tuning parameter and simultaneous
trade+cancel at one level is ambiguous. **The ambiguity must be resolved
pessimistically**: when it cannot be determined whether we advanced by trade
or by cancel, assume the interpretation less favourable to us.

## Fill rules (the conservative defaults)

1. **Join the back of the queue.** On placing at a level, queue-ahead =
   current visible size at that level. Never assume priority over resting
   size.
2. **Advance only on consumption attributable to trades**, by the matching
   rule above. Cancels ahead of us do advance position (they genuinely do),
   but see rule 5.
3. **Fill only when cumulative traded volume at our level exceeds
   queue-ahead.** Partial fills are partial, not rounded up.
4. **A level that is swept past does not guarantee a fill.** If the book
   trades *through* our price in one event with less volume than our
   queue-ahead, we are not filled — we are left resting in a market that has
   just moved against us. This is the single most important asymmetry and the
   one a naive simulator gets backwards.
5. **Latency is non-zero.** Our own actions (place, cancel) apply at
   `event_time + latency`, not instantly. The recorder stores both venue and
   receive timestamps precisely so this can be measured rather than guessed;
   until it is measured, use the observed receive-minus-venue distribution as
   the floor.
6. **No fill at a price better than our limit**, ever, however the book moves.

## What makes a result credible

A maker simulation should be reported with, at minimum:

- **fill rate** and queue-ahead distribution at fill time;
- **markout at t+1m / t+5m / t+1h** per §5.6, as the decisive number;
- **the fraction of fills whose attribution was ambiguous** (trade vs cancel)
  — if that is large, the result rests on the tuning parameter, not the data;
- **a sensitivity run** under the optimistic rule (fill on touch), stated side
  by side. If the strategy is only profitable under the optimistic rule, that
  *is* the finding.

## Deliberate non-goals

- **Market impact of our own orders** is not modelled. A resting order does
  change others' behaviour, and this cannot be recovered from passively
  recorded data. Size the simulated strategy small enough that ignoring it is
  defensible, and say so.
- **Hidden/iceberg liquidity** is invisible by construction.
- **Order placement is not being simulated live.** Phase 6 is closed on
  jurisdiction (see `jurisdiction.md`), so this model exists to produce an
  honest research result, not to warm up a live system.

## Build order

1. **Reconstructor**: replay `book` + `price_change` into a ladder at any
   timestamp, validated by re-syncing against the next full `book` snapshot
   and asserting agreement. Without that validation the replay is unverified.
2. **Trade attribution**: the decrease-matching rule, reporting its own
   ambiguity rate.
3. **Queue simulator**: rules 1-6, with the pessimistic default and the
   optimistic rule available as a labelled comparison.
4. **Markout harness**: §5.6, reading mid-price from the reconstructed book
   at t+1m/5m/1h.
5. Only then: a maker strategy to run through it.

Step 1 is testable against recorded data today. Steps 4 and 5 need enough
recorded history to contain real adverse-selection episodes — days, not hours.
