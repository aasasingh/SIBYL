# SIBYL

A prediction-market fair-value & edge engine for **Polymarket US** and **Kalshi**.
Full design: [SIBYL-build-prompt.md](SIBYL-build-prompt.md).

> The Sibyl gave true prophecies in ambiguous form. Market prices are the same:
> informative, but not the probability they appear to be. This system's job is
> to translate.

## Status

| Phase | Scope | State |
|---|---|---|
| 0 | Skeleton: config, venue clients, kill switch, logging | ✅ done |
| 1 | Ingestion & cost model | ✅ fees, depth-walk execution, APY/capital, §5.2 parity property tests |
| 2 | Historical store & resolution ledger | ⬜ |
| 3 | Calibration engine | ⬜ |
| 4 | Semantic identity graph | ⬜ |
| 5 | Alpha, sizing, paper trading | ⬜ |
| 6 | Live (gated) | ⬜ |

Everything runs against **Kalshi demo** (`demo-api.kalshi.co`) and Polygon
**Amoy** until Phase 6. `live=True` is refused unless the config is coherently
production (mainnet chain, prod Kalshi env).

## Verified domain facts

- Polymarket **US** fees (effective 2026-07-01, verified against
  [docs.polymarket.us/fees](https://docs.polymarket.us/fees) on 2026-07-21):
  taker `0.06·C·p·(1−p)`, maker rebate `0.0125·C·p·(1−p)`, **banker's rounding**
  to the cent, volume rebate tiers 10/25/50%. The golden test pins our formula
  to the venue's published rows ([fixture](tests/fixtures/polymarket_fee_table_100lot.json)).
- Polymarket **International** uses a completely different schedule
  (per-category coefficients, no maker rebate). Deliberately unimplemented;
  pricing INTL books with US fees raises `NotImplementedError`.
- Kalshi books are **bids-only**, ascending, best bid last; asks are derived
  (`yes_ask = 1 − best_no_bid`). Property-tested in
  [tests/test_normalise.py](tests/test_normalise.py), validated against a live
  demo book.

## Commands

```bash
uv run pytest            # offline suite (78 tests)
uv run pytest -m live    # live connectivity smoke tests
uv run mypy              # strict, src + tests
uv run ruff check .
uv run sibyl status      # config + eligibility + venue connectivity
```

## Configuration

Environment variables, prefix `SIBYL_`, nested with `__`:

```
SIBYL_KALSHI__API_KEY_ID=...             # from kalshi demo account settings
SIBYL_KALSHI__PRIVATE_KEY_PATH=...       # PEM path OUTSIDE the repo
SIBYL_POLYMARKET__PLATFORM=us            # us | intl — never both
SIBYL_RISK__DAILY_LOSS_LIMIT_USD=50
```

No secrets in git, ever. The kill switch is the `KILL_SWITCH` file: `touch` it
to halt all trading paths immediately.
