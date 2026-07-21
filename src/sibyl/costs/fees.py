"""Per-venue fee curves (verified July 2026 — see build spec §2.2).

Both venues charge probability-weighted fees proportional to p·(1-p): fees
peak at p=0.50 and vanish at the extremes. All arithmetic is Decimal; floats
are rejected at the boundary.

Rounding conventions differ and matter:
  - Polymarket US: banker's rounding (half-to-even) to the cent, on the total.
  - Kalshi: rounds UP to the next cent, on the total.

The `polymarket_*` functions implement Polymarket US ONLY (verified against
https://docs.polymarket.us/fees, schedule effective 2026-07-01). Polymarket
International runs a different schedule entirely — per-category taker
coefficients, no maker rebate, 5-decimal rounding — and is deliberately
unimplemented; see `polymarket_intl_taker_fee`.
"""

from __future__ import annotations

from decimal import ROUND_CEILING, ROUND_HALF_EVEN, Decimal

from sibyl.venues.normalise import parse_price

CENT = Decimal("0.01")
ONE = Decimal("1")

POLYMARKET_TAKER_COEF = Decimal("0.06")
POLYMARKET_MAKER_REBATE_COEF = Decimal("0.0125")
KALSHI_TAKER_COEF = Decimal("0.07")
KALSHI_MAKER_COEF_STANDARD = Decimal("0")
KALSHI_MAKER_COEF_DESIGNATED = Decimal("0.0175")
# Flat maker fee per contract during designated special events (0.25% of $1).
KALSHI_SPECIAL_EVENT_MAKER_FLAT = Decimal("0.0025")

# Polymarket taker volume rebate by prior-calendar-month notional taker volume.
# Applied as a discount on the taker fee. (thresholds inclusive at the bottom)
_POLYMARKET_VOLUME_TIERS: tuple[tuple[Decimal, Decimal], ...] = (
    (Decimal("10000000"), Decimal("0.50")),
    (Decimal("1000000"), Decimal("0.25")),
    (Decimal("250000"), Decimal("0.10")),
)


def _validate(price: object, contracts: object) -> tuple[Decimal, Decimal]:
    p = parse_price(price)
    c = parse_price(contracts)
    if not (Decimal("0") < p < ONE):
        raise ValueError(f"price must be in (0, 1), got {p}")
    if c < 0:
        raise ValueError(f"contracts must be >= 0, got {c}")
    return p, c


def _quadratic(coef: Decimal, price: Decimal, contracts: Decimal) -> Decimal:
    return coef * contracts * price * (ONE - price)


def polymarket_volume_discount(prior_month_taker_notional: object) -> Decimal:
    """Taker-fee discount fraction for a prior-calendar-month taker notional."""
    notional = parse_price(prior_month_taker_notional)
    for threshold, discount in _POLYMARKET_VOLUME_TIERS:
        if notional >= threshold:
            return discount
    return Decimal("0")


def polymarket_taker_fee(
    price: object,
    contracts: object,
    *,
    volume_discount: Decimal = Decimal("0"),
) -> Decimal:
    """Total taker fee in dollars, banker's-rounded to the cent.

    fee = 0.06 · C · p · (1-p), reduced by any volume-tier discount before
    rounding. Positive = you pay.
    """
    p, c = _validate(price, contracts)
    if not (Decimal("0") <= volume_discount < ONE):
        raise ValueError(f"volume_discount must be in [0, 1), got {volume_discount}")
    raw = _quadratic(POLYMARKET_TAKER_COEF, p, c) * (ONE - volume_discount)
    return raw.quantize(CENT, rounding=ROUND_HALF_EVEN)


def polymarket_maker_rebate(price: object, contracts: object) -> Decimal:
    """Total maker rebate in dollars, banker's-rounded to the cent.

    rebate = 0.0125 · C · p · (1-p), credited at fill. Positive = you receive.
    """
    p, c = _validate(price, contracts)
    raw = _quadratic(POLYMARKET_MAKER_REBATE_COEF, p, c)
    return raw.quantize(CENT, rounding=ROUND_HALF_EVEN)


def _ceil_to_cent(raw: Decimal) -> Decimal:
    return raw.quantize(CENT, rounding=ROUND_CEILING)


def kalshi_taker_fee(price: object, contracts: object) -> Decimal:
    """Total taker fee in dollars: ceil(0.07 · C · p · (1-p)) to the cent."""
    p, c = _validate(price, contracts)
    return _ceil_to_cent(_quadratic(KALSHI_TAKER_COEF, p, c))


def kalshi_maker_fee(
    price: object,
    contracts: object,
    *,
    designated_market: bool = False,
    special_event: bool = False,
) -> Decimal:
    """Total maker fee in dollars.

    Zero on most markets; 0.0175 coefficient on designated markets; a flat
    0.25%/contract during designated special events (NFL championships, NBA
    Finals, presidential elections). The flat fee dominates at low prices —
    ~8% of a 3¢ contract — which is why longshot market-making on Kalshi
    during special events is structurally unprofitable.
    """
    p, c = _validate(price, contracts)
    if special_event:
        return _ceil_to_cent(KALSHI_SPECIAL_EVENT_MAKER_FLAT * c)
    if designated_market:
        return _ceil_to_cent(_quadratic(KALSHI_MAKER_COEF_DESIGNATED, p, c))
    return Decimal("0.00")


def polymarket_intl_taker_fee(price: object, contracts: object) -> Decimal:
    """Polymarket International fees are a different schedule (per-category
    coefficients, zero maker rebate). Unimplemented on purpose: silently
    applying US fees to INTL books would manufacture phantom edge."""
    raise NotImplementedError(
        "Polymarket International fee schedule not implemented; "
        "do not price INTL markets with the US fee curve"
    )


def polymarket_effective_taker_cost(price: object) -> Decimal:
    """Per-contract effective cost of taking YES at `price` (§5.1), unrounded."""
    p, _ = _validate(price, 1)
    return p + POLYMARKET_TAKER_COEF * p * (ONE - p)


def polymarket_effective_maker_cost(price: object) -> Decimal:
    """Per-contract effective cost of a maker fill at `price` (§5.1), unrounded.

    The rebate reduces cost, so this is below `price`.
    """
    p, _ = _validate(price, 1)
    return p - POLYMARKET_MAKER_REBATE_COEF * p * (ONE - p)


def kalshi_effective_taker_cost(price: object) -> Decimal:
    """Per-contract effective cost of taking YES at `price`, unrounded.

    Note: Kalshi's per-order ceil-to-cent makes the true marginal cost
    size-dependent; this is the smooth approximation for ranking. Exact order
    costs must use `kalshi_taker_fee` on the full size.
    """
    p, _ = _validate(price, 1)
    return p + KALSHI_TAKER_COEF * p * (ONE - p)
