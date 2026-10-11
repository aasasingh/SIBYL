"""Per-venue fee curves (verified July 2026 — see build spec §2.2).

Both venues charge probability-weighted fees proportional to p·(1-p): fees
peak at p=0.50 and vanish at the extremes. All arithmetic is Decimal; floats
are rejected at the boundary.

Rounding conventions differ and matter:
  - Polymarket US: banker's rounding (half-to-even) to the cent, on the total.
  - Kalshi: rounds UP to the next cent, on the total.

Two Polymarket venues, two incompatible schedules — never cross them:
  - `polymarket_*`      -> Polymarket US. Taker 0.06, per-fill maker rebate
    0.0125, banker's rounding to the cent. (docs.polymarket.us/fees,
    effective 2026-07-01, verified 2026-07-21.)
  - `polymarket_intl_*` -> Polymarket International. Per-category taker
    coefficients (0 to 0.07), makers never charged, 5-decimal rounding.
    (docs.polymarket.com/polymarket-learn/trading/fees, verified 2026-10-11.)
"""

from __future__ import annotations

from decimal import ROUND_CEILING, ROUND_HALF_EVEN, Decimal
from enum import StrEnum

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


class IntlFeeCategory(StrEnum):
    """Polymarket International's own fee taxonomy.

    Deliberately separate from `semantic.classify.Domain`: this one decides
    what you pay, that one decides how prices are recalibrated, and they do
    not agree. The sharpest divergence is GEOPOLITICS — fee-free on INTL, but
    a `politics` market for calibration purposes.
    """

    CRYPTO = "crypto"
    SPORTS = "sports"
    FINANCE = "finance"
    POLITICS = "politics"
    MENTIONS = "mentions"
    TECH = "tech"
    ECONOMICS = "economics"
    CULTURE = "culture"
    WEATHER = "weather"
    GEOPOLITICS = "geopolitics"
    OTHER = "other"


# Verified against https://docs.polymarket.com/polymarket-learn/trading/fees
# on 2026-10-11. Each coefficient is confirmed by the venue's own published
# 100-share peak: coef * 100 * 0.25 == peak fee at p=0.50.
INTL_TAKER_COEF: dict[IntlFeeCategory, Decimal] = {
    IntlFeeCategory.CRYPTO: Decimal("0.07"),  # peak $1.75
    IntlFeeCategory.SPORTS: Decimal("0.05"),  # peak $1.25
    IntlFeeCategory.ECONOMICS: Decimal("0.05"),
    IntlFeeCategory.CULTURE: Decimal("0.05"),
    IntlFeeCategory.WEATHER: Decimal("0.05"),
    IntlFeeCategory.OTHER: Decimal("0.05"),
    IntlFeeCategory.FINANCE: Decimal("0.04"),  # peak $1.00
    IntlFeeCategory.POLITICS: Decimal("0.04"),
    IntlFeeCategory.MENTIONS: Decimal("0.04"),
    IntlFeeCategory.TECH: Decimal("0.04"),
    IntlFeeCategory.GEOPOLITICS: Decimal("0"),  # fee-free
}

# Maker rebate pool share by category — the fraction of taker fees collected
# that is redistributed to makers, paid DAILY and pro-rata. This is NOT a
# per-fill credit (contrast Polymarket US), so it must never be subtracted
# from an effective cost: you cannot know your share at decision time.
INTL_MAKER_REBATE_SHARE: dict[IntlFeeCategory, Decimal] = {
    IntlFeeCategory.SPORTS: Decimal("0.15"),
    IntlFeeCategory.CRYPTO: Decimal("0.20"),
    IntlFeeCategory.FINANCE: Decimal("0.25"),
    IntlFeeCategory.POLITICS: Decimal("0.25"),
    IntlFeeCategory.ECONOMICS: Decimal("0.25"),
    IntlFeeCategory.CULTURE: Decimal("0.25"),
    IntlFeeCategory.WEATHER: Decimal("0.25"),
    IntlFeeCategory.MENTIONS: Decimal("0.25"),
    IntlFeeCategory.TECH: Decimal("0.25"),
    IntlFeeCategory.OTHER: Decimal("0.25"),
    IntlFeeCategory.GEOPOLITICS: Decimal("0"),  # no fees collected, no pool
}

# INTL rounds to 5 decimals; anything below the minimum charge rounds to zero.
INTL_QUANTUM = Decimal("0.00001")

# Tag keywords -> INTL fee category, highest-priority first. Geopolitics is
# checked before politics because the fee difference is total (0 vs 0.04).
_INTL_TAG_BUCKETS: tuple[tuple[IntlFeeCategory, frozenset[str]], ...] = (
    (
        IntlFeeCategory.GEOPOLITICS,
        frozenset(
            {
                "geopolitics",
                "world affairs",
                "middle east",
                "ukraine",
                "nato",
                "israel",
                "russia",
                "iran",
                "war",
            }
        ),
    ),
    (
        IntlFeeCategory.CRYPTO,
        frozenset(
            {
                "crypto",
                "bitcoin",
                "ethereum",
                "solana",
                "defi",
                "nft",
                "stablecoin",
                "airdrops",
                # Token-launch markets: classed crypto (0.07) rather than
                # finance (0.04). Both the likelier reading and the
                # conservative one — understating a fee invents edge.
                "fdv",
                "pre-market",
                "token sales",
                "public sales",
                "ethgas",
            }
        ),
    ),
    (
        IntlFeeCategory.SPORTS,
        frozenset(
            {
                "sports",
                "nba",
                "nfl",
                "mlb",
                "nhl",
                "soccer",
                "football",
                "basketball",
                "baseball",
                "hockey",
                "tennis",
                "golf",
                "ufc",
                "mma",
                "boxing",
                "cricket",
                "f1",
                "olympics",
                "esports",
                "chess",
            }
        ),
    ),
    (IntlFeeCategory.WEATHER, frozenset({"weather", "climate", "hurricane", "temperature"})),
    (IntlFeeCategory.MENTIONS, frozenset({"mentions", "mention markets"})),
    (IntlFeeCategory.TECH, frozenset({"tech", "ai", "technology", "openai", "spacex"})),
    (
        IntlFeeCategory.ECONOMICS,
        frozenset({"economy", "economics", "inflation", "fed", "fed rates", "macro", "jobs"}),
    ),
    (IntlFeeCategory.FINANCE, frozenset({"finance", "stocks", "earnings", "ipo", "business"})),
    (
        IntlFeeCategory.POLITICS,
        frozenset(
            {
                "politics",
                "elections",
                "election",
                "trump",
                "congress",
                "senate",
                "white house",
                "supreme court",
            }
        ),
    ),
    (
        IntlFeeCategory.CULTURE,
        frozenset(
            {
                "pop culture",
                "culture",
                "entertainment",
                "movies",
                "music",
                "awards",
                "oscars",
                "grammys",
                "celebrities",
                "tv",
                "gaming",
            }
        ),
    ),
)


def intl_fee_category(tags: list[str] | None) -> IntlFeeCategory:
    """Resolve a market's INTL fee category from its gamma tags.

    Falls back to OTHER (0.05) — the conservative choice, since under-pricing
    the fee is what manufactures phantom edge.
    """
    if not tags:
        return IntlFeeCategory.OTHER
    lowered = {t.casefold() for t in tags}
    for category, keywords in _INTL_TAG_BUCKETS:
        if lowered & keywords:
            return category
    return IntlFeeCategory.OTHER


def polymarket_intl_taker_fee(
    price: object,
    contracts: object,
    *,
    category: IntlFeeCategory = IntlFeeCategory.OTHER,
) -> Decimal:
    """Polymarket International taker fee: coef(category) * C * p * (1-p).

    Rounded to 5 decimals; results below the 0.00001 minimum charge are zero.
    Makers are never charged (see `polymarket_intl_maker_fee`).
    """
    p, c = _validate(price, contracts)
    raw = _quadratic(INTL_TAKER_COEF[category], p, c)
    fee = raw.quantize(INTL_QUANTUM, rounding=ROUND_HALF_EVEN)
    return Decimal("0") if fee < INTL_QUANTUM else fee


def polymarket_intl_maker_fee(price: object, contracts: object) -> Decimal:
    """Zero. Makers are never charged on INTL.

    Note what this does NOT model: makers receive a share of the taker-fee
    pool, but it is paid daily and pro-rata across all makers, so it is not
    knowable per fill. Treat it as an unmodelled upside, never as a cost
    reduction — see `INTL_MAKER_REBATE_SHARE`.
    """
    _validate(price, contracts)
    return Decimal("0")


def polymarket_intl_effective_taker_cost(
    price: object, *, category: IntlFeeCategory = IntlFeeCategory.OTHER
) -> Decimal:
    """Per-contract all-in cost of taking YES at `price` on INTL, unrounded."""
    p, _ = _validate(price, 1)
    return p + INTL_TAKER_COEF[category] * p * (ONE - p)


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
