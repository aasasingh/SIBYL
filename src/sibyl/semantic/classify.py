"""Domain taxonomy assignment (Phase 2 needs coarse labels; the LLM
classifier from spec §4.3 refines this in Phase 4).

The six calibration domains from spec §2.3.4 plus `other`. Mapping is
deterministic and conservative: unknown categories land in `other` rather
than guessing, because a wrong domain label silently poisons the per-cell
calibration fit downstream.
"""

from __future__ import annotations

from enum import StrEnum


class Domain(StrEnum):
    POLITICS = "politics"
    SPORTS = "sports"
    CRYPTO = "crypto"
    FINANCE = "finance"
    WEATHER = "weather"
    ENTERTAINMENT = "entertainment"
    OTHER = "other"


# Kalshi event categories observed on the venue (exact-match after casefold).
_KALSHI_CATEGORY: dict[str, Domain] = {
    "politics": Domain.POLITICS,
    "elections": Domain.POLITICS,
    "world": Domain.POLITICS,
    "sports": Domain.SPORTS,
    "crypto": Domain.CRYPTO,
    "cryptocurrency": Domain.CRYPTO,
    "economics": Domain.FINANCE,
    "financials": Domain.FINANCE,
    "companies": Domain.FINANCE,
    "climate and weather": Domain.WEATHER,
    "weather": Domain.WEATHER,
    "entertainment": Domain.ENTERTAINMENT,
    "culture": Domain.ENTERTAINMENT,
}

# Polymarket tag keywords, checked in priority order: the first bucket with a
# matching tag wins. Sports outranks politics so "NFL" + "Washington" events
# don't become politics; weather outranks finance for hurricane-damage markets.
_POLYMARKET_TAG_BUCKETS: tuple[tuple[Domain, frozenset[str]], ...] = (
    (
        Domain.SPORTS,
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
                "formula 1",
                "olympics",
                "esports",
                "chess",
            }
        ),
    ),
    (
        Domain.POLITICS,
        frozenset(
            {
                "politics",
                "elections",
                "election",
                "geopolitics",
                "trump",
                "congress",
                "senate",
                "white house",
                "supreme court",
                "world affairs",
                "middle east",
                "ukraine",
                "china",
                "nato",
            }
        ),
    ),
    (
        Domain.CRYPTO,
        frozenset(
            {"crypto", "bitcoin", "ethereum", "solana", "defi", "nft", "stablecoin", "airdrops"}
        ),
    ),
    (
        Domain.WEATHER,
        frozenset({"weather", "climate", "hurricane", "temperature", "heat", "snow"}),
    ),
    (
        Domain.FINANCE,
        frozenset(
            {
                "economy",
                "business",
                "finance",
                "stocks",
                "fed",
                "fed rates",
                "inflation",
                "macro",
                "earnings",
                "ipo",
                "tech",
                "ai",
            }
        ),
    ),
    (
        Domain.ENTERTAINMENT,
        frozenset(
            {
                "pop culture",
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


def domain_from_kalshi_category(category: str | None) -> Domain:
    if not category:
        return Domain.OTHER
    return _KALSHI_CATEGORY.get(category.casefold(), Domain.OTHER)


def domain_from_polymarket_tags(tags: list[str] | None) -> Domain:
    if not tags:
        return Domain.OTHER
    lowered = {t.casefold() for t in tags}
    for domain, keywords in _POLYMARKET_TAG_BUCKETS:
        if lowered & keywords:
            return domain
    return Domain.OTHER
