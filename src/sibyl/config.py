"""Typed configuration. Jurisdiction and live-trading are explicit config,
never implicit properties of whichever API key happens to be loaded (§2.5).

Environment variables use the SIBYL_ prefix with __ as the nesting delimiter,
e.g. SIBYL_KALSHI__API_KEY_ID. Secrets (key PEMs) are referenced by path and
must live outside git.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from sibyl.venues.normalise import Venue


class KalshiSettings(BaseModel):
    env: Literal["demo", "prod"] = "demo"
    api_key_id: str | None = None
    private_key_path: Path | None = None
    enabled: bool = True


class PolymarketSettings(BaseModel):
    # US and International are legally distinct venues; exactly one is active.
    platform: Literal["us", "intl"] = "us"
    chain_id: int = 80002  # Polygon Amoy testnet until Phase 6; 137 = mainnet
    enabled: bool = True

    @property
    def venue(self) -> Venue:
        return Venue.POLYMARKET_US if self.platform == "us" else Venue.POLYMARKET_INTL


class RiskSettings(BaseModel):
    max_position_usd: Decimal = Decimal("100")
    max_event_exposure_usd: Decimal = Decimal("250")
    daily_loss_limit_usd: Decimal = Decimal("50")
    kill_switch_file: Path = Path("KILL_SWITCH")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SIBYL_",
        env_nested_delimiter="__",
        frozen=True,
    )

    live: bool = False  # Phase 6 gate. Loud on purpose; everything else is paper.
    kalshi: KalshiSettings = KalshiSettings()
    polymarket: PolymarketSettings = PolymarketSettings()
    risk: RiskSettings = RiskSettings()

    @model_validator(mode="after")
    def _no_live_on_demo(self) -> Settings:
        if self.live and self.kalshi.env == "demo":
            raise ValueError(
                "live=True with kalshi.env='demo' is incoherent; refuse to guess intent"
            )
        if self.live and self.polymarket.chain_id != 137:
            raise ValueError("live=True requires polymarket.chain_id=137 (mainnet)")
        return self


class VenueEligibility(BaseModel):
    """Which venues this deployment may touch, and why — resolved once at
    startup and logged. Routing to a disabled venue is a hard error."""

    model_config = {"frozen": True}

    enabled: frozenset[Venue]
    reasons: dict[Venue, str]

    def require(self, venue: Venue) -> None:
        if venue not in self.enabled:
            raise PermissionError(
                f"venue {venue} is disabled: {self.reasons.get(venue, 'not configured')}"
            )


def resolve_eligibility(settings: Settings) -> VenueEligibility:
    enabled: set[Venue] = set()
    reasons: dict[Venue, str] = {}

    if settings.kalshi.enabled:
        enabled.add(Venue.KALSHI)
        reasons[Venue.KALSHI] = f"enabled via config (env={settings.kalshi.env})"
    else:
        reasons[Venue.KALSHI] = "disabled via config"

    active = settings.polymarket.venue
    inactive = Venue.POLYMARKET_INTL if active is Venue.POLYMARKET_US else Venue.POLYMARKET_US
    if settings.polymarket.enabled:
        enabled.add(active)
        reasons[active] = (
            f"enabled via config (platform={settings.polymarket.platform}, "
            f"chain={settings.polymarket.chain_id})"
        )
    else:
        reasons[active] = "disabled via config"
    reasons[inactive] = "not the configured Polymarket platform; books must never be joined"

    return VenueEligibility(enabled=frozenset(enabled), reasons=reasons)
