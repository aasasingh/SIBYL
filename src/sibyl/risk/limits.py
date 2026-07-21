"""Kill switch, position caps, daily loss limit — present from Phase 0 (§3.5).

Every order path (paper included) must call `RiskGovernor.pre_trade(...)`.
The kill switch is file-based so a human can halt the system with `touch
KILL_SWITCH` even if the process is wedged, plus a programmatic engage.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from sibyl.config import RiskSettings


class TradingHalted(RuntimeError):
    """Raised when any risk limit blocks a trade. Never catch-and-continue."""


class KillSwitch:
    def __init__(self, file: Path) -> None:
        self._file = file
        self._engaged_reason: str | None = None

    def engage(self, reason: str) -> None:
        self._engaged_reason = reason
        self._file.write_text(f"{datetime.now(UTC).isoformat()} {reason}\n", encoding="utf-8")

    @property
    def engaged(self) -> bool:
        return self._engaged_reason is not None or self._file.exists()

    @property
    def reason(self) -> str:
        if self._engaged_reason:
            return self._engaged_reason
        if self._file.exists():
            return f"kill-switch file present: {self._file}"
        return "not engaged"


class RiskGovernor:
    """Aggregates the Phase-0 limits. Exposure is tracked per market and per
    event; atomic-outcome aggregation lands with the semantic layer."""

    def __init__(self, settings: RiskSettings) -> None:
        self._settings = settings
        self.kill_switch = KillSwitch(settings.kill_switch_file)
        self._positions_usd: dict[str, Decimal] = {}
        self._event_exposure_usd: dict[str, Decimal] = {}
        self._daily_pnl = Decimal("0")
        self._pnl_day = datetime.now(UTC).date()

    def record_pnl(self, pnl_usd: Decimal, *, now: datetime | None = None) -> None:
        now = now or datetime.now(UTC)
        if now.date() != self._pnl_day:
            self._pnl_day = now.date()
            self._daily_pnl = Decimal("0")
        self._daily_pnl += pnl_usd
        if self._daily_pnl <= -self._settings.daily_loss_limit_usd:
            self.kill_switch.engage(
                f"daily loss limit hit: pnl={self._daily_pnl} "
                f"limit={self._settings.daily_loss_limit_usd}"
            )

    def record_fill(self, market_id: str, event_id: str | None, notional_usd: Decimal) -> None:
        self._positions_usd[market_id] = (
            self._positions_usd.get(market_id, Decimal("0")) + notional_usd
        )
        if event_id is not None:
            self._event_exposure_usd[event_id] = (
                self._event_exposure_usd.get(event_id, Decimal("0")) + notional_usd
            )

    def pre_trade(self, market_id: str, event_id: str | None, notional_usd: Decimal) -> None:
        """Raise TradingHalted unless this order is within every limit."""
        if self.kill_switch.engaged:
            raise TradingHalted(f"kill switch engaged: {self.kill_switch.reason}")
        if notional_usd <= 0:
            raise TradingHalted(f"non-positive notional: {notional_usd}")
        position = self._positions_usd.get(market_id, Decimal("0")) + notional_usd
        if position > self._settings.max_position_usd:
            raise TradingHalted(
                f"position cap: {market_id} would be {position} USD "
                f"(cap {self._settings.max_position_usd})"
            )
        if event_id is not None:
            exposure = self._event_exposure_usd.get(event_id, Decimal("0")) + notional_usd
            if exposure > self._settings.max_event_exposure_usd:
                raise TradingHalted(
                    f"event exposure cap: {event_id} would be {exposure} USD "
                    f"(cap {self._settings.max_event_exposure_usd})"
                )
