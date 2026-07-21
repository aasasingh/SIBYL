"""Phase-0 risk plumbing: kill switch, position/event caps, daily loss limit."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from sibyl.config import RiskSettings
from sibyl.risk.limits import RiskGovernor, TradingHalted


@pytest.fixture
def governor(tmp_path: Path) -> RiskGovernor:
    return RiskGovernor(
        RiskSettings(
            max_position_usd=Decimal("100"),
            max_event_exposure_usd=Decimal("150"),
            daily_loss_limit_usd=Decimal("50"),
            kill_switch_file=tmp_path / "KILL_SWITCH",
        )
    )


class TestKillSwitch:
    def test_clean_governor_allows_trade(self, governor: RiskGovernor) -> None:
        governor.pre_trade("MKT-1", "EVT-1", Decimal("10"))

    def test_file_engages_switch(self, governor: RiskGovernor, tmp_path: Path) -> None:
        (tmp_path / "KILL_SWITCH").touch()
        with pytest.raises(TradingHalted, match="kill switch"):
            governor.pre_trade("MKT-1", "EVT-1", Decimal("10"))

    def test_programmatic_engage_writes_file(self, governor: RiskGovernor, tmp_path: Path) -> None:
        governor.kill_switch.engage("manual halt")
        assert (tmp_path / "KILL_SWITCH").exists()
        with pytest.raises(TradingHalted):
            governor.pre_trade("MKT-1", None, Decimal("10"))


class TestDailyLossLimit:
    def test_loss_limit_engages_kill_switch(self, governor: RiskGovernor) -> None:
        governor.record_pnl(Decimal("-50"))
        with pytest.raises(TradingHalted, match="kill switch"):
            governor.pre_trade("MKT-1", None, Decimal("10"))

    def test_small_loss_does_not_halt(self, governor: RiskGovernor) -> None:
        governor.record_pnl(Decimal("-49.99"))
        governor.pre_trade("MKT-1", None, Decimal("10"))

    def test_pnl_resets_on_new_day_but_kill_switch_does_not(self, governor: RiskGovernor) -> None:
        day1 = datetime(2026, 7, 21, 23, 0, tzinfo=UTC)
        day2 = datetime(2026, 7, 22, 1, 0, tzinfo=UTC)
        governor.record_pnl(Decimal("-50"), now=day1)
        # new day resets the counter, but a tripped kill switch stays tripped
        governor.record_pnl(Decimal("0"), now=day2)
        with pytest.raises(TradingHalted):
            governor.pre_trade("MKT-1", None, Decimal("10"))


class TestExposureCaps:
    def test_position_cap(self, governor: RiskGovernor) -> None:
        governor.record_fill("MKT-1", "EVT-1", Decimal("95"))
        with pytest.raises(TradingHalted, match="position cap"):
            governor.pre_trade("MKT-1", "EVT-1", Decimal("10"))
        governor.pre_trade("MKT-1", "EVT-1", Decimal("5"))  # exactly at cap is fine

    def test_event_exposure_cap_spans_markets(self, governor: RiskGovernor) -> None:
        governor.record_fill("MKT-1", "EVT-1", Decimal("90"))
        governor.record_fill("MKT-2", "EVT-1", Decimal("55"))
        with pytest.raises(TradingHalted, match="event exposure"):
            governor.pre_trade("MKT-3", "EVT-1", Decimal("10"))

    def test_non_positive_notional_rejected(self, governor: RiskGovernor) -> None:
        with pytest.raises(TradingHalted, match="non-positive"):
            governor.pre_trade("MKT-1", None, Decimal("0"))
