"""Phase 6 is closed by construction, not by a flag someone might flip.

The operator is an Ontario resident; Polymarket is under a permanent OSC
undertaking not to serve Ontario residents. These tests pin that as settled
configuration rather than an open question.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sibyl.config import BLOCKED_PROVINCES, Settings


class TestLiveTradingGate:
    def test_default_province_is_ontario_and_blocks_live(self) -> None:
        settings = Settings()
        assert settings.operator_province == "ON"
        assert not settings.live_trading_permitted
        assert not settings.live

    def test_live_true_is_refused_in_a_blocked_province(self) -> None:
        for province in sorted(BLOCKED_PROVINCES):
            with pytest.raises(ValidationError, match="not available"):
                Settings(live=True, operator_province=province)

    def test_blocked_set_covers_the_researched_provinces(self) -> None:
        assert {"ON", "AB", "BC", "QC"} == BLOCKED_PROVINCES

    def test_case_insensitive(self) -> None:
        assert not Settings(operator_province="on").live_trading_permitted

    def test_research_and_paper_are_unaffected(self) -> None:
        """The gate blocks routing real orders, not reading or simulating."""
        settings = Settings()
        assert settings.polymarket.enabled
        assert not settings.live_trading_permitted

    def test_a_permitting_province_still_needs_mainnet_and_prod(self) -> None:
        """Jurisdiction is necessary, not sufficient: the pre-existing
        testnet/demo coherence checks still apply."""
        with pytest.raises(ValidationError):
            Settings(live=True, operator_province="MB")  # still on Amoy + demo
