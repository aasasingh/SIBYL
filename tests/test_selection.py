"""Recorder market-selection tests.

The first recorder ranked markets by lifetime volume and ended up watching
long-dated longshots pinned at 0.001/0.999, where 97.8% of trades were one
cent-sized bot. These pin down the replacement rule.
"""

from __future__ import annotations

import json
from typing import Any

from sibyl.record.recorder import choose_markets


def market(
    mid: str = "0.50",
    *,
    id_: str = "1",
    event: str | None = "e1",
    spread: str = "0.01",
    accepting: bool = True,
    book: bool = True,
) -> dict[str, Any]:
    from decimal import Decimal

    half = Decimal(spread) / 2
    return {
        "id": id_,
        "conditionId": f"0x{id_}",
        "question": f"Q{id_}",
        "bestBid": str(Decimal(mid) - half),
        "bestAsk": str(Decimal(mid) + half),
        "enableOrderBook": book,
        "acceptingOrders": accepting,
        "events": [{"id": event}] if event else [],
        "clobTokenIds": json.dumps([f"{id_}Y", f"{id_}N"]),
        "volume24hr": 1000,
    }


class TestChooseMarkets:
    def test_keeps_rank_order_and_both_legs(self) -> None:
        assets, manifest = choose_markets([market(id_="a", event="x"), market(id_="b", event="y")])
        assert assets == ["aY", "aN", "bY", "bN"]
        assert [m["market_id"] for m in manifest] == ["a", "b"]

    def test_extreme_mids_are_excluded(self) -> None:
        """The failure that motivated the rule: books pinned at the edges."""
        _, manifest = choose_markets(
            [
                market("0.002", id_="longshot", event="a"),
                market("0.998", id_="lock", event="b"),
                market("0.40", id_="live", event="c"),
            ]
        )
        assert [m["market_id"] for m in manifest] == ["live"]

    def test_one_sided_or_crossed_books_are_excluded(self) -> None:
        one_sided = market(id_="a", event="a") | {"bestBid": "0"}
        missing = market(id_="b", event="b") | {"bestAsk": None}
        crossed = market(id_="c", event="c") | {"bestBid": "0.6", "bestAsk": "0.5"}
        assert choose_markets([one_sided, missing, crossed]) == ([], [])

    def test_closed_books_are_excluded(self) -> None:
        _, manifest = choose_markets(
            [market(id_="a", event="a", accepting=False), market(id_="b", event="b", book=False)]
        )
        assert manifest == []

    def test_one_event_cannot_take_every_slot(self) -> None:
        many = [market(id_=str(i), event="election") for i in range(10)]
        other = market(id_="other", event="fed")
        _, manifest = choose_markets([*many, other], max_per_event=3)
        assert [m["market_id"] for m in manifest] == ["0", "1", "2", "other"]

    def test_markets_without_an_event_are_their_own_group(self) -> None:
        _, manifest = choose_markets(
            [market(id_=str(i), event=None) for i in range(5)], max_per_event=1
        )
        assert len(manifest) == 5

    def test_caps_on_markets_and_assets(self) -> None:
        cands = [market(id_=str(i), event=str(i)) for i in range(20)]
        assert len(choose_markets(cands, n_markets=5)[1]) == 5
        assets, _ = choose_markets(cands, max_assets=7)
        assert len(assets) == 6  # never half a market

    def test_manifest_records_the_quote_at_selection(self) -> None:
        _, [m] = choose_markets([market("0.40")])
        assert (m["best_bid"], m["best_ask"], m["event_id"]) == ("0.395", "0.405", "e1")
