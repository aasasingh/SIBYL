"""Order-book replay tests.

Every queue-position and markout number in the fill model reads this
ladder. These tests pin down the rules that keep the replay honest: state is
only trusted after a snapshot, a gap makes it unknown again, and every claim
the venue makes about the book is checked rather than assumed.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from sibyl.fill.ladder import (
    BookSnapshot,
    ChangeFrame,
    Ladder,
    LevelChange,
    Replayer,
    Reset,
    parse_event,
    with_resets,
)
from sibyl.fill.replay import replay
from sibyl.record.writer import EVENTS_SCHEMA

T0 = datetime(2026, 10, 12, 12, 0, tzinfo=UTC)
D = Decimal


def _t(seconds: float) -> datetime:
    return T0 + timedelta(seconds=seconds)


def snap(asset: str, bids: dict[str, str], asks: dict[str, str], at: float = 0.0) -> BookSnapshot:
    return BookSnapshot(
        asset_id=asset,
        recv_ts=_t(at),
        venue_ts=int(at * 1000),
        bids={D(p): D(s) for p, s in bids.items()},
        asks={D(p): D(s) for p, s in asks.items()},
    )


def change(
    asset: str, side: str, price: str, size: str, best_bid: str, best_ask: str
) -> LevelChange:
    return LevelChange(asset, side, D(price), D(size), D(best_bid), D(best_ask))  # type: ignore[arg-type]


def frame(at: float, *changes: LevelChange, venue_ms: int | None = None) -> ChangeFrame:
    return ChangeFrame(
        recv_ts=_t(at),
        venue_ts=venue_ms if venue_ms is not None else int(at * 1000),
        changes=changes,
    )


class TestLadder:
    def test_set_level_adds_updates_and_removes(self) -> None:
        lad = Ladder()
        lad.set_level("bid", D("0.40"), D("100"))
        lad.set_level("bid", D("0.41"), D("50"))
        lad.set_level("ask", D("0.45"), D("10"))
        assert (lad.best_bid, lad.best_ask) == (D("0.41"), D("0.45"))
        lad.set_level("bid", D("0.41"), D("0"))
        assert lad.best_bid == D("0.40")
        assert D("0.41") not in lad.bids

    def test_empty_sides_have_no_touch(self) -> None:
        assert (Ladder().best_bid, Ladder().best_ask) == (None, None)


class TestSnapshotValidation:
    def test_replayed_deltas_that_reach_the_next_snapshot_agree(self) -> None:
        r = Replayer()
        r.apply(snap("A", {"0.40": "100"}, {"0.45": "10"}))
        r.apply(frame(1, change("A", "bid", "0.41", "5", "0.41", "0.45")))
        r.apply(
            frame(
                2,
                change("A", "ask", "0.45", "0", "0.41", "0.46"),
                change("A", "ask", "0.46", "7", "0.41", "0.46"),
            )
        )
        r.apply(snap("A", {"0.40": "100", "0.41": "5"}, {"0.46": "7"}, at=3))
        s = r.run([])
        assert (s.snapshot_checks, s.snapshot_matches) == (1, 1)
        assert s.top_checks == 2 and s.top_matches == 2

    def test_a_disagreeing_snapshot_is_reported_then_resynced(self) -> None:
        r = Replayer()
        r.apply(snap("A", {"0.40": "100"}, {"0.45": "10"}))
        # Venue says 0.40 holds 80, not the 100 we have: a missed delta.
        r.apply(snap("A", {"0.40": "80"}, {"0.45": "10"}, at=1))
        # After resync the next snapshot must agree: errors never propagate.
        r.apply(snap("A", {"0.40": "80"}, {"0.45": "10"}, at=2))
        s = r.run([])
        assert s.snapshot_checks == 2
        assert s.snapshot_matches == 1
        assert s.snapshot_top_mismatches == 0  # deep-book only
        assert s.examples[0].kind == "snapshot"
        assert "bid@0.40: 100->80" in s.examples[0].detail

    def test_a_mismatch_at_the_touch_is_counted_separately(self) -> None:
        r = Replayer()
        r.apply(snap("A", {"0.40": "100"}, {"0.45": "10"}))
        r.apply(snap("A", {"0.40": "100", "0.42": "1"}, {"0.45": "10"}, at=1))
        assert r.run([]).snapshot_top_mismatches == 1

    def test_the_first_snapshot_is_a_sync_not_a_check(self) -> None:
        s = Replayer().run([snap("A", {"0.40": "1"}, {})])
        assert (s.snapshots, s.snapshot_checks) == (1, 0)


class TestUnknownState:
    def test_changes_before_any_snapshot_are_dropped_not_guessed(self) -> None:
        r = Replayer()
        s = r.run([frame(0, change("A", "bid", "0.40", "100", "0.40", "1"))])
        assert s.changes_unsynced == 1
        assert s.changes_applied == 0
        assert "A" not in r.ladders

    def test_a_reset_makes_state_unknown_until_the_next_snapshot(self) -> None:
        r = Replayer()
        s = r.run(
            [
                snap("A", {"0.40": "100"}, {"0.45": "10"}),
                Reset(_t(1), "coverage"),
                frame(2, change("A", "bid", "0.41", "5", "0.41", "0.45")),
                # Resync after the gap is not scored against pre-gap state.
                snap("A", {"0.39": "1"}, {"0.50": "1"}, at=3),
            ]
        )
        assert s.resets == 1
        assert s.changes_unsynced == 1
        assert s.snapshot_checks == 0


class TestTopOfBookValidation:
    def _crossing(self, *, same_message: bool) -> int:
        """A sell crossing a resting bid at 0.988: the venue sends the new ask
        and the consumed bid's removal as two events, both stamped with the
        touch after the whole message (0.987 / 0.988)."""
        r = Replayer()
        second_at = 1.0 if same_message else 1.5
        stats = r.run(
            [
                snap("A", {"0.987": "100", "0.988": "296"}, {"0.989": "50"}),
                frame(1, change("A", "ask", "0.988", "149.5", "0.987", "0.988"), venue_ms=1000),
                frame(second_at, change("A", "bid", "0.988", "0", "0.987", "0.988"), venue_ms=1000),
            ]
        )
        return stats.top_checks - stats.top_matches

    def test_venue_touch_is_checked_per_message_not_per_event(self) -> None:
        assert self._crossing(same_message=True) == 0

    def test_split_across_messages_the_same_events_would_mismatch(self) -> None:
        # Documents why the grouping exists: per-event checking flags a
        # replayed locked book (0.988/0.988) that never existed.
        assert self._crossing(same_message=False) == 1

    def test_venue_sentinels_for_an_empty_side_are_respected(self) -> None:
        r = Replayer()
        stats = r.run(
            [
                snap("A", {"0.40": "1"}, {}),
                frame(1, change("A", "bid", "0.40", "0", "0", "1")),
            ]
        )
        assert stats.top_matches == 1

    def test_a_wrong_touch_is_reported(self) -> None:
        r = Replayer()
        stats = r.run(
            [
                snap("A", {"0.40": "1"}, {"0.45": "1"}),
                frame(1, change("A", "bid", "0.41", "1", "0.40", "0.45")),
            ]
        )
        assert stats.top_matches == 0
        assert stats.examples[0].kind == "top"


class TestParsing:
    def test_book_drops_zero_levels(self) -> None:
        payload = json.dumps(
            {
                "asset_id": "A",
                "bids": [{"price": "0.4", "size": "10"}, {"price": "0.3", "size": "0"}],
                "asks": [{"price": "0.6", "size": "5"}],
            }
        )
        ev = parse_event("book", payload, T0, 1)
        assert isinstance(ev, BookSnapshot)
        assert ev.bids == {D("0.4"): D("10")}
        assert ev.asks == {D("0.6"): D("5")}

    def test_price_change_maps_sides_and_keeps_every_asset(self) -> None:
        payload = json.dumps(
            {
                "price_changes": [
                    {
                        "asset_id": "Y",
                        "price": "0.48",
                        "size": "1875",
                        "side": "BUY",
                        "best_bid": "0.969",
                        "best_ask": "0.97",
                    },
                    {
                        "asset_id": "N",
                        "price": "0.52",
                        "size": "1875",
                        "side": "SELL",
                        "best_bid": "0.03",
                        "best_ask": "0.031",
                    },
                ]
            }
        )
        ev = parse_event("price_change", payload, T0, 1)
        assert isinstance(ev, ChangeFrame)
        assert [(c.asset_id, c.side) for c in ev.changes] == [("Y", "bid"), ("N", "ask")]
        assert ev.changes[0].best_ask == D("0.97")

    def test_other_event_types_are_ignored(self) -> None:
        assert parse_event("last_trade_price", "{}", T0, 1) is None


class TestResets:
    def test_boundaries_and_silences_both_reset(self) -> None:
        events = [
            (_t(0), snap("A", {}, {})),
            (_t(5), snap("A", {}, {}, at=5)),
            (_t(60), snap("A", {}, {}, at=60)),  # 55s of silence
        ]
        out = list(with_resets(events, boundaries=[_t(3)], silence_s=20))
        resets = [e for e in out if isinstance(e, Reset)]
        assert [r.reason for r in resets] == ["coverage", "silence"]
        assert out.index(resets[0]) == 1  # before the event at t=5


def _write_part(path: Path, rows: list[dict[str, object]]) -> None:
    pq.write_table(pa.Table.from_pylist(rows, schema=EVENTS_SCHEMA), path)


class TestReplayFromDisk:
    def test_end_to_end_with_duplicated_recording(self, tmp_path: Path) -> None:
        """Two recorders running at once store every event twice. Replay must
        see each venue event once."""
        events_dir = tmp_path / "book_events"
        events_dir.mkdir()
        book = {
            "event_type": "book",
            "asset_id": "A",
            "timestamp": "1",
            "bids": [{"price": "0.40", "size": "100"}],
            "asks": [{"price": "0.45", "size": "10"}],
        }
        delta = {
            "event_type": "price_change",
            "timestamp": "2",
            "price_changes": [
                {
                    "asset_id": "A",
                    "price": "0.40",
                    "size": "60",
                    "side": "BUY",
                    "best_bid": "0.40",
                    "best_ask": "0.45",
                }
            ],
        }
        book2 = {**book, "timestamp": "3", "bids": [{"price": "0.40", "size": "60"}]}
        rows = [
            {
                "recv_ts": _t(i),
                "venue_ts": i + 1,
                "event_type": e["event_type"],
                "asset_id": "A",
                "market": "m",
                "payload": json.dumps(e),
            }
            for i, e in enumerate([book, delta, book2])
        ]
        _write_part(events_dir / "part-a.parquet", rows)
        once = replay(tmp_path).summary()
        _write_part(events_dir / "part-b.parquet", rows)
        twice = replay(tmp_path).summary()
        assert once == twice
        assert once["snapshot_checks"] == 1
        assert once["snapshot_agreement"] == 1.0
        assert once["changes_applied"] == 1

    def test_no_recordings_is_not_an_error(self, tmp_path: Path) -> None:
        assert replay(tmp_path).summary()["snapshots"] == 0
