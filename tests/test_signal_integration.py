"""End-to-end: book -> depth-walked INTL cost -> recalibrated signal.

This is the seam where Phase 1 (costs), Phase 3 (calibration) and Phase 5
(alpha) meet, and where the two rules that matter get enforced together: the
cost is the one you'd actually pay for your size, and the edge is measured
against the posterior lower bound.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from functools import partial

import numpy as np

from sibyl.calibration.apply import recalibrate
from sibyl.calibration.fit import CellData, fit_cell
from sibyl.costs.execution import buy_yes
from sibyl.costs.fees import (
    IntlFeeCategory,
    polymarket_intl_taker_fee,
)
from sibyl.venues.normalise import BinaryBook, Level, Venue

NOW = datetime(2026, 10, 11, 12, 0, tzinfo=UTC)


def _level(price: str, size: str) -> Level:
    return Level(price=Decimal(price), size=Decimal(size))


def _compressed_cell(slope: float = 2.2, seed: int = 31) -> CellData:
    rng = np.random.default_rng(seed)
    prices, ys, ids = [], [], []
    for m in range(600):
        base = float(rng.uniform(0.08, 0.92))
        true_p = base**slope / (base**slope + (1 - base) ** slope)
        outcome = float(rng.random() < true_p)
        for _ in range(8):
            prices.append(float(np.clip(base + rng.normal(0, 0.01), 0.02, 0.98)))
            ys.append(outcome)
            ids.append(f"M{m:04d}")
    return CellData(
        domain="politics",
        horizon="2d-1w",
        price=np.array(prices),
        y=np.array(ys),
        market_id=np.array(ids, dtype=np.str_),
    )


# A thin book: only 50 contracts at the top, then it gets worse fast.
BOOK = BinaryBook(
    venue=Venue.POLYMARKET_INTL,
    market_id="0xabc",
    ts=NOW,
    yes_bids=(_level("0.68", "100"),),
    yes_asks=(_level("0.70", "50"), _level("0.74", "100"), _level("0.80", "500")),
)

POLITICS_FEE = partial(polymarket_intl_taker_fee, category=IntlFeeCategory.POLITICS)
GEO_FEE = partial(polymarket_intl_taker_fee, category=IntlFeeCategory.GEOPOLITICS)


def test_edge_shrinks_monotonically_as_the_order_walks_the_book() -> None:
    """Edge must decay with size, because cost is depth-walked.

    Pricing size at top-of-book is the error this guards: the quote is the
    same 0.70 at every size, so a system that ignored depth would report an
    identical edge for 10 contracts and 650.
    """
    cell = fit_cell(_compressed_cell(), n_boot=200)
    assert cell.tradeable, "fixture should produce a tradeable cell"

    edges = []
    costs = []
    for size in ("10", "50", "200", "650"):
        est = buy_yes(BOOK, Decimal(size), POLITICS_FEE)
        assert est.effective_price is not None
        costs.append(est.effective_price)
        edges.append(recalibrate(Decimal("0.70"), cell, est.effective_price).edge)

    assert costs == sorted(costs), f"effective cost must rise with size: {costs}"
    assert edges == sorted(edges, reverse=True), f"edge must fall with size: {edges}"
    assert edges[0] > edges[-1]


def test_edge_dies_entirely_on_a_thin_expensive_book() -> None:
    """A tradeable cell is not enough: if the book makes you pay past the
    posterior lower bound, there is no trade."""
    cell = fit_cell(_compressed_cell(), n_boot=200)
    thin = BinaryBook(
        venue=Venue.POLYMARKET_INTL,
        market_id="0xthin",
        ts=NOW,
        yes_bids=(_level("0.68", "10"),),
        yes_asks=(_level("0.70", "10"), _level("0.95", "1000")),
    )
    est = buy_yes(thin, Decimal("500"), POLITICS_FEE)
    assert est.effective_price is not None
    signal = recalibrate(Decimal("0.70"), cell, est.effective_price)
    assert not signal.has_edge, (
        f"edge {signal.edge} survived an effective cost of {est.effective_price} "
        f"against a lower bound of {signal.fair_value_low}"
    )


def test_fee_free_geopolitics_leaves_more_edge_than_politics() -> None:
    """Geopolitics is fee-free on INTL, so the same book leaves more edge.
    This is the §2.2 insight that fees, not opinions, decide where edge
    survives — made concrete."""
    cell = fit_cell(_compressed_cell(), n_boot=200)
    politics = buy_yes(BOOK, Decimal("50"), POLITICS_FEE)
    geopolitics = buy_yes(BOOK, Decimal("50"), GEO_FEE)
    assert geopolitics.fee == 0
    assert politics.fee > 0
    assert geopolitics.effective_price is not None and politics.effective_price is not None

    geo_signal = recalibrate(Decimal("0.70"), cell, geopolitics.effective_price)
    pol_signal = recalibrate(Decimal("0.70"), cell, politics.effective_price)
    assert geo_signal.edge > pol_signal.edge


def test_no_signal_when_cell_has_no_demonstrated_edge() -> None:
    """A calibrated cell must produce nothing, however cheap the book is."""
    rng = np.random.default_rng(21)
    prices, ys, ids = [], [], []
    for m in range(300):
        base = float(rng.uniform(0.08, 0.92))
        outcome = float(rng.random() < base)  # b = 1 exactly
        for _ in range(8):
            prices.append(base)
            ys.append(outcome)
            ids.append(f"M{m:04d}")
    flat = CellData(
        domain="politics",
        horizon="2d-1w",
        price=np.array(prices),
        y=np.array(ys),
        market_id=np.array(ids, dtype=np.str_),
    )
    cell = fit_cell(flat, n_boot=200)
    cheap = buy_yes(BOOK, Decimal("50"), GEO_FEE)  # free fees, tight fill
    assert cheap.effective_price is not None
    signal = recalibrate(Decimal("0.70"), cell, cheap.effective_price)
    assert not signal.has_edge
    assert signal.edge == 0
    assert signal.reason  # must explain itself


def test_partial_fill_is_visible_to_the_caller() -> None:
    """Depth beyond the book must not silently become a filled order."""
    est = buy_yes(BOOK, Decimal("10_000"), POLITICS_FEE)
    assert not est.complete
    assert est.filled == Decimal("650")
    assert est.requested == Decimal("10000")
