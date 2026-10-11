"""Walk-forward out-of-sample validation (spec §6 Phase 3).

An in-sample slope is a description of the past, not a prediction. The only
question that matters is: if we had fitted on data available up to time T,
would the resulting recalibration have beaten the raw price on markets that
resolved after T?

Folds are split on RESOLUTION TIME (close_time), not on row time, and a whole
EVENT belongs to one fold. Two leaks are being closed:

  * Splitting rows would leak directly -- a market's early hours would train
    the model that scores its later hours, and the label is a property of the
    market, so the answer is already in the training set.
  * Splitting markets would still leak through event structure. Markets in a
    NegRisk set are mutually exclusive: exactly one resolves YES. Train on
    "candidate A won" and the test fold's "did candidate B win?" is already
    answered. With 10.66 markets per event in this data, that is not an edge
    case, it is the common case.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from sibyl.calibration.fit import (
    CellData,
    CellFit,
    brier,
    expit,
    fit_cell,
    logit,
)
from sibyl.calibration.hierarchical import partial_pool

FloatArray = npt.NDArray[np.float64]


@dataclass
class Rows:
    """Flat calibration rows, as pulled from the resolution ledger."""

    domain: npt.NDArray[np.str_]
    horizon: npt.NDArray[np.str_]
    price: FloatArray
    y: FloatArray
    market_id: npt.NDArray[np.str_]
    close_time: FloatArray  # unix seconds; fold boundary key
    # Event id (fallback: market id). The unit of both clustering and
    # splitting -- see the module docstring.
    cluster_id: npt.NDArray[np.str_] | None = None

    def __len__(self) -> int:
        return int(self.price.shape[0])

    @property
    def clusters(self) -> npt.NDArray[np.str_]:
        return self.market_id if self.cluster_id is None else self.cluster_id

    def take(self, mask: npt.NDArray[np.bool_]) -> Rows:
        return Rows(
            domain=self.domain[mask],
            horizon=self.horizon[mask],
            price=self.price[mask],
            y=self.y[mask],
            market_id=self.market_id[mask],
            close_time=self.close_time[mask],
            cluster_id=None if self.cluster_id is None else self.cluster_id[mask],
        )

    def cells(self) -> list[CellData]:
        out: list[CellData] = []
        keys = np.char.add(np.char.add(self.domain, "|"), self.horizon)
        for key in np.unique(keys):
            mask = keys == key
            domain, horizon = str(key).split("|", 1)
            out.append(
                CellData(
                    domain=domain,
                    horizon=horizon,
                    price=self.price[mask],
                    y=self.y[mask],
                    market_id=self.market_id[mask],
                    cluster_id=None if self.cluster_id is None else self.cluster_id[mask],
                )
            )
        return out


@dataclass(frozen=True)
class FoldResult:
    fold: int
    train_rows: int
    test_rows: int
    train_markets: int
    test_markets: int
    brier_model: float
    brier_baseline: float
    n_cells_used: int
    n_cells_tradeable: int
    split_time: float
    train_clusters: int = 0
    test_clusters: int = 0

    @property
    def improvement(self) -> float:
        """Positive means the recalibration beat the raw price out of sample."""
        return self.brier_baseline - self.brier_model


def _cluster_level_split(
    rows: Rows, n_folds: int, min_train_frac: float
) -> list[tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_], float]]:
    """Expanding-window splits on close time, whole EVENTS per side.

    A cluster's position in time is its EARLIEST close, so an event cannot
    straddle the boundary and land partly in both folds.
    """
    cluster_keys = rows.clusters
    unique = np.unique(cluster_keys)
    # Earliest close per cluster: min, not first-seen, so ordering is
    # deterministic regardless of row order.
    order_by_key = np.argsort(cluster_keys, kind="stable")
    sorted_keys = cluster_keys[order_by_key]
    sorted_times = rows.close_time[order_by_key]
    boundaries = np.searchsorted(sorted_keys, unique)
    groups = np.split(sorted_times, boundaries[1:])
    cluster_close = np.array([g.min() for g in groups], dtype=np.float64)

    order = np.argsort(cluster_close, kind="stable")
    sorted_markets = unique[order]
    sorted_close = cluster_close[order]
    n_markets = sorted_markets.size

    splits: list[tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_], float]] = []
    start = int(n_markets * min_train_frac)
    if start < 1 or start >= n_markets:
        return splits
    step = max(1, (n_markets - start) // n_folds)
    for fold in range(n_folds):
        cut = start + fold * step
        if cut >= n_markets:
            break
        train_markets = set(sorted_markets[:cut].tolist())
        test_end = min(cut + step, n_markets)
        test_markets = set(sorted_markets[cut:test_end].tolist())
        if not test_markets:
            break
        train_mask = np.isin(cluster_keys, list(train_markets))
        test_mask = np.isin(cluster_keys, list(test_markets))
        splits.append((train_mask, test_mask, float(sorted_close[cut])))
    return splits


def walk_forward(
    rows: Rows,
    *,
    n_folds: int = 4,
    min_train_frac: float = 0.4,
    n_boot: int = 200,
    use_pooled: bool = True,
    tradeable_only: bool = False,
) -> list[FoldResult]:
    """Fit on each expanding window, score the next block of markets.

    `tradeable_only` scores only rows whose cell showed a CI excluding 1 in
    TRAINING — i.e. what the live system would actually act on. The default
    scores every row, which is the harder and more honest headline number.
    """
    results: list[FoldResult] = []
    for fold, (train_mask, test_mask, split_time) in enumerate(
        _cluster_level_split(rows, n_folds, min_train_frac)
    ):
        train, test = rows.take(train_mask), rows.take(test_mask)
        if len(train) == 0 or len(test) == 0:
            continue

        fits = [fit_cell(cell, n_boot=n_boot) for cell in train.cells()]
        if use_pooled:
            fits, mu, _tau = partial_pool(fits)
        else:
            mu = 1.0
        lookup: dict[tuple[str, str], CellFit] = {(f.domain, f.horizon): f for f in fits}

        # Score the test block with only training-time information.
        predicted = np.empty(len(test), dtype=np.float64)
        scored = np.zeros(len(test), dtype=bool)
        used: set[tuple[str, str]] = set()
        for idx in range(len(test)):
            key = (str(test.domain[idx]), str(test.horizon[idx]))
            cell = lookup.get(key)
            price = float(test.price[idx])
            if cell is None or not np.isfinite(cell.effective_slope):
                # Unseen cell: fall back to the population slope, which is
                # what a live system would have to do.
                slope = mu if use_pooled else 1.0
                intercept = 0.0
            else:
                if tradeable_only and not cell.tradeable:
                    continue
                slope = cell.effective_slope
                # A pooled cell can carry a finite (borrowed) slope while its
                # own intercept is NaN, because it never converged. Treat the
                # intercept as 0 there: pooling lends a slope, not an offset.
                intercept = cell.intercept if np.isfinite(cell.intercept) else 0.0
                used.add(key)
            predicted[idx] = float(expit(intercept + slope * logit(price)))
            scored[idx] = True

        if not scored.any():
            continue
        results.append(
            FoldResult(
                fold=fold,
                train_rows=len(train),
                test_rows=int(scored.sum()),
                train_markets=int(np.unique(train.market_id).size),
                test_markets=int(np.unique(test.market_id).size),
                train_clusters=int(np.unique(train.clusters).size),
                test_clusters=int(np.unique(test.clusters).size),
                brier_model=brier(predicted[scored], test.y[scored]),
                brier_baseline=brier(np.clip(test.price[scored], 0.0, 1.0), test.y[scored]),
                n_cells_used=len(used),
                n_cells_tradeable=sum(1 for f in fits if f.tradeable),
                split_time=split_time,
            )
        )
    return results


@dataclass(frozen=True)
class ReliabilityBin:
    """One reliability bin, reported at BOTH row and market level.

    The market-level view is the one to read. Hourly rows within a market are
    the same bet counted many times, so a row-level bin containing 2,900 rows
    may rest on only ~30 independent outcomes — and a row-level gap of +15pp
    then reflects a handful of markets, not a market-wide inefficiency.
    """

    bin_mid: float
    mean_price: float
    observed: float
    n_rows: int
    n_markets: int
    observed_market_level: float

    @property
    def gap(self) -> float:
        return self.observed - self.mean_price

    @property
    def gap_market_level(self) -> float:
        return self.observed_market_level - self.mean_price


def reliability_table(
    price: FloatArray,
    y: FloatArray,
    market_id: npt.NDArray[np.str_] | None = None,
    *,
    n_bins: int = 10,
) -> list[ReliabilityBin]:
    """Reliability bins with both row-level and market-level frequencies."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out: list[ReliabilityBin] = []
    for i in range(n_bins):
        low, high = edges[i], edges[i + 1]
        mask = (
            (price >= low) & (price < high) if i < n_bins - 1 else (price >= low) & (price <= high)
        )
        n = int(mask.sum())
        if n == 0:
            continue
        if market_id is None:
            n_markets, observed_market = n, float(np.mean(y[mask]))
        else:
            ids = market_id[mask]
            unique, first = np.unique(ids, return_index=True)
            n_markets = int(unique.size)
            # One outcome per market: the label is a market property, so
            # taking the first row's y per market is exact, not an average.
            observed_market = float(np.mean(y[mask][first]))
        out.append(
            ReliabilityBin(
                bin_mid=float((low + high) / 2),
                mean_price=float(np.mean(price[mask])),
                observed=float(np.mean(y[mask])),
                n_rows=n,
                n_markets=n_markets,
                observed_market_level=observed_market,
            )
        )
    return out
