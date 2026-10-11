"""Human-readable calibration report (spec §6 Phase 3: "honest report on
which cells have real signal and which are noise")."""

from __future__ import annotations

import numpy as np

from sibyl.calibration.fit import CellFit, fit_cell
from sibyl.calibration.hierarchical import partial_pool, population_slope_ci
from sibyl.calibration.walkforward import FoldResult, Rows, reliability_table, walk_forward
from sibyl.store.resolutions import HORIZON_BUCKETS

_HORIZON_ORDER = {name: i for i, (name, _, _) in enumerate(HORIZON_BUCKETS)}


def _sort_key(cell: CellFit) -> tuple[str, int]:
    return (cell.domain, _HORIZON_ORDER.get(cell.horizon, 99))


def build_report(
    rows: Rows, *, n_boot: int = 400, n_folds: int = 4
) -> tuple[list[CellFit], float, float, list[FoldResult], list[str]]:
    """Fit all cells, pool, walk forward. Returns (cells, mu, tau, folds, lines)."""
    cells = [fit_cell(cell, n_boot=n_boot) for cell in rows.cells()]
    pooled, mu, tau = partial_pool(cells)
    mu_ci = population_slope_ci(cells)
    pooled.sort(key=_sort_key)
    folds = walk_forward(rows, n_folds=n_folds, n_boot=max(100, n_boot // 2))
    return pooled, mu, tau, folds, render(rows, pooled, mu, tau, folds, mu_ci)


def render(
    rows: Rows,
    cells: list[CellFit],
    mu: float,
    tau: float,
    folds: list[FoldResult],
    mu_ci: tuple[float, float] | None = None,
) -> list[str]:
    n_markets = int(np.unique(rows.market_id).size)
    mu_text = f"mu={mu:.4f}"
    if mu_ci is not None:
        excludes_one = mu_ci[0] > 1.0 or mu_ci[1] < 1.0
        mu_text += f" CI [{mu_ci[0]:.4f}, {mu_ci[1]:.4f}]"
        mu_text += " (excludes 1)" if excludes_one else " (STRADDLES 1 — no global edge)"
    lines: list[str] = [
        "=" * 100,
        "SIBYL CALIBRATION REPORT",
        "=" * 100,
        f"rows={len(rows)}  markets={n_markets}  cells={len(cells)}",
        f"population slope {mu_text}  between-cell sd tau={tau:.4f}",
        "",
        "PER-CELL FITS  (b>1 = prices compressed toward 0.5; CI is market-clustered)",
        "-" * 100,
    ]
    lines.extend("  " + c.describe() for c in cells)

    tradeable = [c for c in cells if c.tradeable]
    tested = sum(1 for c in cells if c.slope_ci is not None)
    expected_false = 0.05 * tested
    lines += [
        "",
        f"TRADEABLE CELLS: {len(tradeable)} of {len(cells)}",
        "  Requires all three: the clustered CI excludes b=1; the slope actually traded is",
        f"  still this cell's own (shrinkage <= {CellFit.MAX_SHRINKAGE_FOR_CELL_SIGNAL:.2f});"
        " and pooling did not move it across 1.0.",
        f"  MULTIPLE TESTING: {tested} cells carry an interval, so at 95% confidence about"
        f" {expected_false:.1f} would clear the bar by chance alone.",
        "-" * 100,
    ]
    if tradeable:
        for cell in sorted(tradeable, key=lambda c: -abs(c.effective_slope - 1.0)):
            direction = (
                "prices COMPRESSED (favourites underpriced)"
                if cell.effective_slope > 1
                else "prices OVER-EXTENDED (fade the extremes)"
            )
            shrink = f" shrink={cell.shrinkage:.2f}" if cell.shrinkage is not None else ""
            lines.append(
                f"  {cell.domain}/{cell.horizon}: b={cell.effective_slope:.3f}{shrink}"
                f" -> {direction}"
            )
    else:
        lines.append("  none. No cell shows a demonstrated cell-specific edge on this data.")

    gated = [(c, c.gating_note) for c in cells if not c.tradeable and c.gating_note]
    shrunk_out = [c for c, note in gated if note.startswith("shrinkage")]
    if shrunk_out:
        lines += [
            "",
            f"  REJECTED FOR BORROWED SIGNAL ({len(shrunk_out)}): interval excluded 1, but the",
            "  slope these would trade is the population mean, not their own data. Judge",
            "  these on the population CI above, not per-cell.",
        ]
        lines.extend(
            f"    {c.domain}/{c.horizon}: own b={c.slope:.3f} -> traded "
            f"{c.effective_slope:.3f} (shrink {c.shrinkage:.2f})"
            for c in shrunk_out
            if c.shrinkage is not None
        )

    lines += ["", "WALK-FORWARD OUT-OF-SAMPLE (expanding window, split on market close)", "-" * 100]
    if not folds:
        lines.append("  not enough markets spread over time to form folds")
    else:
        lines.append(
            f"  {'fold':<5} {'train_mkts':<11} {'test_mkts':<10} "
            f"{'tr_evts':<8} {'te_evts':<8} "
            f"{'brier_model':<12} {'brier_raw':<11} {'improvement':<12}"
        )
        for f in folds:
            lines.append(
                f"  {f.fold:<5} {f.train_markets:<11} {f.test_markets:<10} "
                f"{f.train_clusters:<8} {f.test_clusters:<8} "
                f"{f.brier_model:<12.6f} {f.brier_baseline:<11.6f} {f.improvement:<+12.6f}"
            )
        improvements = [f.improvement for f in folds]
        mean_improvement = float(np.mean(improvements))
        wins = sum(1 for i in improvements if i > 0)
        lines += [
            "",
            f"  mean improvement over raw price: {mean_improvement:+.6f} "
            f"({wins}/{len(folds)} folds positive)",
            "  VERDICT: "
            + (
                "recalibration beats the raw price out of sample"
                if mean_improvement > 0 and wins > len(folds) / 2
                else "NO out-of-sample improvement — the raw price is not reliably beaten"
            ),
        ]

    lines += [
        "",
        "RELIABILITY (raw quoted price vs realised frequency)",
        "Read the MARKET-level columns: rows within a market are one bet counted many times.",
        "-" * 100,
        f"  {'bin':<6} {'mean_price':<11} {'obs_row':<9} {'gap_row':<9} "
        f"{'obs_mkt':<9} {'gap_mkt':<9} {'rows':<8} {'markets':<8}",
    ]
    for b in reliability_table(rows.price, rows.y, rows.market_id):
        lines.append(
            f"  {b.bin_mid:<6.2f} {b.mean_price:<11.4f} {b.observed:<9.4f} {b.gap:<+9.4f} "
            f"{b.observed_market_level:<9.4f} {b.gap_market_level:<+9.4f} "
            f"{b.n_rows:<8} {b.n_markets:<8}"
        )
    lines.append("=" * 100)
    return lines
