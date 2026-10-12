"""sibyl CLI: status / backfill / store."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from sibyl import __version__
from sibyl.config import Settings, resolve_eligibility
from sibyl.log import configure_logging, get_logger
from sibyl.store.backfill import backfill_kalshi, backfill_polymarket
from sibyl.store.ledger import Ledger
from sibyl.store.resolutions import resolved_market_counts
from sibyl.venues.kalshi import KalshiClient
from sibyl.venues.normalise import Venue
from sibyl.venues.polymarket import PolymarketReadClient

log = get_logger(__name__)

DEFAULT_DATA_DIR = Path("data")


async def _status() -> int:
    settings = Settings()
    eligibility = resolve_eligibility(settings)
    log.info(
        "sibyl.startup",
        version=__version__,
        live=settings.live,
        kalshi_env=settings.kalshi.env,
        polymarket_platform=settings.polymarket.platform,
        chain_id=settings.polymarket.chain_id,
    )
    for venue in Venue:
        log.info(
            "venue.eligibility",
            venue=venue.value,
            enabled=venue in eligibility.enabled,
            reason=eligibility.reasons.get(venue, "unknown"),
        )

    exit_code = 0
    if Venue.KALSHI in eligibility.enabled:
        try:
            async with KalshiClient(demo=settings.kalshi.env == "demo") as kalshi:
                status = await kalshi.exchange_status()
                log.info("kalshi.exchange_status", **status)
        except Exception as exc:
            log.error("kalshi.unreachable", error=str(exc))
            exit_code = 1

    poly_venue = settings.polymarket.venue
    if poly_venue in eligibility.enabled:
        try:
            async with PolymarketReadClient(poly_venue) as poly:
                markets = await poly.markets(limit=1, active=True)
                log.info("polymarket.reachable", sample_markets=len(markets))
        except Exception as exc:
            log.error("polymarket.unreachable", error=str(exc))
            exit_code = 1
    return exit_code


async def _backfill(args: argparse.Namespace) -> int:
    settings = Settings()
    ledger = Ledger(args.data_dir)
    if args.venue in ("kalshi", "all"):
        async with KalshiClient(demo=settings.kalshi.env == "demo") as kalshi:
            result = await backfill_kalshi(
                kalshi,
                ledger,
                months=args.months,
                max_pages=args.max_pages,
                candle_markets=args.price_markets,
            )
        log.info("backfill.kalshi.done", **result)
    if args.venue in ("polymarket", "all"):
        async with PolymarketReadClient(settings.polymarket.venue) as poly:
            result = await backfill_polymarket(
                poly,
                ledger,
                months=args.months,
                max_pages=args.max_pages,
                tape_markets=args.price_markets,
            )
        log.info("backfill.polymarket.done", **result)
    return 0


def _calibrate(args: argparse.Namespace) -> int:
    from sibyl.calibration.dataset import load_rows
    from sibyl.calibration.report import build_report

    ledger = Ledger(args.data_dir)
    rows = load_rows(
        ledger,
        max_horizon_days=args.max_horizon_days,
        exclude_domains=tuple(args.exclude_domains or ()),
    )
    if len(rows) == 0:
        log.error("calibrate.no_data", hint="run `sibyl backfill --price-markets N` first")
        return 1
    cells, mu, tau, folds, lines = build_report(rows, n_boot=args.n_boot, n_folds=args.n_folds)
    text = "\n".join(lines)
    sys.stdout.write(text + "\n")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
        log.info("calibrate.written", path=str(args.out))
    if args.json_out is not None:
        import json

        from sibyl.calibration.hierarchical import population_slope_ci
        from sibyl.calibration.report import to_json

        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = to_json(rows, cells, mu, tau, folds, population_slope_ci(cells))
        args.json_out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        log.info("calibrate.json_written", path=str(args.json_out))
    return 0


def _eiv(args: argparse.Namespace) -> int:
    """Report measurement-error attenuation per cell (see calibration/eiv.py)."""
    import numpy as np

    from sibyl.calibration.dataset import load_rows
    from sibyl.calibration.eiv import attenuation_for_cell, estimate_noise
    from sibyl.calibration.fit import fit_logistic, logit

    rows = load_rows(Ledger(args.data_dir))
    if len(rows) == 0:
        log.error("eiv.no_data", hint="run `sibyl backfill --price-markets N` first")
        return 1
    pooled = estimate_noise(rows.price, rows.market_id)
    if pooled is not None:
        log.info(
            "eiv.pooled",
            noise_sd=round(float(np.sqrt(pooled.sigma2_noise)), 4),
            reliability=round(pooled.reliability, 4),
            noise_share=round(pooled.noise_share, 4),
            var_observed=round(pooled.var_observed, 3),
            r2=round(pooled.r2, 3),
        )
    crossers = 0
    for cell in rows.cells():
        if cell.n_rows < args.min_rows or cell.n_clusters < 10:
            continue
        _ic, observed, converged = fit_logistic(logit(cell.price), cell.y)
        if not converged or not np.isfinite(observed):
            continue
        result = attenuation_for_cell(cell, observed)
        if result is None:
            continue
        crossers += bool(result.crosses_one)
        log.info(
            "eiv.cell",
            cell=result.cell,
            n=result.n_rows,
            b_observed=round(result.observed_slope, 3),
            reliability=round(result.reliability, 3),
            b_corrected=round(result.corrected_slope, 3),
            noise_sd=round(result.noise_sd, 3),
            crosses_one=result.crosses_one,
        )
    log.info("eiv.summary", cells_crossing_one=crossers)
    return 0


def _power(args: argparse.Namespace) -> int:
    """Multi-seed power curve for the walk-forward test."""
    import numpy as np

    from sibyl.calibration.dataset import load_rows
    from sibyl.calibration.eiv import estimate_noise
    from sibyl.calibration.power import (
        expected_noise_floor,
        power_curve,
        smallest_detectable_slope,
    )

    noise_sd = args.noise_sd
    if noise_sd is None:
        rows = load_rows(Ledger(args.data_dir))
        measured = estimate_noise(rows.price, rows.market_id) if len(rows) else None
        noise_sd = float(np.sqrt(measured.sigma2_noise)) if measured else 0.0
        log.info("power.noise_from_data", noise_sd=round(noise_sd, 4))

    lines = [
        "=" * 96,
        f"WALK-FORWARD POWER  ({args.seeds} seeds/condition, {args.folds} folds, "
        f"noise sd {noise_sd:.3f})",
        "Decision rule is the report's own: mean improvement > 0 AND >half the folds positive.",
        "=" * 96,
    ]
    results = power_curve(
        noise_sd=noise_sd, n_seeds=args.seeds, n_folds=args.folds, n_boot=args.boot
    )
    lines.extend("  " + r.describe() for r in results)
    floor = expected_noise_floor(results)
    smallest = smallest_detectable_slope(results, threshold=args.threshold)
    lines += [
        "",
        f"  null-condition mean improvement (noise floor): {floor:+.6f}"
        if floor is not None
        else "  null condition not tested",
        f"  smallest slope detected at >= {args.threshold:.0%} power: "
        + (f"b = {smallest:.2f}" if smallest is not None else "NONE of the tested slopes"),
        "=" * 96,
    ]
    text = "\n".join(lines)
    sys.stdout.write(text + "\n")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
        log.info("power.written", path=str(args.out))
    if args.json_out is not None:
        import json

        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "n_seeds": args.seeds,
            "n_folds": args.folds,
            "noise_sd": noise_sd,
            "threshold": args.threshold,
            "smallest_detectable": smallest,
            "noise_floor": floor,
            "conditions": [
                {
                    "true_slope": r.true_slope,
                    "noise_sd": r.noise_sd,
                    "n_seeds": r.n_seeds,
                    "detections": r.detections,
                    "detection_rate": r.detection_rate,
                    "mean_fitted_slope": r.mean_fitted_slope,
                    "mean_improvement": r.mean_improvement,
                    "improvement_sd": r.improvement_sd,
                    "is_null": r.is_null,
                }
                for r in results
            ],
        }
        args.json_out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        log.info("power.json_written", path=str(args.json_out))
    return 0


def _tables(args: argparse.Namespace) -> int:
    """Inject generated tables into the write-up. No figure is hand-typed."""
    from sibyl.calibration.tables import inject, load_artifact

    artifacts = {}
    if args.calibration.exists():
        artifacts["calibration"] = load_artifact(args.calibration)
    if args.power.exists():
        artifacts["power"] = load_artifact(args.power)
    if not artifacts:
        log.error(
            "tables.no_artifacts",
            hint="run `sibyl calibrate --json-out` and `sibyl power --json-out` first",
        )
        return 1
    updated = inject(args.findings, artifacts)
    log.info("tables.injected", document=str(args.findings), blocks=updated)
    return 0


def _lag(args: argparse.Namespace) -> int:
    from sibyl.store.resolutions import early_resolution_rate, resolution_lag

    with Ledger(args.data_dir).connect() as con:
        for venue, n, p50, p90, p99, worst in resolution_lag(con):
            log.info(
                "store.resolution_lag_days",
                venue=venue,
                n=n,
                median=round(p50, 3),
                p90=round(p90, 3),
                p99=round(p99, 3),
                max=round(worst, 2),
            )
        for venue, n, early in early_resolution_rate(con):
            log.info("store.early_resolution", venue=venue, n=n, fraction=round(early, 4))
    return 0


async def _record(args: argparse.Namespace) -> int:
    from sibyl.record.recorder import record

    await record(
        data_dir=args.data_dir,
        hours=args.hours,
        n_markets=args.markets,
        max_assets=args.max_assets,
        refresh_hours=args.refresh_hours,
    )
    return 0


def _coverage(args: argparse.Namespace) -> int:
    """Where was book state unknown? The fill model must not cross these."""
    from sibyl.record.coverage import coverage_summary, find_gaps

    summary = coverage_summary(args.data_dir)
    log.info("coverage.summary", **summary)
    gaps = find_gaps(args.data_dir, min_seconds=args.min_seconds)
    for start, end, seconds in gaps[: args.limit]:
        log.info(
            "coverage.gap",
            start=start.isoformat(),
            end=end.isoformat(),
            seconds=round(seconds, 2),
        )
    if len(gaps) > args.limit:
        log.info("coverage.gaps_truncated", shown=args.limit, total=len(gaps))
    return 0


def _replay(args: argparse.Namespace) -> int:
    """Rebuild every recorded book and check it against the venue's own claims."""
    from sibyl.fill.replay import replay

    stats = replay(args.data_dir, silence_s=args.silence_s, max_examples=args.examples)
    log.info("replay.summary", **stats.summary())
    for m in stats.examples:
        log.info(
            "replay.mismatch",
            kind=m.kind,
            recv_ts=m.recv_ts.isoformat(),
            asset=m.asset_id[:12],
            detail=m.detail,
        )
    return 0


def _attribute(args: argparse.Namespace) -> int:
    """Split every recorded size decrease into traded vs cancelled volume."""
    from sibyl.fill.replay import attribute_recorded

    _, stats, _ = attribute_recorded(
        args.data_dir, window_ms=args.window_ms, fallback_ms=args.fallback_ms
    )
    log.info(
        "attribute.summary",
        window_ms=args.window_ms,
        fallback_ms=args.fallback_ms,
        **stats.summary(),
    )
    return 0


def _probe(args: argparse.Namespace) -> int:
    """Run touch-joining probe orders through every fill-rule variant."""
    from datetime import datetime

    from sibyl.fill.queue import summarise
    from sibyl.fill.replay import probe_orders, simulate_recorded, venue_time_range

    start = datetime.fromisoformat(args.start) if args.start else None
    span = venue_time_range(args.data_dir, start=start)
    if span is None:
        log.error("probe.no_data")
        return 1
    orders = probe_orders(
        args.data_dir,
        start_ms=span[0],
        end_ms=span[1],
        every_s=args.every_s,
        life_s=args.life_s,
    )
    results = simulate_recorded(args.data_dir, orders, latency_ms=args.latency_ms, start=start)
    for name, states in results.items():
        log.info(
            "probe.summary", variant=name, latency_ms=args.latency_ms, **summarise(states).as_dict()
        )
    return 0


def _compact(args: argparse.Namespace) -> int:
    merged = Ledger(args.data_dir).compact(min_parts=args.min_parts)
    log.info("store.compacted", **merged)
    return 0


def _store_summary(args: argparse.Namespace) -> int:
    ledger = Ledger(args.data_dir)
    for venue, domain, resolution, n_markets, n_prices in ledger.summary():
        log.info(
            "store.markets",
            venue=venue,
            domain=domain,
            resolution=resolution,
            markets=n_markets,
            price_rows=n_prices,
        )
    with ledger.connect() as con:
        for venue, domain, n_resolved, n_priced in resolved_market_counts(con):
            log.info(
                "store.resolved",
                venue=venue,
                domain=domain,
                resolved=n_resolved,
                with_prices=n_priced,
            )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="sibyl")
    parser.add_argument("--json", action="store_true", help="JSON log output")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="config + eligibility + venue connectivity")

    backfill = sub.add_parser("backfill", help="pull resolved markets + prices into the ledger")
    backfill.add_argument("--venue", choices=["kalshi", "polymarket", "all"], default="all")
    backfill.add_argument("--months", type=int, default=6)
    backfill.add_argument("--max-pages", type=int, default=20, help="market-list pages per venue")
    backfill.add_argument(
        "--price-markets",
        type=int,
        default=0,
        help="how many markets to fetch price paths for (0 = markets only)",
    )
    backfill.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)

    store = sub.add_parser("store", help="summarise the ledger")
    store.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)

    compact = sub.add_parser("compact", help="merge Parquet parts into one file per table")
    compact.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    compact.add_argument("--min-parts", type=int, default=50)

    calibrate = sub.add_parser("calibrate", help="fit recalibration slopes and report honestly")
    calibrate.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    calibrate.add_argument("--n-boot", type=int, default=400, help="clustered bootstrap draws")
    calibrate.add_argument("--n-folds", type=int, default=4, help="walk-forward folds")
    calibrate.add_argument("--max-horizon-days", type=float, default=400.0)
    calibrate.add_argument("--exclude-domains", nargs="*", default=[])
    calibrate.add_argument("--out", type=Path, default=None, help="also write the report here")
    calibrate.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="structured artifact for `sibyl tables` (keeps write-up figures generated)",
    )

    lag = sub.add_parser("lag", help="measured settlement lag (basis for capital buffers)")
    lag.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)

    power = sub.add_parser(
        "power", help="multi-seed power curve: what slope could this test have detected?"
    )
    power.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    power.add_argument("--seeds", type=int, default=60, help="simulations per condition")
    power.add_argument("--folds", type=int, default=4)
    power.add_argument("--boot", type=int, default=40)
    power.add_argument("--threshold", type=float, default=0.80, help="power target")
    power.add_argument(
        "--noise-sd",
        type=float,
        default=None,
        help="logit-space measurement noise (default: measured from the ledger)",
    )
    power.add_argument("--out", type=Path, default=None)
    power.add_argument("--json-out", type=Path, default=None)

    eiv = sub.add_parser("eiv", help="measurement-error attenuation: is the tape hiding a slope?")
    eiv.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    eiv.add_argument("--min-rows", type=int, default=200)

    rec = sub.add_parser(
        "record", help="record live books+trades over WebSocket (input to the fill model)"
    )
    rec.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    rec.add_argument(
        "--hours", type=float, default=None, help="stop after N hours (default: run until killed)"
    )
    rec.add_argument(
        "--markets", type=int, default=60, help="most-traded (24h) mid-priced markets to watch"
    )
    rec.add_argument("--max-assets", type=int, default=150, help="cap on subscribed token ids")
    rec.add_argument(
        "--refresh-hours",
        type=float,
        default=6.0,
        help="re-select the traded market set this often (resolved markets drop out)",
    )

    tbl = sub.add_parser(
        "tables", help="regenerate the write-up's numeric tables from run artifacts"
    )
    tbl.add_argument("--findings", type=Path, default=Path("research/phase3-findings.md"))
    tbl.add_argument("--calibration", type=Path, default=Path("research/calibration.json"))
    tbl.add_argument("--power", type=Path, default=Path("research/power.json"))

    cov = sub.add_parser("coverage", help="recorded-coverage segments and the gaps between them")
    cov.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    cov.add_argument("--min-seconds", type=float, default=1.0, help="ignore gaps below this")
    cov.add_argument("--limit", type=int, default=25)

    rpl = sub.add_parser(
        "replay", help="rebuild recorded order books and validate them against the venue"
    )
    rpl.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    rpl.add_argument(
        "--silence-s",
        type=float,
        default=20.0,
        help="treat a stream silence this long as a disconnect (state reset)",
    )
    rpl.add_argument("--examples", type=int, default=20, help="mismatches to print")

    att = sub.add_parser("attribute", help="classify recorded book decreases as trades or cancels")
    att.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    att.add_argument("--window-ms", type=int, default=500, help="primary trade-match window")
    att.add_argument(
        "--fallback-ms", type=int, default=1500, help="wider window for leftovers (low confidence)"
    )

    prb = sub.add_parser(
        "probe", help="simulate touch-joining probe orders under every fill-rule variant"
    )
    prb.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    prb.add_argument("--start", default=None, help="ISO time; only replay data received after it")
    prb.add_argument("--every-s", type=float, default=300.0, help="probe cadence per book side")
    prb.add_argument("--life-s", type=float, default=300.0, help="how long each probe rests")
    prb.add_argument(
        "--latency-ms", type=int, default=250, help="assumed order latency (place and cancel)"
    )

    args = parser.parse_args()
    configure_logging(json_output=args.json)
    if args.command == "status":
        sys.exit(asyncio.run(_status()))
    elif args.command == "backfill":
        sys.exit(asyncio.run(_backfill(args)))
    elif args.command == "store":
        sys.exit(_store_summary(args))
    elif args.command == "compact":
        sys.exit(_compact(args))
    elif args.command == "calibrate":
        sys.exit(_calibrate(args))
    elif args.command == "lag":
        sys.exit(_lag(args))
    elif args.command == "power":
        sys.exit(_power(args))
    elif args.command == "eiv":
        sys.exit(_eiv(args))
    elif args.command == "tables":
        sys.exit(_tables(args))
    elif args.command == "coverage":
        sys.exit(_coverage(args))
    elif args.command == "probe":
        sys.exit(_probe(args))
    elif args.command == "attribute":
        sys.exit(_attribute(args))
    elif args.command == "replay":
        sys.exit(_replay(args))
    elif args.command == "record":
        sys.exit(asyncio.run(_record(args)))


if __name__ == "__main__":
    main()
