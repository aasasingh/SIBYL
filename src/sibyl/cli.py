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
    _cells, _mu, _tau, _folds, lines = build_report(rows, n_boot=args.n_boot, n_folds=args.n_folds)
    text = "\n".join(lines)
    sys.stdout.write(text + "\n")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
        log.info("calibrate.written", path=str(args.out))
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

    lag = sub.add_parser("lag", help="measured settlement lag (basis for capital buffers)")
    lag.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)

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
    rec.add_argument("--markets", type=int, default=60, help="most-liquid active markets to watch")
    rec.add_argument("--max-assets", type=int, default=150, help="cap on subscribed token ids")

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
    elif args.command == "eiv":
        sys.exit(_eiv(args))
    elif args.command == "record":
        sys.exit(asyncio.run(_record(args)))


if __name__ == "__main__":
    main()
