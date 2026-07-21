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

    args = parser.parse_args()
    configure_logging(json_output=args.json)
    if args.command == "status":
        sys.exit(asyncio.run(_status()))
    elif args.command == "backfill":
        sys.exit(asyncio.run(_backfill(args)))
    elif args.command == "store":
        sys.exit(_store_summary(args))


if __name__ == "__main__":
    main()
