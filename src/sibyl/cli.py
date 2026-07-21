"""`sibyl status` — resolve config, report venue eligibility, smoke-test reads."""

from __future__ import annotations

import argparse
import asyncio
import sys

from sibyl import __version__
from sibyl.config import Settings, resolve_eligibility
from sibyl.log import configure_logging, get_logger
from sibyl.venues.kalshi import KalshiClient
from sibyl.venues.normalise import Venue
from sibyl.venues.polymarket import PolymarketReadClient

log = get_logger(__name__)


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


def main() -> None:
    parser = argparse.ArgumentParser(prog="sibyl")
    parser.add_argument("command", choices=["status"], help="status: config + connectivity check")
    parser.add_argument("--json", action="store_true", help="JSON log output")
    args = parser.parse_args()
    configure_logging(json_output=args.json)
    if args.command == "status":
        sys.exit(asyncio.run(_status()))


if __name__ == "__main__":
    main()
