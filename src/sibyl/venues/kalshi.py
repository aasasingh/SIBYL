"""Kalshi client: hand-rolled RSA-PSS request signer + async REST reads.

Auth (spec §2.1): sign `timestamp_ms + METHOD + path`, where path INCLUDES the
`/trade-api/v2` prefix and EXCLUDES the query string. RSA-PSS with SHA-256 for
both digest and MGF1, salt length = 32 bytes (digest length), base64-encoded.
"""

from __future__ import annotations

import base64
import time
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import httpx
from aiolimiter import AsyncLimiter
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from sibyl.venues.normalise import BinaryBook, normalise_kalshi_book

DEMO_BASE = "https://demo-api.kalshi.co"
PROD_BASE = "https://trading-api.kalshi.com"
API_PREFIX = "/trade-api/v2"

_SALT_LENGTH = 32  # = SHA-256 digest length, per Kalshi's spec


def load_private_key(pem_path: Path) -> rsa.RSAPrivateKey:
    """Load the RSA private key Kalshi generated for the API key pair.

    The key is NOT recoverable from Kalshi — losing it means regenerating the
    pair. It must live outside the repo (see .gitignore).
    """
    key = serialization.load_pem_private_key(pem_path.read_bytes(), password=None)
    if not isinstance(key, rsa.RSAPrivateKey):
        raise TypeError(f"expected an RSA private key, got {type(key).__name__}")
    return key


def signing_message(timestamp_ms: int, method: str, path: str) -> bytes:
    """Build the exact message Kalshi verifies.

    `path` must include the API prefix; any query string is stripped here so a
    caller passing a full request path cannot silently break the signature.
    """
    path_no_query = path.split("?", 1)[0]
    if not path_no_query.startswith(API_PREFIX):
        raise ValueError(f"path must include the {API_PREFIX!r} prefix, got {path!r}")
    return f"{timestamp_ms}{method.upper()}{path_no_query}".encode()


def sign_request(private_key: rsa.RSAPrivateKey, timestamp_ms: int, method: str, path: str) -> str:
    """RSA-PSS signature over the Kalshi signing message, base64-encoded."""
    signature = private_key.sign(
        signing_message(timestamp_ms, method, path),
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=_SALT_LENGTH),
        hashes.SHA256(),
    )
    return base64.b64encode(signature).decode("ascii")


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    return isinstance(exc, httpx.TransportError)


class KalshiClient:
    """Async REST client for Kalshi (demo by default — spec §3.1).

    Public reads work unauthenticated; pass `api_key_id` + `private_key` to
    sign requests. Rate limits per spec: ~10 rps authenticated, ~30 rps public.
    """

    def __init__(
        self,
        *,
        demo: bool = True,
        api_key_id: str | None = None,
        private_key: rsa.RSAPrivateKey | None = None,
        clock_skew_ms: int = 0,
    ) -> None:
        if (api_key_id is None) != (private_key is None):
            raise ValueError("api_key_id and private_key must be provided together")
        self._base = DEMO_BASE if demo else PROD_BASE
        self.demo = demo
        self._api_key_id = api_key_id
        self._private_key = private_key
        self._clock_skew_ms = clock_skew_ms
        rps = 10 if self.authenticated else 30
        self._limiter = AsyncLimiter(max_rate=rps, time_period=1)
        self._http = httpx.AsyncClient(base_url=self._base, http2=True, timeout=15.0)

    @property
    def authenticated(self) -> bool:
        return self._private_key is not None

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self._http.aclose()

    def _auth_headers(self, method: str, path: str) -> dict[str, str]:
        if self._private_key is None or self._api_key_id is None:
            return {}
        timestamp_ms = int(time.time() * 1000) + self._clock_skew_ms
        return {
            "KALSHI-ACCESS-KEY": self._api_key_id,
            "KALSHI-ACCESS-TIMESTAMP": str(timestamp_ms),
            "KALSHI-ACCESS-SIGNATURE": sign_request(self._private_key, timestamp_ms, method, path),
        }

    @retry(
        retry=retry_if_exception(_is_retryable),
        wait=wait_exponential_jitter(initial=1, max=30),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    async def _get(self, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        path = f"{API_PREFIX}{endpoint}"
        async with self._limiter:
            response = await self._http.get(
                path, params=params, headers=self._auth_headers("GET", path)
            )
        response.raise_for_status()
        payload = response.json(parse_float=str)  # never let prices become floats
        if not isinstance(payload, dict):
            raise TypeError(f"expected JSON object from {endpoint}, got {type(payload).__name__}")
        return payload

    async def exchange_status(self) -> dict[str, Any]:
        return await self._get("/exchange/status")

    async def markets(self, **params: Any) -> dict[str, Any]:
        return await self._get("/markets", params=params or None)

    async def market(self, ticker: str) -> dict[str, Any]:
        return await self._get(f"/markets/{ticker}")

    async def events(self, **params: Any) -> dict[str, Any]:
        """GET /events — supports status, limit, cursor, with_nested_markets.
        Events carry the category (domain source) and series_ticker that
        markets themselves lack."""
        return await self._get("/events", params=params or None)

    async def candlesticks(
        self,
        series_ticker: str,
        ticker: str,
        *,
        start_ts: int,
        end_ts: int,
        period_interval: int = 60,
    ) -> dict[str, Any]:
        """GET candlesticks for one market. `period_interval` in minutes
        (1 / 60 / 1440). Dollar fields arrive as strings (fp migration)."""
        return await self._get(
            f"/series/{series_ticker}/markets/{ticker}/candlesticks",
            params={"start_ts": start_ts, "end_ts": end_ts, "period_interval": period_interval},
        )

    async def balance(self) -> dict[str, Any]:
        if not self.authenticated:
            raise RuntimeError("balance requires authentication")
        return await self._get("/portfolio/balance")

    async def book(self, ticker: str) -> BinaryBook:
        """Fetch and normalise the bids-only orderbook for one market."""
        payload = await self._get(f"/markets/{ticker}/orderbook")
        raw_book = payload.get("orderbook_fp") or payload.get("orderbook")
        if not isinstance(raw_book, dict):
            raise TypeError(f"no orderbook in response for {ticker}: {list(payload)}")
        return normalise_kalshi_book(raw_book, market_id=ticker, ts=datetime.now(UTC))
