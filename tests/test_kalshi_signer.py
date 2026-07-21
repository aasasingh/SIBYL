"""RSA-PSS signer tests: exact message construction + verifiable signatures."""

from __future__ import annotations

import base64

import pytest
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from sibyl.venues.kalshi import sign_request, signing_message


@pytest.fixture(scope="module")
def private_key() -> rsa.RSAPrivateKey:
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


class TestSigningMessage:
    def test_message_layout(self) -> None:
        msg = signing_message(1_721_000_000_000, "GET", "/trade-api/v2/portfolio/balance")
        assert msg == b"1721000000000GET/trade-api/v2/portfolio/balance"

    def test_query_string_excluded(self) -> None:
        with_query = signing_message(1, "GET", "/trade-api/v2/markets?limit=5&cursor=abc")
        without = signing_message(1, "GET", "/trade-api/v2/markets")
        assert with_query == without

    def test_method_uppercased(self) -> None:
        assert signing_message(1, "get", "/trade-api/v2/markets") == signing_message(
            1, "GET", "/trade-api/v2/markets"
        )

    def test_prefix_required(self) -> None:
        with pytest.raises(ValueError, match="prefix"):
            signing_message(1, "GET", "/markets")


class TestSignature:
    def test_roundtrip_verifies_with_pss_sha256_salt32(
        self, private_key: rsa.RSAPrivateKey
    ) -> None:
        ts, method, path = 1_721_000_000_000, "GET", "/trade-api/v2/markets"
        sig_b64 = sign_request(private_key, ts, method, path)
        private_key.public_key().verify(
            base64.b64decode(sig_b64),
            signing_message(ts, method, path),
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=32),
            hashes.SHA256(),
        )  # raises InvalidSignature on failure

    def test_signature_bound_to_timestamp(self, private_key: rsa.RSAPrivateKey) -> None:
        sig_b64 = sign_request(private_key, 1_721_000_000_000, "GET", "/trade-api/v2/markets")
        with pytest.raises(InvalidSignature):
            private_key.public_key().verify(
                base64.b64decode(sig_b64),
                signing_message(1_721_000_000_001, "GET", "/trade-api/v2/markets"),
                padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=32),
                hashes.SHA256(),
            )

    def test_signature_is_base64(self, private_key: rsa.RSAPrivateKey) -> None:
        sig = sign_request(private_key, 1, "GET", "/trade-api/v2/markets")
        assert base64.b64encode(base64.b64decode(sig)).decode() == sig
