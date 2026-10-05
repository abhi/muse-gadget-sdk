"""The companion link's boundary: URL and pin parsing, pinned TLS, and the handshake."""

import asyncio
import datetime
import hashlib
import logging
import ssl

import pytest

from fake_companion import TOKEN, FakeCompanion
from musegadget.reachy_capabilities import ReplyStyle
from musegadget.reachy_companion_client import CompanionLink, CompanionNarrator, LinkState, parse_pin, parse_url
from musegadget.reachy_companion_protocol import Narrate, Sender, decode

DIGEST = "4f" * 32


@pytest.mark.parametrize("url, insecure, expected", [
    ("wss://studio.local:8765", False, "wss://studio.local:8765"),
    ("wss://studio.local:8765/", False, "wss://studio.local:8765"),
    ("ws://127.0.0.1:9000", True, "ws://127.0.0.1:9000"),
])
def test_a_companion_url_is_its_base_address(url, insecure, expected):
    assert parse_url(url, insecure=insecure) == expected


@pytest.mark.parametrize("url", [
    "ws://studio.local:8765", "https://studio.local:8765", "wss://studio.local:8765/v1",
    "wss://user:pw@studio.local:8765", "wss://:8765", "studio.local:8765",
])
def test_a_companion_url_with_a_path_credentials_or_plain_ws_is_refused(url):
    with pytest.raises(ValueError, match="companion URL"):
        parse_url(url)


@pytest.mark.parametrize("pin, expected", [
    (f"sha256:{DIGEST}", DIGEST), (f"sha256:{DIGEST.upper()}", None), (DIGEST, None),
    (f"sha256:{DIGEST[:-2]}", None), (f"sha1:{DIGEST}", None),
])
def test_a_pin_is_sha256_and_64_lowercase_hex_digits(pin, expected):
    if expected is None:
        with pytest.raises(ValueError, match="pin"):
            parse_pin(pin)
    else:
        assert parse_pin(pin) == expected


def test_a_wss_companion_needs_a_pin():
    with pytest.raises(ValueError, match="pin"):
        CompanionLink("wss://studio.local:8765", token=TOKEN, out_rate=16000)


@pytest.fixture(scope="module")
def certificate(tmp_path_factory):
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "companion")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(now - datetime.timedelta(days=1))
            .not_valid_after(now + datetime.timedelta(days=1)).sign(key, hashes.SHA256()))
    directory = tmp_path_factory.mktemp("companion-cert")
    (directory / "cert.pem").write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    (directory / "key.pem").write_bytes(key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(directory / "cert.pem", directory / "key.pem")
    pin = "sha256:" + hashlib.sha256(cert.public_bytes(serialization.Encoding.DER)).hexdigest()
    return context, pin


async def first_outcome(companion, **options):
    link = CompanionLink(companion.url[:-3], token=TOKEN, out_rate=16000, backoff_s=(5, 5), **options)
    running = asyncio.ensure_future(link.run())
    try:
        await asyncio.wait_for(link.settled(), 5)
        return link.state, link.models
    finally:
        running.cancel()
        await asyncio.gather(running, return_exceptions=True)


def test_the_pinned_certificate_is_accepted_and_the_link_comes_up(certificate):
    context, pin = certificate

    async def run():
        async with FakeCompanion(ssl=context) as companion:
            return await first_outcome(companion, pin=pin), companion.hellos
    outcome, hellos = asyncio.run(run())
    assert outcome == (LinkState.UP, (("hear", "fake"), ("speak", "tone"), ("narrate", "echo")))
    assert hellos == [TOKEN]


def test_another_certificate_is_refused_before_the_token_is_sent(certificate):
    context, _ = certificate

    async def run():
        async with FakeCompanion(ssl=context) as companion:
            return await first_outcome(companion, pin=f"sha256:{DIGEST}"), companion.hellos
    assert asyncio.run(run()) == ((LinkState.DOWN, ()), [])


def test_a_refused_token_keeps_the_link_down_and_logs_the_companions_reason(caplog):
    async def run():
        async with FakeCompanion(token="rc1_another_robot") as companion:
            return await first_outcome(companion, insecure=True)
    with caplog.at_level(logging.WARNING):
        assert asyncio.run(run()) == (LinkState.DOWN, ())
    assert "the companion refused hello: auth: bad token" in caplog.text


def test_a_pin_for_a_plain_ws_companion_is_refused():
    with pytest.raises(ValueError, match="a pinned companion needs wss://"):
        CompanionLink("ws://127.0.0.1:9000", token=TOKEN, pin=f"sha256:{DIGEST}", insecure=True, out_rate=16000)


def narrated(call):
    """What the fake's echo narrator answers ``call``, and the narrate request Reachy sent it."""
    async def run():
        async with FakeCompanion() as companion:
            link = CompanionLink(companion.url[:-3], token=TOKEN, insecure=True, out_rate=16000, backoff_s=(5, 5))
            running = asyncio.ensure_future(link.run())
            try:
                await asyncio.wait_for(link.settled(), 5)
                said = await call(CompanionNarrator(link))
            finally:
                running.cancel()
                await asyncio.gather(running, return_exceptions=True)
            sent = [decode(frame, sender=Sender.ROBOT) for frame in companion.received if isinstance(frame, str)]
            return said, [message for message in sent if isinstance(message, Narrate)][-1]
    return asyncio.run(run())


def test_a_progress_label_longer_than_the_wire_allows_is_cut_at_a_word():
    status = "Earlier from Muse: " + "checking the castle archives " * 8
    said, sent = narrated(lambda narrator: narrator.say_progress("castles", status, ()))
    expected = ("Earlier from Muse: checking the castle archives checking the castle archives "
                "checking the castle archives checking the")
    assert (sent.status, said.text) == (expected, expected)


def test_a_reply_longer_than_the_wire_allows_is_cut_after_its_last_whole_sentence():
    reply = " ".join(f"Line {n:03d} is here." for n in range(600))
    said, sent = narrated(lambda narrator: narrator.lines("castles", reply, ReplyStyle.PLAIN_SHORT))
    assert (len(sent.reply), sent.reply[-17:]) == (8189, "Line 454 is here.")
    assert [line.text for line in said] == ["Line 000 is here.", "Line 001 is here.", "Line 002 is here."]
