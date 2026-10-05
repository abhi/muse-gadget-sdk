# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Reachy's side of the companion link, and the capabilities a companion answers.

Wire types stay in this module. Callers see Endpoint, Partial and SpokenLine, and every
failure is a CapabilityUnavailable that the failover wrappers turn into on-robot work.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import inspect
import io
import logging
import random
import re
import ssl
import time
import wave
from collections import deque
from enum import Enum
from typing import AsyncIterator, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

from musegadget.reachy_capabilities import (
    CapabilityUnavailable, Endpoint, HeardAudio, Partial, ReplyStyle, SpokenLine,
)
from musegadget.reachy_companion_protocol import (
    MAX_ALREADY_SAID, MAX_LINE_BYTES, MAX_MIC_PCM_BYTES, MAX_OUTSTANDING_NARRATIONS, MAX_REPLY_BYTES,
    MAX_REQUEST_BYTES, MAX_STATUS_BYTES, MAX_TEXT_FRAME_BYTES, VERSION, AudioKind, CloseHow, Error,
    ErrorCode, HearClose, HearEndpoint, HearOpen, HearPartial, Hello, Narrate, NarratedLine, NarrateOp,
    NarrateResult, Op, ProtocolError, Reason, Sender, SequenceCheck, SpeakCancel, SpeakDone, SpeakStart,
    VoiceRole, Welcome, decode, decode_audio, encode, encode_audio,
)

log = logging.getLogger(__name__)

PING_INTERVAL_S = 2.0
PING_TIMEOUT_S = 3.0
HANDSHAKE_TIMEOUT_S = 3.0
BACKOFF_S = (1.0, 30.0)
HEALTHY_S = 10.0                   # a session shorter than this is a failed attempt, not a recovery
ENDPOINT_WAIT_S = 3.0              # how long the companion has to end a stream the robot finished
PARTIAL_GAP_BYTES = 3 * 16000 * 2  # microphone audio sent with no partial back before the companion counts as deaf
_PIN = re.compile(r"sha256:([0-9a-f]{64})")
_FATAL = (ErrorCode.AUTH, ErrorCode.VERSION, ErrorCode.INTERNAL, ErrorCode.MODEL)


def parse_pin(pin: str) -> str:
    """The certificate digest from a pairing pin: "sha256:" and 64 lowercase hex digits."""
    match = _PIN.fullmatch(pin)
    if match is None:
        raise ValueError("the companion pin must be sha256: followed by 64 lowercase hex digits")
    return match[1]


def parse_url(url: str, *, insecure: bool = False) -> str:
    """The companion's base URL as `companion pair` prints it: wss://HOST:PORT, no path."""
    parts = urlsplit(url)
    if parts.scheme not in ("wss", "ws") or not parts.hostname or parts.path not in ("", "/") \
            or parts.query or parts.fragment or parts.username or parts.password:
        raise ValueError("the companion URL must look like wss://HOST:PORT")
    if parts.scheme == "ws" and not insecure:
        raise ValueError("the companion URL must use wss://; plain ws:// is for tests only")
    return f"{parts.scheme}://{parts.netloc}"


class LinkState(Enum):
    CONNECTING = "connecting"      # no attempt has finished yet
    UP = "up"                      # the companion answered welcome
    DOWN = "down"


class _Done:
    pass


_DONE = _Done()


class _Connection:
    """One WebSocket session. Nothing survives it: its streams and narrations fail with it."""

    def __init__(self, ws, ops: Tuple[Op, ...]):
        self.ws = ws
        self.ops = frozenset(ops)
        self.outbox: asyncio.Queue = asyncio.Queue()
        self.hearing: Dict[int, _HearStream] = {}
        self.speaking: Dict[int, asyncio.Queue] = {}
        self.narrations: Dict[int, asyncio.Future] = {}
        self.sequence = SequenceCheck()
        self.closed = False
        self._next_hear = -1
        self._next_speak = 0

    def stream_id(self, op: Op) -> int:
        if op is Op.HEAR:
            self._next_hear = self._next_hear + 2 if self._next_hear < 65533 else 1
            return self._next_hear
        self._next_speak = self._next_speak + 2 if self._next_speak < 65534 else 2
        return self._next_speak

    def send(self, frame) -> None:
        if self.closed:
            raise CapabilityUnavailable("the companion link is down")
        self.outbox.put_nowait(frame if isinstance(frame, bytes) else encode(frame))

    def fail(self, reason: str) -> None:
        if self.closed:
            return
        self.closed = True
        for queue in self.speaking.values():
            queue.put_nowait(CapabilityUnavailable(reason))
        for future in self.narrations.values():
            if not future.done():
                future.set_exception(CapabilityUnavailable(reason))
        for stream in self.hearing.values():
            stream.failure = CapabilityUnavailable(reason)


class _HearStream:
    def __init__(self, connection: _Connection, stream: int):
        self.connection = connection
        self.id = stream
        self.audio = bytearray()
        self.inbox: List[object] = []
        self.failure: Optional[CapabilityUnavailable] = None
        self.truncated = False
        self.end_sample = 0            # where the robot's VAD ended the utterance, in samples fed to the turn
        self.deadline: Optional[float] = None
        self.bytes_since_partial = 0
        self._seq = 0

    def send(self, pcm: bytes) -> None:
        self.audio.extend(pcm)
        self.bytes_since_partial += len(pcm)
        for offset in range(0, len(pcm), MAX_MIC_PCM_BYTES):
            self.connection.send(encode_audio(AudioKind.MIC, self.id, self._seq, pcm[offset:offset + MAX_MIC_PCM_BYTES]))
            self._seq += 1

    def close(self, how: CloseHow) -> None:
        if how is CloseHow.ABORT:
            self.connection.hearing.pop(self.id, None)
        if not self.connection.closed:
            self.connection.send(HearClose(self.id, how))


def _clip(text: str, limit: int) -> str:
    return text.encode("utf-8")[:limit].decode("utf-8", "ignore")


def _fit(text: str, limit: int, boundary: str) -> str:
    """``text`` cut to ``limit`` UTF-8 bytes at the last ``boundary`` that fits, else at the limit."""
    head = _clip(text, limit)
    if head == text:
        return text
    # One character past the cut, so a boundary is not mistaken for one at the cut itself.
    ends = [m.end() for m in re.finditer(boundary, text[:len(head) + 1]) if m.end() <= len(head)]
    return head[:ends[-1]].rstrip() if ends else head


class CompanionLink:
    """The robot's one WebSocket to its companion: pinned TLS, handshake, liveness and reconnect.

    ``transitions`` records every state change. The link is up only after ``welcome``.
    """

    def __init__(self, url: str, *, token: str, out_rate: int, pin: Optional[str] = None,
                 insecure: bool = False, robot: str = "Reachy Mini",
                 ops: Tuple[Op, ...] = (Op.HEAR, Op.SPEAK, Op.NARRATE),
                 backoff_s: Tuple[float, float] = BACKOFF_S, handshake_timeout_s: float = HANDSHAKE_TIMEOUT_S,
                 ping_interval_s: float = PING_INTERVAL_S, ping_timeout_s: float = PING_TIMEOUT_S,
                 healthy_s: float = HEALTHY_S):
        self.url = parse_url(url, insecure=insecure) + "/v1"
        if self.url.startswith("wss:") and pin is None:
            raise ValueError("a wss:// companion needs its certificate pin")
        if self.url.startswith("ws:") and pin is not None:
            raise ValueError("a pinned companion needs wss://; plain ws:// has no certificate to pin")
        self._digest = parse_pin(pin) if pin is not None else None
        self._token = token
        self.out_rate = out_rate
        self._robot = robot
        self._ops = ops
        self._backoff_s = backoff_s
        self._handshake_timeout_s = handshake_timeout_s
        self._ping = (ping_interval_s, ping_timeout_s)
        self._healthy_s = healthy_s
        self.transitions: List[LinkState] = [LinkState.CONNECTING]
        self.models: Tuple[Tuple[str, str], ...] = ()
        self._connection: Optional[_Connection] = None
        self._session: Optional[asyncio.Future] = None
        self._settled: Optional[asyncio.Event] = None
        self._outages = 0
        self._announced = 0
        self._narrate_id = 0
        self._up_since: Optional[float] = None
        self._last_session_healthy = False

    @classmethod
    def from_config(cls, saved: dict, *, out_rate: int) -> CompanionLink:
        return cls(saved["url"], pin=saved["pin"], token=saved["token"], out_rate=out_rate)

    @property
    def state(self) -> LinkState:
        return self.transitions[-1]

    @property
    def up(self) -> bool:
        """Up, and either never down or back for at least ``healthy_s``."""
        return self.can(Op.NARRATE) and (self._outages == 0 or self._healthy())

    def _healthy(self) -> bool:
        return self._up_since is not None and time.monotonic() - self._up_since >= self._healthy_s

    def can(self, op: Op) -> bool:
        return (self.state is LinkState.UP and self._connection is not None
                and not self._connection.closed and op in self._connection.ops)

    def take_outage(self) -> bool:
        if self._outages == self._announced:
            return False
        self._announced = self._outages
        return True

    async def settled(self) -> None:
        """Wait until the first connection attempt has an outcome."""
        if self._settled is None:
            self._settled = asyncio.Event()
        if self.state is LinkState.CONNECTING:
            await self._settled.wait()

    def _set(self, state: LinkState) -> None:
        """Record a state change. Only the first outage, or one after a healthy session, is announced."""
        if state is self.state:
            return
        if state is LinkState.UP:
            self._up_since = time.monotonic()
        else:
            self._last_session_healthy = self._healthy()
            if self._last_session_healthy or self._outages == 0:
                self._outages += 1
            self._up_since = None
        self.transitions.append(state)
        log.info("Reachy companion link %s", state.value)

    async def run(self) -> None:
        if self._settled is None:
            self._settled = asyncio.Event()
        failures = 0
        while True:
            self._last_session_healthy = False
            self._session = asyncio.ensure_future(self._connect())
            try:
                await asyncio.wait({self._session})
            except asyncio.CancelledError:
                self._session.cancel()
                await asyncio.gather(self._session, return_exceptions=True)
                raise
            finally:
                if self._connection is not None:
                    self._connection.fail("the companion link closed")
                self._connection = None
                self._set(LinkState.DOWN)
                self._settled.set()
            failures = 0 if self._last_session_healthy else failures + 1
            if not self._session.cancelled() and self._session.exception() is not None:
                error = self._session.exception()
                (log.warning if failures <= 1 else log.debug)(
                    "Reachy companion link failed: %s: %s", type(error).__name__, error)
            low, high = self._backoff_s
            delay = min(high, low * 2 ** max(0, failures - 1))
            await asyncio.sleep(max(low, random.uniform(delay / 2, delay)))

    def _ssl(self) -> Optional[ssl.SSLContext]:
        if not self.url.startswith("wss:"):
            return None
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        return context

    async def _open(self):
        from websockets.asyncio.client import connect

        options = dict(open_timeout=self._handshake_timeout_s, ping_interval=self._ping[0],
                       ping_timeout=self._ping[1], max_size=2 * MAX_TEXT_FRAME_BYTES)
        if "proxy" in inspect.signature(connect).parameters:
            options["proxy"] = None
        context = self._ssl()
        if context is not None:
            options["ssl"] = context
        return await connect(self.url, **options)

    async def _connect(self) -> None:
        ws = await self._open()
        try:
            if self._digest is not None:
                der = ws.transport.get_extra_info("ssl_object").getpeercert(binary_form=True)
                if not hmac.compare_digest(hashlib.sha256(der or b"").hexdigest(), self._digest):
                    raise ConnectionError("the companion's certificate does not match its pin")
            await ws.send(encode(Hello((VERSION,), self._token, self._ops, self.out_rate, self._robot)))
            first = await asyncio.wait_for(ws.recv(), self._handshake_timeout_s)
            if not isinstance(first, str):
                raise ProtocolError(ErrorCode.BAD_REQUEST, Reason.WRONG_TYPE, "binary frame before welcome")
            welcome = decode(first, sender=Sender.COMPANION)
            if isinstance(welcome, Error):
                raise ConnectionError(f"the companion refused hello: {welcome.code.value}: {welcome.message}")
            if not isinstance(welcome, Welcome) or welcome.version != VERSION:
                raise ProtocolError(ErrorCode.BAD_REQUEST, Reason.WRONG_TYPE, "expected welcome")
            connection = _Connection(ws, welcome.ops)
            self.models = welcome.models
            self._connection = connection
            self._set(LinkState.UP)
            self._settled.set()
            writer = asyncio.ensure_future(self._write(connection))
            try:
                async for frame in ws:
                    self._receive(connection, frame)
            finally:
                writer.cancel()
                await asyncio.gather(writer, return_exceptions=True)
        finally:
            try:
                await asyncio.wait_for(ws.close(), .5)
            except Exception:
                ws.transport.abort()

    async def _write(self, connection: _Connection) -> None:
        while True:
            await connection.ws.send(await connection.outbox.get())

    def _receive(self, connection: _Connection, frame) -> None:
        if isinstance(frame, bytes):
            audio = decode_audio(frame)
            queue = connection.speaking.get(audio.stream)
            if audio.kind is not AudioKind.SPEECH:
                raise ProtocolError(ErrorCode.BAD_REQUEST, Reason.WRONG_DIRECTION, "microphone audio from the companion")
            if queue is not None:
                connection.sequence.accept(audio)
                queue.put_nowait(audio.pcm)
            return
        message = decode(frame, sender=Sender.COMPANION)
        if isinstance(message, (HearPartial, HearEndpoint)):
            stream = connection.hearing.get(message.stream)
            if stream is not None:
                stream.inbox.append(message)
                if isinstance(message, HearEndpoint):
                    del connection.hearing[message.stream]
        elif isinstance(message, SpeakDone):
            queue = connection.speaking.pop(message.stream, None)
            if queue is not None:
                queue.put_nowait(_DONE)
        elif isinstance(message, NarrateResult):
            future = connection.narrations.pop(message.id, None)
            if future is not None and not future.done():
                future.set_result(message.lines)
        elif isinstance(message, Error):
            self._error(connection, message)

    def _error(self, connection: _Connection, error: Error) -> None:
        if error.code in _FATAL:
            raise ConnectionError(f"the companion failed: {error.code.value}: {error.message}")
        failure = CapabilityUnavailable(f"companion {error.code.value}: {error.message}", link_down=False)
        if error.stream is not None and error.stream in connection.hearing:
            connection.hearing.pop(error.stream).failure = failure
        elif error.stream is not None and error.stream in connection.speaking:
            connection.speaking.pop(error.stream).put_nowait(failure)
            connection.sequence.forget(error.stream)
        elif error.id is not None and error.id in connection.narrations:
            future = connection.narrations.pop(error.id)
            if not future.done():
                future.set_exception(failure)
        else:
            log.warning("Reachy's companion reported %s: %s", error.code.value, error.message)

    def _live(self, op: Op) -> _Connection:
        if not self.can(op):
            raise CapabilityUnavailable(f"the companion cannot {op.value} now", link_down=self.state is not LinkState.UP)
        return self._connection

    def open_hear(self) -> _HearStream:
        connection = self._live(Op.HEAR)
        stream = _HearStream(connection, connection.stream_id(Op.HEAR))
        connection.send(HearOpen(stream.id, 0))
        connection.hearing[stream.id] = stream
        return stream

    async def speak(self, text: str, voice: VoiceRole) -> AsyncIterator[bytes]:
        """The companion's speech for one line, as s16le PCM at ``out_rate``."""
        connection = self._live(Op.SPEAK)
        if not text or len(text.encode("utf-8")) > MAX_LINE_BYTES:
            raise CapabilityUnavailable("the line does not fit one companion speech request", link_down=False)
        stream = connection.stream_id(Op.SPEAK)
        queue: asyncio.Queue = asyncio.Queue()
        connection.speaking[stream] = queue
        connection.send(SpeakStart(stream, text, voice))
        finished = False
        try:
            while True:
                item = await queue.get()
                if item is _DONE:
                    finished = True
                    return
                if isinstance(item, CapabilityUnavailable):
                    finished = True
                    raise item
                yield item
        finally:
            connection.sequence.forget(stream)
            if not finished and connection.speaking.pop(stream, None) is not None and not connection.closed:
                connection.send(SpeakCancel(stream))

    async def narrate(self, op: NarrateOp, request: str, *, status: Optional[str] = None,
                      already_said: Tuple[str, ...] = (), reply: Optional[str] = None) -> Tuple[NarratedLine, ...]:
        connection = self._live(Op.NARRATE)
        if len(connection.narrations) >= MAX_OUTSTANDING_NARRATIONS:
            raise CapabilityUnavailable("too many narrations outstanding", link_down=False)
        self._narrate_id = (self._narrate_id + 1) % 2 ** 32
        future = asyncio.get_running_loop().create_future()
        connection.narrations[self._narrate_id] = future
        identity = self._narrate_id
        try:
            connection.send(Narrate(identity, op, _clip(request, MAX_REQUEST_BYTES), status,
                                    tuple(_clip(line, MAX_LINE_BYTES) for line in already_said[-MAX_ALREADY_SAID:]),
                                    reply))
            return await future
        finally:
            connection.narrations.pop(identity, None)


def _wav(pcm: bytes) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(pcm)
    return buffer.getvalue()


class _HeardTurn:
    """A turn the companion already ended and transcribed; its audio ends at ``end_sample``."""

    def __init__(self, endpoint: Endpoint, truncated: bool, end_sample: int):
        self._endpoint = endpoint
        self.truncated = truncated
        self.end_sample = end_sample

    def start(self) -> None:
        pass

    def cancel(self) -> None:
        pass

    async def endpoint(self) -> Optional[Endpoint]:
        return self._endpoint


class CompanionHearingTurn:
    """Robot VAD opens each hear stream; the companion's endpoint ends it.

    When the robot's VAD ends an utterance first, the stream closes with ``finish`` and
    the turn ends when the companion's forced endpoint arrives. A companion that sends no
    partial for ``PARTIAL_GAP_BYTES`` of speech, or no endpoint within ``ENDPOINT_WAIT_S``
    of ``finish``, fails the turn so that the robot finishes it.
    """

    def __init__(self, link: CompanionLink, recorder):
        self._link = link
        self._recorder = recorder
        self._open: Optional[_HearStream] = None
        self._closing: Dict[int, _HearStream] = {}
        self._ended = deque()
        self._fed = 0
        self._vad_ends = deque()
        self.partial: Optional[Partial] = None

    @property
    def active(self) -> bool:
        return self._recorder.active or self._open is not None or bool(self._closing)

    @property
    def speech_active(self) -> bool:
        return self._recorder.speech_active

    def feed(self, samples) -> None:
        self._fed += len(samples)
        if self._recorder.feed(samples) is not None:
            self._vad_ends.append(self._fed)

    def finish_initial_capture(self) -> None:
        self._recorder.finish_initial_capture()

    def take(self) -> HeardAudio:
        from musegadget.voice_audio import SpeechAudio, SpeechEnd

        for event in self._recorder.take_audio_events():
            if isinstance(event, SpeechAudio):
                if self._open is None:
                    self._open = self._link.open_hear()
                self._open.send(event.pcm)
            elif isinstance(event, SpeechEnd) and self._open is not None:
                stream, self._open = self._open, None
                if event.accepted:
                    stream.truncated = self._recorder.last_truncated
                    stream.end_sample = self._vad_ends.popleft() if self._vad_ends else self._fed
                    stream.deadline = time.monotonic() + ENDPOINT_WAIT_S
                    self._closing[stream.id] = stream
                    stream.close(CloseHow.FINISH)
                else:
                    stream.close(CloseHow.ABORT)
                self.partial = None
        failure = None
        for stream in [*([self._open] if self._open is not None else []), *self._closing.values()]:
            messages, stream.inbox = stream.inbox, []
            for message in messages:
                if isinstance(message, HearPartial):
                    stream.bytes_since_partial = 0
                    if stream is self._open:
                        self.partial = Partial(message.text, message.rev, stream.id)
                    continue
                if stream is self._open:
                    self._open = None
                    stream.end_sample = self._fed
                    self._recorder.reset()
                self._closing.pop(stream.id, None)
                self.partial = None
                self._ended.append(_HeardTurn(Endpoint(_wav(bytes(stream.audio)), message.text, message.forced),
                                              stream.truncated, stream.end_sample))
                break
            else:
                failure = failure or stream.failure or self._overdue(stream)
        if self._ended:
            return HeardAudio(self._ended.popleft())
        if failure is not None:
            raise failure
        return HeardAudio()

    def _overdue(self, stream: _HearStream) -> Optional[CapabilityUnavailable]:
        if stream.deadline is not None and time.monotonic() >= stream.deadline:
            return CapabilityUnavailable("the companion did not end the turn", link_down=False)
        if stream is self._open and stream.bytes_since_partial >= PARTIAL_GAP_BYTES:
            return CapabilityUnavailable("the companion sent no transcript for the speech", link_down=False)
        return None

    def abort(self) -> None:
        for stream in [*([self._open] if self._open is not None else []), *self._closing.values()]:
            stream.close(CloseHow.ABORT)
        self._open = None
        self._closing.clear()
        self.partial = None


class CompanionHearing:
    transcribes = True

    def __init__(self, link: CompanionLink):
        self.link = link

    @property
    def available(self) -> bool:
        return self.link.can(Op.HEAR)

    async def start(self) -> None:
        await self.link.settled()

    def open_turn(self, sample_rate: int, *, silence_s: float, vad=None, **options) -> CompanionHearingTurn:
        from musegadget.voice_audio import TurnRecorder

        recorder = TurnRecorder(sample_rate, silence_s=silence_s, max_s=60.0, stream_audio=True,
                                **({"vad": vad} if vad is not None else {}), **options)
        return CompanionHearingTurn(self.link, recorder)

    async def transcribe(self, wav: bytes) -> str:
        raise CapabilityUnavailable("the companion transcribes live turns only", link_down=False)


class CompanionVoice:
    """Speech synthesized by the companion at the robot's output rate."""

    speaks_text = True

    def __init__(self, link: CompanionLink, role: VoiceRole = VoiceRole.MAIN):
        self.link = link
        self.role = role

    @property
    def available(self) -> bool:
        return self.link.can(Op.SPEAK)

    async def warm(self, output_rate: int) -> None:
        pass

    async def stream(self, line: SpokenLine, output_rate: int):
        import numpy as np

        if output_rate != self.link.out_rate:
            raise CapabilityUnavailable("the companion speaks at another rate", link_down=False)
        speech = self.link.speak(line.text, self.role)
        try:
            async for pcm in speech:
                yield np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
        finally:
            await speech.aclose()

    def prepare(self, line: SpokenLine, output_rate: int):
        return None


class CompanionNarrator:
    """The companion's narrator. Ids and request text only; never Muse identifiers."""

    def __init__(self, link: CompanionLink):
        self.link = link

    @property
    def available(self) -> bool:
        return self.link.can(Op.NARRATE)

    async def acknowledge(self, request: str) -> Optional[SpokenLine]:
        lines = await self.link.narrate(NarrateOp.ACKNOWLEDGE, request)
        return _spoken(lines[0], "ack") if lines else None

    async def say_progress(self, request: str, status: str,
                           already_said: Tuple[str, ...]) -> Optional[SpokenLine]:
        lines = await self.link.narrate(NarrateOp.PROGRESS, request, status=_fit(status, MAX_STATUS_BYTES, r"\S+"),
                                        already_said=already_said)
        return _spoken(lines[0], "progress") if lines else None

    async def lines(self, request: str, reply: str, style: ReplyStyle, *,
                    message_id: Optional[str] = None) -> Tuple[SpokenLine, ...]:
        if style is not ReplyStyle.PLAIN_SHORT or not reply:
            raise CapabilityUnavailable("the companion condenses only plain replies", link_down=False)
        if len(reply.encode("utf-8")) > MAX_REPLY_BYTES:
            raise CapabilityUnavailable("the reply is too long for the companion to condense", link_down=False)
        lines = await self.link.narrate(NarrateOp.LINES, request, reply=reply)
        return tuple(_spoken(line, "answer") for line in lines)


def _spoken(line: NarratedLine, role) -> SpokenLine:
    return SpokenLine(line.text, line.expression, role)
