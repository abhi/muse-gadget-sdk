"""An in-process companion speaking wire protocol v1 over plain ws://, for tests.

Scripted hearing, a deterministic tone for speech, an echo narrator, and fault knobs:
- abort_after_frames_both_ways_since_welcome=N: abort the TCP connection, with no close
  frame, once N frames have crossed it after `welcome`, counting both directions.
- stall: after `welcome`, never answer anything again (transport pings still work).
- fabricate: append " It took 17 minutes." to the first narrated line.
- gap: never send speech seq 1, so the robot sees 0, 2, 3, ...
- expression: the expression attached to every narrated line.
- drop_mid_speech: abort the TCP connection right after the first speech frame, once.
- fail_after_welcome: send error{internal} right after every `welcome`, which ends the session.
- mute: never answer speak_start, with audio or speak_done.
- no_endpoint: never send hear_endpoint, even for a stream the robot closed with finish.
- port: listen on this port instead of a free one, so a test can start the companion late.
- ssl: serve wss:// with this server context.
- acknowledges: False answers every acknowledge request with no lines, as a narrator does for small talk.

`hellos` records every hello token the fake was sent, accepted or not.
"""

from __future__ import annotations

import asyncio
import hmac
import math
import re
import struct
from typing import Dict, Optional, Tuple

from websockets.asyncio.server import serve

from musegadget.reachy_capabilities import Expression
from musegadget.reachy_companion_protocol import (
    MAX_ERROR_MESSAGE_BYTES,
    MAX_LINE_BYTES,
    VERSION,
    AudioKind,
    CloseHow,
    Error,
    ErrorCode,
    HearClose,
    HearEndpoint,
    HearOpen,
    HearPartial,
    Hello,
    Narrate,
    NarratedLine,
    NarrateOp,
    NarrateResult,
    Op,
    ProtocolError,
    Sender,
    SpeakCancel,
    SpeakDone,
    SpeakStart,
    Welcome,
    decode,
    decode_audio,
    encode,
    encode_audio,
)

TOKEN = "rc1_test_token_0123456789abcdef0123456789"
MODELS = (("hear", "fake"), ("speak", "tone"), ("narrate", "echo"))
FABRICATION = " It took 17 minutes."


class _Dropped(Exception):
    pass


class FakeCompanion:
    def __init__(
        self,
        *,
        token: str = TOKEN,
        ops: Tuple[Op, ...] = (Op.HEAR, Op.SPEAK, Op.NARRATE),
        partials: Tuple[str, ...] = ("hello",),
        frames_per_partial: int = 5,
        endpoint_after: Optional[int] = None,
        speech_frames: int = 3,
        abort_after_frames_both_ways_since_welcome: Optional[int] = None,
        stall: bool = False,
        fabricate: bool = False,
        gap: bool = False,
        expression: Optional[Expression] = None,
        drop_mid_speech: bool = False,
        fail_after_welcome: bool = False,
        mute: bool = False,
        no_endpoint: bool = False,
        port: int = 0,
        ssl=None,
        acknowledges: bool = True,
    ) -> None:
        self.token = token
        self.ops = ops
        self.partials = partials
        self.frames_per_partial = frames_per_partial
        self.endpoint_after = endpoint_after
        self.speech_frames = speech_frames
        self.abort_after_frames_both_ways_since_welcome = abort_after_frames_both_ways_since_welcome
        self.stall = stall
        self.fabricate = fabricate
        self.gap = gap
        self.expression = expression
        self.drop_mid_speech = drop_mid_speech
        self.fail_after_welcome = fail_after_welcome
        self.mute = mute
        self.no_endpoint = no_endpoint
        self.port = port
        self.ssl = ssl
        self.acknowledges = acknowledges
        self.hellos = []
        self.received = []
        self.url = ""

    @staticmethod
    def tone(rate: int, index: int) -> bytes:
        count = rate // 10
        start = index * count
        return struct.pack(f"<{count}h", *(int(8000 * math.sin(2 * math.pi * 440 * (start + n) / rate)) for n in range(count)))

    async def __aenter__(self) -> FakeCompanion:
        self._server = await serve(self._session, "127.0.0.1", self.port, ssl=self.ssl)
        port = self._server.sockets[0].getsockname()[1]
        self.url = f"{'wss' if self.ssl else 'ws'}://127.0.0.1:{port}/v1"
        return self

    async def __aexit__(self, *exc: object) -> None:
        self._server.close()
        await self._server.wait_closed()

    async def _session(self, ws) -> None:
        session = _Session(self, ws)
        try:
            await session.run()
        except _Dropped:
            ws.transport.abort()


class _Session:
    def __init__(self, fake: FakeCompanion, ws) -> None:
        self.fake = fake
        self.ws = ws
        self.crossed: Optional[int] = None
        self.out_rate = 0
        self.hearing: Dict[int, int] = {}
        self.speaking: Dict[int, asyncio.Task] = {}
        self.sent: Dict[int, int] = {}

    def _count(self) -> None:
        if self.crossed is None:
            return
        self.crossed += 1
        limit = self.fake.abort_after_frames_both_ways_since_welcome
        if limit is not None and self.crossed >= limit:
            raise _Dropped

    async def send(self, frame) -> None:
        if self.fake.stall and self.crossed is not None:
            return
        await self.ws.send(frame if isinstance(frame, bytes) else encode(frame))
        self._count()

    async def run(self) -> None:
        first = await self.ws.recv()
        try:
            hello = decode(first, sender=Sender.ROBOT) if isinstance(first, str) else None
        except ProtocolError:
            hello = None
        if isinstance(hello, Hello):
            self.fake.hellos.append(hello.token)
        if not isinstance(hello, Hello) or not hmac.compare_digest(hello.token.encode(), self.fake.token.encode()):
            await self.ws.send(encode(Error(ErrorCode.AUTH, "bad token")))
            return
        if VERSION not in hello.versions:
            await self.ws.send(encode(Error(ErrorCode.VERSION, f"only v{VERSION}")))
            return
        self.out_rate = hello.out_rate
        ops = tuple(op for op in hello.ops if op in self.fake.ops)
        await self.ws.send(encode(Welcome(VERSION, ops, tuple(m for m in MODELS if Op(m[0]) in ops))))
        if self.fake.fail_after_welcome:
            await self.ws.send(encode(Error(ErrorCode.INTERNAL, "the model crashed")))
            return
        self.crossed = 0
        try:
            async for frame in self.ws:
                self.fake.received.append(frame)
                self._count()
                try:
                    if isinstance(frame, bytes):
                        await self._audio(frame)
                    else:
                        await self._message(decode(frame, sender=Sender.ROBOT))
                except ProtocolError as error:
                    await self.send(Error(error.code, _clip(str(error), MAX_ERROR_MESSAGE_BYTES)))
        finally:
            for task in self.speaking.values():
                task.cancel()

    async def _audio(self, frame: bytes) -> None:
        audio = decode_audio(frame)
        if audio.kind is not AudioKind.MIC or audio.stream not in self.hearing:
            return
        heard = self.hearing[audio.stream] = self.hearing[audio.stream] + 1
        if heard % self.fake.frames_per_partial == 0 and heard // self.fake.frames_per_partial <= len(self.fake.partials):
            rev = heard // self.fake.frames_per_partial - 1
            await self.send(HearPartial(audio.stream, rev, self.fake.partials[rev]))
        if heard == self.fake.endpoint_after and not self.fake.no_endpoint:
            del self.hearing[audio.stream]
            await self.send(HearEndpoint(audio.stream, self.fake.partials[-1], False))

    async def _message(self, message) -> None:
        if isinstance(message, HearOpen):
            self.hearing[message.stream] = 0
        elif isinstance(message, HearClose):
            if (self.hearing.pop(message.stream, None) is not None and message.how is CloseHow.FINISH
                    and not self.fake.no_endpoint):
                await self.send(HearEndpoint(message.stream, self.fake.partials[-1], True))
        elif isinstance(message, SpeakStart) and not self.fake.mute:
            self.speaking[message.stream] = asyncio.ensure_future(self._speak(message.stream))
        elif isinstance(message, SpeakCancel):
            task = self.speaking.pop(message.stream, None)
            if task is not None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                await self.send(SpeakDone(message.stream, self.sent[message.stream]))
        elif isinstance(message, Narrate):
            await self.send(NarrateResult(message.id, self._echo(message)))

    async def _speak(self, stream: int) -> None:
        self.sent[stream] = 0
        try:
            for seq in range(self.fake.speech_frames):
                if self.fake.gap and seq == 1:
                    continue
                await self.send(encode_audio(AudioKind.SPEECH, stream, seq, FakeCompanion.tone(self.out_rate, seq)))
                self.sent[stream] += 1
                if self.fake.drop_mid_speech:
                    self.fake.drop_mid_speech = False
                    raise _Dropped
                await asyncio.sleep(0)
            del self.speaking[stream]
            await self.send(SpeakDone(stream, self.sent[stream]))
        except _Dropped:
            self.ws.transport.abort()

    def _echo(self, message: Narrate) -> Tuple[NarratedLine, ...]:
        if message.op is NarrateOp.ACKNOWLEDGE and not self.fake.acknowledges:
            return ()
        source = {
            NarrateOp.ACKNOWLEDGE: message.request,
            NarrateOp.PROGRESS: message.status,
            NarrateOp.LINES: message.reply,
        }[message.op] or ""
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", source.strip()) if s][:3]
        if self.fake.fabricate and sentences:
            sentences[0] += FABRICATION
        return tuple(NarratedLine(_clip(s), self.fake.expression) for s in sentences)


def _clip(text: str, limit: int = MAX_LINE_BYTES) -> str:
    return text.encode("utf-8")[:limit].decode("utf-8", "ignore")
