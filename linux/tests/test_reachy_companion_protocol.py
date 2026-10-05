import asyncio
import base64
import dataclasses
import json
from enum import Enum
from importlib.resources import files

import pytest
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

from fake_companion import TOKEN, FakeCompanion
from musegadget.reachy_companion_protocol import (
    MESSAGES,
    AudioFrame,
    AudioKind,
    CloseHow,
    ErrorCode,
    HearClose,
    HearOpen,
    Hello,
    Op,
    ProtocolError,
    Reason,
    Sender,
    SequenceCheck,
    SpeakStart,
    VoiceRole,
    decode,
    decode_audio,
    encode,
    encode_audio,
    message_type,
)

VECTORS = files("musegadget") / "data" / "companion_protocol_v1"
VALID = json.loads((VECTORS / "valid.json").read_text(encoding="utf-8"))
INVALID = json.loads((VECTORS / "invalid.json").read_text(encoding="utf-8"))


def plain(value):
    if dataclasses.is_dataclass(value):
        return {f.name: plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    return value


def by_name(vectors):
    return pytest.mark.parametrize("vector", vectors, ids=[v["name"] for v in vectors])


@by_name(VALID)
def test_valid_vector_decodes_and_round_trips(vector):
    if "audio" in vector:
        frame = base64.b64decode(vector["audio"])
        decoded = decode_audio(frame)
        assert {**plain(decoded), "pcm": base64.b64encode(decoded.pcm).decode()} == vector["expect"]
        assert encode_audio(decoded.kind, decoded.stream, decoded.seq, decoded.pcm) == frame
        return
    sender = Sender(vector["sender"])
    message = decode(vector["text"], sender=sender)
    assert {"t": message_type(message), **plain(message)} == vector["expect"]
    assert decode(encode(message), sender=sender) == message


@by_name(INVALID)
def test_invalid_vector_raises_protocol_error(vector):
    with pytest.raises(ProtocolError) as caught:
        if "audio" in vector:
            decode_audio(base64.b64decode(vector["audio"]))
        else:
            decode(vector["text"], sender=Sender(vector["sender"]))
    assert (caught.value.code.value, caught.value.reason.value) == (vector["code"], vector["reason"])


def test_vectors_cover_every_message_type_and_audio_kind():
    decoded = {json.loads(v["text"])["t"] for v in VALID if "text" in v}
    kinds = {v["expect"]["kind"] for v in VALID if "audio" in v}
    assert (decoded, kinds) == (set(MESSAGES), {1, 2})


def test_encode_refuses_what_decode_would_refuse():
    with pytest.raises(ProtocolError) as too_long:
        encode(SpeakStart(stream=2, text="a" * 241, voice=VoiceRole.MAIN))
    with pytest.raises(ProtocolError) as wrong_parity:
        encode(HearOpen(stream=2, pre_roll_frames=0))
    assert (too_long.value.reason, wrong_parity.value.reason) == (Reason.TOO_LONG, Reason.BAD_STREAM)


def test_encode_audio_refuses_seq_overflow():
    with pytest.raises(ProtocolError) as caught:
        encode_audio(AudioKind.MIC, 1, 2**32, b"\x00\x00")
    assert (caught.value.code, caught.value.reason) == (ErrorCode.BAD_REQUEST, Reason.OUT_OF_RANGE)


def test_audio_header_layout_is_kind_pad_stream_seq_little_endian():
    frame = encode_audio(AudioKind.SPEECH, 0x0102, 0x03040506, b"\x07\x08")
    assert frame == b"\x02\x00\x02\x01\x06\x05\x04\x03\x07\x08"


def test_sequence_check_fails_the_stream_on_a_gap():
    check = SequenceCheck()
    check.accept(AudioFrame(AudioKind.SPEECH, 2, 0, b"\x00\x00"))
    check.accept(AudioFrame(AudioKind.SPEECH, 4, 0, b"\x00\x00"))
    with pytest.raises(ProtocolError) as caught:
        check.accept(AudioFrame(AudioKind.SPEECH, 2, 2, b"\x00\x00"))
    assert (caught.value.code, str(caught.value)) == (ErrorCode.GAP, "gap/gap: stream 2 expected seq 1, got 2")


HELLO = Hello(versions=(1,), token=TOKEN, ops=(Op.HEAR, Op.SPEAK, Op.NARRATE), out_rate=24000, robot="test")


async def received(ws, count):
    frames = []
    for _ in range(count):
        frame = await asyncio.wait_for(ws.recv(), 2)
        frames.append(decode_audio(frame) if isinstance(frame, bytes) else decode(frame, sender=Sender.COMPANION))
    return frames


def test_fake_companion_happy_turn():
    async def scenario():
        async with FakeCompanion(partials=("what's", "what's the weather"), frames_per_partial=2, speech_frames=2) as fake:
            async with connect(fake.url) as ws:
                await ws.send(encode(HELLO))
                frames = await received(ws, 1)
                await ws.send(encode(HearOpen(stream=1, pre_roll_frames=0)))
                for seq in range(4):
                    await ws.send(encode_audio(AudioKind.MIC, 1, seq, b"\x00\x00" * 320))
                await ws.send(encode(HearClose(stream=1, how=CloseHow.FINISH)))
                frames += await received(ws, 3)
                await ws.send(encode(SpeakStart(stream=2, text="It's sunny.", voice=VoiceRole.MAIN)))
                frames += await received(ws, 3)
                await ws.send(
                    '{"t":"narrate","id":7,"op":"lines","request":"weather?",'
                    '"reply":"It is 14 degrees. Sunny all day. Wind is calm. Rain tomorrow."}'
                )
                return frames + await received(ws, 1)

    assert [plain(frame) for frame in asyncio.run(scenario())] == [
        {"version": 1, "ops": ["hear", "speak", "narrate"], "models": [["hear", "fake"], ["speak", "tone"], ["narrate", "echo"]]},
        {"stream": 1, "rev": 0, "text": "what's"},
        {"stream": 1, "rev": 1, "text": "what's the weather"},
        {"stream": 1, "text": "what's the weather", "forced": True},
        {"kind": 2, "stream": 2, "seq": 0, "pcm": FakeCompanion.tone(24000, 0)},
        {"kind": 2, "stream": 2, "seq": 1, "pcm": FakeCompanion.tone(24000, 1)},
        {"stream": 2, "frames": 2},
        {"id": 7, "lines": [
            {"text": "It is 14 degrees.", "expression": None},
            {"text": "Sunny all day.", "expression": None},
            {"text": "Wind is calm.", "expression": None},
        ]},
    ]


def test_fake_companion_refuses_a_wrong_token_and_closes():
    async def scenario():
        async with FakeCompanion() as fake:
            async with connect(fake.url) as ws:
                await ws.send(encode(dataclasses.replace(HELLO, token="rc1_wrong")))
                refusal = await received(ws, 1)
                await asyncio.wait_for(ws.wait_closed(), 2)
                return refusal

    assert [plain(m) for m in asyncio.run(scenario())] == [{"code": "auth", "message": "bad token", "stream": None, "id": None}]


def test_fake_companion_gap_and_fabricate_faults():
    async def scenario():
        async with FakeCompanion(speech_frames=3, gap=True, fabricate=True) as fake:
            async with connect(fake.url) as ws:
                await ws.send(encode(HELLO))
                await received(ws, 1)
                await ws.send(encode(SpeakStart(stream=2, text="Hi.", voice=VoiceRole.PROGRESS)))
                spoken = await received(ws, 3)
                await ws.send('{"t":"narrate","id":1,"op":"lines","request":"q","reply":"It is sunny."}')
                return spoken, await received(ws, 1)

    spoken, narrated = asyncio.run(scenario())
    check = SequenceCheck()
    check.accept(spoken[0])
    with pytest.raises(ProtocolError) as caught:
        check.accept(spoken[1])
    assert ([frame.seq for frame in spoken[:2]], caught.value.code) == ([0, 2], ErrorCode.GAP)
    assert plain(spoken[2]) == {"stream": 2, "frames": 2}
    assert plain(narrated[0]) == {"id": 1, "lines": [{"text": "It is sunny. It took 17 minutes.", "expression": None}]}


def test_fake_companion_abort_after_frames_closes_without_a_close_frame():
    async def scenario():
        async with FakeCompanion(abort_after_frames_both_ways_since_welcome=2, speech_frames=5) as fake:
            async with connect(fake.url) as ws:
                await ws.send(encode(HELLO))
                await received(ws, 1)
                await ws.send(encode(SpeakStart(stream=2, text="Hi.", voice=VoiceRole.MAIN)))
                frames = []
                try:
                    while True:
                        frames.append(await asyncio.wait_for(ws.recv(), 2))
                except ConnectionClosed as closed:
                    return len(frames), closed.rcvd

    assert asyncio.run(scenario()) == (1, None)


@pytest.mark.parametrize("frame, reasons", [
    ('{"t": "speak_start", "stream": 2, "text": "\\ud800", "voice": "main"}', {Reason.WRONG_TYPE}),
    ("[" * 6000 + "]" * 6000, {Reason.NOT_JSON, Reason.NOT_OBJECT}),
], ids=["lone-surrogate", "deep-nesting"])
def test_hostile_text_frames_raise_protocol_errors(frame, reasons):
    with pytest.raises(ProtocolError) as caught:
        decode(frame, sender=Sender.ROBOT)
    assert caught.value.code == ErrorCode.BAD_REQUEST
    assert caught.value.reason in reasons
