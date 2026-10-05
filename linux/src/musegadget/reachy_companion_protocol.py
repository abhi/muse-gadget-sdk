# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Wire protocol v1 between a Reachy robot and its companion. The spec is docs/reachy-companion-protocol.md."""

from __future__ import annotations

import json
import struct
from dataclasses import MISSING, dataclass, field, fields
from enum import Enum, IntEnum
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

from musegadget.reachy_capabilities import Expression

VERSION = 1

MAX_TEXT_FRAME_BYTES = 16 * 1024
MAX_LINE_BYTES = 240
MAX_NARRATED_LINES = 3
MAX_TRANSCRIPT_BYTES = 4096
MAX_REQUEST_BYTES = 4096
MAX_REPLY_BYTES = 8192
MAX_STATUS_BYTES = 120
MAX_ALREADY_SAID = 8
MAX_ERROR_MESSAGE_BYTES = 500
MAX_PRE_ROLL_FRAMES = 150
MAX_OUTSTANDING_NARRATIONS = 4
MIN_OUT_RATE = 8000
MAX_OUT_RATE = 48000
U16_MAX = 0xFFFF
U32_MAX = 0xFFFFFFFF

AUDIO_HEADER = struct.Struct("<BxHI")
MIC_RATE = 16000
MAX_MIC_PCM_BYTES = MIC_RATE * 20 // 1000 * 2
MAX_SPEECH_PCM_BYTES = MAX_OUT_RATE * 100 // 1000 * 2


class Sender(str, Enum):
    ROBOT = "robot"
    COMPANION = "companion"


class ErrorCode(str, Enum):
    AUTH = "auth"
    VERSION = "version"
    UNKNOWN_TYPE = "unknown_type"
    BAD_REQUEST = "bad_request"
    GAP = "gap"
    OVERLOADED = "overloaded"
    MODEL = "model"
    INTERNAL = "internal"


class Reason(str, Enum):
    """Why a frame was refused. Local detail; only the ErrorCode goes on the wire."""

    TOO_LARGE = "too_large"
    NOT_JSON = "not_json"
    NOT_OBJECT = "not_object"
    UNKNOWN_TYPE = "unknown_type"
    WRONG_DIRECTION = "wrong_direction"
    MISSING_FIELD = "missing_field"
    WRONG_TYPE = "wrong_type"
    BAD_ENUM = "bad_enum"
    OUT_OF_RANGE = "out_of_range"
    TOO_MANY = "too_many"
    TOO_LONG = "too_long"
    EMPTY = "empty"
    BAD_STREAM = "bad_stream"
    SHORT_FRAME = "short_frame"
    ODD_PCM = "odd_pcm"
    GAP = "gap"


class ProtocolError(Exception):
    def __init__(self, code: ErrorCode, reason: Reason, detail: str = "") -> None:
        super().__init__(f"{code.value}/{reason.value}: {detail}" if detail else f"{code.value}/{reason.value}")
        self.code = code
        self.reason = reason
        self.detail = detail


def _bad(reason: Reason, detail: str) -> ProtocolError:
    return ProtocolError(ErrorCode.BAD_REQUEST, reason, detail)


class Op(str, Enum):
    HEAR = "hear"
    SPEAK = "speak"
    NARRATE = "narrate"


class CloseHow(str, Enum):
    FINISH = "finish"
    ABORT = "abort"


class VoiceRole(str, Enum):
    MAIN = "main"
    PROGRESS = "progress"


class NarrateOp(str, Enum):
    ACKNOWLEDGE = "acknowledge"
    PROGRESS = "progress"
    LINES = "lines"


class AudioKind(IntEnum):
    MIC = 1
    SPEECH = 2


@dataclass(frozen=True)
class _Codec:
    parse: Callable[[str, Any], Any]
    dump: Callable[[Any], Any] = lambda value: value


def _int(lo: int, hi: int) -> _Codec:
    def parse(name: str, raw: Any) -> int:
        if type(raw) is not int:
            raise _bad(Reason.WRONG_TYPE, f"{name} must be an integer")
        if not lo <= raw <= hi:
            raise _bad(Reason.OUT_OF_RANGE, f"{name} must be in {lo}..{hi}")
        return raw

    return _Codec(parse)


class _StreamUse(Enum):
    """Which streams a field may name; the value is the id parity that use requires."""

    HEAR = 1
    SPEAK = 0
    EITHER = None


def _stream(use: _StreamUse) -> _Codec:
    in_range = _int(0, U16_MAX).parse
    parity = use.value

    def parse(name: str, raw: Any) -> int:
        value = in_range(name, raw)
        if parity is not None and (value == 0 or value % 2 != parity):
            raise _bad(Reason.BAD_STREAM, f"{name} {value} has the wrong parity")
        return value

    return _Codec(parse)


def _bool() -> _Codec:
    def parse(name: str, raw: Any) -> bool:
        if type(raw) is not bool:
            raise _bad(Reason.WRONG_TYPE, f"{name} must be a boolean")
        return raw

    return _Codec(parse)


def _text(max_bytes: int, *, non_empty: bool = False) -> _Codec:
    def parse(name: str, raw: Any) -> str:
        if not isinstance(raw, str):
            raise _bad(Reason.WRONG_TYPE, f"{name} must be a string")
        try:
            size = len(raw.encode("utf-8"))
        except UnicodeEncodeError:
            raise _bad(Reason.WRONG_TYPE, f"{name} is not valid UTF-8") from None
        if size > max_bytes:
            raise _bad(Reason.TOO_LONG, f"{name} exceeds {max_bytes} bytes")
        if non_empty and not raw.strip():
            raise _bad(Reason.EMPTY, f"{name} must not be empty")
        return raw

    return _Codec(parse)


def _enum(kind: Type[Enum]) -> _Codec:
    def parse(name: str, raw: Any) -> Enum:
        if not isinstance(raw, str):
            raise _bad(Reason.WRONG_TYPE, f"{name} must be a string")
        try:
            return kind(raw)
        except ValueError:
            raise _bad(Reason.BAD_ENUM, f"{name} {raw!r} is not one of {[m.value for m in kind]}") from None

    return _Codec(parse, lambda value: value.value)


def _list(item: _Codec, max_items: int, *, min_items: int = 0, drop_unknown: bool = False) -> _Codec:
    def parse(name: str, raw: Any) -> tuple:
        if not isinstance(raw, list):
            raise _bad(Reason.WRONG_TYPE, f"{name} must be a list")
        if len(raw) > max_items:
            raise _bad(Reason.TOO_MANY, f"{name} has more than {max_items} items")
        items = []
        for index, value in enumerate(raw):
            try:
                items.append(item.parse(f"{name}[{index}]", value))
            except ProtocolError as error:
                if not (drop_unknown and error.reason is Reason.BAD_ENUM):
                    raise
        if len(items) < min_items:
            raise _bad(Reason.EMPTY, f"{name} needs at least {min_items} item")
        return tuple(items)

    return _Codec(parse, lambda values: [item.dump(value) for value in values])


def _optional(inner: _Codec) -> _Codec:
    return _Codec(
        lambda name, raw: None if raw is None else inner.parse(name, raw),
        lambda value: None if value is None else inner.dump(value),
    )


def _models() -> _Codec:
    key, value = _text(32, non_empty=True), _text(120)

    def parse(name: str, raw: Any) -> Tuple[Tuple[str, str], ...]:
        if not isinstance(raw, dict):
            raise _bad(Reason.WRONG_TYPE, f"{name} must be an object")
        if len(raw) > 8:
            raise _bad(Reason.TOO_MANY, f"{name} has more than 8 entries")
        return tuple((key.parse(f"{name} key", k), value.parse(f"{name}.{k}", v)) for k, v in raw.items())

    return _Codec(parse, lambda pairs: dict(pairs))


@dataclass(frozen=True)
class NarratedLine:
    text: str
    expression: Optional[Expression]


def _narrated_line() -> _Codec:
    text = _text(MAX_LINE_BYTES, non_empty=True)

    def parse(name: str, raw: Any) -> NarratedLine:
        if not isinstance(raw, list) or len(raw) != 2:
            raise _bad(Reason.WRONG_TYPE, f"{name} must be [text, expression]")
        expression = raw[1]
        if expression is not None and not isinstance(expression, str):
            raise _bad(Reason.WRONG_TYPE, f"{name} expression must be a string or null")
        return NarratedLine(text.parse(f"{name} text", raw[0]), Expression.parse(expression))

    return _Codec(parse, lambda line: [line.text, None if line.expression is None else line.expression.value])


def _wire(codec: _Codec, **kwargs: Any) -> Any:
    return field(metadata={"codec": codec}, **kwargs)


HEAR_STREAM = _stream(_StreamUse.HEAR)
SPEAK_STREAM = _stream(_StreamUse.SPEAK)
U32 = _int(0, U32_MAX)


@dataclass(frozen=True)
class Hello:
    versions: Tuple[int, ...] = _wire(_list(_int(1, U16_MAX), 8, min_items=1))
    token: str = _wire(_text(128, non_empty=True))
    ops: Tuple[Op, ...] = _wire(_list(_enum(Op), 8, drop_unknown=True))
    out_rate: int = _wire(_int(MIN_OUT_RATE, MAX_OUT_RATE))
    robot: str = _wire(_text(64))


@dataclass(frozen=True)
class HearOpen:
    stream: int = _wire(HEAR_STREAM)
    pre_roll_frames: int = _wire(_int(0, MAX_PRE_ROLL_FRAMES))


@dataclass(frozen=True)
class HearClose:
    stream: int = _wire(HEAR_STREAM)
    how: CloseHow = _wire(_enum(CloseHow))


@dataclass(frozen=True)
class SpeakStart:
    stream: int = _wire(SPEAK_STREAM)
    text: str = _wire(_text(MAX_LINE_BYTES, non_empty=True))
    voice: VoiceRole = _wire(_enum(VoiceRole))


@dataclass(frozen=True)
class SpeakCancel:
    stream: int = _wire(SPEAK_STREAM)


@dataclass(frozen=True)
class Narrate:
    id: int = _wire(U32)
    op: NarrateOp = _wire(_enum(NarrateOp))
    request: str = _wire(_text(MAX_REQUEST_BYTES))
    status: Optional[str] = _wire(_optional(_text(MAX_STATUS_BYTES, non_empty=True)), default=None)
    already_said: Tuple[str, ...] = _wire(_list(_text(MAX_LINE_BYTES), MAX_ALREADY_SAID), default=())
    reply: Optional[str] = _wire(_optional(_text(MAX_REPLY_BYTES, non_empty=True)), default=None)


@dataclass(frozen=True)
class Welcome:
    version: int = _wire(_int(1, U16_MAX))
    ops: Tuple[Op, ...] = _wire(_list(_enum(Op), 8))
    models: Tuple[Tuple[str, str], ...] = _wire(_models(), default=())


@dataclass(frozen=True)
class HearPartial:
    stream: int = _wire(HEAR_STREAM)
    rev: int = _wire(U32)
    text: str = _wire(_text(MAX_TRANSCRIPT_BYTES))


@dataclass(frozen=True)
class HearEndpoint:
    stream: int = _wire(HEAR_STREAM)
    text: str = _wire(_text(MAX_TRANSCRIPT_BYTES))
    forced: bool = _wire(_bool())


@dataclass(frozen=True)
class SpeakDone:
    stream: int = _wire(SPEAK_STREAM)
    frames: int = _wire(U32)


@dataclass(frozen=True)
class NarrateResult:
    id: int = _wire(U32)
    lines: Tuple[NarratedLine, ...] = _wire(_list(_narrated_line(), MAX_NARRATED_LINES))


@dataclass(frozen=True)
class Error:
    code: ErrorCode = _wire(_enum(ErrorCode))
    message: str = _wire(_text(MAX_ERROR_MESSAGE_BYTES), default="")
    stream: Optional[int] = _wire(_optional(_stream(_StreamUse.EITHER)), default=None)
    id: Optional[int] = _wire(_optional(U32), default=None)


Message = Union[
    Hello, HearOpen, HearClose, SpeakStart, SpeakCancel, Narrate,
    Welcome, HearPartial, HearEndpoint, SpeakDone, NarrateResult, Error,
]


def _narrate_has_op_inputs(message: Narrate) -> None:
    needed = {NarrateOp.PROGRESS: "status", NarrateOp.LINES: "reply"}.get(message.op)
    if needed is not None and getattr(message, needed) is None:
        raise _bad(Reason.MISSING_FIELD, f"narrate op {message.op.value} needs {needed}")


@dataclass(frozen=True)
class _Spec:
    cls: type
    sender: Sender
    check: Optional[Callable[[Any], None]] = None


MESSAGES: Dict[str, _Spec] = {
    "hello": _Spec(Hello, Sender.ROBOT),
    "hear_open": _Spec(HearOpen, Sender.ROBOT),
    "hear_close": _Spec(HearClose, Sender.ROBOT),
    "speak_start": _Spec(SpeakStart, Sender.ROBOT),
    "speak_cancel": _Spec(SpeakCancel, Sender.ROBOT),
    "narrate": _Spec(Narrate, Sender.ROBOT, _narrate_has_op_inputs),
    "welcome": _Spec(Welcome, Sender.COMPANION),
    "hear_partial": _Spec(HearPartial, Sender.COMPANION),
    "hear_endpoint": _Spec(HearEndpoint, Sender.COMPANION),
    "speak_done": _Spec(SpeakDone, Sender.COMPANION),
    "narrate_result": _Spec(NarrateResult, Sender.COMPANION),
    "error": _Spec(Error, Sender.COMPANION),
}
_TYPE_OF: Dict[type, str] = {spec.cls: name for name, spec in MESSAGES.items()}


def message_type(message: Message) -> str:
    return _TYPE_OF[type(message)]


def _build(spec: _Spec, obj: Dict[str, Any]) -> Message:
    values = {}
    for f in fields(spec.cls):
        codec: _Codec = f.metadata["codec"]
        if f.name in obj:
            values[f.name] = codec.parse(f.name, obj[f.name])
        elif f.default is MISSING:
            raise _bad(Reason.MISSING_FIELD, f"{f.name} is required")
    message = spec.cls(**values)
    if spec.check is not None:
        spec.check(message)
    return message


def decode(text: str, *, sender: Sender) -> Message:
    """Parse one text frame from `sender`. Raises ProtocolError; never returns a dict."""
    if len(text.encode("utf-8")) > MAX_TEXT_FRAME_BYTES:
        raise _bad(Reason.TOO_LARGE, f"text frame exceeds {MAX_TEXT_FRAME_BYTES} bytes")
    try:
        obj = json.loads(text)
    except (ValueError, RecursionError):
        raise _bad(Reason.NOT_JSON, "text frame is not JSON") from None
    if not isinstance(obj, dict):
        raise _bad(Reason.NOT_OBJECT, "text frame is not a JSON object")
    kind = obj.get("t")
    if not isinstance(kind, str):
        raise _bad(Reason.MISSING_FIELD, "t is required")
    spec = MESSAGES.get(kind)
    if spec is None:
        raise ProtocolError(ErrorCode.UNKNOWN_TYPE, Reason.UNKNOWN_TYPE, kind)
    if spec.sender is not sender:
        raise ProtocolError(ErrorCode.UNKNOWN_TYPE, Reason.WRONG_DIRECTION, f"{kind} is not sent by the {sender.value}")
    return _build(spec, obj)


def encode(message: Message) -> str:
    """Serialize one message, validating it with the same rules decode applies."""
    name = _TYPE_OF[type(message)]
    obj: Dict[str, Any] = {"t": name}
    for f in fields(message):
        value = getattr(message, f.name)
        if f.default is not MISSING and value == f.default:
            continue
        obj[f.name] = f.metadata["codec"].dump(value)
    text = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    _build(MESSAGES[name], obj)
    if len(text.encode("utf-8")) > MAX_TEXT_FRAME_BYTES:
        raise _bad(Reason.TOO_LARGE, f"text frame exceeds {MAX_TEXT_FRAME_BYTES} bytes")
    return text


@dataclass(frozen=True)
class AudioFrame:
    kind: AudioKind
    stream: int
    seq: int
    pcm: bytes


_AUDIO_RULES: Dict[AudioKind, Tuple[_Codec, int]] = {
    AudioKind.MIC: (HEAR_STREAM, MAX_MIC_PCM_BYTES),
    AudioKind.SPEECH: (SPEAK_STREAM, MAX_SPEECH_PCM_BYTES),
}


def _check_audio(kind: Any, stream: int, seq: int, pcm: bytes) -> AudioFrame:
    try:
        audio_kind = AudioKind(kind)
    except ValueError:
        raise _bad(Reason.BAD_ENUM, f"audio kind {kind} is unknown") from None
    stream_codec, max_bytes = _AUDIO_RULES[audio_kind]
    stream_codec.parse("stream", stream)
    U32.parse("seq", seq)
    if not pcm:
        raise _bad(Reason.EMPTY, "audio frame has no samples")
    if len(pcm) % 2:
        raise _bad(Reason.ODD_PCM, "s16le audio needs an even byte count")
    if len(pcm) > max_bytes:
        raise _bad(Reason.TOO_LARGE, f"{audio_kind.name.lower()} audio exceeds {max_bytes} bytes")
    return AudioFrame(audio_kind, stream, seq, pcm)


def encode_audio(kind: AudioKind, stream: int, seq: int, pcm: bytes) -> bytes:
    frame = _check_audio(kind, stream, seq, pcm)
    return AUDIO_HEADER.pack(frame.kind, frame.stream, frame.seq) + frame.pcm


def decode_audio(frame: bytes) -> AudioFrame:
    if len(frame) < AUDIO_HEADER.size:
        raise _bad(Reason.SHORT_FRAME, f"audio frame shorter than {AUDIO_HEADER.size} bytes")
    kind, stream, seq = AUDIO_HEADER.unpack_from(frame)
    return _check_audio(kind, stream, seq, bytes(frame[AUDIO_HEADER.size:]))


class SequenceCheck:
    """Per-stream receive order. Every stream starts at seq 0; a skipped or repeated seq is a gap."""

    def __init__(self) -> None:
        self._next: Dict[int, int] = {}

    def accept(self, frame: AudioFrame) -> None:
        expected = self._next.get(frame.stream, 0)
        if frame.seq != expected:
            raise ProtocolError(ErrorCode.GAP, Reason.GAP, f"stream {frame.stream} expected seq {expected}, got {frame.seq}")
        self._next[frame.stream] = expected + 1

    def forget(self, stream: int) -> None:
        self._next.pop(stream, None)
