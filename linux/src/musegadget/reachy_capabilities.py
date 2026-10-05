# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, AsyncIterator, Iterable, Optional, Protocol, Tuple

if TYPE_CHECKING:
    from typing import Literal

    import numpy as np

    from musegadget.reachy_progress import ProgressPlan

    SpeechRole = Literal["ack", "progress", "answer", "notice"]


class Expression(str, Enum):
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    SURPRISED = "surprised"
    CURIOUS = "curious"
    NOD = "nod"
    SHAKE = "shake"
    LISTENING = "listening"
    THINKING = "thinking"

    @classmethod
    def parse(cls, name: object) -> Optional[Expression]:
        """Unknown names mean no gesture; parsing never raises."""
        try:
            return cls(name)
        except ValueError:
            return None


class ReplyStyle(Enum):
    """The Muse prompt a request was sent with, which also picks its reply parser."""

    MUSE_VOICE = "muse_voice"      # Muse speaks; a trailing [reachy:NAME] picks the expression
    MARKER = "marker"              # local speech of a reply with a trailing [reachy:NAME]
    EXPRESSIVE_JSON = "json"       # local speech of one JSON sentence frame per line


@dataclass(frozen=True)
class Partial:
    text: str
    revision: int


@dataclass(frozen=True)
class Endpoint:
    """One finished user turn. Without text, Muse receives the audio itself."""

    audio: bytes                   # the turn as a 16 kHz mono PCM16 WAV
    text: Optional[str]            # None: not transcribed here; "": nothing intelligible
    forced: bool = False


@dataclass(frozen=True)
class SpokenLine:
    text: str
    expression: Optional[Expression]
    role: SpeechRole
    muse_message_id: Optional[str] = None


@dataclass(frozen=True)
class HeardAudio:
    recognition: Optional[asyncio.Future] = None
    failures: int = 0              # recognizer failures; each one owes the user a retry notice
    transcript_lost: bool = False
    dropped: bool = False


class HearingTurn(Protocol):
    """One endpointed recording. ``feed`` may run off the event loop; ``take`` may not."""

    active: bool
    speech_active: bool
    last_truncated: bool

    def feed(self, samples: np.ndarray) -> Optional[bytes]: ...
    def take(self) -> HeardAudio: ...
    def finish_initial_capture(self) -> None: ...
    def abort(self) -> None: ...


class Hearing(Protocol):
    transcribes: bool

    async def start(self) -> None: ...
    def open_turn(self, sample_rate: int, *, silence_s: float, vad: object = None,
                  **options) -> HearingTurn: ...
    def recognize(self, wav: bytes) -> Optional[asyncio.Future]: ...
    async def transcribe(self, wav: bytes) -> str: ...
    async def endpoint(self, wav: bytes, recognition: Optional[asyncio.Future]) -> Optional[Endpoint]: ...


class PreparedVoice(Protocol):
    """Speech synthesized ahead of playback, as float32 mono at the requested rate."""

    def __aiter__(self) -> AsyncIterator[np.ndarray]: ...
    def cancel(self) -> None: ...
    async def aclose(self) -> None: ...


class Voice(Protocol):
    speaks_text: bool              # False: speaks only Muse's own audio for a Muse message

    def stream(self, line: SpokenLine, output_rate: int) -> AsyncIterator[np.ndarray]: ...
    def prepare(self, line: SpokenLine, output_rate: int) -> Optional[PreparedVoice]: ...
    async def cache(self, phrases: Iterable[str], output_rate: int) -> None: ...
    def cached(self, text: str) -> Optional[Tuple[np.ndarray, ...]]: ...


class Narrator(Protocol):
    acknowledgements: Tuple[str, ...]
    progress_phrases: Tuple[str, ...]

    async def acknowledge(self, request: str) -> Optional[SpokenLine]: ...
    def progress(self, request: str, started: float) -> ProgressPlan: ...
    def lines(self, reply: str, style: ReplyStyle, *,
              message_id: Optional[str] = None) -> Tuple[SpokenLine, ...]: ...


@dataclass(frozen=True)
class Backends:
    hearing: Hearing
    voice: Voice
    narrator: Narrator
    reply_style: ReplyStyle
    progress_voice: Optional[Voice] = None

    def __post_init__(self) -> None:
        if not self.voice.speaks_text and self.reply_style is not ReplyStyle.MUSE_VOICE:
            raise ValueError("a text reply style requires local speech")
        if self.voice.speaks_text and self.reply_style is ReplyStyle.MUSE_VOICE:
            raise ValueError("Muse voice replies require Muse speech")
