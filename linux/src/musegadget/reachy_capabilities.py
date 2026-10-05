# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, AsyncIterator, Optional, Protocol, Tuple

if TYPE_CHECKING:
    from typing import Literal

    import numpy as np

    from musegadget.reachy_progress import ProgressPlan

    SpeechRole = Literal["ack", "progress", "answer", "notice"]


WAKE_CUE = "Yes?"


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
    MUSE_VOICE = "muse_voice"
    MARKER = "marker"
    EXPRESSIVE_JSON = "json"


class Mode(Enum):
    MUSE_VOICE = "muse-voice"
    ON_ROBOT = "on-robot"


@dataclass(frozen=True)
class Partial:
    """One utterance's streaming transcript so far; ``revision`` rises with each change of text."""

    text: str
    revision: int
    utterance_id: int              # a new id for every utterance, even one repeating the last


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


class EndedTurn(Protocol):
    """One user turn whose end the hearing decided; its transcript may still be on the way."""

    truncated: bool                # the turn hit the 60-second recording cap

    def start(self) -> None: ...   # begin recognition now, if it waits for the caller; idempotent
    def cancel(self) -> None: ...
    async def endpoint(self) -> Optional[Endpoint]: ...   # None: its transcription was lost


@dataclass(frozen=True)
class HeardAudio:
    """What one feed told the caller."""

    ended: Optional[EndedTurn] = None   # at most one ended turn per take; feed nothing to drain the next
    failures: int = 0              # recognizer failures; each one owes the user a retry notice
    transcript_lost: bool = False  # an utterance ended without a usable transcript


class HearingTurn(Protocol):
    """One listening window, which may hold several user turns.

    The robot decides when the window opens and closes; the implementation decides where
    each user turn inside it ends. ``feed`` may run off the event loop; ``take`` may not.
    """

    active: bool                   # inside a user turn that has not ended yet
    speech_active: bool
    partial: Optional[Partial]     # the open utterance's transcript so far; never sent to Muse

    def feed(self, samples: np.ndarray) -> None: ...
    def take(self) -> HeardAudio: ...
    def finish_initial_capture(self) -> None: ...
    def abort(self) -> None: ...


class Hearing(Protocol):
    transcribes: bool

    async def start(self) -> None: ...
    def open_turn(self, sample_rate: int, *, silence_s: float, vad: object = None,
                  **options) -> HearingTurn: ...
    async def transcribe(self, wav: bytes) -> str: ...


class PreparedVoice(Protocol):
    """Speech synthesized ahead of playback, as float32 mono at the requested rate."""

    def __aiter__(self) -> AsyncIterator[np.ndarray]: ...
    def cancel(self) -> None: ...
    async def aclose(self) -> None: ...


class Voice(Protocol):
    speaks_text: bool              # False: speaks only Muse's own audio for a Muse message

    async def warm(self, output_rate: int) -> None: ...
    def stream(self, line: SpokenLine, output_rate: int) -> AsyncIterator[np.ndarray]: ...
    def prepare(self, line: SpokenLine, output_rate: int) -> Optional[PreparedVoice]: ...


class Narrator(Protocol):
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
