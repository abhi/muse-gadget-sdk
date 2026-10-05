# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Optional, Tuple

from musegadget.reachy_capabilities import (
    Backends, Endpoint, Expression, HeardAudio, ReplyStyle, SpokenLine,
)
from musegadget.reachy_expression import spoken_reply
from musegadget.reachy_progress import PUBLIC_PROGRESS_PHRASES, ProgressPlan

log = logging.getLogger(__name__)
ACKNOWLEDGEMENTS = (
    "Let me check the weather.", "Let me look that up.", "Let me explain that.",
    "Let me work out a plan.", "Let me work that out.", "Let me look into that trip.",
)


def contextual_acknowledgement(text: str) -> str | None:
    words = text.casefold().strip()
    request = re.search(r"\b(?:what|why|how|when|where|who|which|can|could|would|please|"
                        r"find|search|look|check|tell|explain|plan|calculate|solve|help)\b", words)
    if not request:
        return None
    if re.search(r"\b(?:weather|forecast|temperature|rain|snow)\b", words):
        return ACKNOWLEDGEMENTS[0]
    if re.search(r"\b(?:calculate|solve|multiply|divide)\b|\d\s*[+*/=]\s*\d", words):
        return ACKNOWLEDGEMENTS[4]
    if re.search(r"\b(?:trip|travel|flight|flights|hotel|hotels|itinerary)\b", words):
        return ACKNOWLEDGEMENTS[5]
    if re.search(r"\b(?:plan|planning|schedule)\b", words):
        return ACKNOWLEDGEMENTS[3]
    if re.search(r"\b(?:explain|why)\b|\bhow (?:does|do|is|are)\b", words):
        return ACKNOWLEDGEMENTS[2]
    if re.search(r"\b(?:search|find|latest|current|news)\b|\blook (?:up|for)\b", words):
        return ACKNOWLEDGEMENTS[1]
    return None


class LocalHearingTurn:
    def __init__(self, recorder, recognizer=None):
        self._recorder = recorder
        self._recognizer = recognizer
        self._turn = None
        self._failed = False

    @property
    def active(self) -> bool:
        return self._recorder.active

    @property
    def speech_active(self) -> bool:
        return getattr(self._recorder, "speech_active", self._recorder.active)

    @property
    def last_truncated(self) -> bool:
        return self._recorder.last_truncated

    def feed(self, samples):
        return self._recorder.feed(samples)

    def finish_initial_capture(self) -> None:
        self._recorder.finish_initial_capture()

    def take(self) -> HeardAudio:
        from musegadget.voice_audio import SpeechAudio, SpeechEnd

        events = self._recorder.take_audio_events() if self._recognizer is not None else ()
        recognition = None
        failures = 0
        transcript_lost = accepted_turn_lost = False
        audio = bytearray()
        for event in events:
            if isinstance(event, SpeechAudio):
                audio.extend(event.pcm)
            elif isinstance(event, SpeechEnd):
                failures += self._listen(audio)
                if self._failed:
                    transcript_lost = True
                    accepted_turn_lost = accepted_turn_lost or event.accepted
                elif self._turn is not None:
                    if event.accepted:
                        recognition = self._turn.finish()
                    else:
                        self._turn.abort()
                self._turn = None
                self._failed = False
        failures += self._listen(audio)
        return HeardAudio(recognition, failures, transcript_lost, accepted_turn_lost)

    def _listen(self, audio: bytearray) -> int:
        from musegadget.streaming_transcription import StreamingTranscriptionError, StreamingWorkerError

        if not audio or self._failed:
            audio.clear()
            return 0
        try:
            if self._turn is None:
                self._turn = self._recognizer.open_turn()
            self._turn.feed(bytes(audio))
        except StreamingWorkerError:
            raise
        except StreamingTranscriptionError:
            self.abort()
            self._failed = True
            return 1
        finally:
            audio.clear()
        return 0

    def abort(self) -> None:
        if self._turn is not None:
            self._turn.abort()
            self._turn = None


class LocalHearing:
    def __init__(self, transcriber=None):
        self.transcriber = transcriber
        self._streaming = callable(getattr(transcriber, "open_turn", None))
        self._batch = callable(getattr(transcriber, "transcribe", None))
        self._lock = None

    @property
    def transcribes(self) -> bool:
        return self.transcriber is not None

    async def start(self) -> None:
        if self._streaming:
            await self.transcriber.start()
            log.info("Reachy streaming recognition ready: audio processed before endpoint")

    def open_turn(self, sample_rate: int, *, silence_s: float, vad=None, **options) -> LocalHearingTurn:
        from musegadget.voice_audio import TurnRecorder

        recorder = TurnRecorder(sample_rate, silence_s=silence_s, max_s=60.0,
                                **({"stream_audio": True} if self._streaming else {}),
                                **({"vad": vad} if vad is not None else {}), **options)
        return LocalHearingTurn(recorder, self.transcriber if self._streaming else None)

    def recognize(self, wav: bytes) -> Optional[asyncio.Task]:
        return asyncio.create_task(self.transcribe(wav)) if self._batch else None

    async def transcribe(self, wav: bytes) -> str:
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            started = time.monotonic()
            text = await asyncio.wait_for(self.transcriber.transcribe(wav), 60)
            log.info("Speech recognition took %.1fs", time.monotonic() - started)
            return text

    async def endpoint(self, wav: bytes, recognition: Optional[asyncio.Future]) -> Optional[Endpoint]:
        """The turn as Muse receives it; None when its transcription was lost."""
        from musegadget.streaming_transcription import StreamingTranscriptionError, StreamingWorkerError

        if recognition is None:
            return Endpoint(wav, None)
        try:
            text = await recognition
        except StreamingWorkerError:
            raise
        except StreamingTranscriptionError:
            return None
        return Endpoint(wav, text)


class LocalVoice:
    """A local speech engine such as Piper, with fixed phrases synthesized in advance."""

    speaks_text = True

    def __init__(self, speech):
        self.speech = speech
        self.phrases = {}

    def stream(self, line: SpokenLine, output_rate: int):
        return self.speech.stream(line.text, output_rate)

    def prepare(self, line: SpokenLine, output_rate: int):
        if not callable(getattr(self.speech, "prepare", None)):
            return None
        return self.speech.prepare(line.text, output_rate)

    async def cache(self, phrases, output_rate: int) -> None:
        for phrase in phrases:
            stream = self.speech.stream(phrase, output_rate)
            audio = []
            try:
                async for chunk in stream:
                    audio.append(chunk)
            finally:
                await stream.aclose()
            self.phrases[phrase] = tuple(audio)

    def cached(self, text: str):
        return self.phrases.get(text)


class MuseVoice:
    speaks_text = False

    def __init__(self, session):
        self.session = session

    def stream(self, line: SpokenLine, output_rate: int):
        from musegadget.voice_audio import Mp3Decoder

        decoder = Mp3Decoder(output_rate)
        mp3 = self.session.stream_tts(line.muse_message_id)

        async def decoded():
            try:
                async for chunk in mp3:
                    for samples in decoder.feed(chunk):
                        yield samples
            finally:
                await mp3.aclose()
            for samples in decoder.finish():
                yield samples
        return decoded()

    def prepare(self, line: SpokenLine, output_rate: int):
        return None

    async def cache(self, phrases, output_rate: int) -> None:
        pass

    def cached(self, text: str):
        return None


class RuleNarrator:
    """Fixed acknowledgement and progress phrases, and Muse's reply as spoken lines."""

    progress_phrases = PUBLIC_PROGRESS_PHRASES

    def __init__(self, *, acknowledge: bool = False):
        self.acknowledgements = ACKNOWLEDGEMENTS if acknowledge else ()

    async def acknowledge(self, request: str) -> Optional[SpokenLine]:
        phrase = contextual_acknowledgement(request)
        return SpokenLine(phrase, None, "ack") if phrase in self.acknowledgements else None

    def progress(self, request: str, started: float) -> ProgressPlan:
        return ProgressPlan(request, started)

    def lines(self, reply: str, style: ReplyStyle, *,
              message_id: Optional[str] = None) -> Tuple[SpokenLine, ...]:
        if style is ReplyStyle.EXPRESSIVE_JSON:
            raise ValueError("sentence frames are parsed as they stream, by the reply tracker")
        text, expression = spoken_reply(reply)
        return (SpokenLine(text, Expression.parse(expression), "answer", message_id),)


def robot_backends(session, *, speech=None, progress_speech=None, transcriber=None,
                   stream_replies: bool = False) -> Backends:
    """One conversation's parts from the robot's installed speech models."""
    style = (ReplyStyle.EXPRESSIVE_JSON if stream_replies else
             ReplyStyle.MUSE_VOICE if speech is None else ReplyStyle.MARKER)
    return Backends(
        hearing=LocalHearing(transcriber),
        voice=MuseVoice(session) if speech is None else LocalVoice(speech),
        narrator=RuleNarrator(acknowledge=speech is not None and transcriber is not None),
        reply_style=style,
        progress_voice=None if progress_speech is None else LocalVoice(progress_speech),
    )
