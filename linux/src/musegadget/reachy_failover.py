# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Companion capabilities that fall back to Reachy's own, so a turn never waits on a dead link."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import Optional, Tuple

from musegadget.reachy_capabilities import (
    Backends, CapabilityUnavailable, HeardAudio, ReplyStyle, SpokenLine, grounded,
)
from musegadget.reachy_companion_protocol import MAX_REPLY_BYTES
from musegadget.reachy_progress import ProgressPlan

log = logging.getLogger(__name__)

REPLAY_LIMIT_S = 60.0              # audio kept for a local replay of an unfinished turn
IDLE_KEEP_S = 1.0                  # between turns, keep only this much
ACK_DEADLINE_S = 0.9
PROGRESS_DEADLINE_S = 1.5
LINES_DEADLINE_S = 4.0
SPEECH_GAP_S = 2.0                 # companion speech that pauses this long falls back
STRIKES = 3                        # consecutive failures of one companion op that rest it
COOLDOWN_S = 60.0                  # how long a rested op stays on the robot before the companion is tried again


class _Rest:
    """One companion op's failures in a row; ``STRIKES`` of them rest the op for ``COOLDOWN_S``."""

    def __init__(self, op: str):
        self._op = op
        self._strikes = 0
        self._until = 0.0

    @property
    def resting(self) -> bool:
        return time.monotonic() < self._until

    def fail(self, reason: str) -> None:
        self._strikes += 1
        log.warning("Reachy used its own %s: %s (%d in a row)", self._op, reason, self._strikes)
        if self._strikes >= STRIKES:
            self._strikes = 0
            self._until = time.monotonic() + COOLDOWN_S
            log.warning("Reachy rests its companion's %s for %.0f s", self._op, COOLDOWN_S)

    def succeed(self) -> None:
        self._strikes = 0


class CompanionRoute:
    """Whether the companion voices and narrates the current turn, decided once at its start.

    A link that comes back mid-turn waits for the next turn, so one answer never switches
    voices, and an op that keeps failing rests on its own while the link stays up.
    """

    def __init__(self, link):
        self.link = link
        self.this_turn = False
        self.hearing = _Rest("hearing")
        self.speech = _Rest("voice")
        self.narration = _Rest("narration")

    def begin_turn(self) -> bool:
        self.this_turn = self.link.up
        return self.this_turn

    def take_outage(self) -> bool:
        return self.link.take_outage()


class FailoverHearingTurn:
    """One listening window on the companion, finished locally if the companion fails.

    While the companion hears, fed audio is kept from the end of its last ended turn. If it
    fails mid-turn, a local turn opens and the kept audio replays into it at the next feed,
    which runs off the event loop. The local turn hands back to the companion between user
    turns once the link is up.
    """

    def __init__(self, hearing: FailoverHearing, sample_rate: int, options: dict):
        self._hearing = hearing
        self._rate = sample_rate
        self._options = options
        self._kept = deque()
        self._kept_samples = 0
        self._fed = 0
        self._turn_start_sample = 0
        self._replay = None
        self._on_companion = hearing.companion_ready
        self._turn = (hearing.companion if self._on_companion else hearing.local).open_turn(sample_rate, **options)

    @property
    def active(self) -> bool:
        return self._replay is not None or self._turn.active

    @property
    def speech_active(self) -> bool:
        return self._turn.speech_active

    @property
    def partial(self):
        return self._turn.partial

    def feed(self, samples) -> None:
        self._fed += len(samples)
        if self._replay is not None:
            import numpy as np
            samples, self._replay = np.concatenate([*self._replay, samples]), None
        elif self._on_companion:
            self._keep(samples)
        self._turn.feed(samples)

    def _keep(self, samples) -> None:
        self._kept.append(samples.copy())
        self._kept_samples += len(samples)
        self._trim(REPLAY_LIMIT_S)

    def _trim(self, seconds: float) -> None:
        while len(self._kept) > 1 and self._kept_samples - len(self._kept[0]) >= seconds * self._rate:
            self._kept_samples -= len(self._kept.popleft())

    def _forget_until(self, sample: int) -> None:
        """Drop kept audio before ``sample``, counted in samples fed to this window."""
        excess = sample - (self._fed - self._kept_samples)
        while excess > 0 and self._kept:
            oldest = self._kept[0]
            if len(oldest) <= excess:
                self._kept.popleft()
            else:
                self._kept[0] = oldest[excess:]
            dropped = min(len(oldest), excess)
            self._kept_samples -= dropped
            excess -= dropped

    def take(self) -> HeardAudio:
        if not self._on_companion:
            heard = self._turn.take()
            if (self._hearing.companion_ready and heard.ended is None and self._replay is None
                    and not self._turn.active):
                self._switch(on_companion=True)
            return heard
        try:
            heard = self._turn.take()
        except CapabilityUnavailable as error:
            log.warning("Reachy's companion stopped hearing (%s); finishing the turn on the robot", error)
            if not error.link_down:
                self._hearing.rest.fail(str(error))
            self._replay = tuple(self._kept)
            self._switch(on_companion=False)
            return HeardAudio()
        if heard.ended is not None:
            self._hearing.rest.succeed()
            self._forget_until(self._turn_start_sample + heard.ended.end_sample)
        elif not self._turn.active:
            self._trim(IDLE_KEEP_S)
        return heard

    def _switch(self, *, on_companion: bool) -> None:
        self._turn.abort()
        self._kept.clear()
        self._kept_samples = 0
        self._on_companion = on_companion
        self._turn_start_sample = self._fed
        hearing = self._hearing.companion if on_companion else self._hearing.local
        self._turn = hearing.open_turn(self._rate, **self._options)

    def finish_initial_capture(self) -> None:
        self._turn.finish_initial_capture()

    def abort(self) -> None:
        self._turn.abort()


class FailoverHearing:
    def __init__(self, companion, local, rest: _Rest):
        self.companion = companion
        self.local = local
        self.rest = rest

    @property
    def companion_ready(self) -> bool:
        return self.companion.available and not self.rest.resting

    @property
    def transcribes(self) -> bool:
        return self.local.transcribes

    async def start(self) -> None:
        await self.local.start()
        await self.companion.start()

    def open_turn(self, sample_rate: int, **options) -> FailoverHearingTurn:
        return FailoverHearingTurn(self, sample_rate, options)

    async def transcribe(self, wav: bytes) -> str:
        return await self.local.transcribe(wav)


class FailoverVoice:
    """Companion speech; on failure the whole line is spoken again by the local voice."""

    speaks_text = True

    def __init__(self, companion, local, route: CompanionRoute):
        self.companion = companion
        self.local = local
        self.route = route

    def _on_companion(self, line: SpokenLine) -> bool:
        return (self.route.this_turn and self.companion.available and not self.route.speech.resting
                and not self.local.is_presynthesized(line.text))

    async def warm(self, output_rate: int) -> None:
        await self.local.warm(output_rate)

    def is_presynthesized(self, text: str) -> bool:
        return self.local.is_presynthesized(text)

    def prepare(self, line: SpokenLine, output_rate: int):
        if self._on_companion(line):
            return None
        return self.local.prepare(line, output_rate)

    async def stream(self, line: SpokenLine, output_rate: int):
        if self._on_companion(line):
            speech = self.companion.stream(line, output_rate)
            spoke = False
            try:
                while True:
                    try:
                        chunk = await asyncio.wait_for(speech.__anext__(), SPEECH_GAP_S)
                    except StopAsyncIteration:
                        self.route.speech.succeed()
                        return
                    spoke = True
                    yield chunk
            except (CapabilityUnavailable, asyncio.TimeoutError) as error:
                reason = (f"companion speech stopped ({type(error).__name__}); "
                          + ("repeating the line" if spoke else "speaking it"))
                if isinstance(error, CapabilityUnavailable) and error.link_down:
                    log.warning("Reachy's %s", reason)
                else:
                    self.route.speech.fail(reason)
            finally:
                await speech.aclose()
        speech = self.local.stream(line, output_rate)
        try:
            async for chunk in speech:
                yield chunk
        finally:
            await speech.aclose()


class FailoverNarrator:
    """The companion narrates within deadlines; Reachy's rules answer whenever it can't."""

    def __init__(self, companion, local, route: CompanionRoute):
        self.companion = companion
        self.local = local
        self.route = route

    @property
    def _on_companion(self) -> bool:
        return self.route.this_turn and self.companion.available and not self.route.narration.resting

    async def _ask(self, call, deadline_s: float, sources: Tuple[str, ...]) -> Optional[Tuple[SpokenLine, ...]]:
        """The companion's lines, possibly none, or None when they are late, failed or ungrounded."""
        try:
            result = await asyncio.wait_for(call, deadline_s)
        except CapabilityUnavailable as error:
            if not error.link_down:
                self.route.narration.fail(str(error))
            return None
        except asyncio.TimeoutError:
            self.route.narration.fail("narration timed out")
            return None
        lines = result if isinstance(result, tuple) else (() if result is None else (result,))
        if not all(grounded(line.text, *sources) for line in lines):
            self.route.narration.fail("narration added facts that are not in the request or reply")
            return None
        self.route.narration.succeed()
        return lines

    async def acknowledge(self, request: str) -> Optional[SpokenLine]:
        if self._on_companion:
            lines = await self._ask(self.companion.acknowledge(request), ACK_DEADLINE_S, (request,))
            if lines is not None:
                # A healthy companion that chose no acknowledgement means silence, not a miss.
                return lines[0] if lines else None
        return await self.local.acknowledge(request)

    def progress(self, request: str, started: float) -> ProgressPlan:
        return self.local.progress(request, started)

    async def say_progress(self, request: str, status: str,
                           already_said: Tuple[str, ...]) -> Optional[SpokenLine]:
        if self._on_companion:
            lines = await self._ask(self.companion.say_progress(request, status, already_said),
                                    PROGRESS_DEADLINE_S, (status, request))
            if lines:
                return lines[0]
        return await self.local.say_progress(request, status, already_said)

    async def lines(self, request: str, reply: str, style: ReplyStyle, *,
                    message_id: Optional[str] = None) -> Tuple[SpokenLine, ...]:
        local = await self.local.lines(request, reply, style, message_id=message_id)
        if style is ReplyStyle.PLAIN_SHORT and local and self._on_companion:
            spoken = " ".join(line.text for line in local)
            if len(spoken.encode("utf-8")) > MAX_REPLY_BYTES:
                log.info("Reachy speaks a %d-byte reply itself; the companion narrates at most %d",
                         len(spoken.encode("utf-8")), MAX_REPLY_BYTES)
                self.route.this_turn = False
                return local
            result = await self._ask(self.companion.lines(request, spoken, style), LINES_DEADLINE_S,
                                     (request, spoken))
            if result:
                return result
        return local


def companion_backends(local: Backends, link) -> Backends:
    """Mode 3: the companion's parts, each backed by the on-robot part of mode 2."""
    from musegadget.reachy_companion_client import (
        CompanionHearing, CompanionNarrator, CompanionVoice, VoiceRole,
    )

    route = CompanionRoute(link)
    progress = (None if local.progress_voice is None
                else FailoverVoice(CompanionVoice(link, VoiceRole.PROGRESS), local.progress_voice, route))
    return Backends(
        hearing=FailoverHearing(CompanionHearing(link), local.hearing, route.hearing),
        voice=FailoverVoice(CompanionVoice(link), local.voice, route),
        narrator=FailoverNarrator(CompanionNarrator(link), local.narrator, route),
        local_style=local.local_style,
        progress_voice=progress,
        companion=route,
    )
