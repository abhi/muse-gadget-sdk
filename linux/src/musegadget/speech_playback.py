# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Pause speech by retaining unsent PCM, without flushing the speaker pipeline."""

from __future__ import annotations

import asyncio
import math
import time
from collections import deque
from typing import Awaitable, Callable


_TIMING_WINDOW = 512


def _percentiles_ms(samples: deque[float]) -> tuple[float, float]:
    if not samples:
        return 0.0, 0.0
    ordered = sorted(samples)
    values = []
    for quantile in (0.5, 0.95):
        position = (len(ordered) - 1) * quantile
        lower, upper = math.floor(position), math.ceil(position)
        values.append((ordered[lower] + (ordered[upper] - ordered[lower])
                       * (position - lower)) * 1000)
    return values[0], values[1]


class SpeechPlayback:
    """A shared output gate, updated on the asyncio loop by authorized input VAD.

    The cumulative pause clock includes an ongoing hold. A timeout owner can
    subtract the change since it started, without waiting for the user to stop.
    Updates from a capture thread must use the loop's ``call_soon_threadsafe``.
    """

    def __init__(self, *, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._quiet = asyncio.Event()
        self._quiet.set()
        self._pause_started: float | None = None
        self._paused_s = 0.0

    @property
    def user_speaking(self) -> bool:
        return self._pause_started is not None

    @property
    def paused_s(self) -> float:
        ongoing = 0 if self._pause_started is None else self._clock() - self._pause_started
        return self._paused_s + max(0.0, ongoing)

    def set_user_speaking(self, speaking: bool) -> None:
        if type(speaking) is not bool:
            raise TypeError("User speaking must be a boolean")
        if speaking == self.user_speaking:
            return
        if speaking:
            self._pause_started = self._clock()
            self._quiet.clear()
        else:
            self._paused_s = self.paused_s
            self._pause_started = None
            self._quiet.set()

    async def wait_until_quiet(self) -> float:
        """Wait without blocking input, synthesis, or Muse; return time waited."""
        started = self._clock()
        while self.user_speaking:
            await self._quiet.wait()
        return max(0.0, self._clock() - started)


class PCMPlayer:
    """One sentence's PCM cursor, owned by a single sequential output task.

    ``push`` must await acceptance of one chunk and finish any physical write
    before propagating cancellation. Chunks already accepted are drained on a
    pause; all remaining samples stay in the caller's stream and current cursor.
    The player never flushes, changes the SDK pipeline state, or repeats audio.

    Software queue lead is bounded by ``lead_s + chunk_s``. ``drain_margin_s``
    allows for sink buffering; it is an estimate, not a DAC consumption cursor.
    Callbacks are async: start once before the first push, pause after accepted
    audio drains, and resume before the next push of the same sentence.
    """

    def __init__(
        self,
        sample_rate: int,
        push: Callable[[object], Awaitable[None]],
        gate: SpeechPlayback,
        *,
        on_start: Callable[[], Awaitable[None]] | None = None,
        on_pause: Callable[[], Awaitable[None]] | None = None,
        on_resume: Callable[[], Awaitable[None]] | None = None,
        chunk_s: float = 0.04,
        lead_s: float = 0.04,
        drain_margin_s: float = 0.05,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        diagnostics: bool = False,
    ):
        if type(sample_rate) is not int or sample_rate <= 0:
            raise ValueError("Speaker sample rate must be a positive integer")
        if (not all(math.isfinite(value) for value in (chunk_s, lead_s, drain_margin_s))
                or chunk_s <= 0 or lead_s < 0 or drain_margin_s < 0):
            raise ValueError("Speaker pacing durations must be finite and nonnegative")
        self._rate = sample_rate
        self._chunk_samples = max(1, math.floor(chunk_s * sample_rate))
        self._lead_s = lead_s
        self._drain_margin_s = drain_margin_s
        self._push = push
        self._gate = gate
        self._on_start, self._on_pause, self._on_resume = on_start, on_pause, on_resume
        self._clock, self._sleep = clock, sleep
        self._end = clock()
        self._started = False
        self._paused = False
        self._samples_sent = 0
        self._pause_baseline = gate.paused_s
        self._diagnostics = {
            "play_calls": 0, "input_frames": 0, "play_wall_s": 0.0,
            "drain_wall_s": 0.0, "push_count": 0, "push_wait_ms_max": 0.0,
            "scheduler_wake_count": 0, "scheduler_wake_lateness_ms_max": 0.0,
            "source_gap_ms_max": 0.0,
        } if diagnostics else None
        self._push_wait_s: deque[float] = deque(maxlen=_TIMING_WINDOW)
        self._wake_lateness_s: deque[float] = deque(maxlen=_TIMING_WINDOW)
        self._last_play_end: float | None = None
        self._last_play_pause_s = 0.0
        self._between_play_drain_s = 0.0

    @property
    def paused_s(self) -> float:
        return max(0.0, self._gate.paused_s - self._pause_baseline)

    @property
    def metrics(self) -> dict | None:
        """Return an opt-in snapshot containing timing and counts only.

        Push waits include the entire awaited callback, including failed or
        cancelled attempts; pushed frames count successful returns. Sleep
        lateness measures delay after the scheduled wake, not preexisting debt.
        Percentiles use the latest 512 observations; counts and maxima cover
        the player's lifetime. Producer gaps exclude user holds and explicit
        finish drains. These software timings do not measure DAC consumption.
        """
        if self._diagnostics is None:
            return None
        result = dict(self._diagnostics)
        push_p50, push_p95 = _percentiles_ms(self._push_wait_s)
        wake_p50, wake_p95 = _percentiles_ms(self._wake_lateness_s)
        result.update({
            "input_audio_s": result["input_frames"] / self._rate,
            "pushed_frames": self._samples_sent,
            "pushed_audio_s": self._samples_sent / self._rate,
            "push_wait_ms_p50": push_p50, "push_wait_ms_p95": push_p95,
            "scheduler_wake_lateness_ms_p50": wake_p50,
            "scheduler_wake_lateness_ms_p95": wake_p95,
            "percentile_window": _TIMING_WINDOW,
            "push_wait_window_count": len(self._push_wait_s),
            "scheduler_wake_window_count": len(self._wake_lateness_s),
            "paused_s": self.paused_s,
        })
        return result

    async def _sleep_until(self, target: float) -> None:
        started = self._clock()
        await self._sleep(max(0.0, target - started))
        if self._diagnostics is not None:
            lateness = max(0.0, self._clock() - max(started, target))
            self._wake_lateness_s.append(lateness)
            self._diagnostics["scheduler_wake_count"] += 1
            self._diagnostics["scheduler_wake_lateness_ms_max"] = max(
                self._diagnostics["scheduler_wake_lateness_ms_max"], lateness * 1000)

    async def _drain(self) -> None:
        if self._samples_sent:
            await self._sleep_until(self._end + self._drain_margin_s)

    async def _pause(self) -> None:
        if self._started and not self._paused:
            self._paused = True
            if self._on_pause is not None:
                await self._on_pause()

    async def _ready(self) -> None:
        while True:
            if self._gate.user_speaking:
                await self._drain()
                if self._gate.user_speaking:
                    await self._pause()
                    await self._gate.wait_until_quiet()
            if not self._started:
                self._started = True
                if self._on_start is not None:
                    await self._on_start()
            elif self._paused:
                self._paused = False
                if self._on_resume is not None:
                    await self._on_resume()
            # Pose callbacks can await hardware while another VAD update arrives.
            if not self._gate.user_speaking:
                return

    async def play(self, samples) -> None:
        """Accept an arbitrary PCM chunk and preserve its order across pauses."""
        metrics = self._diagnostics
        if metrics is not None:
            started = self._clock()
            metrics["play_calls"] += 1
            metrics["input_frames"] += len(samples)
            if self._last_play_end is not None:
                held = max(0.0, self._gate.paused_s - self._last_play_pause_s)
                gap = max(0.0, started - self._last_play_end - held
                          - self._between_play_drain_s)
                metrics["source_gap_ms_max"] = max(metrics["source_gap_ms_max"], gap * 1000)
            self._between_play_drain_s = 0.0
        try:
            for offset in range(0, len(samples), self._chunk_samples):
                chunk = samples[offset:offset + self._chunk_samples]
                await self._ready()
                if metrics is None:
                    await self._push(chunk)
                else:
                    push_started = self._clock()
                    try:
                        await self._push(chunk)
                    finally:
                        waited = max(0.0, self._clock() - push_started)
                        self._push_wait_s.append(waited)
                        metrics["push_count"] += 1
                        metrics["push_wait_ms_max"] = max(metrics["push_wait_ms_max"], waited * 1000)
                self._samples_sent += len(chunk)
                self._end = max(self._end, self._clock()) + len(chunk) / self._rate
                await self._sleep_until(self._end - self._lead_s)
        finally:
            if metrics is not None:
                self._last_play_end = self._clock()
                self._last_play_pause_s = self._gate.paused_s
                metrics["play_wall_s"] += max(0.0, self._last_play_end - started)

    async def finish(self) -> None:
        """Drain accepted audio; there is no unsent tail to wait for quiet on."""
        if self._diagnostics is not None:
            started, held_before = self._clock(), self._gate.paused_s
        try:
            await self._drain()
            if self._gate.user_speaking:
                await self._pause()
        finally:
            if self._diagnostics is not None:
                elapsed = max(0.0, self._clock() - started)
                held = max(0.0, self._gate.paused_s - held_before)
                self._diagnostics["drain_wall_s"] += elapsed
                self._between_play_drain_s += max(0.0, elapsed - held)
