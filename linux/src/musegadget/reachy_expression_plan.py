# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, Optional, Tuple, Union

from musegadget.reachy_capabilities import Expression


class State(str, Enum):
    """The resting states the planner may put Reachy in; speech and errors belong elsewhere."""

    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"


@dataclass(frozen=True)
class SetState:
    state: State
    expression: Optional[Expression] = None


Cue = SetState


@dataclass(frozen=True)
class Started:
    """The microphone loop is ready."""


@dataclass(frozen=True)
class WakeHeard:
    pass


@dataclass(frozen=True)
class WakeClosed:
    """The wake window expired with nothing said."""


@dataclass(frozen=True)
class CaptureGap:
    """Microphone audio lost continuity and the speech detectors were reset."""

    wake_open: bool
    muted: bool
    turn_running: bool


@dataclass(frozen=True)
class Recording:
    """Without a wake word, the recorder started or stopped capturing a turn."""

    active: bool
    turn_running: bool


@dataclass(frozen=True)
class UserSpeech:
    speaking: bool
    output: bool
    turn_running: bool
    turns_waiting: bool


@dataclass(frozen=True)
class Working:
    """Reachy is handling a request: dequeued, started, transcribed, or awaiting a sentence."""

    output: bool
    user_speaking: bool


@dataclass(frozen=True)
class TurnDone:
    turns_waiting: bool
    output: bool
    recording: bool
    wake_open: bool


@dataclass(frozen=True)
class OutputIdle:
    """The speaker finished its queue and nothing else is ready to play."""

    user_speaking: bool
    turn_open: bool
    wake_open: bool


Event = Union[Started, WakeHeard, WakeClosed, CaptureGap, Recording, UserSpeech, Working,
              TurnDone, OutputIdle]


class ExpressionPlanner:
    def __init__(self) -> None:
        self._handlers: Dict[type, Callable[..., Tuple[Cue, ...]]] = {
            Started: self._started,
            WakeHeard: self._wake_heard,
            WakeClosed: self._wake_closed,
            CaptureGap: self._capture_gap,
            Recording: self._recording,
            UserSpeech: self._user_speech,
            Working: self._working,
            TurnDone: self._turn_done,
            OutputIdle: self._output_idle,
        }

    def on(self, event: Event, now: float) -> Tuple[Cue, ...]:
        return self._handlers[type(event)](event, now)

    def _started(self, event: Started, now: float) -> Tuple[Cue, ...]:
        return (SetState(State.IDLE),)

    def _wake_heard(self, event: WakeHeard, now: float) -> Tuple[Cue, ...]:
        return (SetState(State.LISTENING, Expression.HAPPY),)

    def _wake_closed(self, event: WakeClosed, now: float) -> Tuple[Cue, ...]:
        return (SetState(State.IDLE),)

    def _capture_gap(self, event: CaptureGap, now: float) -> Tuple[Cue, ...]:
        if event.wake_open or event.muted or event.turn_running:
            return ()
        return (SetState(State.IDLE),)

    def _recording(self, event: Recording, now: float) -> Tuple[Cue, ...]:
        if event.turn_running:
            return ()
        return (SetState(State.LISTENING if event.active else State.IDLE),)

    def _user_speech(self, event: UserSpeech, now: float) -> Tuple[Cue, ...]:
        if event.speaking:
            return (SetState(State.LISTENING),) if event.output or event.turn_running else ()
        if not event.output and (event.turn_running or event.turns_waiting):
            return (SetState(State.THINKING),)
        return ()

    def _working(self, event: Working, now: float) -> Tuple[Cue, ...]:
        if event.output or event.user_speaking:
            return ()
        return (SetState(State.THINKING),)

    def _turn_done(self, event: TurnDone, now: float) -> Tuple[Cue, ...]:
        if event.turns_waiting or event.output:
            return ()
        return (SetState(State.LISTENING if event.recording or event.wake_open else State.IDLE),)

    def _output_idle(self, event: OutputIdle, now: float) -> Tuple[Cue, ...]:
        if event.user_speaking:
            return (SetState(State.LISTENING),)
        if event.turn_open:
            return (SetState(State.THINKING),)
        return (SetState(State.LISTENING if event.wake_open else State.IDLE),)
