# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Callable, Dict, Optional, Tuple, Union

from musegadget.reachy_capabilities import Expression, Partial
from musegadget.reachy_hardware import GESTURE_DURATION_S


TILT_WORDS = 3
GESTURE_GAP_S = GESTURE_DURATION_S
NOD_EVERY_S = 3.0
CLAUSE_PAUSE_S = 0.6
CLAUSE_WORDS = ("and", "so")
VARY_EVERY_S = 4.0


class State(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"


@dataclass(frozen=True)
class SetState:
    state: State
    expression: Optional[Expression] = None


@dataclass(frozen=True)
class Gesture:
    name: str


NOD = "nod"
TILT = "tilt"

Cue = Union[SetState, Gesture]


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
class TurnStarted:
    """A new request begins; per-request body language starts over."""

    output: bool
    user_speaking: bool


@dataclass(frozen=True)
class Heard:
    """The open utterance's transcript, observed on every microphone chunk while the user talks.

    ``speech_active`` is the recorder's voice activity: false means the user is silent right now.
    ``output`` is whether Reachy has speech playing or queued.
    """

    partial: Partial
    speech_active: bool
    output: bool


@dataclass(frozen=True)
class MuseStatus:
    """Muse's latest reported phase and allowlisted activity, observed while Reachy waits."""

    phase: str
    activity: Optional[str]
    output: bool


@dataclass(frozen=True)
class AnswerArrived:
    """Muse's first reply line is ready to speak."""

    output: bool


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
              TurnStarted, Heard, MuseStatus, AnswerArrived, TurnDone, OutputIdle]


class ExpressionPlanner:
    def __init__(self) -> None:
        self._state: Optional[State] = None
        self._gesture_at = -math.inf
        self._nod_at = -math.inf
        self._utterance_id: Optional[int] = None
        self._words = 0
        self._words_at_gesture = 0
        self._quiet_since: Optional[float] = None
        self._nodded: Optional[Tuple[int, int]] = None
        self._answered = False
        self._status_shown: Optional[Tuple[str, Optional[str]]] = None
        self._varied_at = -math.inf
        self._variations = 0
        self._thinking_expression: Optional[Expression] = None
        self._handlers: Dict[type, Callable[..., Tuple[Cue, ...]]] = {
            Started: self._started,
            WakeHeard: self._wake_heard,
            WakeClosed: self._wake_closed,
            CaptureGap: self._capture_gap,
            Recording: self._recording,
            UserSpeech: self._user_speech,
            Working: self._working,
            TurnStarted: self._turn_started,
            Heard: self._heard,
            MuseStatus: self._muse_status,
            AnswerArrived: self._answer_arrived,
            TurnDone: self._turn_done,
            OutputIdle: self._output_idle,
        }

    def on(self, event: Event, now: float) -> Tuple[Cue, ...]:
        cues = self._handlers[type(event)](event, now)
        for cue in cues:
            if isinstance(cue, SetState):
                self._state = cue.state
        return cues

    def resting(self, state: State) -> SetState:
        """The cue that holds ``state``; thinking keeps the lean Muse's work last chose."""
        return SetState(state, self._thinking_expression if state is State.THINKING else None)

    def _gesture(self, name: str, now: float) -> Tuple[Cue, ...]:
        self._gesture_at = now
        self._words_at_gesture = self._words
        return (Gesture(name),)

    def _started(self, event: Started, now: float) -> Tuple[Cue, ...]:
        return (SetState(State.IDLE),)

    def _wake_heard(self, event: WakeHeard, now: float) -> Tuple[Cue, ...]:
        self._gesture_at = now
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
            return (self.resting(State.THINKING),)
        return ()

    def _working(self, event: Working, now: float) -> Tuple[Cue, ...]:
        if event.output or event.user_speaking:
            return ()
        return (self.resting(State.THINKING),)

    def _turn_started(self, event: TurnStarted, now: float) -> Tuple[Cue, ...]:
        self._answered = False
        self._status_shown = None
        self._varied_at = -math.inf
        self._variations = 0
        self._thinking_expression = None
        return self._working(Working(event.output, event.user_speaking), now)

    def _heard(self, event: Heard, now: float) -> Tuple[Cue, ...]:
        partial = event.partial
        words = partial.text.split()
        if partial.utterance_id != self._utterance_id:
            self._utterance_id = partial.utterance_id
            self._words_at_gesture = 0
            self._quiet_since = None
        self._words = len(words)
        self._words_at_gesture = min(self._words_at_gesture, self._words)
        if event.speech_active:
            self._quiet_since = None
        elif self._quiet_since is None:
            self._quiet_since = now
        if event.output or self._state is not State.LISTENING or now - self._gesture_at < GESTURE_GAP_S:
            return ()
        paused = self._quiet_since is not None and now - self._quiet_since >= CLAUSE_PAUSE_S
        clause_end = partial.text.endswith(",") or (bool(words) and words[-1].casefold() in CLAUSE_WORDS) or paused
        heard = (partial.utterance_id, partial.revision)
        if clause_end and heard != self._nodded and now - self._nod_at >= NOD_EVERY_S:
            self._nodded = heard
            self._nod_at = now
            return self._gesture(NOD, now)
        if self._words - self._words_at_gesture >= TILT_WORDS:
            return self._gesture(TILT, now)
        return ()

    def _muse_status(self, event: MuseStatus, now: float) -> Tuple[Cue, ...]:
        shown = (event.phase, event.activity)
        if (shown == self._status_shown or event.output or self._state is not State.THINKING
                or now - self._varied_at < VARY_EVERY_S):
            return ()
        self._status_shown = shown
        self._varied_at = now
        self._variations += 1
        self._thinking_expression = Expression.CURIOUS if self._variations % 2 else Expression.THINKING
        return (self.resting(State.THINKING),)

    def _answer_arrived(self, event: AnswerArrived, now: float) -> Tuple[Cue, ...]:
        if self._answered:
            return ()
        self._answered = True
        if event.output or self._state is not State.THINKING or now - self._gesture_at < GESTURE_GAP_S:
            return ()
        self._nod_at = now
        return self._gesture(NOD, now)

    def _turn_done(self, event: TurnDone, now: float) -> Tuple[Cue, ...]:
        if event.turns_waiting or event.output:
            return ()
        return (SetState(State.LISTENING if event.recording or event.wake_open else State.IDLE),)

    def _output_idle(self, event: OutputIdle, now: float) -> Tuple[Cue, ...]:
        if event.user_speaking:
            return (SetState(State.LISTENING),)
        if event.turn_open:
            return (self.resting(State.THINKING),)
        return (SetState(State.LISTENING if event.wake_open else State.IDLE),)
