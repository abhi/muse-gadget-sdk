# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Bounded, turn-scoped spoken progress without inventing backend activity."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal


ProgressSource = Literal["backend", "frame", "fallback"]
# These limits apply to incoming public frames. Trusted history attribution
# adds three words when an older frame is spoken without truncating its report.
MAX_PROGRESS_BYTES = 240

_FLIGHT_WAIT = "I haven't received a flight progress update yet."
_WEATHER_WAIT = "I haven't received a weather progress update yet."
_MUSE_WAIT = "I haven't received a detailed progress update yet."
_WORKING = "Muse says it's working on your request."
_RESPONDING = "Muse is preparing a reply."
_LAST_WORKING = "Muse's latest status is that it's working on your request."
_LAST_RESPONDING = "Muse's latest status is that it's preparing a reply."
_FLIGHTS = "Looking at flights now."
_FARES = "Checking fares now."
_COMPARE = "Comparing the options."
_WEB = "Searching the web now."
_SOURCES = "Searching sources now."
_WEBSITE = "Checking a website now."
_CONNECTOR = "Using a connector now."

_LATEST_FLIGHTS = "Muse's last reported step was looking at flights."
_LATEST_FARES = "Muse's last reported step was checking fares."
_LATEST_COMPARE = "Muse's last reported step was comparing options."
_LATEST_WEB = "Muse's last reported step was searching the web."
_LATEST_SOURCES = "Muse's last reported step was searching sources."
_LATEST_WEBSITE = "Muse's last reported step was checking a website."
_LATEST_CONNECTOR = "Muse's last reported step was using a connector."

_RESPONDING_HISTORY = {
    latest: _RESPONDING + " Its" + latest[len("Muse's"):]
    for latest in (_LATEST_FLIGHTS, _LATEST_FARES, _LATEST_COMPARE, _LATEST_WEB,
                   _LATEST_SOURCES, _LATEST_WEBSITE, _LATEST_CONNECTOR)
}

PUBLIC_PROGRESS_PHRASES = (
    _FLIGHT_WAIT, _WEATHER_WAIT, _MUSE_WAIT,
    _FLIGHTS, _FARES, _COMPARE, _WEB, _SOURCES, _WEBSITE, _CONNECTOR,
    _LATEST_FLIGHTS, _LATEST_FARES, _LATEST_COMPARE, _LATEST_WEB,
    _LATEST_SOURCES, _LATEST_WEBSITE, _LATEST_CONNECTOR,
    _WORKING, _RESPONDING, _LAST_WORKING, _LAST_RESPONDING,
    *_RESPONDING_HISTORY.values(),
)

_PRIVATE_PAYLOAD = re.compile(
    r"https?://|\bwww\.|[{}\[\]]|```|"
    r"\b(?:bearer|password|api[ _-]?key|(?:access|refresh)[ _-]?token)\b",
    re.IGNORECASE,
)
_ACTIVITY_PREFIX = re.compile(r"^(?:i(?:'m|’m| am)\s+)?(?:currently\s+)?")


@dataclass(frozen=True)
class ProgressUpdate:
    text: str
    source: ProgressSource = "frame"


@dataclass(frozen=True)
class BackendActivity:
    current_text: str
    latest_text: str


@dataclass(frozen=True)
class BackendStatus:
    phase: Literal["working", "responding", "idle", "unknown"]
    activity: BackendActivity | None = None


_BACKEND_ACTIVITIES = {
    _FLIGHTS: BackendActivity(_FLIGHTS, _LATEST_FLIGHTS),
    _FARES: BackendActivity(_FARES, _LATEST_FARES),
    _COMPARE: BackendActivity(_COMPARE, _LATEST_COMPARE),
    _WEB: BackendActivity(_WEB, _LATEST_WEB),
    _SOURCES: BackendActivity(_SOURCES, _LATEST_SOURCES),
    _WEBSITE: BackendActivity(_WEBSITE, _LATEST_WEBSITE),
    _CONNECTOR: BackendActivity(_CONNECTOR, _LATEST_CONNECTOR),
}


def validate_public_progress(text: object) -> str | None:
    """Accept one short public sentence, excluding raw payloads and credentials."""
    if not isinstance(text, str) or any(ord(char) < 32 or ord(char) == 127 for char in text):
        return None
    clean = " ".join(text.split())
    if not clean or _PRIVATE_PAYLOAD.search(clean):
        return None
    try:
        if len(clean.encode("utf-8")) > MAX_PROGRESS_BYTES:
            return None
    except UnicodeEncodeError:
        return None
    return clean


def waiting_progress(request_text: str) -> str:
    """Describe waiting for a result, without claiming that a tool is running."""
    if re.search(r"\b(?:flights?|airfares?|fares?)\b", request_text, re.IGNORECASE):
        return _FLIGHT_WAIT
    if re.search(r"\b(?:weather|forecast)\b", request_text, re.IGNORECASE):
        return _WEATHER_WAIT
    return _MUSE_WAIT


def progress_from_activity(activity_text: object) -> str | None:
    """Map explicit public actions to fixed cues; never return raw status text."""
    clean = validate_public_progress(activity_text)
    if clean is None:
        return None
    action = _ACTIVITY_PREFIX.sub("", clean.casefold())
    if re.match(r"^(?:searching(?: for)?|looking for) (?:available )?flights?\b", action):
        return _FLIGHTS
    if re.match(r"^(?:checking|searching(?: for)?) (?:flight )?(?:air)?fares?\b", action):
        return _FARES
    if re.match(r"^comparing\b", action):
        return _COMPARE
    if re.match(r"^searching (?:on )?(?:the )?web\b", action):
        return _WEB
    if re.match(r"^searching (?:for )?sources\b", action):
        return _SOURCES
    if re.match(r"^(?:checking|opening|reading|visiting) "
                r"(?:(?:a|an|the) |[a-z0-9_-]+ )?website\b", action):
        return _WEBSITE
    if re.match(r"^(?:using|checking|querying|opening) "
                r"(?:(?:a|the|public) |[a-z0-9_-]+ )?connector\b", action):
        return _CONNECTOR
    return None


def backend_activity_from_status(activity_text: object) -> BackendActivity | None:
    """Convert an explicit public status to fixed current and historical speech."""
    return _BACKEND_ACTIVITIES.get(progress_from_activity(activity_text))


def backend_status_from_event(activity_code: object, activity_text: object) -> BackendStatus:
    """Separate public phase information from an allowlisted specific action."""
    phase = activity_code if activity_code in ("working", "responding", "idle") else "unknown"
    if activity_code == "online":
        phase = "idle"
    return BackendStatus(phase, backend_activity_from_status(activity_text))


class ProgressPlan:
    """Speak distinct turn-scoped updates, at most once per cadence interval."""

    def __init__(self, request_text: str, started: float, *, first_delay_s: float = 20,
                 interval_s: float = 20, expiry_s: float = 10,
                 max_updates: int | None = None):
        self._waiting = waiting_progress(request_text)
        self._first_due = started + first_delay_s
        self._interval_s = interval_s
        self._expiry_s = expiry_s
        self._max_updates = max_updates
        self._last_cue = None
        self._emitted = 0
        self._spoken = set()
        self._spoken_activities = set()
        self._spoken_phases = set()
        self._pending = None
        self._status = None
        self._milestone = None
        self._stopped = False

    def offer(self, text: object, now: float, *, source: ProgressSource = "frame") -> bool:
        """Replace pending progress with a valid update not already spoken."""
        clean = validate_public_progress(text)
        if (self._stopped or clean is None or clean in self._spoken
                or (self._pending is not None and clean == self._pending[0].text)):
            return False
        self._pending = (ProgressUpdate(clean, source), now)
        return True

    def set_status(self, status: BackendStatus, now: float) -> bool:
        """Update the phase; generic reports preserve the last specific milestone."""
        if (self._stopped or status.phase not in ("working", "responding", "idle", "unknown")
                or (status.activity is not None and status.activity not in _BACKEND_ACTIVITIES.values())):
            return False
        self._status = (status, now)
        if status.activity is not None:
            self._milestone = (status.activity, now)
        return True

    def take(self, now: float) -> ProgressUpdate | None:
        fresh_public_frame = (self._pending is not None
                              and self._pending[0].source == "frame"
                              and now - self._pending[1] < self._expiry_s)
        fresh_specific_activity = (self._milestone is not None
                                   and self._milestone[0] not in self._spoken_activities
                                   and now - self._milestone[1] < self._expiry_s)
        first_specific_cue = (self._last_cue is None
                              and (fresh_public_frame or fresh_specific_activity))
        if (self._stopped or (self._max_updates is not None and self._emitted >= self._max_updates)
                or (now < self._first_due and not first_specific_cue)
                or (self._last_cue is not None and now - self._last_cue < self._interval_s)):
            return None
        update = None
        reported_text = None
        if (self._pending is not None and fresh_specific_activity
                and now - self._pending[1] >= self._expiry_s):
            self._spoken.add(self._pending[0].text)
            self._pending = None
        if self._pending is not None:
            pending, offered = self._pending
            reported_text = pending.text
            text = pending.text
            if now - offered >= self._expiry_s:
                # The trusted attribution adds three words to the bounded
                # incoming frame and preserves every word of its report.
                text = "Earlier from Muse: " + text
            update = ProgressUpdate(text, pending.source)
            self._spoken.add(pending.text)
            self._pending = None
        if update is None and self._status is not None:
            status, observed = self._status
            fresh = now - observed < self._expiry_s
            milestone = self._milestone[0] if self._milestone is not None else None
            if milestone is not None and milestone not in self._spoken_activities:
                text = milestone.latest_text
                if fresh and status.phase == "working" and status.activity == milestone:
                    text = milestone.current_text
                elif fresh and status.phase == "responding":
                    text = _RESPONDING_HISTORY[text]
                    self._spoken_phases.add("responding")
                self._spoken_activities.add(milestone)
                self._spoken_phases.add("working")
            else:
                text = None
                if status.phase not in self._spoken_phases:
                    text = {"working": _WORKING if fresh else _LAST_WORKING,
                            "responding": _RESPONDING if fresh else _LAST_RESPONDING}.get(status.phase)
                    if text is not None:
                        self._spoken_phases.add(status.phase)
            if text is not None:
                update = ProgressUpdate(text, "backend")
        if update is None:
            if self._emitted:
                return None
            update = ProgressUpdate(self._waiting, "fallback")
        self._last_cue = now
        self._emitted += 1
        self._spoken.add(update.text)
        for activity in _BACKEND_ACTIVITIES.values():
            if (reported_text or update.text) in (activity.current_text, activity.latest_text,
                                                 _RESPONDING_HISTORY[activity.latest_text]):
                self._spoken_activities.add(activity)
                self._spoken_phases.add("working")
                self._spoken.update((activity.current_text, activity.latest_text))
        if (reported_text or update.text) in (_WORKING, _LAST_WORKING):
            self._spoken_phases.add("working")
            self._spoken.update((_WORKING, _LAST_WORKING))
        if (reported_text or update.text) in (_RESPONDING, _LAST_RESPONDING):
            self._spoken_phases.add("responding")
            self._spoken.update((_RESPONDING, _LAST_RESPONDING))
        return update

    def stop(self) -> None:
        self._stopped = True
        self._pending = None
        self._status = None
        self._milestone = None
