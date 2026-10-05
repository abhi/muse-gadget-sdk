# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Hands-free Muse conversations through Reachy Mini's own audio hardware."""

from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import logging
import math
import re
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Awaitable, Callable

from musegadget import __version__
from musegadget import reachy_expression_plan as plan
from musegadget.link_client import DeviceDescription, LinkSession, Outcome
from musegadget.reachy_capabilities import WAKE_CUE, Backends, Expression, ReplyStyle, SpokenLine
from musegadget.reachy_expression import transcript_text
from musegadget.reachy_progress import BackendStatus, ProgressPlan, backend_status_from_event
from musegadget.service import DEFAULT_NOISE_HOST, Service
from musegadget.speech_playback import PCMPlayer, SpeechPlayback

log = logging.getLogger(__name__)
REPLY_TIMEOUT_S = 180.0
SPEECH_TIMEOUT_S = 180.0
REPLY_QUIET_S = 3.0
EMPTY_REPLY_GRACE_S = 1.0
ECHO_TAIL_S = 0.35
PENDING_TURN_LIMIT = 8
INPUT_OVERFLOW_CUE = "My question queue is full. Please repeat that after I finish."
TRANSCRIPTION_RETRY_CUE = "I missed part of that. Please say it again."


def strip_wake_prefix(text: str, phrase: str) -> str:
    """Strip one exact leading wake phrase, allowing ASR punctuation between words."""
    tokens = phrase.split()
    if not tokens:
        return text.strip()
    prefix = r"[\s,!.?:;-]*".join(re.escape(token) for token in tokens)
    return re.sub(r"^\s*" + prefix + r"\b[\s,!.?:;-]*", "", text,
                  count=1, flags=re.IGNORECASE).strip()


def question_after_wake(text: str, phrase: str) -> str | None:
    """Discard pre-wake words; None means ASR did not confirm the invocation."""
    prefix = r"[\s,!.?:;-]*".join(re.escape(token) for token in phrase.split())
    match = re.search(r"\b" + prefix + r"\b[\s,!.?:;-]*", text, flags=re.IGNORECASE)
    return text[match.end():].strip() if match is not None else None
_ACTIVITY_CODES = frozenset(("working", "responding", "idle", "online", "thinking", "researching",
                             "planning", "listening", "speaking", "processing", "generating", "executing"))
_TERMINAL_TASK_STATUSES = frozenset(("completed", "failed", "errored", "cancelled", "canceled"))


@dataclass(frozen=True)
class SpeechSegment:
    message_id: str
    index: int
    text: str
    expression: str | None = None


@dataclass(frozen=True)
class ProgressSegment:
    text: str
    source: str = "frame"


@dataclass(frozen=True)
class BackendStatusSegment:
    status: BackendStatus


@dataclass(frozen=True)
class TaskFinished:
    """Muse reported the turn's task terminal; the waiting turn rechecks completion now."""


class TurnOutcome(Enum):
    EMPTY = "empty"
    WAKE_CUE = "wake_cue"
    ACCEPTED = "accepted"


class _SpeechSuperseded(Exception):
    """An unheard sentence was replaced before its audio could be queued."""


@dataclass(frozen=True)
class _CapturedAudio:
    samples: object
    discard: bool = False
    gap: bool = False
    gap_reason: str | None = None


@dataclass(frozen=True)
class _RecordedTurn:
    wav: bytes
    recognition: asyncio.Future | None = None
    endpoint_at: float = field(default_factory=time.monotonic)


@dataclass(frozen=True)
class _QueuedTurn:
    wav: bytes
    wake_strip_required: bool
    wake_epoch: int
    recognition: asyncio.Future | None = None
    endpoint_at: float = field(default_factory=time.monotonic)


@dataclass(frozen=True)
class _SpeechJob:
    message_id: str | None
    text: str
    expression: str | None = None
    prepared_stream: object = None
    is_current: Callable[[], bool] | None = None
    on_start: Callable[[], None] | None = None
    started: float | None = None
    state: str = "speaking"


_SPEECH_LABELS = {
    "ack": ("Reachy acknowledgement", "acknowledgement audio"),
    "progress": ("Reachy progress", "progress audio"),
    "answer": ("Muse speech", "Muse audio"),
    "notice": ("Reachy notice", "notice audio"),
}


def _line(message_id: str | None, text: str, expression: str | None, *,
          progress: bool = False, role: str | None = None) -> SpokenLine:
    role = role or ("progress" if progress else "answer" if message_id is not None else "notice")
    return SpokenLine(text, Expression.parse(expression), role, message_id)


class _CaptureBuffer:
    """Bounded microphone audio, with explicit discontinuities on loss."""

    def __init__(self, sample_rate: int, *, capacity_s: float = 1.0):
        self.limit = int(sample_rate * capacity_s)
        self.batch_limit = int(sample_rate * .1)
        self.samples = 0
        self.chunks = deque()
        self.available = asyncio.Event()
        self.error = None
        self.gap = False
        self.gap_reason = None

    def push(self, samples, *, discard: bool = False, gap: bool = False,
             gap_reason: str | None = None) -> None:
        if not len(samples):
            return
        if gap:
            self.chunks.clear()
            self.samples = 0
            self.gap = True
            self.gap_reason = gap_reason or "upstream gap"
        if len(samples) > self.limit:
            samples = samples[-self.limit:].copy()
            self.gap = True
            self.gap_reason = "oversized capture chunk"
        if self.samples + len(samples) > self.limit:
            self.chunks.clear()
            self.samples = 0
            self.gap = True
            self.gap_reason = "capture buffer overflow"
        self.chunks.append(_CapturedAudio(samples, discard))
        self.samples += len(samples)
        self.available.set()

    def fail(self, error: Exception) -> None:
        self.error = error
        self.available.set()

    async def get(self) -> _CapturedAudio | None:
        if not self.chunks and self.error is None:
            self.available.clear()
            try:
                await asyncio.wait_for(self.available.wait(), .01)
            except asyncio.TimeoutError:
                return None
        if self.error is not None:
            raise self.error
        chunk = self.chunks.popleft()
        self.samples -= len(chunk.samples)
        # Drain available chunks together without waiting for a batch to fill.
        # A discard boundary must never become authorized microphone input.
        parts = [chunk.samples]
        size = len(chunk.samples)
        while (self.chunks and size < self.batch_limit
               and self.chunks[0].discard == chunk.discard):
            following = self.chunks.popleft().samples
            parts.append(following)
            size += len(following)
            self.samples -= len(following)
        if len(parts) > 1:
            import numpy as np
            samples = np.concatenate(parts)
        else:
            samples = chunk.samples
        result = _CapturedAudio(samples, chunk.discard, self.gap, self.gap_reason)
        self.gap = False
        self.gap_reason = None
        return result


class ReplayScope:
    """Remember a bounded set of retired IDs across this connection's turns."""

    def __init__(self, limit: int = 2048):
        self.limit = limit
        self.ids: set[tuple[str, str]] = set()
        self.order: deque[tuple[str, str]] = deque()

    @staticmethod
    def event_ids(event: dict, *, include_parents: bool = True) -> set[tuple[str, str]]:
        payload = event.get("payload")
        if not isinstance(payload, dict):
            return set()
        result = set()
        keys = ("message_id", "id", "task_id")
        if include_parents:
            keys += ("reply_to_message_id", "parent_message_id")
        for key in keys:
            value = payload.get(key)
            if isinstance(value, str) and value:
                if len(value) > 512:
                    raise ValueError("Muse returned an oversized chat identifier")
                result.add(("task" if key == "task_id" else "message", value))
        value = event.get("message_id")
        if isinstance(value, str) and value:
            if len(value) > 512:
                raise ValueError("Muse returned an oversized chat identifier")
            result.add(("message", value))
        return result

    def retire(self, ids) -> None:
        for identifier in ids:
            if identifier not in self.ids:
                self.ids.add(identifier)
                self.order.append(identifier)
        while len(self.order) > self.limit:
            self.ids.discard(self.order.popleft())

    def observe_idle(self, event: dict, session_id: str | None) -> None:
        payload = event.get("payload")
        if (event.get("type") == "event" and isinstance(payload, dict) and session_id
                and (payload.get("session_id") or event.get("session_id")) == session_id
                and event.get("event") in ("agent.status", "task.status", "delta.message_start",
                                           "delta.text_append", "delta.message_done", "message.assistant")):
            self.retire(self.event_ids(event))


@dataclass
class ReplyTracker:
    """Bind assistant messages to one acknowledged microphone turn."""

    session_id: str | None = None
    user_ids: set[str] = field(default_factory=set)
    messages: dict[str, dict] = field(default_factory=dict)
    acknowledged: bool = False
    busy: bool = False
    task_finished: bool = False
    last_activity: float = field(default_factory=time.monotonic)
    pending: list[dict] = field(default_factory=list)
    style: ReplyStyle = ReplyStyle.MARKER
    activity_code: str | None = None
    pending_text_bytes: int = 0
    owns_chat: bool = False
    replay_scope: ReplayScope | None = None
    observed_ids: set[tuple[str, str]] = field(default_factory=set)
    running_task_ids: set[str] = field(default_factory=set)

    def retire(self) -> None:
        if self.owns_chat and self.replay_scope is not None:
            self.replay_scope.retire(self.observed_ids)
            self.replay_scope.retire(("message", value) for value in self.user_ids)
            for event in self.pending:
                self.replay_scope.retire(ReplayScope.event_ids(event))

    def acknowledge(self, response: dict) -> list[
            str | SpeechSegment | ProgressSegment | BackendStatusSegment | TaskFinished]:
        result = response.get("result", response)
        if not isinstance(result, dict):
            raise ValueError("Muse returned an invalid chat acknowledgement")
        self.user_ids = {result[k] for k in ("message_id", "reply_to_message_id")
                         if isinstance(result.get(k), str) and result[k]}
        if not self.user_ids:
            raise ValueError("Muse chat acknowledgement omitted the user message ID")
        if self.owns_chat and any(len(value) > 512 for value in self.user_ids):
            raise ValueError("Muse returned an oversized chat identifier")
        acknowledged_session = result.get("session_id")
        if self.owns_chat and (not self.session_id or acknowledged_session != self.session_id
                               or result.get("is_thread") is not True):
            raise ValueError("Muse did not acknowledge Reachy's dedicated side chat")
        if self.session_id is None and isinstance(acknowledged_session, str):
            self.session_id = acknowledged_session
        self.acknowledged = True
        queued = []
        pending, self.pending = self.pending, []
        self.pending_text_bytes = 0
        for event in pending:
            queued.extend(self.event(event))
        return queued

    def event(self, event: dict) -> list[
            str | SpeechSegment | ProgressSegment | BackendStatusSegment | TaskFinished]:
        if event.get("type") != "event":
            return []
        payload = event.get("payload")
        if not isinstance(payload, dict):
            return []
        event_session = payload.get("session_id") or event.get("session_id")
        if self.owns_chat:
            if not self.session_id or event_session != self.session_id:
                return []
            identifiers = ReplayScope.event_ids(event)
            if self.replay_scope is not None:
                own_ids = ReplayScope.event_ids(event, include_parents=False)
                if own_ids & self.replay_scope.ids:
                    return []
                parents = identifiers - own_ids
                authorized_parents = {("message", value) for value in self.user_ids}
                if self.acknowledged and (parents - authorized_parents) & self.replay_scope.ids:
                    return []
            self.observed_ids.update(identifiers)
            if len(self.observed_ids) > 256:
                raise ValueError("Muse returned too many activity identifiers for one turn")
        if event_session and self.session_id and event_session != self.session_id:
            return []
        if not self.acknowledged:
            if len(self.pending) >= 64:
                raise ValueError("too many chat events before Muse acknowledged the turn")
            self.pending_text_bytes += sum(len(payload[key].encode("utf-8")) for key in
                                           ("text", "display_text", "content", "transcript")
                                           if isinstance(payload.get(key), str))
            if self.pending_text_bytes > 64 * 1024:
                raise ValueError("Muse returned more than 64 KiB of pending reply text")
            self.pending.append(event)
            return []
        scoped = bool(self.owns_chat and self.session_id and event_session == self.session_id)
        name = event.get("event")
        if name in ("agent.status", "task.status"):
            activity = payload.get("activity_code")
            status = payload.get("status")
            parent = payload.get("reply_to_message_id") or payload.get("parent_message_id")
            message_id = payload.get("message_id")
            explicitly_linked = (parent in self.user_ids
                                 or self.messages.get(message_id, {}).get("progress_linked", False))
            terminal = isinstance(status, str) and status in _TERMINAL_TASK_STATUSES
            was_finished = self.task_finished
            task_id = payload.get("task_id")
            if not isinstance(task_id, str):
                task_id = None
            if self.owns_chat and name == "task.status":
                if terminal and task_id not in self.running_task_ids and not explicitly_linked:
                    return []
                if status == "running" and isinstance(task_id, str) and task_id:
                    self.running_task_ids.add(task_id)
                if terminal:
                    self.running_task_ids.discard(task_id)
            if name == "task.status" and isinstance(status, str):
                self.task_finished = terminal and not (self.owns_chat and self.running_task_ids)
            if isinstance(activity, str):
                self.busy = activity not in ("", "online", "idle") and activity not in _TERMINAL_TASK_STATUSES
                if name == "agent.status":
                    self.activity_code = activity if activity in _ACTIVITY_CODES else "other"
            elif isinstance(status, str):
                self.busy = status not in ("", "idle") and not terminal
            if self.owns_chat and self.running_task_ids:
                self.task_finished = False
                self.busy = True
            elif name == "task.status" and self.task_finished:
                self.busy = False
            self.last_activity = time.monotonic()
            linked = scoped or explicitly_linked
            if name == "agent.status" and linked and not self.task_finished:
                return [BackendStatusSegment(backend_status_from_event(activity, payload.get("activity_text")))]
            return [TaskFinished()] if self.task_finished and not was_finished else []
        if name not in ("delta.message_start", "delta.text_append",
                        "delta.message_done", "message.assistant"):
            return []
        message_id = payload.get("message_id") or event.get("message_id") or payload.get("id")
        if not isinstance(message_id, str) or not message_id or payload.get("role") == "user":
            return []
        parent = payload.get("reply_to_message_id") or payload.get("parent_message_id")
        if (parent and parent not in self.user_ids and parent not in self.messages
                and not (scoped and parent == message_id)):
            return []
        if message_id not in self.messages:
            if len(self.messages) >= 32:
                raise ValueError("Muse returned too many messages for one voice turn")
            self.messages[message_id] = {"text": "", "done": False, "queued": False,
                                         "progress_linked": False}
            if self.style is ReplyStyle.EXPRESSIVE_JSON:
                from musegadget.reachy_expression import SentenceStream
                self.messages[message_id].update(stream=SentenceStream(), delta_text="", stream_fed=0,
                                                 stream_closed=False, stream_index=0, stream_spoken=0,
                                                 stream_revised=False, stream_skip_before=0, linked=not parent)
        message = self.messages[message_id]
        if scoped:
            message["progress_linked"] = True
        if parent:
            message["progress_linked"] = (scoped or parent in self.user_ids
                                          or self.messages.get(parent, {}).get("progress_linked", False))
        if self.style is ReplyStyle.EXPRESSIVE_JSON and parent:
            message["linked"] = (scoped or parent in self.user_ids
                                 or self.messages.get(parent, {}).get("linked", False))
        self.last_activity = time.monotonic()
        if name == "delta.text_append" and payload.get("text") and not message["done"]:
            if isinstance(payload["text"], str):
                message["text"] += payload["text"]
                if self.style is ReplyStyle.EXPRESSIVE_JSON:
                    message["delta_text"] += payload["text"]
        if name in ("delta.message_done", "message.assistant"):
            completed_text = payload.get("display_text")
            if not isinstance(completed_text, str):
                transcript = payload.get("transcript")
                completed_text = transcript if isinstance(transcript, str) else transcript_text(transcript, message_id)
                if not completed_text and not isinstance(transcript, str):
                    completed_text = payload.get("content")
            if isinstance(completed_text, str):
                message["text"] = completed_text
            if name == "delta.message_done" or payload.get("display_text_ready") is not False:
                message["done"] = True
        if len(message["text"].encode("utf-8")) > 64 * 1024 or (
            self.style is ReplyStyle.EXPRESSIVE_JSON and len(message["delta_text"].encode("utf-8")) > 64 * 1024
        ):
            raise ValueError("Muse returned more than 64 KiB of reply text")
        if self.style is ReplyStyle.EXPRESSIVE_JSON:
            return self._stream_segments()
        if message["done"] and message["text"].strip() and not message["queued"]:
            message["queued"] = True
            return [message_id]
        return []

    def _stream_segments(self) -> list[SpeechSegment | ProgressSegment]:
        from musegadget.reachy_expression import ReplyProtocolError, ReplyRevisionError

        queued = []
        for message_id, message in self.messages.items():
            if message["stream_closed"]:
                continue
            if not message["done"] and not message["linked"]:
                continue
            decoder = message["stream"]
            try:
                if message["done"]:
                    try:
                        sentences = decoder.finish(message["text"])
                    except ReplyRevisionError:
                        if message["stream_spoken"]:
                            log.warning("Muse revised a spoken reply; remaining speech for that message skipped")
                            message["stream_revised"] = True
                            sentences = []
                        else:
                            from musegadget.reachy_expression import SentenceStream
                            message["stream_skip_before"] = message["stream_index"]
                            message["stream"] = SentenceStream()
                            sentences = message["stream"].finish(message["text"])
                    message["stream_closed"] = True
                else:
                    sentences = decoder.feed(message["delta_text"][message["stream_fed"]:])
                    message["stream_fed"] = len(message["delta_text"])
            except ReplyProtocolError:
                log.warning("Muse returned an invalid sentence frame; speech for that message stopped")
                message.update(stream_closed=True, stream_revised=True, done=True,
                               queued=bool(message["stream_spoken"]))
                sentences = []
            for sentence in sentences:
                if sentence.kind == "progress":
                    if message["progress_linked"] and not message["done"]:
                        queued.append(ProgressSegment(sentence.text))
                    continue
                index = message["stream_index"]
                message["stream_index"] += 1
                if sentence.text.strip() or sentence.expression:
                    queued.append(SpeechSegment(message_id, index, sentence.text, sentence.expression))
                    message["queued"] = True
            if len(queued) > 32:
                raise ValueError("Muse returned too many queued speech segments")
            if not message["done"]:
                break
        return queued

    def complete(self, now: float, played: bool) -> bool:
        return (played and not self.busy and all(m["done"] for m in self.messages.values())
                and (self.task_finished or now - self.last_activity >= REPLY_QUIET_S))

    def finished_without_text(self, now: float) -> bool:
        return (self.task_finished and not self.busy
                and all(m["done"] for m in self.messages.values())
                and not any(m["queued"] for m in self.messages.values())
                and now - self.last_activity >= (EMPTY_REPLY_GRACE_S if self.messages else REPLY_QUIET_S))


class VoiceConversation:
    def __init__(self, session: LinkSession, hardware, *, backends: Backends,
                 session_id: str | None = None,
                 silence_s: float = 2.0, reply_timeout_s: float = REPLY_TIMEOUT_S,
                 wake_detector=None, wake_timeout_s: float = 10.0,
                 owns_chat: bool = False, speech_gate=None,
                 audio_diagnostics: bool = False) -> None:
        if not math.isfinite(wake_timeout_s) or wake_timeout_s < 0:
            raise ValueError("wake timeout must be a finite, nonnegative number")
        self.session = session
        self.hardware = hardware
        self.session_id = session_id
        self.owns_chat = owns_chat
        self.silence_s = silence_s
        self.reply_timeout_s = reply_timeout_s
        self.backends = backends
        self._progress: ProgressPlan | None = None
        self._prepared_speech = {}
        self._closing_speech = set()
        self.wake_detector = wake_detector
        self.speech_gate = speech_gate
        self.audio_diagnostics = audio_diagnostics
        self.wake_timeout_s = wake_timeout_s
        self._wake_deadline = None
        self._wake_strip_required = False
        self._wake_first_request_pending = False
        self._voice_context_sent = False
        self.tracker: ReplyTracker | None = None
        self._replies: asyncio.Queue[str | SpeechSegment | TaskFinished] = asyncio.Queue(maxsize=32)
        self._seen_sequences: set[int] = set()
        self._sequence_order = deque()
        self._muted = False
        self._echo_until = 0.0
        self._input_overflow = False
        self._turn_started = None
        self._logged_activity = None
        self._activity_log_count = 0
        self._logged_progress_status = None
        self._replay_scope = ReplayScope()
        self._playback = SpeechPlayback()
        self._speaker_lock = asyncio.Lock()
        self._speech_pending = 0
        self._audio_clear_attempted = False
        self._output_queue: asyncio.Queue[_SpeechJob] = asyncio.Queue(maxsize=32)
        self._output_busy = False
        self._output_prefetched = False
        self._defer_playback = False
        self._planner = plan.ExpressionPlanner()
        self._muse_status: BackendStatus | None = None

    async def _plan(self, event: plan.Event) -> None:
        for cue in self._planner.on(event, time.monotonic()):
            await self._apply_cue(cue)

    async def _apply_cue(self, cue: plan.Cue) -> None:
        if cue.expression is None:
            await asyncio.to_thread(self.hardware.set_state, cue.state.value)
        else:
            await asyncio.to_thread(self.hardware.set_state, cue.state.value,
                                    expression=cue.expression.value)

    def _has_output(self) -> bool:
        return (self._speech_pending > 0 or self._speaker_lock.locked() or self._output_busy
                or self._output_prefetched or not self._output_queue.empty())

    async def _speech_with_timeout(self, speaking) -> None:
        task = asyncio.create_task(speaking)
        started = time.monotonic()
        held = self._playback.paused_s
        try:
            while not task.done():
                elapsed = time.monotonic() - started - (self._playback.paused_s - held)
                if elapsed >= SPEECH_TIMEOUT_S:
                    raise TimeoutError("Reachy's speech playback did not complete in time")
                await asyncio.wait({task}, timeout=min(.1, SPEECH_TIMEOUT_S - elapsed))
            task.result()
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def _play_output(self) -> None:
        """One speaker owner and one sentence of synthesis ahead of it."""
        async def next_job():
            job = await self._output_queue.get()
            self._output_prefetched = True
            try:
                if job.prepared_stream is None and job.text:
                    job = replace(job, prepared_stream=self.backends.voice.prepare(
                        _line(job.message_id, job.text, job.expression), self.hardware.output_sample_rate))
            except BaseException:
                self._output_prefetched = False
                self._output_queue.task_done()
                raise
            return job
        async def close_job(job):
            try:
                if job.prepared_stream is not None:
                    await job.prepared_stream.aclose()
            finally:
                self._output_queue.task_done()
        following = asyncio.create_task(next_job())
        try:
            while True:
                job = await asyncio.shield(following)
                self._output_prefetched = False
                following = asyncio.create_task(next_job())
                self._output_busy = True
                try:
                    await self._speech_with_timeout(self._speak(
                        job.message_id, text=job.text, expression=job.expression,
                        state=job.state, stream_segment=True, prepared_stream=job.prepared_stream,
                        is_current=job.is_current, on_start=job.on_start,
                        turn_started=job.started))
                except _SpeechSuperseded:
                    pass
                finally:
                    await close_job(job)
                    self._output_busy = False
                if self._output_queue.empty() and not following.done():
                    await self._plan(plan.OutputIdle(user_speaking=self._playback.user_speaking,
                                                     turn_open=self.tracker is not None,
                                                     wake_open=self._wake_deadline is not None))
                    if self._wake_deadline is not None:
                        self._wake_deadline = time.monotonic() + (self.wake_timeout_s or 10.0)
        finally:
            if not following.done():
                following.cancel()
            result, = await asyncio.gather(following, return_exceptions=True)
            if isinstance(result, _SpeechJob):
                await close_job(result)
            while not self._output_queue.empty():
                await close_job(self._output_queue.get_nowait())
            self._output_prefetched = False

    def _input_blocked(self) -> bool:
        """Unknown audio routes suppress input only while speaker audio can echo."""
        if getattr(self.hardware, "echo_cancelled_input", False) is True:
            return False
        return self._muted or time.monotonic() < self._echo_until

    async def run(self) -> None:
        await self.session.registered.wait()
        await self.warm_up()
        subscriber = asyncio.create_task(self._subscribe())
        microphone = asyncio.create_task(self._microphone())
        try:
            done, _ = await asyncio.wait({subscriber, microphone},
                                         return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
            raise ConnectionError("Muse voice stream ended")
        finally:
            for task in (subscriber, microphone):
                task.cancel()
            await asyncio.gather(subscriber, microphone, return_exceptions=True)
            await asyncio.to_thread(self.hardware.clear_audio)

    async def warm_up(self) -> None:
        rate = self.hardware.output_sample_rate
        await self.backends.voice.warm(rate)
        if self.backends.progress_voice is not None:
            await self.backends.progress_voice.warm(rate)

    async def _subscribe(self) -> None:
        subscription = self.session.subscribe_chat(self.session_id)
        try:
            async for event in subscription:
                seq = event.get("seq")
                if isinstance(seq, int) and seq > 0:
                    if seq in self._seen_sequences:
                        continue
                    self._seen_sequences.add(seq)
                    self._sequence_order.append(seq)
                    if len(self._sequence_order) > 2048:
                        self._seen_sequences.discard(self._sequence_order.popleft())
                tracker = self.tracker
                if tracker is not None:
                    had_message = bool(tracker.messages)
                    had_text = any(message["text"].strip() for message in tracker.messages.values())
                    completed_before = {message_id for message_id, message in tracker.messages.items() if message["done"]}
                    self._queue_replies(tracker.event(event), tracker)
                    self._log_backend_activity()
                    if self._turn_started is not None:
                        elapsed = time.monotonic() - self._turn_started
                        for message_id, message in tracker.messages.items():
                            if message["done"] and message_id not in completed_before:
                                log.info("Muse completed a reply message %.1fs after the voice turn", elapsed)
                        if not had_message and tracker.messages:
                            log.info("Muse first assistant event arrived %.1fs after the voice turn", elapsed)
                        if not had_text and any(message["text"].strip() for message in tracker.messages.values()):
                            log.info("Muse first reply text arrived %.1fs after the voice turn", elapsed)
                elif self.owns_chat:
                    self._replay_scope.observe_idle(event, self.session_id)
        finally:
            await subscription.aclose()

    def _log_backend_activity(self) -> None:
        code = self.tracker.activity_code if self.tracker is not None else None
        if code is not None and code != self._logged_activity:
            self._logged_activity = code
            if self._activity_log_count < 32:
                elapsed = time.monotonic() - self._turn_started if self._turn_started is not None else 0
                log.info("Muse backend activity %s %.1fs after the voice turn", code, elapsed)
                self._activity_log_count += 1

    def _queue_replies(self, replies, tracker: ReplyTracker) -> None:
        pending = []
        while not self._replies.empty():
            pending.append(self._replies.get_nowait())
        for reply in (*pending, *replies):
            if isinstance(reply, BackendStatusSegment):
                self._muse_status = reply.status
                if self._progress is not None:
                    accepted = self._progress.set_status(reply.status, time.monotonic())
                    if accepted and reply.status != self._logged_progress_status:
                        self._logged_progress_status = reply.status
                        if self._activity_log_count < 32:
                            elapsed = (time.monotonic() - self._turn_started
                                       if self._turn_started is not None else 0)
                            log.info("Muse public progress phase=%s stage=%s %.1fs after the voice turn",
                                     reply.status.phase,
                                     reply.status.activity.current_text if reply.status.activity else "none", elapsed)
                            self._activity_log_count += 1
                continue
            if isinstance(reply, ProgressSegment):
                if self._progress is not None:
                    self._progress.offer(reply.text, time.monotonic(), source=reply.source)
                continue
            if isinstance(reply, SpeechSegment):
                message = tracker.messages[reply.message_id]
                if message["stream_revised"] or reply.index < message["stream_skip_before"]:
                    self._discard_prepared((reply.message_id, reply.index))
                    continue
            if self._progress is not None:
                self._progress.stop()
            try:
                self._replies.put_nowait(reply)
            except asyncio.QueueFull:
                raise ValueError("Muse returned too many queued speech segments") from None
            key = (reply.message_id, reply.index) if isinstance(reply, SpeechSegment) else None
            if (not self._defer_playback and key is not None and key not in self._prepared_speech
                    and reply.text.strip() and len(self._prepared_speech) < 2):
                prepared = self.backends.voice.prepare(
                    _line(reply.message_id, reply.text, reply.expression), self.hardware.output_sample_rate)
                if prepared is not None:
                    self._prepared_speech[key] = prepared

    def _discard_prepared(self, key) -> None:
        prepared = self._prepared_speech.pop(key, None)
        if prepared is not None:
            prepared.cancel()
            closing = asyncio.create_task(prepared.aclose())
            self._closing_speech.add(closing)
            closing.add_done_callback(self._closing_speech.discard)

    async def _microphone(self) -> None:
        hearing = self.backends.hearing
        await self.session.chat_subscribed.wait()
        await hearing.start()
        drain_until = time.monotonic() + 0.25
        for _ in range(1000):
            if time.monotonic() >= drain_until or await asyncio.to_thread(self.hardware.read_audio) is None:
                break
        gap_quiet_samples = 0
        recorder = None
        retry_notice_pending = False

        def transcription_failed():
            nonlocal retry_notice_pending
            retry_notice_pending = True
            log.warning("Reachy discarded an incomplete transcription; retry notice pending")

        def make_recorder(**options):
            nonlocal gap_quiet_samples
            if recorder is not None:
                recorder.abort()
            if self._playback.user_speaking:
                gap_quiet_samples = math.ceil(self.silence_s * self.hardware.sample_rate)
            return hearing.open_turn(self.hardware.sample_rate, silence_s=self.silence_s,
                                     vad=self.speech_gate, **options)

        recorder = make_recorder()
        recorder_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="reachy-recorder")
        last_slow_feed = 0.0
        async def record(sample):
            nonlocal last_slow_feed, gap_quiet_samples
            # Keep VAD ahead of transcription/TTS work in the shared executor.
            started = time.monotonic()
            feed = asyncio.get_running_loop().run_in_executor(recorder_worker, recorder.feed, sample)
            try:
                result = await asyncio.shield(feed)
            except asyncio.CancelledError:
                await asyncio.gather(feed, return_exceptions=True)
                raise
            endpoint_at = time.monotonic()
            heard = recorder.take()
            for _ in range(heard.failures):
                transcription_failed()
            if heard.transcript_lost:
                if heard.dropped:
                    result = None
                self._wake_strip_required = False
                self._wake_first_request_pending = False
                recorder.finish_initial_capture()
            recognition = heard.recognition
            if recognition is not None:
                recognitions.add(recognition)
            speaking = recorder.speech_active
            if speaking:
                gap_quiet_samples = 0
            else:
                gap_quiet_samples = max(0, gap_quiet_samples - len(sample))
            was_speaking = self._playback.user_speaking
            self._playback.set_user_speaking(speaking or gap_quiet_samples > 0)
            if self._playback.user_speaking != was_speaking:
                await self._plan(plan.UserSpeech(speaking=self._playback.user_speaking,
                                                 output=self._has_output(),
                                                 turn_running=turn_task is not None,
                                                 turns_waiting=bool(pending_turns)))
            partial = recorder.partial
            if partial is not None:
                await self._plan(plan.Partial(partial.text, partial.revision, output=self._has_output()))
            elapsed = time.monotonic() - started
            if elapsed > max(.1, len(sample) / self.hardware.sample_rate * 2) and started - last_slow_feed > 10:
                log.warning("Reachy microphone processing took %.3fs for %.3fs of audio",
                            elapsed, len(sample) / self.hardware.sample_rate)
                last_slow_feed = started
            return _RecordedTurn(result, recognition, endpoint_at) if result is not None else None

        wake = self.wake_detector
        if wake is not None:
            await asyncio.to_thread(wake.reset)
            self._wake_deadline = None
            self._wake_strip_required = False
            self._wake_first_request_pending = False
        pre_roll = deque()
        pre_roll_samples = 0
        pre_roll_limit = 3 * self.hardware.sample_rate
        def trim_pre_roll(limit: int) -> None:
            nonlocal pre_roll_samples
            while pre_roll_samples > limit:
                excess = pre_roll_samples - limit
                oldest = pre_roll.popleft()
                if len(oldest) > excess:
                    pre_roll.appendleft(oldest[excess:])
                    pre_roll_samples -= excess
                else:
                    pre_roll_samples -= len(oldest)
        wake_resampler = None
        if wake is not None and self.hardware.sample_rate != 16000:
            import av
            wake_resampler = av.AudioResampler(format="flt", layout="mono", rate=16000)

        async def detects_wake(sample) -> tuple[bool, int | None]:
            if wake_resampler is None:
                detected = await asyncio.to_thread(wake.feed, sample)
                event = getattr(wake, "last_detection", None) if detected else None
                tail = getattr(event, "post_wake_samples", None)
                if (type(tail) is int and 0 <= tail <= pre_roll_samples
                        and getattr(event, "epoch", None) == getattr(wake, "epoch", None)
                        and getattr(event, "feed_id", None) == getattr(wake, "feed_id", None)
                        and type(getattr(event, "epoch", None)) is int
                        and type(getattr(event, "feed_id", None)) is int):
                    return detected, tail
                return detected, None
            import av
            import numpy as np
            frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(sample[None, :]),
                                              format="flt", layout="mono")
            frame.sample_rate = self.hardware.sample_rate
            for chunk in wake_resampler.resample(frame):
                if await asyncio.to_thread(wake.feed, chunk.to_ndarray().reshape(-1)):
                    return True, None
            return False, None

        await self._plan(plan.Started())
        turn_task = (asyncio.create_task(self.turn(None))
                     if self.backends.reply_style is not ReplyStyle.MUSE_VOICE
                     and wake is None and not self.owns_chat else None)
        pending_turns = deque()
        recognitions = set()
        wake_epoch = 0
        active_wake_epoch = 0
        async def queue_turn(recorded: _RecordedTurn) -> None:
            nonlocal recorder
            if recorder.last_truncated:
                log.info("voice turn reached the 60-second recording limit; continuing capture")
            if len(pending_turns) < PENDING_TURN_LIMIT:
                recognition = recorded.recognition
                if recognition is None:
                    recognition = hearing.recognize(recorded.wav)
                    if recognition is not None:
                        recognitions.add(recognition)
                pending_turns.append(_QueuedTurn(recorded.wav, self._wake_strip_required, wake_epoch,
                                                recognition, recorded.endpoint_at))
                if self._wake_strip_required:
                    recorder.finish_initial_capture()
                self._wake_strip_required = False
                self._wake_first_request_pending = False
                log.info("Reachy queued a voice turn; %d pending", len(pending_turns))
            else:
                if recorded.recognition is not None:
                    recorded.recognition.cancel()
                    recognitions.discard(recorded.recognition)
                self._input_overflow = True
                log.warning("Reachy question queue full; saved turns retained, retry notice pending")
            if wake is not None and self.wake_timeout_s == 0:
                self._wake_deadline = None
                self._wake_strip_required = False
                self._wake_first_request_pending = False
                recorder = make_recorder()
                await asyncio.to_thread(wake.reset)
        capture = _CaptureBuffer(self.hardware.sample_rate, capacity_s=3.0)
        producer = asyncio.create_task(self._capture_microphone(capture), name="reachy-microphone-capture")
        speaker = asyncio.create_task(self._play_output(), name="reachy-speaker")
        ready = False
        blocked_episode = False
        try:
            while True:
                if speaker.done():
                    speaker.result()
                    raise ConnectionError("Reachy speaker ended")
                if retry_notice_pending and not self._output_queue.full():
                    self._output_queue.put_nowait(_SpeechJob(
                        None, TRANSCRIPTION_RETRY_CUE, expression="curious", state="listening"))
                    retry_notice_pending = False
                if turn_task is not None and turn_task.done():
                    outcome = turn_task.result()
                    turn_task = None
                    if wake is not None and active_wake_epoch == wake_epoch:
                        if outcome in (TurnOutcome.WAKE_CUE, TurnOutcome.ACCEPTED):
                            self._wake_first_request_pending = False
                            # A confirmed cue also opens time for a question in wake-every-turn mode.
                            window = self.wake_timeout_s or (10.0 if outcome is TurnOutcome.WAKE_CUE else 0.0)
                            self._wake_deadline = time.monotonic() + window
                        elif (self._wake_deadline is not None
                              and time.monotonic() >= self._wake_deadline and not recorder.active):
                            # False speech must not keep authorizing new audio forever.
                            # Already admitted WAVs retain their own wake boundary.
                            self._wake_deadline = None
                            self._wake_strip_required = False
                            self._wake_first_request_pending = False
                            recorder = make_recorder()
                            pre_roll.clear()
                            pre_roll_samples = 0
                            await asyncio.to_thread(wake.reset)
                            if wake_resampler is not None:
                                import av
                                wake_resampler = av.AudioResampler(format="flt", layout="mono", rate=16000)
                            log.info("Reachy's wake window closed after empty recognition and inactivity")
                    await self._plan(plan.TurnDone(
                        turns_waiting=bool(pending_turns), output=self._has_output(), recording=recorder.active,
                        wake_open=wake is not None and self._wake_deadline is not None))
                if turn_task is None and pending_turns:
                    request = pending_turns.popleft()
                    active_wake_epoch = request.wake_epoch
                    await self._plan(plan.Working(output=self._has_output(),
                                                  user_speaking=self._playback.user_speaking))
                    async def process(request=request):
                        try:
                            endpoint = await hearing.endpoint(request.wav, request.recognition)
                            if endpoint is None:
                                transcription_failed()
                                return TurnOutcome.EMPTY
                            if endpoint.text is None:
                                return await self.turn(endpoint.audio,
                                    wake_strip_required=request.wake_strip_required)
                            log.info("Reachy transcript ready for dispatch %.3fs after endpoint",
                                     time.monotonic() - request.endpoint_at)
                            return await self.turn(endpoint.audio,
                                wake_strip_required=request.wake_strip_required,
                                recognized_text=endpoint.text, defer_playback=True)
                        finally:
                            recognitions.discard(request.recognition)
                    turn_task = asyncio.create_task(process())
                if not ready and turn_task is None:
                    log.info("Reachy speech yield ready: microphone priority, ordered background replies")
                    log.info("Reachy continuous capture ready: silence %.1fs, pending limit %d, %s",
                             self.silence_s, PENDING_TURN_LIMIT,
                             "echo-cancelled duplex" if getattr(self.hardware, "echo_cancelled_input", False) is True
                             else "playback echo gate")
                    log.info("Reachy is ready. %s", "Say Hey Muse to open the microphone." if wake is not None
                             else "Speak naturally; pause to send your turn to Muse.")
                    ready = True
                if (wake is not None and not self._input_blocked() and turn_task is None
                        and not pending_turns and not self._has_output()):
                    if (self._wake_deadline is not None and time.monotonic() >= self._wake_deadline
                            and not recorder.active):
                        self._wake_deadline = None
                        self._wake_strip_required = False
                        self._wake_first_request_pending = False
                        recorder = make_recorder()
                        await asyncio.to_thread(wake.reset)
                        if wake_resampler is not None:
                            import av
                            wake_resampler = av.AudioResampler(format="flt", layout="mono", rate=16000)
                        await self._plan(plan.WakeClosed())
                        log.info("Reachy's wake window closed after inactivity")
                captured = await capture.get()
                if captured is None:
                    continue
                sample = captured.samples
                if captured.gap:
                    recorder = make_recorder()
                    pre_roll.clear()
                    pre_roll_samples = 0
                    if self._wake_deadline is not None:
                        # The acoustic invocation still authorizes fresh audio.
                        # All audio requiring a pre-wake text boundary was discarded.
                        self._wake_strip_required = False
                    if wake is not None:
                        await asyncio.to_thread(wake.reset)
                    if wake_resampler is not None:
                        import av
                        wake_resampler = av.AudioResampler(format="flt", layout="mono", rate=16000)
                    await self._plan(plan.CaptureGap(wake_open=self._wake_deadline is not None,
                                                     muted=self._muted, turn_running=turn_task is not None))
                    log.warning("Reachy microphone capture lost continuity (%s); speech detectors reset; %s",
                                captured.gap_reason or "upstream gap",
                                "microphone remains open" if self._wake_deadline is not None else "waiting for wake")
                if captured.discard or self._input_blocked():
                    # Unqualified speaker paths cannot stitch user speech across echo.
                    if not blocked_episode:
                        recorder = make_recorder()
                        pre_roll.clear()
                        pre_roll_samples = 0
                        if self._wake_first_request_pending:
                            self._wake_deadline = None
                            self._wake_strip_required = False
                            self._wake_first_request_pending = False
                        if wake is not None:
                            await asyncio.to_thread(wake.reset)
                        if wake_resampler is not None:
                            import av
                            wake_resampler = av.AudioResampler(format="flt", layout="mono", rate=16000)
                        blocked_episode = True
                    continue
                blocked_episode = False
                if wake is not None:
                    now = time.monotonic()
                    if self._wake_deadline is None:
                        pre_roll.append(sample.copy())
                        pre_roll_samples += len(sample)
                        trim_pre_roll(pre_roll_limit)
                        detected, wake_tail = await detects_wake(sample)
                        if not detected:
                            continue
                        wake_epoch += 1
                        self._wake_deadline = now + (self.wake_timeout_s or 10.0)
                        self._wake_strip_required = wake_tail is None
                        self._wake_first_request_pending = True
                        if self._has_output() or self._muted:
                            self._playback.set_user_speaking(True)
                        await self._plan(plan.WakeHeard())
                        await asyncio.to_thread(wake.reset)
                        log.info("Reachy heard its wake phrase; microphone open")
                        if wake_tail is None:
                            # Untimed detectors retain the exact text boundary guard.
                            recorder = make_recorder(initial_min_s=.06)
                        else:
                            # The acoustic detector supplies a fresh cut after the
                            # keyword. Discard all earlier audio and capture only
                            # the request; do not transcribe the invocation again.
                            trim_pre_roll(wake_tail)
                            recorder = make_recorder()
                            log.info("Wake handoff retained %.3fs of post-keyword audio; request capture open",
                                     wake_tail / self.hardware.sample_rate)
                        wav = None
                        for captured in pre_roll:
                            completed = await record(captured)
                            while completed is not None:
                                if wake_tail is not None:
                                    await queue_turn(completed)
                                    if self.wake_timeout_s == 0:
                                        break
                                else:
                                    if wav is not None and wav.recognition is not None:
                                        wav.recognition.cancel()
                                        recognitions.discard(wav.recognition)
                                    wav = completed
                                # Drain any same-buffer speech after an older pre-roll endpoint.
                                completed = await record(captured[:0])
                            if wake_tail is not None and self.wake_timeout_s == 0 and self._wake_deadline is None:
                                break
                        if recorder.active:
                            if wav is not None and wav.recognition is not None:
                                wav.recognition.cancel()
                                recognitions.discard(wav.recognition)
                            wav = None
                        pre_roll.clear()
                        pre_roll_samples = 0
                        if wake_tail is not None or wav is None:
                            continue
                    else:
                        wav = await record(sample)
                    if wav is None:
                        continue
                else:
                    active_before = recorder.active
                    wav = await record(sample)
                    if recorder.active != active_before:
                        await self._plan(plan.Recording(active=recorder.active,
                                                        turn_running=turn_task is not None))
                if wav is None:
                    continue
                while wav is not None:
                    await queue_turn(wav)
                    if wake is not None and self.wake_timeout_s == 0:
                        break
                    # Retain resampler/VAD state and drain endpoints in this SDK chunk.
                    wav = await record(sample[:0])
        finally:
            producer.cancel()
            await asyncio.gather(producer, return_exceptions=True)
            recorder.abort()
            if turn_task is not None:
                turn_task.cancel()
                await asyncio.gather(turn_task, return_exceptions=True)
            for recognition in recognitions:
                recognition.cancel()
            await asyncio.gather(*recognitions, return_exceptions=True)
            speaker.cancel()
            await asyncio.gather(speaker, return_exceptions=True)
            self._playback.set_user_speaking(False)
            recorder_worker.shutdown(wait=True)

    async def _capture_microphone(self, capture: _CaptureBuffer) -> None:
        """Drain the SDK independently of keyword inference and conversation work."""
        from musegadget.reachy_hardware import ReachyHardwareError
        last_sample = time.monotonic()
        previous_duration = None
        try:
            while True:
                muted_at_start = self._input_blocked()
                read = asyncio.create_task(asyncio.to_thread(self.hardware.read_audio))
                try:
                    sample = await asyncio.shield(read)
                except asyncio.CancelledError:
                    await asyncio.gather(read, return_exceptions=True)
                    raise
                now = time.monotonic()
                if sample is None or not len(sample):
                    if now - last_sample > 15:
                        raise ReachyHardwareError("Reachy microphone supplied no audio for 15 seconds")
                    await asyncio.sleep(.002)
                    continue
                duration = len(sample) / self.hardware.sample_rate
                # SDK audio has no PTS. A long wall-time stall is a conservative gap signal.
                gap = previous_duration is not None and now - last_sample > max(duration, previous_duration) + .25
                previous_duration = duration
                last_sample = now
                capture.push(sample, discard=muted_at_start or self._input_blocked(), gap=gap,
                             gap_reason="SDK read or capture scheduling stall" if gap else None)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            capture.fail(exc)

    async def turn(self, wav: bytes | None, *, wake_strip_required: bool | None = None,
                   recognized_text: str | None = None, defer_playback: bool = False) -> TurnOutcome:
        if wav is None and self.wake_detector is not None:
            return TurnOutcome.EMPTY
        self._defer_playback = defer_playback
        started = self._turn_started = time.monotonic()
        log.info("Reachy voice turn started")
        acknowledgement_task = None
        progress_task = None
        turn_cancelled = False
        completed = False
        async def finish_progress(*, cancel: bool = False) -> None:
            nonlocal progress_task
            if progress_task is not None:
                task, progress_task = progress_task, None
                if cancel:
                    task.cancel()
                result, = await asyncio.gather(task, return_exceptions=True)
                if isinstance(result, Exception):
                    from musegadget.reachy_hardware import ReachyHardwareError
                    if isinstance(result, ReachyHardwareError) and not turn_cancelled:
                        raise result
                    log.warning("Spoken progress failed: %s", type(result).__name__)
        async def stop_progress() -> None:
            if self._progress is not None:
                self._progress.stop()
            await finish_progress(cancel=True)
        async def announce(text: str) -> None:
            if defer_playback:
                await self._output_queue.put(_SpeechJob(None, text, started=started, state="listening"))
            else:
                await self._speak(None, text=text, state="listening", stream_segment=True)
        try:
            await self._plan(plan.TurnStarted(output=self._has_output(),
                                              user_speaking=self._playback.user_speaking))
            text = None
            if wav is not None and (self.backends.hearing.transcribes or recognized_text is not None):
                text = (recognized_text if recognized_text is not None
                        else await self.backends.hearing.transcribe(wav))
                strip_required = (self._wake_strip_required if wake_strip_required is None
                                  else wake_strip_required)
                if wake_strip_required is None:
                    self._wake_strip_required = False
                if not text.strip():
                    log.info("Speech recognition outcome empty")
                    if self.wake_detector is not None and strip_required:
                        await announce(WAKE_CUE)
                        completed = True
                        return TurnOutcome.WAKE_CUE
                    log.info("No words recognized in microphone input; listening again")
                    completed = True
                    return TurnOutcome.EMPTY
                if self.wake_detector is not None:
                    if strip_required:
                        question = question_after_wake(text, self.wake_detector.phrase)
                        # The wake detector confirms the acoustic trigger, but ASR must identify its boundary.
                        log.info("Wake transcription outcome %s", "missing_boundary" if question is None
                                 else "request" if question else "wake_only")
                        text = question or ""
                    else:
                        text = strip_wake_prefix(text, self.wake_detector.phrase)
                    if not text:
                        await announce(WAKE_CUE)
                        completed = True
                        return TurnOutcome.WAKE_CUE
                log.info("Recognized %d characters from Reachy's microphone", len(text))
                await self._plan(plan.Working(output=self._has_output(),
                                              user_speaking=self._playback.user_speaking))
            self.tracker = ReplyTracker(self.session_id, style=self.backends.reply_style,
                                        owns_chat=self.owns_chat, replay_scope=self._replay_scope)
            self._logged_activity = None
            self._activity_log_count = 0
            self._logged_progress_status = None
            self._muse_status = None
            self._replies = asyncio.Queue(maxsize=32)
            request_text = text or ""
            response_deadline = time.monotonic() + self.reply_timeout_s
            options = {}
            if self.backends.reply_style is ReplyStyle.MUSE_VOICE:
                options["output_modality"] = "voice"
            if text is not None and not self._has_output() and not self._playback.user_speaking:
                line = await self.backends.narrator.acknowledge(text)
                if line is not None:
                    log.info("Reachy selected a question acknowledgement %.1fs after the voice turn",
                             time.monotonic() - started)
                    acknowledgement_task = asyncio.create_task(self._speak(
                        None, text=line.text, state="thinking", role=line.role))
            if wav is None:
                context = self._voice_context()
                setup = (" For setup, return one sentence frame with text 'Ready to talk' and expression 'nod'. "
                         "Keep using this sentence protocol for subsequent spoken messages."
                         if self.backends.reply_style is ReplyStyle.EXPRESSIVE_JSON else
                         " For setup, say 'Ready to talk' and append [reachy:nod]. "
                         "Keep using this expression channel for subsequent spoken messages.")
                ack = await self.session.send_chat(
                    context + setup,
                    self.session_id)
            elif text is not None:
                if (not self.owns_chat and self.wake_detector is not None
                        and not self._voice_context_sent):
                    context = self._voice_context()
                    text = context + "\n\nThe user's spoken request is: " + text
                ack = await self.session.send_chat(text, self.session_id, **options)
            else:
                ack = await self.session.send_voice(wav, self.session_id, **options)
            if not ack.get("ok"):
                raise ConnectionError(f"Muse rejected conversation request: HTTP {ack.get('status')}")
            response = ack.get("response")
            if not isinstance(response, dict):
                raise ValueError("Muse did not acknowledge the voice note")
            self._voice_context_sent = True
            if wav is not None and self.backends.progress_voice is not None:
                self._progress = self.backends.narrator.progress(request_text, time.monotonic())
            self._queue_replies(self.tracker.acknowledge(response), self.tracker)
            self._log_backend_activity()
            if wav is None:
                log.info("Muse accepted Reachy's expression setup")
            else:
                log.info("Muse accepted a %.1f-second voice turn", max(0, len(wav) - 44) / 32000)
            if acknowledgement_task is not None:
                await acknowledgement_task
            played = False
            waiting_for_segment = False
            while True:
                now = time.monotonic()
                if self.tracker.task_finished:
                    await stop_progress()
                if self.tracker.complete(now, played) and self._replies.empty():
                    completed = True
                    return TurnOutcome.ACCEPTED
                if not played and self._replies.empty() and self.tracker.finished_without_text(now):
                    await stop_progress()
                    log.warning("Muse finished without reply text; returning to listening")
                    if self.backends.voice.speaks_text:
                        from musegadget.reachy_hardware import ReachyHardwareError
                        try:
                            await announce("Muse returned an empty reply. Please try again.")
                        except ReachyHardwareError:
                            raise
                        except Exception as exc:
                            log.warning("Empty-reply announcement failed: %s", type(exc).__name__)
                    completed = True
                    return TurnOutcome.ACCEPTED
                if now >= response_deadline:
                    raise TimeoutError("Muse's voice reply did not complete in time")
                try:
                    reply = await asyncio.wait_for(self._replies.get(), 0.1)
                except asyncio.TimeoutError:
                    if progress_task is not None and progress_task.done():
                        await finish_progress()
                    status = self._muse_status
                    if status is not None:
                        await self._plan(plan.MuseStatus(
                            status.phase, status.activity.current_text if status.activity else None,
                            output=self._has_output()))
                    if (self._input_overflow and progress_task is None and self.backends.voice.speaks_text
                            and not self._has_output() and not self._playback.user_speaking):
                        self._input_overflow = False
                        progress_task = asyncio.create_task(self._speak(
                            None, text=INPUT_OVERFLOW_CUE, state="thinking", stream_segment=True))
                    if (self._progress is not None and progress_task is None and not played
                            and not self._has_output() and not self._playback.user_speaking):
                        progress = self._progress.take(time.monotonic())
                        if progress is not None:
                            log.info("Reachy selected %s progress %.1fs after the voice turn",
                                     progress.source, time.monotonic() - started)
                            progress_task = asyncio.create_task(self._speak(
                                None, text=progress.text, state="thinking", stream_segment=True,
                                voice=self.backends.progress_voice, progress=True))
                    if (self.backends.reply_style is ReplyStyle.EXPRESSIVE_JSON and not waiting_for_segment
                            and not self._has_output() and not self._playback.user_speaking):
                        await self._plan(plan.Working(output=False, user_speaking=False))
                        waiting_for_segment = True
                    continue
                if isinstance(reply, TaskFinished):
                    continue
                log.info("%s next Muse reply %.1fs after the voice turn",
                         "Queueing" if defer_playback else "Playing", now - started)
                if not played and wav is not None:
                    await self._plan(plan.AnswerArrived(output=self._has_output()))
                if isinstance(reply, SpeechSegment):
                    message = self.tracker.messages[reply.message_id]
                    if message["stream_revised"] or reply.index < message["stream_skip_before"]:
                        self._discard_prepared((reply.message_id, reply.index))
                        continue
                    await stop_progress()
                    def is_current(message=message, reply=reply):
                        return not message["stream_revised"] and reply.index >= message["stream_skip_before"]
                    if not is_current():
                        self._discard_prepared((reply.message_id, reply.index))
                        continue
                    def mark_spoken(message=message):
                        message["stream_spoken"] += 1
                    prepared = self._prepared_speech.get((reply.message_id, reply.index))
                    if defer_playback:
                        self._prepared_speech.pop((reply.message_id, reply.index), None)
                        queued_at = time.monotonic()
                        await self._output_queue.put(_SpeechJob(
                            reply.message_id, reply.text, reply.expression, prepared,
                            is_current, mark_spoken, started))
                        response_deadline += time.monotonic() - queued_at
                        self._queue_replies([], self.tracker)
                        played = True
                        continue
                    speaking = self._speak(reply.message_id, text=reply.text, expression=reply.expression,
                                           stream_segment=True,
                                           prepared_stream=prepared,
                                           is_current=is_current, on_start=mark_spoken)
                else:
                    await stop_progress()
                    if defer_playback:
                        line, = self.backends.narrator.lines(
                            self.tracker.messages[reply]["text"], self.backends.reply_style, message_id=reply)
                        queued_at = time.monotonic()
                        await self._output_queue.put(_SpeechJob(
                            reply, line.text, line.expression.value if line.expression else None,
                            started=started))
                        response_deadline += time.monotonic() - queued_at
                        played = True
                        continue
                    prepared = None
                    speaking = self._speak(reply)
                waiting_for_segment = False
                speech_started = time.monotonic()
                try:
                    await self._speech_with_timeout(speaking)
                except _SpeechSuperseded:
                    self._discard_prepared((reply.message_id, reply.index))
                    continue
                finally:
                    # Response waiting and draining accepted audio have separate budgets.
                    response_deadline += time.monotonic() - speech_started
                    if prepared is not None:
                        self._prepared_speech.pop((reply.message_id, reply.index), None)
                        await prepared.aclose()
                if isinstance(reply, SpeechSegment):
                    # Refill the single sentence of lookahead after the current one finishes.
                    self._queue_replies([], self.tracker)
                played = True
        except asyncio.CancelledError:
            turn_cancelled = True
            raise
        finally:
            try:
                try:
                    await stop_progress()
                finally:
                    self._progress = None
                    if acknowledgement_task is not None:
                        acknowledgement_task.cancel()
                        await asyncio.gather(acknowledgement_task, return_exceptions=True)
                        if not turn_cancelled and not acknowledgement_task.cancelled():
                            acknowledgement_task.result()
            finally:
                prepared, self._prepared_speech = self._prepared_speech, {}
                await asyncio.gather(*(audio.aclose() for audio in prepared.values()), return_exceptions=True)
                await asyncio.gather(*self._closing_speech, return_exceptions=True)
                self._closing_speech.clear()
                if self.tracker is not None:
                    self.tracker.retire()
                self.tracker = None
                try:
                    # Deferred output owns its cleanup; backend failures cannot
                    # pause an earlier reply's shared GStreamer pipeline.
                    if (not defer_playback and (not completed
                            or getattr(self.hardware, "echo_cancelled_input", False) is not True)):
                        await self._clear_idle_audio()
                except Exception:
                    if not turn_cancelled:
                        raise
                finally:
                    log.info("Reachy voice turn ended")

    def _voice_context(self) -> str:
        from musegadget.reachy_expression import voice_context
        return voice_context(
            stream_replies=self.backends.reply_style is ReplyStyle.EXPRESSIVE_JSON,
            motion_enabled=getattr(self.hardware, "motion_enabled", True),
            antenna_mode=getattr(self.hardware, "antenna_mode", "both"),
            face_tracking_enabled=getattr(self.hardware, "face_tracking_enabled", False),
        )

    async def _speak(self, message_id: str | None, *, text: str | None = None,
                     state: str = "speaking",
                     expression: str | None = None, stream_segment: bool = False,
                     voice=None, progress: bool = False, prepared_stream=None,
                     is_current=None, on_start=None, turn_started: float | None = None,
                     role: str | None = None) -> None:
        # Direct acknowledgements/progress and deferred replies share this owner.
        # Interrupted audio is flushed before a waiting speaker acquires it.
        self._speech_pending += 1
        try:
            async with self._speaker_lock:
                try:
                    await self._speak_owned(message_id, text=text, state=state,
                        expression=expression,
                        stream_segment=stream_segment, voice=voice,
                        progress=progress, prepared_stream=prepared_stream,
                        is_current=is_current, on_start=on_start, turn_started=turn_started,
                        role=role)
                except BaseException as exc:
                    try:
                        await self._clear_owned_audio(propagate_cancel=False)
                    except Exception:
                        if not isinstance(exc, asyncio.CancelledError):
                            raise
                    raise
        finally:
            self._speech_pending -= 1

    async def _clear_owned_audio(self, *, propagate_cancel: bool = True) -> None:
        """Flush under speaker ownership, joining any physical clear on cancellation."""
        if self._audio_clear_attempted:
            return
        self._audio_clear_attempted = True
        task = asyncio.create_task(asyncio.to_thread(self.hardware.clear_audio))
        cancelled = False
        while True:
            try:
                await asyncio.shield(task)
                break
            except asyncio.CancelledError:
                if task.cancelled():
                    raise
                cancelled = True
        self._echo_until = min(self._echo_until, time.monotonic() + ECHO_TAIL_S)
        if cancelled and propagate_cancel:
            raise asyncio.CancelledError

    async def _clear_idle_audio(self) -> None:
        # A backend turn has no authority over another turn's active playback.
        # Count awakened lock waiters too: release can precede their acquisition.
        if not self._speech_pending and not self._speaker_lock.locked():
            async with self._speaker_lock:
                await self._clear_owned_audio()

    async def _speak_owned(self, message_id: str | None, *, text: str | None = None,
                           state: str = "speaking",
                           expression: str | None = None, stream_segment: bool = False,
                           voice=None, progress: bool = False, prepared_stream=None,
                           is_current=None, on_start=None, turn_started: float | None = None,
                           role: str | None = None) -> None:
        voice = voice if voice is not None else self.backends.voice
        samples_played = 0
        write_timing = {"dispatch_ms_max": 0.0, "write_ms_max": 0.0, "return_ms_max": 0.0}
        started = self._turn_started if turn_started is None else turn_started
        async def hardware_call(function, *args, **kwargs):
            task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
            cancelled = False
            while True:
                try:
                    result = await asyncio.shield(task)
                    break
                except asyncio.CancelledError:
                    if task.cancelled():
                        raise
                    # Repeated cancellation must still wait for the pending hardware command.
                    cancelled = True
            if cancelled:
                raise asyncio.CancelledError
            return result
        async def begin() -> None:
            if is_current is not None and not is_current():
                raise _SpeechSuperseded
            if progress:
                await hardware_call(self.hardware.cue_progress)
            await hardware_call(self.hardware.set_state, state, expression=expression)
            if is_current is not None and not is_current():
                raise _SpeechSuperseded
            if on_start is not None:
                on_start()
            if started is not None:
                log.info("%s started %.1fs after the voice turn",
                         _SPEECH_LABELS[line.role][0],
                         time.monotonic() - started)
            if expression and expression not in ("nod", "shake"):
                log.info("Muse expressed %s while Reachy spoke", expression)
        async def pause() -> None:
            self._muted = False
            await hardware_call(self.hardware.set_state, "listening")
            log.info("Reachy speech paused for microphone input")
        async def resume() -> None:
            await hardware_call(self.hardware.set_state, state, expression=expression)
            log.info("Reachy speech resumed after microphone silence")
        async def push(samples) -> None:
            nonlocal samples_played
            self._audio_clear_attempted = False
            self._muted = True
            self._echo_until = time.monotonic() + .13 + ECHO_TAIL_S
            if self.audio_diagnostics:
                dispatched = time.monotonic()
                entered = finished = dispatched
                def measured_write():
                    nonlocal entered, finished
                    entered = time.monotonic()
                    try:
                        self.hardware.play_audio(samples)
                    finally:
                        finished = time.monotonic()
                try:
                    await hardware_call(measured_write)
                finally:
                    resumed = time.monotonic()
                    for key, elapsed in (("dispatch_ms_max", entered - dispatched),
                                         ("write_ms_max", finished - entered),
                                         ("return_ms_max", resumed - finished)):
                        write_timing[key] = max(write_timing[key], elapsed * 1000)
            else:
                await hardware_call(self.hardware.play_audio, samples)
            samples_played += len(samples)
        player = PCMPlayer(self.hardware.output_sample_rate, push, self._playback,
                           on_start=begin, on_pause=pause, on_resume=resume,
                           diagnostics=self.audio_diagnostics)
        if text is None and self.tracker is not None:
            line, = self.backends.narrator.lines(
                self.tracker.messages[message_id]["text"], self.backends.reply_style, message_id=message_id)
            text, expression = line.text, line.expression.value if line.expression else None
        text = text or ""
        line = _line(message_id, text, expression, progress=progress, role=role)
        if voice.speaks_text and not text:
            if expression:
                await self._express(expression)
                return
            raise ValueError("Muse returned an empty spoken response")
        if prepared_stream is not None:
            speech = prepared_stream
        else:
            speech = voice.stream(line, self.hardware.output_sample_rate)
        try:
            try:
                async for samples in speech:
                    await player.play(samples)
            finally:
                await speech.aclose()
            if samples_played == 0:
                if is_current is not None and not is_current():
                    raise _SpeechSuperseded
                raise ValueError("Muse returned no playable speech")
            await player.finish()
            if not stream_segment:
                await asyncio.sleep(.25)
                await hardware_call(self.hardware.set_state, "thinking")
            log.info("Reachy spoke %.1f seconds of %s",
                     samples_played / self.hardware.output_sample_rate,
                     _SPEECH_LABELS[line.role][1])
        except BaseException:
            if self._muted:
                # Hold the unknown echo path closed until its owner flushes output.
                self._echo_until = math.inf
            raise
        finally:
            self._muted = False
            if self.audio_diagnostics:
                log.info("Reachy playback timing %s", {**player.metrics, **write_timing})

    async def _express(self, expression: str) -> None:
        result = await asyncio.to_thread(self.hardware.run_command, "reachy.expression",
                                        {"name": expression}, 20000)
        if result.get("ok"):
            log.info("Muse expressed %s through Reachy", expression)
        else:
            log.warning("Reachy rejected Muse expression %s: %s", expression, result.get("error"))


def _backends_required():
    raise TypeError("ReachyService requires backends")


@dataclass
class ReachyService(Service):
    session_id: str | None = None
    prepare_chat: Callable[[LinkSession, str], Awaitable[str]] | None = None
    owns_chat: bool = False
    silence_s: float = 2.0
    # Required: Python 3.9 dataclasses cannot follow inherited defaults with a field that has none.
    backends: Callable[[LinkSession], Backends] = field(default_factory=_backends_required)
    audio_diagnostics: bool = False
    wake_detector: object | None = None
    wake_timeout_s: float = 10.0
    speech_gate: object | None = None

    async def _session(self, vm: dict, pairing: dict) -> tuple[Outcome, float]:
        from musegadget.reachy_hardware import COMMAND_SPECS, ReachyHardwareError

        hardware = self.executor
        device = DeviceDescription(self.identity.node_id, "Reachy Mini", __version__, COMMAND_SPECS)
        session = LinkSession(noise_host=pairing.get("noise_host") or DEFAULT_NOISE_HOST,
                              vm_id=vm["vm_id"] or vm["vm_name"],
                              vm_auth_token=vm["vm_auth_token"], device=device,
                              run_command=hardware.run_command)
        self._current = session
        stop_link = asyncio.Event()
        async def converse():
            await session.registered.wait()
            session_id = self.session_id
            if self.prepare_chat is not None:
                session_id = await self.prepare_chat(session, vm["vm_id"] or vm["vm_name"])
            if self.owns_chat and (not isinstance(session_id, str) or not session_id):
                raise ValueError("Reachy's dedicated side chat was not selected")
            voice = VoiceConversation(session, hardware, backends=self.backends(session),
                                      session_id=session_id, silence_s=self.silence_s,
                                      wake_detector=self.wake_detector, wake_timeout_s=self.wake_timeout_s,
                                      owns_chat=self.owns_chat, speech_gate=self.speech_gate,
                                      audio_diagnostics=self.audio_diagnostics)
            await voice.run()
        link_task = asyncio.create_task(session.run(stop_link))
        voice_task = asyncio.create_task(converse())
        stop_task = asyncio.create_task(self._stop.wait())
        started = time.monotonic()
        outcome = Outcome.CLOSED
        try:
            done, _ = await asyncio.wait({link_task, voice_task, stop_task},
                                         return_when=asyncio.FIRST_COMPLETED)
            if stop_task in done:
                outcome = Outcome.STOPPED
            elif link_task in done:
                outcome = link_task.result()
            else:
                voice_task.result()
        except ReachyHardwareError:
            raise
        except Exception as exc:
            log.warning("Reachy conversation ended: %s: %s", type(exc).__name__, exc)
            await asyncio.to_thread(hardware.set_state, "error")
        finally:
            stop_link.set()
            voice_task.cancel()
            stop_task.cancel()
            await asyncio.gather(voice_task, stop_task, return_exceptions=True)
            with contextlib.suppress(Exception):
                await asyncio.wait_for(link_task, 5)
            self._current = None
        return outcome, time.monotonic() - (session.registered_at or started)
