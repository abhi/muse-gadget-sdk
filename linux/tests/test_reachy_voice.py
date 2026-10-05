# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

from __future__ import annotations

import asyncio
import io
import json
import queue
import threading
import time
import wave

import pytest

from musegadget import reachy_local_backends, reachy_voice
from musegadget.reachy_capabilities import Mode, ReplyStyle
from musegadget.reachy_local_backends import backends_for
from musegadget.reachy_progress import BackendActivity, BackendStatus
from musegadget.reachy_voice import (
    BackendStatusSegment, ProgressSegment, ReplayScope, ReplyTracker, SpeechSegment, TaskFinished, TurnOutcome,
    VoiceConversation,
)


def sentence_frame(text, expression="neutral"):
    return json.dumps({"text": text, "expression": expression}) + "\n"


def progress_frame(text="Looking at flights now."):
    return json.dumps({"kind": "progress", "text": text, "expression": "thinking"}) + "\n"


def status_segment(activity=None, phase="working"):
    return BackendStatusSegment(BackendStatus(phase, activity))


def owned_tracker(scope=None, user_id="user-1"):
    tracker = ReplyTracker("robot-chat", style=ReplyStyle.EXPRESSIVE_JSON, owns_chat=True, replay_scope=scope)
    tracker.acknowledge({"message_id": user_id, "session_id": "robot-chat", "is_thread": True})
    return tracker


def test_retired_turn_ids_reject_late_status_frames_and_terminals_before_ack():
    scope = ReplayScope()
    old = owned_tracker(scope)
    old.event(chat_event("task.status", parent=None, task_id="old-task", status="running"))
    old.event(chat_event("agent.status", "old-worker", parent=None,
                         activity_code="working", activity_text="Searching web"))
    old.event(chat_event("delta.text_append", "old-answer", parent=None, text=progress_frame()))
    old.retire()
    current = ReplyTracker("robot-chat", style=ReplyStyle.EXPRESSIVE_JSON, owns_chat=True, replay_scope=scope)
    events = [
        chat_event("agent.status", "old-worker", parent=None,
                   activity_code="working", activity_text="Searching web"),
        chat_event("delta.text_append", "old-answer", parent="old-answer", text=progress_frame()),
        chat_event("delta.text_append", "previously-unseen-answer", parent="user-1", text=progress_frame()),
        chat_event("task.status", parent=None, task_id="old-task", status="completed"),
    ]
    for event in events:
        assert current.event(event) == []
    # An unknown message's parent may be authorized by the still-pending ACK.
    assert current.pending == [events[2]]
    assert current.acknowledge({"message_id": "user-2", "session_id": "robot-chat", "is_thread": True}) == []
    for event in events:
        assert current.event(event) == []
    assert current.messages == {} and not current.busy and not current.task_finished


@pytest.mark.parametrize("session_id", [None, "main", "another-side"])
@pytest.mark.parametrize("acknowledged", [False, True])
def test_owned_status_cannot_mutate_or_buffer_without_exact_chat(session_id, acknowledged):
    tracker = owned_tracker() if acknowledged else ReplyTracker("robot-chat", owns_chat=True)
    for name, details in [
        ("task.status", {"task_id": "other-task", "status": "completed"}),
        ("agent.status", {"activity_code": "working", "activity_text": "Searching web"}),
    ]:
        assert tracker.event(chat_event(name, session_id=session_id, **details)) == []
    assert not tracker.busy and not tracker.task_finished and tracker.pending == []
    assert tracker.observed_ids == set()


def test_unseen_terminal_snapshot_is_ignored_and_retired_but_running_task_finishes():
    scope = ReplayScope()
    tracker = owned_tracker(scope)
    assert tracker.event(chat_event("task.status", parent=None, task_id="startup-task", status="completed")) == []
    assert not tracker.task_finished and not tracker.finished_without_text(time.monotonic() + 10)
    tracker.event(chat_event("task.status", parent=None, task_id="current-task", status="running"))
    assert tracker.busy
    tracker.event(chat_event("task.status", parent=None, task_id="current-task", status="completed"))
    assert tracker.task_finished and not tracker.busy
    tracker.retire()
    assert {("task", "startup-task"), ("task", "current-task")} <= scope.ids


def test_explicit_current_ancestry_allows_terminal_without_running_event():
    tracker = owned_tracker()
    tracker.event(chat_event("task.status", task_id="current-task", status="errored"))
    assert tracker.task_finished and not tracker.busy


@pytest.mark.parametrize("first_status", ["completed", "failed", "errored", "cancelled", "canceled"])
def test_owned_turn_remains_active_until_all_running_tasks_finish(first_status):
    tracker = owned_tracker(ReplayScope())
    tracker.event(chat_event("task.status", parent=None, task_id="root-task", status="running"))
    tracker.event(chat_event("task.status", parent=None, task_id="sub-task", status="running"))
    tracker.event(chat_event("task.status", parent=None, task_id="sub-task", status=first_status))
    assert tracker.running_task_ids == {"root-task"}
    assert tracker.busy and not tracker.task_finished
    assert not tracker.finished_without_text(time.monotonic() + 10)
    tracker.event(chat_event("agent.status", "root-worker", parent=None, activity_code="idle"))
    assert tracker.busy and not tracker.task_finished
    assert tracker.event(chat_event("agent.status", "root-worker", parent=None,
                                    activity_code="working", activity_text="Searching web")) == [
        status_segment(BackendActivity("Searching the web now.",
                                               "Muse's last reported step was searching the web."))]
    tracker.event(chat_event("task.status", parent=None, task_id="root-task", status="completed"))
    assert tracker.running_task_ids == set()
    assert tracker.task_finished and not tracker.busy
    assert tracker.finished_without_text(time.monotonic() + 10)


@pytest.mark.parametrize("spelling", ["canceled", "cancelled"])
@pytest.mark.parametrize("owned", [False, True])
def test_both_task_cancellation_spellings_are_terminal(spelling, owned):
    tracker = owned_tracker() if owned else ReplyTracker("robot-chat")
    if not owned:
        tracker.acknowledge({"message_id": "user-1"})
    tracker.event(chat_event("task.status", parent=None, task_id="current-task", status="running"))
    tracker.event(chat_event("task.status", parent=None, task_id="current-task", status=spelling))
    assert tracker.task_finished and not tracker.busy and tracker.running_task_ids == set()


@pytest.mark.parametrize("spelling", ["canceled", "cancelled"])
def test_agent_cancellation_code_does_not_keep_a_shared_turn_busy(spelling):
    tracker = ReplyTracker("robot-chat")
    tracker.acknowledge({"message_id": "user-1"})
    tracker.event(chat_event("agent.status", activity_code=spelling))
    assert not tracker.busy


def test_current_worker_id_can_report_repeated_stages_within_a_turn():
    tracker = owned_tracker(ReplayScope())
    first = tracker.event(chat_event("agent.status", "worker-1", parent=None,
                                     activity_code="working", activity_text="Searching web"))
    second = tracker.event(chat_event("agent.status", "worker-1", parent=None,
                                      activity_code="working", activity_text="Searching sources"))
    assert first == [status_segment(BackendActivity(
        "Searching the web now.", "Muse's last reported step was searching the web."))]
    assert second == [status_segment(BackendActivity(
        "Searching sources now.", "Muse's last reported step was searching sources."))]


def test_retired_id_memory_is_bounded():
    scope = ReplayScope(limit=4)
    scope.retire(("task", str(index)) for index in range(100))
    assert scope.ids == {("task", str(index)) for index in range(96, 100)}
    assert len(scope.order) == 4
    scope.retire([("task", "99")])
    assert len(scope.order) == 4


def test_owned_active_id_memory_is_bounded():
    tracker = owned_tracker()
    with pytest.raises(ValueError, match="too many activity identifiers"):
        for index in range(257):
            tracker.event(chat_event("agent.status", str(index), parent=None,
                                     activity_code="working", activity_text="is working"))


@pytest.mark.parametrize("before_ack", [False, True])
def test_current_ack_can_authorize_a_retired_parent_without_unretiring_old_messages(before_ack):
    scope = ReplayScope()
    scope.retire([("message", "acknowledged-root")])
    tracker = ReplyTracker("robot-chat", style=ReplyStyle.EXPRESSIVE_JSON, owns_chat=True, replay_scope=scope)
    event = chat_event("delta.text_append", "current-reply", parent="acknowledged-root", text=progress_frame())
    ack = {"message_id": "user-2", "reply_to_message_id": "acknowledged-root",
           "session_id": "robot-chat", "is_thread": True}
    if before_ack:
        assert tracker.event(event) == []
        assert tracker.acknowledge(ack) == [ProgressSegment("Looking at flights now.")]
    else:
        tracker.acknowledge(ack)
        assert tracker.event(event) == [ProgressSegment("Looking at flights now.")]
    assert tracker.event(chat_event("delta.text_append", "acknowledged-root", parent="user-2",
                                    text=progress_frame())) == []


def test_owned_ack_identifiers_obey_retirement_memory_limit():
    tracker = ReplyTracker("robot-chat", owns_chat=True)
    with pytest.raises(ValueError, match="oversized chat identifier"):
        tracker.acknowledge({"message_id": "x" * 513, "session_id": "robot-chat", "is_thread": True})


def test_idle_subscription_snapshots_are_retired_before_the_next_ack():
    async def scenario():
        session = FakeSession()
        conversation = VoiceConversation(session, FakeHardware(),
                                         backends=backends_for(Mode.MUSE_VOICE, session),
                                         session_id="robot-chat", owns_chat=True)
        subscriber = asyncio.create_task(conversation._subscribe())
        try:
            await session.events.put(chat_event("task.status", parent=None, task_id="idle-task", status="completed"))
            await session.delivered.get()
            await session.events.put(chat_event("agent.status", "idle-worker", parent=None, seq=2,
                                                activity_code="working", activity_text="Searching web"))
            await session.delivered.get()
            tracker = owned_tracker(conversation._replay_scope)
            conversation.tracker = tracker
            await session.events.put(chat_event("task.status", parent=None, task_id="idle-task", status="running", seq=3))
            await session.delivered.get()
            await session.events.put(chat_event("agent.status", "idle-worker", parent=None, seq=4,
                                                activity_code="working", activity_text="Searching sources"))
            await session.delivered.get()
            assert not tracker.busy and tracker.messages == {}
        finally:
            await cancel_task(subscriber)
    asyncio.run(bounded(scenario()))


def test_cancelled_unacknowledged_owned_turn_retires_pending_ids(monkeypatch):
    monkeypatch.setattr(reachy_voice, "ECHO_TAIL_S", 0)
    async def scenario():
        started = asyncio.Event()
        class BlockedSession(FakeSession):
            async def send_voice(self, wav, session_id, **kwargs):
                started.set()
                await asyncio.Event().wait()
        session = BlockedSession()
        conversation = VoiceConversation(session, FakeHardware(),
                                         backends=backends_for(Mode.MUSE_VOICE, session),
                                         session_id="robot-chat", owns_chat=True)
        turn = asyncio.create_task(conversation.turn(b"wav"))
        await started.wait()
        tracker = conversation.tracker
        tracker.event(chat_event("task.status", parent=None, task_id="cancelled-task", status="running"))
        tracker.event(chat_event("delta.message_start", "cancelled-message", parent=None))
        await cancel_task(turn)
        assert conversation.tracker is None
        assert {("task", "cancelled-task"), ("message", "cancelled-message")} <= conversation._replay_scope.ids
    asyncio.run(bounded(scenario()))


def test_progress_is_ephemeral_and_never_counts_as_an_answer():
    tracker = ReplyTracker("robot-chat", style=ReplyStyle.EXPRESSIVE_JSON)
    tracker.acknowledge({"message_id": "user-1"})
    progress = progress_frame()
    assert tracker.event(chat_event("delta.text_append", "reply-1", text=progress)) == [
        ProgressSegment("Looking at flights now.")]
    message = tracker.messages["reply-1"]
    assert not message["queued"] and message["stream_index"] == 0
    answer = sentence_frame("Here are your options.", "happy")
    assert tracker.event(chat_event("delta.text_append", "reply-1", text=answer)) == [
        SpeechSegment("reply-1", 0, "Here are your options.", "happy")]
    assert tracker.event(chat_event("delta.message_done", "reply-1", content=answer)) == []


@pytest.mark.parametrize("parent", [None, "reply-1"])
def test_parentless_or_self_parent_progress_is_not_current_turn_proof(parent):
    tracker = ReplyTracker("robot-chat", style=ReplyStyle.EXPRESSIVE_JSON)
    tracker.acknowledge({"message_id": "user-1"})
    tracker.event(chat_event("delta.message_start", "reply-1", parent=None))
    assert tracker.event(chat_event("delta.text_append", "reply-1", parent=parent,
                                    text=progress_frame())) == []
    assert not tracker.messages["reply-1"]["progress_linked"]


def test_only_explicitly_linked_public_status_can_offer_a_stage():
    tracker = ReplyTracker("robot-chat", style=ReplyStyle.EXPRESSIVE_JSON)
    tracker.acknowledge({"message_id": "user-1"})
    assert tracker.event(chat_event("agent.status", parent=None,
                                    activity_text="Searching flights", activity_code="working")) == []
    assert tracker.event(chat_event("agent.status", parent="user-1",
                                    activity_text="Searching flights", activity_code="working")) == [
        status_segment(BackendActivity("Looking at flights now.",
                                               "Muse's last reported step was looking at flights."))]
    assert tracker.event(chat_event("agent.status", parent="user-1",
                                    activity_text="is working", activity_code="working")) == [status_segment(None)]
    assert tracker.event(chat_event("agent.status", parent="user-1",
                                    activity_text="Checking fares https://private.test",
                                    activity_code="working")) == [status_segment(None)]


@pytest.mark.parametrize("parent", [None, "reply-1"])
def test_dedicated_side_chat_attributes_exact_session_progress_without_parent_ancestry(parent):
    tracker = ReplyTracker("robot-chat", style=ReplyStyle.EXPRESSIVE_JSON, owns_chat=True)
    tracker.acknowledge({"message_id": "user-1", "session_id": "robot-chat", "is_thread": True})
    assert tracker.event(chat_event("delta.text_append", "reply-1", parent=parent,
                                    text=progress_frame())) == [ProgressSegment("Looking at flights now.")]
    assert not tracker.messages["reply-1"]["queued"]
    assert tracker.event(chat_event("agent.status", parent=None,
                                    activity_text="Searching web", activity_code="working")) == [
        status_segment(BackendActivity("Searching the web now.",
                                               "Muse's last reported step was searching the web."))]
    assert tracker.event(chat_event("agent.status", parent=None,
                                    activity_text="is responding", activity_code="responding")) == [
        status_segment(None, "responding")]


@pytest.mark.parametrize("session_id", [None, "main", "other-device-chat"])
def test_dedicated_side_chat_never_attributes_other_or_missing_session_progress(session_id):
    tracker = ReplyTracker("robot-chat", style=ReplyStyle.EXPRESSIVE_JSON, owns_chat=True)
    tracker.acknowledge({"message_id": "user-1", "session_id": "robot-chat", "is_thread": True})
    assert tracker.event(chat_event("agent.status", parent=None, session_id=session_id,
                                    activity_text="Searching web", activity_code="working")) == []
    assert tracker.event(chat_event("delta.text_append", "reply-1", parent=None, session_id=session_id,
                                    text=progress_frame())) == []


@pytest.mark.parametrize("ack", [
    {"session_id": "main", "is_thread": True},
    {"session_id": "robot-chat", "is_thread": False},
    {"session_id": "robot-chat"},
    {"is_thread": True},
])
def test_dedicated_chat_ack_must_confirm_the_selected_thread(ack):
    tracker = ReplyTracker("robot-chat", owns_chat=True)
    with pytest.raises(ValueError, match="dedicated side chat"):
        tracker.acknowledge({"message_id": "user-1", **ack})


def test_owned_chat_backend_stages_are_spoken_once_and_answer_stops_them(monkeypatch):
    from musegadget.reachy_progress import ProgressPlan
    clock = 0.0
    monkeypatch.setattr(reachy_voice.time, "monotonic", lambda: clock)
    session = FakeSession()
    conversation = VoiceConversation(session, FakeHardware(),
                                     backends=backends_for(Mode.MUSE_VOICE, session),
                                     session_id="robot-chat", owns_chat=True)
    tracker = ReplyTracker("robot-chat", style=ReplyStyle.EXPRESSIVE_JSON, owns_chat=True)
    tracker.acknowledge({"message_id": "user-1", "session_id": "robot-chat", "is_thread": True})
    conversation._progress = ProgressPlan("Look up the documentation.", 0)
    clock = 8.153
    conversation._queue_replies(tracker.event(chat_event("agent.status", parent=None,
                                                        activity_text="Searching web")), tracker)
    assert conversation._progress.take(20).text == "Muse's last reported step was searching the web."
    clock = 22.183
    conversation._queue_replies(tracker.event(chat_event("agent.status", parent=None,
                                                        activity_text="Searching sources")), tracker)
    assert conversation._progress.take(40).text == "Muse's last reported step was searching sources."
    assert conversation._progress.take(60) is None
    clock = 61
    conversation._queue_replies(tracker.event(chat_event("agent.status", parent=None,
                                                        activity_text="is responding")), tracker)
    assert conversation._progress.take(80) is None
    assert conversation._replies.empty()
    clock = 81
    conversation._queue_replies(tracker.event(chat_event("delta.text_append", "answer", parent=None,
                                                        text=sentence_frame("Here is the documentation."))), tracker)
    assert conversation._progress.take(100) is None
    assert conversation._replies.get_nowait() == SpeechSegment("answer", 0, "Here is the documentation.", "neutral")


def test_lunch_request_reports_working_phase_before_delayed_answer(monkeypatch):
    from musegadget.reachy_progress import ProgressPlan
    clock = 0.0
    monkeypatch.setattr(reachy_voice.time, "monotonic", lambda: clock)
    session = FakeSession()
    conversation = VoiceConversation(session, FakeHardware(),
                                     backends=backends_for(Mode.MUSE_VOICE, session),
                                     session_id="robot-chat", owns_chat=True)
    tracker = owned_tracker()
    conversation._progress = ProgressPlan("What should I cook for lunch?", 0)
    clock = 2
    conversation._queue_replies(tracker.event(chat_event(
        "agent.status", parent=None, activity_code="working", activity_text="is working")), tracker)
    update = conversation._progress.take(20)
    assert update.source == "backend"
    assert update.text == "Muse's latest status is that it's working on your request."
    clock = 27
    conversation._queue_replies(tracker.event(chat_event(
        "agent.status", parent=None, activity_code="responding", activity_text="is responding")), tracker)
    conversation._queue_replies(tracker.event(chat_event(
        "delta.text_append", "lunch-answer", parent=None,
        text=sentence_frame("Try a quick vegetable omelette.", "happy"))), tracker)
    assert conversation._progress.take(40) is None
    assert conversation._replies.get_nowait() == SpeechSegment(
        "lunch-answer", 0, "Try a quick vegetable omelette.", "happy")


def test_early_public_milestone_survives_the_first_twenty_second_cue(monkeypatch):
    from musegadget.reachy_progress import ProgressPlan
    clock = 0.0
    monkeypatch.setattr(reachy_voice.time, "monotonic", lambda: clock)
    session = FakeSession()
    conversation = VoiceConversation(session, FakeHardware(),
                                     backends=backends_for(Mode.MUSE_VOICE, session),
                                     session_id="robot-chat", owns_chat=True)
    tracker = owned_tracker()
    conversation._progress = ProgressPlan("Find a recipe.", 0)
    clock = 1
    conversation._queue_replies(tracker.event(chat_event(
        "delta.text_append", "recipe-progress", parent=None,
        text=progress_frame("I found three recipes that match your ingredients."))), tracker)
    assert conversation._progress.take(20).text == (
        "Earlier from Muse: I found three recipes that match your ingredients.")
    assert conversation._replies.empty()
    assert not tracker.complete(20, False)


def test_progress_only_final_does_not_mark_reply_played_or_complete():
    tracker = ReplyTracker("robot-chat", style=ReplyStyle.EXPRESSIVE_JSON)
    tracker.acknowledge({"message_id": "user-1"})
    tracker.event(chat_event("delta.text_append", "reply-1", text=progress_frame()))
    assert tracker.event(chat_event("delta.message_done", "reply-1", content=progress_frame())) == []
    tracker.event(chat_event("task.status", status="completed"))
    assert not tracker.complete(time.monotonic() + 4, False)
    assert tracker.finished_without_text(time.monotonic() + 4)


@pytest.mark.parametrize("ending", ["answer", "cancel", "timeout", "terminal"])
def test_progress_playback_is_preempted_and_closed_before_answer_or_turn_cleanup(
        monkeypatch, streaming_speech, ending):
    np = pytest.importorskip("numpy")
    from musegadget.reachy_progress import ProgressPlan
    monkeypatch.setattr(reachy_local_backends, "ProgressPlan",
                        lambda text, started: ProgressPlan(text, started, first_delay_s=0))

    async def scenario():
        hardware = FakeHardware()
        entered, closed = asyncio.Event(), asyncio.Event()

        class ProgressSpeech:
            async def stream(self, text, rate):
                entered.set()
                try:
                    yield np.full(8000, .1, dtype=np.float32)
                    await asyncio.Event().wait()
                finally:
                    closed.set()

        class AnswerSpeech(streaming_speech):
            async def stream(self, text, rate):
                assert closed.is_set(), "progress must close before an answer starts"
                async for chunk in super().stream(text, rate):
                    yield chunk

        session = FakeSession()
        speech = AnswerSpeech(hardware)
        conversation = VoiceConversation(session, hardware,
                                         backends=backends_for(Mode.ON_ROBOT, session, speech=speech,
                                                               progress_speech=ProgressSpeech(),
                                                               stream_replies=True),
                                         session_id="robot-chat",
                                         reply_timeout_s=.25 if ending == "timeout" else 3)
        subscriber = asyncio.create_task(conversation._subscribe())
        turn = asyncio.create_task(conversation.turn(b"voice"))
        try:
            await entered.wait()
            while not hardware.played:
                await asyncio.sleep(.001)
            assert ("thinking", None) in hardware.state_expressions
            assert conversation.tracker.messages == {}
            if ending == "cancel":
                await cancel_task(turn)
            elif ending == "timeout":
                with pytest.raises(TimeoutError):
                    await turn
            else:
                if ending == "answer":
                    answer = sentence_frame("Found it.", "happy")
                    await session.events.put(chat_event("delta.text_append", "reply-1", text=answer))
                    await session.delivered.get()
                    await session.events.put(chat_event("delta.message_done", "reply-1", content=answer, seq=2))
                    await session.delivered.get()
                await session.events.put(chat_event("task.status", status="completed", seq=3))
                await session.delivered.get()
                assert await turn is TurnOutcome.ACCEPTED
                assert speech.requests == (["Found it."] if ending == "answer" else
                                           ["Muse returned an empty reply. Please try again."])
            assert closed.is_set() and not conversation._muted
            assert conversation.tracker is None and conversation._progress is None
            assert hardware.cleared == (1 if ending in ('cancel', 'timeout') else 2)
            assert not hardware.commands
        finally:
            if not turn.done():
                turn.cancel()
                await asyncio.gather(turn, return_exceptions=True)
            await cancel_task(subscriber)
    asyncio.run(bounded(scenario()))


def test_answer_queued_before_progress_due_suppresses_the_cue(monkeypatch, streaming_speech):
    from musegadget.reachy_progress import ProgressPlan
    monkeypatch.setattr(reachy_local_backends, "ProgressPlan",
                        lambda text, started: ProgressPlan(text, started, first_delay_s=0))

    async def scenario():
        hardware = FakeHardware()
        speech = streaming_speech(hardware)

        class ProgressSpeech:
            async def stream(self, text, rate):
                pytest.fail("queued answer takes priority over a due progress cue")
                yield

        async def send(session):
            answer = sentence_frame("Already done.", "happy")
            await session.events.put(chat_event("delta.message_done", "reply-1", content=answer))
            await session.delivered.get()
            await session.events.put(chat_event("task.status", status="completed", seq=2))
            await session.delivered.get()

        session = FakeSession(send_hook=send)
        conversation = VoiceConversation(session, hardware,
                                         backends=backends_for(Mode.ON_ROBOT, session, speech=speech,
                                                               progress_speech=ProgressSpeech(),
                                                               stream_replies=True),
                                         session_id="robot-chat")
        subscriber = asyncio.create_task(conversation._subscribe())
        try:
            assert await conversation.turn(b"voice") is TurnOutcome.ACCEPTED
            assert speech.requests == ["Already done."]
        finally:
            await cancel_task(subscriber)
    asyncio.run(bounded(scenario()))


def test_progress_clock_starts_after_muse_acknowledges(monkeypatch, streaming_speech):
    from musegadget.reachy_progress import ProgressPlan
    origins = []
    acknowledged = []

    def make_plan(text, started):
        origins.append(started)
        return ProgressPlan(text, started)

    monkeypatch.setattr(reachy_local_backends, "ProgressPlan", make_plan)

    async def scenario():
        hardware = FakeHardware()
        speech = streaming_speech(hardware)

        async def send(session):
            await asyncio.sleep(.05)
            answer = sentence_frame("Done.")
            await session.events.put(chat_event("delta.message_done", "reply-1", content=answer))
            await session.delivered.get()
            await session.events.put(chat_event("task.status", status="completed", seq=2))
            await session.delivered.get()
            acknowledged.append(time.monotonic())

        session = FakeSession(send_hook=send)
        conversation = VoiceConversation(session, hardware,
                                         backends=backends_for(Mode.ON_ROBOT, session, speech=speech,
                                                               progress_speech=speech, stream_replies=True),
                                         session_id="robot-chat")
        subscriber = asyncio.create_task(conversation._subscribe())
        try:
            await conversation.turn(b"voice")
            assert origins[0] >= acknowledged[0]
        finally:
            await cancel_task(subscriber)
    asyncio.run(bounded(scenario()))


def test_completed_progress_hardware_error_survives_answer_race(monkeypatch, streaming_speech):
    from musegadget.reachy_hardware import ReachyHardwareError
    from musegadget.reachy_progress import ProgressPlan
    monkeypatch.setattr(reachy_local_backends, "ProgressPlan",
                        lambda text, started: ProgressPlan(text, started, first_delay_s=0))

    async def scenario():
        hardware = FakeHardware()
        failed = asyncio.Event()

        class ProgressSpeech:
            async def stream(self, text, rate):
                failed.set()
                raise ReachyHardwareError("speaker failed")
                yield

        session = FakeSession()
        speech = streaming_speech(hardware)
        conversation = VoiceConversation(session, hardware,
                                         backends=backends_for(Mode.ON_ROBOT, session, speech=speech,
                                                               progress_speech=ProgressSpeech(),
                                                               stream_replies=True),
                                         session_id="robot-chat")
        subscriber = asyncio.create_task(conversation._subscribe())
        turn = asyncio.create_task(conversation.turn(b"voice"))
        try:
            await failed.wait()
            await session.events.put(chat_event("delta.text_append", "reply-1", text=sentence_frame("Done.")))
            await session.delivered.get()
            with pytest.raises(ReachyHardwareError, match="speaker failed"):
                await turn
            assert speech.requests == []
            assert conversation.tracker is None and conversation._progress is None
            assert not conversation._muted
        finally:
            await cancel_task(subscriber)
    asyncio.run(bounded(scenario()))


@pytest.mark.parametrize("blocked_call", ["play_audio", "set_state"])
@pytest.mark.parametrize("fail", [False, True])
def test_preemption_finishes_pending_hardware_call_before_flush_and_answer(
        monkeypatch, streaming_speech, blocked_call, fail):
    np = pytest.importorskip("numpy")
    from musegadget.reachy_hardware import ReachyHardwareError
    from musegadget.reachy_progress import ProgressPlan
    monkeypatch.setattr(reachy_local_backends, "ProgressPlan",
                        lambda text, started: ProgressPlan(text, started, first_delay_s=0))

    async def scenario():
        entered, release = threading.Event(), threading.Event()
        calls = []

        class Hardware(FakeHardware):
            def play_audio(self, samples):
                if blocked_call == "play_audio" and not entered.is_set():
                    entered.set()
                    assert release.wait(2)
                    calls.append("progress push finished")
                    if fail:
                        raise ReachyHardwareError("delayed speaker failure")
                super().play_audio(samples)

            def set_state(self, state, **kwargs):
                if blocked_call == "set_state" and state == "thinking" and "expression" in kwargs:
                    entered.set()
                    assert release.wait(2)
                    calls.append("progress pose finished")
                    if fail:
                        raise ReachyHardwareError("delayed pose failure")
                super().set_state(state, **kwargs)

            def clear_audio(self):
                calls.append("flush")
                super().clear_audio()

        class ProgressSpeech:
            async def stream(self, text, rate):
                yield np.full(8000, .1, dtype=np.float32)
                await asyncio.Event().wait()

        hardware = Hardware()
        speech = streaming_speech(hardware)
        session = FakeSession()
        conversation = VoiceConversation(session, hardware,
                                         backends=backends_for(Mode.ON_ROBOT, session, speech=speech,
                                                               progress_speech=ProgressSpeech(),
                                                               stream_replies=True),
                                         session_id="robot-chat")
        subscriber = asyncio.create_task(conversation._subscribe())
        turn = asyncio.create_task(conversation.turn(b"voice"))
        try:
            while not entered.is_set():
                await asyncio.sleep(.001)
            answer = sentence_frame("Done.")
            await session.events.put(chat_event("delta.message_done", "reply-1", content=answer))
            await session.delivered.get()
            await asyncio.sleep(.02)
            assert speech.requests == [] and "flush" not in calls
            release.set()
            await session.events.put(chat_event("task.status", status="completed", seq=2))
            await session.delivered.get()
            if fail:
                with pytest.raises(ReachyHardwareError, match="delayed"):
                    await turn
            else:
                await turn
            assert calls[0].startswith("progress") and calls[1] == "flush"
            assert speech.requests == ([] if fail else ["Done."])
        finally:
            release.set()
            if not turn.done():
                turn.cancel()
                await asyncio.gather(turn, return_exceptions=True)
            await cancel_task(subscriber)
    asyncio.run(bounded(scenario()))


def test_repeated_cancellation_still_joins_pending_progress_push(streaming_speech):
    async def scenario():
        entered, release = threading.Event(), threading.Event()

        class Hardware(FakeHardware):
            def play_audio(self, samples):
                entered.set()
                assert release.wait(2)
                super().play_audio(samples)

        hardware = Hardware()
        speech = streaming_speech(hardware)
        session = FakeSession()
        conversation = VoiceConversation(session, hardware,
                                         backends=backends_for(Mode.ON_ROBOT, session, speech=speech))
        task = asyncio.create_task(conversation._speak(None, text="Waiting.", state="thinking",
                                                       stream_segment=True, progress=True))
        try:
            while not entered.is_set():
                await asyncio.sleep(.001)
            task.cancel()
            await asyncio.sleep(.01)
            task.cancel()
            await asyncio.sleep(.01)
            assert not task.done() and not hardware.played
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert len(hardware.played) == 1 and speech.closed == ["Waiting."]
        finally:
            release.set()
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
    asyncio.run(bounded(scenario()))


@pytest.mark.parametrize('progress', [False, True])
def test_queued_notice_waits_for_paused_direct_acknowledgement_or_progress(progress):
    np = pytest.importorskip('numpy')

    async def scenario():
        direct_audio = np.full(640, .1, dtype=np.float32)
        notice_audio = np.full(640, .2, dtype=np.float32)
        hardware = FakeHardware()
        session = FakeSession()
        conversation = VoiceConversation(session, hardware,
                                         backends=backends_for(Mode.ON_ROBOT, session, speech=object()))
        conversation.backends.voice.phrases.update({'Public cue.': (direct_audio,),
                                                    'Public retry.': (notice_audio,)})
        conversation._playback.set_user_speaking(True)
        speaker = asyncio.create_task(conversation._play_output())
        direct = asyncio.create_task(conversation._speak(None, text='Public cue.',
            state='thinking', progress=progress, stream_segment=True))
        try:
            while not conversation._speaker_lock.locked():
                await asyncio.sleep(0)
            conversation._output_queue.put_nowait(reachy_voice._SpeechJob(None,
                'Public retry.', state='listening'))
            while not conversation._output_busy:
                await asyncio.sleep(0)
            assert hardware.played == []
            conversation._playback.set_user_speaking(False)
            await direct
            await conversation._output_queue.join()
            np.testing.assert_array_equal(np.concatenate(hardware.played),
                                          np.concatenate((direct_audio, notice_audio)))
            assert hardware.cleared == 0
        finally:
            conversation._playback.set_user_speaking(False)
            if not direct.done():
                direct.cancel()
            await asyncio.gather(direct, return_exceptions=True)
            await cancel_task(speaker)

    asyncio.run(bounded(scenario()))


@pytest.mark.parametrize('failed_turn', [False, True])
def test_waiting_or_failed_later_turn_cannot_flush_an_earlier_answer(failed_turn):
    np = pytest.importorskip('numpy')

    async def scenario():
        first_chunk, release = asyncio.Event(), asyncio.Event()
        waveform = np.full(640, .1, dtype=np.float32)

        class Speech:
            async def stream(self, text, rate):
                yield waveform
                first_chunk.set()
                await release.wait()
                yield waveform

        hardware = FakeHardware()
        session = FakeSession(acknowledgement={'ok': False, 'status': 403})
        conversation = VoiceConversation(session, hardware,
                                         backends=backends_for(Mode.ON_ROBOT, session, speech=Speech()))
        owner = asyncio.create_task(conversation._speak(None, text='Earlier answer.',
                                                       stream_segment=True))
        try:
            await first_chunk.wait()
            if failed_turn:
                with pytest.raises(ConnectionError, match='rejected'):
                    await conversation.turn(b'public-waveform', defer_playback=True)
            else:
                waiter = asyncio.create_task(conversation._speak(None, text='Public progress.',
                                                                 progress=True))
                await asyncio.sleep(0)
                await cancel_task(waiter)
            assert hardware.cleared == 0
            assert conversation._muted
            release.set()
            await owner
            np.testing.assert_array_equal(np.concatenate(hardware.played),
                                          np.concatenate((waveform, waveform)))
        finally:
            release.set()
            if not owner.done():
                owner.cancel()
            await asyncio.gather(owner, return_exceptions=True)

    asyncio.run(bounded(scenario()))


def test_idle_cleanup_cannot_join_a_speaker_lock_handoff():
    np = pytest.importorskip('numpy')

    async def scenario():
        hardware = FakeHardware()
        session = FakeSession()
        conversation = VoiceConversation(session, hardware,
                                         backends=backends_for(Mode.ON_ROBOT, session, speech=object()))
        conversation.backends.voice.phrases['Public answer.'] = (np.full(640, .1, dtype=np.float32),)
        await conversation._speaker_lock.acquire()
        waiting = asyncio.create_task(conversation._speak(None, text='Public answer.',
            stream_segment=True))
        await asyncio.sleep(0)
        conversation._speaker_lock.release()
        assert not conversation._speaker_lock.locked()
        # The awakened speaker has not run yet. Cleanup must neither queue
        # behind it nor flush the audio it is about to own.
        await conversation._clear_idle_audio()
        assert not waiting.done() and hardware.cleared == 0
        await waiting
        assert hardware.cleared == 0
        np.testing.assert_array_equal(np.concatenate(hardware.played),
                                      np.full(640, .1, dtype=np.float32))

    asyncio.run(bounded(scenario()))


def test_failed_deferred_turn_does_not_flush_the_output_pipeline():
    async def scenario():
        hardware = FakeHardware()
        session = FakeSession(acknowledgement={'ok': False, 'status': 403})
        conversation = VoiceConversation(session,
                                         hardware, backends=backends_for(Mode.MUSE_VOICE, session))
        with pytest.raises(ConnectionError, match='rejected'):
            await conversation.turn(b'public-waveform', defer_playback=True)
        assert hardware.cleared == 0 and not conversation._muted

    asyncio.run(bounded(scenario()))


def test_standard_progress_audio_is_cached_without_playback_or_answer_synthesis(streaming_speech):
    from musegadget.reachy_progress import PUBLIC_PROGRESS_PHRASES

    async def scenario():
        hardware = FakeHardware()
        answer = streaming_speech(hardware)
        progress = streaming_speech(hardware)
        # The fake voice records poses; preparation itself must not move the hardware.
        hardware.state_expressions.append(("idle", None))
        session = FakeSession()
        conversation = VoiceConversation(session, hardware,
                                         backends=backends_for(Mode.ON_ROBOT, session, speech=answer,
                                                               progress_speech=progress))
        await conversation.backends.progress_voice.warm(hardware.output_sample_rate)
        assert progress.requests == list(PUBLIC_PROGRESS_PHRASES)
        assert progress.closed == progress.requests and answer.requests == []
        assert set(conversation.backends.progress_voice.phrases) == set(PUBLIC_PROGRESS_PHRASES)
        assert not hardware.states and not hardware.played
        for phrase in PUBLIC_PROGRESS_PHRASES:
            await conversation._speak(None, text=phrase, state="thinking", stream_segment=True,
                                      voice=conversation.backends.progress_voice, progress=True)
        assert progress.requests == list(PUBLIC_PROGRESS_PHRASES)
        assert len(hardware.played) == len(PUBLIC_PROGRESS_PHRASES)
        assert hardware.progress_cues == len(PUBLIC_PROGRESS_PHRASES)
        assert not hardware.commands
    asyncio.run(bounded(scenario()))


@pytest.fixture
def streaming_speech(monkeypatch):
    np = pytest.importorskip("numpy")
    monkeypatch.setattr(reachy_voice, "REPLY_QUIET_S", 0)
    monkeypatch.setattr(reachy_voice, "EMPTY_REPLY_GRACE_S", 0)
    monkeypatch.setattr(reachy_voice, "ECHO_TAIL_S", 0)

    class Speech:
        def __init__(self, hardware):
            self.hardware = hardware
            self.requests = []
            self.poses = []
            self.closed = []
            self.active = False

        async def stream(self, text, rate):
            assert not self.active, "speech iterators must never overlap"
            self.active = True
            self.requests.append(text)
            self.poses.append(self.hardware.state_expressions[-1] if self.hardware.state_expressions else None)
            try:
                yield np.full(80, .1, dtype=np.float32)
            finally:
                self.active = False
                self.closed.append(text)
    return Speech


async def streaming_turn(script, speech, hardware, *, before_ack=False):
    producers = []
    conversation = None

    async def deliver(session):
        sequence = 100
        for item in script:
            if callable(item):
                await item(conversation)
                continue
            sequence += 1
            await session.events.put({**item, "seq": sequence})
            await session.delivered.get()
        await session.events.put(chat_event("task.status", seq=sequence + 1, status="completed"))
        await session.delivered.get()

    async def send(session):
        if before_ack:
            await deliver(session)
        else:
            producers.append(asyncio.create_task(deliver(session)))

    session = FakeSession(send_hook=send)
    conversation = VoiceConversation(session, hardware,
                                     backends=backends_for(Mode.ON_ROBOT, session, speech=speech, stream_replies=True),
                                     session_id="robot-chat", reply_timeout_s=3)
    subscriber = asyncio.create_task(conversation._subscribe())
    try:
        await session.chat_subscribed.wait()
        await conversation.turn(b"voice")
        await asyncio.gather(*producers)
        assert not conversation._muted and conversation.tracker is None
        assert hardware.cleared == 1
    finally:
        for producer in producers:
            if not producer.done():
                producer.cancel()
        await asyncio.gather(*producers, return_exceptions=True)
        await cancel_task(subscriber)
    return conversation


def test_protocol_segments_are_immutable_indexed_and_final_prefix_does_not_repeat():
    tracker = ReplyTracker("robot-chat", style=ReplyStyle.EXPRESSIVE_JSON)
    tracker.acknowledge({"message_id": "user-1"})
    first = sentence_frame("Hello!", "happy")
    assert tracker.event(chat_event("delta.text_append", "reply-1", text=first)) == [
        SpeechSegment("reply-1", 0, "Hello!", "happy")]
    final = first + sentence_frame("Yes.", "nod")
    assert tracker.event(chat_event("delta.message_done", "reply-1", content=final)) == [
        SpeechSegment("reply-1", 1, "Yes.", "nod")]
    assert tracker.event(chat_event("delta.message_done", "reply-1", content=final)) == []


def test_parentless_streaming_preserves_session_and_explicit_parent_filters():
    tracker = ReplyTracker("robot-chat", style=ReplyStyle.EXPRESSIVE_JSON)
    tracker.acknowledge({"message_id": "user-1"})
    text = sentence_frame("Hello.")
    assert tracker.event(chat_event("delta.text_append", "wrong-session", parent=None,
                                    session_id="phone-chat", text=text)) == []
    assert tracker.event(chat_event("delta.text_append", "wrong-parent", parent="another-user", text=text)) == []
    assert tracker.event(chat_event("delta.text_append", "reply-1", parent=None, text=text)) == [
        SpeechSegment("reply-1", 0, "Hello.", "neutral")]
    assert tracker.event(chat_event("delta.text_append", "reply-1", parent="another-user", text=text)) == []


def test_streaming_holds_later_messages_until_earlier_message_finishes():
    tracker = ReplyTracker("robot-chat", style=ReplyStyle.EXPRESSIVE_JSON)
    tracker.acknowledge({"message_id": "user-1"})
    first = sentence_frame("First.")
    second = sentence_frame("Second.", "curious")
    assert tracker.event(chat_event("delta.text_append", "first", text=first))[0].text == "First."
    assert tracker.event(chat_event("delta.message_done", "second", parent="first", content=second)) == []
    result = tracker.event(chat_event("delta.message_done", "first", content=first))
    assert result == [SpeechSegment("second", 0, "Second.", "curious")]


@pytest.mark.parametrize("text", ["é" * 33000, sentence_frame("Hi.") * 33])
def test_streaming_text_and_segment_batches_are_bounded(text):
    tracker = ReplyTracker("robot-chat", style=ReplyStyle.EXPRESSIVE_JSON)
    tracker.acknowledge({"message_id": "user-1"})
    with pytest.raises(ValueError, match="64 KiB|too many queued"):
        tracker.event(chat_event("delta.text_append", "reply-1", text=text))


@pytest.mark.parametrize("gesture", ["nod", "shake"])
def test_streaming_first_frame_plays_before_done_and_expressions_match_playback(streaming_speech, gesture):
    async def scenario():
        hardware = FakeHardware()
        speech = streaming_speech(hardware)
        first = sentence_frame("Wow!", "surprised")
        final = first + sentence_frame("Yes.", gesture)

        async def heard_before_done(conversation):
            while not hardware.played:
                await asyncio.sleep(.001)
            assert not conversation.tracker.messages["reply-1"]["done"]
            assert speech.requests == ["Wow!"]

        await streaming_turn([
            chat_event("delta.text_append", "reply-1", parent=None, text=first),
            heard_before_done,
            chat_event("delta.text_append", "reply-1", parent=None, text=sentence_frame("Yes.", gesture)),
            chat_event("delta.message_done", "reply-1", parent=None, content=final),
            chat_event("delta.message_done", "reply-1", parent=None, content=final),
        ], speech, hardware)
        assert speech.requests == ["Wow!", "Yes."]
        assert hardware.play_expressions == [("speaking", "surprised"), ("speaking", gesture)]
        assert hardware.commands == []
        assert speech.closed == speech.requests
        assert [state for state in hardware.state_expressions if state[0] == "thinking"] == [
            ("thinking", None), ("thinking", "nod")]
    asyncio.run(bounded(scenario()))


def test_legacy_streaming_withholds_split_marker_and_only_speaks_final_tail(streaming_speech):
    async def scenario():
        hardware = FakeHardware()
        speech = streaming_speech(hardware)

        async def heard(conversation):
            while not hardware.played:
                await asyncio.sleep(.001)
            assert speech.requests == ["Hello."]

        await streaming_turn([
            chat_event("delta.text_append", "reply-1", text="Hello. Next sentence. ["),
            heard,
            chat_event("delta.text_append", "reply-1", text="reachy:no"),
            chat_event("delta.text_append", "reply-1", text="d]"),
            chat_event("delta.message_done", "reply-1", content="Hello. Next sentence. [reachy:nod]"),
        ], speech, hardware)
        assert speech.requests == ["Hello.", "Next sentence."]
        # The old text format reveals this silent gesture after both sentences.
        assert hardware.commands == [("reachy.expression", {"name": "nod"})]
    asyncio.run(bounded(scenario()))


def test_unheard_delta_revision_before_ack_is_replaced_by_authoritative_answer(streaming_speech):
    async def scenario():
        hardware = FakeHardware()
        speech = streaming_speech(hardware)
        await streaming_turn([
            chat_event("delta.text_append", "reply-1", text=sentence_frame("Draft.")),
            chat_event("delta.message_done", "reply-1", content=sentence_frame("Final.", "happy")),
        ], speech, hardware, before_ack=True)
        assert speech.requests == ["Final."]
    asyncio.run(bounded(scenario()))


def test_spoken_prefix_revision_drops_already_queued_tail(streaming_speech, caplog):
    async def scenario():
        hardware = FakeHardware()
        speech = streaming_speech(hardware)
        draft = sentence_frame("Original.") + sentence_frame("Unspoken draft.")

        async def heard(conversation):
            while not hardware.played:
                await asyncio.sleep(.001)
            assert speech.requests == ["Original."]

        await streaming_turn([
            chat_event("delta.text_append", "reply-1", text=draft),
            heard,
            chat_event("delta.message_done", "reply-1", content=sentence_frame("Changed private answer.")),
        ], speech, hardware)
        assert speech.requests == ["Original."]
    asyncio.run(bounded(scenario()))
    assert "revised a spoken reply" in caplog.text
    assert "Changed private answer" not in caplog.text


def test_streaming_multiple_messages_keep_audio_order(streaming_speech):
    async def scenario():
        hardware = FakeHardware()
        speech = streaming_speech(hardware)
        first = sentence_frame("First.")
        await streaming_turn([
            chat_event("delta.text_append", "first", text=first),
            chat_event("delta.message_done", "second", parent="first", content=sentence_frame("Second.")),
            chat_event("delta.message_done", "first", content=first + sentence_frame("First tail.")),
        ], speech, hardware)
        assert speech.requests == ["First.", "First tail.", "Second."]
    asyncio.run(bounded(scenario()))


@pytest.mark.parametrize("content,expected", [("", ["Muse returned an empty reply. Please try again."]),
                                              ("Transcript only.", ["Transcript only."])])
def test_streaming_blank_and_transcript_only_completion_recover(streaming_speech, content, expected):
    async def scenario():
        hardware = FakeHardware()
        speech = streaming_speech(hardware)
        await streaming_turn([chat_event("delta.message_done", "reply-1", transcript=content)], speech, hardware)
        assert speech.requests == expected
    asyncio.run(bounded(scenario()))


def test_cancelling_streaming_closes_active_speech_before_capture_resumes(streaming_speech):
    np = pytest.importorskip("numpy")
    async def scenario():
        started, closed = asyncio.Event(), asyncio.Event()

        class StalledSpeech:
            async def stream(self, text, rate):
                try:
                    yield np.full(80, .1, dtype=np.float32)
                    started.set()
                    await asyncio.Event().wait()
                finally:
                    closed.set()

        async def send(session):
            await session.events.put(chat_event("delta.text_append", "reply-1", text=sentence_frame("Hello.")))
            await session.delivered.get()

        hardware = FakeHardware()
        muse = FakeSession(send_hook=send)
        conversation = VoiceConversation(muse, hardware,
                                         backends=backends_for(Mode.ON_ROBOT, muse, speech=StalledSpeech(),
                                                               stream_replies=True))
        subscriber = asyncio.create_task(conversation._subscribe())
        await conversation.session.chat_subscribed.wait()
        turn = asyncio.create_task(conversation.turn(b"voice"))
        await started.wait()
        await cancel_task(turn)
        assert closed.is_set() and hardware.cleared == 1
        assert not conversation._muted and conversation.tracker is None
        await cancel_task(subscriber)
    asyncio.run(bounded(scenario()))


@pytest.mark.parametrize("mode,expected", [("both", "Both antennas are enabled"),
                                         ("left", "Only the left antenna is enabled"),
                                         ("right", "Only the right antenna is enabled"),
                                         ("none", "Antenna movement is disabled")])
def test_muse_robot_awareness_reflects_antenna_settings_without_capture_mechanics(mode, expected):
    hardware = FakeHardware()
    hardware.antenna_mode = mode
    hardware.motion_enabled = True
    wake = type("Wake", (), {"phrase": "hey muse"})()
    session = FakeSession()
    conversation = VoiceConversation(session, hardware,
                                     backends=backends_for(Mode.ON_ROBOT, session, speech=object(), transcriber=object()),
                                     wake_detector=wake, wake_timeout_s=10)
    context = conversation._voice_context()
    assert expected in context
    assert "hey muse" not in context
    assert "idle seconds" not in context
    assert "silence" not in context
    assert "echo-cancelled" not in context
    assert "six degrees of freedom" in context and "does not send camera images" in context


def test_muse_robot_awareness_reports_disabled_movement():
    hardware = FakeHardware()
    hardware.motion_enabled = False
    session = FakeSession()
    context = VoiceConversation(session, hardware, backends=backends_for(Mode.MUSE_VOICE, session))._voice_context()
    assert "Movement is disabled in this session" in context
    assert "Both antennas are enabled" not in context


def test_muse_knows_local_face_follow_does_not_supply_camera_vision_or_identity():
    hardware = FakeHardware()
    hardware.face_tracking_enabled = True
    session = FakeSession()
    context = VoiceConversation(session, hardware, backends=backends_for(Mode.MUSE_VOICE, session))._voice_context()
    assert "Local face tracking is enabled" in context
    assert "camera images stay on Reachy" in context
    assert "no visual information or identity recognition" in context


@pytest.fixture
def prepared_speech():
    from musegadget.local_speech import PreparedSpeech
    np = pytest.importorskip("numpy")

    class Speech:
        def __init__(self, hardware, *, stall_tail=False):
            self.requests = []
            self.closed = []
            self.prepared = []
            self.tail_started = asyncio.Event()
            self.lock = asyncio.Lock()
            self.stall_tail = stall_tail

        def prepare(self, text, rate):
            audio = PreparedSpeech(self.stream(text, rate))
            self.prepared.append(audio)
            return audio

        async def stream(self, text, rate):
            async with self.lock:
                self.requests.append(text)
                try:
                    if text == "Second.":
                        self.tail_started.set()
                        if self.stall_tail:
                            await asyncio.Event().wait()
                    yield np.full(rate // 5, .2 if text == "Second." else .1, dtype=np.float32)
                finally:
                    self.closed.append(text)
    return Speech


def test_sentence_lookahead_synthesizes_during_playback_and_preserves_expressions(prepared_speech):
    async def scenario():
        hardware = FakeHardware()
        speech = prepared_speech(hardware)
        final = sentence_frame("First.", "happy") + sentence_frame("Second.", "curious")

        async def heard(conversation):
            while not hardware.played:
                await asyncio.sleep(.001)
            await asyncio.wait_for(speech.tail_started.wait(), .1)
            assert hardware.played and all(float(chunk[0]) == pytest.approx(.1) for chunk in hardware.played)
            assert hardware.state_expressions[-1] == ("speaking", "happy")
            assert len(conversation._prepared_speech) == 2

        conversation = await streaming_turn([
            chat_event("delta.text_append", "reply-1", text=final), heard,
            chat_event("delta.message_done", "reply-1", content=final),
        ], speech, hardware)
        assert speech.requests == ["First.", "Second."]
        np = pytest.importorskip("numpy")
        np.testing.assert_array_equal(np.concatenate(hardware.played),
            np.concatenate([np.full(hardware.output_sample_rate // 5, value, dtype=np.float32)
                            for value in (.1, .2)]))
        assert hardware.state_expressions[-2:] == [("speaking", "happy"), ("speaking", "curious")]
        assert not conversation._prepared_speech
        assert all(audio._task.done() for audio in speech.prepared)
    asyncio.run(bounded(scenario()))


def test_revised_unspoken_lookahead_is_cancelled_and_never_played(prepared_speech):
    async def scenario():
        hardware = FakeHardware()
        speech = prepared_speech(hardware, stall_tail=True)
        draft = sentence_frame("First.") + sentence_frame("Second.", "curious")

        async def heard(conversation):
            while not hardware.played:
                await asyncio.sleep(.001)
            await speech.tail_started.wait()

        conversation = await streaming_turn([
            chat_event("delta.text_append", "reply-1", text=draft), heard,
            chat_event("delta.message_done", "reply-1", content=sentence_frame("Revised.")),
        ], speech, hardware)
        assert sum(len(chunk) for chunk in hardware.played) == hardware.output_sample_rate // 5
        assert all(float(chunk[0]) == pytest.approx(.1) for chunk in hardware.played)
        assert ("speaking", "curious") not in hardware.state_expressions
        assert speech.closed == ["First.", "Second."]
        assert not conversation._prepared_speech
        assert all(audio._task.done() for audio in speech.prepared)
    asyncio.run(bounded(scenario()))


def test_cancelled_turn_reaps_current_and_lookahead_before_microphone_reopens(prepared_speech):
    async def scenario():
        hardware = FakeHardware()
        speech = prepared_speech(hardware, stall_tail=True)
        frames = sentence_frame("First.") + sentence_frame("Second.")

        async def send(session):
            await session.events.put(chat_event("delta.text_append", "reply-1", text=frames))
            await session.delivered.get()

        muse = FakeSession(send_hook=send)
        conversation = VoiceConversation(muse, hardware,
                                         backends=backends_for(Mode.ON_ROBOT, muse, speech=speech, stream_replies=True))
        subscriber = asyncio.create_task(conversation._subscribe())
        await conversation.session.chat_subscribed.wait()
        turn = asyncio.create_task(conversation.turn(b"voice"))
        await speech.tail_started.wait()
        await cancel_task(turn)
        assert not conversation._muted and not conversation._prepared_speech
        assert hardware.cleared == 1 and speech.closed == ["First.", "Second."]
        assert all(audio._task.done() for audio in speech.prepared)
        await cancel_task(subscriber)
    asyncio.run(bounded(scenario()))


def test_authoritative_revision_during_progress_shutdown_speaks_only_replacement(
        monkeypatch, prepared_speech):
    np = pytest.importorskip("numpy")
    from musegadget.reachy_progress import ProgressPlan
    monkeypatch.setattr(reachy_local_backends, "ProgressPlan",
                        lambda text, started: ProgressPlan(text, started, first_delay_s=0))

    async def scenario():
        closing, release = asyncio.Event(), asyncio.Event()

        class ProgressSpeech:
            async def stream(self, text, rate):
                try:
                    yield np.full(rate // 5, .4, dtype=np.float32)
                    await asyncio.Event().wait()
                finally:
                    closing.set()
                    await release.wait()

        hardware = FakeHardware()
        speech = prepared_speech(hardware)
        session = FakeSession()
        conversation = VoiceConversation(session, hardware,
                                         backends=backends_for(Mode.ON_ROBOT, session, speech=speech,
                                                               progress_speech=ProgressSpeech(),
                                                               stream_replies=True),
                                         reply_timeout_s=3)
        subscriber = asyncio.create_task(conversation._subscribe())
        await session.chat_subscribed.wait()
        turn = asyncio.create_task(conversation.turn(b"voice"))
        try:
            while not hardware.played:
                await asyncio.sleep(.001)
            draft = sentence_frame("First.", "happy")
            await session.events.put(chat_event("delta.text_append", "reply-1", text=draft, seq=10))
            await session.delivered.get()
            await closing.wait()
            await session.events.put(chat_event("delta.message_done", "reply-1",
                                                content=sentence_frame("Second.", "curious"), seq=11))
            await session.delivered.get()
            await session.events.put(chat_event("task.status", status="completed", seq=12))
            await session.delivered.get()
            release.set()
            assert await turn is TurnOutcome.ACCEPTED
            from itertools import groupby
            assert [marker for marker, _ in groupby(round(float(chunk[0]), 1)
                    for chunk in hardware.played)] == [.4, .2]
            assert sum(len(chunk) for chunk in hardware.played
                       if float(chunk[0]) == pytest.approx(.2)) == hardware.output_sample_rate // 5
            assert hardware.play_expressions[-1] == ("speaking", "curious")
            assert ("speaking", "happy") not in hardware.play_expressions
            assert not conversation._prepared_speech and not conversation._closing_speech
            assert all(audio._task.done() for audio in speech.prepared)
        finally:
            release.set()
            if not turn.done():
                await cancel_task(turn)
            await cancel_task(subscriber)
    asyncio.run(bounded(scenario()))


def chat_event(name, message_id=None, *, seq=1, parent="user-1", session_id="robot-chat", **payload):
    data = {"session_id": session_id, **payload}
    if message_id is not None:
        data["message_id"] = message_id
    if parent is not None:
        data["reply_to_message_id"] = parent
    return {"type": "event", "seq": seq, "event": name, "payload": data}


def test_events_before_ack_bind_only_after_the_user_message_is_known():
    tracker = ReplyTracker("robot-chat")
    assert tracker.event(chat_event("delta.message_done", "other", parent="other-user", content="other")) == []
    assert tracker.event(chat_event("delta.message_start", "reply-1", seq=2)) == []
    assert tracker.event(chat_event("delta.text_append", "reply-1", seq=3, text="Hello")) == []
    assert tracker.event(chat_event("delta.message_done", "reply-1", seq=4)) == []
    assert tracker.acknowledge({"result": {"message_id": "user-1"}}) == ["reply-1"]
    assert set(tracker.messages) == {"reply-1"}
    assert tracker.pending == []


def test_multi_message_replies_queue_once_and_can_parent_the_previous_reply():
    tracker = ReplyTracker("robot-chat")
    tracker.acknowledge({"message_id": "user-1"})
    assert tracker.event(chat_event("message.assistant", "reply-1", content="First.")) == ["reply-1"]
    assert tracker.event(chat_event("delta.message_done", "reply-1", seq=2, content="First.")) == []
    assert tracker.event(chat_event("delta.message_done", "reply-2", seq=3,
                                    parent="reply-1", display_text="Second.")) == ["reply-2"]
    assert len(tracker.messages) == 2


def test_assistant_display_text_waits_for_readiness_before_tts():
    tracker = ReplyTracker("robot-chat")
    tracker.acknowledge({"reply_to_message_id": "user-1"})
    assert tracker.event(chat_event("message.assistant", "reply-1", content="Draft", display_text_ready=False)) == []
    assert tracker.event(chat_event("message.assistant", "reply-1", seq=2,
                                    display_text="Finished", display_text_ready=True)) == ["reply-1"]


def test_unrelated_chat_user_messages_and_unknown_events_do_not_bind():
    tracker = ReplyTracker("robot-chat")
    tracker.acknowledge({"message_id": "user-1"})
    events = [
        chat_event("message.assistant", "other-session", session_id="phone-chat", content="Other"),
        chat_event("message.assistant", "other-parent", parent="unrelated-user", content="Other"),
        chat_event("message.assistant", "user-1", role="user", content="My question"),
        chat_event("message.tool", "tool", content="Tool output"),
        {"type": "ack", "payload": {}},
        {"type": "event", "event": "delta.message_done", "payload": None},
    ]
    assert [tracker.event(event) for event in events] == [[]] * len(events)
    assert tracker.messages == {}


def test_turn_completion_waits_for_speech_message_completion_and_agent_idle():
    tracker = ReplyTracker("robot-chat")
    tracker.acknowledge({"message_id": "user-1"})
    tracker.event(chat_event("delta.message_start", "reply-1"))
    now = time.monotonic() + 10
    assert not tracker.complete(now, played=True)
    tracker.event(chat_event("delta.message_done", "reply-1", content="Answer"))
    assert not tracker.complete(now, played=False)
    tracker.event(chat_event("agent.status", activity_code="researching"))
    assert not tracker.complete(now, played=True)
    tracker.event(chat_event("agent.status", activity_code="idle"))
    assert tracker.complete(now, played=True)
    assert not tracker.complete(time.monotonic(), played=True)


def test_errored_task_is_terminal_and_does_not_hold_a_spoken_reply_open():
    tracker = ReplyTracker()
    tracker.acknowledge({"message_id": "user-1"})
    tracker.event({"type": "event", "seq": 17979, "event": "task.status",
                   "payload": {"task_id": "task-1", "status": "running"}})
    assert tracker.busy
    assert tracker.event({
        "type": "event", "seq": 17984, "event": "delta.message_done",
        "payload": {"message_id": "reply-1", "reply_to_message_id": "user-1",
                    "status": "completed", "content": "Test response"},
    }) == ["reply-1"]
    tracker.event({"type": "event", "seq": 17985, "event": "task.status",
                   "payload": {"task_id": "task-1", "status": "errored"}})
    assert not tracker.busy
    assert tracker.complete(time.monotonic() + 10, played=True)


def test_terminal_task_returns_to_listening_without_an_extra_quiet_period():
    tracker = ReplyTracker()
    tracker.acknowledge({"message_id": "user-1"})
    tracker.event(chat_event("delta.message_done", "reply-1", content="Hello"))
    tracker.event(chat_event("task.status", status="completed"))
    assert tracker.complete(time.monotonic(), played=True)


def test_terminal_status_with_activity_does_not_keep_the_turn_busy():
    tracker = ReplyTracker()
    tracker.acknowledge({"message_id": "user-1"})
    tracker.event(chat_event("delta.message_done", "reply-1", content="Hello"))
    tracker.event(chat_event("task.status", status="completed", activity_code="thinking"))
    assert tracker.complete(time.monotonic(), played=True)


@pytest.mark.parametrize("order", [("blank", "terminal"), ("terminal", "blank"), ("terminal",)])
def test_empty_finished_reply_waits_for_reordered_text_but_not_the_full_timeout(order):
    tracker = ReplyTracker()
    tracker.acknowledge({"message_id": "user-1"})
    for item in order:
        event = (chat_event("task.status", status="completed") if item == "terminal" else
                 chat_event("delta.message_done", "reply-1", content="  "))
        assert tracker.event(event) == ([TaskFinished()] if item == "terminal" else [])
    grace = 1 if "blank" in order else 3
    assert not tracker.finished_without_text(tracker.last_activity + grace - 0.01)
    assert tracker.finished_without_text(tracker.last_activity + grace + 0.01)
    assert tracker.event(chat_event("delta.message_done", "reply-1", content="Late answer")) == ["reply-1"]
    assert not tracker.finished_without_text(tracker.last_activity + 10)


def test_nonterminal_empty_reply_does_not_end_the_turn():
    tracker = ReplyTracker()
    tracker.acknowledge({"message_id": "user-1"})
    tracker.event(chat_event("delta.message_done", "reply-1", content=""))
    assert not tracker.finished_without_text(tracker.last_activity + 200)


def test_transcript_only_completed_message_queues_tts_without_text_deltas():
    tracker = ReplyTracker()
    tracker.acknowledge({"result": {"message_id": "user-1"}})
    assert tracker.event({
        "type": "event", "seq": 17984, "event": "delta.message_done",
        "payload": {"message_id": "reply-1", "reply_to_message_id": "user-1",
                    "status": "completed", "transcript": "Test response"},
    }) == ["reply-1"]
    assert tracker.complete(time.monotonic() + 10, played=True)


def test_local_voice_speaks_muse_text_and_executes_its_expression(monkeypatch):
    np = pytest.importorskip("numpy")
    monkeypatch.setattr(reachy_voice, "REPLY_QUIET_S", 0)
    monkeypatch.setattr(reachy_voice, "ECHO_TAIL_S", 0)

    class LocalSpeech:
        def __init__(self):
            self.requests = []

        async def stream(self, text, output_rate):
            self.requests.append((text, output_rate))
            yield np.full(80, 0.1, dtype=np.float32)

    async def scenario():
        async def reply(session):
            await session.events.put(chat_event("delta.message_done", "reply-1", transcript={
                "messages": [{"id": "reply-1", "role": "assistant", "content": [
                    {"type": "text", "text": "Yes, I can nod. [reachy:nod]"},
                ]}],
            }))
            await session.delivered.get()

        speech = LocalSpeech()
        session = FakeSession(send_hook=reply)
        hardware = FakeHardware()
        conversation = VoiceConversation(session, hardware,
                                         backends=backends_for(Mode.ON_ROBOT, session, speech=speech), reply_timeout_s=2)
        subscriber = asyncio.create_task(conversation._subscribe())
        await session.chat_subscribed.wait()
        await conversation.turn(b"recording")
        assert "message" not in session.sent_options[0]
        assert "output_modality" not in session.sent_options[0]
        assert speech.requests == [("Yes, I can nod.", 8000)]
        assert session.tts_requests == []
        assert sum(len(chunk) for chunk in hardware.played) == 80
        assert hardware.commands == []
        assert hardware.play_expressions == [("speaking", "nod")]
        assert not conversation._muted
        await cancel_task(subscriber)

    asyncio.run(bounded(scenario()))


def test_expression_setup_uses_a_separate_text_turn_and_speaks_readiness(monkeypatch):
    np = pytest.importorskip("numpy")
    monkeypatch.setattr(reachy_voice, "REPLY_QUIET_S", 0)
    monkeypatch.setattr(reachy_voice, "ECHO_TAIL_S", 0)
    spoken = []

    class LocalSpeech:
        async def stream(self, text, output_rate):
            spoken.append(text)
            yield np.full(80, 0.1, dtype=np.float32)

    async def scenario():
        async def reply(session):
            await session.events.put(chat_event("delta.message_done", "reply-1",
                                                content="Ready to talk. [reachy:nod]"))
            await session.delivered.get()

        session = FakeSession(send_hook=reply)
        hardware = FakeHardware()
        conversation = VoiceConversation(session, hardware,
                                         backends=backends_for(Mode.ON_ROBOT, session, speech=LocalSpeech()))
        subscriber = asyncio.create_task(conversation._subscribe())
        await session.chat_subscribed.wait()
        await conversation.turn(None)
        assert session.sent == []
        assert "subsequent spoken messages" in session.setup_messages[0][0]
        assert spoken == ["Ready to talk."]
        assert hardware.commands == []
        assert hardware.play_expressions == [("speaking", "nod")]
        await cancel_task(subscriber)

    asyncio.run(bounded(scenario()))


@pytest.mark.parametrize("recognized", ["Please say pineapple sunshine and nod.", ""])
def test_local_recognition_sends_words_to_muse_and_ignores_empty_speech(monkeypatch, recognized):
    np = pytest.importorskip("numpy")
    monkeypatch.setattr(reachy_voice, "REPLY_QUIET_S", 0)
    monkeypatch.setattr(reachy_voice, "ECHO_TAIL_S", 0)
    recordings = []

    class LocalRecognition:
        async def transcribe(self, wav):
            recordings.append(wav)
            return recognized

    class LocalSpeech:
        async def stream(self, text, rate):
            assert text == "Pineapple sunshine!"
            yield np.full(80, 0.1, dtype=np.float32)

    async def scenario():
        async def reply(session):
            await session.events.put(chat_event("delta.message_done", "reply-1",
                                                content="Pineapple sunshine! [reachy:nod]"))
            await session.delivered.get()

        session = FakeSession(send_hook=reply)
        hardware = FakeHardware()
        conversation = VoiceConversation(session, hardware,
                                         backends=backends_for(Mode.ON_ROBOT, session, speech=LocalSpeech(),
                                                               transcriber=LocalRecognition()))
        subscriber = asyncio.create_task(conversation._subscribe())
        await session.chat_subscribed.wait()
        await conversation.turn(b"captured microphone audio")
        assert recordings == [b"captured microphone audio"]
        assert session.sent == []
        assert session.setup_messages == ([(recognized, None)] if recognized else [])
        assert hardware.commands == []
        assert hardware.play_expressions == ([("speaking", "nod")] if recognized else [])
        assert not conversation._muted
        assert conversation.tracker is None
        await cancel_task(subscriber)

    asyncio.run(bounded(scenario()))


@pytest.mark.parametrize("speech_fails", [False, True])
def test_empty_muse_reply_announces_the_problem_and_resumes_without_reconnecting(monkeypatch, speech_fails):
    np = pytest.importorskip("numpy")
    monkeypatch.setattr(reachy_voice, "EMPTY_REPLY_GRACE_S", 0)
    monkeypatch.setattr(reachy_voice, "ECHO_TAIL_S", 0)
    spoken = []

    class LocalSpeech:
        async def stream(self, text, rate):
            spoken.append(text)
            if speech_fails:
                raise RuntimeError("synthesis failed")
            yield np.full(80, 0.1, dtype=np.float32)

    async def scenario():
        async def reply(session):
            for event in [chat_event("delta.message_done", "reply-1", content=""),
                          chat_event("task.status", seq=2, status="completed")]:
                await session.events.put(event)
                await session.delivered.get()

        session = FakeSession(send_hook=reply)
        hardware = FakeHardware()
        conversation = VoiceConversation(session, hardware,
                                         backends=backends_for(Mode.ON_ROBOT, session, speech=LocalSpeech()))
        subscriber = asyncio.create_task(conversation._subscribe())
        await session.chat_subscribed.wait()
        await conversation.turn(b"voice")
        assert spoken == ["Muse returned an empty reply. Please try again."]
        assert not conversation._muted
        assert conversation.tracker is None
        assert not session.subscription_closed
        assert hardware.cleared == 1
        await cancel_task(subscriber)

    asyncio.run(bounded(scenario()))


def test_reply_emotion_is_present_during_speech_without_a_delayed_extra_move(monkeypatch):
    np = pytest.importorskip("numpy")
    monkeypatch.setattr(reachy_voice, "ECHO_TAIL_S", 0)

    class LocalSpeech:
        async def stream(self, text, rate):
            assert text == "That's wonderful."
            yield np.full(80, 0.1, dtype=np.float32)

    async def scenario():
        hardware = FakeHardware()
        session = FakeSession()
        conversation = VoiceConversation(session, hardware,
                                         backends=backends_for(Mode.ON_ROBOT, session, speech=LocalSpeech()))
        conversation.tracker = ReplyTracker()
        conversation.tracker.messages["reply-1"] = {"text": "That's wonderful. [reachy:happy]"}
        await conversation._speak("reply-1")
        assert ("speaking", "happy") in hardware.state_expressions
        assert hardware.commands == []
        assert hardware.played

    asyncio.run(bounded(scenario()))


@pytest.mark.parametrize("recognized, expected_ack", [
    ("What's the weather tomorrow?", "Let me check the weather."),
    ("Hello!", None), ("Say hello.", None), ("", None),
])
def test_contextual_acknowledgement_waits_for_words_and_skips_unmatched_input(monkeypatch, recognized, expected_ack):
    np = pytest.importorskip("numpy")
    monkeypatch.setattr(reachy_voice, "ECHO_TAIL_S", 0)
    monkeypatch.setattr(reachy_voice, "REPLY_QUIET_S", 0)
    synthesized = []
    hardware = FakeHardware()

    class LocalRecognition:
        async def transcribe(self, wav, *, on_speech=None):
            assert on_speech is None
            assert not hardware.played
            await asyncio.sleep(.01)
            assert not hardware.played
            return recognized

    class LocalSpeech:
        async def stream(self, text, rate):
            synthesized.append(text)
            yield np.full(80, .1, dtype=np.float32)

    async def scenario():
        async def reply(session):
            if expected_ack:
                while not hardware.played:
                    await asyncio.sleep(.001)
            await session.events.put(chat_event("delta.message_done", "reply-1", content="Hello. [reachy:happy]"))
            await session.delivered.get()

        session = FakeSession(send_hook=reply)
        conversation = VoiceConversation(session, hardware,
                                         backends=backends_for(Mode.ON_ROBOT, session, speech=LocalSpeech(),
                                                               transcriber=LocalRecognition()))
        conversation.backends.voice.phrases = {phrase: (np.full(80, .1, dtype=np.float32),)
                                               for phrase in reachy_local_backends.ACKNOWLEDGEMENTS}
        subscriber = asyncio.create_task(conversation._subscribe())
        await session.chat_subscribed.wait()
        await conversation.turn(b"voice")
        assert synthesized == (["Hello."] if recognized else [])
        assert len(hardware.played) == (int(bool(expected_ack)) + 1 if recognized else 0)
        assert not conversation._muted
        await cancel_task(subscriber)

    asyncio.run(bounded(scenario()))


def test_cancelling_request_stops_contextual_acknowledgement_before_listening_resumes(monkeypatch):
    np = pytest.importorskip("numpy")
    monkeypatch.setattr(reachy_voice, "ECHO_TAIL_S", 0)

    class LocalRecognition:
        async def transcribe(self, wav):
            return "What's the weather?"

    async def scenario():
        async def waiting_request(session):
            await asyncio.Event().wait()

        session = FakeSession(send_hook=waiting_request)
        hardware = FakeHardware()
        conversation = VoiceConversation(session, hardware,
                                         backends=backends_for(Mode.ON_ROBOT, session, speech=object(),
                                                               transcriber=LocalRecognition()))
        conversation.backends.voice.phrases = {"Let me check the weather.":
                                               (np.full(8000, .1, dtype=np.float32),) * 3}
        turn = asyncio.create_task(conversation.turn(b"voice"))
        while not hardware.played:
            await asyncio.sleep(.001)
        await cancel_task(turn)
        assert 0 < sum(len(chunk) for chunk in hardware.played) <= hardware.output_sample_rate * .08
        assert hardware.cleared == 1
        assert conversation.tracker is None
        assert not conversation._muted

    asyncio.run(bounded(scenario()))


@pytest.mark.parametrize("request_fails", [False, True])
def test_completed_acknowledgement_hardware_failure_propagates_and_restores_capture(monkeypatch, request_fails):
    np = pytest.importorskip("numpy")
    from musegadget.reachy_hardware import ReachyHardwareError
    monkeypatch.setattr(reachy_voice, "ECHO_TAIL_S", 0)

    class BrokenSpeaker(FakeHardware):
        def play_audio(self, samples):
            raise ReachyHardwareError("speaker disconnected")

    class LocalRecognition:
        async def transcribe(self, wav):
            return "What's the weather?"

    async def scenario():
        acknowledgement_finished = asyncio.Event()
        async def request(session):
            await acknowledgement_finished.wait()
            if request_fails:
                raise ConnectionError("request failed")

        hardware = BrokenSpeaker()
        session = FakeSession(send_hook=request)
        conversation = VoiceConversation(session, hardware,
                                         backends=backends_for(Mode.ON_ROBOT, session, speech=object(),
                                                               transcriber=LocalRecognition()))
        conversation.backends.voice.phrases = {"Let me check the weather.": (np.full(80, .1, dtype=np.float32),)}
        speak = conversation._speak
        async def observed_speech(*args, **kwargs):
            try:
                await speak(*args, **kwargs)
            finally:
                acknowledgement_finished.set()
        conversation._speak = observed_speech
        with pytest.raises(ReachyHardwareError, match="speaker disconnected"):
            await conversation.turn(b"voice")
        assert hardware.cleared == 1
        assert conversation.tracker is None
        assert not conversation._muted

    asyncio.run(bounded(scenario()))


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_turn_cancellation_preserves_cancellation_after_acknowledgement_hardware_failure(monkeypatch, cleanup_fails):
    np = pytest.importorskip("numpy")
    from musegadget.reachy_hardware import ReachyHardwareError
    monkeypatch.setattr(reachy_voice, "ECHO_TAIL_S", 0)

    class BrokenSpeaker(FakeHardware):
        def play_audio(self, samples):
            raise ReachyHardwareError("speaker disconnected")
        def clear_audio(self):
            super().clear_audio()
            if cleanup_fails:
                raise ReachyHardwareError("speaker unavailable while flushing")

    class LocalRecognition:
        async def transcribe(self, wav):
            return "What's the weather?"

    async def scenario():
        acknowledgement_finished = asyncio.Event()
        request_waiting = asyncio.Event()
        async def request(session):
            await acknowledgement_finished.wait()
            request_waiting.set()
            await asyncio.Event().wait()

        hardware = BrokenSpeaker()
        session = FakeSession(send_hook=request)
        conversation = VoiceConversation(session, hardware,
                                         backends=backends_for(Mode.ON_ROBOT, session, speech=object(),
                                                               transcriber=LocalRecognition()))
        conversation.backends.voice.phrases = {"Let me check the weather.": (np.full(80, .1, dtype=np.float32),)}
        speak = conversation._speak
        async def observed_speech(*args, **kwargs):
            try:
                await speak(*args, **kwargs)
            finally:
                acknowledgement_finished.set()
        conversation._speak = observed_speech
        turn = asyncio.create_task(conversation.turn(b"voice"))
        await request_waiting.wait()
        await cancel_task(turn)
        assert hardware.cleared == 1
        assert conversation.tracker is None
        assert not conversation._muted

    asyncio.run(bounded(scenario()))


@pytest.mark.parametrize("text, expected", [
    ("What's the weather today?", "Let me check the weather."),
    ("Find the latest news.", "Let me look that up."),
    ("Explain how a radio works.", "Let me explain that."),
    ("Help me plan my week.", "Let me work out a plan."),
    ("Please calculate 14 * 8.", "Let me work that out."),
    ("Could you plan a trip to Kyoto?", "Let me look into that trip."),
    ("Hello, Muse!", None), ("Weather is nice.", None),
    ("Say pineapple sunshine.", None), ("I am sad.", None),
])
def test_question_acknowledgement_matches_intent_without_a_generic_fallback(text, expected):
    assert reachy_local_backends.contextual_acknowledgement(text) == expected


@pytest.mark.parametrize("text, expected", [
    ("Hey, Muse! What is the weather?", "What is the weather?"),
    ("hey muse", ""), ("Hey museum opens today", "Hey museum opens today"),
    ("Please say Hey Muse.", "Please say Hey Muse."),
])
def test_only_exact_leading_wake_phrase_is_stripped_from_followups(text, expected):
    assert reachy_voice.strip_wake_prefix(text, "hey muse") == expected


def test_first_wake_question_discards_words_before_exact_invocation():
    assert reachy_voice.question_after_wake("Background private words. Hey, Muse! What time is it?", "hey muse") == "What time is it?"
    assert reachy_voice.question_after_wake("Background private words only.", "hey muse") is None


def test_wake_and_contextual_cues_are_cached_without_overlapping_synthesis(monkeypatch):
    np = pytest.importorskip("numpy")
    synthesized = []
    class Speech:
        async def stream(self, text, rate):
            synthesized.append(text)
            yield np.full(80, .1, dtype=np.float32)
    class Wake:
        phrase = "hey muse"
    async def scenario():
        session = FakeSession()
        conversation = VoiceConversation(session, FakeHardware(),
                                         backends=backends_for(Mode.ON_ROBOT, session, speech=Speech(),
                                                               transcriber=object(), wake=True),
                                         wake_detector=Wake())
        await conversation.warm_up()
        assert synthesized == [*reachy_local_backends.ACKNOWLEDGEMENTS, "Yes?"]
        assert set(conversation.backends.voice.phrases) == set(synthesized)
        assert not conversation.hardware.played
    asyncio.run(bounded(scenario()))


@pytest.mark.parametrize("recognized", ["Hey, Muse!", "Private background words without a recognized invocation"])
def test_wake_only_or_unconfirmed_boundary_is_local_then_clean_followup_gets_context_once(monkeypatch, recognized):
    np = pytest.importorskip("numpy")
    monkeypatch.setattr(reachy_voice, "ECHO_TAIL_S", 0)
    monkeypatch.setattr(reachy_voice, "REPLY_QUIET_S", 0)
    class Wake:
        phrase = "hey muse"
    class Recognition:
        async def transcribe(self, wav):
            return recognized if wav == b"wake" else "Say hello."
    class Speech:
        async def stream(self, text, rate):
            yield np.full(80, .1, dtype=np.float32)
    async def scenario():
        async def reply(session):
            count = len(session.setup_messages)
            await session.events.put(chat_event("delta.message_done", f"reply-{count}", seq=count,
                                                 content=sentence_frame("Hello!", "happy")))
            await session.delivered.get()
        session = FakeSession(send_hook=reply)
        hardware = FakeHardware()
        conversation = VoiceConversation(session, hardware,
                                         backends=backends_for(Mode.ON_ROBOT, session, speech=Speech(),
                                                               transcriber=Recognition(),
                                                               stream_replies=True),
                                         wake_detector=Wake())
        conversation.backends.voice.phrases["Yes?"] = (np.full(80, .1, dtype=np.float32),)
        conversation._wake_strip_required = True
        await conversation.turn(None)
        assert session.setup_messages == [] and not hardware.states
        outcome = await conversation.turn(b"wake")
        assert outcome is TurnOutcome.WAKE_CUE
        assert session.setup_messages == [] and len(hardware.played) == 1
        assert hardware.states[-1] == "listening"
        subscriber = asyncio.create_task(conversation._subscribe())
        await session.chat_subscribed.wait()
        await conversation.turn(b"question")
        assert len(session.setup_messages) == 1
        assert session.setup_messages[0][0].endswith("The user's spoken request is: Say hello.")
        assert "sentence" in session.setup_messages[0][0].casefold()
        await conversation.turn(b"followup")
        assert session.setup_messages[1][0] == "Say hello."
        assert not conversation._muted
        await cancel_task(subscriber)
    asyncio.run(bounded(scenario()))


def test_same_breath_wake_question_strips_pre_wake_words_and_sends_one_request(monkeypatch):
    np = pytest.importorskip("numpy")
    monkeypatch.setattr(reachy_voice, "ECHO_TAIL_S", 0)
    monkeypatch.setattr(reachy_voice, "REPLY_QUIET_S", 0)
    class Wake:
        phrase = "hey muse"
    class Recognition:
        def __init__(self, hardware):
            self.hardware = hardware
        async def transcribe(self, wav):
            assert self.hardware.state_expressions[-1] == ("thinking", None)
            return "Private background chatter. Hey Muse, say pineapple sunshine."
    class Speech:
        async def stream(self, text, rate):
            yield np.full(80, .1, dtype=np.float32)
    async def scenario():
        async def reply(session):
            await session.events.put(chat_event("delta.message_done", "reply-1", content="Pineapple sunshine."))
            await session.delivered.get()
        session = FakeSession(send_hook=reply)
        hardware = FakeHardware()
        conversation = VoiceConversation(session, hardware,
                                         backends=backends_for(Mode.ON_ROBOT, session, speech=Speech(),
                                                               transcriber=Recognition(hardware)),
                                         wake_detector=Wake())
        conversation._wake_strip_required = True
        subscriber = asyncio.create_task(conversation._subscribe())
        await session.chat_subscribed.wait()
        await conversation.turn(b"question")
        assert len(session.setup_messages) == 1
        request = session.setup_messages[0][0]
        assert request.endswith("The user's spoken request is: say pineapple sunshine.")
        assert "Private background chatter" not in request
        assert len(hardware.played) == 1
        await cancel_task(subscriber)
    asyncio.run(bounded(scenario()))


def test_wake_microphone_gates_asr_preserves_preroll_and_closes_only_after_turn_finishes(monkeypatch):
    np = pytest.importorskip("numpy")
    from musegadget import voice_audio
    feeds = []
    preroll_complete = threading.Event()
    class Recorder:
        def finish_initial_capture(self):
            pass
        active = False
        last_truncated = False
        def __init__(self, *args, **kwargs):
            pass
        def reset(self):
            self.active = False
        def feed(self, sample):
            if not len(sample):
                return None
            marker = int(sample[0])
            feeds.append(marker)
            self.active = marker in (2, 4)
            if marker == 2:
                preroll_complete.set()
            return {1: b"old background", 3: b"question", 5: b"followup"}.get(marker)
    class Wake:
        phrase = "hey muse"
        def __init__(self):
            self.feeds = []
            self.resets = 0
        def feed(self, sample):
            self.feeds.append(int(sample[0]))
            return int(sample[0]) == 2
        def reset(self):
            self.resets += 1
    monkeypatch.setattr(voice_audio, "TurnRecorder", Recorder)
    async def scenario():
        hardware = FakeHardware()
        session = FakeSession()
        session.chat_subscribed.set()
        wake = Wake()
        conversation = VoiceConversation(session, hardware,
                                         backends=backends_for(Mode.ON_ROBOT, session, speech=object(),
                                                               transcriber=object()),
                                         wake_detector=wake, wake_timeout_s=.06)
        turns = []
        release = asyncio.Event()
        async def turn(wav, **options):
            turns.append(wav)
            conversation._muted = True
            try:
                await release.wait()
            finally:
                conversation._muted = False
            return TurnOutcome.ACCEPTED
        conversation.turn = turn
        microphone = asyncio.create_task(conversation._microphone())
        try:
            while not hardware.states:
                await asyncio.sleep(.001)
            assert wake.resets == 1
            hardware.samples.put(np.full(160, 1, dtype=np.float32))
            while hardware.read_count < 1:
                await asyncio.sleep(.001)
            await asyncio.sleep(.01)
            assert feeds == [] and turns == [] and hardware.states == ["idle"]
            hardware.samples.put(np.full(160, 2, dtype=np.float32))
            assert await asyncio.to_thread(preroll_complete.wait, 1)
            assert feeds == [1, 2] and turns == []
            assert hardware.state_expressions[-1] == ("listening", "happy")
            assert not hardware.commands
            hardware.samples.put(np.full(160, 3, dtype=np.float32))
            while not turns:
                await asyncio.sleep(.001)
            assert turns == [b"question"]
            await asyncio.sleep(.09)
            assert hardware.states[-1] == "thinking" and conversation._wake_deadline is not None
            release.set()
            while hardware.states.count("listening") < 2:
                await asyncio.sleep(.001)
            quiet_deadline = conversation._wake_deadline
            hardware.samples.put(np.full(160, 4, dtype=np.float32))
            while 4 not in feeds:
                await asyncio.sleep(.001)
            await asyncio.sleep(.09)
            assert hardware.states[-1] == "listening"
            assert conversation._wake_deadline == quiet_deadline
            hardware.samples.put(np.full(160, 5, dtype=np.float32))
            while len(turns) < 2:
                await asyncio.sleep(.001)
            assert turns == [b"question", b"followup"]
            assert wake.feeds == [1, 2]
            while hardware.states[-1] != "idle":
                await asyncio.sleep(.005)
            assert conversation._wake_deadline is None
            hardware.samples.put(np.full(160, 1, dtype=np.float32))
            while len(wake.feeds) < 3:
                await asyncio.sleep(.001)
            assert turns == [b"question", b"followup"] and feeds == [1, 2, 3, 4, 5]
        finally:
            await cancel_task(microphone)
        assert not conversation._muted
    asyncio.run(bounded(scenario()))


@pytest.mark.parametrize("pending_same_chunk", [False, True])
def test_delayed_wake_endpoint_and_pending_same_chunk_keep_the_question(monkeypatch, pending_same_chunk):
    np = pytest.importorskip("numpy")
    from musegadget import voice_audio
    class Recorder:
        def finish_initial_capture(self):
            pass
        active = False
        last_truncated = False
        def __init__(self, *args, **kwargs):
            self.pending = False
        def reset(self):
            self.active = False
        def feed(self, sample):
            if not len(sample):
                if self.pending:
                    self.pending = False
                    return b"wake question"
                return None
            if int(sample[0]) == 1:
                return b"background"
            if int(sample[0]) == 2:
                self.pending = pending_same_chunk
                return b"background" if pending_same_chunk else b"wake question"
            return None
    class Wake:
        phrase = "hey muse"
        def reset(self):
            pass
        def feed(self, sample):
            return int(sample[0]) == 3
    monkeypatch.setattr(voice_audio, "TurnRecorder", Recorder)
    async def scenario():
        session = FakeSession()
        session.chat_subscribed.set()
        hardware = FakeHardware()
        conversation = VoiceConversation(session, hardware,
                                         backends=backends_for(Mode.ON_ROBOT, session, speech=object(),
                                                               transcriber=object()),
                                         wake_detector=Wake())
        turns = []
        async def turn(wav, **options):
            turns.append(wav)
            conversation._muted = False
            return TurnOutcome.ACCEPTED
        conversation.turn = turn
        microphone = asyncio.create_task(conversation._microphone())
        try:
            while not hardware.states:
                await asyncio.sleep(.001)
            for marker in (1, 2, 3):
                hardware.samples.put(np.full(160, marker, dtype=np.float32))
            while not turns:
                await asyncio.sleep(.001)
            assert turns == [b"wake question"]
        finally:
            await cancel_task(microphone)
    asyncio.run(bounded(scenario()))


def test_sleeping_detector_receives_stateful_16k_resampling_and_preroll_is_bounded(monkeypatch):
    np = pytest.importorskip("numpy")
    pytest.importorskip("av")
    from musegadget import voice_audio
    captured = []
    captured_complete = threading.Event()
    class Recorder:
        def finish_initial_capture(self):
            pass
        active = True
        last_truncated = False
        def __init__(self, *args, **kwargs):
            pass
        def reset(self):
            pass
        def feed(self, sample):
            captured.append(sample.copy())
            if sum(len(chunk) for chunk in captured) == 3 * 48000:
                captured_complete.set()
            return None
    class Wake:
        phrase = "hey muse"
        def __init__(self):
            self.received = []
        def reset(self):
            pass
        def feed(self, sample):
            self.received.append(sample.copy())
            return sum(len(chunk) for chunk in self.received) >= 4 * 16000 - 32
    monkeypatch.setattr(voice_audio, "TurnRecorder", Recorder)
    async def scenario():
        session = FakeSession()
        session.chat_subscribed.set()
        hardware = FakeHardware()
        hardware.sample_rate = 48000
        wake = Wake()
        conversation = VoiceConversation(session, hardware,
                                         backends=backends_for(Mode.ON_ROBOT, session, speech=object(),
                                                               transcriber=object()),
                                         wake_detector=wake)
        microphone = asyncio.create_task(conversation._microphone())
        try:
            while not hardware.states:
                await asyncio.sleep(.001)
            for _ in range(40):
                hardware.samples.put(np.full(4800, .25, dtype=np.float32))
            assert await asyncio.to_thread(captured_complete.wait, 1)
            assert sum(len(chunk) for chunk in captured) == 3 * 48000
            output = np.concatenate(wake.received)
            assert 4 * 16000 - 32 <= len(output) <= 4 * 16000
            assert output.dtype == np.float32 and output.ndim == 1
            assert np.max(np.abs(output - .25)) < .0001
        finally:
            await cancel_task(microphone)
    asyncio.run(bounded(scenario()))


def test_continuous_capture_preserves_realtime_frames_during_slow_keyword_bursts(monkeypatch):
    np = pytest.importorskip("numpy")
    from musegadget import voice_audio
    class Recorder:
        def finish_initial_capture(self):
            pass
        active = False
        last_truncated = False
        def __init__(self, *args, **kwargs):
            self.frames = []
        def reset(self):
            self.frames = []
            self.active = False
        def feed(self, sample):
            if not len(sample):
                return None
            markers = [int(round(float(value) * 100)) for value in sample[::256]]
            self.frames.extend(markers)
            self.active = True
            if 63 in markers:
                self.active = False
                return bytes(self.frames)
            return None
    class Wake:
        phrase = "hey muse"
        def __init__(self):
            self.frames = []
        def reset(self):
            pass
        def feed(self, sample):
            for value in sample[::256]:
                marker = int(round(float(value) * 100))
                self.frames.append(marker)
                if marker % 10 == 0:
                    time.sleep(.08)
                if marker == 30:
                    return True
            return False
    monkeypatch.setattr(voice_audio, "TurnRecorder", Recorder)
    async def scenario():
        session = FakeSession()
        session.chat_subscribed.set()
        hardware = FakeHardware()
        hardware.samples = queue.Queue(maxsize=2)
        wake = Wake()
        conversation = VoiceConversation(session, hardware,
                                         backends=backends_for(Mode.ON_ROBOT, session, speech=object(),
                                                               transcriber=object()),
                                         wake_detector=wake)
        turns = []
        async def turn(wav, **options):
            turns.append(wav)
            conversation._muted = False
            return TurnOutcome.ACCEPTED
        conversation.turn = turn
        dropped = []
        async def supply():
            for marker in range(64):
                await asyncio.sleep(.016)
                sample = np.full(256, marker / 100, dtype=np.float32)
                if hardware.samples.full():
                    dropped.append(hardware.samples.get_nowait())
                hardware.samples.put_nowait(sample)
        microphone = asyncio.create_task(conversation._microphone())
        supplier = None
        try:
            while not hardware.states:
                await asyncio.sleep(.001)
            supplier = asyncio.create_task(supply())
            while not turns:
                await asyncio.sleep(.002)
            await supplier
            assert dropped == []
            assert wake.frames == list(range(31))
            assert turns == [bytes(range(64))]
            assert hardware.state_expressions[1] == ("listening", "happy")
            assert not hardware.commands
        finally:
            if supplier is not None and not supplier.done():
                await cancel_task(supplier)
            await cancel_task(microphone)
        assert not [task for task in asyncio.all_tasks() if task.get_name() == "reachy-microphone-capture"]
    asyncio.run(bounded(scenario()))


def test_capture_fifo_bounds_audio_duration_and_marks_every_loss_boundary():
    np = pytest.importorskip("numpy")
    async def scenario():
        buffer = reachy_voice._CaptureBuffer(10)
        buffer.push(np.array([1] * 4, dtype=np.float32))
        buffer.push(np.array([2] * 4, dtype=np.float32))
        buffer.push(np.array([3] * 8, dtype=np.float32))
        chunk = await buffer.get()
        np.testing.assert_array_equal(chunk.samples, [3] * 8)
        assert chunk.gap and buffer.samples == 0
        assert chunk.gap_reason == "capture buffer overflow"
        buffer.push(np.array([4] * 2, dtype=np.float32), discard=True)
        chunk = await buffer.get()
        assert chunk.discard and not chunk.gap
        assert chunk.gap_reason is None
        buffer.push(np.arange(14, dtype=np.float32))
        chunk = await buffer.get()
        np.testing.assert_array_equal(chunk.samples, list(range(4, 14)))
        assert chunk.gap and len(chunk.samples) == 10
        assert chunk.gap_reason == "oversized capture chunk"
        assert chunk.samples.base is None
        buffer.push(np.array([5] * 4, dtype=np.float32))
        buffer.push(np.array([6] * 4, dtype=np.float32), gap=True, gap_reason="SDK read stall")
        chunk = await buffer.get()
        np.testing.assert_array_equal(chunk.samples, [6] * 4)
        assert chunk.gap and buffer.samples == 0
        assert chunk.gap_reason == "SDK read stall"
    asyncio.run(bounded(scenario()))


def test_capture_overflow_resets_keyword_and_recording_before_retained_audio(monkeypatch, caplog):
    np = pytest.importorskip("numpy")
    from musegadget import voice_audio
    class Recorder:
        def finish_initial_capture(self):
            pass
        active = False
        last_truncated = False
        def __init__(self, *args, **kwargs):
            pass
        def reset(self):
            pass
        def feed(self, sample):
            raise AssertionError("sleeping capture must not invoke recording")
    release = threading.Event()
    class Wake:
        phrase = "hey muse"
        def __init__(self, loop):
            self.loop = loop
            self.entered = asyncio.Event()
            self.completed = asyncio.Event()
            self.reset_count = 0
            self.observed = []
        def reset(self):
            self.reset_count += 1
        def feed(self, sample):
            self.observed.append((self.reset_count, int(sample[0])))
            if len(self.observed) == 1:
                self.loop.call_soon_threadsafe(self.entered.set)
                release.wait()
            if len(self.observed) == 2:
                self.loop.call_soon_threadsafe(self.completed.set)
            return False
    monkeypatch.setattr(voice_audio, "TurnRecorder", Recorder)
    async def scenario():
        loop = asyncio.get_running_loop()
        ready = asyncio.Event()
        overflow_committed = asyncio.Event()
        original_push = reachy_voice._CaptureBuffer.push
        def push(buffer, sample, **options):
            original_push(buffer, sample, **options)
            if int(sample[0]) == 8:
                overflow_committed.set()
        monkeypatch.setattr(reachy_voice._CaptureBuffer, "push", push)
        class Hardware(FakeHardware):
            def set_state(self, state, **options):
                super().set_state(state, **options)
                loop.call_soon_threadsafe(ready.set)
        session = FakeSession()
        session.chat_subscribed.set()
        hardware = Hardware()
        wake = Wake(loop)
        conversation = VoiceConversation(session, hardware,
                                         backends=backends_for(Mode.ON_ROBOT, session, speech=object(),
                                                               transcriber=object()),
                                         wake_detector=wake)
        microphone = asyncio.create_task(conversation._microphone())
        try:
            await ready.wait()
            hardware.samples.put(np.full(8000, 1, dtype=np.float32))
            await wake.entered.wait()
            for marker in range(2, 9):
                hardware.samples.put(np.full(8000, marker, dtype=np.float32))
            await overflow_committed.wait()
            release.set()
            await wake.completed.wait()
            assert wake.observed == [(1, 1), (2, 8)]
            assert "lost continuity" in caplog.text
        finally:
            release.set()
            await cancel_task(microphone)
    asyncio.run(bounded(scenario()))


def test_capture_hardware_failure_propagates_and_cancels_its_producer():
    pytest.importorskip("webrtcvad")
    pytest.importorskip("av")
    from musegadget.reachy_hardware import ReachyHardwareError
    class Hardware(FakeHardware):
        broken = False
        def read_audio(self):
            if self.broken:
                raise ReachyHardwareError("capture disconnected")
            return super().read_audio()
    async def scenario():
        session = FakeSession()
        session.chat_subscribed.set()
        hardware = Hardware()
        conversation = VoiceConversation(session, hardware, backends=backends_for(Mode.MUSE_VOICE, session))
        microphone = asyncio.create_task(conversation._microphone())
        while not hardware.states:
            await asyncio.sleep(.001)
        hardware.broken = True
        with pytest.raises(ReachyHardwareError, match="capture disconnected"):
            await asyncio.wait_for(microphone, 1)
        assert not [task for task in asyncio.all_tasks() if task.get_name() == "reachy-microphone-capture"]
    asyncio.run(bounded(scenario()))


def test_slow_recording_feed_keeps_microphone_producer_draining_and_joins_on_cancel(monkeypatch):
    from musegadget import voice_audio
    np = pytest.importorskip("numpy")
    release, returned = threading.Event(), threading.Event()

    async def scenario():
        loop = asyncio.get_running_loop()
        ready, entered, committed = asyncio.Event(), asyncio.Event(), asyncio.Event()

        class Recorder:
            active = False
            last_truncated = False

            def __init__(self, *_args, **_options):
                pass

            def feed(self, samples):
                if len(samples) and samples[0] == 1:
                    loop.call_soon_threadsafe(entered.set)
                    release.wait(1)
                    returned.set()
                return None

        class Hardware(FakeHardware):
            def set_state(self, state, **options):
                super().set_state(state, **options)
                loop.call_soon_threadsafe(ready.set)

        original_push = reachy_voice._CaptureBuffer.push

        def push(buffer, samples, **options):
            original_push(buffer, samples, **options)
            if samples[0] == 2:
                committed.set()

        monkeypatch.setattr(voice_audio, "TurnRecorder", Recorder)
        monkeypatch.setattr(reachy_voice._CaptureBuffer, "push", push)
        session, hardware = FakeSession(), Hardware()
        session.chat_subscribed.set()
        microphone = asyncio.create_task(VoiceConversation(session, hardware,
                                                           backends=backends_for(Mode.MUSE_VOICE, session))._microphone())
        try:
            await ready.wait()
            hardware.samples.put(np.full(320, 1, np.float32))
            await entered.wait()
            hardware.samples.put(np.full(320, 2, np.float32))
            await committed.wait()
            assert not returned.is_set(), "SDK audio must drain while recording inference is held"
            microphone.cancel()
            await asyncio.sleep(0)
            assert not microphone.done(), "cancellation must join the outstanding recorder worker"
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await microphone
            assert returned.is_set()
        finally:
            release.set()
            if not microphone.done():
                await cancel_task(microphone)

    asyncio.run(bounded(scenario()))


def test_capture_cancellation_waits_for_the_outstanding_sdk_read():
    pytest.importorskip("webrtcvad")
    pytest.importorskip("av")
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    class Hardware(FakeHardware):
        def read_audio(self):
            if self.states:
                entered.set()
                release.wait(2)
                finished.set()
            return None
    async def scenario():
        session = FakeSession()
        session.chat_subscribed.set()
        hardware = Hardware()
        conversation = VoiceConversation(session, hardware, backends=backends_for(Mode.MUSE_VOICE, session))
        microphone = asyncio.create_task(conversation._microphone())
        try:
            while not entered.is_set():
                await asyncio.sleep(.001)
            microphone.cancel()
            await asyncio.sleep(.01)
            assert not microphone.done()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await microphone
            assert finished.is_set()
            assert not [task for task in asyncio.all_tasks() if task.get_name() == "reachy-microphone-capture"]
        finally:
            release.set()
            await cancel_task(microphone)
    asyncio.run(bounded(scenario()))


def test_read_started_during_playback_stays_discarded_after_unmute():
    np = pytest.importorskip("numpy")
    entered = threading.Event()
    release = threading.Event()
    class Hardware(FakeHardware):
        first = True
        def read_audio(self):
            if self.first:
                self.first = False
                entered.set()
                release.wait(2)
                return np.full(256, .1, dtype=np.float32)
            return None
    async def scenario():
        hardware = Hardware()
        session = FakeSession()
        conversation = VoiceConversation(session, hardware, backends=backends_for(Mode.MUSE_VOICE, session))
        conversation._muted = True
        buffer = reachy_voice._CaptureBuffer(16000)
        producer = asyncio.create_task(conversation._capture_microphone(buffer))
        try:
            while not entered.is_set():
                await asyncio.sleep(.001)
            conversation._muted = False
            release.set()
            chunk = None
            while chunk is None:
                chunk = await buffer.get()
            assert chunk.discard
        finally:
            release.set()
            await cancel_task(producer)
    asyncio.run(bounded(scenario()))


def test_non_wake_capture_gap_clears_the_listening_pose(monkeypatch):
    np = pytest.importorskip("numpy")
    from musegadget import voice_audio
    class Recorder:
        def finish_initial_capture(self):
            pass
        active = False
        last_truncated = False
        def __init__(self, *args, **kwargs):
            pass
        def reset(self):
            self.active = False
        def feed(self, sample):
            self.active = bool(sample[0])
            return None
    monkeypatch.setattr(voice_audio, "TurnRecorder", Recorder)
    async def scenario():
        session = FakeSession()
        session.chat_subscribed.set()
        hardware = FakeHardware()
        conversation = VoiceConversation(session, hardware, backends=backends_for(Mode.MUSE_VOICE, session))
        microphone = asyncio.create_task(conversation._microphone())
        try:
            while not hardware.states:
                await asyncio.sleep(.001)
            hardware.samples.put(np.full(256, .1, dtype=np.float32))
            while hardware.states[-1] != "listening":
                await asyncio.sleep(.001)
            hardware.samples.put(np.zeros(32000, dtype=np.float32))
            while hardware.states[-1] != "idle":
                await asyncio.sleep(.001)
            assert hardware.states == ["idle", "listening", "idle"]
        finally:
            await cancel_task(microphone)
    asyncio.run(bounded(scenario()))


def test_blank_first_wake_gives_local_cue_but_blank_followup_is_empty(monkeypatch):
    np = pytest.importorskip("numpy")
    monkeypatch.setattr(reachy_voice, "ECHO_TAIL_S", 0)
    class Wake:
        phrase = "hey muse"
    class Recognition:
        async def transcribe(self, wav):
            return ""
    async def scenario():
        hardware = FakeHardware()
        session = FakeSession()
        conversation = VoiceConversation(session, hardware,
                                         backends=backends_for(Mode.ON_ROBOT, session, speech=object(),
                                                               transcriber=Recognition()),
                                         wake_detector=Wake())
        conversation.backends.voice.phrases["Yes?"] = (np.full(80, .1, dtype=np.float32),)
        conversation._wake_strip_required = True
        assert await conversation.turn(b"first wake") is TurnOutcome.WAKE_CUE
        assert len(hardware.played) == 1 and hardware.states[-1] == "listening"
        assert await conversation.turn(b"empty followup") is TurnOutcome.EMPTY
        assert len(hardware.played) == 1
        assert not session.setup_messages and not session.sent and not conversation._muted
    asyncio.run(bounded(scenario()))


def test_missing_asr_wake_boundary_opens_time_for_question_after_slow_recognition(monkeypatch):
    np = pytest.importorskip("numpy")
    from musegadget import voice_audio
    monkeypatch.setattr(reachy_voice, "ECHO_TAIL_S", 0)
    class Recorder:
        def finish_initial_capture(self):
            pass
        active = False
        last_truncated = False
        def __init__(self, *args, **kwargs):
            pass
        def reset(self):
            self.active = False
        def feed(self, sample):
            if not len(sample):
                return None
            self.active = int(sample[0]) == 2
            return None if self.active else b"wake recording"
    class Wake:
        phrase = "hey muse"
        def reset(self):
            pass
        def feed(self, sample):
            return int(sample[0]) == 2
    class Recognition:
        async def transcribe(self, wav):
            await asyncio.sleep(.06)
            return "An uncertain transcript without the invocation."
    monkeypatch.setattr(voice_audio, "TurnRecorder", Recorder)
    async def scenario():
        session = FakeSession()
        session.chat_subscribed.set()
        hardware = FakeHardware()
        conversation = VoiceConversation(session, hardware,
                                         backends=backends_for(Mode.ON_ROBOT, session, speech=object(),
                                                               transcriber=Recognition()),
                                         wake_detector=Wake(), wake_timeout_s=.03)
        conversation.backends.voice.phrases["Yes?"] = (np.full(80, .1, dtype=np.float32),)
        microphone = asyncio.create_task(conversation._microphone())
        try:
            while not hardware.states:
                await asyncio.sleep(.001)
            hardware.samples.put(np.full(128, 2, dtype=np.float32))
            while hardware.states[-1] != "listening":
                await asyncio.sleep(.001)
            original_deadline = conversation._wake_deadline
            hardware.samples.put(np.ones(128, dtype=np.float32))
            while conversation._wake_deadline <= original_deadline:
                await asyncio.sleep(.001)
            assert time.monotonic() > original_deadline
            assert conversation._wake_deadline is not None
            assert conversation._wake_deadline > time.monotonic()
            assert hardware.states[-1] == "listening" and len(hardware.played) == 1
            assert not session.setup_messages and not session.sent
        finally:
            await cancel_task(microphone)
    asyncio.run(bounded(scenario()))


def test_repeated_empty_vad_turns_cannot_keep_a_wake_session_open(monkeypatch):
    np = pytest.importorskip("numpy")
    from musegadget import voice_audio
    monkeypatch.setattr(reachy_voice, "ECHO_TAIL_S", 0)
    class Recorder:
        def finish_initial_capture(self):
            pass
        active = False
        last_truncated = False
        def __init__(self, *args, **kwargs):
            pass
        def reset(self):
            self.active = False
        def feed(self, sample):
            if not len(sample):
                return None
            self.active = int(sample[0]) == 2
            return b"false VAD turn" if not self.active else None
    class Wake:
        phrase = "hey muse"
        def reset(self):
            pass
        def feed(self, sample):
            return int(sample[0]) == 2
    class Recognition:
        def __init__(self):
            self.calls = 0
        async def transcribe(self, wav):
            self.calls += 1
            await asyncio.sleep(.02)
            return ""
    monkeypatch.setattr(voice_audio, "TurnRecorder", Recorder)
    async def scenario():
        session = FakeSession()
        session.chat_subscribed.set()
        hardware = FakeHardware()
        recognition = Recognition()
        conversation = VoiceConversation(session, hardware,
                                         backends=backends_for(Mode.ON_ROBOT, session, speech=object(),
                                                               transcriber=recognition),
                                         wake_detector=Wake(), wake_timeout_s=.08)
        conversation.backends.voice.phrases["Yes?"] = (np.full(80, .1, dtype=np.float32),)
        microphone = asyncio.create_task(conversation._microphone())
        supplier = None
        async def background():
            for _ in range(45):
                hardware.samples.put(np.ones(128, dtype=np.float32))
                await asyncio.sleep(.008)
        try:
            while not hardware.states:
                await asyncio.sleep(.001)
            hardware.samples.put(np.full(128, 2, dtype=np.float32))
            while hardware.states[-1] != "listening":
                await asyncio.sleep(.001)
            supplier = asyncio.create_task(background())
            await supplier
            while conversation._wake_deadline is not None or hardware.states[-1] != "idle":
                await asyncio.sleep(.001)
            assert 2 <= recognition.calls < 45
            assert conversation._wake_deadline is None and hardware.states[-1] == "idle"
            assert len(hardware.played) == 1
            assert not session.setup_messages and not session.sent
        finally:
            if supplier is not None and not supplier.done():
                await cancel_task(supplier)
            await cancel_task(microphone)
    asyncio.run(bounded(scenario()))


@pytest.mark.parametrize("marker, expected", [("nod", "nod"), ("unknown", None)])
def test_marker_only_response_moves_silently_or_fails_without_motion(marker, expected):
    class UnusedSpeech:
        def stream(self, *_):
            raise AssertionError("empty text must not reach the speech engine")

    async def scenario():
        hardware = FakeHardware()
        session = FakeSession()
        conversation = VoiceConversation(session, hardware,
                                         backends=backends_for(Mode.ON_ROBOT, session, speech=UnusedSpeech()))
        conversation.tracker = ReplyTracker()
        conversation.tracker.messages["reply-1"] = {"text": f"[reachy:{marker}]"}
        if expected:
            await conversation._speak("reply-1")
            assert hardware.commands == [("reachy.expression", {"name": expected})]
        else:
            with pytest.raises(ValueError, match="empty spoken response"):
                await conversation._speak("reply-1")
            assert hardware.commands == []
        assert hardware.played == []

    asyncio.run(bounded(scenario()))


@pytest.mark.parametrize("response", [{}, {"result": []}, {"message_id": ""}, {"message_id": 10}])
def test_acknowledgement_requires_a_user_message_id(response):
    with pytest.raises(ValueError, match="acknowledgement"):
        ReplyTracker("robot-chat").acknowledge(response)


def test_events_waiting_for_acknowledgement_are_bounded():
    tracker = ReplyTracker("robot-chat")
    for seq in range(64):
        tracker.event(chat_event("delta.text_append", "reply-1", seq=seq + 1, text="word"))
    with pytest.raises(ValueError, match="too many chat events"):
        tracker.event(chat_event("delta.message_done", "reply-1", seq=65))


class FakeHardware:
    sample_rate = 16000
    output_sample_rate = 8000

    def __init__(self):
        self.states = []
        self.state_expressions = []
        self.played = []
        self.play_expressions = []
        self.cleared = 0
        self.progress_cues = 0
        self.samples = queue.Queue()
        self.read_count = 0
        self.commands = []

    def run_command(self, name, params, timeout_ms):
        self.commands.append((name, params))
        return {"ok": True, "payload": params}

    def set_state(self, state, *, expression=None):
        self.states.append(state)
        self.state_expressions.append((state, expression))

    def play_audio(self, samples):
        self.played.append(samples.copy())
        self.play_expressions.append(self.state_expressions[-1] if self.state_expressions else None)

    def clear_audio(self):
        self.cleared += 1

    def cue_progress(self):
        self.progress_cues += 1

    def read_audio(self):
        try:
            sample = self.samples.get_nowait()
        except queue.Empty:
            return None
        self.read_count += 1
        return sample


class FakeSession:
    def __init__(self, mp3=b"", *, acknowledgement=None, send_hook=None, speech_error=None):
        self.registered = asyncio.Event()
        self.registered.set()
        self.chat_subscribed = asyncio.Event()
        self.events = asyncio.Queue()
        self.delivered = asyncio.Queue()
        self.sent = []
        self.sent_options = []
        self.setup_messages = []
        self.tts_requests = []
        self.tts_closed = []
        self.subscription_closed = False
        self.acknowledgement = acknowledgement or {
            "ok": True, "status": 200, "response": {"result": {"message_id": "user-1"}},
        }
        self.send_hook = send_hook
        self.mp3 = mp3
        self.speech_error = speech_error

    async def send_voice(self, wav, session_id, **options):
        self.sent.append((wav, session_id))
        self.sent_options.append(options)
        if self.send_hook is not None:
            await self.send_hook(self)
        return self.acknowledgement

    async def send_chat(self, message, session_id, **options):
        self.setup_messages.append((message, session_id))
        if self.send_hook is not None:
            await self.send_hook(self)
        return self.acknowledgement

    async def subscribe_chat(self, session_id):
        self.chat_subscribed.set()
        try:
            while True:
                event = await self.events.get()
                if isinstance(event, Exception):
                    raise event
                if event is None:
                    return
                yield event
                self.delivered.put_nowait(event)
        finally:
            self.chat_subscribed.clear()
            self.subscription_closed = True

    async def stream_tts(self, message_id):
        self.tts_requests.append(message_id)
        try:
            if self.speech_error is not None:
                raise self.speech_error
            for offset in range(0, len(self.mp3), 257):
                yield self.mp3[offset:offset + 257]
        finally:
            self.tts_closed.append(message_id)


@pytest.fixture
def mp3_tone():
    av = pytest.importorskip("av")
    np = pytest.importorskip("numpy")
    buffer = io.BytesIO()
    with av.open(buffer, "w", format="mp3") as container:
        stream = container.add_stream("libmp3lame", rate=16000)
        stream.layout = "mono"
        samples = (0.25 * np.sin(2 * np.pi * 220 * np.arange(1280) / 16000)).astype(np.float32)
        frame = av.AudioFrame.from_ndarray(samples[None, :], format="fltp", layout="mono")
        frame.sample_rate = 16000
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    return buffer.getvalue()


async def bounded(scenario):
    await asyncio.wait_for(scenario, 4)


async def cancel_task(task):
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def test_ack_race_and_multiple_messages_play_real_mp3_once_each(mp3_tone, monkeypatch, caplog):
    np = pytest.importorskip("numpy")
    monkeypatch.setattr(reachy_voice, "REPLY_QUIET_S", 0.01)
    monkeypatch.setattr(reachy_voice, "ECHO_TAIL_S", 0)

    async def before_ack(session):
        events = [
            chat_event("message.assistant", "other", parent="other-user", seq=1, content="Other"),
            chat_event("delta.message_done", "reply-1", seq=2, content="private spoken words"),
            chat_event("delta.message_done", "reply-1", seq=2, content="private spoken words"),
            chat_event("delta.message_done", "reply-2", seq=3, parent="reply-1", content="Second"),
        ]
        for event in events:
            await session.events.put(event)
            await session.delivered.get()

    async def scenario():
        session = FakeSession(mp3_tone, send_hook=before_ack)
        hardware = FakeHardware()
        conversation = VoiceConversation(session, hardware,
                                         backends=backends_for(Mode.MUSE_VOICE, session),
                                         session_id="robot-chat", reply_timeout_s=2)
        subscriber = asyncio.create_task(conversation._subscribe())
        await session.chat_subscribed.wait()
        await conversation.turn(b"voice recording")
        assert session.sent == [(b"voice recording", "robot-chat")]
        assert session.tts_requests == ["reply-1", "reply-2"]
        assert session.tts_closed == ["reply-1", "reply-2"]
        assert hardware.state_expressions == [("thinking", None), ("thinking", "nod"), ("speaking", None),
                                              ("thinking", None), ("speaking", None), ("thinking", None)]
        pcm = np.concatenate(hardware.played)
        assert pcm.ndim == 1
        assert np.isfinite(pcm).all()
        assert np.max(np.abs(pcm)) > 0.1
        spectrum = np.abs(np.fft.rfft(pcm[:len(pcm) // 2]))
        frequencies = np.fft.rfftfreq(len(pcm) // 2, 1 / hardware.output_sample_rate)
        assert abs(frequencies[np.argmax(spectrum)] - 220) < 15
        assert hardware.cleared == 1
        assert not conversation._muted
        assert conversation.tracker is None
        await cancel_task(subscriber)
        assert session.subscription_closed

    asyncio.run(bounded(scenario()))
    assert "private spoken words" not in caplog.text


def test_replayed_subscription_events_do_not_queue_old_replies():
    async def scenario():
        session = FakeSession()
        hardware = FakeHardware()
        conversation = VoiceConversation(session, hardware,
                                         backends=backends_for(Mode.MUSE_VOICE, session), session_id="robot-chat")
        subscriber = asyncio.create_task(conversation._subscribe())
        await session.chat_subscribed.wait()
        await session.events.put(chat_event("delta.message_done", "old", seq=10, content="old"))
        await session.delivered.get()
        conversation.tracker = ReplyTracker("robot-chat")
        conversation.tracker.acknowledge({"message_id": "user-1"})
        for event in [
            chat_event("delta.message_done", "old", seq=10, content="old"),
            chat_event("delta.message_done", "unrelated", seq=9, parent="other-user", content="old"),
            chat_event("delta.message_done", "reply-1", seq=11, content="current"),
            chat_event("delta.message_done", "reply-2", seq=8, parent="reply-1", content="next"),
            chat_event("delta.message_done", "reply-2", seq=8, parent="reply-1", content="next"),
        ]:
            await session.events.put(event)
            await session.delivered.get()
        assert conversation._replies.get_nowait() == "reply-1"
        assert conversation._replies.get_nowait() == "reply-2"
        assert conversation._replies.empty()
        await cancel_task(subscriber)

    asyncio.run(bounded(scenario()))


@pytest.mark.parametrize("acknowledgement, error_type", [
    ({"ok": False, "status": 403}, ConnectionError),
    ({"ok": True, "response": "invalid"}, ValueError),
    ({"ok": True, "response": {}}, ValueError),
])
def test_rejected_or_invalid_voice_ack_restores_capture_and_clears_playback(
    acknowledgement, error_type, monkeypatch,
):
    monkeypatch.setattr(reachy_voice, "ECHO_TAIL_S", 0)

    async def scenario():
        session = FakeSession(acknowledgement=acknowledgement)
        hardware = FakeHardware()
        conversation = VoiceConversation(session, hardware, backends=backends_for(Mode.MUSE_VOICE, session))
        with pytest.raises(error_type):
            await conversation.turn(b"recording")
        assert session.tts_requests == []
        assert hardware.cleared == 1
        assert not conversation._muted
        assert conversation.tracker is None

    asyncio.run(bounded(scenario()))


def test_voice_reply_timeout_restores_capture(monkeypatch):
    monkeypatch.setattr(reachy_voice, "ECHO_TAIL_S", 0)

    async def scenario():
        hardware = FakeHardware()
        session = FakeSession()
        conversation = VoiceConversation(session, hardware,
                                         backends=backends_for(Mode.MUSE_VOICE, session), reply_timeout_s=0.01)
        with pytest.raises(TimeoutError, match="did not complete"):
            await conversation.turn(b"recording")
        assert hardware.cleared == 1
        assert not conversation._muted

    asyncio.run(bounded(scenario()))


@pytest.mark.parametrize("recognition_delay", [0.0, 0.2])
def test_timely_final_reply_finishes_all_sentences_after_response_deadline(
        monkeypatch, streaming_speech, recognition_delay, caplog):
    from types import SimpleNamespace
    np = pytest.importorskip("numpy")
    clock = [0.0]
    monkeypatch.setattr(reachy_voice, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    caplog.set_level("INFO", logger="musegadget.reachy_voice")

    async def scenario():
        hardware = FakeHardware()
        sentences = [("First, choose a movie.", "happy"),
                     ("Then order dinner.", "curious"),
                     ("Finally, enjoy your evening.", "neutral")]
        final = "".join(sentence_frame(text, expression) for text, expression in sentences)

        class Transcriber:
            async def transcribe(self, wav):
                clock[0] += recognition_delay
                return "Help me plan movie night."

        class Speech(streaming_speech):
            async def stream(self, text, rate):
                clock[0] += .12
                async for chunk in super().stream(text, rate):
                    yield chunk

        async def deliver(session):
            # The complete backend answer arrives just before its .15s budget.
            clock[0] += .14
            await session.events.put(chat_event("delta.message_done", "reply-1", content=final, seq=1))
            await session.delivered.get()
            await session.events.put(chat_event("task.status", status="completed", seq=2))
            await session.delivered.get()

        session = FakeSession(send_hook=deliver)
        speech = Speech(hardware)
        conversation = VoiceConversation(session, hardware,
                                         backends=backends_for(Mode.ON_ROBOT, session, speech=speech,
                                                               transcriber=Transcriber(),
                                                               stream_replies=True),
                                         session_id="robot-chat", reply_timeout_s=.15)
        subscriber = asyncio.create_task(conversation._subscribe())
        try:
            await session.chat_subscribed.wait()
            assert await conversation.turn(b"public recording") is TurnOutcome.ACCEPTED
            assert speech.requests == [text for text, _ in sentences]
            assert speech.closed == speech.requests
            assert hardware.play_expressions == [("speaking", expression) for _, expression in sentences]
            assert len(hardware.played) == 3 and np.concatenate(hardware.played).size == 240
            assert clock[0] > recognition_delay + .15
            assert not conversation._muted and conversation.tracker is None
            assert "Reachy voice turn started" in caplog.text
            assert "Reachy voice turn ended" in caplog.text
        finally:
            await cancel_task(subscriber)

    asyncio.run(bounded(scenario()))


def test_stalled_answer_speech_has_its_own_bound_and_closes_playback(monkeypatch, streaming_speech):
    monkeypatch.setattr(reachy_voice, "SPEECH_TIMEOUT_S", .03)

    async def scenario():
        entered, closed = asyncio.Event(), asyncio.Event()

        class StalledSpeech:
            async def stream(self, text, rate):
                entered.set()
                try:
                    await asyncio.Event().wait()
                    yield None
                finally:
                    closed.set()

        async def deliver(session):
            await session.events.put(chat_event("delta.message_done", "reply-1",
                                                content=sentence_frame("Here is your answer."), seq=1))
            await session.delivered.get()
            await session.events.put(chat_event("task.status", status="completed", seq=2))
            await session.delivered.get()

        hardware = FakeHardware()
        session = FakeSession(send_hook=deliver)
        conversation = VoiceConversation(session, hardware,
                                         backends=backends_for(Mode.ON_ROBOT, session, speech=StalledSpeech(),
                                                               stream_replies=True),
                                         session_id="robot-chat", reply_timeout_s=2)
        subscriber = asyncio.create_task(conversation._subscribe())
        try:
            await session.chat_subscribed.wait()
            with pytest.raises(TimeoutError):
                await conversation.turn(b"public recording")
            assert entered.is_set() and closed.is_set()
            assert hardware.played == [] and hardware.cleared == 1
            assert not conversation._muted and conversation.tracker is None
        finally:
            await cancel_task(subscriber)

    asyncio.run(bounded(scenario()))


def test_subscription_error_ends_conversation_and_flushes_playback(monkeypatch):
    monkeypatch.setattr(reachy_voice, "ECHO_TAIL_S", 0)

    async def scenario():
        session = FakeSession()
        hardware = FakeHardware()
        await session.events.put(ConnectionError("subscription refused"))
        conversation = VoiceConversation(session, hardware, backends=backends_for(Mode.MUSE_VOICE, session))
        with pytest.raises(ConnectionError, match="subscription refused"):
            await conversation.run()
        assert session.subscription_closed
        assert hardware.cleared >= 1
        assert not hardware.played

    asyncio.run(bounded(scenario()))


def test_cancelling_speech_closes_tts_before_returning(mp3_tone):
    async def scenario():
        session = FakeSession(mp3_tone)
        hardware = FakeHardware()
        conversation = VoiceConversation(session, hardware, backends=backends_for(Mode.MUSE_VOICE, session))
        speaking = asyncio.create_task(conversation._speak("reply-1"))
        while not hardware.played:
            await asyncio.sleep(0.001)
        await cancel_task(speaking)
        assert session.tts_closed == ["reply-1"]

    asyncio.run(bounded(scenario()))


def test_microphone_continues_draining_echo_while_reply_is_active(monkeypatch):
    class Recorder:
        def finish_initial_capture(self):
            pass
        active = False
        last_truncated = False

        def __init__(self, *args, **kwargs):
            pass

        def feed(self, sample):
            return bytes(sample) if len(sample) else None

        def reset(self):
            pass

    from musegadget import voice_audio
    monkeypatch.setattr(voice_audio, "TurnRecorder", Recorder)

    async def scenario():
        session = FakeSession()
        session.chat_subscribed.set()
        hardware = FakeHardware()
        conversation = VoiceConversation(session, hardware, backends=backends_for(Mode.MUSE_VOICE, session))
        started = asyncio.Event()
        release = asyncio.Event()
        recorded = []

        async def delayed_turn(wav, **options):
            recorded.append(wav)
            conversation._muted = True
            started.set()
            try:
                await release.wait()
            finally:
                conversation._muted = False

        conversation.turn = delayed_turn
        hardware.samples.put(b"stale speech from before readiness")
        microphone = asyncio.create_task(conversation._microphone())
        while hardware.states.count("idle") < 1:
            await asyncio.sleep(0.001)
        assert hardware.read_count == 1
        assert recorded == []
        hardware.samples.put(b"user speech")
        await started.wait()
        hardware.samples.put(b"speaker echo one")
        hardware.samples.put(b"speaker echo two")
        while hardware.read_count < 4:
            await asyncio.sleep(0.001)
        assert recorded == [b"user speech"]
        release.set()
        while hardware.states.count("idle") < 2:
            await asyncio.sleep(0.001)
        hardware.samples.put(b"next user speech")
        while len(recorded) < 2:
            await asyncio.sleep(0.001)
        assert recorded == [b"user speech", b"next user speech"]
        await cancel_task(microphone)
        assert not conversation._muted

    asyncio.run(bounded(scenario()))


def test_real_noise_voice_note_gets_text_completion_and_paced_tts_burst(mp3_tone, monkeypatch):
    from test_link_client import registered_session
    from musegadget.noise import ApplicationResponse, BodyChunk, ServiceFrame

    monkeypatch.setattr(reachy_voice, "REPLY_QUIET_S", 0.01)
    monkeypatch.setattr(reachy_voice, "ECHO_TAIL_S", 0)

    async def scenario():
        session, vm, stop, task = await registered_session()
        hardware = FakeHardware()
        conversation = VoiceConversation(session, hardware, backends=backends_for(Mode.MUSE_VOICE, session))
        subscriber = asyncio.create_task(conversation._subscribe())
        subscription = await vm.next_frame()
        assert subscription.value.path == "/chat/subscribe"
        assert json.loads(subscription.value.body) == {}
        await vm.send_frame(ServiceFrame.response(subscription.stream_id, ApplicationResponse(status=200)))
        await session.chat_subscribed.wait()
        recording = io.BytesIO()
        with wave.open(recording, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(b"\x01\x00" * 1280)
        speaking = asyncio.create_task(conversation.turn(recording.getvalue()))
        voice_note = await vm.next_frame()
        assert voice_note.value.path == "/chat/stream"
        uploaded = json.loads(voice_note.value.body)
        assert uploaded["output_modality"] == "voice"
        assert uploaded["items"][0]["mime_type"] == "audio/wav"
        await vm.send_frame(ServiceFrame.response(voice_note.stream_id, ApplicationResponse(
            status=200, body=b'{"result":{"message_id":"user-1"}}', end_body=True,
        )))
        await vm.send_frame(ServiceFrame.body_chunk(subscription.stream_id, BodyChunk(
            data=json.dumps(chat_event("delta.message_done", "reply-1", session_id=None,
                                       content="Test response")).encode() + b"\n",
        )))
        request = await vm.next_frame()
        assert request.value.path == "/api/voice/tts-stream?message_id=reply-1"
        await vm.send_frame(ServiceFrame.response(request.stream_id, ApplicationResponse(status=200)))
        # A VM can deliver compressed audio much faster than the speaker plays
        # it. Tiny network chunks must not consume one bounded queue slot each.
        assert len(mp3_tone) > 300
        for offset in range(0, len(mp3_tone), 3):
            await vm.send_frame(ServiceFrame.body_chunk(request.stream_id, BodyChunk(
                data=mp3_tone[offset:offset + 3], end_body=offset + 3 >= len(mp3_tone),
            )))
        await speaking
        assert hardware.state_expressions == [("thinking", None), ("thinking", "nod"), ("speaking", None),
                                              ("thinking", None)]
        assert sum(len(samples) for samples in hardware.played) >= 640
        assert not conversation._muted
        await cancel_task(subscriber)
        reset = await vm.next_frame()
        assert (reset.kind, reset.stream_id) == ("reset", subscription.stream_id)
        assert not session._streams
        stop.set()
        await task

    asyncio.run(bounded(scenario()))


def test_initial_motion_failure_restores_capture_and_releases_tracker(monkeypatch):
    monkeypatch.setattr(reachy_voice, "ECHO_TAIL_S", 0)

    class FailedHardware(FakeHardware):
        def set_state(self, state):
            raise RuntimeError("motion hardware disconnected")

    async def scenario():
        session = FakeSession()
        hardware = FailedHardware()
        conversation = VoiceConversation(session, hardware, backends=backends_for(Mode.MUSE_VOICE, session))
        with pytest.raises(RuntimeError, match="hardware disconnected"):
            await conversation.turn(b"recording")
        assert session.sent == []
        assert hardware.cleared == 1
        assert conversation.tracker is None
        assert not conversation._muted

    asyncio.run(bounded(scenario()))


def test_playback_flush_failure_still_restores_capture(monkeypatch):
    monkeypatch.setattr(reachy_voice, "ECHO_TAIL_S", 0)

    class FailedHardware(FakeHardware):
        def clear_audio(self):
            raise RuntimeError("playback flush unavailable")

    async def scenario():
        session = FakeSession(acknowledgement={"ok": False, "status": 403})
        conversation = VoiceConversation(session, FailedHardware(), backends=backends_for(Mode.MUSE_VOICE, session))
        with pytest.raises(RuntimeError, match="flush unavailable"):
            await conversation.turn(b"recording")
        assert conversation.tracker is None
        assert not conversation._muted

    asyncio.run(bounded(scenario()))


def test_cancelling_an_unacknowledged_turn_flushes_audio_and_restores_capture(monkeypatch):
    monkeypatch.setattr(reachy_voice, "ECHO_TAIL_S", 0)

    async def scenario():
        sending = asyncio.Event()

        async def wait_for_ack(session):
            sending.set()
            await asyncio.Event().wait()

        hardware = FakeHardware()
        session = FakeSession(send_hook=wait_for_ack)
        conversation = VoiceConversation(session, hardware, backends=backends_for(Mode.MUSE_VOICE, session))
        turning = asyncio.create_task(conversation.turn(b"recording"))
        await sending.wait()
        assert not conversation._input_blocked()
        await cancel_task(turning)
        assert hardware.cleared == 1
        assert session.tts_requests == []
        assert conversation.tracker is None
        assert not conversation._muted

    asyncio.run(bounded(scenario()))


def test_hardware_failure_leaves_service_for_a_fresh_controller(monkeypatch):
    from musegadget.identity import Identity
    from musegadget.link_client import Outcome
    from musegadget.reachy_hardware import ReachyHardwareError

    class IdleLink:
        registered_at = None
        stopped = False

        def __init__(self, **kwargs):
            self.registered = asyncio.Event()
            self.registered.set()

        async def run(self, stop):
            await stop.wait()
            self.stopped = True
            return Outcome.STOPPED

    class FailedConversation:
        def __init__(self, *args, **kwargs):
            pass

        async def run(self):
            raise ReachyHardwareError("microphone disconnected")

    monkeypatch.setattr(reachy_voice, "LinkSession", IdleLink)
    monkeypatch.setattr(reachy_voice, "VoiceConversation", FailedConversation)

    async def scenario():
        hardware = FakeHardware()
        hardware.run_command = lambda *args: {"ok": True}
        service = reachy_voice.ReachyService(identity=Identity("02:00:00:ab:cd:ef"), executor=hardware,
                                             backends=lambda session: backends_for(Mode.MUSE_VOICE, session))
        with pytest.raises(ReachyHardwareError, match="microphone disconnected"):
            await service._session({"vm_id": "vm-1", "vm_auth_token": "token"}, {})
        assert service._current is None
        assert hardware.states == []

    asyncio.run(bounded(scenario()))


def test_cli_closes_failed_hardware_and_exits_for_service_restart(monkeypatch, tmp_path, capsys):
    from musegadget import config, reachy_cli
    from musegadget.identity import Identity
    from musegadget.reachy_hardware import ReachyHardwareError

    class Hardware:
        starts = 0
        closes = 0

        def start(self):
            self.starts += 1

        def close(self):
            self.closes += 1

    class FailedService:
        runs = 0

        def __init__(self, **kwargs):
            pass

        def stop(self):
            pass

        async def run(self):
            type(self).runs += 1
            raise ReachyHardwareError("microphone disconnected")

    hardware = Hardware()
    monkeypatch.setenv(config.STATE_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(config, "load_json", lambda *args: {"paired": True})
    monkeypatch.setattr(config, "sdk_token", lambda: None)
    monkeypatch.setattr(reachy_cli.identity, "load_or_create", lambda: Identity("02:00:00:ab:cd:ef"))
    monkeypatch.setattr(reachy_cli, "_hardware", lambda *args, **kwargs: hardware)
    monkeypatch.setattr(reachy_voice, "ReachyService", FailedService)
    assert reachy_cli.main(["--state-dir", str(tmp_path), "run"]) == 1
    assert (hardware.starts, hardware.closes, FailedService.runs) == (1, 1, 1)
    assert "Reachy: microphone disconnected" in capsys.readouterr().err


@pytest.mark.parametrize("chat_args, expected_session", [
    ([], None), (["--session-id", "existing-chat"], "existing-chat"),
])
def test_cli_passes_explicit_session_or_default_route_to_service(
    chat_args, expected_session, monkeypatch, tmp_path,
):
    from musegadget import config, reachy_cli
    from musegadget.identity import Identity

    class Hardware:
        def start(self):
            pass

        def close(self):
            pass

    selected = []

    class Service:
        def __init__(self, **kwargs):
            selected.append(kwargs["session_id"])

        def stop(self):
            pass

        async def run(self):
            pass

    monkeypatch.setenv(config.STATE_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(config, "load_json", lambda *args: {"paired": True})
    monkeypatch.setattr(config, "sdk_token", lambda: None)
    monkeypatch.setattr(reachy_cli.identity, "load_or_create", lambda: Identity("02:00:00:ab:cd:ef"))
    monkeypatch.setattr(reachy_cli, "_hardware", lambda *args, **kwargs: Hardware())
    monkeypatch.setattr(reachy_voice, "ReachyService", Service)
    assert reachy_cli.main(["--state-dir", str(tmp_path), "run", *chat_args]) == 0
    assert selected == [expected_session]
