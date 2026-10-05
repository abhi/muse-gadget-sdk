# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Behavioral contract for persistent Reachy side-chat instructions."""

import asyncio

import pytest

from musegadget import reachy_voice
from musegadget.reachy_capabilities import Mode
from musegadget.reachy_local_backends import backends_for
from musegadget.reachy_voice import TurnOutcome, VoiceConversation


class Session:
    def __init__(self):
        self.registered = asyncio.Event()
        self.registered.set()
        self.chat_subscribed = asyncio.Event()
        self.chat_subscribed.set()
        self.sent = []
        self.conversation = None

    async def send_chat(self, text, session_id, **options):
        self.sent.append((text, session_id, options))
        for status in ("running", "completed"):
            self.conversation.job.tracker.event({
                "type": "event",
                "event": "task.status",
                "payload": {
                    "session_id": session_id,
                    "task_id": "task-1",
                    "reply_to_message_id": "user-1",
                    "status": status,
                },
            })
        return {"ok": True, "response": {"result": {
            "session_id": session_id,
            "is_thread": True,
            "message_id": "user-1",
        }}}


class Hardware:
    sample_rate = 16000
    output_sample_rate = 16000
    motion_enabled = True
    antenna_mode = "both"
    face_tracking_enabled = False

    def __init__(self):
        self.states = []
        self.cleared = 0

    def set_state(self, state, **options):
        self.states.append(state)

    def clear_audio(self):
        self.cleared += 1

    def read_audio(self):
        return None


async def owned_turn(text):
    session = Session()
    conversation = VoiceConversation(
        session,
        Hardware(), backends=backends_for(Mode.ON_ROBOT, session, speech=object(), transcriber=object()),
        session_id="owned-chat",
        owns_chat=True,
        wake_detector=type("Wake", (), {"phrase": "hey muse"})(),
    )
    session.conversation = conversation
    outcome = await conversation.turn(b"wav", recognized_text=text)
    return outcome, session.sent


def test_owned_chat_sends_only_the_recognized_user_transcript(monkeypatch):
    monkeypatch.setattr(reachy_voice, "EMPTY_REPLY_GRACE_S", 0)
    outcome, sent = asyncio.run(owned_turn("What should I cook for lunch?"))
    assert outcome is TurnOutcome.ACCEPTED
    assert sent == [("What should I cook for lunch?", "owned-chat", {})]


def test_new_owned_conversation_cannot_reinject_context_after_reconnect(monkeypatch):
    monkeypatch.setattr(reachy_voice, "EMPTY_REPLY_GRACE_S", 0)

    async def check():
        first = await owned_turn("First request")
        reconnected = await owned_turn("Second request")
        return first, reconnected

    (first_outcome, first_sent), (second_outcome, second_sent) = asyncio.run(check())
    assert first_outcome is second_outcome is TurnOutcome.ACCEPTED
    assert first_sent[0][0] == "First request"
    assert second_sent[0][0] == "Second request"
    assert "The user's spoken request is" not in first_sent[0][0] + second_sent[0][0]


@pytest.mark.parametrize("owns_chat, expected", [(True, []), (False, [None])])
def test_owned_microphone_does_not_send_expression_bootstrap(owns_chat, expected):
    pytest.importorskip("webrtcvad")
    pytest.importorskip("av")
    async def check():
        calls = []
        session = Session()
        conversation = VoiceConversation(
            session, Hardware(),
                                         backends=backends_for(Mode.ON_ROBOT, session, speech=object()),
                                         session_id="owned-chat" if owns_chat else None,
            owns_chat=owns_chat,
        )

        async def turn(wav, **options):
            calls.append(wav)
            await asyncio.Event().wait()

        conversation.turn = turn
        microphone = asyncio.create_task(conversation._microphone())
        await asyncio.sleep(0.03)
        microphone.cancel()
        with pytest.raises(asyncio.CancelledError):
            await microphone
        assert calls == expected

    asyncio.run(check())
