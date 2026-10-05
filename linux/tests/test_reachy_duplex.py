"""Conversation input and ordered speech remain independent without talking over users."""

import asyncio

import pytest

from musegadget import reachy_voice
from musegadget.reachy_capabilities import Mode
from musegadget.reachy_local_backends import backends_for
from musegadget.reachy_voice import VoiceConversation, _SpeechJob
from test_reachy_duplex_voice import duplex
from test_reachy_voice import FakeHardware, FakeSession, bounded, cancel_task, chat_event, sentence_frame


class OrderedSpeech:
    def __init__(self):
        self.requests = []
        self.closed = []
        self.active = False

    async def stream(self, text, rate):
        np = pytest.importorskip("numpy")
        assert not self.active, "only the global speaker may consume speech"
        self.active = True
        self.requests.append(text)
        try:
            marker = (len(self.requests) + 1) / 10
            yield np.full(80, marker, dtype=np.float32)
        finally:
            self.active = False
            self.closed.append(text)


async def _deliver_reply(session, number, text, expression):
    parent = f"user-{number}"
    await session.events.put(chat_event(
        "delta.message_done", f"reply-{number}", parent=parent, seq=number * 2,
        content=sentence_frame(text, expression),
    ))
    await session.delivered.get()
    await session.events.put(chat_event(
        "task.status", parent=parent, task_id=f"task-{number}",
        status="completed", seq=number * 2 + 1,
    ))
    await session.delivered.get()


def test_paused_old_answer_does_not_block_next_request_and_jobs_keep_turn_scope(monkeypatch):
    monkeypatch.setattr(reachy_voice, "REPLY_QUIET_S", 0)
    monkeypatch.setattr(reachy_voice, "ECHO_TAIL_S", 0)

    async def scenario():
        hardware = FakeHardware()
        speech = OrderedSpeech()
        producers = []

        async def send(session):
            number = len(session.setup_messages)
            session.acknowledgement = {
                "ok": True, "status": 200,
                "response": {"message_id": f"user-{number}"},
            }
            producers.append(asyncio.create_task(_deliver_reply(
                session, number, f"Answer {number}.", "happy" if number == 1 else "curious",
            )))

        session = FakeSession(send_hook=send)
        conversation = VoiceConversation(
            session, hardware,
                                         backends=backends_for(Mode.ON_ROBOT, session, speech=speech, stream_replies=True),
                                         session_id="robot-chat",
        )
        subscriber = asyncio.create_task(conversation._subscribe())
        speaker = asyncio.create_task(conversation._play_output())
        conversation._playback.set_user_speaking(True)
        try:
            await conversation.turn(
                b"first", recognized_text="First question.", defer_playback=True,
            )
            assert conversation.tracker is None and conversation._has_output()
            await conversation.turn(
                b"second", recognized_text="Second question.", defer_playback=True,
            )
            assert len(session.setup_messages) == 2
            assert conversation.tracker is None and conversation._has_output()
            assert not hardware.played and not hardware.play_expressions

            conversation._playback.set_user_speaking(False)
            await conversation._output_queue.join()
            assert speech.requests == ["Answer 1.", "Answer 2."]
            assert speech.closed == speech.requests
            assert hardware.play_expressions == [
                ("speaking", "happy"), ("speaking", "curious"),
            ]
        finally:
            conversation._playback.set_user_speaking(False)
            speaker.cancel()
            await asyncio.gather(speaker, *producers, return_exceptions=True)
            await cancel_task(subscriber)

    asyncio.run(bounded(scenario()))


def test_cancelling_global_speaker_closes_current_stream_and_drops_queued_tail():
    async def scenario():
        hardware = FakeHardware()
        speech = OrderedSpeech()
        session = FakeSession()
        conversation = VoiceConversation(session, hardware,
                                         backends=backends_for(Mode.ON_ROBOT, session, speech=speech))
        conversation._playback.set_user_speaking(True)
        conversation._output_queue.put_nowait(_SpeechJob(None, "First."))
        conversation._output_queue.put_nowait(_SpeechJob(None, "Second."))
        speaker = asyncio.create_task(conversation._play_output())
        try:
            while speech.requests != ["First."]:
                await asyncio.sleep(0)
            await cancel_task(speaker)
            assert speech.closed == ["First."]
            assert not hardware.played
            assert conversation._output_queue.empty()
        finally:
            conversation._playback.set_user_speaking(False)
            if not speaker.done():
                await cancel_task(speaker)

    asyncio.run(bounded(scenario()))


def test_final_revision_while_output_is_held_drops_every_draft_job(monkeypatch):
    monkeypatch.setattr(reachy_voice, "REPLY_QUIET_S", 0)

    async def scenario():
        hardware = FakeHardware()
        speech = OrderedSpeech()
        draft = sentence_frame("Draft one.") + sentence_frame("Draft two.", "curious")

        async def send(session):
            async def deliver():
                await session.events.put(chat_event(
                    "delta.text_append", "reply-1", text=draft,
                ))
                await session.delivered.get()
                await session.events.put(chat_event(
                    "delta.message_done", "reply-1",
                    content=sentence_frame("Authoritative answer.", "happy"), seq=2,
                ))
                await session.delivered.get()
                await session.events.put(chat_event(
                    "task.status", task_id="task-1", status="completed", seq=3,
                ))
                await session.delivered.get()
            asyncio.create_task(deliver())

        session = FakeSession(send_hook=send)
        conversation = VoiceConversation(
            session, hardware,
                                         backends=backends_for(Mode.ON_ROBOT, session, speech=speech, stream_replies=True),
        )
        subscriber = asyncio.create_task(conversation._subscribe())
        speaker = asyncio.create_task(conversation._play_output())
        conversation._playback.set_user_speaking(True)
        try:
            await conversation.turn(
                b"voice", recognized_text="Question.", defer_playback=True,
            )
            assert not hardware.played
            conversation._playback.set_user_speaking(False)
            await conversation._output_queue.join()
            assert hardware.play_expressions == [("speaking", "happy")]
            assert speech.closed[-1] == "Authoritative answer."
        finally:
            conversation._playback.set_user_speaking(False)
            speaker.cancel()
            await asyncio.gather(speaker, return_exceptions=True)
            await cancel_task(subscriber)

    asyncio.run(bounded(scenario()))


def test_wake_and_question_are_admitted_while_qualified_output_is_playing(duplex):
    np = pytest.importorskip("numpy")
    started = asyncio.Event()
    finish = asyncio.Event()

    class Speech:
        async def stream(self, text, rate):
            yield np.full(80, .1, np.float32)
            started.set()
            await finish.wait()

    async def run(ctx):
        speaking = asyncio.create_task(ctx.conversation._speak(
            None, text="An older answer.", stream_segment=True,
        ))
        try:
            await started.wait()
            await ctx.feed((.75, 7), (.2, 16), (0, 6))
            await ctx.entered[0].wait()
            assert ctx.labels() == [2]
            assert ctx.conversation._playback.user_speaking is False
        finally:
            finish.set()
            ctx.release[0].set()
            await speaking

    asyncio.run(bounded(duplex(run, echo=True, wake=True, speech=Speech())))


def test_confirmed_wake_alone_pauses_old_output_until_fresh_silence(duplex):
    np = pytest.importorskip("numpy")
    started = asyncio.Event()

    class Speech:
        async def stream(self, text, rate):
            started.set()
            yield np.full(rate * 2, .1, np.float32)

    async def run(ctx):
        speaking = asyncio.create_task(ctx.conversation._speak(
            None, text="An older answer that must yield.", stream_segment=True,
        ))
        await started.wait()
        await ctx.playback.get()
        await ctx.feed((.75, 7), (0, 1))
        assert ctx.conversation._wake_deadline is not None
        assert ctx.conversation._playback.user_speaking
        played = len(ctx.hardware.played)
        await asyncio.sleep(.02)
        assert len(ctx.hardware.played) == played

        # The wake chunk already supplied one 20 ms quiet frame. Fourteen
        # more complete the configured 300 ms fresh-silence hold.
        await ctx.feed((0, 14))
        while ctx.conversation._playback.user_speaking:
            await asyncio.sleep(0)
        await speaking
        assert len(ctx.hardware.played) > played
        assert ctx.records == []

    asyncio.run(bounded(duplex(
        run, echo=True, wake=True, speech=Speech(), silence_s=.3,
    )))


def test_capture_gap_during_user_speech_requires_a_fresh_quiet_interval(duplex):
    np = pytest.importorskip("numpy")
    started = asyncio.Event()

    class Speech:
        async def stream(self, text, rate):
            started.set()
            yield np.full(rate, .1, np.float32)

    async def run(ctx):
        speaking = asyncio.create_task(ctx.conversation._speak(
            None, text="A sentence with an unsent tail.", stream_segment=True,
        ))
        await started.wait()
        await ctx.playback.get()
        await ctx.feed((.2, 3))
        assert ctx.conversation._playback.user_speaking
        played = len(ctx.hardware.played)

        await ctx.feed((0, 1), gap=True)
        await ctx.feed((0, 3))
        await asyncio.sleep(.02)
        assert ctx.conversation._playback.user_speaking
        assert len(ctx.hardware.played) == played

        await ctx.feed((0, 1))
        while ctx.conversation._playback.user_speaking:
            await asyncio.sleep(0)
        await speaking
        assert len(ctx.hardware.played) > played

    asyncio.run(bounded(duplex(
        run, echo=True, speech=Speech(), silence_s=.1, owned=True,
    )))


def test_next_asr_runs_before_the_previous_backend_turn_finishes(duplex):
    async def run(ctx):
        acknowledged = [asyncio.Event(), asyncio.Event()]

        async def acknowledge(session):
            number = len(session.setup_messages)
            session.acknowledgement = {
                "ok": True, "status": 200,
                "response": {"message_id": f"user-{number}", "session_id": "robot-chat", "is_thread": True},
            }
            acknowledged[number - 1].set()

        ctx.session.send_hook = acknowledge
        subscriber = asyncio.create_task(ctx.conversation._subscribe())
        try:
            await ctx.question(.1)
            await ctx.entered[0].wait()
            await ctx.question(.2)
            assert not ctx.entered[1].is_set()

            ctx.release[0].set()
            await acknowledged[0].wait()
            # The serial ASR worker advances even though turn one still owns
            # Muse and has received neither an answer nor a terminal event.
            await ctx.entered[1].wait()
            assert len(ctx.session.setup_messages) == 1
            ctx.release[1].set()

            await _deliver_reply(ctx.session, 1, "First answer.", "happy")
            await acknowledged[1].wait()
            await _deliver_reply(ctx.session, 2, "Second answer.", "curious")
            await ctx.playback.get()
            await ctx.playback.get()
            assert len(ctx.session.setup_messages) == 2
        finally:
            await cancel_task(subscriber)

    asyncio.run(bounded(duplex(
        run, echo=True, real_turn=True, hold_recognition=True, speech=OrderedSpeech(), owned=True,
    )))
