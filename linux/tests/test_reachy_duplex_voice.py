"""Continuous authorized input with one owner of Muse turns and speech."""

import asyncio
from dataclasses import replace
import io
import wave

import pytest

from musegadget import reachy_voice, voice_audio
from musegadget.reachy_capabilities import Mode, ReplyStyle
from musegadget.reachy_local_backends import LocalHearing, backends_for
from musegadget.reachy_voice import TurnOutcome, VoiceConversation
from musegadget.wake_word import WakeDetection
from test_reachy_voice import FakeHardware, FakeSession, bounded, cancel_task, chat_event, sentence_frame


@pytest.fixture
def duplex(monkeypatch):
    np = pytest.importorskip("numpy")
    pytest.importorskip("av")
    monkeypatch.setattr(reachy_voice, "ECHO_TAIL_S", 0)
    monkeypatch.setattr(reachy_voice, "REPLY_QUIET_S", .01)

    class Vad:
        def is_speech(self, pcm, rate):
            return bool(np.any(np.frombuffer(pcm, dtype="<i2")))

    def read(wav):
        with wave.open(io.BytesIO(wav), "rb") as source:
            return np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").copy()

    async def scenario(run, *, echo=True, wake=False, real_turn=False, speech=None, silence_s=.1,
                       wake_delay_feeds=0, hold_recognition=False, transcriber=None, owned=False):
        loop = asyncio.get_running_loop()
        ready = asyncio.Event()
        consumer_ready = asyncio.Event()
        acknowledgements = {}
        gaps = set()
        records = []
        entered = [asyncio.Event() for _ in range(12)]
        release = [asyncio.Event() for _ in range(12)]
        finished = [asyncio.Event() for _ in range(12)]
        playback = asyncio.Queue()
        active = 0
        maximum_active = 0

        class Hardware(FakeHardware):
            echo_cancelled_input = echo

            def set_state(self, state, **options):
                super().set_state(state, **options)
                loop.call_soon_threadsafe(ready.set)

            def play_audio(self, samples):
                super().play_audio(samples)
                loop.call_soon_threadsafe(playback.put_nowait, samples.copy())

        original_buffer = reachy_voice._CaptureBuffer

        class Buffer(original_buffer):
            previous = None

            def push(self, samples, **options):
                if id(samples) in gaps:
                    options["gap"] = True
                    gaps.remove(id(samples))
                super().push(samples, **options)

            async def get(self):
                consumer_ready.set()
                if self.previous is not None and not self.previous.done():
                    self.previous.set_result(None)
                self.previous = None
                chunk = await super().get()
                if chunk is not None:
                    self.previous = acknowledgements.pop(id(chunk.samples), None)
                return chunk

        monkeypatch.setattr(reachy_voice, "_CaptureBuffer", Buffer)
        session = FakeSession()
        session.chat_subscribed.set()
        hardware = Hardware()

        class Wake:
            sample_rate = 16000
            phrase = "hey muse"
            epoch = 0
            feed_id = 0
            last_detection = None
            pending_samples = None
            pending_feeds = 0

            def reset(self):
                self.epoch += 1
                self.last_detection = None
                self.pending_samples = None

            def feed(self, audio):
                self.feed_id += 1
                self.last_detection = None
                indices = np.flatnonzero(audio == np.float32(.75))
                if len(indices):
                    self.pending_samples = len(audio) - int(indices[-1]) - 1
                    self.pending_feeds = wake_delay_feeds
                elif self.pending_samples is not None:
                    self.pending_samples += len(audio)
                    self.pending_feeds -= 1
                else:
                    return False
                if self.pending_feeds:
                    return False
                self.last_detection = WakeDetection(self.pending_samples, self.epoch, self.feed_id)
                return True

        class Recognition:
            async def transcribe(self, wav):
                records.append(read(wav))
                index = len(records) - 1
                entered[index].set()
                if hold_recognition:
                    await release[index].wait()
                return "Question one."

        options = {} if silence_s is None else {"silence_s": silence_s}
        backends = (backends_for(Mode.ON_ROBOT, session, speech=speech or object(),
                                 transcriber=transcriber or Recognition())
                    if speech is not None or wake or transcriber is not None or hold_recognition
                    else backends_for(Mode.MUSE_VOICE, session))
        if owned:
            options.update(session_id="robot-chat", owns_chat=True)
            session.acknowledgement["response"]["result"].update(session_id="robot-chat", is_thread=True)
        conversation = VoiceConversation(session, hardware, **options, backends=backends, speech_gate=Vad(),
                                         wake_detector=Wake() if wake else None)

        async def turn(wav, **options):
            nonlocal active, maximum_active
            index = len(records)
            records.append(read(wav))
            active += 1
            maximum_active = max(maximum_active, active)
            entered[index].set()
            try:
                await release[index].wait()
                return TurnOutcome.ACCEPTED
            finally:
                active -= 1
                finished[index].set()

        if not real_turn:
            # These scenarios replace the whole Muse turn with the controlled
            # function below.  Do not also start the production ASR preprocessor;
            # the replacement records each admitted WAV itself.
            conversation.backends = replace(backends, hearing=LocalHearing())
            conversation.turn = turn
        microphone = asyncio.create_task(conversation._microphone())

        class Context:
            def __init__(self):
                self.conversation = conversation
                self.hardware = hardware
                self.session = session
                self.records = records
                self.entered = entered
                self.release = release
                self.finished = finished
                self.microphone = microphone
                self.playback = playback

            async def feed(self, *parts, gap=False):
                audio = np.concatenate([np.full(frames * 320, marker, np.float32)
                                        for marker, frames in parts])
                assert len(audio) <= 16000
                processed = loop.create_future()
                acknowledgements[id(audio)] = processed
                if gap:
                    gaps.add(id(audio))
                hardware.samples.put(audio)
                done, _ = await asyncio.wait({processed, microphone}, timeout=1,
                                             return_when=asyncio.FIRST_COMPLETED)
                if microphone in done:
                    microphone.result()
                    pytest.fail("microphone ended while accepting input")
                assert processed in done, "microphone did not finish consuming the queued frame"

            async def question(self, marker):
                await self.feed((marker, 16), (0, 6))

            def labels(self):
                return [round(int(pcm[pcm != 0][0]) / 32768 * 10) for pcm in records]

            def maximum_active(self):
                return maximum_active

        try:
            await ready.wait()
            await consumer_ready.wait()
            await run(Context())
        finally:
            for event in release:
                event.set()
            await cancel_task(microphone)

    return scenario


@pytest.mark.parametrize("echo", [True, False])
def test_three_requests_are_retained_while_first_muse_turn_waits_and_run_in_order(duplex, echo):
    async def run(ctx):
        await ctx.question(.1)
        await ctx.entered[0].wait()
        await ctx.question(.2)
        await ctx.question(.3)
        assert ctx.labels() == [1]
        ctx.release[0].set()
        await ctx.entered[1].wait()
        assert ctx.labels() == [1, 2]
        ctx.release[1].set()
        await ctx.entered[2].wait()
        assert ctx.labels() == [1, 2, 3]
        assert ctx.maximum_active() == 1

    asyncio.run(bounded(duplex(run, echo=echo)))


def test_next_first_word_survives_previous_turn_completion(duplex):
    async def run(ctx):
        await ctx.question(.1)
        await ctx.entered[0].wait()
        await ctx.feed((.2, 3))
        ctx.release[0].set()
        await ctx.finished[0].wait()
        await ctx.feed((.3, 13), (0, 6))
        await ctx.entered[1].wait()
        assert ctx.labels() == [1, 2]
        assert ctx.records[1][ctx.records[1] != 0][:960].tolist() == [6554] * 960
        assert 9830 in ctx.records[1]

    asyncio.run(bounded(duplex(run)))


def test_two_endpoints_in_one_capture_chunk_are_both_retained(duplex):
    async def run(ctx):
        await ctx.feed((.1, 16), (0, 6), (.2, 16), (0, 6))
        await ctx.entered[0].wait()
        ctx.release[0].set()
        await ctx.entered[1].wait()
        assert ctx.labels() == [1, 2]
        assert ctx.maximum_active() == 1

    asyncio.run(bounded(duplex(run)))


@pytest.mark.parametrize("echo", [True, False, None, 1])
def test_actual_playback_accepts_input_only_with_exact_echo_capability(duplex, echo):
    np = pytest.importorskip("numpy")
    played = asyncio.Event()
    stop = asyncio.Event()

    class Speech:
        async def stream(self, text, rate):
            yield np.zeros(80, np.float32)
            played.set()
            await stop.wait()

    async def run(ctx):
        speaking = asyncio.create_task(ctx.conversation._speak(None, text="The first answer.", stream_segment=True))
        try:
            await played.wait()
            await ctx.question(.1)
            if echo is True:
                await ctx.entered[0].wait()
                assert ctx.labels() == [1]
            else:
                assert ctx.records == []
            stop.set()
            await speaking
            # Commit the read that may have begun during playback before speaking again.
            while ctx.conversation._input_blocked():
                await asyncio.sleep(.005)
            await ctx.feed((0, 1))
            await ctx.question(.2)
            if echo is True:
                ctx.release[0].set()
                await ctx.entered[1].wait()
                assert ctx.labels() == [1, 2]
            else:
                await ctx.entered[0].wait()
                assert ctx.labels() == [2]
        finally:
            stop.set()
            if not speaking.done():
                await cancel_task(speaking)
            else:
                await speaking

    asyncio.run(bounded(duplex(run, echo=echo, speech=Speech(), owned=True)))


def test_wake_privacy_and_continuous_followups_while_muse_waits(duplex):
    async def run(ctx):
        await ctx.question(.4)
        assert ctx.records == []
        await ctx.feed((.75, 7), (.1, 16), (0, 6))
        await ctx.entered[0].wait()
        await ctx.question(.2)
        ctx.release[0].set()
        await ctx.entered[1].wait()
        assert ctx.labels() == [1, 2]
        assert all(24576 not in pcm and 13107 not in pcm for pcm in ctx.records)

    asyncio.run(bounded(duplex(run, wake=True)))


def test_timed_wake_pre_roll_keeps_both_completed_questions_and_a_partial_next_one(duplex):
    async def run(ctx):
        await ctx.question(.4)
        await ctx.feed((.75, 7))
        await ctx.question(.1)
        await ctx.question(.2)
        assert ctx.records == []
        await ctx.feed((.3, 3))
        await ctx.entered[0].wait()
        await ctx.feed((.3, 13), (0, 6))
        ctx.release[0].set()
        await ctx.entered[1].wait()
        ctx.release[1].set()
        await ctx.entered[2].wait()
        assert ctx.labels() == [1, 2, 3]
        assert all(24576 not in pcm and 13107 not in pcm for pcm in ctx.records)

    asyncio.run(bounded(duplex(run, wake=True, wake_delay_feeds=3)))


def test_next_question_is_captured_while_first_question_is_still_transcribing(duplex):
    async def run(ctx):
        acknowledged = [asyncio.Event(), asyncio.Event()]

        async def acknowledge(session):
            index = len(session.setup_messages) - 1
            session.acknowledgement = {"ok": True, "status": 200,
                                       "response": {"message_id": f"user-{index + 1}"}}
            acknowledged[index].set()

        ctx.session.send_hook = acknowledge
        subscriber = asyncio.create_task(ctx.conversation._subscribe())
        try:
            await ctx.question(.1)
            await ctx.entered[0].wait()
            await ctx.question(.2)
            assert ctx.labels() == [1]
            assert ctx.session.setup_messages == []
            ctx.release[0].set()
            await acknowledged[0].wait()
            await ctx.session.events.put(chat_event("task.status", task_id="first", status="completed"))
            await ctx.session.delivered.get()
            await ctx.entered[1].wait()
            assert ctx.labels() == [1, 2]
            assert len(ctx.session.setup_messages) == 1
        finally:
            await cancel_task(subscriber)

    asyncio.run(bounded(duplex(run, echo=False, real_turn=True, hold_recognition=True, owned=True)))


def test_capture_gap_drops_partial_words_but_keeps_already_queued_requests(duplex):
    async def run(ctx):
        await ctx.question(.1)
        await ctx.entered[0].wait()
        await ctx.question(.2)
        await ctx.feed((.3, 3))
        await ctx.feed((.4, 16), (0, 6), gap=True)
        ctx.release[0].set()
        await ctx.entered[1].wait()
        ctx.release[1].set()
        await ctx.entered[2].wait()
        assert ctx.labels() == [1, 2, 4]
        assert 9830 not in ctx.records[2]

    asyncio.run(bounded(duplex(run)))


def test_cancellation_reaps_serial_turn_without_sending_queued_requests(duplex):
    async def run(ctx):
        await ctx.question(.1)
        await ctx.entered[0].wait()
        await ctx.question(.2)
        await cancel_task(ctx.microphone)
        assert ctx.finished[0].is_set()
        assert ctx.labels() == [1]
        assert not ctx.entered[1].is_set()
        assert not [task for task in asyncio.all_tasks() if task.get_name() == "reachy-microphone-capture"]

    asyncio.run(bounded(duplex(run)))


def test_default_voice_endpoint_groups_a_short_pause_and_waits_two_seconds(duplex):
    async def run(ctx):
        await ctx.feed((.1, 20))
        await ctx.feed((0, 50))
        await ctx.feed((0, 25))
        assert ctx.records == []
        await ctx.feed((.2, 20))
        await ctx.feed((0, 50))
        await ctx.feed((0, 49))
        assert ctx.records == []
        await ctx.feed((0, 1))
        await ctx.entered[0].wait()
        assert ctx.labels() == [1]
        assert 6554 in ctx.records[0]

    asyncio.run(bounded(duplex(run, silence_s=None)))


def test_full_queue_preserves_eight_questions_and_speaks_one_retry_notice(duplex):
    np = pytest.importorskip("numpy")
    notice = asyncio.Event()
    spoken = []
    overflow_text = "My question queue is full. Please repeat that after I finish."

    class Speech:
        async def stream(self, text, rate):
            spoken.append(text)
            if text == overflow_text:
                notice.set()
            yield np.zeros(80, np.float32)

    async def run(ctx):
        acknowledged = [asyncio.Event() for _ in range(9)]
        ctx.conversation.backends = replace(ctx.conversation.backends, reply_style=ReplyStyle.EXPRESSIVE_JSON)

        async def acknowledge(session):
            index = len(session.setup_messages) - 1
            session.acknowledgement = {"ok": True, "status": 200,
                                       "response": {"message_id": f"user-{index + 1}",
                                                    "session_id": "robot-chat", "is_thread": True}}
            acknowledged[index].set()

        ctx.session.send_hook = acknowledge
        subscriber = asyncio.create_task(ctx.conversation._subscribe())
        try:
            await ctx.question(.1)
            await acknowledged[0].wait()
            for marker in range(2, 11):
                await ctx.question(marker / 10)
            await notice.wait()
            await ctx.playback.get()
            assert ctx.labels() == list(range(1, 10))
            assert spoken == [overflow_text]
            for index in range(9):
                await acknowledged[index].wait()
                parent = f"user-{index + 1}"
                expression = "happy" if index % 2 == 0 else "curious"
                events = [chat_event("task.status", parent=parent, seq=index * 3 + 1,
                                     task_id=f"task-{index}", status="running"),
                          chat_event("delta.message_done", f"reply-{index}", parent=parent,
                                     seq=index * 3 + 2,
                                     display_text=sentence_frame(f"Answer {index + 1}.", expression)),
                          chat_event("task.status", parent=parent, seq=index * 3 + 3,
                                     task_id=f"task-{index}", status="completed")]
                for event in events:
                    await ctx.session.events.put(event)
                    await ctx.session.delivered.get()
            for _ in range(9):
                await ctx.playback.get()
            assert ctx.labels() == list(range(1, 10))
            assert len(ctx.session.setup_messages) == 9
            assert spoken == [overflow_text] + [f"Answer {i}." for i in range(1, 10)]
            assert ctx.hardware.play_expressions == [("thinking", None)] + [
                ("speaking", "happy" if i % 2 == 0 else "curious") for i in range(9)]
        finally:
            await cancel_task(subscriber)

    asyncio.run(bounded(duplex(run, real_turn=True, speech=Speech(), owned=True)))
