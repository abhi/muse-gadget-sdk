"""Confirmed short invocations use the real recorder and conversation path."""

import asyncio
import io
import wave

import pytest

from musegadget import reachy_voice, voice_audio
from musegadget.reachy_capabilities import Mode
from musegadget.reachy_local_backends import backends_for
from musegadget.reachy_voice import VoiceConversation
from test_reachy_voice import FakeHardware, FakeSession, bounded, cancel_task


@pytest.fixture
def wake_scenario(monkeypatch):
    np = pytest.importorskip("numpy")
    pytest.importorskip("av")
    monkeypatch.setattr(reachy_voice, "ECHO_TAIL_S", 0)
    real_recorder = voice_audio.TurnRecorder

    class Vad:
        def is_speech(self, pcm, rate):
            return bool(np.any(np.frombuffer(pcm, dtype="<i2")))

    async def scenario(run, *, timeout=10, boundary=True, pause_recognition=False):
        processed = asyncio.Queue()
        incoming = asyncio.Queue()
        decision = asyncio.Event()
        recognition = []
        loop = asyncio.get_running_loop()

        class Recorder(real_recorder):
            def __init__(self, rate, **kwargs):
                super().__init__(rate, vad=Vad(), **kwargs)

            def feed(self, samples):
                result = super().feed(samples)
                if len(samples):
                    loop.call_soon_threadsafe(processed.put_nowait, result is not None)
                return result

        class Wake:
            phrase = "hey muse"

            def reset(self):
                pass

            def feed(self, samples):
                found = bool(len(samples) and np.any(samples == np.float32(.25)))
                if not found:
                    loop.call_soon_threadsafe(processed.put_nowait, False)
                return found

        class Recognition:
            def __init__(self):
                self.entered = asyncio.Event()
                self.proceed = asyncio.Event()
                if not pause_recognition:
                    self.proceed.set()

            async def transcribe(self, wav):
                self.entered.set()
                await self.proceed.wait()
                with wave.open(io.BytesIO(wav), "rb") as recording:
                    pcm = np.frombuffer(recording.readframes(recording.getnframes()), dtype="<i2")
                if np.max(pcm) > 12000:
                    text = "What is two plus two?"
                    if np.any(pcm == 8192):
                        text = "Hey Muse, " + text
                else:
                    text = "Hey Muse." if boundary else "Unconfirmed private speech."
                recognition.append(text)
                return text

        async def request(session):
            decision.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(voice_audio, "TurnRecorder", Recorder)
        session = FakeSession(send_hook=request)
        session.chat_subscribed.set()
        hardware = FakeHardware()
        conversation = VoiceConversation(session, hardware,
                                         backends=backends_for(Mode.ON_ROBOT, session, speech=object(),
                                                               transcriber=Recognition()),
                                         wake_detector=Wake(), silence_s=.14, wake_timeout_s=timeout)
        conversation.backends.voice.phrases["Yes?"] = (np.full(80, .1, np.float32),)
        actual_turn = conversation.turn

        async def turn(wav, **options):
            result = await actual_turn(wav, **options)
            decision.set()
            return result

        conversation.turn = turn

        async def capture(buffer):
            while True:
                sample, gap = await incoming.get()
                buffer.push(sample, discard=conversation._muted, gap=gap)

        conversation._capture_microphone = capture
        microphone = asyncio.create_task(conversation._microphone())
        try:
            while not hardware.states:
                await asyncio.sleep(.001)

            async def feed(marker, voiced_frames, *, gap=False, end=True, wake_frames=0):
                decision.clear()
                audio = np.concatenate((np.full(wake_frames * 320, .25, np.float32),
                                        np.full(voiced_frames * 320, marker, np.float32),
                                        np.zeros((7 if end else 0) * 320, np.float32)))
                await incoming.put((audio, gap))
                completed = await processed.get()
                if completed:
                    old_deadline = conversation._wake_deadline
                    await decision.wait()
                    if not session.setup_messages:
                        # A turn can finish before its independently owned cue plays.
                        await conversation._output_queue.join()
                        while (conversation._input_blocked()
                               or conversation._wake_deadline == old_deadline):
                            await asyncio.sleep(.001)
                    while not processed.empty():
                        processed.get_nowait()
                return completed

            async def capture_gap():
                await incoming.put((np.zeros(320, np.float32), True))

            feed.capture_gap = capture_gap

            await run(conversation, session, hardware, recognition, feed)
        finally:
            await cancel_task(microphone)

    return scenario


def test_confirmed_short_wake_keeps_the_next_question(wake_scenario):
    async def run(conversation, session, hardware, recognition, feed):
        await feed(.25, 14)
        await feed(.5, 20)

        sent = [message.split("The user's spoken request is: ")[-1]
                for message, _ in session.setup_messages]
        assert sent == ["What is two plus two?"]
        assert recognition == ["Hey Muse.", "What is two plus two?"]

    asyncio.run(bounded(wake_scenario(run)))


def test_short_followup_still_requires_ordinary_sustained_speech(wake_scenario):
    async def run(conversation, session, hardware, recognition, feed):
        await feed(.25, 14)
        assert not await feed(.5, 14)
        assert recognition == ["Hey Muse."]
        assert session.setup_messages == []
        await feed(.5, 20)
        assert session.setup_messages[0][0].endswith("The user's spoken request is: What is two plus two?")

    asyncio.run(bounded(wake_scenario(run)))


def test_first_wake_capture_gap_keeps_fresh_question_authorized(wake_scenario):
    async def run(conversation, session, hardware, recognition, feed):
        assert not await feed(.25, 3, end=False)
        assert await feed(.5, 20, gap=True)
        assert recognition == ["What is two plus two?"]
        assert conversation._wake_deadline is not None
        assert session.setup_messages[0][0].endswith("The user's spoken request is: What is two plus two?")

    asyncio.run(bounded(wake_scenario(run)))


def test_followup_gap_preserves_authorization_and_normal_minimum(wake_scenario):
    async def run(conversation, session, hardware, recognition, feed):
        await feed(.25, 14)
        assert not await feed(.5, 14, gap=True)
        assert recognition == ["Hey Muse."]
        await feed(.5, 20)
        assert session.setup_messages[0][0].endswith("The user's spoken request is: What is two plus two?")

    asyncio.run(bounded(wake_scenario(run)))


def test_expired_pending_wake_requires_a_fresh_invocation(wake_scenario):
    async def run(conversation, session, hardware, recognition, feed):
        assert not await feed(.25, 2, end=False)
        while hardware.states[-1] != "idle":
            await asyncio.sleep(.001)
        assert not await feed(.5, 20)
        assert recognition == [] and session.setup_messages == []
        await feed(.5, 20, wake_frames=14)
        assert session.setup_messages[0][0].endswith("The user's spoken request is: What is two plus two?")
        assert recognition == ["Hey Muse, What is two plus two?"]

    asyncio.run(bounded(wake_scenario(run, timeout=.03)))


def test_short_confirmed_audio_does_not_bypass_the_asr_boundary(wake_scenario):
    async def run(conversation, session, hardware, recognition, feed):
        await feed(.25, 14)
        assert recognition == ["Unconfirmed private speech."]
        assert session.setup_messages == []
        assert len(hardware.played) == 1

    asyncio.run(bounded(wake_scenario(run, boundary=False)))


def test_same_breath_request_is_not_truncated_by_short_wake_capture(wake_scenario):
    async def run(conversation, session, hardware, recognition, feed):
        await feed(.5, 20, wake_frames=14)
        assert recognition == ["Hey Muse, What is two plus two?"]
        assert session.setup_messages[0][0].endswith("The user's spoken request is: What is two plus two?")

    asyncio.run(bounded(wake_scenario(run)))


def test_capture_gap_during_asr_preserves_the_inflight_boundary(wake_scenario, caplog):
    async def run(conversation, session, hardware, recognition, feed):
        first_wake = asyncio.create_task(feed(.25, 14))
        try:
            await conversation.backends.hearing.transcriber.entered.wait()
            await feed.capture_gap()
            while "lost continuity" not in caplog.text:
                await asyncio.sleep(.001)
            conversation.backends.hearing.transcriber.proceed.set()
            await first_wake
            assert recognition == ["Unconfirmed private speech."]
            assert session.setup_messages == []
            assert len(hardware.played) == 1
        finally:
            first_wake.cancel()
            await asyncio.gather(first_wake, return_exceptions=True)

    asyncio.run(bounded(wake_scenario(run, boundary=False, pause_recognition=True)))
