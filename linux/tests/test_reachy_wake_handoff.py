"""Trusted acoustic wake boundaries hand off only request audio to ASR."""

import asyncio
from dataclasses import dataclass
import io
import wave

import pytest

from musegadget import reachy_voice, voice_audio
from musegadget.reachy_capabilities import Mode
from musegadget.reachy_local_backends import backends_for
from musegadget.reachy_voice import VoiceConversation
from test_reachy_voice import FakeHardware, FakeSession, bounded, cancel_task


@dataclass(frozen=True)
class Detection:
    post_wake_samples: int
    epoch: int
    feed_id: int


@pytest.fixture
def handoff(monkeypatch):
    np = pytest.importorskip("numpy")
    pytest.importorskip("av")
    monkeypatch.setattr(reachy_voice, "ECHO_TAIL_S", 0)
    real_recorder = voice_audio.TurnRecorder

    class Vad:
        def is_speech(self, pcm, rate):
            samples = np.frombuffer(pcm, dtype="<i2")
            return bool(np.any((samples != 0) & (samples != 2048)))

    class Recorder(real_recorder):
        def __init__(self, rate, **kwargs):
            super().__init__(rate, vad=Vad(), **kwargs)

    class Wake:
        phrase = "hey muse"
        sample_rate = 16000

        def __init__(self, *, trusted=True, invalid=None, delayed=False):
            self.trusted = trusted
            self.invalid = invalid
            self.delayed = delayed
            self.epoch = 0
            self.feed_id = 0
            self.last_detection = None
            self.samples = 0
            self.keyword_end = None

        def reset(self):
            self.epoch += 1
            self.last_detection = None
            self.samples = 0
            self.keyword_end = None

        def feed(self, samples):
            self.feed_id += 1
            self.last_detection = None
            previous_samples = self.samples
            self.samples += len(samples)
            positions = np.flatnonzero(samples == np.float32(.25))
            if len(positions):
                self.keyword_end = previous_samples + int(positions[-1]) + 1
                if self.delayed:
                    return False
            elif self.keyword_end is None:
                return False
            if self.trusted:
                tail = self.samples - self.keyword_end
                if self.invalid == "boolean_tail":
                    tail = True
                elif self.invalid == "negative_tail":
                    tail = -1
                elif self.invalid == "oversized_tail":
                    tail = 48001
                self.last_detection = Detection(
                    post_wake_samples=tail,
                    epoch=self.epoch - 1 if self.invalid == "old_epoch" else self.epoch,
                    feed_id=self.feed_id - 1 if self.invalid == "old_feed" else self.feed_id,
                )
            return True

    async def scenario(run, *, trusted=True, invalid=None, recognized=None, pause_asr=False, delayed=False):
        incoming = asyncio.Queue()
        accepted = asyncio.Event()
        consumer_ready = asyncio.Event()
        recordings = []

        class Recognition:
            def __init__(self):
                self.entered = asyncio.Event()
                self.proceed = asyncio.Event()
                if not pause_asr:
                    self.proceed.set()

            async def transcribe(self, wav):
                with wave.open(io.BytesIO(wav), "rb") as recording:
                    pcm = np.frombuffer(recording.readframes(recording.getnframes()), dtype="<i2").copy()
                recordings.append(pcm)
                self.entered.set()
                await self.proceed.wait()
                return recognized if recognized is not None else "Find flights from Boston to Paris."

        async def request(session):
            accepted.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(voice_audio, "TurnRecorder", Recorder)
        session = FakeSession(send_hook=request)
        session.chat_subscribed.set()
        hardware = FakeHardware()
        wake = Wake(trusted=trusted, invalid=invalid, delayed=delayed)
        recognition = Recognition()
        conversation = VoiceConversation(session, hardware,
                                         backends=backends_for(Mode.ON_ROBOT, session, speech=object(),
                                                               transcriber=recognition),
                                         wake_detector=wake, silence_s=.14)
        conversation.backends.voice.phrases["Yes?"] = (np.full(80, .1, np.float32),)

        async def capture(buffer):
            original_get = buffer.get
            acknowledgements = {}
            previous = None

            async def get():
                nonlocal previous
                consumer_ready.set()
                if previous is not None and not previous.done():
                    previous.set_result(None)
                previous = None
                chunk = await original_get()
                if chunk is not None:
                    previous = acknowledgements.pop(id(chunk.samples), None)
                return chunk

            buffer.get = get
            while True:
                sample, gap, processed = await incoming.get()
                acknowledgements[id(sample)] = processed
                buffer.push(sample, discard=conversation._muted, gap=gap)

        conversation._capture_microphone = capture
        microphone = asyncio.create_task(conversation._microphone())
        try:
            await consumer_ready.wait()

            async def feed(*parts, gap=False):
                samples = np.concatenate([np.full(frames * 320, marker, np.float32)
                                          for marker, frames in parts])
                processed = asyncio.get_running_loop().create_future()
                await incoming.put((samples, gap, processed))
                await processed

            await run(conversation, session, hardware, wake, recognition, recordings, accepted, feed)
        finally:
            recognition.proceed.set()
            await cancel_task(microphone)

    return scenario


def test_trusted_wake_and_quiet_movement_noise_do_not_start_wake_asr(handoff):
    async def run(conversation, session, hardware, wake, recognition, recordings, accepted, feed):
        await feed((.25, 14), (0, 7))
        await feed((.0625, 20), (0, 7))
        assert recordings == [] and not recognition.entered.is_set()
        assert hardware.played == [] and not conversation._muted
        await feed((.5, 20), (0, 7))
        await accepted.wait()
        assert session.setup_messages[0][0].endswith("The user's spoken request is: Find flights from Boston to Paris.")
        assert len(recordings) == 1
        assert 16384 in recordings[0] and 8192 not in recordings[0]

    asyncio.run(bounded(handoff(run)))


def test_same_breath_request_keeps_first_word_and_removes_all_prewake_pcm(handoff):
    async def run(conversation, session, hardware, wake, recognition, recordings, accepted, feed):
        await feed((.125, 8), (.25, 14), (.625, 3), (.5, 17), (0, 7))
        await accepted.wait()
        assert len(recordings) == 1
        assert recordings[0][:960].tolist() == [20480] * 960
        assert 16384 in recordings[0]
        assert 4096 not in recordings[0] and 8192 not in recordings[0]
        assert session.setup_messages[0][0].endswith("The user's spoken request is: Find flights from Boston to Paris.")
        assert hardware.played == []

    asyncio.run(bounded(handoff(run)))


def test_wake_only_never_blocks_the_first_question_on_delayed_asr(handoff):
    async def run(conversation, session, hardware, wake, recognition, recordings, accepted, feed):
        await feed((.25, 14), (0, 7))
        assert not recognition.entered.is_set() and recordings == []
        await feed((.625, 3), (.5, 17), (0, 7))
        await recognition.entered.wait()
        assert len(recordings) == 1
        first_word = recordings[0][recordings[0] != 0][:960]
        assert first_word.tolist() == [20480] * 960
        assert 8192 not in recordings[0] and hardware.played == []
        recognition.proceed.set()
        await accepted.wait()
        assert session.setup_messages[0][0].endswith("The user's spoken request is: Find flights from Boston to Paris.")

    asyncio.run(bounded(handoff(run, pause_asr=True)))


def test_delayed_keyword_hit_keeps_question_start_from_an_earlier_capture_chunk(handoff):
    async def run(conversation, session, hardware, wake, recognition, recordings, accepted, feed):
        await feed((.125, 8), (.25, 14), (.625, 3))
        assert recordings == [] and session.setup_messages == []
        await feed((.5, 17), (0, 7))
        await accepted.wait()
        assert len(recordings) == 1
        assert recordings[0][:960].tolist() == [20480] * 960
        assert 4096 not in recordings[0] and 8192 not in recordings[0]
        assert session.setup_messages[0][0].endswith("The user's spoken request is: Find flights from Boston to Paris.")

    asyncio.run(bounded(handoff(run, delayed=True)))


def test_detectors_without_timing_keep_exact_asr_privacy_guard(handoff):
    async def run(conversation, session, hardware, wake, recognition, recordings, accepted, feed):
        await feed((.125, 8), (.25, 14), (.5, 20), (0, 7))
        await recognition.entered.wait()
        while not hardware.played:
            await asyncio.sleep(0)
        while conversation._muted:
            await asyncio.sleep(0)
        assert session.setup_messages == [] and not accepted.is_set()
        assert len(hardware.played) == 1

    asyncio.run(bounded(handoff(run, trusted=False,
                               recognized="Private speech without the confirmed wake boundary.")))


@pytest.mark.parametrize("invalid", ["old_epoch", "old_feed", "boolean_tail", "negative_tail", "oversized_tail"])
def test_invalid_or_stale_boundary_cannot_bypass_exact_asr_guard(handoff, invalid):
    async def run(conversation, session, hardware, wake, recognition, recordings, accepted, feed):
        await feed((.125, 8), (.25, 14), (.5, 20), (0, 7))
        await recognition.entered.wait()
        while conversation._muted:
            await asyncio.sleep(0)
        while not hardware.played:
            await asyncio.sleep(0)
        assert len(recordings) == 1 and 4096 in recordings[0] and 8192 in recordings[0]
        assert session.setup_messages == [] and not accepted.is_set()
        assert len(hardware.played) == 1

    asyncio.run(bounded(handoff(run, invalid=invalid,
                               recognized="Private speech without the confirmed wake boundary.")))


@pytest.mark.parametrize("trusted", [True, False])
def test_capture_gap_after_wake_keeps_listening_and_discards_incomplete_audio(handoff, trusted):
    async def run(conversation, session, hardware, wake, recognition, recordings, accepted, feed):
        await feed((.125, 8), (.25, 14), (.375, 5))
        assert recordings == []
        deadline = conversation._wake_deadline
        await feed((.625, 3), (.5, 17), (0, 7), gap=True)
        await accepted.wait()
        assert len(recordings) == 1 and recordings[0][:960].tolist() == [20480] * 960
        assert all(marker not in recordings[0] for marker in (4096, 8192, 12288))
        assert conversation._wake_deadline == deadline
        assert hardware.states.count("idle") == 1
        assert session.setup_messages[0][0].endswith("The user's spoken request is: Find flights from Boston to Paris.")

    asyncio.run(bounded(handoff(run, trusted=trusted)))


def test_capture_gap_while_asleep_never_authorizes_question_audio(handoff):
    async def run(conversation, session, hardware, wake, recognition, recordings, accepted, feed):
        await feed((.125, 8), (0, 7))
        await feed((.625, 3), (.5, 17), (0, 7), gap=True)
        assert conversation._wake_deadline is None
        assert recordings == [] and session.setup_messages == []
        assert not accepted.is_set() and hardware.played == []

    asyncio.run(bounded(handoff(run)))


def test_next_authorized_utterance_is_captured_separately_while_first_asr_waits(handoff):
    async def run(conversation, session, hardware, wake, recognition, recordings, accepted, feed):
        await feed((.25, 14), (0, 7))
        await feed((.5, 20), (0, 7))
        await recognition.entered.wait()
        await feed((.75, 20), (0, 7))
        recognition.proceed.set()
        await accepted.wait()
        while len(recordings) < 2:
            await asyncio.sleep(0)
        assert 24576 not in recordings[0]
        assert 8192 not in recordings[0]
        assert 24576 in recordings[1] and 8192 not in recordings[1]
        assert hardware.played == []
        # Muse remains serialized: the second recognized request waits for the
        # first backend owner even though its audio and ASR are already safe.
        assert len(session.setup_messages) == 1
        assert session.setup_messages[0][0].endswith("The user's spoken request is: Find flights from Boston to Paris.")

    asyncio.run(bounded(handoff(run, pause_asr=True)))
