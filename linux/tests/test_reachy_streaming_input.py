"""Streaming ASR receives authorized speech before the user finishes talking."""

import asyncio

import pytest

from test_reachy_duplex_voice import duplex
from test_reachy_duplex import OrderedSpeech
from test_reachy_voice import bounded
from musegadget.streaming_transcription import (
    StreamingBufferOverflow, StreamingTranscriptionError, TranscriptRevision,
)


class StreamingRecognizer:
    def __init__(self):
        self.turns = []

    async def start(self):
        pass

    def open_turn(self):
        turn = Stream()
        self.turns.append(turn)
        return turn

    async def transcribe(self, wav):
        pytest.fail("Streaming capture must not start a whole-WAV decoder at endpoint")


class Stream:
    partial = TranscriptRevision(0, "")

    def __init__(self):
        self.audio = bytearray()
        self.ended = False
        self.aborted = False
        self.result = asyncio.get_running_loop().create_future()

    def feed(self, pcm):
        assert not self.ended and not self.aborted
        self.audio.extend(pcm)

    def finish(self):
        self.ended = True
        return self.result

    def abort(self):
        self.aborted = True
        self.result.cancel()


def test_audio_is_transcribed_during_speech_and_endpoint_sends_only_final_text(duplex):
    recognizer = StreamingRecognizer()

    async def run(ctx):
        await ctx.feed((.2, 25))
        stream, = recognizer.turns
        assert len(stream.audio) == 25 * 320 * 2
        assert not stream.ended
        assert ctx.session.setup_messages == []
        await ctx.feed((0, 50))
        await ctx.feed((0, 49))
        assert not stream.ended
        await ctx.feed((0, 1))
        assert stream.ended
        assert ctx.session.setup_messages == []
        stream.result.set_result("Please keep the last three words.")
        while not ctx.session.setup_messages:
            await asyncio.sleep(0)
        assert ctx.session.setup_messages == [("Please keep the last three words.", "robot-chat")]
        assert ctx.session.sent == []

        # A second utterance is decoded while Muse still owns the first turn.
        await ctx.feed((.3, 25))
        second = recognizer.turns[1]
        assert second.audio and not second.ended
        assert len(ctx.session.setup_messages) == 1

    asyncio.run(bounded(duplex(run, real_turn=True, transcriber=recognizer, silence_s=2, owned=True)))
    assert recognizer.turns[1].aborted


def test_sleeping_audio_never_enters_stream_and_timed_wake_keeps_first_words(duplex):
    np = pytest.importorskip("numpy")
    recognizer = StreamingRecognizer()

    async def run(ctx):
        await ctx.feed((.1, 25))
        assert recognizer.turns == []
        await ctx.feed((.75, 3), (.2, 16))
        stream, = recognizer.turns
        pcm = np.frombuffer(stream.audio, dtype="<i2")
        assert len(pcm) == 16 * 320
        assert set(pcm) == {6554}
        assert not stream.ended

    asyncio.run(bounded(duplex(run, wake=True, real_turn=True, transcriber=recognizer)))
    assert recognizer.turns[0].aborted


def test_capture_gap_aborts_old_recognition_and_new_audio_starts_a_fresh_turn(duplex):
    recognizer = StreamingRecognizer()

    async def run(ctx):
        await ctx.feed((.2, 16))
        first, = recognizer.turns
        await ctx.feed((.3, 16), gap=True)
        assert first.aborted
        second = recognizer.turns[1]
        assert second.audio and not second.ended
        await ctx.feed((0, 5))
        assert second.ended
        second.result.set_result("Only the complete new request.")
        while not ctx.session.setup_messages:
            await asyncio.sleep(0)
        assert ctx.session.setup_messages == [("Only the complete new request.", "robot-chat")]

    asyncio.run(bounded(duplex(run, real_turn=True, transcriber=recognizer, owned=True)))


def test_vad_rejected_short_sound_never_commits_its_partial_transcript(duplex):
    recognizer = StreamingRecognizer()

    async def run(ctx):
        await ctx.feed((.2, 3), (0, 5))
        stream, = recognizer.turns
        assert stream.aborted and not stream.ended
        assert ctx.session.setup_messages == []

    asyncio.run(bounded(duplex(run, real_turn=True, transcriber=recognizer, owned=True)))


def test_recognition_overflow_drops_the_whole_turn_and_keeps_listening(duplex):
    recognizer = StreamingRecognizer()
    speech = OrderedSpeech()
    original_open = recognizer.open_turn

    def open_turn():
        turn = original_open()
        if len(recognizer.turns) == 1:
            def overflow(pcm):
                raise StreamingBufferOverflow("test inference backlog")
            turn.feed = overflow
        return turn

    recognizer.open_turn = open_turn

    async def run(ctx):
        await ctx.feed((.2, 16))
        await ctx.feed((.2, 16))
        assert len(recognizer.turns) == 1
        assert recognizer.turns[0].aborted
        await ctx.feed((0, 5))
        assert ctx.session.setup_messages == []
        await ctx.feed((.3, 16), (0, 5))
        second = recognizer.turns[1]
        assert second.ended
        second.result.set_result("A complete replacement request.")
        while not ctx.session.setup_messages:
            await asyncio.sleep(0)
        assert ctx.session.setup_messages == [("A complete replacement request.", "robot-chat")]
        assert not ctx.microphone.done()

    asyncio.run(bounded(duplex(run, real_turn=True, transcriber=recognizer, speech=speech, owned=True)))


def test_failed_finalization_does_not_cancel_a_later_complete_utterance(duplex):
    recognizer = StreamingRecognizer()

    async def run(ctx):
        await ctx.feed((.2, 16), (0, 5))
        await ctx.feed((.3, 16), (0, 5))
        first, second = recognizer.turns
        first.result.set_exception(StreamingTranscriptionError("test failed native line"))
        second.result.set_result("Keep this second request.")
        while not ctx.session.setup_messages:
            await asyncio.sleep(0)
        assert ctx.session.setup_messages == [("Keep this second request.", "robot-chat")]
        assert not ctx.microphone.done()

    asyncio.run(bounded(duplex(run, real_turn=True, transcriber=recognizer, speech=OrderedSpeech(), owned=True)))
