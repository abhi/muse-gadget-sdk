# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import io
import wave
from fractions import Fraction
from itertools import cycle

import pytest

np = pytest.importorskip("numpy")
av = pytest.importorskip("av")

from musegadget.voice_audio import Mp3Decoder, SpeechAudio, SpeechEnd, TurnRecorder


class ScriptVad:
    def __init__(self, voiced):
        self.voiced = iter(voiced)
        self.frames = []

    def is_speech(self, frame, sample_rate):
        assert sample_rate == 16_000
        assert len(frame) == 640
        self.frames.append(frame)
        return next(self.voiced, False)


def chunks(data, sizes):
    offset = 0
    for size in cycle(sizes):
        if offset >= len(data):
            break
        yield data[offset:offset + size]
        offset += size


def collect_turn(recorder, audio, sizes=(431, 1, 53, 2048, 177)):
    for chunk in chunks(audio, sizes):
        result = recorder.feed(chunk)
        if result is not None:
            return result
    return None


def read_wav(data):
    with wave.open(io.BytesIO(data), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 16_000
        assert wav.getcomptype() == "NONE"
        return np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")


def test_recorded_wav_preserves_preroll_speech_and_endpoint_silence():
    script = [False] * 20 + [True] * 25 + [False] * 40
    audio = np.repeat(np.asarray(script, np.float32) * 0.25, 320)
    recorder = TurnRecorder(16_000, vad=ScriptVad(script))

    recorded = read_wav(collect_turn(recorder, audio))

    expected = np.concatenate((np.zeros(7 * 320), np.full(25 * 320, 8192), np.zeros(35 * 320)))
    np.testing.assert_array_equal(recorded, expected)
    assert not recorder.active
    assert not recorder.last_truncated


def test_streaming_audio_arrives_before_endpoint_and_matches_the_whole_recording():
    script = [False] * 20 + [True] * 25 + [False] * 100
    audio = np.repeat(np.asarray(script, np.float32) * .25, 320)
    recorder = TurnRecorder(16000, vad=ScriptVad(script), silence_s=2, stream_audio=True)
    assert recorder.feed(audio[:45 * 320]) is None
    early = recorder.take_audio_events()
    assert early and all(isinstance(event, SpeechAudio) for event in early)
    assert recorder.take_audio_events() == []

    result = recorder.feed(audio[45 * 320:])
    late = recorder.take_audio_events()
    assert late[-1] == SpeechEnd(accepted=True)
    streamed = b"".join(event.pcm for event in early + late if isinstance(event, SpeechAudio))
    expected = np.concatenate((np.zeros(7 * 320), np.full(25 * 320, 8192), np.zeros(100 * 320)))
    np.testing.assert_array_equal(np.frombuffer(streamed, dtype="<i2"), expected)
    np.testing.assert_array_equal(read_wav(result), expected)


def test_short_rejected_speech_closes_stream_without_committing_a_turn():
    recorder = TurnRecorder(16000, vad=ScriptVad([True] * 3 + [False] * 5),
                            silence_s=.1, stream_audio=True)
    assert recorder.feed(np.full(8 * 320, .25, np.float32)) is None
    events = recorder.take_audio_events()
    assert isinstance(events[0], SpeechAudio)
    assert events[-1] == SpeechEnd(accepted=False)
    assert not recorder.active


def test_streaming_recorder_reset_discards_pending_audio_and_short_turns():
    recorder = TurnRecorder(16000, vad=ScriptVad([True] * 3), stream_audio=True)
    recorder.feed(np.full(3 * 320, .25, np.float32))
    recorder.reset()
    assert recorder.take_audio_events() == []
    assert not recorder.active


@pytest.mark.parametrize("input_rate", [16_000, 24_000, 44_100, 48_000])
def test_resampling_and_framing_are_identical_across_arbitrary_chunks(input_rate):
    script = [False] * 20 + [True] * 25 + [False] * 40
    t = np.arange(int(input_rate * 1.7), dtype=np.float64) / input_rate
    audio = (0.3 * np.sin(2 * np.pi * 427 * t)).astype(np.float32)
    once = TurnRecorder(input_rate, vad=ScriptVad(script))
    incremental = TurnRecorder(input_rate, vad=ScriptVad(script))

    expected = read_wav(collect_turn(once, audio, (len(audio),)))
    actual = read_wav(collect_turn(incremental, audio, (1, 11, 587, 960, 3, 1777)))

    np.testing.assert_array_equal(actual, expected)
    assert len(actual) == 67 * 320
    assert np.max(np.abs(actual.astype(np.int32))) < 10_000


def test_a_single_transient_does_not_start_a_turn():
    script = [False] * 10 + [True] + [False] * 50
    recorder = TurnRecorder(16_000, vad=ScriptVad(script))

    assert collect_turn(recorder, np.full(len(script) * 320, 0.1, np.float32)) is None
    assert not recorder.active


def test_minimum_duration_counts_speech_instead_of_preroll_or_silence():
    script = [False] * 10 + [True] * 3 + [False] * 50
    recorder = TurnRecorder(16_000, vad=ScriptVad(script), min_s=0.3)

    assert collect_turn(recorder, np.full(len(script) * 320, 0.1, np.float32)) is None
    assert not recorder.active


def test_short_pauses_do_not_end_a_turn():
    script = [True] * 10 + [False] * 10 + [True] * 10 + [False] * 35
    recorder = TurnRecorder(16_000, vad=ScriptVad(script), min_s=0.3)

    recorded = read_wav(collect_turn(recorder, np.full(len(script) * 320, 0.1, np.float32)))

    assert len(recorded) == 65 * 320
    assert not recorder.last_truncated


def test_decimal_durations_end_on_the_expected_twenty_millisecond_frame():
    recorder = TurnRecorder(
        16_000, vad=ScriptVad([True] * 14 + [False] * 7), min_s=0.28, silence_s=0.14
    )
    wav = recorder.feed(np.full(21 * 320, 0.1, np.float32))
    assert len(read_wav(wav)) == 21 * 320

    recorder = TurnRecorder(16_000, vad=ScriptVad([True] * 40), max_s=0.58)
    wav = recorder.feed(np.full(40 * 320, 0.1, np.float32))
    assert len(read_wav(wav)) == 29 * 320
    assert recorder.last_truncated


def test_active_requires_three_consecutive_voiced_frames_and_reset_discards_capture():
    recorder = TurnRecorder(16_000, vad=ScriptVad([True] * 20))
    frame = np.full(320, 0.25, np.float32)

    assert recorder.feed(frame) is None
    assert not recorder.active
    assert recorder.feed(frame) is None
    assert not recorder.active
    assert recorder.feed(frame) is None
    assert recorder.active

    recorder.reset()
    assert not recorder.active
    assert recorder.feed(frame[:319]) is None
    assert not recorder.active
    assert recorder.feed(frame[:1]) is None
    assert not recorder.active


def test_long_speech_stops_at_twenty_seconds_with_an_observable_cutoff():
    recorder = TurnRecorder(16_000, vad=ScriptVad([True] * 1100))
    audio = np.full(21 * 16_000, 0.25, np.float32)

    wav = collect_turn(recorder, audio)

    assert len(wav) == 640_044
    np.testing.assert_array_equal(read_wav(wav), np.full(320_000, 8192))
    assert recorder.last_truncated
    assert not recorder.active


def test_samples_remaining_after_a_turn_can_produce_the_next_turn():
    script = ([True] * 20 + [False] * 35) * 2
    recorder = TurnRecorder(16_000, vad=ScriptVad(script))
    audio = np.full(len(script) * 320, 0.1, np.float32)

    first = recorder.feed(audio)
    second = recorder.feed(np.empty(0, dtype=np.float32))

    assert len(read_wav(first)) == 55 * 320
    assert len(read_wav(second)) == 55 * 320


@pytest.mark.parametrize("kwargs", [
    {"input_rate": 0},
    {"input_rate": 16000.5},
    {"input_rate": 16000, "min_s": 21},
    {"input_rate": 16000, "max_s": 61},
    {"input_rate": 16000, "silence_s": float("nan")},
])
def test_invalid_recorder_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        TurnRecorder(vad=ScriptVad([]), **kwargs)


@pytest.mark.parametrize("samples", [
    np.zeros((2, 320), np.float32),
    np.array([float("nan")]),
    np.array([float("inf")]),
])
def test_invalid_microphone_samples_are_rejected(samples):
    recorder = TurnRecorder(16_000, vad=ScriptVad([]))
    with pytest.raises(ValueError, match="mono array"):
        recorder.feed(samples)


@pytest.fixture
def mp3():
    """A real stereo MP3 container, including its ID3 prefix."""
    output = io.BytesIO()
    with av.open(output, "w", format="mp3") as container:
        stream = container.add_stream("libmp3lame", rate=24_000)
        stream.layout = "stereo"
        t = np.arange(24_000, dtype=np.float64) / 24_000
        audio = np.stack((
            0.3 * np.sin(2 * np.pi * 440 * t),
            0.2 * np.sin(2 * np.pi * 880 * t),
        )).astype(np.float32)
        frame = av.AudioFrame.from_ndarray(audio, format="fltp", layout="stereo")
        frame.sample_rate = 24_000
        frame.time_base = Fraction(1, 24_000)
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    return output.getvalue()


def decode_chunks(data, sizes, rate):
    decoder = Mp3Decoder(rate)
    during_feed = []
    for chunk in chunks(data, sizes):
        during_feed.extend(decoder.feed(chunk))
    final = decoder.finish()
    assert decoder.finish() == []
    assert during_feed
    assert final
    all_chunks = during_feed + final
    assert all(array.ndim == 1 and array.dtype == np.float32 for array in all_chunks)
    return np.concatenate(all_chunks)


@pytest.mark.parametrize("output_rate", [16_000, 48_000])
def test_real_mp3_decodes_incrementally_without_lost_samples_or_chunk_boundaries(mp3, output_rate):
    complete = decode_chunks(mp3, (len(mp3),), output_rate)
    incremental = decode_chunks(mp3, (1, 2, 7, 17, 197, 53, 509), output_rate)

    np.testing.assert_array_equal(incremental, complete)
    assert output_rate <= len(incremental) <= int(output_rate * 1.15)
    assert np.max(np.abs(incremental)) > 0.1
    assert np.max(np.abs(np.diff(incremental))) < 0.15
    spectrum = np.abs(np.fft.rfft(incremental))
    frequencies = np.fft.rfftfreq(len(incremental), 1 / output_rate)
    peaks = frequencies[np.argsort(spectrum)[-8:]]
    assert np.min(np.abs(peaks - 440)) < 2
    assert np.min(np.abs(peaks - 880)) < 2


def test_empty_mp3_feed_does_not_flush_a_partial_packet(mp3):
    decoder = Mp3Decoder(48_000)
    output = decoder.feed(mp3[:173])
    assert decoder.feed(b"") == []
    output.extend(decoder.feed(mp3[173:]))
    output.extend(decoder.finish())

    reference = decode_chunks(mp3, (len(mp3),), 48_000)
    np.testing.assert_array_equal(np.concatenate(output), reference)
    with pytest.raises(ValueError, match="after finish"):
        decoder.feed(mp3)


def test_elementary_mp3_matches_independent_container_decoding():
    encoder = av.CodecContext.create("libmp3lame", "w")
    encoder.sample_rate = 24_000
    encoder.layout = "mono"
    encoder.format = "fltp"
    t = np.arange(24_000, dtype=np.float64) / 24_000
    audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)[None, :]
    frame = av.AudioFrame.from_ndarray(audio, format="fltp", layout="mono")
    frame.sample_rate = 24_000
    encoded = b"".join(bytes(packet) for packet in encoder.encode(frame) + encoder.encode(None))

    reference = []
    resampler = av.AudioResampler(format="fltp", layout="mono", rate=48_000)
    with av.open(io.BytesIO(encoded), format="mp3") as container:
        for decoded in container.decode(audio=0):
            reference.extend(output.to_ndarray().reshape(-1) for output in resampler.resample(decoded))
    reference.extend(output.to_ndarray().reshape(-1) for output in resampler.resample(None))

    actual = decode_chunks(encoded, (3, 13, 99, 1, 277), 48_000)

    assert len(actual) == 50_688
    np.testing.assert_array_equal(actual, np.concatenate(reference))


def test_incomplete_mp3_metadata_is_reported():
    decoder = Mp3Decoder(48_000)
    assert decoder.feed(b"ID3\x04\x00\x00\x00\x00\x00\x20") == []
    with pytest.raises(ValueError, match="Incomplete"):
        decoder.finish()


def test_empty_mp3_response_finishes_without_audio():
    decoder = Mp3Decoder(48_000)
    assert decoder.feed(b"") == []
    assert decoder.finish() == []
    assert decoder.finish() == []


def test_two_seconds_of_quiet_end_a_turn_without_splitting_a_shorter_pause():
    script = [True] * 20 + [False] * 75 + [True] * 20 + [False] * 100
    recorder = TurnRecorder(16000, vad=ScriptVad(script), silence_s=2)
    first = np.full(20 * 320, .25, np.float32)
    pause = np.zeros(75 * 320, np.float32)
    second = np.full(20 * 320, .5, np.float32)
    assert recorder.feed(np.concatenate((first, pause, second, np.zeros(99 * 320, np.float32)))) is None
    assert recorder.active
    wav = recorder.feed(np.zeros(320, np.float32))
    pcm = read_wav(wav)
    np.testing.assert_array_equal(pcm, np.concatenate((np.full(20 * 320, 8192),
                                                     np.zeros(75 * 320), np.full(20 * 320, 16384),
                                                     np.zeros(100 * 320))))
    assert not recorder.active and not recorder.last_truncated


def test_sixty_second_option_preserves_the_tail_inside_local_asr_packet_budget():
    recorder = TurnRecorder(16000, vad=ScriptVad([True] * 3000), max_s=60)
    audio = np.concatenate((np.full(59 * 16000, .25, np.float32),
                            np.full(16000, .5, np.float32)))
    wav = recorder.feed(audio)
    pcm = read_wav(wav)
    assert len(wav) == 1920044 and len(wav) < 2 * 1024 * 1024
    np.testing.assert_array_equal(pcm[-16000:], np.full(16000, 16384))
    assert recorder.last_truncated and not recorder.active


def test_speech_activity_stays_paused_through_recording_caps_until_two_seconds_of_quiet():
    recorder = TurnRecorder(16000, vad=ScriptVad([True] * 23 + [False] * 100),
                            silence_s=2, min_s=.06, max_s=.4)
    frame = np.full(320, .25, np.float32)
    for _ in range(2):
        assert recorder.feed(frame) is None
        assert not recorder.speech_active
    recorder.feed(frame)
    assert recorder.speech_active
    for _ in range(17):
        wav = recorder.feed(frame)
    assert wav is not None and recorder.last_truncated
    assert not recorder.active
    assert recorder.speech_active
    for _ in range(3):
        recorder.feed(frame)
        assert recorder.speech_active
    for _ in range(99):
        recorder.feed(np.zeros(320, np.float32))
        assert recorder.speech_active
    recorder.feed(np.zeros(320, np.float32))
    assert not recorder.speech_active


def test_speech_activity_ignores_transients_and_reset_requires_fresh_sustained_speech():
    recorder = TurnRecorder(16000, vad=ScriptVad([True] * 2 + [False] + [True] * 6),
                            silence_s=2)
    frame = np.full(320, .25, np.float32)
    for _ in range(3):
        recorder.feed(frame)
        assert not recorder.speech_active
    for _ in range(3):
        recorder.feed(frame)
    assert recorder.speech_active
    recorder.reset()
    assert not recorder.speech_active
    for _ in range(2):
        recorder.feed(frame)
        assert not recorder.speech_active
    recorder.feed(frame)
    assert recorder.speech_active
