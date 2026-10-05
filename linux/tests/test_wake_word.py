# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

from __future__ import annotations

import importlib
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")

from musegadget.wake_word import MODEL_FILES, PHONETIC_MODEL_FILES, WakeDetection, WakeWordDetector, _encoder_timing


def protobuf_string(field, value):
    data = value.encode() if isinstance(value, str) else value

    def integer(number):
        result = bytearray()
        while number >= 128:
            result.append((number & 127) | 128)
            number >>= 7
        result.append(number)
        return bytes(result)

    return integer(field * 8 + 2) + integer(len(data)) + data


def encoder_metadata(**values):
    return protobuf_string(7, b'graph skipped') + b''.join(
        protobuf_string(14, protobuf_string(1, key) + protobuf_string(2, str(value)))
        for key, value in values.items())


class FakeStream:
    def __init__(self):
        self.queue = []
        self.history = []
        self.result = ""
        self.accepted = []

    def accept_waveform(self, rate, audio):
        assert rate == 16000 and audio.dtype == np.float32 and audio.ndim == 1
        assert audio.flags.c_contiguous
        self.accepted.append(audio.copy())
        self.queue.extend(audio.tolist())


class FakeSpotter:
    def __init__(self, **config):
        self.config = config
        self.keyword = Path(config["keywords_file"]).read_text()
        self.label = self.keyword.split("@")[-1].strip()
        self.streams = []
        self.force_result = None

    def create_stream(self):
        stream = FakeStream()
        self.streams.append(stream)
        return stream

    def is_ready(self, stream):
        return len(stream.queue) >= 4

    def decode_stream(self, stream):
        stream.history.extend(stream.queue[:4])
        del stream.queue[:4]
        pattern = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32).tolist()
        if stream.history[-4:] == pattern:
            stream.result = self.label
        elif self.force_result is not None:
            stream.result = self.force_result

    def get_result(self, stream):
        return stream.result


class TimedSpotter(FakeSpotter):
    """A consumptive structured binding with the real model's decode stride."""

    def __init__(self, **config):
        super().__init__(**config)
        self.keyword_spotter = SimpleNamespace(get_result=self.structured_result)
        self.invalid_timestamps = None
        self.result_tokens = [' HE', 'Y', ' MU', 'SE']
        self.snapshot_calls = 0

    def create_stream(self):
        stream = super().create_stream()
        stream.raw_result = None
        stream.last_marker = -1
        stream.blank_steps = 0
        stream.clock_origin = 0
        return stream

    def is_ready(self, stream):
        return len(stream.queue) >= 7200

    def decode_stream(self, stream):
        if stream.blank_steps >= 5:
            stream.clock_origin = len(stream.history)
            stream.blank_steps = 0
        stream.history.extend(stream.queue[:5120])
        del stream.queue[:5120]
        pattern = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32).tolist()
        index = next((index for index in range(max(0, stream.last_marker + 1), len(stream.history) - 3)
                      if stream.history[index:index + 4] == pattern), None)
        if index is None:
            stream.blank_steps += 1
            return
        stream.last_marker = index
        stream.blank_steps = 0
        timestamps = [(index + number - stream.clock_origin) / 16000 for number in range(4)]
        if self.invalid_timestamps is not None:
            timestamps = self.invalid_timestamps
        stream.raw_result = SimpleNamespace(keyword=self.label, tokens=self.result_tokens,
                                            timestamps=timestamps)

    def structured_result(self, stream):
        self.snapshot_calls += 1
        result, stream.raw_result = stream.raw_result, None
        return result or SimpleNamespace(keyword='', tokens=[], timestamps=[])

    def get_result(self, stream):
        raise AssertionError('The string-only getter would consume the structured timing result')


@pytest.fixture
def models(tmp_path, monkeypatch):
    for filename in MODEL_FILES.values():
        (tmp_path / filename).write_bytes(b"model")
    (tmp_path / "tokens.txt").write_text("<unk> 0\n▁HE 1\nY 2\n▁MU 3\nSE 4\n")
    instances = []

    def spotter(**config):
        instance = FakeSpotter(**config)
        instances.append(instance)
        return instance

    class Tokenizer:
        def __init__(self, *, model_file):
            assert Path(model_file).name == "bpe.model"

        def encode(self, text, *, out_type):
            assert text == "HEY MUSE" and out_type is str
            return ["▁HE", "Y", "▁MU", "SE"]

    monkeypatch.setitem(sys.modules, "sherpa_onnx", SimpleNamespace(KeywordSpotter=spotter))
    monkeypatch.setitem(sys.modules, "sentencepiece", SimpleNamespace(SentencePieceProcessor=Tokenizer))
    return tmp_path, instances


@pytest.fixture
def timed_models(models, monkeypatch):
    directory, instances = models
    (directory / MODEL_FILES['encoder']).write_bytes(encoder_metadata(
        model_type='zipformer2', version=1, T=45, decode_chunk_len=32))

    def create(**config):
        spotter = TimedSpotter(**config)
        instances.append(spotter)
        return spotter

    monkeypatch.setitem(sys.modules, 'sherpa_onnx', SimpleNamespace(KeywordSpotter=create))
    return directory, instances


def test_keyword_tail_includes_unaccepted_large_caller_remainder(timed_models):
    directory, instances = timed_models
    detector = WakeWordDetector(directory)
    audio = np.zeros(12000, dtype=np.float32)
    audio[4:8] = [.1, .2, .3, .4]
    assert detector.feed(audio)
    assert detector.last_detection == WakeDetection(11993, detector.epoch, detector.feed_id)
    # The primary spotter confirmed after8000samples;4000caller samples were
    # never accepted by it, yet all of those request samples remain in the tail.
    assert sum(map(len, instances[0].streams[0].accepted)) == 8000


@pytest.mark.parametrize('tokens', [
    ['HE', 'Y', 'MU', 'SE'], [' HE', 'Y', ' MU', 'S'],
    [' HE', 'Y', ' MU', 'SE', 'extra'],
])
def test_confirmed_label_with_different_tokens_cannot_authorize_a_cut(timed_models, tokens):
    directory, instances = timed_models
    detector = WakeWordDetector(directory)
    instances[0].result_tokens = tokens
    audio = np.zeros(8000, np.float32)
    audio[:4] = [.1, .2, .3, .4]
    assert detector.feed(audio)
    assert detector.last_detection is None


@pytest.mark.parametrize('step_s, timed', [(.174, True), (.176, False)])
def test_alignment_accepts_measured_replay_cost_but_keeps_a_bounded_deadline(timed_models, monkeypatch, step_s, timed):
    import musegadget.wake_word as module

    directory, instances = timed_models
    detector = WakeWordDetector(directory)
    spotter = instances[0]
    primary = detector._stream
    clock = SimpleNamespace(now=0.)
    monkeypatch.setattr(module.time, 'monotonic', lambda: clock.now)
    original = spotter.decode_stream

    def decode(stream):
        original(stream)
        if stream is not primary:
            clock.now += step_s

    spotter.decode_stream = decode
    audio = np.zeros(24000, np.float32)
    audio[20000:20004] = [.1, .2, .3, .4]
    assert detector.feed(audio)
    assert (detector.last_detection is not None) is timed


def test_internal_silence_reset_does_not_shift_verified_cut_into_private_audio(timed_models):
    directory, instances = timed_models
    detector = WakeWordDetector(directory)
    assert not detector.feed(np.zeros(64000, dtype=np.float32))
    assert instances[0].streams[0].clock_origin > 0
    request = np.zeros(8000, dtype=np.float32)
    request[:4] = [.1, .2, .3, .4]
    assert detector.feed(request)
    assert detector.last_detection.post_wake_samples == 7997
    assert detector.last_detection.post_wake_samples < len(request)


def test_detected_event_matches_current_epoch_and_feed_and_is_invalidated(timed_models):
    directory, _ = timed_models
    detector = WakeWordDetector(directory)
    audio = np.zeros(8000, dtype=np.float32)
    audio[:4] = [.1, .2, .3, .4]
    assert detector.feed(audio)
    event = detector.last_detection
    assert (event.epoch, event.feed_id) == (detector.epoch, detector.feed_id)
    assert not detector.feed(np.zeros(512, dtype=np.float32))
    assert detector.last_detection is None and detector.feed_id > event.feed_id
    assert detector.feed(audio)
    epoch, feed_id = detector.epoch, detector.feed_id
    detector.reset()
    assert detector.epoch == epoch + 1 and detector.feed_id == feed_id
    assert detector.last_detection is None
    assert detector.feed(audio)
    assert detector.last_detection.epoch == epoch + 1


def test_invalid_feed_clears_previously_valid_event(timed_models):
    directory, _ = timed_models
    detector = WakeWordDetector(directory)
    audio = np.zeros(8000, dtype=np.float32)
    audio[:4] = [.1, .2, .3, .4]
    assert detector.feed(audio)
    with pytest.raises(ValueError):
        detector.feed(np.array([float('nan')], dtype=np.float32))
    assert detector.last_detection is None


@pytest.mark.parametrize('timestamps', [
    [], [0, .01], [0, 0, 0, 0], [0, .1, .2, float('nan')], [0, .1, .2, float('inf')],
    [0, .1, .2, True], [-.1, 0, .1, .2], [0, .2, .1, .3], [0, .1, .2, 9],
])
def test_malformed_timing_never_authorizes_trimming(timed_models, timestamps):
    directory, instances = timed_models
    detector = WakeWordDetector(directory)
    instances[0].invalid_timestamps = timestamps
    audio = np.zeros(8000, dtype=np.float32)
    audio[:4] = [.1, .2, .3, .4]
    assert detector.feed(audio)
    assert detector.last_detection is None


def test_alignment_and_audio_history_have_explicit_bounds(timed_models):
    directory, instances = timed_models
    detector = WakeWordDetector(directory)
    assert detector._history_limit == 28320 and detector._alignment_steps == 5
    assert not detector.feed(np.zeros(160000, dtype=np.float32))
    assert detector._history_samples == 28320
    assert sum(map(len, detector._history)) == 28320
    audio = np.zeros(48000, dtype=np.float32)
    audio[:4] = [.1, .2, .3, .4]
    assert detector.feed(audio)
    assert detector.last_detection.post_wake_samples == 47997
    assert detector._history_samples == 0 and not detector._history
    detector.close()
    assert detector.last_detection is None


@pytest.mark.parametrize('metadata', [
    {}, {'model_type': 'unknown', 'version': '1', 'T': '45', 'decode_chunk_len': '32'},
    {'model_type': 'zipformer2', 'version': '2', 'T': '45', 'decode_chunk_len': '32'},
    {'model_type': 'zipformer2', 'version': '1', 'T': '32', 'decode_chunk_len': '32'},
    {'model_type': 'zipformer2', 'version': '1', 'T': '45', 'decode_chunk_len': '31'},
    {'model_type': 'zipformer2', 'version': '1', 'T': '5000', 'decode_chunk_len': '32'},
    {'model_type': 'zipformer2', 'version': '1', 'T': '77', 'decode_chunk_len': '64'},
    {'model_type': 'zipformer2', 'version': '1', 'T': '141', 'decode_chunk_len': '128'},
])
def test_unknown_encoder_metadata_cannot_enable_timed_handoff(tmp_path, metadata):
    path = tmp_path / 'encoder.onnx'
    path.write_bytes(encoder_metadata(**metadata))
    assert _encoder_timing(path) is None


def test_encoder_metadata_rejects_duplicate_keys_and_truncated_graph(tmp_path):
    path = tmp_path / 'encoder.onnx'
    entry = protobuf_string(14, protobuf_string(1, 'version') + protobuf_string(2, '1'))
    path.write_bytes(encoder_metadata(model_type='zipformer2', version=1, T=45, decode_chunk_len=32) + entry)
    assert _encoder_timing(path) is None
    path.write_bytes(protobuf_string(7, b'graph')[:-1])
    assert _encoder_timing(path) is None


def test_supported_timing_is_exposed_for_startup_readiness(timed_models, caplog):
    directory, _ = timed_models
    caplog.set_level('INFO', logger='musegadget.wake_word')
    detector = WakeWordDetector(directory)
    assert detector.timing_supported
    assert 'Acoustic wake-tail timing ready' in caplog.text
    detector.close()
    assert not detector.timing_supported


def test_legacy_string_only_backend_does_not_claim_timed_handoff(models):
    directory, _ = models
    detector = WakeWordDetector(directory)
    assert not detector.timing_supported


def test_constructor_registers_only_the_exact_phrase_with_small_cpu_models(models):
    directory, instances = models
    detector = WakeWordDetector(directory, phrase=" Hey   Muse ")
    spotter = instances[0]
    assert detector.phrase == "hey muse" and detector.sample_rate == 16000
    assert spotter.keyword == "▁HE Y ▁MU SE @HEY_MUSE\n"
    assert spotter.config["num_threads"] == 1
    assert spotter.config["provider"] == "cpu"
    assert spotter.config["keywords_threshold"] == 0.25
    assert all(spotter.config[name].endswith(".int8.onnx") for name in ("encoder", "decoder", "joiner"))
    assert not Path(spotter.config["keywords_file"]).exists()


@pytest.fixture
def phonetic_models(models, monkeypatch):
    directory, instances = models
    for filename in MODEL_FILES.values():
        (directory / filename).unlink()
    for filename in PHONETIC_MODEL_FILES.values():
        (directory / filename).write_bytes(b"model")
    (directory / "tokens.txt").write_text("<unk> 0\nHH 1\nEY1 2\nM 3\nY 4\nUW1 5\nZ 6\n")
    (directory / "en.phone").write_text("HEY HH EY1\nMUSE M Y UW1 Z\n")
    monkeypatch.setitem(sys.modules, "sentencepiece", None)
    return directory, instances


def test_phonetic_model_uses_lexicon_and_mixed_quantization_without_sentencepiece(phonetic_models):
    directory, instances = phonetic_models
    detector = WakeWordDetector(directory, phrase=" Hey   Muse ")
    spotter = instances[0]
    assert detector.phrase == "hey muse"
    assert spotter.keyword == "HH EY1 M Y UW1 Z @HEY_MUSE\n"
    assert {name: Path(spotter.config[name]).name for name in ("encoder", "decoder", "joiner")} == {
        "encoder": "encoder-epoch-13-avg-2-chunk-16-left-64.int8.onnx",
        "decoder": "decoder-epoch-13-avg-2-chunk-16-left-64.onnx",
        "joiner": "joiner-epoch-13-avg-2-chunk-16-left-64.int8.onnx",
    }
    assert spotter.config["keywords_threshold"] == .25
    assert spotter.config["keywords_score"] == 1
    assert spotter.config["max_active_paths"] == 4
    assert spotter.config["num_threads"] == 1
    assert spotter.config["provider"] == "cpu"
    assert not Path(spotter.config["keywords_file"]).exists()
    assert detector.feed(np.array([.1, .2, .3, .4], dtype=np.float32))


def test_phonetic_model_requires_its_own_lexicon(phonetic_models):
    directory, instances = phonetic_models
    (directory / "en.phone").unlink()
    with pytest.raises(ValueError, match="model is missing: en.phone"):
        WakeWordDetector(directory)
    assert instances == []


@pytest.mark.parametrize("lexicon,error", [
    ("HEY HH EY1\n", "lexicon is missing: MUSE"),
    ("HEY HH EY1\nMUSE M Y UNKNOWN Z\n", "incompatible with the tokens"),
    ("HEY HH EY1\nMUSE <unk>\n", "incompatible with the tokens"),
    ("HEY\nMUSE M Y UW1 Z\n", "invalid fields"),
])
def test_phonetic_model_rejects_invalid_pronunciations(phonetic_models, lexicon, error):
    directory, instances = phonetic_models
    (directory / "en.phone").write_text(lexicon)
    with pytest.raises(ValueError, match=error):
        WakeWordDetector(directory)
    assert instances == []


def test_phonetic_model_rejects_an_oversized_lexicon(phonetic_models):
    directory, instances = phonetic_models
    with (directory / "en.phone").open("wb") as lexicon:
        lexicon.truncate(8 * 1024 * 1024 + 1)
    with pytest.raises(ValueError, match="lexicon exceeds its size limit"):
        WakeWordDetector(directory)
    assert instances == []


def test_arbitrary_chunk_boundaries_preserve_detection_state(models):
    directory, _ = models
    phrase = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    for split in range(5):
        detector = WakeWordDetector(directory)
        results = [detector.feed(phrase[:split]), detector.feed(phrase[split:])]
        assert results.count(True) == 1
        assert not detector.feed(np.zeros(4, dtype=np.float32))


def test_large_chunks_are_decoded_in_bounded_blocks(models):
    directory, instances = models
    detector = WakeWordDetector(directory)
    assert not detector.feed(np.zeros(17000, dtype=np.float32))
    parts = instances[0].streams[0].accepted
    assert max(map(len, parts)) <= 1600
    assert sum(map(len, parts)) == 17000


def test_strided_audio_is_accepted_without_changing_sample_order(models):
    directory, instances = models
    detector = WakeWordDetector(directory)
    original = np.arange(20, dtype=np.float32) / 20
    assert not detector.feed(original[::2])
    accepted = np.concatenate(instances[0].streams[0].accepted)
    np.testing.assert_array_equal(accepted, original[::2])


def test_reset_discards_queued_audio_and_previous_hypotheses(models):
    directory, instances = models
    detector = WakeWordDetector(directory)
    assert not detector.feed(np.array([0.1, 0.2], dtype=np.float32))
    first = instances[0].streams[-1]
    detector.reset()
    assert instances[0].streams[-1] is not first
    assert not detector.feed(np.array([0.3, 0.4, 0, 0], dtype=np.float32))
    detector.reset()
    assert detector.feed(np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32))
    assert instances[0].streams[-1].accepted == []


def test_an_unregistered_result_never_wakes_the_conversation(models):
    directory, instances = models
    detector = WakeWordDetector(directory)
    instances[0].force_result = "HEY_MUSIC"
    assert not detector.feed(np.zeros(4, dtype=np.float32))
    assert len(instances[0].streams) == 2


def test_microphone_summaries_continue_after_startup_with_fresh_window_stats(models, monkeypatch, caplog):
    import musegadget.wake_word as module

    directory, _ = models
    detector = WakeWordDetector(directory)
    caplog.set_level("INFO", logger="musegadget.wake_word")

    def feed(at, value, count=16000, delay=0):
        times = iter((at, at + delay))
        monkeypatch.setattr(module.time, "monotonic", lambda: next(times))
        assert not detector.feed(np.full(count, value, dtype=np.float32))

    feed(0, .1)
    for at in range(10, 61, 10):
        feed(at, .1)
    assert caplog.messages == [
        "Wake microphone: 2.00s audio over 10.00s wall, coverage 0.200, "
        "RMS 0.1000, peak 0.100, clipped 0.0000, max inference 0.0ms, min chunk 1000.0ms",
        *["Wake microphone: 1.00s audio over 10.00s wall, coverage 0.100, "
          "RMS 0.1000, peak 0.100, clipped 0.0000, max inference 0.0ms, min chunk 1000.0ms"] * 5,
    ]
    caplog.clear()

    feed(61, .999, 8000)
    feed(119, .5, 4000)
    assert caplog.messages == []
    feed(120, .25, 1600, delay=.05)
    assert caplog.messages == [
        "Wake microphone: 0.85s audio over 60.05s wall, coverage 0.014, "
        "RMS 0.8173, peak 0.999, clipped 0.5882, max inference 50.0ms, min chunk 100.0ms",
    ]
    caplog.clear()

    feed(121, .999, 8000)
    detector.reset()
    feed(200, .2)
    feed(210, .2)
    assert caplog.messages == []
    feed(260, .2)
    assert caplog.messages == [
        "Wake microphone: 3.00s audio over 60.00s wall, coverage 0.050, "
        "RMS 0.2000, peak 0.200, clipped 0.0000, max inference 0.0ms, min chunk 1000.0ms",
    ]


@pytest.mark.parametrize("audio", [
    np.zeros((2, 2), dtype=np.float32),
    np.array([float("nan")]), np.array([float("inf")]),
    np.array([1.1]), np.array([-1.1]), np.array(0.0),
])
def test_invalid_audio_is_rejected_before_decoding(models, audio):
    directory, instances = models
    detector = WakeWordDetector(directory)
    with pytest.raises(ValueError, match="finite mono"):
        detector.feed(audio)
    assert instances[0].streams[0].accepted == []


def test_close_releases_stream_and_rejects_further_audio(models):
    directory, _ = models
    detector = WakeWordDetector(directory)
    detector.close()
    detector.close()
    with pytest.raises(RuntimeError, match="closed"):
        detector.feed(np.zeros(4, dtype=np.float32))
    with pytest.raises(RuntimeError, match="closed"):
        detector.reset()


@pytest.mark.parametrize("operation", ["close", "reset"])
def test_lifecycle_waits_for_an_inference_worker_to_finish(models, operation):
    directory, instances = models
    detector = WakeWordDetector(directory)
    decoding = threading.Event()
    release = threading.Event()
    operation_started = threading.Event()
    completed = threading.Event()
    events = []
    original = instances[0].decode_stream

    def blocked_decode(stream):
        decoding.set()
        assert release.wait(2)
        original(stream)
        events.append("decoded")

    def lifecycle():
        operation_started.set()
        getattr(detector, operation)()
        events.append(operation)
        completed.set()

    instances[0].decode_stream = blocked_decode
    feeding = threading.Thread(target=lambda: detector.feed(np.zeros(4, dtype=np.float32)))
    finishing = threading.Thread(target=lifecycle)
    feeding.start()
    try:
        assert decoding.wait(1)
        finishing.start()
        assert operation_started.wait(1)
        assert not completed.wait(0.03)
    finally:
        release.set()
        feeding.join(2)
        if finishing.ident is not None:
            finishing.join(2)
    assert not feeding.is_alive() and not finishing.is_alive()
    assert events == ["decoded", operation]
    if operation == "close":
        with pytest.raises(RuntimeError, match="closed"):
            detector.feed(np.zeros(4, dtype=np.float32))
    else:
        assert not detector.feed(np.zeros(4, dtype=np.float32))


@pytest.mark.parametrize("phrase", ["", "hey/muse", "你好", "word " * 30, None])
def test_invalid_phrase_is_rejected_without_loading_models(tmp_path, phrase):
    with pytest.raises(ValueError, match="English words"):
        WakeWordDetector(tmp_path, phrase=phrase)


@pytest.mark.parametrize("setting", [
    {"threshold": 0}, {"threshold": 1.1}, {"threshold": float("nan")},
    {"score": 0}, {"score": float("inf")},
])
def test_invalid_decoder_settings_are_rejected(tmp_path, setting):
    with pytest.raises(ValueError):
        WakeWordDetector(tmp_path, **setting)


def test_missing_model_files_fail_before_optional_imports(tmp_path):
    with pytest.raises(ValueError, match="model is missing"):
        WakeWordDetector(tmp_path)


def test_optional_dependencies_are_imported_only_when_a_detector_is_created(models, monkeypatch):
    directory, _ = models
    monkeypatch.setitem(sys.modules, "sherpa_onnx", None)
    import musegadget.wake_word as module
    importlib.reload(module)
    with pytest.raises(RuntimeError, match="requires sherpa-onnx and sentencepiece"):
        WakeWordDetector(directory)


@pytest.mark.parametrize("table", [
    "▁HE 1\nY 2\n▁M 3\n", "▁HE 1\nY 2\n▁M 3\nUSE 3\n",
    "▁HE 1\nY 2\n▁M 3\nUSE invalid\n", "▁HE 1\nY 2\n▁M 3\nUSE 4 extra\n",
])
def test_incompatible_vocabulary_is_rejected(models, table):
    directory, _ = models
    (directory / "tokens.txt").write_text(table)
    with pytest.raises(ValueError):
        WakeWordDetector(directory)
