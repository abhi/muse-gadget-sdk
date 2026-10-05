# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")

from musegadget.vosk_wake import VoskWakeWordDetector


class FakeRecognizer:
    def __init__(self, model, rate, grammar):
        self.model = model
        self.rate = rate
        self.grammar = json.loads(grammar)
        self.plan = []
        self.received = []
        self.last = (False, "")

    def AcceptWaveform(self, pcm):
        self.received.append(np.frombuffer(pcm, dtype="<i2").copy())
        self.last = self.plan.pop(0) if self.plan else (False, "")
        return self.last[0]

    def Result(self):
        return json.dumps({"text": self.last[1]})

    def PartialResult(self):
        return json.dumps({"partial": self.last[1]})


@pytest.fixture
def models(tmp_path, monkeypatch):
    for name in ("am/final.mdl", "conf/model.conf", "conf/mfcc.conf",
                 "graph/HCLr.fst", "graph/Gr.fst"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"model")
    instances = []
    vocabulary = {"hey", "muse", "news", "music", "moose", "mouse", "[unk]"}

    class Model:
        def __init__(self, path):
            assert Path(path) == tmp_path

        def vosk_model_find_word(self, word):
            return 1 if word in vocabulary else -1

    def recognize(*args):
        instance = FakeRecognizer(*args)
        instances.append(instance)
        return instance

    monkeypatch.setitem(sys.modules, "vosk", SimpleNamespace(Model=Model, KaldiRecognizer=recognize))
    return tmp_path, instances, vocabulary


def test_endpoint_mode_keeps_the_native_competitor_grammar_and_normalizes_phrase(models):
    directory, instances, _ = models
    detector = VoskWakeWordDetector(directory, phrase=" Hey   Muse ")
    assert detector.phrase == "hey muse"
    assert detector.mode == "endpoint" and detector.sample_rate == 16000
    assert instances[0].rate == 16000
    assert instances[0].grammar == [
        "hey muse", "hey news", "hey music", "hey moose", "hey mouse", "[unk]",
    ]


@pytest.mark.parametrize("text", [
    "hey muse", "HEY MUSE", "hey muse [unk]", "[unk] hey muse [unk]",
    "hey muse what time is it",
])
def test_endpoint_recognizes_exact_wake_tokens_with_a_same_breath_question(models, text):
    directory, instances, _ = models
    detector = VoskWakeWordDetector(directory)
    instances[-1].plan = [(True, text)]
    assert detector.feed(np.zeros(1600, dtype=np.float32))
    assert len(instances) == 2
    assert not detector.feed(np.zeros(1600, dtype=np.float32))


@pytest.mark.parametrize("text", [
    "hey news", "hey music", "hey moose", "hey mouse", "hey museum", "hey musician",
    "hello muse", "hey [unk] muse", "muse", "[unk]", "I need to check the news",
])
def test_confusables_and_nonadjacent_words_do_not_activate(models, text):
    directory, instances, _ = models
    detector = VoskWakeWordDetector(directory)
    instances[-1].plan = [(True, text)]
    assert not detector.feed(np.zeros(1600, dtype=np.float32))
    assert len(instances) == 1


def test_default_mode_waits_for_endpoint_even_when_partial_predictions_persist(models):
    directory, instances, _ = models
    detector = VoskWakeWordDetector(directory)
    instances[-1].plan = [(False, "hey muse")] * 10 + [(True, "hey music")]
    assert not detector.feed(np.zeros(17600, dtype=np.float32))


def test_partial_mode_stabilizes_in_audio_time_while_the_question_continues(models):
    directory, instances, _ = models
    detector = VoskWakeWordDetector(directory, mode="stable_partial")
    instances[-1].plan = [(False, "hey muse")] + [(False, "hey muse [unk]")] * 6
    assert not detector.feed(np.zeros(1600, dtype=np.float32))
    assert not detector.feed(np.zeros(4800, dtype=np.float32))
    assert detector.feed(np.zeros(4800, dtype=np.float32))


def test_changing_or_reset_partial_hypotheses_cannot_complete_stabilization(models):
    directory, instances, _ = models
    detector = VoskWakeWordDetector(directory, mode="stable_partial")
    instances[-1].plan = [(False, "hey muse")] * 3 + [(False, "hey music")] * 4
    assert not detector.feed(np.zeros(11200, dtype=np.float32))
    instances[-1].plan = [(False, "hey muse")] * 3
    assert not detector.feed(np.zeros(4800, dtype=np.float32))
    detector.reset()
    instances[-1].plan = [(False, "hey muse")] * 3
    assert not detector.feed(np.zeros(4800, dtype=np.float32))


def test_pcm_conversion_preserves_extrema_and_strided_sample_order_in_bounded_blocks(models):
    directory, instances, _ = models
    detector = VoskWakeWordDetector(directory)
    audio = np.array([-1, .2, -.5, .2, 0, .2, .5, .2, 1], dtype=np.float32)
    assert not detector.feed(audio[::2])
    assert instances[0].received[0].tolist() == [-32767, -16384, 0, 16384, 32767]
    assert not detector.feed(np.zeros(17000, dtype=np.float32))
    assert max(map(len, instances[0].received)) <= 1600
    assert sum(map(len, instances[0].received)) == 17005


def test_empty_audio_is_a_noop_even_after_a_logged_window(models, monkeypatch):
    import musegadget.vosk_wake as module

    directory, instances, _ = models
    detector = VoskWakeWordDetector(directory)
    times = iter((0, 0, 10, 10))
    monkeypatch.setattr(module.time, "monotonic", lambda: next(times))
    assert not detector.feed(np.zeros(1600, dtype=np.float32))
    assert not detector.feed(np.zeros(1600, dtype=np.float32))
    monkeypatch.setattr(module.time, "monotonic", lambda: 100)
    assert not detector.feed(np.zeros(0, dtype=np.float32))
    assert len(instances[0].received) == 2


@pytest.mark.parametrize("audio", [
    np.zeros((2, 2)), np.array([np.nan]), np.array([np.inf]),
    np.array([1.1]), np.array([-1.1]), np.array(0.0),
])
def test_invalid_audio_is_rejected_before_native_inference(models, audio):
    directory, instances, _ = models
    detector = VoskWakeWordDetector(directory)
    with pytest.raises(ValueError, match="finite mono floats"):
        detector.feed(audio)
    assert instances[0].received == []


@pytest.mark.parametrize("raw", ['[]', '{"text": 3}', 'invalid JSON'])
def test_invalid_native_result_is_not_accepted_as_a_wake(models, monkeypatch, raw):
    directory, instances, _ = models
    detector = VoskWakeWordDetector(directory)
    instances[-1].plan = [(True, "hey muse")]
    monkeypatch.setattr(instances[-1], "Result", lambda: raw)
    with pytest.raises(ValueError):
        detector.feed(np.zeros(1600, dtype=np.float32))


def test_health_logs_continue_after_startup_and_reset_keeps_the_logging_cadence(models, monkeypatch, caplog):
    import musegadget.vosk_wake as module

    directory, _, _ = models
    detector = VoskWakeWordDetector(directory)
    caplog.set_level("INFO", logger="musegadget.vosk_wake")

    def feed(at, value, count=16000, delay=0):
        times = iter((at, at + delay))
        monkeypatch.setattr(module.time, "monotonic", lambda: next(times))
        assert not detector.feed(np.full(count, value, dtype=np.float32))

    feed(0, .1)
    for at in range(10, 61, 10):
        feed(at, .1)
    assert len(caplog.messages) == 6
    assert caplog.messages[0] == (
        "Wake microphone: 2.00s audio over 10.00s wall, coverage 0.200, "
        "RMS 0.1000, peak 0.100, clipped 0.0000, max inference 0.0ms, min chunk 1000.0ms"
    )
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
    detector.reset()
    feed(200, .2)
    feed(210, .2)
    assert caplog.messages == []
    feed(260, .2)
    assert caplog.messages == [
        "Wake microphone: 3.00s audio over 60.00s wall, coverage 0.050, "
        "RMS 0.2000, peak 0.200, clipped 0.0000, max inference 0.0ms, min chunk 1000.0ms",
    ]


@pytest.mark.parametrize("operation", ["reset", "close"])
def test_lifecycle_waits_for_native_inference_to_finish(models, monkeypatch, operation):
    directory, instances, _ = models
    detector = VoskWakeWordDetector(directory)
    entered = threading.Event()
    release = threading.Event()
    operation_started = threading.Event()
    completed = threading.Event()
    accept = instances[0].AcceptWaveform

    def blocked(pcm):
        entered.set()
        assert release.wait(2)
        return accept(pcm)

    def lifecycle():
        operation_started.set()
        getattr(detector, operation)()
        completed.set()

    monkeypatch.setattr(instances[0], "AcceptWaveform", blocked)
    feeding = threading.Thread(target=lambda: detector.feed(np.zeros(1600, dtype=np.float32)))
    finishing = threading.Thread(target=lifecycle)
    feeding.start()
    try:
        assert entered.wait(1)
        finishing.start()
        assert operation_started.wait(1)
        assert not completed.wait(.03)
    finally:
        release.set()
        feeding.join(2)
        if finishing.ident is not None:
            finishing.join(2)
    assert not feeding.is_alive() and not finishing.is_alive()
    assert completed.is_set()
    if operation == "close":
        with pytest.raises(RuntimeError, match="closed"):
            detector.feed(np.zeros(1600, dtype=np.float32))
    else:
        assert not detector.feed(np.zeros(1600, dtype=np.float32))


def test_close_is_idempotent_and_releases_native_resources(models):
    directory, _, _ = models
    detector = VoskWakeWordDetector(directory)
    detector.close()
    detector.close()
    with pytest.raises(RuntimeError, match="closed"):
        detector.reset()
    with pytest.raises(RuntimeError, match="closed"):
        detector.feed(np.zeros(1, dtype=np.float32))


@pytest.mark.parametrize("phrase", [None, "", "hey music", "hello muse", "你好"])
def test_unverified_phrases_fail_before_loading_models(tmp_path, phrase):
    with pytest.raises(ValueError, match="supports only the phrase Hey Muse"):
        VoskWakeWordDetector(tmp_path, phrase=phrase)


@pytest.mark.parametrize("settings", [
    {"mode": "unknown"}, {"stable_partial_s": 0},
    {"stable_partial_s": float("nan")}, {"stable_partial_s": float("inf")},
])
def test_invalid_modes_or_durations_fail_before_loading_models(tmp_path, settings):
    with pytest.raises(ValueError):
        VoskWakeWordDetector(tmp_path, **settings)


def test_missing_required_dynamic_grammar_assets_fail_before_optional_imports(models, monkeypatch):
    directory, instances, _ = models
    (directory / "graph/Gr.fst").unlink()
    monkeypatch.setitem(sys.modules, "vosk", None)
    with pytest.raises(ValueError, match="model is missing: graph/Gr.fst"):
        VoskWakeWordDetector(directory)
    assert instances == []


def test_model_vocabulary_must_include_wake_and_competing_words(models):
    directory, instances, vocabulary = models
    vocabulary.remove("muse")
    with pytest.raises(ValueError, match="vocabulary is missing: muse"):
        VoskWakeWordDetector(directory)
    assert instances == []


def test_optional_dependencies_load_only_when_constructing_a_detector(models, monkeypatch):
    directory, _, _ = models
    import musegadget.vosk_wake as module
    monkeypatch.setitem(sys.modules, "vosk", None)
    monkeypatch.setitem(sys.modules, "numpy", None)
    importlib.reload(module)
    with pytest.raises(RuntimeError, match="requires vosk and numpy"):
        VoskWakeWordDetector(directory)
