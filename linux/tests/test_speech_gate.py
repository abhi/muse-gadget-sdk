from types import SimpleNamespace

import numpy as np
import pytest

from musegadget.speech_gate import SileroSpeechGate


class FakeSession:
    def __init__(self, probabilities):
        self.probabilities = iter(probabilities)
        self.calls = []

    def get_inputs(self):
        return [SimpleNamespace(name=name) for name in ("input", "h", "c")]

    def get_outputs(self):
        return [SimpleNamespace(name=name) for name in ("speech_probs", "hn", "cn")]

    def run(self, outputs, inputs):
        self.calls.append({key: value.copy() for key, value in inputs.items()})
        probability = next(self.probabilities)
        state = np.full((1, 1, 128), len(self.calls), dtype=np.float32)
        return [np.array([[probability]], dtype=np.float32), state, -state]


def pcm(value=0):
    return np.full(320, value, dtype="<i2").tobytes()


def test_buffers_twenty_ms_frames_and_carries_context_and_state():
    session = FakeSession([.8, .7])
    gate = SileroSpeechGate(session=session)

    assert gate.is_speech(pcm(100), 16_000) is False
    assert gate.pending_samples == 320
    assert gate.is_speech(pcm(200), 16_000) is True
    assert gate.pending_samples == 128
    assert len(session.calls) == 1
    assert session.calls[0]["input"].shape == (1, 576)
    assert np.all(session.calls[0]["input"][0, :64] == 0)

    assert gate.is_speech(pcm(300), 16_000) is True
    assert gate.is_speech(pcm(400), 16_000) is True
    assert len(session.calls) == 2
    assert np.all(session.calls[1]["input"][0, :64] == pytest.approx(200 / 32768))
    assert np.all(session.calls[1]["h"] == 1)
    assert np.all(session.calls[1]["c"] == -1)
    assert gate.pending_samples < gate.WINDOW_SAMPLES


def test_hysteresis_holds_ambiguous_probability_then_releases():
    gate = SileroSpeechGate(session=FakeSession([.5, .4, .34]))
    decisions = []
    for _ in range(5):
        decisions.append(gate.is_speech(pcm(), 16_000))
    assert decisions == [False, True, True, True, False]


def test_reset_clears_audio_model_state_and_decision():
    session = FakeSession([.9, .9])
    gate = SileroSpeechGate(session=session)
    gate.is_speech(pcm(100), 16_000)
    assert gate.is_speech(pcm(200), 16_000)

    gate.reset()
    assert gate.pending_samples == 0
    assert gate.is_speech(pcm(300), 16_000) is False
    assert gate.is_speech(pcm(400), 16_000) is True
    assert np.all(session.calls[1]["input"][0, :64] == 0)
    assert np.all(session.calls[1]["h"] == 0)
    assert np.all(session.calls[1]["c"] == 0)


@pytest.mark.parametrize("rate, frame", [(8_000, pcm()), (16_000, b""), (16_000, bytearray(640))])
def test_rejects_audio_outside_recorder_contract(rate, frame):
    gate = SileroSpeechGate(session=FakeSession([]))
    with pytest.raises(ValueError):
        gate.is_speech(frame, rate)


@pytest.mark.parametrize("threshold, release", [(float("nan"), .35), (.5, float("inf")), (0, -.1), (.5, .5), (.5, .6), (1, .5)])
def test_rejects_invalid_thresholds(threshold, release):
    with pytest.raises(ValueError):
        SileroSpeechGate(session=FakeSession([]), threshold=threshold, neg_threshold=release)


def test_rejects_unknown_model_schema():
    session = FakeSession([])
    session.get_inputs = lambda: [SimpleNamespace(name="samples")]
    with pytest.raises(RuntimeError, match="unsupported schema"):
        SileroSpeechGate(session=session)


@pytest.mark.parametrize("result", [
    [np.array([[np.nan]], dtype=np.float32), np.zeros((1, 1, 128)), np.zeros((1, 1, 128))],
    [np.array([[1.1]], dtype=np.float32), np.zeros((1, 1, 128)), np.zeros((1, 1, 128))],
    [np.array([[.8]], dtype=np.float32), np.zeros((1, 1, 12)), np.zeros((1, 1, 128))],
    [np.array([[.8]], dtype=np.float32), np.zeros((1, 1, 128)), np.full((1, 1, 128), np.inf)],
])
def test_rejects_invalid_native_results(result):
    session = FakeSession([])
    session.run = lambda outputs, inputs: result
    gate = SileroSpeechGate(session=session)
    gate.is_speech(pcm(), 16_000)
    with pytest.raises(RuntimeError, match="Silero VAD returned"):
        gate.is_speech(pcm(), 16_000)
