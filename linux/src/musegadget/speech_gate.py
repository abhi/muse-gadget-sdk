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

"""Optional streaming Silero speech gate for 16 kHz microphone frames."""

from __future__ import annotations

import importlib.metadata
import math
from pathlib import Path


class SileroSpeechGate:
    """Classify PCM16 microphone frames with faster-whisper's bundled Silero VAD.

    The public method matches ``webrtcvad.Vad.is_speech`` so a recorder can use
    either implementation. Silero consumes 512 samples at a time; 20 ms input
    frames are buffered and the most recent decision is held between inference
    windows. ``reset`` must be called whenever microphone continuity is lost.
    """

    RATE = 16_000
    INPUT_SAMPLES = 320
    WINDOW_SAMPLES = 512
    CONTEXT_SAMPLES = 64
    STATE_SHAPE = (1, 1, 128)

    def __init__(
        self,
        *,
        threshold: float = 0.5,
        neg_threshold: float = 0.35,
        session=None,
        model_path: Path | None = None,
    ):
        if not all(isinstance(value, (int, float)) and math.isfinite(value)
                   for value in (threshold, neg_threshold)):
            raise ValueError("Speech thresholds must be finite numbers")
        if not 0 < neg_threshold < threshold < 1:
            raise ValueError("Speech thresholds must satisfy 0 < release < onset < 1")
        self._np = _numpy()
        self._threshold = float(threshold)
        self._neg_threshold = float(neg_threshold)
        self._session = session if session is not None else _create_session(model_path)
        _validate_session(self._session)
        self.reset()

    @property
    def pending_samples(self) -> int:
        """Return buffered samples, primarily for bounded-state diagnostics."""
        return int(self._pending.size)

    def reset(self) -> None:
        """Forget buffered audio, recurrent model state, and speech hysteresis."""
        self._pending = self._np.empty(0, dtype=self._np.float32)
        self._context = self._np.zeros((1, self.CONTEXT_SAMPLES), dtype=self._np.float32)
        self._h = self._np.zeros(self.STATE_SHAPE, dtype=self._np.float32)
        self._c = self._np.zeros(self.STATE_SHAPE, dtype=self._np.float32)
        self._speech = False

    def is_speech(self, frame: bytes, sample_rate: int) -> bool:
        """Return the current speech decision for one 20 ms PCM16 frame."""
        if sample_rate != self.RATE:
            raise ValueError("Silero speech gate requires 16 kHz audio")
        if not isinstance(frame, bytes) or len(frame) != self.INPUT_SAMPLES * 2:
            raise ValueError("Silero speech gate requires one 20 ms PCM16 bytes frame")
        samples = self._np.frombuffer(frame, dtype="<i2").astype(self._np.float32)
        samples *= 1.0 / 32768.0
        self._pending = self._np.concatenate((self._pending, samples))
        while self._pending.size >= self.WINDOW_SAMPLES:
            window = self._pending[:self.WINDOW_SAMPLES]
            self._pending = self._pending[self.WINDOW_SAMPLES:]
            model_input = self._np.concatenate((self._context, window[None, :]), axis=1)
            values = self._session.run(
                None,
                {"input": model_input, "h": self._h, "c": self._c},
            )
            if not isinstance(values, (tuple, list)) or len(values) != 3:
                raise RuntimeError("Silero VAD returned an invalid output set")
            probability = self._probability(values[0])
            self._h = self._state(values[1], "h")
            self._c = self._state(values[2], "c")
            self._context = window[-self.CONTEXT_SAMPLES:][None, :].copy()
            if probability >= self._threshold:
                self._speech = True
            elif probability < self._neg_threshold:
                self._speech = False
        return self._speech

    def _probability(self, value) -> float:
        array = self._np.asarray(value, dtype=self._np.float32)
        if array.size != 1:
            raise RuntimeError("Silero VAD returned an invalid speech probability")
        probability = float(array.reshape(-1)[0])
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise RuntimeError("Silero VAD returned an invalid speech probability")
        return probability

    def _state(self, value, name: str):
        array = self._np.asarray(value, dtype=self._np.float32)
        if array.shape != self.STATE_SHAPE or not self._np.isfinite(array).all():
            raise RuntimeError(f"Silero VAD returned invalid {name} state")
        return array.copy()


def _numpy():
    try:
        import numpy
    except ImportError as exc:
        raise RuntimeError("Silero speech gating requires numpy") from exc
    return numpy


def _model_path() -> Path:
    try:
        distribution = importlib.metadata.distribution("faster-whisper")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("Silero speech gating requires faster-whisper") from exc
    path = Path(distribution.locate_file("faster_whisper/assets/silero_vad_v6.onnx"))
    if not path.is_file():
        raise RuntimeError("The faster-whisper Silero VAD asset is missing")
    return path


def _create_session(model_path: Path | None):
    try:
        import onnxruntime
    except ImportError as exc:
        raise RuntimeError("Silero speech gating requires onnxruntime") from exc
    path = Path(model_path) if model_path is not None else _model_path()
    if not path.is_file():
        raise RuntimeError("The Silero VAD model is missing")
    options = onnxruntime.SessionOptions()
    options.inter_op_num_threads = 1
    options.intra_op_num_threads = 1
    options.enable_cpu_mem_arena = False
    options.log_severity_level = 4
    return onnxruntime.InferenceSession(
        str(path), providers=["CPUExecutionProvider"], sess_options=options
    )


def _validate_session(session) -> None:
    try:
        inputs = [item.name for item in session.get_inputs()]
        outputs = [item.name for item in session.get_outputs()]
    except (AttributeError, TypeError) as exc:
        raise RuntimeError("Silero VAD session does not expose its schema") from exc
    if inputs != ["input", "h", "c"] or outputs != ["speech_probs", "hn", "cn"]:
        raise RuntimeError("Silero VAD session has an unsupported schema")
