# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Optional local Vosk wake recognition with explicit competing phrases."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
import threading
import time

log = logging.getLogger(__name__)
MODEL_FILES = (
    "am/final.mdl", "conf/model.conf", "conf/mfcc.conf",
    "graph/HCLr.fst", "graph/Gr.fst",
)
GRAMMAR = ("hey muse", "hey news", "hey music", "hey moose", "hey mouse", "[unk]")


class VoskWakeWordDetector:
    """Detect Hey Muse in mono 16 kHz floats using a small dynamic-grammar model.

    Endpoint mode waits for a complete recognition segment. The experimental
    stable_partial mode may trigger before a hypothesis subsequently changes.
    Its duration counts input audio, so caller scheduling cannot advance it.
    Lifecycle operations serialize with inference. Capture pre-roll belongs to
    the conversation runner, not this detector.
    """

    sample_rate = 16000

    def __init__(self, model_dir: str | Path, phrase: str = "hey muse", *,
                 mode: str = "endpoint", stable_partial_s: float = .5) -> None:
        if not isinstance(phrase, str) or " ".join(phrase.lower().split()) != "hey muse":
            raise ValueError("Vosk wake detection supports only the phrase Hey Muse.")
        if mode not in ("endpoint", "stable_partial"):
            raise ValueError("Vosk wake mode must be endpoint or stable_partial.")
        if not math.isfinite(stable_partial_s) or stable_partial_s <= 0:
            raise ValueError("Stable partial duration must be positive and finite.")
        directory = Path(model_dir).expanduser()
        missing = [name for name in MODEL_FILES
                   if not (directory / name).is_file() or not (directory / name).stat().st_size]
        if missing:
            raise ValueError("Vosk wake model is missing: " + ", ".join(missing))
        try:
            import numpy as np
            import vosk
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "Vosk wake detection requires vosk and numpy; "
                "install the Reachy Vosk wake dependencies."
            ) from exc
        self.phrase = "hey muse"
        self.mode = mode
        self._stable_partial_s = stable_partial_s
        self._lock = threading.RLock()
        self._np = np
        self._model = None
        self._recognizer = None
        self._recognizer_factory = vosk.KaldiRecognizer
        self._grammar = json.dumps(GRAMMAR)
        self._diagnostic_windows = 0
        try:
            self._model = vosk.Model(str(directory))
            words = {word for item in GRAMMAR for word in item.split()}
            missing_words = sorted(word for word in words if self._model.vosk_model_find_word(word) < 0)
            if missing_words:
                raise ValueError("Vosk wake vocabulary is missing: " + ", ".join(missing_words))
            self.reset()
        except BaseException:
            self.close()
            raise
        log.info("Using local Vosk wake detection (%s)", mode)

    def _reset_diagnostics(self, started=None) -> None:
        self._diagnostic_start = started
        self._diagnostic_samples = 0
        self._diagnostic_energy = 0.0
        self._diagnostic_peak = 0.0
        self._diagnostic_clipped = 0
        self._diagnostic_max_feed = 0.0
        self._diagnostic_min_chunk = math.inf

    def reset(self) -> None:
        """Discard recognition, partial hypotheses, and the current audio window."""
        with self._lock:
            if self._model is None:
                raise RuntimeError("Vosk wake detector is closed.")
            self._recognizer = self._recognizer_factory(self._model, self.sample_rate, self._grammar)
            self._audio_seconds = 0.0
            self._partial_started = None
            self._reset_diagnostics()

    @staticmethod
    def _result_has_phrase(raw: str, field: str) -> bool:
        result = json.loads(raw)
        if not isinstance(result, dict) or not isinstance(result.get(field, ""), str):
            raise ValueError("Vosk returned an invalid wake recognition result.")
        words = result.get(field, "").lower().split()
        return any(words[index:index + 2] == ["hey", "muse"]
                   for index in range(len(words) - 1))

    def feed(self, samples) -> bool:
        """Consume normalized float audio; never activate on a partial word match."""
        with self._lock:
            if self._recognizer is None:
                raise RuntimeError("Vosk wake detector is closed.")
            np = self._np
            audio = np.asarray(samples, dtype=np.float32)
            if audio.ndim != 1 or not np.isfinite(audio).all() or (np.abs(audio) > 1).any():
                raise ValueError("Wake-word audio must be finite mono floats between -1 and 1.")
            if not audio.size:
                return False
            started = time.monotonic()
            if self._diagnostic_start is None:
                self._diagnostic_start = started
            self._diagnostic_samples += audio.size
            self._diagnostic_energy += float(np.dot(audio, audio))
            self._diagnostic_peak = max(self._diagnostic_peak, float(np.max(np.abs(audio))))
            self._diagnostic_clipped += int(np.count_nonzero(np.abs(audio) >= .999))
            self._diagnostic_min_chunk = min(self._diagnostic_min_chunk, audio.size / self.sample_rate)
            for offset in range(0, audio.size, 1600):
                part = audio[offset:offset + 1600]
                self._audio_seconds += part.size / self.sample_rate
                pcm = np.rint(part * 32767).astype("<i2").tobytes()
                if self._recognizer.AcceptWaveform(pcm):
                    detected = self._result_has_phrase(self._recognizer.Result(), "text")
                    self._partial_started = None
                    if detected:
                        self.reset()
                        return True
                elif self.mode == "stable_partial":
                    if not self._result_has_phrase(self._recognizer.PartialResult(), "partial"):
                        self._partial_started = None
                    elif self._partial_started is None:
                        self._partial_started = self._audio_seconds
                    elif self._audio_seconds - self._partial_started >= self._stable_partial_s:
                        self.reset()
                        return True
            if self._diagnostic_start is not None:
                ended = time.monotonic()
                self._diagnostic_max_feed = max(self._diagnostic_max_feed, ended - started)
                elapsed = ended - self._diagnostic_start
                interval = 10 if self._diagnostic_windows < 6 else 60
                if elapsed >= interval:
                    log.info("Wake microphone: %.2fs audio over %.2fs wall, coverage %.3f, "
                             "RMS %.4f, peak %.3f, clipped %.4f, max inference %.1fms, min chunk %.1fms",
                             self._diagnostic_samples / self.sample_rate, elapsed,
                             self._diagnostic_samples / self.sample_rate / elapsed,
                             math.sqrt(self._diagnostic_energy / self._diagnostic_samples),
                             self._diagnostic_peak, self._diagnostic_clipped / self._diagnostic_samples,
                             self._diagnostic_max_feed * 1000,
                             self._diagnostic_min_chunk * 1000)
                    self._diagnostic_windows += 1
                    self._reset_diagnostics(ended)
            return False

    def close(self) -> None:
        with self._lock:
            self._recognizer = None
            self._model = None
