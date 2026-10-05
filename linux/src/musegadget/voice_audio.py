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

"""Streaming microphone endpoint detection and Muse speech decoding.

The audio dependencies load when a recorder or decoder is constructed, so
installations that do not use voice keep the base SDK's dependencies.
"""

from __future__ import annotations

import io
import math
import wave
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def _audio_dependencies():
    try:
        import av
        import numpy
    except ImportError as exc:
        raise RuntimeError("Voice audio requires pip install 'musegadget[reachy]'") from exc
    return av, numpy


@dataclass(frozen=True)
class SpeechAudio:
    """Authorized mono PCM16 at 16 kHz for the current utterance."""

    pcm: bytes


@dataclass(frozen=True)
class SpeechEnd:
    """The recorder's endpoint, including rejected short utterances."""

    accepted: bool


class TurnRecorder:
    """Collect sustained speech as a mono, 16 kHz, PCM16 WAV.

    Input chunks may have any length. Resampling and VAD framing retain their
    state between calls. Three consecutive voiced 20 ms frames start a turn;
    the preceding 200 ms preserve the start of the word. ``min_s`` counts
    voiced audio, while ``max_s`` caps the whole WAV, including its silence.
    ``last_truncated`` reports whether the last returned turn hit that cap.
    ``initial_min_s`` can temporarily lower the threshold for invocation audio;
    call ``finish_initial_capture`` once that initial capture has been admitted.

    ``feed`` returns at most one turn. Any remaining input stays buffered for
    the next call. Call ``reset`` when suspending capture for speaker playback
    to discard audio from before that suspension.
    """

    RATE = 16_000
    FRAME_SAMPLES = 320
    FRAME_BYTES = FRAME_SAMPLES * 2
    FRAME_SECONDS = 0.02

    def __init__(
        self,
        input_rate: int,
        vad=None,
        silence_s: float = 0.7,
        min_s: float = 0.3,
        max_s: float = 20.0,
        initial_min_s: float | None = None,
        stream_audio: bool = False,
    ):
        if not isinstance(input_rate, int) or input_rate <= 0:
            raise ValueError("input_rate must be a positive integer")
        if not all(math.isfinite(value) for value in (silence_s, min_s, max_s)):
            raise ValueError("Voice durations must be finite")
        if silence_s <= 0 or min_s <= 0 or not 0.06 <= max_s <= 60 or min_s > max_s:
            raise ValueError("Voice durations must be positive, with min_s <= max_s <= 60")
        self._av, self._np = _audio_dependencies()
        if vad is None:
            try:
                import webrtcvad
            except ImportError as exc:
                raise RuntimeError("Voice audio requires pip install 'musegadget[reachy]'") from exc
            vad = webrtcvad.Vad(2)
        self._vad = vad
        self._input_rate = input_rate
        self._silence_frames = max(
            1, math.ceil(math.nextafter(silence_s / self.FRAME_SECONDS, -math.inf))
        )
        self._min_frames = max(
            1, math.ceil(math.nextafter(min_s / self.FRAME_SECONDS, -math.inf))
        )
        self._normal_min_frames = self._min_frames
        if initial_min_s is not None and (
                not math.isfinite(initial_min_s) or not 0 < initial_min_s <= max_s):
            raise ValueError("Initial voice duration must be positive and within max_s")
        self._initial_min_frames = (self._min_frames if initial_min_s is None else max(
            1, math.ceil(math.nextafter(initial_min_s / self.FRAME_SECONDS, -math.inf))))
        self._initial_capture = initial_min_s is not None
        self._max_frames = math.floor(math.nextafter(max_s / self.FRAME_SECONDS, math.inf))
        self._history = deque(maxlen=min(10, self._max_frames))
        self._stream_audio = stream_audio
        self._audio_events: list[SpeechAudio | SpeechEnd] = []
        self.last_truncated = False
        self.reset()

    @property
    def active(self) -> bool:
        return bool(self._turn)

    @property
    def speech_active(self) -> bool:
        """Sustained speech remains active until the silence endpoint, across WAV caps."""
        return self._speech_active

    def reset(self) -> None:
        """Discard capture, pending samples, and resampling filter history."""
        reset_vad = getattr(self._vad, "reset", None)
        if reset_vad is not None:
            reset_vad()
        self._resampler = self._av.AudioResampler(format="s16", layout="mono", rate=self.RATE)
        self._pending = bytearray()
        self._audio_events.clear()
        self._speech_active = False
        self._speech_run = 0
        self._speech_quiet = 0
        self.last_truncated = False
        self._clear_turn()
        self._min_frames = self._initial_min_frames

    def take_audio_events(self) -> list[SpeechAudio | SpeechEnd]:
        """Drain audio and endpoints in capture order after each ``feed``.

        Audio starts with the same pre-roll retained in the WAV. Internal VAD
        decisions never omit audio within an utterance. The caller owns aborting
        its recognizer when this recorder is reset or replaced.
        """
        events, self._audio_events = self._audio_events, []
        return events

    def finish_initial_capture(self) -> None:
        """Restore the normal speech minimum without losing buffered samples."""
        self._initial_capture = False
        self._min_frames = self._normal_min_frames

    def _clear_turn(self) -> None:
        self._min_frames = (self._initial_min_frames if self._initial_capture
                            else self._normal_min_frames)
        self._history.clear()
        self._turn: list[bytes] = []
        self._voice_run = 0
        self._voiced_frames = 0
        self._quiet_frames = 0

    def feed(self, samples: np.ndarray) -> bytes | None:
        """Consume float mono samples and return an ended speech turn, if any."""
        samples = self._np.asarray(samples, dtype=self._np.float32)
        if samples.ndim != 1 or not self._np.isfinite(samples).all():
            raise ValueError("Microphone samples must be a finite, one-dimensional mono array")
        if samples.size:
            samples = self._np.ascontiguousarray(self._np.clip(samples, -1.0, 1.0)[None, :])
            frame = self._av.AudioFrame.from_ndarray(samples, format="flt", layout="mono")
            frame.sample_rate = self._input_rate
            for output in self._resampler.resample(frame):
                self._pending.extend(output.to_ndarray().astype("<i2", copy=False).tobytes())

        offset = 0
        result = None
        while len(self._pending) - offset >= self.FRAME_BYTES:
            pcm = bytes(self._pending[offset:offset + self.FRAME_BYTES])
            offset += self.FRAME_BYTES
            voiced = bool(self._vad.is_speech(pcm, self.RATE))
            self._speech_run = self._speech_run + 1 if voiced else 0
            self._speech_quiet = 0 if voiced else self._speech_quiet + 1
            if self._speech_run >= 3:
                self._speech_active = True
            elif self._speech_quiet >= self._silence_frames:
                self._speech_active = False
            if not self.active:
                self._history.append((pcm, voiced))
                self._voice_run = self._voice_run + 1 if voiced else 0
                if self._voice_run < 3:
                    continue
                self._turn = [audio for audio, _ in self._history]
                if self._stream_audio:
                    self._audio_events.append(SpeechAudio(b"".join(self._turn)))
                self._voiced_frames = sum(speech for _, speech in self._history)
                self._history.clear()
            else:
                self._turn.append(pcm)
                if self._stream_audio:
                    self._audio_events.append(SpeechAudio(pcm))
                self._voiced_frames += int(voiced)
            self._quiet_frames = 0 if voiced else self._quiet_frames + 1
            truncated = len(self._turn) >= self._max_frames
            if truncated or self._quiet_frames >= self._silence_frames:
                accepted = self._voiced_frames >= self._min_frames
                if self._stream_audio:
                    self._audio_events.append(SpeechEnd(accepted))
                if accepted:
                    result = self._wav()
                    self.last_truncated = truncated
                self._clear_turn()
                if result is not None:
                    break
        del self._pending[:offset]
        return result

    def _wav(self) -> bytes:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(self.RATE)
            output.writeframes(b"".join(self._turn))
        return buffer.getvalue()


class Mp3Decoder:
    """Decode incremental MP3 bytes to mono float32 at the speaker's rate."""

    def __init__(self, output_rate: int):
        if not isinstance(output_rate, int) or output_rate <= 0:
            raise ValueError("output_rate must be a positive integer")
        self._av, self._np = _audio_dependencies()
        self._codec = self._av.CodecContext.create("mp3float", "r")
        self._resampler = self._av.AudioResampler(format="fltp", layout="mono", rate=output_rate)
        self._prefix = bytearray()
        self._at_start = True
        self._tag_remaining = 0
        self._finished = False

    def feed(self, data: bytes) -> list[np.ndarray]:
        if self._finished:
            raise ValueError("Cannot feed an MP3 decoder after finish")
        if not data:
            return []
        # Codec parsers consume elementary MP3 frames, not the ID3 metadata
        # that commonly prefixes an MP3 response. Its header may span chunks.
        if self._at_start:
            self._prefix.extend(data)
            if len(self._prefix) < 3:
                return []
            if self._prefix[:3] == b"ID3":
                if len(self._prefix) < 10:
                    return []
                size_bytes = self._prefix[6:10]
                if any(value & 0x80 for value in size_bytes):
                    raise ValueError("Invalid MP3 ID3 tag size")
                size = sum(value << (7 * (3 - index)) for index, value in enumerate(size_bytes))
                if size > 1_048_576:
                    raise ValueError("MP3 ID3 metadata exceeds 1 MiB")
                self._tag_remaining = 10 + size + (10 if self._prefix[5] & 0x10 else 0)
            data = bytes(self._prefix)
            self._prefix.clear()
            self._at_start = False
        if self._tag_remaining:
            skipped = min(self._tag_remaining, len(data))
            self._tag_remaining -= skipped
            data = data[skipped:]
        return self._decode(self._codec.parse(data)) if data else []

    def finish(self) -> list[np.ndarray]:
        """Flush the parser, decoder, and resampler once at response EOF."""
        if self._finished:
            return []
        self._finished = True
        if self._tag_remaining or self._prefix:
            raise ValueError("Incomplete MP3 header")
        output = self._decode(self._codec.parse(None))
        output.extend(self._resample(self._codec.decode(None)))
        output.extend(self._arrays(self._resampler.resample(None)))
        return output

    def _decode(self, packets) -> list[np.ndarray]:
        return [
            samples
            for packet in packets
            for samples in self._resample(self._codec.decode(packet))
        ]

    def _resample(self, frames) -> list[np.ndarray]:
        return [
            samples
            for frame in frames
            for samples in self._arrays(self._resampler.resample(frame))
        ]

    def _arrays(self, frames) -> list[np.ndarray]:
        return [self._np.ascontiguousarray(frame.to_ndarray().reshape(-1)) for frame in frames]
