# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Optional local speech transcription in a persistent CPU worker."""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import math
from pathlib import Path
import struct
import sys
from typing import Callable, Optional
import zlib

MAX_WAV_BYTES = 2 * 1024 * 1024
MAX_TRANSCRIPT_BYTES = 64 * 1024


class _LocalTranscriber:
    """Transcribe WAV turns without blocking the microphone's event loop.

    The model remains in one child process. Requests serialize; cancellation
    kills and reaps the worker so the next request can restart it. Neither
    transcripts nor worker stderr are written to logs.
    """

    _backend = "whisper"
    _name = "Whisper"
    _required_files = ("model.bin", "config.json", "tokenizer.json")

    def __init__(self, model_path: Path):
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_dir() or any(
            not (self.model_path / name).is_file() for name in self._required_files
        ):
            raise ValueError(f"{self._name} model directory is missing its model or tokenizer: {self.model_path}")
        self._process = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """Load the model and wait until the worker is ready."""
        async with self._lock:
            try:
                await self._ensure_worker()
            except BaseException:
                await self._stop_worker()
                raise

    async def close(self) -> None:
        """Kill and reap the worker after active transcription has stopped."""
        async with self._lock:
            await self._stop_worker()

    async def _ensure_worker(self) -> None:
        if self._process is not None and self._process.returncode is None:
            return
        await self._stop_worker()
        backend_args = ("--backend", self._backend) if self._backend != "whisper" else ()
        spawn = asyncio.create_task(asyncio.create_subprocess_exec(
            sys.executable, "-m", "musegadget.local_transcription", "--model", str(self.model_path),
            *backend_args,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        ))
        try:
            self._process = await asyncio.shield(spawn)
        except asyncio.CancelledError:
            self._process = await spawn
            raise
        if await self._packet_size() != 0:
            raise ValueError(f"{self._name} worker returned an invalid ready handshake")

    async def _stop_worker(self) -> None:
        process, self._process = self._process, None
        if process is not None:
            process.stdin.close()
            if process.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    process.kill()
            await process.communicate()

    async def _packet_size(self) -> int:
        try:
            header = await self._process.stdout.readexactly(4)
        except asyncio.IncompleteReadError:
            status = await self._process.wait()
            raise RuntimeError(f"{self._name} worker ended with exit status {status}; check the model and installation") from None
        size = struct.unpack("<I", header)[0]
        if size > MAX_TRANSCRIPT_BYTES:
            raise ValueError(f"{self._name} transcript exceeds 64 KiB")
        return size

    async def transcribe(self, wav: bytes, *, on_speech: Optional[Callable[[], None]] = None) -> str:
        """Recognize English speech, notifying once before decoding when VAD finds it."""
        if not isinstance(wav, bytes) or not 44 <= len(wav) <= MAX_WAV_BYTES:
            raise ValueError(f"{self._name} requires a WAV recording of at most 2 MiB")
        if wav[:4] != b"RIFF" or wav[8:12] != b"WAVE":
            raise ValueError(f"{self._name} requires a WAV recording")
        if on_speech is not None and not callable(on_speech):
            raise TypeError("on_speech must be a synchronous callable")
        async with self._lock:
            succeeded = False
            try:
                await self._ensure_worker()
                self._process.stdin.write(struct.pack("<I", len(wav)) + wav)
                await self._process.stdin.drain()
                speech_reported = False
                while True:
                    size = await self._packet_size()
                    try:
                        data = await self._process.stdout.readexactly(size)
                    except asyncio.IncompleteReadError:
                        raise ValueError(f"{self._name} returned an incomplete transcript packet") from None
                    try:
                        response = json.loads(data)
                    except (UnicodeDecodeError, ValueError):
                        raise ValueError(f"{self._name} returned an invalid transcript response") from None
                    if isinstance(response, dict) and "speech_detected" in response:
                        if response.keys() != {"speech_detected"} or response["speech_detected"] is not True:
                            raise ValueError(f"{self._name} returned an invalid speech notification")
                        if speech_reported:
                            raise ValueError(f"{self._name} returned a duplicate speech notification")
                        speech_reported = True
                        if on_speech is not None:
                            on_speech()
                        continue
                    text = response.get("text") if isinstance(response, dict) else None
                    if not isinstance(text, str):
                        raise ValueError(f"{self._name} returned an invalid transcript response")
                    succeeded = True
                    return text.strip()
            finally:
                if not succeeded:
                    await self._stop_worker()


class WhisperTranscriber(_LocalTranscriber):
    """Recognize English with a locally installed faster-whisper model."""


class MoonshineTranscriber(_LocalTranscriber):
    """Recognize complete English turns with the local Sherpa Moonshine model."""

    _backend = "moonshine"
    _name = "Moonshine"
    _required_files = ("preprocess.onnx", "encode.int8.onnx", "uncached_decode.int8.onnx",
                       "cached_decode.int8.onnx", "tokens.txt")


def _read_exact(source, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        part = source.read(size - len(data))
        if not part:
            raise EOFError("Whisper input packet is incomplete")
        data.extend(part)
    return bytes(data)


def _write_response(output, response: dict) -> None:
    data = json.dumps(response, ensure_ascii=False).encode("utf-8")
    if len(data) > MAX_TRANSCRIPT_BYTES:
        raise ValueError("Whisper transcript exceeds 64 KiB")
    output.write(struct.pack("<I", len(data)))
    output.write(data)
    output.flush()


def _transcribe_audio(model, tokenizer, suppressed_tokens, audio) -> str:
    """Decode a short turn without encoding thirty seconds of padding.

    Keep a second of padding and at least five seconds of context. Longer,
    uncertain, or repetitive turns use the complete decoder. A generous
    duration-based token budget bounds runaway repetition without accepting
    a truncated question.
    """
    import ctranslate2
    import numpy as np

    prompt = tokenizer.sot_sequence + [tokenizer.no_timestamps]
    token_budget = min(model.max_length - len(prompt), max(64, math.ceil(len(audio) / 16000 * 20)))
    if len(audio) <= 8 * 16000:
        features = model.feature_extractor(audio)
        frames = max(500, ((features.shape[-1] + 199) // 100) * 100)
        if frames <= 3000:
            features = np.pad(features, ((0, 0), (0, frames - features.shape[-1])))
            encoded = model.model.encode(ctranslate2.StorageView.from_array(
                np.ascontiguousarray(features[None, :], dtype=np.float32)))
            result = model.model.generate(
                encoded, [prompt], beam_size=1, max_length=len(prompt) + token_budget,
                suppress_blank=True, suppress_tokens=suppressed_tokens,
                return_scores=True, return_no_speech_prob=True,
                sampling_temperature=0.0,
            )[0]
            tokens = result.sequences_ids[0]
            valid_tokens = all(isinstance(token, int) and not isinstance(token, bool) and token >= 0
                               for token in tokens)
            text = tokenizer.decode(tokens).strip() if valid_tokens else ""
            average_logprob = result.scores[0] * len(tokens) / (len(tokens) + 1)
            encoded_text = text.encode("utf-8")
            compression_ratio = len(encoded_text) / len(zlib.compress(encoded_text))
            if (text and len(tokens) < token_budget
                    and math.isfinite(average_logprob) and average_logprob > -1.0
                    and math.isfinite(result.no_speech_prob) and 0 <= result.no_speech_prob <= 1
                    and compression_ratio <= 2.4):
                return text

    segments, _ = model.transcribe(audio, language="en", beam_size=3,
                                   condition_on_previous_text=False, temperature=(0.0, .2, .4),
                                   vad_filter=False, no_speech_threshold=0.6,
                                   log_prob_threshold=-1.0, compression_ratio_threshold=2.4,
                                   max_new_tokens=token_budget)
    parts = []
    for segment in segments:
        if (len(segment.tokens) >= token_budget or not math.isfinite(segment.avg_logprob)
                or not math.isfinite(segment.no_speech_prob)
                or not 0 <= segment.no_speech_prob <= 1
                or not math.isfinite(segment.compression_ratio) or segment.compression_ratio > 2.4):
            return ""
        if segment.text.strip():
            parts.append(segment.text.strip())
    return " ".join(parts)


def _worker(model_path: Path, backend: str = "whisper") -> None:
    from faster_whisper.audio import decode_audio
    from faster_whisper.vad import VadOptions, get_speech_timestamps, get_vad_model

    if backend == "moonshine":
        import sherpa_onnx
        model = sherpa_onnx.OfflineRecognizer.from_moonshine(
            preprocessor=str(model_path / "preprocess.onnx"),
            encoder=str(model_path / "encode.int8.onnx"),
            uncached_decoder=str(model_path / "uncached_decode.int8.onnx"),
            cached_decoder=str(model_path / "cached_decode.int8.onnx"),
            tokens=str(model_path / "tokens.txt"), num_threads=2, debug=False, provider="cpu")
    else:
        from faster_whisper import WhisperModel
        from faster_whisper.tokenizer import Tokenizer
        from faster_whisper.transcribe import get_suppressed_tokens

        model = WhisperModel(str(model_path), device="cpu", compute_type="int8",
                             cpu_threads=2, num_workers=1, local_files_only=True)
        tokenizer = Tokenizer(model.hf_tokenizer, model.model.is_multilingual,
                              task="transcribe", language="en")
        suppressed_tokens = get_suppressed_tokens(tokenizer, [-1])
    vad_options = VadOptions()
    get_vad_model()
    source = sys.stdin.buffer
    output = sys.stdout.buffer
    output.write(struct.pack("<I", 0))
    output.flush()
    while True:
        header = source.read(4)
        if not header:
            return
        if len(header) < 4:
            header += _read_exact(source, 4 - len(header))
        size = struct.unpack("<I", header)[0]
        if not 44 <= size <= MAX_WAV_BYTES:
            raise ValueError("Whisper input WAV size is invalid")
        wav = _read_exact(source, size)
        audio = decode_audio(io.BytesIO(wav), sampling_rate=16000)
        speech_chunks = get_speech_timestamps(audio, vad_options)
        if not speech_chunks:
            _write_response(output, {"text": ""})
            continue
        _write_response(output, {"speech_detected": True})
        if backend == "moonshine":
            # VAD gates silence; retain the original waveform so soft words
            # and the end of a long request cannot be removed by cropping.
            stream = model.create_stream()
            stream.accept_waveform(16000, audio)
            model.decode_stream(stream)
            text = stream.result.text.strip()
        else:
            text = _transcribe_audio(model, tokenizer, suppressed_tokens, audio)
        _write_response(output, {"text": text})


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Muse local transcription worker")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--backend", choices=("whisper", "moonshine"), default="whisper")
    args = parser.parse_args()
    _worker(args.model, args.backend)
