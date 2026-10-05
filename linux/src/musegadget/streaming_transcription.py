# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Incremental, local Sherpa ONNX transcription with bounded PCM queues.

Only the child process imports Sherpa and mutates its model/streams. Callers
feed authorized mono PCM16 at 16 kHz and decide the utterance endpoint. Partial
hypotheses stay in RAM; native endpoint detection is disabled.
"""

from __future__ import annotations

import asyncio
from collections import deque
import contextlib
from dataclasses import dataclass
import math
from pathlib import Path
import struct
import sys

SAMPLE_RATE = 16000
MAX_TURN_SAMPLES = 60 * SAMPLE_RATE
MAX_PACKET_BYTES = 64 * 1024
MODEL_FILES = (
    "encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
    "decoder-epoch-99-avg-1-chunk-16-left-128.onnx",
    "joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
    "tokens.txt",
)
_COMMAND_HEADER = struct.Struct("<BII")
_OPEN, _FEED, _FINISH, _ABORT = range(1, 5)


class StreamingTranscriptionError(RuntimeError):
    """The whole utterance failed; no partial result may be submitted."""


class StreamingBufferOverflow(StreamingTranscriptionError):
    """Inference did not keep up with the bounded microphone queue."""


class StreamingWorkerError(StreamingTranscriptionError):
    """The model process failed; restart it before accepting another turn."""


@dataclass(frozen=True)
class TranscriptRevision:
    revision: int
    text: str


@dataclass(frozen=True)
class _Command:
    operation: int
    turn: StreamingTurn
    pcm: bytes = b""


class StreamingTurn:
    """An event-loop-owned utterance. finish seals input without waiting."""

    def __init__(self, owner: SherpaStreamingTranscriber, turn_id: int):
        self._owner = owner
        self._id = turn_id
        self._future = asyncio.get_running_loop().create_future()
        self._future.add_done_callback(self._completed)
        self._sealed = False
        self._discarded = False
        self._opened = False
        self._samples = 0
        self._partial = TranscriptRevision(0, "")

    @property
    def partial(self) -> TranscriptRevision:
        return self._partial

    def _completed(self, future) -> None:
        if future.cancelled():
            self.abort()
        else:
            # A failed feed raises synchronously too; still retain the exception
            # for finish/await without an unobserved-Future warning.
            future.exception()

    def feed(self, pcm16_16khz: bytes) -> None:
        """Queue PCM immediately. Overflow invalidates this entire utterance."""
        self._owner._check_loop()
        if self._sealed or self._discarded:
            raise StreamingTranscriptionError("Streaming turn is no longer accepting audio")
        if not isinstance(pcm16_16khz, bytes) or not pcm16_16khz or len(pcm16_16khz) % 2:
            error = StreamingTranscriptionError("Streaming audio requires complete nonempty PCM16 samples")
            self._owner._discard(self, error)
            raise error
        samples = len(pcm16_16khz) // 2
        if self._samples + samples > MAX_TURN_SAMPLES:
            error = StreamingTranscriptionError("Streaming utterance exceeds sixty seconds")
        elif self._owner._pending_samples + samples > self._owner._max_samples:
            error = StreamingBufferOverflow("Streaming audio queue exceeded its bounded capacity")
        elif len(self._owner._commands) >= 256:
            error = StreamingBufferOverflow("Streaming command queue exceeded its bounded capacity")
        else:
            self._samples += samples
            self._owner._pending_samples += samples
            self._owner._enqueue(_Command(_FEED, self, pcm16_16khz))
            return
        self._owner._discard(self, error)
        raise error

    def finish(self) -> asyncio.Future[str]:
        """Seal now; final text arrives after earlier feeds and native flush."""
        self._owner._check_loop()
        if not self._sealed and not self._discarded:
            self._sealed = True
            self._owner._enqueue(_Command(_FINISH, self))
        return self._future

    def abort(self) -> None:
        """Discard PCM/hypotheses. A pending native result cannot revive it."""
        self._owner._check_loop()
        self._owner._discard(self)


class SherpaStreamingTranscriber:
    """One persistent model process and independently sealed turn handles.

The shared three-second budget includes in-flight, unacknowledged PCM, not
just waiting Python commands. Native calls run sequentially. A later turn can
accept audio while an earlier turn finalizes, but its work queues behind that
call. A hung call kills/reaps the process and fails every affected turn.
"""

    def __init__(self, model_path: Path, *, max_buffer_s: float = 3.0, max_turns: int = 8,
                 startup_timeout_s: float = 30.0, command_timeout_s: float = 10.0):
        self.model_path = Path(model_path).expanduser().resolve()
        if any(not (self.model_path / name).is_file() or not (self.model_path / name).stat().st_size
               for name in MODEL_FILES):
            raise ValueError("Sherpa streaming model directory is missing required local assets")
        if not math.isfinite(max_buffer_s) or not 0 < max_buffer_s <= 3:
            raise ValueError("Streaming buffer must hold at most three seconds")
        if isinstance(max_turns, bool) or not isinstance(max_turns, int) or not 1 <= max_turns <= 8:
            raise ValueError("Streaming transcriber accepts between one and eight turns")
        if any(not math.isfinite(value) or value <= 0
               for value in (startup_timeout_s, command_timeout_s)):
            raise ValueError("Streaming worker timeouts must be positive and finite")
        self._max_samples = int(max_buffer_s * SAMPLE_RATE)
        self._max_turns = max_turns
        self._startup_timeout_s = startup_timeout_s
        self._command_timeout_s = command_timeout_s
        self._lifecycle_lock = asyncio.Lock()
        self._process = None
        self._pump = None
        self._accepting = False
        self._loop = None
        self._commands = deque()
        self._pending_samples = 0
        self._inflight = None
        self._turns = {}
        self._next_id = 0
        self._wake = asyncio.Event()

    def _check_loop(self) -> None:
        if asyncio.get_running_loop() is not self._loop:
            raise RuntimeError("Streaming transcription must stay on its owning event loop")

    async def start(self) -> None:
        async with self._lifecycle_lock:
            if self._pump is not None and not self._pump.done():
                self._check_loop()
                if self._accepting:
                    return
                await self._pump
            self._loop = asyncio.get_running_loop()
            spawn = asyncio.create_task(asyncio.create_subprocess_exec(
                sys.executable, "-m", "musegadget.streaming_transcription", "--model",
                str(self.model_path),
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL))
            try:
                try:
                    self._process = await asyncio.shield(spawn)
                except asyncio.CancelledError:
                    self._process = await spawn
                    raise
                ready = await asyncio.wait_for(self._read_response(), self._startup_timeout_s)
                if ready != {"ready": True}:
                    raise StreamingTranscriptionError("Streaming worker returned an invalid readiness packet")
                self._accepting = True
                self._pump = asyncio.create_task(self._run_commands())
            except BaseException:
                await self._stop_process()
                raise

    def open_turn(self) -> StreamingTurn:
        self._check_loop()
        if not self._accepting or self._pump is None or self._pump.done():
            raise StreamingWorkerError("Streaming worker is not ready")
        if len(self._turns) >= self._max_turns:
            raise StreamingBufferOverflow("Streaming utterance queue is full")
        self._next_id += 1
        turn = StreamingTurn(self, self._next_id)
        self._turns[turn._id] = turn
        self._enqueue(_Command(_OPEN, turn))
        return turn

    def _enqueue(self, command: _Command) -> None:
        self._commands.append(command)
        self._wake.set()

    def _discard(self, turn: StreamingTurn, error=None) -> None:
        if turn._discarded:
            return
        turn._discarded = turn._sealed = True
        turn._partial = TranscriptRevision(turn._partial.revision + 1, "")
        retained = deque()
        for command in self._commands:
            if command.turn is turn:
                self._pending_samples -= len(command.pcm) // 2
            else:
                retained.append(command)
        self._commands = retained
        self._turns.pop(turn._id, None)
        if turn._opened or self._inflight is not None and self._inflight.turn is turn:
            self._enqueue(_Command(_ABORT, turn))
        if not turn._future.done():
            if error is None:
                turn._future.cancel()
            else:
                turn._future.set_exception(error)

    async def _read_response(self):
        import json

        try:
            header = await self._process.stdout.readexactly(4)
            size = struct.unpack("<I", header)[0]
            if not 0 < size <= MAX_PACKET_BYTES:
                raise StreamingTranscriptionError("Streaming worker response exceeds its bounded packet size")
            response = json.loads(await self._process.stdout.readexactly(size))
        except (asyncio.IncompleteReadError, ValueError, UnicodeDecodeError):
            raise StreamingTranscriptionError("Streaming worker returned an incomplete or invalid packet") from None
        if not isinstance(response, dict):
            raise StreamingTranscriptionError("Streaming worker returned an invalid response")
        return response

    async def _exchange(self, command: _Command):
        self._process.stdin.write(_COMMAND_HEADER.pack(
            command.operation, command.turn._id, len(command.pcm)) + command.pcm)
        await self._process.stdin.drain()
        response = await self._read_response()
        if response.get("turn") != command.turn._id:
            raise StreamingTranscriptionError("Streaming worker returned a mismatched utterance")
        if response == {"turn": command.turn._id, "error": True}:
            return None
        expected = {"turn", "text"} if command.operation in {_FEED, _FINISH} else {"turn"}
        if response.keys() != expected or "text" in response and not isinstance(response["text"], str):
            raise StreamingTranscriptionError("Streaming worker returned invalid utterance data")
        return response

    async def _run_commands(self) -> None:
        try:
            while True:
                await self._wake.wait()
                self._wake.clear()
                while self._commands:
                    command = self._commands.popleft()
                    self._inflight = command
                    try:
                        response = await asyncio.wait_for(self._exchange(command), self._command_timeout_s)
                    finally:
                        self._pending_samples -= len(command.pcm) // 2
                        self._inflight = None
                    turn = command.turn
                    if response is None:
                        self._discard(turn, StreamingTranscriptionError("Streaming model failed this utterance"))
                    elif not turn._discarded:
                        if command.operation == _OPEN:
                            turn._opened = True
                        elif command.operation in {_FEED, _FINISH}:
                            text = response["text"].strip()
                            if text != turn.partial.text:
                                turn._partial = TranscriptRevision(turn.partial.revision + 1, text)
                            if command.operation == _FINISH:
                                self._turns.pop(turn._id, None)
                                if not turn._future.done():
                                    turn._future.set_result(text)
        except asyncio.CancelledError:
            raise
        except Exception:
            self._accepting = False
            # Keep native exceptions and transcript-bearing stderr private.
            for turn in list(self._turns.values()):
                self._discard(turn, StreamingWorkerError("Streaming worker failed or exceeded its time limit"))
        finally:
            self._accepting = False
            self._commands.clear()
            self._pending_samples = 0
            await self._stop_process()

    async def _stop_process(self) -> None:
        process, self._process = self._process, None
        if process is not None:
            process.stdin.close()
            if process.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    process.kill()
            reap = asyncio.create_task(process.communicate())
            try:
                await asyncio.shield(reap)
            except asyncio.CancelledError:
                await reap
                raise

    async def close(self) -> None:
        async with self._lifecycle_lock:
            self._accepting = False
            for turn in list(self._turns.values()):
                self._discard(turn)
            pump, self._pump = self._pump, None
            if pump is not None:
                pump.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await pump
            await self._stop_process()


class _SherpaTurn:
    def __init__(self, model):
        self.model = model
        self.stream = model.create_stream()

    def text(self) -> str:
        while self.model.is_ready(self.stream):
            self.model.decode_stream(self.stream)
        text = self.model.get_result(self.stream)
        if not isinstance(text, str):
            raise StreamingTranscriptionError("Streaming model returned an invalid hypothesis")
        if len(text.encode("utf-8")) > MAX_PACKET_BYTES - 1024:
            raise StreamingTranscriptionError("Streaming transcript exceeds its bounded capacity")
        return text.strip()

    def feed(self, pcm: bytes) -> str:
        import numpy as np

        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
        self.stream.accept_waveform(SAMPLE_RATE, audio)
        return self.text()

    def finish(self) -> str:
        import numpy as np

        # InputFinished does not pad an incomplete encoder chunk. The ordinary
        # endpoint has silence, but a sixty-second cap can end during a word.
        # Supply native lookahead without adding a microphone listening delay.
        self.stream.accept_waveform(SAMPLE_RATE, np.zeros(int(.66 * SAMPLE_RATE), np.float32))
        self.stream.input_finished()
        return self.text()

    def close(self) -> None:
        # OnlineStream has no close API; release its native reference here.
        self.stream = self.model = None


def _load_model(model_path: Path):
    from importlib.metadata import version

    if version("sherpa-onnx") != "1.13.8":
        raise StreamingTranscriptionError("Streaming adapter requires Sherpa ONNX 1.13.8")
    import sherpa_onnx

    return sherpa_onnx.OnlineRecognizer.from_transducer(
        **{name: str(model_path / path) for name, path in
           zip(("encoder", "decoder", "joiner", "tokens"), MODEL_FILES)},
        num_threads=1, sample_rate=SAMPLE_RATE, feature_dim=80,
        decoding_method="modified_beam_search", max_active_paths=4,
        enable_endpoint_detection=False, provider="cpu", debug=False,
        model_type="zipformer2",
    )


def _worker(model_path: Path, *, model=None, source=None, output=None) -> None:
    import json

    model = model if model is not None else _load_model(model_path)
    source = source if source is not None else sys.stdin.buffer
    output = output if output is not None else sys.stdout.buffer
    turns = {}

    def send(response):
        data = json.dumps(response, ensure_ascii=False).encode("utf-8")
        if len(data) > MAX_PACKET_BYTES:
            raise StreamingTranscriptionError("Streaming response exceeds its bounded capacity")
        output.write(struct.pack("<I", len(data)) + data)
        output.flush()

    def read_exact(size):
        data = bytearray()
        while len(data) < size:
            part = source.read(size - len(data))
            if not part:
                raise EOFError
            data.extend(part)
        return bytes(data)

    try:
        send({"ready": True})
        while True:
            first = source.read(1)
            if not first:
                return
            operation, turn_id, size = _COMMAND_HEADER.unpack(first + read_exact(_COMMAND_HEADER.size - 1))
            if operation not in {_OPEN, _FEED, _FINISH, _ABORT} or not turn_id or size > 3 * SAMPLE_RATE * 2:
                raise StreamingTranscriptionError("Streaming input packet is invalid")
            if operation != _FEED and size or operation == _FEED and (not size or size % 2):
                raise StreamingTranscriptionError("Streaming input PCM packet is invalid")
            pcm = read_exact(size)
            try:
                response = {"turn": turn_id}
                if operation == _OPEN:
                    if turn_id in turns or len(turns) >= 8:
                        raise StreamingTranscriptionError("Streaming model utterance queue is full")
                    turns[turn_id] = _SherpaTurn(model)
                elif operation == _FEED:
                    response["text"] = turns[turn_id].feed(pcm)
                elif operation == _FINISH:
                    response["text"] = turns[turn_id].finish()
                if operation in {_FINISH, _ABORT}:
                    turn = turns.pop(turn_id, None)
                    if turn is not None:
                        turn.close()
                send(response)
            except Exception:
                turn = turns.pop(turn_id, None)
                if turn is not None:
                    with contextlib.suppress(Exception):
                        turn.close()
                send({"turn": turn_id, "error": True})
    finally:
        for turn in turns.values():
            with contextlib.suppress(Exception):
                turn.close()
        turns.clear()
        model = None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    args = parser.parse_args()
    _worker(args.model)
