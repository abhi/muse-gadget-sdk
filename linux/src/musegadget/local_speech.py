# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Optional Piper speech synthesis, isolated in a cancellable child process."""

from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path
import struct
import sys
from typing import AsyncIterator, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class PreparedSpeech:
    """Synthesize ahead of playback with at most 2 MiB of float32 audio queued."""

    def __init__(self, stream):
        self._stream = stream
        self._chunks = asyncio.Queue(maxsize=128)
        self._task = asyncio.create_task(self._produce())

    async def _produce(self):
        try:
            async for chunk in self._stream:
                # Bound memory even if a speech backend yields a whole sentence.
                for offset in range(0, len(chunk), 4096):
                    await self._chunks.put(chunk[offset:offset + 4096].copy())
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._chunks.put(exc)
        finally:
            await self._stream.aclose()
        await self._chunks.put(None)

    def __aiter__(self):
        return self

    async def __anext__(self):
        chunk = await self._chunks.get()
        if chunk is None:
            raise StopAsyncIteration
        if isinstance(chunk, Exception):
            raise chunk
        return chunk

    def cancel(self):
        self._task.cancel()
        while not self._chunks.empty():
            self._chunks.get_nowait()
        # Wake a consumer already waiting for a canceled preparation.
        self._chunks.put_nowait(None)

    async def aclose(self):
        self.cancel()
        await asyncio.gather(self._task, return_exceptions=True)


class PiperSpeech:
    """Speak through an installed Piper ONNX voice and its JSON sidecar.

    One child process retains the loaded model. Conversation text goes through
    stdin; errors report the exit status without exposing Piper's stderr, which
    may contain input text. Close active streams before closing the backend.
    The caller supplies the playback and conversation timeout.
    """

    def __init__(self, model_path: Path):
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_file():
            raise ValueError(f"Piper voice model is missing: {self.model_path}")
        config_path = Path(str(self.model_path) + ".json")
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"Piper voice configuration is missing or invalid: {config_path}") from exc
        audio = config.get("audio") if isinstance(config, dict) else None
        rate = audio.get("sample_rate") if isinstance(audio, dict) else None
        if isinstance(rate, bool) or not isinstance(rate, int) or not 8000 <= rate <= 192_000:
            raise ValueError("Piper voice configuration needs a valid audio.sample_rate")
        self.sample_rate = rate
        self._process = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """Load the model once and wait until its worker is ready."""
        async with self._lock:
            try:
                await self._ensure_worker()
            except BaseException:
                await self._stop_worker()
                raise

    async def close(self) -> None:
        """Kill and reap the worker after the conversation has stopped."""
        async with self._lock:
            await self._stop_worker()

    def prepare(self, text: str, output_rate: int) -> PreparedSpeech:
        """Prepare one upcoming sentence while the speaker plays the previous one."""
        return PreparedSpeech(self.stream(text, output_rate))

    async def _ensure_worker(self) -> None:
        if self._process is not None and self._process.returncode is None:
            return
        await self._stop_worker()
        spawn = asyncio.create_task(asyncio.create_subprocess_exec(
            sys.executable, "-m", "musegadget.local_speech", "--model", str(self.model_path),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        ))
        try:
            self._process = await asyncio.shield(spawn)
        except asyncio.CancelledError:
            self._process = await spawn
            raise
        if await self._packet_size(self._process) != 0:
            raise ValueError("Piper worker returned an invalid ready handshake")

    async def _stop_worker(self) -> None:
        process, self._process = self._process, None
        if process is not None:
            process.stdin.close()
            if process.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    process.kill()
            # Drain stdout after killing the worker. A full asyncio pipe can
            # otherwise keep wait() blocked even after the child has exited.
            await process.communicate()

    @staticmethod
    async def _packet_size(process) -> int:
        try:
            header = await process.stdout.readexactly(4)
        except asyncio.IncompleteReadError:
            status = await process.wait()
            raise RuntimeError(f"Piper worker ended with exit status {status}; check the model and installation") from None
        size = struct.unpack("<I", header)[0]
        if size > 65536:
            raise ValueError("Piper audio packet exceeds 64 KiB")
        return size

    async def stream(self, text: str, output_rate: int) -> AsyncIterator[np.ndarray]:
        """Yield contiguous mono float32 chunks resampled to the speaker rate."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Piper speech requires nonempty text")
        if isinstance(output_rate, bool) or not isinstance(output_rate, int) or output_rate <= 0:
            raise ValueError("output_rate must be a positive integer")
        try:
            import av
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("Local speech requires the Muse Reachy audio dependencies") from exc
        async with self._lock:
            succeeded = False
            try:
                await self._ensure_worker()
                process = self._process
                process.stdin.write((json.dumps({"text": text}, ensure_ascii=False) + "\n").encode("utf-8"))
                await process.stdin.drain()
                resampler = av.AudioResampler(format="fltp", layout="mono", rate=output_rate)
                pending = bytearray()
                samples_emitted = 0
                while True:
                    size = await self._packet_size(process)
                    if not size:
                        break
                    try:
                        pending.extend(await process.stdout.readexactly(size))
                    except asyncio.IncompleteReadError:
                        raise ValueError("Piper returned an incomplete audio packet") from None
                    complete = len(pending) // 2 * 2
                    if not complete:
                        continue
                    pcm = bytes(pending[:complete])
                    del pending[:complete]
                    samples = np.frombuffer(pcm, dtype="<i2").reshape(1, -1)
                    frame = av.AudioFrame.from_ndarray(samples, format="s16", layout="mono")
                    frame.sample_rate = self.sample_rate
                    for output in resampler.resample(frame):
                        samples = np.ascontiguousarray(output.to_ndarray().reshape(-1))
                        samples_emitted += len(samples)
                        yield samples
                if pending:
                    raise ValueError("Piper returned an incomplete PCM16 sample")
                for output in resampler.resample(None):
                    samples = np.ascontiguousarray(output.to_ndarray().reshape(-1))
                    samples_emitted += len(samples)
                    yield samples
                if not samples_emitted:
                    raise RuntimeError("Piper returned no speech audio")
                succeeded = True
            finally:
                if not succeeded:
                    await self._stop_worker()


def _worker(model_path: Path) -> None:
    import onnxruntime
    from piper import PiperVoice
    from piper.config import PiperConfig

    config = PiperConfig.from_dict(json.loads(Path(str(model_path) + ".json").read_text(encoding="utf-8")))
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 2
    voice = PiperVoice(config=config, session=onnxruntime.InferenceSession(
        str(model_path), sess_options=options, providers=["CPUExecutionProvider"],
    ))
    output = sys.stdout.buffer
    output.write(struct.pack("<I", 0))
    output.flush()
    for line in sys.stdin:
        request = json.loads(line)
        text = request["text"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Piper worker requires nonempty text")
        for chunk in voice.synthesize(text):
            audio = chunk.audio_int16_bytes
            for offset in range(0, len(audio), 4096):
                packet = audio[offset:offset + 4096]
                output.write(struct.pack("<I", len(packet)))
                output.write(packet)
                output.flush()
        output.write(struct.pack("<I", 0))
        output.flush()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Muse local speech worker")
    parser.add_argument("--model", required=True, type=Path)
    _worker(parser.parse_args().model)
