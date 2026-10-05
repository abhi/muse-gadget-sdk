# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

from __future__ import annotations

import asyncio
import json
import sys

import pytest

np = pytest.importorskip("numpy")
av = pytest.importorskip("av")

from musegadget import local_speech
from musegadget.local_speech import PiperSpeech


@pytest.fixture
def model(tmp_path):
    model_path = tmp_path / "voice.onnx"
    model_path.write_bytes(b"fake model for the worker")
    model_path.with_suffix(".onnx.json").write_text(json.dumps({"audio": {"sample_rate": 22050}}))
    return model_path


@pytest.fixture
def worker(tmp_path, monkeypatch):
    """A real child process with PCM output, split samples, and error modes."""
    script = tmp_path / "speech_worker.py"
    script.write_text("""
import json, math, os, struct, sys, time
from pathlib import Path
mode = os.environ.get('PIPER_TEST_MODE', 'normal')
output = sys.stdout.buffer
output.write(struct.pack('<I', 0))
output.flush()
for line in sys.stdin:
    text = json.loads(line)['text']
    Path(os.environ['PIPER_TEST_INPUT']).write_text(text)
    if mode == 'fail':
        sys.stderr.write(text)
        sys.exit(7)
    pcm = b''.join(struct.pack('<h', round(12000 * math.sin(2 * math.pi * 440 * i / 22050))) for i in range(2205))
    if mode == 'empty':
        pcm = b''
    if mode == 'stall':
        output.write(struct.pack('<I', 512) + pcm[:512])
        output.flush()
        time.sleep(3600)
    if mode == 'flood':
        while True:
            output.write(struct.pack('<I', 512) + pcm[:512])
            output.flush()
    if mode == 'odd':
        pcm = pcm[:-1]
    if mode == 'oversized':
        output.write(struct.pack('<I', 65537))
        output.flush()
        time.sleep(3600)
    for offset in range(0, len(pcm), 97):
        packet = pcm[offset:offset + 97]
        output.write(struct.pack('<I', len(packet)) + packet)
        output.flush()
        time.sleep(.001)
    output.write(struct.pack('<I', 0))
    output.flush()
""")
    input_path = tmp_path / "worker_input.txt"
    monkeypatch.setenv("PIPER_TEST_INPUT", str(input_path))
    original = asyncio.create_subprocess_exec
    calls = []
    processes = []

    async def create(*args, **kwargs):
        calls.append((args, kwargs))
        process = await original(sys.executable, str(script), **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(local_speech.asyncio, "create_subprocess_exec", create)
    return calls, processes, input_path


async def collect(speech, text="Hello Muse.", rate=48000, *, close=True):
    try:
        arrays = [chunk async for chunk in speech.stream(text, rate)]
        assert all(chunk.ndim == 1 and chunk.dtype == np.float32 and chunk.flags.c_contiguous
                   for chunk in arrays)
        return np.concatenate(arrays)
    finally:
        if close:
            await speech.close()


def test_real_process_streams_resampled_pcm_and_keeps_text_off_command_line(model, worker):
    calls, processes, input_path = worker
    text = "A private Muse reply.\nUnicode works too: café."

    actual = asyncio.run(collect(PiperSpeech(model), text))

    assert calls[0][0] == (sys.executable, "-m", "musegadget.local_speech", "--model", str(model))
    assert calls[0][1]["stderr"] == asyncio.subprocess.DEVNULL
    assert input_path.read_text() == text
    assert processes[0].returncode == -9
    assert len(actual) == 4800
    assert 0.35 < np.max(np.abs(actual)) < 0.38
    assert np.max(np.abs(np.diff(actual))) < 0.03

    source = np.rint(12000 * np.sin(2 * np.pi * 440 * np.arange(2205) / 22050)).astype(np.int16)
    frame = av.AudioFrame.from_ndarray(source[None, :], format="s16", layout="mono")
    frame.sample_rate = 22050
    resampler = av.AudioResampler(format="fltp", layout="mono", rate=48000)
    expected = np.concatenate([output.to_ndarray().reshape(-1)
                               for output in resampler.resample(frame) + resampler.resample(None)])
    np.testing.assert_array_equal(actual, expected)


def test_process_failure_reports_status_without_private_text(model, worker, monkeypatch, capsys):
    monkeypatch.setenv("PIPER_TEST_MODE", "fail")
    text = "This private reply must never appear in a diagnostic."
    with pytest.raises(RuntimeError, match="exit status 7") as error:
        asyncio.run(collect(PiperSpeech(model), text))
    assert text not in str(error.value)
    output = capsys.readouterr()
    assert text not in output.out + output.err
    assert worker[1][0].returncode == 7


@pytest.mark.parametrize("mode, error, message", [
    ("empty", RuntimeError, "no speech audio"),
    ("odd", ValueError, "incomplete PCM16"),
    ("oversized", ValueError, "exceeds 64 KiB"),
])
def test_invalid_process_audio_is_reported(model, worker, monkeypatch, mode, error, message):
    monkeypatch.setenv("PIPER_TEST_MODE", mode)
    with pytest.raises(error, match=message):
        asyncio.run(collect(PiperSpeech(model)))
    assert worker[1][0].returncode == -9


def test_start_warms_one_worker_and_reuses_it_for_sequential_replies(model, worker):
    async def scenario():
        speech = PiperSpeech(model)
        await speech.start()
        await speech.start()
        assert len(worker[1]) == 1
        assert worker[1][0].returncode is None
        first = await collect(speech, "First reply.", close=False)
        second = await collect(speech, "Second reply.", close=False)
        assert len(worker[1]) == 1
        assert worker[1][0].returncode is None
        np.testing.assert_array_equal(first, second)
        assert worker[2].read_text() == "Second reply."
        await speech.close()
        await speech.close()
        assert worker[1][0].returncode == -9

    asyncio.run(scenario())


def test_concurrent_requests_are_serialized_through_one_worker(model, worker):
    async def scenario():
        speech = PiperSpeech(model)
        first, second = await asyncio.gather(
            collect(speech, "First reply.", close=False),
            collect(speech, "Second reply.", close=False),
        )
        assert len(worker[1]) == 1
        np.testing.assert_array_equal(first, second)
        await speech.close()

    asyncio.run(scenario())


def test_cancellation_kills_and_reaps_a_stalled_synthesizer(model, worker, monkeypatch):
    monkeypatch.setenv("PIPER_TEST_MODE", "stall")

    async def scenario():
        stream = PiperSpeech(model).stream("Hello Muse.", 48000)
        first = await asyncio.wait_for(stream.__anext__(), 3)
        assert len(first) > 0
        pending = asyncio.create_task(stream.__anext__())
        await asyncio.sleep(0.02)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert worker[1][0].returncode == -9
        await stream.aclose()

    asyncio.run(scenario())


def test_closing_a_paused_stream_kills_its_process(model, worker, monkeypatch):
    monkeypatch.setenv("PIPER_TEST_MODE", "stall")

    async def scenario():
        stream = PiperSpeech(model).stream("Hello Muse.", 48000)
        await asyncio.wait_for(stream.__anext__(), 3)
        await stream.aclose()
        assert worker[1][0].returncode == -9

    asyncio.run(scenario())


def test_next_reply_restarts_after_cancellation(model, worker, monkeypatch):
    monkeypatch.setenv("PIPER_TEST_MODE", "stall")

    async def scenario():
        speech = PiperSpeech(model)
        stream = speech.stream("Cancelled reply.", 48000)
        await asyncio.wait_for(stream.__anext__(), 3)
        await stream.aclose()
        assert worker[1][0].returncode == -9
        monkeypatch.setenv("PIPER_TEST_MODE", "normal")
        samples = await collect(speech, "New reply.")
        assert len(samples) == 4800
        assert len(worker[1]) == 2
        assert worker[1][1].returncode == -9

    asyncio.run(scenario())


def test_closing_a_stream_with_full_stdout_pipe_still_reaps_its_process(model, worker, monkeypatch):
    monkeypatch.setenv("PIPER_TEST_MODE", "flood")

    async def scenario():
        stream = PiperSpeech(model).stream("A longer reply.", 48000)
        await asyncio.wait_for(stream.__anext__(), 3)
        await asyncio.sleep(0.1)
        await asyncio.wait_for(stream.aclose(), 3)
        assert worker[1][0].returncode == -9

    asyncio.run(scenario())


def test_cancellation_during_spawn_still_reaps_the_process(model, worker, monkeypatch):
    monkeypatch.setenv("PIPER_TEST_MODE", "stall")
    original = local_speech.asyncio.create_subprocess_exec
    spawned = asyncio.Event()
    release = asyncio.Event()

    async def delayed(*args, **kwargs):
        process = await original(*args, **kwargs)
        spawned.set()
        await release.wait()
        return process

    monkeypatch.setattr(local_speech.asyncio, "create_subprocess_exec", delayed)

    async def scenario():
        stream = PiperSpeech(model).stream("Hello Muse.", 48000)
        pending = asyncio.create_task(stream.__anext__())
        await asyncio.wait_for(spawned.wait(), 3)
        pending.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert worker[1][0].returncode == -9

    asyncio.run(scenario())


@pytest.mark.parametrize("config", [{}, [], {"audio": []}, {"audio": {"sample_rate": 0}},
                                    {"audio": {"sample_rate": True}}])
def test_invalid_voice_metadata_is_rejected(model, config):
    model.with_suffix(".onnx.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="audio.sample_rate"):
        PiperSpeech(model)


def test_missing_model_and_sidecar_are_reported(model):
    with pytest.raises(ValueError, match="model is missing"):
        PiperSpeech(model.with_name("absent.onnx"))
    model.with_suffix(".onnx.json").unlink()
    with pytest.raises(ValueError, match="configuration is missing"):
        PiperSpeech(model)


def test_prepared_sentences_finish_synthesis_before_playback_consumes_first_audio(model, worker):
    async def scenario():
        speech = PiperSpeech(model)
        first = second = None
        try:
            await speech.start()
            process = speech._process
            first = speech.prepare("First sentence.", 48000)
            second = speech.prepare("Second sentence.", 48000)
            # Neither sentence has been consumed by the speaker yet.
            await asyncio.wait_for(asyncio.gather(first._task, second._task), 3)
            audio = [np.concatenate([chunk async for chunk in prepared]) for prepared in (first, second)]
            assert all(len(samples) == 4800 for samples in audio)
            assert speech._process is process and process.returncode is None
            assert worker[2].read_text() == "Second sentence."
        finally:
            await asyncio.gather(*(prepared.aclose() for prepared in (first, second) if prepared is not None))
            await speech.close()
    asyncio.run(scenario())


def test_prepared_audio_is_bounded_and_cancellation_closes_its_producer():
    async def scenario():
        closed = asyncio.Event()

        async def endless_audio():
            try:
                while True:
                    yield np.ones(8192, dtype=np.float32)
            finally:
                closed.set()

        prepared = local_speech.PreparedSpeech(endless_audio())
        while not prepared._chunks.full():
            await asyncio.sleep(0)
        assert prepared._chunks.qsize() == 128
        samples = [prepared._chunks.get_nowait() for _ in range(128)]
        assert sum(chunk.nbytes for chunk in samples) == 2 * 1024 * 1024
        assert not prepared._task.done()
        await prepared.aclose()
        assert prepared._task.done() and closed.is_set()
    asyncio.run(scenario())


def test_prepared_worker_failure_reaches_playback_and_the_next_request_recovers(model, worker, monkeypatch):
    async def scenario():
        speech = PiperSpeech(model)
        prepared = None
        try:
            monkeypatch.setenv("PIPER_TEST_MODE", "fail")
            prepared = speech.prepare("Private failed sentence.", 48000)
            with pytest.raises(RuntimeError, match="exit status 7"):
                await asyncio.wait_for(prepared.__anext__(), 3)
            await prepared.aclose()
            assert worker[1][0].returncode == 7
            monkeypatch.setenv("PIPER_TEST_MODE", "normal")
            prepared = speech.prepare("Recovered sentence.", 48000)
            samples = np.concatenate([chunk async for chunk in prepared])
            assert len(samples) == 4800 and speech._process.returncode is None
        finally:
            if prepared is not None:
                await prepared.aclose()
            await speech.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("text, rate", [("", 48000), ("   ", 48000), (None, 48000),
                                      ("Hello", 0), ("Hello", True)])
def test_invalid_stream_inputs_do_not_start_a_child(model, worker, text, rate):
    with pytest.raises(ValueError):
        asyncio.run(collect(PiperSpeech(model), text, rate))
    assert worker[0] == []


def test_production_worker_keeps_voice_config_and_reuses_two_thread_cpu_session(model, tmp_path, monkeypatch):
    observations = tmp_path / "session_calls.json"
    config_path = tmp_path / "voice_config.json"
    voice_config = {"audio": {"sample_rate": 22050}, "espeak": {"voice": "en-us"},
                    "phoneme_type": "espeak", "phoneme_id_map": {"a": [12]},
                    "inference": {"noise_scale": 0.667, "length_scale": 1.0}}
    model.with_suffix(".onnx.json").write_text(json.dumps(voice_config))
    (tmp_path / "onnxruntime.py").write_text("""
import json, os
from pathlib import Path
class SessionOptions:
    def __init__(self):
        self.intra_op_num_threads = 0
class InferenceSession:
    def __init__(self, model, sess_options, providers):
        self.configured = True
        Path(os.environ['PIPER_SESSION_CALLS']).write_text(json.dumps({
            'model': model, 'threads': sess_options.intra_op_num_threads, 'providers': providers}))
""")
    package = tmp_path / "piper"
    package.mkdir()
    (package / "config.py").write_text("""
class PiperConfig:
    @staticmethod
    def from_dict(config):
        return config
""")
    (package / "__init__.py").write_text("""
import json, os
from pathlib import Path
from types import SimpleNamespace
class PiperVoice:
    def __init__(self, config, session):
        assert session.configured
        Path(os.environ['PIPER_VOICE_CONFIG']).write_text(json.dumps(config))
    def synthesize(self, text):
        yield SimpleNamespace(audio_int16_bytes=b'\\0\\x20' * 2205)
""")
    monkeypatch.setenv("PIPER_SESSION_CALLS", str(observations))
    monkeypatch.setenv("PIPER_VOICE_CONFIG", str(config_path))
    monkeypatch.setenv("PYTHONPATH", str(tmp_path) + ":" + ":".join(sys.path))

    async def scenario():
        speech = PiperSpeech(model)
        try:
            await speech.start()
            pid = speech._process.pid
            first = await collect(speech, "Hello Muse.", 16000, close=False)
            second = await collect(speech, "Another reply.", 16000, close=False)
            assert speech._process.pid == pid
            assert len(first) == 1600 and float(np.max(np.abs(first))) == pytest.approx(.25)
            np.testing.assert_array_equal(first, second)
        finally:
            await speech.close()
    asyncio.run(scenario())
    assert json.loads(observations.read_text()) == {"model": str(model), "threads": 2,
                                                  "providers": ["CPUExecutionProvider"]}
    assert json.loads(config_path.read_text()) == voice_config
