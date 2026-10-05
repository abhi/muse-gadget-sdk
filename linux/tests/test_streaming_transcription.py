# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

from __future__ import annotations

import asyncio
import struct
import sys
from types import SimpleNamespace

import pytest

from musegadget import streaming_transcription as streaming
from musegadget.streaming_transcription import (
    SherpaStreamingTranscriber, StreamingBufferOverflow, StreamingTranscriptionError,
    StreamingWorkerError,
)


def pcm(value=1, samples=1600):
    return struct.pack("<h", value) * samples


@pytest.fixture
def model(tmp_path):
    path = tmp_path / "model"
    path.mkdir()
    for name in streaming.MODEL_FILES:
        (path / name).write_bytes(b"public fake model")
    return path


@pytest.fixture
def worker(tmp_path, monkeypatch):
    """Run the actual framed worker with a fake streaming model in a child."""
    script = tmp_path / "public_model.py"
    script.write_text("""
import os, sys, time
from pathlib import Path
import numpy
from musegadget.streaming_transcription import _worker

if os.environ.get('STREAM_TEST_BOOTSTRAPPED'):
    Path(os.environ['STREAM_TEST_BOOTSTRAPPED']).touch()
if os.environ.get('STREAM_TEST_STARTUP') == 'stall':
    Path(os.environ['STREAM_TEST_ENTERED']).touch()
    time.sleep(3600)
if os.environ.get('STREAM_TEST_STARTUP') == 'fail':
    sys.stderr.write('private native error must not escape')
    sys.exit(7)

start_failed = False

class Stream:
    def __init__(self):
        self.text = ''
        self.pending = []
        self.tail_waiting = False
    def accept_waveform(self, rate, audio):
        assert rate == 16000
        value = round(audio[0] * 32768)
        if value == 8:
            assert len(audio) == 1600 and (audio == 8 / 32768).all()
        if self.tail_waiting and value == 0:
            assert len(audio) == 10560 and not audio.any()
            self.tail_waiting = False
            self.pending.append(10)
        if value == 7:
            Path(os.environ['STREAM_TEST_ENTERED']).touch()
            time.sleep(3600)
        if value == 9: raise RuntimeError('private text must not escape')
        self.pending.append(value)
    def input_finished(self): self.pending.append('final')

class Model:
    def create_stream(self):
        global start_failed
        if os.environ.get('STREAM_TEST_STREAM_START') == 'fail_once' and not start_failed:
            start_failed = True
            raise RuntimeError('public native stream creation failure')
        return Stream()
    def is_ready(self, stream): return bool(stream.pending)
    def decode_stream(self, stream):
        value = stream.pending.pop(0)
        if value == 1: stream.text = 'Find a train'
        if value == 2: stream.text = 'Find a flight to Paris'
        if value == 3: stream.text = 'Find a flight to Paris. on Tuesday'
        if value == 4: stream.text = 'Weather tomorrow'
        if value == 5: stream.text = None
        if value == 6: stream.text = ''
        if value == 8:
            stream.text = 'Remember the'
            stream.tail_waiting = True
        if value == 10: stream.text = 'Remember the violet umbrella'
        if value == 'final' and stream.text and not stream.text.endswith('.'):
            stream.text += '.'
    def get_result(self, stream): return stream.text

_worker(None, model=Model())
""")
    original = asyncio.create_subprocess_exec
    processes, calls = [], []
    entered = tmp_path / "native_call_entered"
    monkeypatch.setenv("STREAM_TEST_ENTERED", str(entered))

    async def create(*args, **kwargs):
        calls.append((args, kwargs))
        process = await original(sys.executable, str(script), **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(streaming.asyncio, "create_subprocess_exec", create)
    return processes, calls, entered


async def partial_is(turn, text):
    async def wait():
        while turn.partial.text != text:
            await asyncio.sleep(0)
    await asyncio.wait_for(wait(), 3)


async def native_entered(path):
    async def wait():
        while not path.exists():
            await asyncio.sleep(0)
    await asyncio.wait_for(wait(), 3)


def hold_command(transcriber, operation):
    """Hold one acknowledged exchange to model an in-flight native call."""
    entered, release = asyncio.Event(), asyncio.Event()
    original = transcriber._exchange

    async def exchange(command):
        response = await original(command)
        if command.operation == operation and not entered.is_set():
            entered.set()
            await release.wait()
        return response

    transcriber._exchange = exchange
    return entered, release


def test_audio_is_processed_before_endpoint_and_revisions_replace_whole_hypothesis(model, worker, capsys):
    async def scenario():
        transcriber = SherpaStreamingTranscriber(model)
        try:
            await transcriber.start()
            await transcriber.start()
            turn = transcriber.open_turn()
            turn.feed(pcm(1))
            await partial_is(turn, "Find a train")
            first = turn.partial
            turn.feed(pcm(2))
            await partial_is(turn, "Find a flight to Paris")
            assert turn.partial.revision > first.revision
            turn.feed(pcm(3))
            await partial_is(turn, "Find a flight to Paris. on Tuesday")
            # A multi-sentence hypothesis did not seal the caller's utterance.
            turn.feed(pcm(0))
            final = turn.finish()
            assert final is turn.finish()
            assert await final == "Find a flight to Paris. on Tuesday."
            with pytest.raises(StreamingTranscriptionError, match="accepting"):
                turn.feed(pcm())
            assert len(worker[0]) == 1
        finally:
            await transcriber.close()
            await transcriber.close()
    asyncio.run(scenario())
    assert worker[0][0].returncode is not None
    assert worker[1][0][1]["stderr"] == asyncio.subprocess.DEVNULL
    output = capsys.readouterr()
    assert "Find a" not in output.out + output.err


def test_second_turn_accepts_audio_while_previous_finalization_is_pending(model, worker):
    async def scenario():
        transcriber = SherpaStreamingTranscriber(model)
        try:
            await transcriber.start()
            entered, release = hold_command(transcriber, streaming._FINISH)
            first = transcriber.open_turn()
            first.feed(pcm(2))
            final1 = first.finish()
            await asyncio.wait_for(entered.wait(), 3)
            second = transcriber.open_turn()
            second.feed(pcm(4))
            final2 = second.finish()
            assert not final1.done() and not final2.done()
            release.set()
            assert await final1 == "Find a flight to Paris."
            assert await final2 == "Weather tomorrow."
        finally:
            await transcriber.close()
    asyncio.run(scenario())


def test_endpoint_without_silence_preserves_the_last_words_once(model, worker):
    async def scenario():
        transcriber = SherpaStreamingTranscriber(model)
        try:
            await transcriber.start()
            turn = transcriber.open_turn()
            # The last word's frames are waiting for the encoder's lookahead.
            turn.feed(pcm(8))
            await partial_is(turn, "Remember the")
            assert await turn.finish() == "Remember the violet umbrella."
            assert await turn.finish() == "Remember the violet umbrella."
            other = transcriber.open_turn()
            other.feed(pcm(4))
            assert await other.finish() == "Weather tomorrow."
        finally:
            await transcriber.close()
    asyncio.run(scenario())


def test_overflow_counts_inflight_pcm_discards_whole_turn_and_preserves_other_turn(model, worker):
    async def scenario():
        transcriber = SherpaStreamingTranscriber(model)
        try:
            await transcriber.start()
            entered, release = hold_command(transcriber, streaming._FEED)
            failed = transcriber.open_turn()
            failed.feed(pcm(2))  # 0.1 seconds remains unacknowledged.
            await asyncio.wait_for(entered.wait(), 3)
            failed.feed(pcm(2, 46400))  # Exactly three seconds outstanding.
            with pytest.raises(StreamingBufferOverflow):
                failed.feed(pcm(2, 1))
            with pytest.raises(StreamingBufferOverflow):
                await failed.finish()
            assert failed.partial.text == ""
            other = transcriber.open_turn()
            other.feed(pcm(4))
            final = other.finish()
            release.set()
            assert await final == "Weather tomorrow."
            assert failed.partial.text == ""
        finally:
            await transcriber.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("action", ["abort", "cancel_final"])
def test_cancelled_inflight_result_cannot_revive_partial_or_final(model, worker, action):
    async def scenario():
        transcriber = SherpaStreamingTranscriber(model)
        try:
            await transcriber.start()
            entered, release = hold_command(transcriber, streaming._FEED)
            first = transcriber.open_turn()
            first.feed(pcm(2))
            final = first.finish()
            await asyncio.wait_for(entered.wait(), 3)
            if action == "abort":
                first.abort()
            else:
                final.cancel()
                await asyncio.sleep(0)  # Deliver Future cancellation callback.
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await final
            next_turn = transcriber.open_turn()
            next_turn.feed(pcm(4))
            assert await next_turn.finish() == "Weather tomorrow."
            assert first.partial.text == ""
        finally:
            await transcriber.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("value", [5, 9])
def test_native_error_or_invalid_hypothesis_never_returns_prior_partial(model, worker, value, capsys):
    async def scenario():
        transcriber = SherpaStreamingTranscriber(model)
        try:
            await transcriber.start()
            failed = transcriber.open_turn()
            failed.feed(pcm(2))
            await partial_is(failed, "Find a flight to Paris")
            failed.feed(pcm(value))
            with pytest.raises(StreamingTranscriptionError, match="failed"):
                await failed.finish()
            assert failed.partial.text == ""
            other = transcriber.open_turn()
            other.feed(pcm(4))
            assert await other.finish() == "Weather tomorrow."
        finally:
            await transcriber.close()
    asyncio.run(scenario())
    output = capsys.readouterr()
    assert "private text" not in output.out + output.err


def test_failed_native_stream_creation_preserves_the_next_turn(model, worker, monkeypatch):
    monkeypatch.setenv("STREAM_TEST_STREAM_START", "fail_once")

    async def scenario():
        transcriber = SherpaStreamingTranscriber(model)
        try:
            await transcriber.start()
            failed = transcriber.open_turn()
            failed.feed(pcm(2))
            with pytest.raises(StreamingTranscriptionError, match="failed"):
                await failed.finish()
            other = transcriber.open_turn()
            other.feed(pcm(4))
            assert await other.finish() == "Weather tomorrow."
            assert len(worker[0]) == 1
        finally:
            await transcriber.close()
    asyncio.run(scenario())


def test_hung_native_call_is_bounded_fails_all_turns_and_reaps_worker(model, worker):
    async def scenario():
        transcriber = SherpaStreamingTranscriber(model, command_timeout_s=.1)
        try:
            await transcriber.start()
            entered, _ = hold_command(transcriber, streaming._FINISH)
            first = transcriber.open_turn()
            first.feed(pcm(2))
            final1 = first.finish()
            await asyncio.wait_for(entered.wait(), 3)
            second = transcriber.open_turn()
            second.feed(pcm(4))
            final2 = second.finish()
            for final in (final1, final2):
                with pytest.raises(StreamingWorkerError, match="time limit"):
                    await asyncio.wait_for(final, 3)
            with pytest.raises(StreamingWorkerError, match="not ready"):
                transcriber.open_turn()
        finally:
            await transcriber.close()
        assert worker[0][0].returncode is not None
    asyncio.run(scenario())


def test_actually_blocked_child_is_killed_without_losing_other_futures(model, worker):
    async def scenario():
        transcriber = SherpaStreamingTranscriber(model, command_timeout_s=.2)
        try:
            await transcriber.start()
            first = transcriber.open_turn()
            first.feed(pcm(7))
            final1 = first.finish()
            await native_entered(worker[2])
            second = transcriber.open_turn()
            second.feed(pcm(4))
            final2 = second.finish()
            for final in (final1, final2):
                with pytest.raises(StreamingTranscriptionError, match="time limit"):
                    await asyncio.wait_for(final, 3)
        finally:
            await transcriber.close()
        assert worker[0][0].returncode is not None
    asyncio.run(scenario())


def test_restart_joins_failed_worker_cleanup_before_accepting_new_audio(model, worker):
    async def scenario():
        transcriber = SherpaStreamingTranscriber(model)
        await transcriber.start()
        entered, release, restarting = asyncio.Event(), asyncio.Event(), asyncio.Event()
        original_stop = transcriber._stop_process

        async def stop():
            if not entered.is_set():
                entered.set()
                await release.wait()
            await original_stop()

        async def failed_exchange(command):
            raise ConnectionError("public fake worker lost its pipe")

        original_exchange = transcriber._exchange
        transcriber._stop_process = stop
        transcriber._exchange = failed_exchange
        failed = transcriber.open_turn()
        failed.feed(pcm(2))
        with pytest.raises(StreamingWorkerError):
            await failed.finish()
        await entered.wait()
        transcriber._exchange = original_exchange

        async def restart():
            restarting.set()
            await transcriber.start()

        task = asyncio.create_task(restart())
        try:
            await restarting.wait()
            assert not task.done()
            with pytest.raises(StreamingWorkerError):
                transcriber.open_turn()
            release.set()
            await asyncio.wait_for(task, 3)
            assert worker[0][0].returncode is not None
            turn = transcriber.open_turn()
            turn.feed(pcm(4))
            assert await turn.finish() == "Weather tomorrow."
        finally:
            release.set()
            await transcriber.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("mode", ["stall", "fail"])
def test_startup_failure_reaps_child_and_allows_a_fresh_start(model, worker, monkeypatch, capsys, mode):
    monkeypatch.setenv("STREAM_TEST_STARTUP", mode)
    bootstrapped = worker[2].with_name("python_bootstrapped")
    monkeypatch.setenv("STREAM_TEST_BOOTSTRAPPED", str(bootstrapped))
    spawn = asyncio.create_subprocess_exec

    async def bootstrapped_child(*args, **kwargs):
        process = await spawn(*args, **kwargs)
        # The short injected model timeout must not measure a cold Python import.
        await native_entered(bootstrapped)
        return process

    monkeypatch.setattr(streaming.asyncio, "create_subprocess_exec", bootstrapped_child)

    async def scenario():
        transcriber = SherpaStreamingTranscriber(model, startup_timeout_s=.2)
        error_type = asyncio.TimeoutError if mode == "stall" else StreamingTranscriptionError
        with pytest.raises(error_type):
            await transcriber.start()
        assert worker[0][0].returncode is not None
        monkeypatch.delenv("STREAM_TEST_STARTUP")
        bootstrapped.unlink()
        transcriber._startup_timeout_s = 5
        try:
            await transcriber.start()
            turn = transcriber.open_turn()
            turn.feed(pcm(4))
            assert await turn.finish() == "Weather tomorrow."
        finally:
            await transcriber.close()
    asyncio.run(scenario())
    output = capsys.readouterr()
    assert "private native error" not in output.out + output.err


def test_close_cancels_open_and_sealed_turns_without_waiting_for_native_call(model, worker):
    async def scenario():
        transcriber = SherpaStreamingTranscriber(model)
        await transcriber.start()
        entered, _ = hold_command(transcriber, streaming._FEED)
        first = transcriber.open_turn()
        first.feed(pcm(2))
        await asyncio.wait_for(entered.wait(), 3)
        second = transcriber.open_turn()
        second.feed(pcm(4))
        final2 = second.finish()
        await asyncio.wait_for(transcriber.close(), 3)
        assert first.finish().cancelled() and final2.cancelled()
        assert first.partial.text == second.partial.text == ""
        assert worker[0][0].returncode is not None
    asyncio.run(scenario())


def test_silence_and_removed_hypothesis_finalize_empty_and_model_is_reused(model, worker):
    async def scenario():
        transcriber = SherpaStreamingTranscriber(model)
        try:
            await transcriber.start()
            first = transcriber.open_turn()
            first.feed(pcm(0))
            assert await first.finish() == ""
            second = transcriber.open_turn()
            second.feed(pcm(2))
            await partial_is(second, "Find a flight to Paris")
            second.feed(pcm(6))
            assert await second.finish() == ""
            third = transcriber.open_turn()
            third.feed(pcm(4))
            assert await third.finish() == "Weather tomorrow."
            assert len(worker[0]) == 1
        finally:
            await transcriber.close()
    asyncio.run(scenario())


def test_queue_capacity_refuses_new_turn_without_losing_existing_turn(model, worker):
    async def scenario():
        transcriber = SherpaStreamingTranscriber(model, max_turns=1)
        try:
            await transcriber.start()
            first = transcriber.open_turn()
            first.feed(pcm(4))
            with pytest.raises(StreamingBufferOverflow, match="full"):
                transcriber.open_turn()
            assert await first.finish() == "Weather tomorrow."
            second = transcriber.open_turn()
            second.feed(pcm(2))
            assert await second.finish() == "Find a flight to Paris."
        finally:
            await transcriber.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("audio", [b"", b"a", bytearray(b"ab")])
def test_invalid_pcm_invalidates_prior_partial_instead_of_submitting_it(model, worker, audio):
    async def scenario():
        transcriber = SherpaStreamingTranscriber(model)
        try:
            await transcriber.start()
            first = transcriber.open_turn()
            first.feed(pcm(2))
            await partial_is(first, "Find a flight to Paris")
            with pytest.raises(StreamingTranscriptionError, match="PCM16"):
                first.feed(audio)
            with pytest.raises(StreamingTranscriptionError, match="PCM16"):
                await first.finish()
            assert first.partial.text == ""
        finally:
            await transcriber.close()
    asyncio.run(scenario())


def test_missing_assets_are_rejected_without_optional_backend_import(tmp_path):
    with pytest.raises(ValueError, match="local assets"):
        SherpaStreamingTranscriber(tmp_path)


def test_model_loader_pins_optional_api_and_uses_local_assets_without_download(model, monkeypatch):
    import importlib.metadata

    created = []

    def constructor(**options):
        created.append(options)
        return "loaded model"

    monkeypatch.setattr(importlib.metadata, "version", lambda name: "1.13.8")
    monkeypatch.setitem(sys.modules, "sherpa_onnx", SimpleNamespace(
        OnlineRecognizer=SimpleNamespace(from_transducer=constructor)))
    assert streaming._load_model(model) == "loaded model"
    assert created == [{
        "encoder": str(model / "encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx"),
        "decoder": str(model / "decoder-epoch-99-avg-1-chunk-16-left-128.onnx"),
        "joiner": str(model / "joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx"),
        "tokens": str(model / "tokens.txt"),
        "num_threads": 1, "sample_rate": 16000, "feature_dim": 80,
        "decoding_method": "modified_beam_search", "max_active_paths": 4,
        "enable_endpoint_detection": False, "provider": "cpu", "debug": False,
        "model_type": "zipformer2",
    }]
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "1.13.9")
    with pytest.raises(StreamingTranscriptionError, match="1.13.8"):
        streaming._load_model(model)
