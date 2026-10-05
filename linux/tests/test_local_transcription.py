# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import sys
from types import SimpleNamespace
import wave

import pytest

from musegadget import local_transcription
from musegadget.local_transcription import MoonshineTranscriber, WhisperTranscriber


@pytest.fixture
def model(tmp_path):
    path = tmp_path / "whisper"
    path.mkdir()
    for name in ("model.bin", "config.json", "tokenizer.json"):
        (path / name).write_bytes(b"test model")
    return path


@pytest.fixture
def moonshine_model(tmp_path):
    path = tmp_path / "moonshine"
    path.mkdir()
    for name in ("preprocess.onnx", "encode.int8.onnx", "uncached_decode.int8.onnx",
                 "cached_decode.int8.onnx", "tokens.txt"):
        (path / name).write_bytes(b"test model")
    return path


def recording(amplitude=1000):
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(amplitude.to_bytes(2, "little", signed=True) * 1600)
    return output.getvalue()


@pytest.fixture
def worker(tmp_path, monkeypatch):
    """Exercise the binary protocol against a real, independently framed child."""
    script = tmp_path / "transcription_worker.py"
    script.write_text("""
import hashlib, io, json, os, struct, sys, time, wave
from pathlib import Path
mode = os.environ.get('WHISPER_TEST_MODE', 'normal')
output = sys.stdout.buffer
source = sys.stdin.buffer
if mode == 'startup_fail':
    sys.stderr.write('private recognized speech must not appear')
    sys.exit(7)
if mode == 'startup_stall':
    time.sleep(3600)
output.write(struct.pack('<I', 1 if mode == 'bad_ready' else 0))
output.flush()
while True:
    header = source.read(4)
    if not header:
        break
    size = struct.unpack('<I', header)[0]
    data = bytearray()
    while len(data) < size:
        part = source.read(size - len(data))
        if not part:
            sys.exit(6)
        data.extend(part)
    with open(os.environ['WHISPER_TEST_INPUT'], 'a') as log:
        log.write(hashlib.sha256(data).hexdigest() + '\\n')
    if mode == 'stall':
        time.sleep(3600)
    if mode == 'fail':
        sys.stderr.write('private recognized speech must not appear')
        sys.exit(7)
    if mode == 'oversized':
        output.write(struct.pack('<I', 65537))
        output.flush()
        time.sleep(3600)
    if mode == 'flood':
        output.write(struct.pack('<I', 65537))
        output.flush()
        while True:
            output.write(b'x' * 65536)
            output.flush()
    if mode == 'truncated':
        output.write(struct.pack('<I', 80) + b'{')
        output.flush()
        sys.exit(0)
    with wave.open(io.BytesIO(data), 'rb') as wav:
        frames = wav.readframes(wav.getnframes())
    if any(frames):
        notice = {'speech_detected': True}
        if mode == 'notice_false':
            notice = {'speech_detected': False}
        if mode == 'notice_integer':
            notice = {'speech_detected': 1}
        if mode == 'notice_extra':
            notice['text'] = 'unexpected'
        notification = json.dumps(notice).encode('utf-8')
        for _ in range(2 if mode == 'notice_duplicate' else 1):
            packet = struct.pack('<I', len(notification)) + notification
            for offset in range(0, len(packet), 3):
                output.write(packet[offset:offset+3])
                output.flush()
                time.sleep(.001)
        if mode == 'notice_stall':
            time.sleep(3600)
        time.sleep(.02)
    text = '  Pineapple sunshine. Café.\\n' if any(frames) else ''
    response = json.dumps({'text': text}, ensure_ascii=False).encode('utf-8')
    if mode == 'invalid_json':
        response = b'{private speech'
    if mode == 'invalid_utf8':
        response = b'\\xff'
    if mode == 'invalid_text':
        response = b'{"text": 7}'
    packet = struct.pack('<I', len(response)) + response
    for offset in range(0, len(packet), 3):
        output.write(packet[offset:offset+3])
        output.flush()
        time.sleep(.001)
""")
    input_path = tmp_path / "wav_hashes.txt"
    monkeypatch.setenv("WHISPER_TEST_INPUT", str(input_path))
    original = asyncio.create_subprocess_exec
    calls, processes = [], []

    async def create(*args, **kwargs):
        calls.append((args, kwargs))
        process = await original(sys.executable, str(script), **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(local_transcription.asyncio, "create_subprocess_exec", create)
    return calls, processes, input_path


async def transcribe_and_close(transcriber, wav):
    try:
        return await transcriber.transcribe(wav)
    finally:
        await transcriber.close()


def test_transcribes_real_wav_packet_without_text_in_argv_or_diagnostics(model, worker, capsys):
    wav = recording()
    text = asyncio.run(transcribe_and_close(WhisperTranscriber(model), wav))
    assert text == "Pineapple sunshine. Café."
    assert worker[2].read_text().splitlines() == [hashlib.sha256(wav).hexdigest()]
    assert worker[0][0][0] == (sys.executable, "-m", "musegadget.local_transcription", "--model", str(model))
    assert worker[0][0][1]["stderr"] == asyncio.subprocess.DEVNULL
    assert worker[1][0].returncode == -9
    captured = capsys.readouterr()
    assert text not in captured.out + captured.err


def test_start_and_silence_keep_one_warm_worker(model, worker):
    async def scenario():
        transcriber = WhisperTranscriber(model)
        try:
            await transcriber.start()
            await transcriber.start()
            assert await transcriber.transcribe(recording(0)) == ""
            assert await transcriber.transcribe(recording()) == "Pineapple sunshine. Café."
            assert len(worker[1]) == 1
            assert worker[1][0].returncode is None
        finally:
            await transcriber.close()
            await transcriber.close()
    asyncio.run(scenario())
    assert worker[1][0].returncode == -9


def test_speech_callback_arrives_before_recognition_and_never_for_silence(model, worker):
    async def scenario():
        transcriber = WhisperTranscriber(model)
        detected = asyncio.Event()
        callbacks = []

        def on_speech():
            callbacks.append("speech")
            detected.set()

        try:
            await transcriber.start()
            task = asyncio.create_task(transcriber.transcribe(recording(), on_speech=on_speech))
            await asyncio.wait_for(detected.wait(), 2)
            assert callbacks == ["speech"] and not task.done()
            assert await task == "Pineapple sunshine. Café."
            assert await transcriber.transcribe(recording(0), on_speech=on_speech) == ""
            assert callbacks == ["speech"]
            assert len(worker[1]) == 1
        finally:
            await transcriber.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("mode,message,callback_count", [
    ("notice_duplicate", "duplicate speech notification", 1),
    ("notice_false", "invalid speech notification", 0),
    ("notice_integer", "invalid speech notification", 0),
    ("notice_extra", "invalid speech notification", 0),
])
def test_invalid_speech_notifications_are_bounded_and_reap_worker(model, worker, monkeypatch,
                                                                 mode, message, callback_count):
    monkeypatch.setenv("WHISPER_TEST_MODE", mode)
    callbacks = []
    with pytest.raises(ValueError, match=message):
        asyncio.run(WhisperTranscriber(model).transcribe(recording(),
                                                        on_speech=lambda: callbacks.append("speech")))
    assert callbacks == ["speech"] * callback_count
    assert worker[1][0].returncode == -9


def test_cancellation_after_speech_notification_reaps_and_restarts(model, worker, monkeypatch):
    async def scenario():
        transcriber = WhisperTranscriber(model)
        detected = asyncio.Event()
        monkeypatch.setenv("WHISPER_TEST_MODE", "notice_stall")
        task = asyncio.create_task(transcriber.transcribe(recording(), on_speech=detected.set))
        await asyncio.wait_for(detected.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)
        assert worker[1][0].returncode == -9
        monkeypatch.setenv("WHISPER_TEST_MODE", "normal")
        try:
            assert await transcriber.transcribe(recording()) == "Pineapple sunshine. Café."
        finally:
            await transcriber.close()
    asyncio.run(scenario())
    assert len(worker[1]) == 2 and all(process.returncode == -9 for process in worker[1])


def test_speech_callback_failure_stops_and_reaps_worker(model, worker):
    def failed():
        raise RuntimeError("speech callback failed")
    with pytest.raises(RuntimeError, match="speech callback failed"):
        asyncio.run(WhisperTranscriber(model).transcribe(recording(), on_speech=failed))
    assert worker[1][0].returncode == -9


def test_invalid_speech_callback_is_rejected_before_start(model, worker):
    with pytest.raises(TypeError, match="synchronous callable"):
        asyncio.run(WhisperTranscriber(model).transcribe(recording(), on_speech="speech"))
    assert not worker[1]


def test_concurrent_requests_serialize_complete_wav_packets(model, worker):
    async def scenario():
        transcriber = WhisperTranscriber(model)
        try:
            return await asyncio.gather(transcriber.transcribe(recording(0)),
                                        transcriber.transcribe(recording()))
        finally:
            await transcriber.close()
    assert asyncio.run(scenario()) == ["", "Pineapple sunshine. Café."]
    assert len(worker[1]) == 1
    assert worker[2].read_text().splitlines() == [hashlib.sha256(recording(0)).hexdigest(),
                                                hashlib.sha256(recording()).hexdigest()]


@pytest.mark.parametrize("mode,error,message", [
    ("fail", RuntimeError, "exit status 7"),
    ("startup_fail", RuntimeError, "exit status 7"),
    ("bad_ready", ValueError, "ready handshake"),
    ("oversized", ValueError, "exceeds 64 KiB"),
    ("flood", ValueError, "exceeds 64 KiB"),
    ("truncated", ValueError, "incomplete transcript packet"),
    ("invalid_json", ValueError, "invalid transcript response"),
    ("invalid_utf8", ValueError, "invalid transcript response"),
    ("invalid_text", ValueError, "invalid transcript response"),
])
def test_invalid_worker_response_stops_and_reaps_without_private_data(model, worker, monkeypatch, capsys,
                                                                    mode, error, message):
    monkeypatch.setenv("WHISPER_TEST_MODE", mode)
    with pytest.raises(error, match=message) as raised:
        asyncio.run(asyncio.wait_for(transcribe_and_close(WhisperTranscriber(model), recording()), 3))
    assert "private" not in str(raised.value)
    output = capsys.readouterr()
    assert "private" not in output.out + output.err
    assert worker[1][0].returncode is not None


@pytest.mark.parametrize("mode", ["startup_stall", "stall"])
def test_cancel_reaps_worker_and_next_request_restarts(model, worker, monkeypatch, mode):
    async def scenario():
        transcriber = WhisperTranscriber(model)
        monkeypatch.setenv("WHISPER_TEST_MODE", mode)
        task = asyncio.create_task(transcriber.transcribe(recording()))
        while not worker[1] or (mode == "stall" and not worker[2].exists()):
            await asyncio.sleep(.005)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)
        assert worker[1][0].returncode == -9
        monkeypatch.setenv("WHISPER_TEST_MODE", "normal")
        try:
            assert await transcriber.transcribe(recording()) == "Pineapple sunshine. Café."
            assert len(worker[1]) == 2
        finally:
            await transcriber.close()
    asyncio.run(scenario())
    assert all(process.returncode == -9 for process in worker[1])


def test_cancel_during_spawn_waits_for_child_and_reaps_it(model, worker, monkeypatch):
    original = local_transcription.asyncio.create_subprocess_exec
    spawned = asyncio.Event()
    release = asyncio.Event()

    async def delayed(*args, **kwargs):
        process = await original(*args, **kwargs)
        spawned.set()
        await release.wait()
        return process

    monkeypatch.setattr(local_transcription.asyncio, "create_subprocess_exec", delayed)

    async def scenario():
        task = asyncio.create_task(WhisperTranscriber(model).start())
        await spawned.wait()
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)
    asyncio.run(scenario())
    assert worker[1][0].returncode == -9


@pytest.mark.parametrize("wav", [None, bytearray(recording()), b"", b"x" * 44,
                                  b"RIFFxxxxWAVE" + b"x" * (2 * 1024 * 1024)])
def test_invalid_input_is_rejected_before_starting_worker(model, worker, wav):
    with pytest.raises(ValueError, match="WAV"):
        asyncio.run(WhisperTranscriber(model).transcribe(wav))
    assert not worker[1]


@pytest.mark.parametrize("missing", ["model.bin", "config.json", "tokenizer.json"])
def test_incomplete_model_directory_is_rejected(model, missing):
    (model / missing).unlink()
    with pytest.raises(ValueError, match="model or tokenizer"):
        WhisperTranscriber(model)


@pytest.mark.parametrize("missing", ["preprocess.onnx", "encode.int8.onnx",
                                    "uncached_decode.int8.onnx", "cached_decode.int8.onnx",
                                    "tokens.txt"])
def test_moonshine_rejects_missing_assets_before_spawning(moonshine_model, missing, worker):
    (moonshine_model / missing).unlink()
    with pytest.raises(ValueError, match="Moonshine model"):
        MoonshineTranscriber(moonshine_model)
    assert not worker[1]


def test_moonshine_uses_one_warm_private_packet_worker(moonshine_model, worker, capsys):
    async def scenario():
        transcriber = MoonshineTranscriber(moonshine_model)
        callbacks = []
        try:
            await transcriber.start()
            await transcriber.start()
            assert await transcriber.transcribe(recording(), on_speech=lambda: callbacks.append(True)) == "Pineapple sunshine. Café."
            assert await transcriber.transcribe(recording(0), on_speech=lambda: callbacks.append(False)) == ""
            assert await transcriber.transcribe(recording()) == "Pineapple sunshine. Café."
            assert callbacks == [True]
            assert len(worker[1]) == 1 and worker[1][0].returncode is None
        finally:
            await transcriber.close()
    asyncio.run(scenario())
    assert worker[0][0][0] == (sys.executable, "-m", "musegadget.local_transcription",
                             "--model", str(moonshine_model), "--backend", "moonshine")
    assert worker[0][0][1]["stderr"] == asyncio.subprocess.DEVNULL
    assert worker[1][0].returncode == -9
    captured = capsys.readouterr()
    assert "Pineapple" not in captured.out + captured.err


def test_moonshine_cancellation_reaps_worker_and_next_turn_restarts(moonshine_model, worker, monkeypatch):
    async def scenario():
        transcriber = MoonshineTranscriber(moonshine_model)
        detected = asyncio.Event()
        monkeypatch.setenv("WHISPER_TEST_MODE", "notice_stall")
        task = asyncio.create_task(transcriber.transcribe(recording(), on_speech=detected.set))
        await asyncio.wait_for(detected.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert worker[1][0].returncode == -9
        monkeypatch.setenv("WHISPER_TEST_MODE", "normal")
        try:
            assert await transcriber.transcribe(recording()) == "Pineapple sunshine. Café."
        finally:
            await transcriber.close()
    asyncio.run(scenario())
    assert len(worker[1]) == 2
    assert all(process.returncode == -9 for process in worker[1])


def test_moonshine_worker_keeps_full_twenty_second_audio_and_gates_silence(
        moonshine_model, tmp_path, monkeypatch):
    np = pytest.importorskip("numpy")
    package = tmp_path / "faster_whisper"
    package.mkdir()
    (package / "__init__.py").write_text("class WhisperModel:\n    def __init__(self, *args, **kwargs):\n        raise RuntimeError('Whisper must not load for Moonshine')\n")
    (package / "audio.py").write_text("""
import wave
import numpy as np
def decode_audio(source, sampling_rate):
    assert sampling_rate == 16000
    with wave.open(source, 'rb') as wav:
        return np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2').astype(np.float32) / 32768
""")
    (package / "vad.py").write_text("""
import os
from pathlib import Path
import numpy as np
class VadOptions:
    pass
def get_vad_model():
    Path(os.environ['MOONSHINE_VAD_READY']).write_text('ready')
def get_speech_timestamps(audio, options):
    return [{'start': 200, 'end': 1000}] if np.any(audio) else []
def collect_chunks(*args):
    raise AssertionError('Moonshine must retain the complete recording')
""")
    (tmp_path / "sherpa_onnx.py").write_text("""
import hashlib, json, os
from pathlib import Path
from types import SimpleNamespace
class OfflineRecognizer:
    @staticmethod
    def from_moonshine(**kwargs):
        return Recognizer(kwargs)
class Recognizer:
    def __init__(self, settings):
        self.observed = {'load': settings, 'turns': []}
    def create_stream(self):
        class Stream:
            def accept_waveform(self, rate, audio):
                assert rate == 16000
                self.audio = audio
        return Stream()
    def decode_stream(self, stream):
        self.observed['turns'].append({'samples': len(stream.audio),
            'audio_sha256': hashlib.sha256(stream.audio.tobytes()).hexdigest()})
        Path(os.environ['MOONSHINE_MODEL_CALLS']).write_text(json.dumps(self.observed))
        stream.result = SimpleNamespace(text='  Keep the final instruction: violet.  ')
""")
    observations = tmp_path / "moonshine_calls.json"
    vad_ready = tmp_path / "vad_ready"
    monkeypatch.setenv("MOONSHINE_MODEL_CALLS", str(observations))
    monkeypatch.setenv("MOONSHINE_VAD_READY", str(vad_ready))
    monkeypatch.setenv("PYTHONPATH", str(tmp_path) + ":" + ":".join(sys.path))
    samples = np.arange(20 * 16000, dtype=np.int16)
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(samples.tobytes())
    full_wav = output.getvalue()

    async def scenario():
        transcriber = MoonshineTranscriber(moonshine_model)
        callbacks = []
        try:
            await transcriber.start()
            assert vad_ready.read_text() == "ready"
            assert await transcriber.transcribe(full_wav, on_speech=lambda: callbacks.append(True)) == "Keep the final instruction: violet."
            assert await transcriber.transcribe(recording(0), on_speech=lambda: callbacks.append(False)) == ""
            assert callbacks == [True]
        finally:
            await transcriber.close()
    asyncio.run(scenario())
    actual = json.loads(observations.read_text())
    assert actual["turns"] == [{"samples": 320000, "audio_sha256": hashlib.sha256(
        (samples.astype(np.float32) / 32768).tobytes()).hexdigest()}]
    assert actual["load"] == {
        "preprocessor": str(moonshine_model / "preprocess.onnx"),
        "encoder": str(moonshine_model / "encode.int8.onnx"),
        "uncached_decoder": str(moonshine_model / "uncached_decode.int8.onnx"),
        "cached_decoder": str(moonshine_model / "cached_decode.int8.onnx"),
        "tokens": str(moonshine_model / "tokens.txt"), "num_threads": 2,
        "debug": False, "provider": "cpu",
    }


@pytest.fixture
def short_decoder(monkeypatch):
    np = pytest.importorskip("numpy")
    monkeypatch.setitem(sys.modules, "ctranslate2", SimpleNamespace(
        StorageView=SimpleNamespace(from_array=lambda array: array)))
    state = SimpleNamespace(features=np.full((80, 321), .25, dtype=np.float32),
                            encoded=[], fallback_audio=[], generated=[])
    state.result = SimpleNamespace(sequences_ids=[[1, 2, 3]], scores=[-.2], no_speech_prob=.05)

    def encode(features):
        state.encoded.append(features.copy())
        return features

    def generate(encoded, prompts, **options):
        state.generated.append((prompts, options))
        return [state.result]

    def transcribe(audio, **options):
        state.fallback_audio.append((audio, options))
        metadata = {"tokens": [1, 2], "avg_logprob": -.2, "no_speech_prob": .05,
                    "compression_ratio": .5}
        return iter([SimpleNamespace(text=" The complete question, ", **metadata),
                     SimpleNamespace(text=" including the last instruction. ", **metadata)]), None

    model = SimpleNamespace(feature_extractor=lambda audio: state.features,
                            model=SimpleNamespace(encode=encode, generate=generate),
                            transcribe=transcribe, max_length=448)
    tokenizer = SimpleNamespace(sot_sequence=[101], no_timestamps=102,
                                decode=lambda tokens: "  What is the weather?  ")
    return np, model, tokenizer, state


def test_short_speech_uses_small_encoder_and_preserves_decoder_policy(short_decoder):
    np, model, tokenizer, state = short_decoder
    audio = np.ones(32000, dtype=np.float32)

    assert local_transcription._transcribe_audio(model, tokenizer, [7, 8], audio) == "What is the weather?"
    assert state.encoded[0].shape == (1, 80, 500)
    np.testing.assert_array_equal(state.encoded[0][0, :, :321], state.features)
    np.testing.assert_array_equal(state.encoded[0][0, :, 321:], 0)
    assert state.generated == [([[101, 102]], {
        "beam_size": 1, "max_length": 66, "suppress_blank": True,
        "suppress_tokens": [7, 8], "return_scores": True,
        "return_no_speech_prob": True, "sampling_temperature": 0.0,
    })]
    assert not state.fallback_audio


def test_eight_second_turn_keeps_its_last_word_and_extra_encoder_context(short_decoder):
    np, model, tokenizer, state = short_decoder
    state.features = np.zeros((80, 801), dtype=np.float32)
    state.features[:, -1] = .875
    audio = np.ones(8 * 16000, dtype=np.float32)

    assert local_transcription._transcribe_audio(model, tokenizer, [], audio) == "What is the weather?"
    assert state.encoded[0].shape == (1, 80, 1000)
    np.testing.assert_array_equal(state.encoded[0][0, :, 800], .875)
    np.testing.assert_array_equal(state.encoded[0][0, :, 801:], 0)
    assert not state.fallback_audio


@pytest.mark.parametrize("failure", ["empty", "low_confidence", "threshold", "nonfinite_score",
                                    "nonfinite_speech_probability", "invalid_probability", "token_limit",
                                    "duration_token_limit", "invalid_token", "repetition"])
def test_uncertain_or_incomplete_short_result_uses_complete_original_audio(short_decoder, failure):
    np, model, tokenizer, state = short_decoder
    audio = np.ones(32000, dtype=np.float32)
    if failure == "empty":
        tokenizer.decode = lambda tokens: "  "
    elif failure == "low_confidence":
        state.result.scores = [-1.5]
    elif failure == "threshold":
        state.result.scores = [-4 / 3]
        state.result.no_speech_prob = .7
    elif failure == "nonfinite_score":
        state.result.scores = [float("nan")]
    elif failure == "nonfinite_speech_probability":
        state.result.no_speech_prob = float("nan")
    elif failure == "invalid_probability":
        state.result.no_speech_prob = 1.1
    elif failure == "duration_token_limit":
        state.result.sequences_ids = [[1] * 64]
    elif failure == "invalid_token":
        state.result.sequences_ids = [[float("nan")]]
    elif failure == "repetition":
        tokenizer.decode = lambda tokens: "Buy milk and eggs. " * 50
    else:
        model.max_length = 5

    assert local_transcription._transcribe_audio(model, tokenizer, [], audio) == (
        "The complete question, including the last instruction.")
    assert state.fallback_audio == [(audio, {
        "language": "en", "beam_size": 3, "condition_on_previous_text": False,
        "temperature": (0.0, .2, .4), "vad_filter": False, "no_speech_threshold": .6,
        "log_prob_threshold": -1.0, "compression_ratio_threshold": 2.4,
        "max_new_tokens": 3 if failure == "token_limit" else 64,
    })]


def test_confident_speech_retained_even_if_no_speech_probability_is_high(short_decoder):
    np, model, tokenizer, state = short_decoder
    state.result.no_speech_prob = .9
    assert local_transcription._transcribe_audio(model, tokenizer, [], np.ones(32000)) == "What is the weather?"
    assert not state.fallback_audio


@pytest.mark.parametrize("failure", ["repetition", "nonfinite_score", "nonfinite_probability",
                                    "nonfinite_compression", "invalid_probability", "token_limit"])
def test_complete_fallback_rejects_hallucinated_or_truncated_question(short_decoder, failure):
    np, model, tokenizer, state = short_decoder
    segment = SimpleNamespace(text=" Buy milk and eggs. " * 50, tokens=[1, 2],
                              avg_logprob=-.2, no_speech_prob=.1, compression_ratio=.5)
    if failure == "repetition":
        segment.compression_ratio = 12
    elif failure == "nonfinite_score":
        segment.avg_logprob = float("nan")
    elif failure == "nonfinite_probability":
        segment.no_speech_prob = float("inf")
    elif failure == "nonfinite_compression":
        segment.compression_ratio = float("nan")
    elif failure == "invalid_probability":
        segment.no_speech_prob = -1
    else:
        segment.tokens = [1] * 400
    model.transcribe = lambda *args, **kwargs: (iter([segment]), None)
    assert local_transcription._transcribe_audio(model, tokenizer, [], np.ones(20 * 16000)) == ""


@pytest.mark.parametrize("samples", [8 * 16000 + 1, 20 * 16000, 30 * 16000])
def test_long_recording_uses_normal_segmented_decoder_without_truncation(short_decoder, samples):
    np, model, tokenizer, state = short_decoder
    audio = np.ones(samples, dtype=np.float32)
    audio[-1] = .875

    assert local_transcription._transcribe_audio(model, tokenizer, [], audio) == (
        "The complete question, including the last instruction.")
    assert not state.encoded
    assert state.fallback_audio[0][0] is audio
    assert state.fallback_audio[0][0][-1] == .875


def test_production_worker_uses_cpu_int8_and_isolated_english_vad_settings(model, tmp_path, monkeypatch):
    np = pytest.importorskip("numpy")
    package = tmp_path / "faster_whisper"
    package.mkdir()
    observations = tmp_path / "model_calls.json"
    (package / "__init__.py").write_text("""
import hashlib, json, os
import numpy as np
from pathlib import Path
from types import SimpleNamespace
class WhisperModel:
    def __init__(self, model, **kwargs):
        self.observations = {'model': model, 'load': kwargs, 'turns': []}
        self.hf_tokenizer = None
        self.max_length = 448
        self.model = SimpleNamespace(is_multilingual=False, encode=lambda features: features,
            generate=lambda *args, **kwargs: [SimpleNamespace(
                sequences_ids=[[1, 2]], scores=[-2.0], no_speech_prob=.2)])
    def feature_extractor(self, audio):
        return np.zeros((80, len(audio) // 160 + 1), dtype=np.float32)
    def transcribe(self, audio, **kwargs):
        assert isinstance(audio, np.ndarray) and audio.dtype == np.float32 and audio.ndim == 1
        self.observations['turns'].append({'settings': kwargs,
                                         'audio_sha256': hashlib.sha256(audio.tobytes()).hexdigest()})
        Path(os.environ['WHISPER_MODEL_CALLS']).write_text(json.dumps(self.observations))
        metadata = {'tokens': [1, 2], 'avg_logprob': -.2, 'no_speech_prob': .2,
                    'compression_ratio': .5}
        return iter([SimpleNamespace(text=' Hello.', **metadata), SimpleNamespace(text='  ', **metadata),
                     SimpleNamespace(text=' A second segment. ', **metadata)]), None
""")
    (package / "audio.py").write_text("""
import io, wave
import numpy as np
def decode_audio(source, sampling_rate):
    assert isinstance(source, io.BytesIO) and sampling_rate == 16000
    with wave.open(source, 'rb') as wav:
        assert wav.getframerate() == 16000 and wav.getnchannels() == 1
        return np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2').astype(np.float32) / 32768.0
""")
    (package / "tokenizer.py").write_text("""
class Tokenizer:
    sot_sequence = [101]
    no_timestamps = 102
    def __init__(self, *args, **kwargs):
        pass
    def decode(self, tokens):
        return 'Uncertain short decode.'
""")
    (package / "transcribe.py").write_text("""
def get_suppressed_tokens(tokenizer, tokens):
    return [7, 8]
""")
    (tmp_path / "ctranslate2.py").write_text("""
class StorageView:
    @staticmethod
    def from_array(array):
        return array
""")
    vad_warmed = tmp_path / "vad_ready"
    (package / "vad.py").write_text("""
import os
from pathlib import Path
import numpy as np
class VadOptions:
    pass
def get_vad_model():
    Path(os.environ['WHISPER_VAD_READY']).write_text('ready')
def get_speech_timestamps(audio, options):
    assert isinstance(options, VadOptions)
    return [{'start': 200, 'end': 1000}] if np.any(audio) else []
def collect_chunks(audio, chunks):
    return [audio[c['start']:c['end']] for c in chunks], [{}]
""")
    monkeypatch.setenv("WHISPER_MODEL_CALLS", str(observations))
    monkeypatch.setenv("WHISPER_VAD_READY", str(vad_warmed))
    monkeypatch.setenv("PYTHONPATH", str(tmp_path) + ":" + ":".join(sys.path))

    async def scenario():
        transcriber = WhisperTranscriber(model)
        try:
            await transcriber.start()
            assert vad_warmed.read_text() == "ready"
            callbacks = []
            assert await transcriber.transcribe(recording(), on_speech=lambda: callbacks.append("speech")) == "Hello. A second segment."
            full_output = io.BytesIO()
            full_samples = np.zeros(20 * 16000, dtype=np.int16)
            full_samples[:64] = 1000
            full_samples[-64:] = 2000
            with wave.open(full_output, "wb") as wav:
                wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                wav.writeframes(full_samples.tobytes())
            assert await transcriber.transcribe(full_output.getvalue(), on_speech=lambda: callbacks.append("speech")) == "Hello. A second segment."
            assert await transcriber.transcribe(recording(0), on_speech=lambda: callbacks.append("silence")) == ""
            assert callbacks == ["speech", "speech"]
        finally:
            await transcriber.close()
    asyncio.run(scenario())
    actual = json.loads(observations.read_text())
    assert actual["model"] == str(model)
    assert actual["load"] == {"device": "cpu", "compute_type": "int8", "cpu_threads": 2,
                              "num_workers": 1, "local_files_only": True}
    expected_settings = {"language": "en", "beam_size": 3, "condition_on_previous_text": False,
                         "temperature": [0.0, .2, .4], "vad_filter": False, "no_speech_threshold": 0.6,
                         "log_prob_threshold": -1.0, "compression_ratio_threshold": 2.4,
                         "max_new_tokens": 64}
    expected_audio = np.full(1600, 1000 / 32768.0, dtype=np.float32)
    full_audio = np.zeros(320000, dtype=np.float32)
    full_audio[:64] = 1000 / 32768
    full_audio[-64:] = 2000 / 32768
    assert actual["turns"] == [{"settings": expected_settings,
                                "audio_sha256": hashlib.sha256(expected_audio.tobytes()).hexdigest()},
                               {"settings": dict(expected_settings, max_new_tokens=400),
                                "audio_sha256": hashlib.sha256(full_audio.tobytes()).hexdigest()}]
