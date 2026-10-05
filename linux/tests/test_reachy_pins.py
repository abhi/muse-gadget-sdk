import asyncio
import functools
import json
import os
from pathlib import Path

import pytest

from musegadget import reachy_voice
from musegadget.identity import Identity
from musegadget.link_client import Outcome
from musegadget.reachy_capabilities import Mode
from musegadget.reachy_local_backends import backends_for
from test_reachy_voice import FakeHardware, FakeSession, chat_event, mp3_tone, sentence_frame  # noqa: F401

np = pytest.importorskip("numpy")
pytest.importorskip("av")

PINS = Path(__file__).parent / "pins"
CHUNK = 320
WAKE_SAMPLE = np.float32(.75)
VOICE_SAMPLE = np.float32(.3)


class Recording:
    def __init__(self):
        self.events = []
        self.synthesized = []
        self.codes = {}
        self.labels = {}
        self.tts_requests = []

    def add(self, event):
        self.events.append(event)

    def code(self, label):
        if label not in self.codes:
            value = np.float32((len(self.codes) + 1) / 1024)
            self.codes[label] = value
            self.labels[float(value)] = label
        return self.codes[label]

    def label(self, samples):
        label = self.labels.get(float(samples[0])) if len(samples) else None
        return label or f"mp3:{self.tts_requests[-1]}"

    def plays(self):
        return sum(event.get("robot") == "play" for event in self.events)


class PinHardware(FakeHardware):
    def __init__(self, recording):
        super().__init__()
        self.recording = recording

    def run_command(self, name, params, timeout_ms):
        self.recording.add({"robot": "command", "name": name, "params": params})
        return super().run_command(name, params, timeout_ms)

    def set_state(self, state, *, expression=None):
        self.recording.add({"robot": "set_state", "state": state, "expression": expression})
        super().set_state(state, expression=expression)

    def play_audio(self, samples):
        event = {"robot": "play", "audio": self.recording.label(samples)}
        if self.recording.events[-1:] != [event]:
            self.recording.add(event)
        super().play_audio(samples)

    def clear_audio(self):
        self.recording.add({"robot": "clear_audio"})
        super().clear_audio()

    def cue_progress(self):
        self.recording.add({"robot": "cue_progress"})
        super().cue_progress()


class PinSession(FakeSession):
    registered_at = None

    def __init__(self, recording, muse, *, mp3=b"", owned=True, rejected=False):
        super().__init__(mp3)
        self.recording = recording
        self.muse = muse
        self.chat = "robot-chat" if owned else "main-chat"
        self.owned = owned
        self.rejected = rejected
        self.requests = 0
        self.scripts = []
        self.seq = 0

    async def run(self, stop):
        await stop.wait()
        return Outcome.STOPPED

    def _acknowledge(self):
        self.requests += 1
        if self.rejected:
            return {"ok": False, "status": 503}
        result = {"message_id": f"user-{self.requests}", "session_id": self.chat}
        if self.owned:
            result["is_thread"] = True
        self.scripts.append(asyncio.create_task(self.muse(self, self.requests)))
        return {"ok": True, "status": 200, "response": {"result": result}}

    async def send_voice(self, wav, session_id, **options):
        self.recording.add({"muse": "send_voice", "session_id": session_id, "options": options,
                            "wav_bytes": len(wav)})
        return self._acknowledge()

    async def send_chat(self, message, session_id=None, **options):
        self.recording.add({"muse": "send_chat", "session_id": session_id, "options": options,
                            "text": message})
        return self._acknowledge()

    async def subscribe_chat(self, session_id):
        self.recording.add({"muse": "subscribe_chat", "session_id": session_id})
        async for event in super().subscribe_chat(session_id):
            yield event

    async def stream_tts(self, message_id):
        self.recording.add({"muse": "stream_tts", "message_id": message_id})
        self.recording.tts_requests.append(message_id)
        async for chunk in super().stream_tts(message_id):
            yield chunk

    async def emit(self, name, message_id=None, *, request, **payload):
        self.seq += 1
        parent = payload.pop("parent", f"user-{request}")
        await self.events.put(chat_event(name, message_id, seq=self.seq, parent=parent,
                                         session_id=self.chat, **payload))
        await self.delivered.get()

    async def answer(self, request, content):
        await self.emit("delta.message_done", f"reply-{request}", request=request, content=content)
        await self.emit("task.status", request=request, status="completed")


class TextCodedVoice:
    def __init__(self, recording, name):
        self.recording = recording
        self.name = name

    def stream(self, text, output_rate):
        label = f"{self.name}:{text}"
        self.recording.synthesized.append(label)
        samples = np.full(80, self.recording.code(label), dtype=np.float32)

        async def audio():
            yield samples
        return audio()

    def prepare(self, text, output_rate):
        from musegadget.local_speech import PreparedSpeech
        return PreparedSpeech(self.stream(text, output_rate))


class Vad:
    def is_speech(self, pcm, rate):
        return bool(np.any(np.frombuffer(pcm, dtype="<i2")))


class Transcriber:
    def __init__(self, text):
        self.text = text

    async def transcribe(self, wav):
        return self.text


class FailingStreamingTranscriber:
    async def start(self):
        pass

    def open_turn(self):
        from musegadget.streaming_transcription import StreamingTranscriptionError

        class Turn:
            def feed(self, pcm):
                raise StreamingTranscriptionError("recognizer lost the turn")

            def abort(self):
                pass
        return Turn()


class ScriptedStreamingTranscriber:
    def __init__(self):
        from musegadget.streaming_transcription import TranscriptRevision
        self.partial = TranscriptRevision(0, "")

    def say(self, text):
        from musegadget.streaming_transcription import TranscriptRevision
        self.partial = TranscriptRevision(self.partial.revision + 1, text)

    async def start(self):
        pass

    def open_turn(self):
        transcriber = self

        class Turn:
            @property
            def partial(self):
                return transcriber.partial

            def feed(self, pcm):
                pass

            def finish(self):
                result = asyncio.get_running_loop().create_future()
                result.set_result(transcriber.partial.text)
                return result

            def abort(self):
                pass
        return Turn()


class Wake:
    phrase = "hey muse"

    def feed(self, samples):
        return bool(np.any(samples == WAKE_SAMPLE))

    def reset(self):
        pass

    def close(self):
        pass


class Rig:
    def __init__(self, monkeypatch):
        self.monkeypatch = monkeypatch
        self.recording = Recording()
        self.hardware = PinHardware(self.recording)
        self.session = None

    def service(self, muse, *, mp3=b"", owned=True, rejected=False, speech=None, progress_speech=None,
                transcriber=None, stream_replies=False, **fields):
        self.session = PinSession(self.recording, muse, mp3=mp3, owned=owned, rejected=rejected)
        self.monkeypatch.setattr(reachy_voice, "LinkSession", lambda **_: self.session)

        async def prepare_chat(session, vm):
            return "robot-chat"
        mode = Mode.MUSE_VOICE if speech is None else Mode.ON_ROBOT
        backends = functools.partial(backends_for, mode, speech=speech, progress_speech=progress_speech,
                                     transcriber=transcriber, stream_replies=stream_replies,
                                     wake="wake_detector" in fields)
        return reachy_voice.ReachyService(
            identity=Identity("02:00:00:ab:cd:ef"), executor=self.hardware, silence_s=.2,
            speech_gate=Vad(), backends=backends,
            **({"prepare_chat": prepare_chat, "owns_chat": True} if owned else {}), **fields)

    def voice(self, name="tts"):
        return TextCodedVoice(self.recording, name)

    async def speak_live(self, transcriber, script):
        for words, seconds in script:
            transcriber.say(words)
            for _ in range(round(seconds * 16000 / CHUNK)):
                self.hardware.samples.put(np.full(CHUNK, VOICE_SAMPLE, dtype=np.float32))
                await asyncio.sleep(CHUNK / 16000)
        self.say(voiced_s=0)

    def say(self, *, voiced_s=.5, wake=False):
        chunks = [np.full(CHUNK, WAKE_SAMPLE, dtype=np.float32)] if wake else []
        chunks += [np.full(CHUNK, VOICE_SAMPLE, dtype=np.float32)] * round(voiced_s * 16000 / CHUNK)
        chunks += [np.zeros(CHUNK, dtype=np.float32)] * round(.5 * 16000 / CHUNK)
        for chunk in chunks:
            self.hardware.samples.put(chunk.copy())

    async def wait_until_or_timeout(self, predicate, timeout=3):
        deadline = asyncio.get_running_loop().time() + timeout
        while not predicate() and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(.005)

    async def quiet(self, span=.6):
        seen = -1
        while seen != len(self.recording.events):
            seen = len(self.recording.events)
            await asyncio.sleep(span)


async def converse(rig, service, user):
    session = asyncio.create_task(service._session({"vm_id": "vm-1", "vm_auth_token": "token"}, {}))
    try:
        await rig.quiet()
        await user()
        await rig.quiet()
    finally:
        service.stop()
        await asyncio.wait_for(session, 5)
        await asyncio.gather(*rig.session.scripts)


def run_scenario(rig, service, user):
    asyncio.run(asyncio.wait_for(converse(rig, service, user), 20))
    return {"events": rig.recording.events, "synthesized": rig.recording.synthesized}


def render(record):
    lines = ["{"]
    for index, key in enumerate(("events", "synthesized")):
        items = [json.dumps(item, sort_keys=True) for item in record[key]]
        body = ",\n".join("  " + item for item in items)
        lines.append(f' "{key}": [' + ("\n" + body + "\n ]" if items else "]") + ("," if index == 0 else ""))
    lines.append("}")
    return "\n".join(lines) + "\n"


def check(name, record):
    path = PINS / f"{name}.json"
    actual = render(record)
    if os.environ.get("PIN_UPDATE") == "1":
        PINS.mkdir(exist_ok=True)
        path.write_text(actual)
    assert actual.splitlines() == path.read_text().splitlines()


def test_muse_voice_turn_sends_wav_and_plays_muse_mp3_with_marker_expression(monkeypatch, mp3_tone):
    rig = Rig(monkeypatch)

    async def muse(session, request):
        await session.answer(request, "Octopuses have three hearts. [reachy:surprised]")

    service = rig.service(muse, mp3=mp3_tone)

    async def user():
        rig.say()
    check("muse_voice", run_scenario(rig, service, user))


def test_on_robot_marker_turn_sets_up_main_chat_then_speaks_reply(monkeypatch):
    rig = Rig(monkeypatch)

    async def muse(session, request):
        reply = "Ready to talk. [reachy:nod]" if request == 1 else "You are doing great. [reachy:happy]"
        await session.answer(request, reply)

    service = rig.service(muse, owned=False, speech=rig.voice(), progress_speech=rig.voice("progress_tts"),
                          transcriber=Transcriber("Please say something nice."))

    async def user():
        rig.say()
    check("on_robot_marker", run_scenario(rig, service, user))


def test_task_status_trailing_the_answer_still_ends_idle_without_a_thinking_flash(monkeypatch):
    rig = Rig(monkeypatch)

    async def muse(session, request):
        reply = "Ready to talk. [reachy:nod]" if request == 1 else "You are doing great. [reachy:happy]"
        await session.emit("delta.message_done", f"reply-{request}", request=request, content=reply)
        await asyncio.sleep(.01)
        await session.emit("task.status", request=request, status="completed")

    service = rig.service(muse, owned=False, speech=rig.voice(), progress_speech=rig.voice("progress_tts"),
                          transcriber=Transcriber("Please say something nice."))

    async def user():
        rig.say()
    check("on_robot_marker", run_scenario(rig, service, user))


def test_on_robot_json_turn_speaks_each_sentence_with_its_expression(monkeypatch):
    rig = Rig(monkeypatch)

    async def muse(session, request):
        frames = [sentence_frame("Sunlight bends inside raindrops.", "curious"),
                  sentence_frame("Each color bends a little differently.", "happy"),
                  sentence_frame("That spreads white light into a bow.", "surprised")]
        await session.emit("delta.text_append", "reply-1", request=request, text=frames[0])
        await session.answer(request, "".join(frames))

    service = rig.service(muse, speech=rig.voice(), progress_speech=rig.voice("progress_tts"),
                          transcriber=Transcriber("How does a rainbow form?"), stream_replies=True)

    async def user():
        rig.say()
    check("on_robot_json", run_scenario(rig, service, user))


def test_wake_turn_strips_wake_phrase_and_prefixes_voice_context(monkeypatch):
    rig = Rig(monkeypatch)

    async def muse(session, request):
        await session.answer(request, "Sunny and warm all day. [reachy:happy]")

    service = rig.service(muse, owned=False, speech=rig.voice(), progress_speech=rig.voice("progress_tts"),
                          transcriber=Transcriber("Hey Muse, what's the weather tomorrow?"),
                          wake_detector=Wake(), wake_timeout_s=.3)

    async def user():
        rig.say(wake=True)
    check("wake_marker", run_scenario(rig, service, user))


def test_slow_turn_acknowledges_then_reports_backend_progress_before_answer(monkeypatch):
    rig = Rig(monkeypatch)

    async def muse(session, request):
        await rig.wait_until_or_timeout(lambda: rig.recording.plays() == 1)
        await session.emit("agent.status", request=request, parent=None,
                           activity_code="working", activity_text="Searching web")
        await rig.wait_until_or_timeout(lambda: rig.recording.plays() == 2)
        await session.answer(request, "Robots learned to fold laundry. [reachy:happy]")

    service = rig.service(muse, speech=rig.voice(), progress_speech=rig.voice("progress_tts"),
                          transcriber=Transcriber("Can you search for the latest robot news?"))

    async def user():
        rig.say()
    check("slow_progress", run_scenario(rig, service, user))


def test_empty_transcript_sends_nothing_and_returns_to_idle(monkeypatch):
    rig = Rig(monkeypatch)

    async def muse(session, request):
        raise AssertionError("an empty transcript must not reach Muse")

    service = rig.service(muse, speech=rig.voice(), progress_speech=rig.voice("progress_tts"),
                          transcriber=Transcriber(""))

    async def user():
        rig.say()
    check("empty_transcript", run_scenario(rig, service, user))


def test_streaming_transcription_failure_speaks_retry_cue(monkeypatch):
    rig = Rig(monkeypatch)

    async def muse(session, request):
        raise AssertionError("a failed transcription must not reach Muse")

    service = rig.service(muse, speech=rig.voice(), progress_speech=rig.voice("progress_tts"),
                          transcriber=FailingStreamingTranscriber())

    async def user():
        rig.say()
    check("transcription_retry_cue", run_scenario(rig, service, user))


def test_rejected_muse_request_ends_the_conversation_in_error(monkeypatch):
    rig = Rig(monkeypatch)

    async def muse(session, request):
        raise AssertionError("a rejected request has no reply")

    service = rig.service(muse, rejected=True, speech=rig.voice(), progress_speech=rig.voice("progress_tts"),
                          transcriber=Transcriber("Please say something nice."))

    async def user():
        rig.say()
    check("muse_rejects", run_scenario(rig, service, user))


def test_streaming_partials_tilt_and_nod_while_the_user_talks_and_never_reach_muse(monkeypatch):
    rig = Rig(monkeypatch)
    transcriber = ScriptedStreamingTranscriber()

    async def muse(session, request):
        await asyncio.sleep(.5)
        await session.answer(request, "Castles kept dragons out. [reachy:happy]")

    service = rig.service(muse, speech=rig.voice(), progress_speech=rig.voice("progress_tts"),
                          transcriber=transcriber)

    async def user():
        await rig.speak_live(transcriber, [
            ("tell me", .4), ("tell me about", .4), ("tell me about the old", .4),
            ("tell me about the old castles", .4), ("tell me about the old castles and", .8),
            ("tell me about the old castles and dragons", .5)])
    check("streaming_partials", run_scenario(rig, service, user))
