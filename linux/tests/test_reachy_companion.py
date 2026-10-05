"""Companion mode against the in-process fake companion: whole turns, failover and the wire."""

import asyncio
import functools
import socket

import pytest

from fake_companion import TOKEN, FakeCompanion
from musegadget import reachy_voice
from musegadget.identity import Identity
from musegadget.reachy_capabilities import COMPANION_NOTICE, Expression, Mode
from musegadget.reachy_companion_client import CompanionLink, LinkState
from musegadget.reachy_companion_protocol import HearOpen, Narrate, Sender, SpeakStart, decode
from musegadget.reachy_local_backends import backends_for
from test_reachy_pins import CHUNK, VOICE_SAMPLE, PinHardware, PinSession, Recording, TextCodedVoice, Transcriber, Vad

np = pytest.importorskip("numpy")
pytest.importorskip("av")

VM_TOKEN = "vm-secret-7f3a91"
IDENTITY = Identity("02:00:00:ab:cd:ef")
QUESTION = "tell me about the old castles and dragons"
PARTIALS = ("tell me about", "tell me about the old castles", QUESTION)
PLAIN_PROMPT = "Answer in one to three short, plain spoken sentences"


class CompanionHardware(PinHardware):
    """Labels companion speech, which carries none of the local voice's text codes."""

    def play_audio(self, samples):
        if float(samples[0]) not in self.recording.labels:
            event = {"robot": "play", "audio": "companion"}
            if self.recording.events[-1:] != [event]:
                self.recording.add(event)
            self.played.append(samples.copy())
            return
        super().play_audio(samples)


class Conversation:
    """One robot in companion mode, its scripted Muse, and a user talking over room tone."""

    def __init__(self, monkeypatch, url, muse, *, owned=True, transcript=QUESTION, backoff_s=(.05, .2), **link):
        self.recording = Recording()
        self.hardware = CompanionHardware(self.recording)
        self.session = PinSession(self.recording, muse, owned=owned)
        monkeypatch.setattr(reachy_voice, "LinkSession", lambda **_: self.session)
        self.link = CompanionLink(url, token=TOKEN, insecure=True, out_rate=self.hardware.output_sample_rate,
                                  backoff_s=backoff_s, **link)

        async def prepare_chat(session, vm):
            return "robot-chat"
        backends = functools.partial(backends_for, Mode.COMPANION, speech=TextCodedVoice(self.recording, "tts"),
                                     progress_speech=TextCodedVoice(self.recording, "progress_tts"),
                                     transcriber=Transcriber(transcript), companion=self.link)
        self.service = reachy_voice.ReachyService(
            identity=IDENTITY, executor=self.hardware, silence_s=.2, speech_gate=Vad(), backends=backends,
            **({"prepare_chat": prepare_chat, "owns_chat": True} if owned else {}))
        self._voiced = 0

    async def run(self, user):
        tasks = [asyncio.ensure_future(self.link.run()), asyncio.ensure_future(self._room_tone())]
        session = asyncio.ensure_future(self.service._session({"vm_id": "vm-1", "vm_auth_token": VM_TOKEN}, {}))
        try:
            await self.quiet()
            await user()
            await self.quiet()
        finally:
            self.service.stop()
            await asyncio.wait_for(session, 5)
            await asyncio.gather(*self.session.scripts)
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _room_tone(self):
        """Microphone audio in real time: the user's voice when they talk, silence otherwise."""
        while True:
            voiced = bool(self._voiced)
            if voiced:
                self._voiced -= 1
            self.hardware.samples.put(np.full(CHUNK, VOICE_SAMPLE if voiced else 0, dtype=np.float32))
            await asyncio.sleep(CHUNK / 16000)

    async def speak(self, voiced_s):
        self._voiced += round(voiced_s * 16000 / CHUNK)
        await self.until(lambda: not self._voiced)

    async def until(self, predicate, timeout=5):
        deadline = asyncio.get_running_loop().time() + timeout
        while not predicate() and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(.01)
        assert predicate()

    async def quiet(self, span=.6, timeout=20):
        """Waits until the robot is back to idle and nothing else happens for ``span``.

        Silence alone is not enough. Playing the acknowledgement records no event for as long
        as ``span`` while the robot is still thinking about the reply.
        """
        deadline = asyncio.get_running_loop().time() + timeout
        seen = -1
        while seen != len(self.recording.events) or self.states()[-1:] != ["idle"]:
            assert asyncio.get_running_loop().time() < deadline, self.states()[-5:]
            seen = len(self.recording.events)
            await asyncio.sleep(span)

    def states(self):
        return [event["state"] for event in self.recording.events if event.get("robot") == "set_state"]

    def requests(self):
        return [event["text"] for event in self.recording.events if event.get("muse") == "send_chat"]

    def spoken(self):
        """What the robot played, in order: local voice labels and companion speech."""
        return [event["audio"] for event in self.recording.events if event.get("robot") == "play"]

    async def turns(self, count):
        """The user asks ``count`` questions, each after the last reply is spoken."""
        for turn in range(1, count + 1):
            await self.speak(.8)
            await self.until(lambda: len(self.session.scripts) == turn and self.session.scripts[-1].done(), 15)
            await self.quiet()
        self.transitions_before_stop = list(self.link.transitions)

    def notices(self):
        return [label for label in self.recording.synthesized if label == f"tts:{COMPANION_NOTICE}"]


def narrated_by_companion(companion):
    return [message.op.value for message in (decode(frame, sender=Sender.ROBOT) for frame in companion.received
                                             if isinstance(frame, str)) if isinstance(message, Narrate)]


def spoken_by_companion(companion):
    return [message.text for message in (decode(frame, sender=Sender.ROBOT) for frame in companion.received
                                         if isinstance(frame, str)) if isinstance(message, SpeakStart)]


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def scenario(coroutine):
    asyncio.run(asyncio.wait_for(coroutine, 30))


def answer(reply):
    async def muse(session, request):
        await session.answer(request, reply)
    return muse


def test_happy_turn_nods_along_to_partials_and_speaks_condensed_lines_with_their_expression(monkeypatch):
    async def run():
        # 7 frames of room tone before the onset, then 40 voiced frames: the companion ends the turn
        # on the last voiced frame, the way Smart Turn would.
        async with FakeCompanion(partials=PARTIALS, frames_per_partial=12, endpoint_after=46,
                                 expression=Expression.HAPPY) as companion:
            robot = Conversation(monkeypatch, companion.url[:-3],
                                 answer("Castles kept dragons out. They had tall walls."))
            await robot.run(lambda: robot.speak(.8))
            prompt, request = robot.requests()[0].rsplit("\n\n", 1)
            assert (len(robot.requests()), request) == (1, f"The user's spoken request is: {QUESTION}")
            assert PLAIN_PROMPT in prompt
            assert spoken_by_companion(companion) == [QUESTION, "Castles kept dragons out.", "They had tall walls."]
            assert {"robot": "gesture", "name": "tilt"} in robot.recording.events
            assert {"robot": "set_state", "state": "speaking", "expression": "happy"} in robot.recording.events
            assert robot.notices() == []
    scenario(run())


def test_unreachable_companion_is_announced_once_and_muse_is_set_up_for_on_robot_speech_once(monkeypatch):
    async def run():
        robot = Conversation(monkeypatch, f"ws://127.0.0.1:{free_port()}",
                             answer("Ready to talk. [reachy:nod]"), owned=False)

        async def user():
            await asyncio.sleep(1)
        await robot.run(user)
        setup, = robot.requests()
        assert setup.endswith(" For setup, say 'Ready to talk' and append [reachy:nod]. "
                              "Keep using this expression channel for subsequent spoken messages.")
        assert sorted(robot.spoken()) == [f"tts:{COMPANION_NOTICE}", "tts:Ready to talk."]
        assert robot.link.transitions[:2] == [LinkState.CONNECTING, LinkState.DOWN]
        assert LinkState.UP not in robot.link.transitions
    scenario(run())


def test_companion_dropping_mid_hearing_finishes_the_turn_on_the_robot_from_replayed_audio(monkeypatch):
    async def run():
        async with FakeCompanion(partials=PARTIALS, frames_per_partial=12,
                                 abort_after_frames_both_ways_since_welcome=20) as companion:
            robot = Conversation(monkeypatch, companion.url[:-3],
                                 answer("Castles kept dragons out. [reachy:happy]"), backoff_s=(10, 10))

            async def user():
                await robot.until(lambda: robot.link.state is LinkState.UP)
                await robot.speak(.8)
            await robot.run(user)
            assert robot.requests() == [QUESTION]
            assert robot.spoken() == [f"tts:{COMPANION_NOTICE}", "tts:Castles kept dragons out."]
    scenario(run())


def test_companion_dropping_mid_speech_has_the_robot_say_the_whole_line_again(monkeypatch):
    async def run():
        async with FakeCompanion(partials=PARTIALS, frames_per_partial=12, drop_mid_speech=True) as companion:
            robot = Conversation(monkeypatch, companion.url[:-3], answer("Castles kept dragons out."),
                                 backoff_s=(10, 10))
            await robot.run(lambda: robot.speak(.8))
            # The fake's narrator acknowledges by echoing the request; that is the line it drops.
            assert spoken_by_companion(companion) == [QUESTION]
            assert robot.spoken() == ["companion", f"tts:{QUESTION}", f"tts:{COMPANION_NOTICE}",
                                      "tts:Castles kept dragons out."]
    scenario(run())


def test_a_fabricated_figure_is_replaced_by_the_robots_own_split_of_the_reply(monkeypatch):
    async def run():
        async with FakeCompanion(partials=PARTIALS, frames_per_partial=12, fabricate=True) as companion:
            robot = Conversation(monkeypatch, companion.url[:-3],
                                 answer("Castles kept dragons out. They had tall walls."))
            await robot.run(lambda: robot.speak(.8))
            # The echoed acknowledgement carries the fabrication too, so Reachy skips it.
            assert spoken_by_companion(companion) == ["Castles kept dragons out.", "They had tall walls."]
            assert robot.notices() == []
    scenario(run())


def test_recovered_companion_gets_the_plain_setup_at_the_next_turn_not_the_running_one(monkeypatch):
    async def run():
        port = free_port()
        started = {}

        async def muse(session, request):
            if request == 1:
                started["companion"] = await FakeCompanion(port=port, partials=PARTIALS,
                                                           frames_per_partial=12).__aenter__()
                await robot.until(lambda: robot.link.state is LinkState.UP)
                await session.answer(request, "Castles kept dragons out. [reachy:happy]")
            else:
                await session.answer(request, "They had tall walls.")
        robot = Conversation(monkeypatch, f"ws://127.0.0.1:{port}", muse, healthy_s=.5)

        async def user():
            await robot.until(lambda: robot.link.state is LinkState.DOWN)
            await robot.speak(.8)
            await robot.until(lambda: len(robot.session.scripts) == 1 and robot.session.scripts[0].done())
            await robot.quiet()
            await robot.speak(.8)
        try:
            await robot.run(user)
        finally:
            await started["companion"].__aexit__(None, None, None)
        first, second = robot.requests()
        assert first == QUESTION
        assert PLAIN_PROMPT in second and second.endswith(f"The user's spoken request is: {QUESTION}")
        # The first reply was requested with the marker prompt, so the marker parser reads it and the
        # robot's voice speaks it, even though the companion is back by the time it is spoken.
        assert {"robot": "set_state", "state": "speaking", "expression": "happy"} in robot.recording.events
        assert "tts:Castles kept dragons out." in robot.spoken()
        assert spoken_by_companion(started["companion"]) == [QUESTION, "They had tall walls."]
        assert robot.notices() == [f"tts:{COMPANION_NOTICE}"]
    scenario(run())


def test_nothing_that_identifies_muse_or_the_robot_crosses_to_the_companion(monkeypatch):
    async def run():
        async with FakeCompanion(partials=PARTIALS, frames_per_partial=12) as companion:
            robot = Conversation(monkeypatch, companion.url[:-3], answer("Castles kept dragons out."))
            await robot.run(lambda: robot.speak(.8))
            assert spoken_by_companion(companion) == [QUESTION, "Castles kept dragons out."]
            secrets = ("robot-chat", "user-1", "reply-1", IDENTITY.node_id, VM_TOKEN)
            leaks = [(secret, frame) for frame in companion.received for secret in secrets
                     if (secret if isinstance(frame, str) else secret.encode()) in frame]
            assert leaks == []
    scenario(run())


def test_a_companion_that_fails_after_every_welcome_is_announced_once_and_never_flips_the_style(monkeypatch):
    async def run():
        async with FakeCompanion(fail_after_welcome=True) as companion:
            robot = Conversation(monkeypatch, companion.url[:-3], answer("Castles kept dragons out."))
            styles = []

            async def user():
                for _ in range(300):
                    if styles[-1:] != [robot.link.up]:
                        styles.append(robot.link.up)
                    await asyncio.sleep(.02)
            await robot.run(user)
            assert len(companion.hellos) > 5
            assert robot.notices() == [f"tts:{COMPANION_NOTICE}"]
            assert styles in ([False], [True, False])
    scenario(run())


def test_three_ungrounded_narrations_rest_only_the_narrator(monkeypatch):
    async def run():
        async with FakeCompanion(partials=PARTIALS, frames_per_partial=12, fabricate=True) as companion:
            robot = Conversation(monkeypatch, companion.url[:-3],
                                 answer("Castles kept dragons out. They had tall walls."))
            await robot.run(lambda: robot.turns(3))
            # Turn one strikes the acknowledgement and the lines, turn two's acknowledgement is the
            # third strike, and the narrator rests for the rest of the test.
            assert narrated_by_companion(companion) == ["acknowledge", "lines", "acknowledge"]
            assert spoken_by_companion(companion) == ["Castles kept dragons out.", "They had tall walls."] * 3
            assert [request.endswith(QUESTION) for request in robot.requests()] == [True] * 3
            assert robot.notices() == []
            assert robot.transitions_before_stop == [LinkState.CONNECTING, LinkState.UP]
    scenario(run())


def test_companion_speech_that_never_starts_rests_only_the_companion_voice(monkeypatch):
    async def run():
        async with FakeCompanion(partials=PARTIALS, frames_per_partial=12, mute=True) as companion:
            async def muse(session, request):
                await asyncio.sleep(2.5)
                walls = ("tall", "thick")[request - 1]
                await session.answer(request, f"Castles kept dragons out. They had {walls} walls.")
            robot = Conversation(monkeypatch, companion.url[:-3], muse)
            await robot.run(lambda: robot.turns(2))
            # The acknowledgement and both lines of turn one are three silent lines in a row.
            assert spoken_by_companion(companion) == [QUESTION, "Castles kept dragons out.", "They had tall walls."]
            assert [line for line in robot.spoken() if "walls" in line] == ["tts:They had tall walls.",
                                                                             "tts:They had thick walls."]
            assert narrated_by_companion(companion).count("lines") == 2
            assert robot.notices() == []
            assert robot.transitions_before_stop == [LinkState.CONNECTING, LinkState.UP]
    scenario(run())


LOCAL_REQUEST = "The user's spoken request is: what the robot heard"


def test_a_turn_the_companion_never_ends_is_finished_on_the_robot_within_the_deadline(monkeypatch):
    from musegadget.reachy_companion_client import ENDPOINT_WAIT_S

    async def run():
        async with FakeCompanion(partials=PARTIALS, frames_per_partial=12, no_endpoint=True) as companion:
            robot = Conversation(monkeypatch, companion.url[:-3], answer("Castles kept dragons out."),
                                 transcript="what the robot heard")
            waited = {}

            async def user():
                await robot.until(lambda: robot.link.state is LinkState.UP)
                await robot.speak(.8)
                stopped = asyncio.get_running_loop().time()
                await robot.until(lambda: robot.requests(), ENDPOINT_WAIT_S + 2)
                waited["s"] = asyncio.get_running_loop().time() - stopped
            await robot.run(user)
            assert waited["s"] < ENDPOINT_WAIT_S + 1
            assert [request.endswith(LOCAL_REQUEST) for request in robot.requests()] == [True]
    scenario(run())


def test_a_companion_that_hears_nothing_mid_utterance_is_replaced_by_the_robot(monkeypatch):
    async def run():
        async with FakeCompanion(partials=PARTIALS, frames_per_partial=12, stall=True) as companion:
            robot = Conversation(monkeypatch, companion.url[:-3], answer("Castles kept dragons out."),
                                 transcript="what the robot heard")
            waited = {}

            async def user():
                await robot.until(lambda: robot.link.state is LinkState.UP)
                await robot.speak(4)
                stopped = asyncio.get_running_loop().time()
                await robot.until(lambda: robot.requests(), 3)
                waited["s"] = asyncio.get_running_loop().time() - stopped
            await robot.run(user)
            assert waited["s"] < 1
            assert [request.endswith(LOCAL_REQUEST) for request in robot.requests()] == [True]
    scenario(run())


def test_a_reply_too_long_to_narrate_is_spoken_whole_by_the_robot_without_resting_the_narrator(monkeypatch):
    walls = f"Castles kept {'very ' * 900}tall walls."
    moats = f"Their moats were {'very ' * 900}deep."

    async def muse(session, request):
        await session.answer(request, f"{walls} {moats}" if request <= 3 else "Castles kept dragons out.")

    async def run():
        async with FakeCompanion(partials=PARTIALS, frames_per_partial=12) as companion:
            robot = Conversation(monkeypatch, companion.url[:-3], muse)
            await robot.run(lambda: robot.turns(4))
            assert narrated_by_companion(companion) == ["acknowledge"] * 4 + ["lines"]
            assert spoken_by_companion(companion) == [QUESTION] * 4 + ["Castles kept dragons out."]
            assert [line for line in robot.spoken() if line != "companion"] == [f"tts:{walls}", f"tts:{moats}"] * 3
            assert robot.notices() == []
    scenario(run())
