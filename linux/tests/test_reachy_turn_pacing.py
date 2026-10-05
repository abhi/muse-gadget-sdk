"""When Reachy speaks its own lines around a Muse answer: acknowledgements, progress and busy cues."""

import asyncio
import time

import pytest

from musegadget import reachy_local_backends, reachy_voice
from musegadget.reachy_capabilities import Mode
from musegadget.reachy_follow_up import is_small_talk
from musegadget.reachy_local_backends import backends_for
from musegadget.reachy_voice import VoiceConversation
from test_reachy_voice import FakeHardware, FakeSession, cancel_task, chat_event

np = pytest.importorskip("numpy")


@pytest.mark.parametrize("text, expected", [
    ("Hi!", True),
    ("Hello Muse.", True),
    ("How are you?", True),
    ("Hi Muse, how are you doing today?", True),
    ("Good morning, Reachy!", True),
    ("Thank you so much!", True),
    ("Thanks.", True),
    ("Yes.", True),
    ("No, thanks.", True),
    ("Okay.", True),
    ("Tell me more", True),
    ("What's the weather tomorrow?", False),
    ("How are you supposed to cook rice?", False),
    ("Can you search for the latest robot news?", False),
    ("Thanks, now what's the weather in Paris?", False),
    ("", True),
])
def test_small_talk_is_greetings_thanks_yes_no_or_three_words_at_most(text, expected):
    assert is_small_talk(text) is expected


class Labels:
    """Distinct audio for each line, so the order the speaker played them can be read back."""

    def __init__(self):
        self.codes = {}

    def samples(self, text, count=80):
        code = self.codes.setdefault(text, np.float32((len(self.codes) + 1) / 1024))
        return np.full(count, code, dtype=np.float32)

    def text(self, chunk):
        return next(text for text, code in self.codes.items() if code == chunk[0])


class PacedHardware(FakeHardware):
    """A speaker that takes real time per chunk, recording when each line began."""

    def __init__(self, labels, chunk_s=0.0):
        super().__init__()
        self.labels = labels
        self.chunk_s = chunk_s
        self.timeline = []

    def play_audio(self, samples):
        text = self.labels.text(samples)
        if not self.timeline or self.timeline[-1][0] != text:
            self.timeline.append((text, time.monotonic()))
        super().play_audio(samples)
        time.sleep(self.chunk_s)

    def spoken(self):
        return [text for text, _ in self.timeline]


def conversation_for(labels, hardware, request, *, ack_chunks=1):
    class Recognition:
        async def transcribe(self, wav):
            return request

    class Speech:
        async def stream(self, text, rate):
            yield labels.samples(text)

    def build(session):
        conversation = VoiceConversation(session, hardware, backends=backends_for(
            Mode.ON_ROBOT, session, speech=Speech(), progress_speech=Speech(), transcriber=Recognition()))
        conversation.backends.voice.phrases = {
            phrase: tuple(labels.samples(phrase) for _ in range(ack_chunks))
            for phrase in reachy_local_backends.ACKNOWLEDGEMENTS}
        return conversation
    return build


def answer_after(delay_s, reply, *, when=None):
    def send_hook(session):
        async def answer():
            if when is not None:
                while not when():
                    await asyncio.sleep(.005)
            else:
                await asyncio.sleep(delay_s)
            await session.events.put(chat_event("delta.message_done", "reply-1", content=reply))
            await session.delivered.get()
            await session.events.put(chat_event("task.status", status="completed"))
            await session.delivered.get()
        session.scripts.append(asyncio.create_task(answer()))
        return asyncio.sleep(0)
    return send_hook


def run_turn(monkeypatch, build, send_hook, *, timeout=6):
    monkeypatch.setattr(reachy_voice, "ECHO_TAIL_S", 0)

    async def scenario():
        session = FakeSession(send_hook=send_hook)
        session.scripts = []
        conversation = build(session)
        subscriber = asyncio.create_task(conversation._subscribe())
        await session.chat_subscribed.wait()
        dispatched = time.monotonic()
        await conversation.turn(b"voice")
        await asyncio.gather(*session.scripts)
        await cancel_task(subscriber)
        return dispatched

    return asyncio.run(asyncio.wait_for(scenario(), timeout))


def test_small_talk_is_not_acknowledged_even_when_muse_is_slow(monkeypatch):
    labels = Labels()
    hardware = PacedHardware(labels)
    run_turn(monkeypatch, conversation_for(labels, hardware, "How are you?"),
             answer_after(1.5, "I'm well. [reachy:happy]"))
    assert hardware.spoken() == ["I'm well."]


def test_an_answer_within_the_grace_plays_without_an_acknowledgement(monkeypatch):
    labels = Labels()
    hardware = PacedHardware(labels)
    run_turn(monkeypatch, conversation_for(labels, hardware, "What's the weather tomorrow?"),
             answer_after(.4, "Sunny all day. [reachy:happy]"))
    assert hardware.spoken() == ["Sunny all day."]


def test_a_slow_answer_is_acknowledged_once_the_grace_has_passed(monkeypatch):
    labels = Labels()
    hardware = PacedHardware(labels)
    dispatched = run_turn(monkeypatch, conversation_for(labels, hardware, "What's the weather tomorrow?"),
                          answer_after(1.6, "Sunny all day. [reachy:happy]"))
    assert hardware.spoken() == ["Let me check the weather.", "Sunny all day."]
    ack_started = hardware.timeline[0][1] - dispatched
    assert reachy_voice.ACK_GRACE_S <= ack_started < reachy_voice.ACK_GRACE_S + .3


def test_an_answer_arriving_mid_acknowledgement_lets_it_finish_then_plays_at_once(monkeypatch):
    labels = Labels()
    hardware = PacedHardware(labels, chunk_s=.05)
    ack = "Let me check the weather."
    run_turn(monkeypatch, conversation_for(labels, hardware, "What's the weather tomorrow?", ack_chunks=10),
             answer_after(None, "Sunny all day. [reachy:happy]", when=lambda: hardware.spoken() == [ack]))
    assert hardware.spoken() == [ack, "Sunny all day."]
    assert sum(1 for chunk in hardware.played if labels.text(chunk) == ack) == 10
    finished_ack = hardware.timeline[0][1] + 10 * .05
    assert hardware.timeline[1][1] - finished_ack < .3
