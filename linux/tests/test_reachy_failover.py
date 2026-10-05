"""The failover wrappers against scripted companion and on-robot parts."""

import asyncio

import pytest

from musegadget import reachy_failover
from musegadget.reachy_capabilities import CapabilityUnavailable, HeardAudio, SpokenLine
from musegadget.reachy_companion_client import CompanionHearingTurn, _HearStream
from musegadget.reachy_companion_protocol import HearEndpoint
from musegadget.reachy_failover import CompanionRoute, FailoverHearing, FailoverNarrator
from musegadget.voice_audio import SpeechAudio, SpeechEnd

np = pytest.importorskip("numpy")


class EndedLate:
    truncated = False

    def __init__(self, end_sample):
        self.end_sample = end_sample


class ScriptedTurn:
    """A hearing turn whose takes are scripted: a HeardAudio to return or an error to raise."""

    active = True
    speech_active = True
    partial = None

    def __init__(self, takes):
        self.fed = []
        self._takes = list(takes)

    def feed(self, samples):
        self.fed.append(samples.copy())

    def take(self):
        outcome = self._takes.pop(0) if self._takes else HeardAudio()
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def abort(self):
        pass


class ScriptedHearing:
    def __init__(self, *turns, available=True):
        self.available = available
        self.turns = list(turns)
        self.opened = []

    def open_turn(self, sample_rate, **options):
        self.opened.append(self.turns.pop(0))
        return self.opened[-1]


def test_a_late_endpoint_keeps_the_next_utterances_opening_for_a_replay():
    # The first utterance ends at sample 500; its endpoint arrives after 1000 samples were fed.
    companion = ScriptedHearing(ScriptedTurn([HeardAudio(EndedLate(500)), CapabilityUnavailable("gone")]))
    local = ScriptedHearing(ScriptedTurn([]))
    turn = FailoverHearing(companion, local, CompanionRoute(Link()).hearing).open_turn(16000)
    for chunk in range(10):
        turn.feed(np.full(100, chunk, dtype=np.float32))
    assert turn.take().ended is not None
    assert turn.take().ended is None
    turn.feed(np.full(100, 10, dtype=np.float32))
    replayed, = local.opened[0].fed
    assert replayed.tolist() == [float(n // 100) for n in range(500, 1100)]


class Connection:
    closed = False

    def __init__(self):
        self.hearing = {}

    def send(self, frame):
        pass


class HearingLink:
    def __init__(self):
        self.connection = Connection()
        self._next = -1

    def open_hear(self):
        self._next += 2
        stream = self.connection.hearing[self._next] = _HearStream(self.connection, self._next)
        return stream


class Recorder:
    active = False
    speech_active = False
    last_truncated = False

    def __init__(self, *events):
        self._events = list(events)

    def take_audio_events(self):
        return self._events.pop(0) if self._events else []

    def feed(self, samples):
        return None

    def reset(self):
        pass


def test_an_ended_turn_waiting_behind_another_is_returned_before_the_companion_fails():
    voice = SpeechAudio(b"\0\0" * 160)
    link = HearingLink()
    turn = CompanionHearingTurn(link, Recorder([voice, SpeechEnd(True), voice, SpeechEnd(True)], [voice]))
    assert turn.take().ended is None
    link.connection.hearing[1].inbox.append(HearEndpoint(1, "first", True))
    link.connection.hearing[3].inbox.append(HearEndpoint(3, "second", True))
    first = turn.take().ended
    link.connection.hearing[5].failure = CapabilityUnavailable("gone", link_down=False)
    second = turn.take().ended

    async def texts():
        return [(await ended.endpoint()).text for ended in (first, second)]
    assert asyncio.run(texts()) == ["first", "second"]
    with pytest.raises(CapabilityUnavailable, match="gone"):
        turn.take()


class Link:
    up = True


class InventingNarrator:
    available = True

    def __init__(self):
        self.asked = 0

    async def acknowledge(self, request):
        self.asked += 1
        return SpokenLine("It took 17 minutes.", None, "ack")


class RobotNarrator:
    async def acknowledge(self, request):
        return SpokenLine("Okay.", None, "ack")


def test_a_rested_narrator_is_asked_again_after_the_cooldown(monkeypatch):
    monkeypatch.setattr(reachy_failover, "COOLDOWN_S", .2)
    companion = InventingNarrator()
    route = CompanionRoute(Link())
    route.begin_turn()
    narrator = FailoverNarrator(companion, RobotNarrator(), route)

    async def run():
        said = [(await narrator.acknowledge("castles")).text for _ in range(4)]
        await asyncio.sleep(.3)
        said.append((await narrator.acknowledge("castles")).text)
        return said
    assert asyncio.run(run()) == ["Okay."] * 5
    assert companion.asked == 4


def test_rested_hearing_stays_on_the_robot_and_is_tried_again_after_the_cooldown(monkeypatch):
    monkeypatch.setattr(reachy_failover, "COOLDOWN_S", .2)
    gone = CapabilityUnavailable("the companion did not end the turn", link_down=False)
    lost = CapabilityUnavailable("the link dropped")
    companion = ScriptedHearing(*(ScriptedTurn([outcome]) for outcome in (gone, lost, gone, gone)), ScriptedTurn([]))
    local = ScriptedHearing(*(ScriptedTurn([]) for _ in range(6)))
    hearing = FailoverHearing(companion, local, CompanionRoute(Link()).hearing)

    def window():
        hearing.open_turn(16000).take()
        return (len(companion.opened), len(local.opened))

    async def run():
        heard = [window() for _ in range(5)]
        await asyncio.sleep(.3)
        return heard + [window()]
    # A lost link is no strike, so the third failure that rests hearing is the fourth window's. The
    # fifth window is heard on the robot alone, and after the cooldown the companion hears again.
    assert asyncio.run(run()) == [(1, 1), (2, 2), (3, 3), (4, 4), (4, 5), (5, 5)]
