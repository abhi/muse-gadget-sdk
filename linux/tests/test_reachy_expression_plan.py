import pytest

from musegadget.reachy_capabilities import Expression
from musegadget.reachy_expression_plan import (
    AnswerArrived, CaptureGap, ExpressionPlanner, MuseStatus, OutputIdle, Partial, Recording, SetState,
    Started, State, TurnDone, TurnStarted, UserSpeech, WakeClosed, WakeHeard, Working,
)

IDLE = SetState(State.IDLE)
LISTENING = SetState(State.LISTENING)
THINKING = SetState(State.THINKING)
TILT = SetState(State.LISTENING, Expression.LISTENING)
NOD = SetState(State.LISTENING, Expression.NOD)
GOT_IT = SetState(State.THINKING, Expression.NOD)
CURIOUS = SetState(State.THINKING, Expression.CURIOUS)
PONDER = SetState(State.THINKING, Expression.THINKING)


def plan(script):
    planner = ExpressionPlanner()
    return [(now, planner.on(event, now)) for now, event in script]


RESTING_STATES = {
    "startup is idle": (
        [(0, Started())],
        [(0, (IDLE,))]),
    "wake perks up and listens": (
        [(0, WakeHeard())],
        [(0, (SetState(State.LISTENING, Expression.HAPPY),))]),
    "an expired wake window goes idle": (
        [(0, WakeClosed())],
        [(0, (IDLE,))]),
    "a capture gap idles only when nothing is open": (
        [(0, CaptureGap(wake_open=False, muted=False, turn_running=False)),
         (1, CaptureGap(wake_open=True, muted=False, turn_running=False)),
         (2, CaptureGap(wake_open=False, muted=True, turn_running=False)),
         (3, CaptureGap(wake_open=False, muted=False, turn_running=True))],
        [(0, (IDLE,)), (1, ()), (2, ()), (3, ())]),
    "recording follows the recorder unless a turn runs": (
        [(0, Recording(active=True, turn_running=False)),
         (1, Recording(active=False, turn_running=False)),
         (2, Recording(active=True, turn_running=True))],
        [(0, (LISTENING,)), (1, (IDLE,)), (2, ())]),
    "user speech over a busy robot listens, then thinks when work remains": (
        [(0, UserSpeech(speaking=True, output=True, turn_running=False, turns_waiting=False)),
         (1, UserSpeech(speaking=True, output=False, turn_running=True, turns_waiting=False)),
         (2, UserSpeech(speaking=True, output=False, turn_running=False, turns_waiting=True)),
         (3, UserSpeech(speaking=False, output=False, turn_running=False, turns_waiting=True)),
         (4, UserSpeech(speaking=False, output=True, turn_running=True, turns_waiting=False)),
         (5, UserSpeech(speaking=False, output=False, turn_running=False, turns_waiting=False))],
        [(0, (LISTENING,)), (1, (LISTENING,)), (2, ()), (3, (THINKING,)), (4, ()), (5, ())]),
    "work shows thinking unless Reachy speaks or the user talks": (
        [(0, Working(output=False, user_speaking=False)),
         (1, Working(output=True, user_speaking=False)),
         (2, Working(output=False, user_speaking=True))],
        [(0, (THINKING,)), (1, ()), (2, ())]),
    "a finished turn listens while a recording or wake window is open": (
        [(0, TurnDone(turns_waiting=False, output=False, recording=True, wake_open=False)),
         (1, TurnDone(turns_waiting=False, output=False, recording=False, wake_open=True)),
         (2, TurnDone(turns_waiting=False, output=False, recording=False, wake_open=False)),
         (3, TurnDone(turns_waiting=True, output=False, recording=False, wake_open=False)),
         (4, TurnDone(turns_waiting=False, output=True, recording=False, wake_open=False))],
        [(0, (LISTENING,)), (1, (LISTENING,)), (2, (IDLE,)), (3, ()), (4, ())]),
    "drained output returns to whatever comes next": (
        [(0, OutputIdle(user_speaking=True, turn_open=True, wake_open=False)),
         (1, OutputIdle(user_speaking=False, turn_open=True, wake_open=True)),
         (2, OutputIdle(user_speaking=False, turn_open=False, wake_open=True)),
         (3, OutputIdle(user_speaking=False, turn_open=False, wake_open=False))],
        [(0, (LISTENING,)), (1, (THINKING,)), (2, (LISTENING,)), (3, (IDLE,))]),
}


@pytest.mark.parametrize("script, expected", RESTING_STATES.values(), ids=RESTING_STATES.keys())
def test_resting_states(script, expected):
    assert plan(script) == expected


LISTENING_GESTURES = {
    "a tilt needs three new words and a quiet 1.5 s": (
        [(0, Recording(active=True, turn_running=False)),
         (.2, Partial("what is", 1, output=False)),
         (.4, Partial("what is the", 2, output=False)),
         (.6, Partial("what is the tallest mountain", 3, output=False)),
         (1.0, Partial("what is the tallest mountain in", 4, output=False)),
         (1.9, Partial("what is the tallest mountain in the world", 5, output=False))],
        [(0, (LISTENING,)), (.2, ()), (.4, (TILT,)), (.6, ()), (1.0, ()), (1.9, (TILT,))]),
    "a clause end nods at most every 3 s": (
        [(0, Recording(active=True, turn_running=False)),
         (.5, Partial("I went out and", 1, output=False)),
         (2.5, Partial("I went out and it rained so", 2, output=False)),
         (3.6, Partial("I went out and it rained so", 2, output=False)),
         (4.1, Partial("I went out and it rained so", 2, output=False)),
         (6.0, Partial("I went out and it rained so", 2, output=False))],
        [(0, (LISTENING,)), (.5, (NOD,)), (2.5, (TILT,)), (3.6, ()), (4.1, (NOD,)), (6.0, ())]),
    "a comma ends a clause": (
        [(0, Recording(active=True, turn_running=False)),
         (.3, Partial("well,", 1, output=False))],
        [(0, (LISTENING,)), (.3, (NOD,))]),
    "a pause nods once per transcript revision": (
        [(0, Recording(active=True, turn_running=False)),
         (.1, Partial("tell me", 1, output=False)),
         (.6, Partial("tell me", 1, output=False)),
         (.7, Partial("tell me", 1, output=False)),
         (5.0, Partial("tell me", 1, output=False))],
        [(0, (LISTENING,)), (.1, ()), (.6, ()), (.7, (NOD,)), (5.0, ())]),
    "nothing moves while Reachy speaks, then gestures resume": (
        [(0, UserSpeech(speaking=True, output=True, turn_running=False, turns_waiting=False)),
         (.5, Partial("stop stop stop and", 1, output=True)),
         (1.5, Partial("stop stop stop and", 1, output=True)),
         (2.0, Partial("stop stop stop and wait", 2, output=False))],
        [(0, (LISTENING,)), (.5, ()), (1.5, ()), (2.0, (TILT,))]),
    "partials move Reachy only while it listens": (
        [(0, Working(output=False, user_speaking=False)),
         (.5, Partial("one two three and", 1, output=False)),
         (1, Started()),
         (1.5, Partial("one two three and four", 2, output=False))],
        [(0, (THINKING,)), (.5, ()), (1, (IDLE,)), (1.5, ())]),
    "the wake greeting counts as a gesture": (
        [(0, WakeHeard()),
         (1.0, Partial("one two three", 1, output=False)),
         (1.5, Partial("one two three four", 2, output=False))],
        [(0, (SetState(State.LISTENING, Expression.HAPPY),)), (1.0, ()), (1.5, (TILT,))]),
    "a new utterance starts its word count over": (
        [(0, Recording(active=True, turn_running=False)),
         (.1, Partial("one two three", 3, output=False)),
         (2.0, Partial("four", 1, output=False)),
         (2.2, Partial("four five six", 2, output=False))],
        [(0, (LISTENING,)), (.1, (TILT,)), (2.0, ()), (2.2, (TILT,))]),
}

THINKING_MOTION = {
    "each new Muse activity alternates the thinking motion, at most every 4 s": (
        [(0, TurnStarted(output=False, user_speaking=False)),
         (.5, MuseStatus("working", "Searching the web now.", output=False)),
         (.6, MuseStatus("working", "Searching the web now.", output=False)),
         (1.0, MuseStatus("working", "Checking a website now.", output=False)),
         (4.5, MuseStatus("working", "Checking a website now.", output=False)),
         (9.0, MuseStatus("responding", None, output=False))],
        [(0, (THINKING,)), (.5, (CURIOUS,)), (.6, ()), (1.0, ()), (4.5, (PONDER,)), (9.0, (CURIOUS,))]),
    "speech and a talking user hold the thinking motion": (
        [(0, TurnStarted(output=False, user_speaking=False)),
         (.5, MuseStatus("working", "Searching the web now.", output=True)),
         (.6, UserSpeech(speaking=True, output=False, turn_running=True, turns_waiting=False)),
         (.7, MuseStatus("working", "Searching the web now.", output=False)),
         (.8, UserSpeech(speaking=False, output=False, turn_running=True, turns_waiting=False)),
         (.9, MuseStatus("working", "Searching the web now.", output=False))],
        [(0, (THINKING,)), (.5, ()), (.6, (LISTENING,)), (.7, ()), (.8, (THINKING,)), (.9, (CURIOUS,))]),
    "a new turn starts the alternation over": (
        [(0, TurnStarted(output=False, user_speaking=False)),
         (.5, MuseStatus("working", "Searching the web now.", output=False)),
         (10, TurnStarted(output=False, user_speaking=False)),
         (10.1, MuseStatus("working", "Searching the web now.", output=False))],
        [(0, (THINKING,)), (.5, (CURIOUS,)), (10, (THINKING,)), (10.1, (CURIOUS,))]),
}

GOT_IT_NODS = {
    "the answer gets one nod per turn": (
        [(0, TurnStarted(output=False, user_speaking=False)),
         (2, AnswerArrived(output=False)),
         (3, AnswerArrived(output=False)),
         (4, TurnStarted(output=False, user_speaking=False)),
         (6, AnswerArrived(output=False))],
        [(0, (THINKING,)), (2, (GOT_IT,)), (3, ()), (4, (THINKING,)), (6, (GOT_IT,))]),
    "an answer during speech is not nodded at later": (
        [(0, TurnStarted(output=False, user_speaking=False)),
         (1, AnswerArrived(output=True)),
         (2, AnswerArrived(output=False))],
        [(0, (THINKING,)), (1, ()), (2, ())]),
    "a gesture in the last second suppresses the nod": (
        [(0, Recording(active=True, turn_running=False)),
         (.1, Partial("one two three", 1, output=False)),
         (.2, TurnStarted(output=False, user_speaking=False)),
         (.9, AnswerArrived(output=False))],
        [(0, (LISTENING,)), (.1, (TILT,)), (.2, (THINKING,)), (.9, ())]),
    "a gesture over a second ago does not": (
        [(0, Recording(active=True, turn_running=False)),
         (.1, Partial("one two three", 1, output=False)),
         (.2, TurnStarted(output=False, user_speaking=False)),
         (1.1, AnswerArrived(output=False))],
        [(0, (LISTENING,)), (.1, (TILT,)), (.2, (THINKING,)), (1.1, (GOT_IT,))]),
    "no nod unless Reachy is thinking": (
        [(0, Started()),
         (1, AnswerArrived(output=False))],
        [(0, (IDLE,)), (1, ())]),
}


@pytest.mark.parametrize("script, expected", LISTENING_GESTURES.values(), ids=LISTENING_GESTURES.keys())
def test_listening_gestures(script, expected):
    assert plan(script) == expected


@pytest.mark.parametrize("script, expected", THINKING_MOTION.values(), ids=THINKING_MOTION.keys())
def test_thinking_motion(script, expected):
    assert plan(script) == expected


@pytest.mark.parametrize("script, expected", GOT_IT_NODS.values(), ids=GOT_IT_NODS.keys())
def test_got_it_nod(script, expected):
    assert plan(script) == expected
