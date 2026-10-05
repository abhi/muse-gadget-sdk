import pytest

from musegadget.reachy_capabilities import Expression, Partial
from musegadget.reachy_expression_plan import (
    AnswerArrived, CaptureGap, ExpressionPlanner, Gesture, Heard, MuseStatus, OutputIdle, Recording, SetState,
    Started, State, TurnDone, TurnStarted, UserSpeech, WakeClosed, WakeHeard, Working,
)

IDLE = SetState(State.IDLE)
LISTENING = SetState(State.LISTENING)
THINKING = SetState(State.THINKING)
TILT = Gesture("tilt")
NOD = Gesture("nod")
CURIOUS = SetState(State.THINKING, Expression.CURIOUS)
PONDER = SetState(State.THINKING, Expression.THINKING)


def heard(text, revision, *, output=False, speaking=True, utterance=1):
    return Heard(Partial(text, revision, utterance), speech_active=speaking, output=output)


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
    "a tilt needs three new words and the previous gesture finished": (
        [(0, Recording(active=True, turn_running=False)),
         (.2, heard("what is", 1, output=False)),
         (.4, heard("what is the", 2, output=False)),
         (.6, heard("what is the tallest mountain", 3, output=False)),
         (1.0, heard("what is the tallest mountain in", 4, output=False)),
         (1.9, heard("what is the tallest mountain in the world", 5, output=False)),
         (2.0, heard("what is the tallest mountain in the world", 5, output=False))],
        [(0, (LISTENING,)), (.2, ()), (.4, (TILT,)), (.6, ()), (1.0, ()), (1.9, ()), (2.0, (TILT,))]),
    "a clause end nods at most every 3 s": (
        [(0, Recording(active=True, turn_running=False)),
         (.5, heard("I went out and", 1, output=False)),
         (2.5, heard("I went out and it rained so", 2, output=False)),
         (3.6, heard("I went out and it rained so", 2, output=False)),
         (4.2, heard("I went out and it rained so", 2, output=False)),
         (6.0, heard("I went out and it rained so", 2, output=False))],
        [(0, (LISTENING,)), (.5, (NOD,)), (2.5, (TILT,)), (3.6, ()), (4.2, (NOD,)), (6.0, ())]),
    "a comma ends a clause": (
        [(0, Recording(active=True, turn_running=False)),
         (.3, heard("well,", 1, output=False))],
        [(0, (LISTENING,)), (.3, (NOD,))]),
    "0.6 s of silence nods once per transcript revision": (
        [(0, Recording(active=True, turn_running=False)),
         (.1, heard("tell me", 1)),
         (.5, heard("tell me", 1, speaking=False)),
         (1.0, heard("tell me", 1, speaking=False)),
         (1.1, heard("tell me", 1, speaking=False)),
         (5.0, heard("tell me", 1, speaking=False))],
        [(0, (LISTENING,)), (.1, ()), (.5, ()), (1.0, ()), (1.1, (NOD,)), (5.0, ())]),
    "talking resets the silence": (
        [(0, Recording(active=True, turn_running=False)),
         (.1, heard("tell me", 1, speaking=False)),
         (.5, heard("tell me", 1)),
         (.8, heard("tell me", 1, speaking=False))],
        [(0, (LISTENING,)), (.1, ()), (.5, ()), (.8, ())]),
    "an unchanged transcript does not nod while the user keeps talking": (
        [(0, Recording(active=True, turn_running=False)),
         (.1, heard("tell me", 1)),
         (.8, heard("tell me", 1)),
         (3.0, heard("tell me", 1))],
        [(0, (LISTENING,)), (.1, ()), (.8, ()), (3.0, ())]),
    "nothing moves while Reachy speaks, then gestures resume": (
        [(0, UserSpeech(speaking=True, output=True, turn_running=False, turns_waiting=False)),
         (.5, heard("stop stop stop and", 1, output=True)),
         (1.5, heard("stop stop stop and", 1, output=True)),
         (2.0, heard("stop stop stop and wait", 2, output=False))],
        [(0, (LISTENING,)), (.5, ()), (1.5, ()), (2.0, (TILT,))]),
    "partials move Reachy only while it listens": (
        [(0, Working(output=False, user_speaking=False)),
         (.5, heard("one two three and", 1, output=False)),
         (1, Started()),
         (1.5, heard("one two three and four", 2, output=False))],
        [(0, (THINKING,)), (.5, ()), (1, (IDLE,)), (1.5, ())]),
    "the wake greeting counts as a gesture": (
        [(0, WakeHeard()),
         (1.0, heard("one two three", 1, output=False)),
         (1.5, heard("one two three four", 2, output=False)),
         (1.6, heard("one two three four", 2, output=False))],
        [(0, (SetState(State.LISTENING, Expression.HAPPY),)), (1.0, ()), (1.5, ()), (1.6, (TILT,))]),
    "a new utterance starts its word count over": (
        [(0, Recording(active=True, turn_running=False)),
         (.1, heard("one two three", 3, output=False)),
         (2.0, heard("four", 1, utterance=2)),
         (2.2, heard("four five six", 2, utterance=2))],
        [(0, (LISTENING,)), (.1, (TILT,)), (2.0, ()), (2.2, (TILT,))]),
    "a shortened rewrite counts new words from its own length": (
        [(0, Recording(active=True, turn_running=False)),
         (.1, heard("one two three four five six", 1)),
         (2.0, heard("one two", 2)),
         (2.2, heard("one two three four five", 3))],
        [(0, (LISTENING,)), (.1, (TILT,)), (2.0, ()), (2.2, (TILT,))]),
    "a repeated identical utterance still nods": (
        [(0, Recording(active=True, turn_running=False)),
         (.1, heard("well,", 1, utterance=1)),
         (4.0, heard("well,", 1, utterance=2))],
        [(0, (LISTENING,)), (.1, (NOD,)), (4.0, (NOD,))]),
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
    "every thinking cue keeps the lean Muse's work chose until the next turn": (
        [(0, TurnStarted(output=False, user_speaking=False)),
         (.5, MuseStatus("working", "Searching the web now.", output=False)),
         (1, UserSpeech(speaking=True, output=False, turn_running=True, turns_waiting=False)),
         (2, UserSpeech(speaking=False, output=False, turn_running=True, turns_waiting=False)),
         (3, Working(output=False, user_speaking=False)),
         (4, OutputIdle(user_speaking=False, turn_open=True, wake_open=False)),
         (5, TurnStarted(output=False, user_speaking=False))],
        [(0, (THINKING,)), (.5, (CURIOUS,)), (1, (LISTENING,)), (2, (CURIOUS,)), (3, (CURIOUS,)),
         (4, (CURIOUS,)), (5, (THINKING,))]),
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
        [(0, (THINKING,)), (2, (NOD,)), (3, ()), (4, (THINKING,)), (6, (NOD,))]),
    "an answer during speech is not nodded at later": (
        [(0, TurnStarted(output=False, user_speaking=False)),
         (1, AnswerArrived(output=True)),
         (2, AnswerArrived(output=False))],
        [(0, (THINKING,)), (1, ()), (2, ())]),
    "a gesture still playing suppresses the nod": (
        [(0, Recording(active=True, turn_running=False)),
         (.1, heard("one two three", 1, output=False)),
         (.2, TurnStarted(output=False, user_speaking=False)),
         (.9, AnswerArrived(output=False))],
        [(0, (LISTENING,)), (.1, (TILT,)), (.2, (THINKING,)), (.9, ())]),
    "a finished gesture does not": (
        [(0, Recording(active=True, turn_running=False)),
         (.1, heard("one two three", 1, output=False)),
         (.2, TurnStarted(output=False, user_speaking=False)),
         (1.8, AnswerArrived(output=False))],
        [(0, (LISTENING,)), (.1, (TILT,)), (.2, (THINKING,)), (1.8, (NOD,))]),
    "the got-it nod leaves the thinking lean in place": (
        [(0, TurnStarted(output=False, user_speaking=False)),
         (.5, MuseStatus("working", "Searching the web now.", output=False)),
         (2, AnswerArrived(output=False)),
         (3, Working(output=False, user_speaking=False))],
        [(0, (THINKING,)), (.5, (CURIOUS,)), (2, (NOD,)), (3, (CURIOUS,))]),
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
