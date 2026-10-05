import pytest

from musegadget.reachy_capabilities import Expression
from musegadget.reachy_expression_plan import (
    CaptureGap, ExpressionPlanner, OutputIdle, Recording, SetState, Started, State, TurnDone,
    UserSpeech, WakeClosed, WakeHeard, Working,
)

IDLE = SetState(State.IDLE)
LISTENING = SetState(State.LISTENING)
THINKING = SetState(State.THINKING)


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
