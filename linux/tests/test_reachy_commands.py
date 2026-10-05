# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

import pytest

from musegadget.reachy.commands import Command, parse_command


def test_a_move_gets_its_default_timing_and_timeout():
    assert parse_command("reachy.move", {"yaw": -10, "left_antenna": 30}) == Command(
        "reachy.move", {"yaw": -10.0, "left_antenna": 30.0, "duration": 0.8, "hold": 0.5}, 15.0)


def test_an_expression_keeps_its_name_and_takes_an_explicit_timeout():
    assert parse_command("reachy.expression", {"name": "happy1", "intensity": 1},
                         timeout_ms=50) == Command(
        "reachy.expression", {"name": "happy1", "intensity": 1.0, "duration": 1.5}, 0.05)


@pytest.mark.parametrize("name, params, antenna_mode, error", [
    ("reachy.dance", {}, "both", "unsupported command: reachy.dance"),
    ("reachy.move", [], "both", "command parameters must be an object"),
    ("reachy.move", {"yaw": 1, "speed": 2, "angle": 3}, "both", "unknown parameter: angle"),
    ("reachy.move", {"duration": 1}, "both", "provide at least one head, antenna, or body coordinate"),
    ("reachy.move", {"right_antenna": 10, "x": 99}, "left",
     "right_antenna is inactive for antenna_mode=left"),
    ("reachy.move", {"left_antenna": 10}, "none", "left_antenna is inactive for antenna_mode=none"),
    ("reachy.move", {"x": 16}, "both", "x must be between -15 and 15"),
    ("reachy.move", {"x": True}, "both", "x must be a number"),
    ("reachy.move", {"z": float("nan")}, "both", "z must be between -10 and 20"),
    ("reachy.expression", {"name": ""}, "both", "name must be an expression name"),
    ("reachy.expression", {"name": "nod", "duration": 9}, "both", "duration must be between 0.2 and 5"),
])
def test_an_invalid_invocation_names_its_first_problem(name, params, antenna_mode, error):
    with pytest.raises(ValueError) as problem:
        parse_command(name, params, antenna_mode=antenna_mode)
    assert str(problem.value) == error


def test_a_timeout_beyond_thirty_seconds_is_refused():
    with pytest.raises(ValueError, match="^timeout_ms must be between 1 and 30000$"):
        parse_command("reachy.move", {"x": 1}, timeout_ms=60000)
