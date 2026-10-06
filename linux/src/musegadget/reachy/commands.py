# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The commands Muse can invoke on Reachy, and the checking of their parameters."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

POSE_LIMITS = {
    "x": (-15.0, 15.0), "y": (-15.0, 15.0), "z": (-10.0, 20.0),
    "roll": (-20.0, 20.0), "pitch": (-20.0, 20.0), "yaw": (-35.0, 35.0),
    "body_yaw": (-35.0, 35.0),
    "right_antenna": (-90.0, 90.0), "left_antenna": (-90.0, 90.0),
}

COMMAND_SPECS = {
    "reachy.expression": {
        "description": (
            "Express an emotion with Reachy Mini. Built-in names: neutral, happy, "
            "sad, surprised, curious, nod, shake, listening, thinking, error. "
            "Any other name must exactly match a recorded emotion on the robot. "
            "No sound effect is played, so this can accompany Muse speech."
        ),
        "required": {"name": {"type": "string", "description": "Expression name."}},
        "optional": {
            "intensity": {"type": "number", "description": "Motion strength, 0.1 to 1. Default 0.6."},
            "duration": {"type": "number", "description": "Built-in expression duration in seconds, 0.2 to 5. Default 1.5. Recorded moves keep their timing."},
        },
        "timeout_ms": 20000,
    },
    "reachy.move": {
        "description": (
            "Move Reachy Mini smoothly. Head translation uses millimeters; "
            "rotation and antenna positions use degrees. Commands are bounded "
            "and serialized with expressions. Omitted coordinates keep their "
            "current position."
        ),
        "required": {},
        "optional": {
            **{key: {"type": "number", "description": f"{key}: {low:g} to {high:g} {'mm' if key in ('x', 'y', 'z') else 'degrees'}."}
               for key, (low, high) in POSE_LIMITS.items()},
            "duration": {"type": "number", "description": "Travel time in seconds, 0.2 to 5. Default 0.8."},
            "hold": {"type": "number", "description": "Hold the resulting pose for 0 to 5 seconds. Default 0.5."},
        },
        "timeout_ms": 15000,
    },
}


@dataclass(frozen=True)
class Command:
    """One invocation that passed its spec, with defaults filled in."""

    name: str
    params: dict
    timeout_s: float


def _number(params: dict, key: str, default: float, low: float, high: float) -> float:
    value = params.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{key} must be a number")
    value = float(value)
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{key} must be between {low:g} and {high:g}")
    return value


def parse_command(name: str, params: object, *, antenna_mode: str = "both",
                  timeout_ms: Optional[int] = None) -> Command:
    """Check one invocation from Muse; raises ValueError naming the first problem.

    ``antenna_mode`` is the session's antenna setting: an antenna it leaves
    inactive cannot be moved.
    """
    if name not in COMMAND_SPECS:
        raise ValueError(f"unsupported command: {name}")
    if not isinstance(params, dict):
        raise ValueError("command parameters must be an object")
    spec = COMMAND_SPECS[name]
    unknown = set(params) - set(spec["required"]) - set(spec["optional"])
    if unknown:
        raise ValueError(f"unknown parameter: {sorted(unknown)[0]}")
    checked = dict(params)
    if name == "reachy.expression":
        expression = params.get("name")
        if not isinstance(expression, str) or not expression or len(expression) > 80:
            raise ValueError("name must be an expression name")
        checked["intensity"] = _number(params, "intensity", 0.6, 0.1, 1)
        checked["duration"] = _number(params, "duration", 1.5, 0.2, 5)
    else:
        if not set(params).intersection(POSE_LIMITS):
            raise ValueError("provide at least one head, antenna, or body coordinate")
        for key, side in (("right_antenna", "right"), ("left_antenna", "left")):
            if antenna_mode not in ("both", side) and key in params:
                raise ValueError(f"{key} is inactive for antenna_mode={antenna_mode}")
        for key, (low, high) in POSE_LIMITS.items():
            if key in params:
                checked[key] = _number(params, key, 0, low, high)
        checked["duration"] = _number(params, "duration", 0.8, 0.2, 5)
        checked["hold"] = _number(params, "hold", 0.5, 0, 5)
    timeout_s = _number({"timeout_ms": timeout_ms if timeout_ms is not None else spec["timeout_ms"]},
                        "timeout_ms", spec["timeout_ms"], 1, 30000) / 1000
    return Command(name, checked, timeout_s)
