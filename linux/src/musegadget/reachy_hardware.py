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

"""Reachy Mini audio and motion, with one thread owning the target poses.

The public methods are synchronous. An asyncio caller should offload start,
close, read_audio, clear_audio, and run_command. The SDK's speech wobbler adds
offsets on the daemon; it does not replace this driver's primary poses.
"""

from __future__ import annotations

import logging
import math
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    import numpy as np

log = logging.getLogger(__name__)
EMOTIONS_DATASET = "pollen-robotics/reachy-mini-emotions-library"
STATES = frozenset({"idle", "listening", "thinking", "speaking", "error"})
BUILTIN_EXPRESSIONS = frozenset({
    "neutral", "happy", "sad", "surprised", "curious", "nod", "shake",
    "listening", "thinking", "error",
})
EXPRESSION_POSES = {
    "neutral": ({}, [-10, 10]), "happy": ({"z": 7, "roll": -8}, [-35, 35]),
    "sad": ({"z": -5, "pitch": 10}, [-55, 55]),
    "surprised": ({"z": 10, "pitch": -7}, [-5, 5]),
    "curious": ({"roll": 12, "pitch": -4}, [-20, 5]),
    "listening": ({"pitch": -3, "z": 3}, [-12, 12]),
    "thinking": ({"roll": 12, "pitch": -5, "z": 5}, [-45, 45]),
    "error": ({"pitch": 8, "z": -3}, [-35, 35]),
}
POSE_LIMITS = {
    "x": (-15.0, 15.0), "y": (-15.0, 15.0), "z": (-10.0, 20.0),
    "roll": (-20.0, 20.0), "pitch": (-20.0, 20.0), "yaw": (-35.0, 35.0),
    "body_yaw": (-35.0, 35.0),
    "right_antenna": (-90.0, 90.0), "left_antenna": (-90.0, 90.0),
}
_MOTION_PERIOD_S = 0.025
_SPEECH_GESTURE_S = 1.6
_SPEECH_GESTURE_KEYFRAMES = ((0., 0.), (.25, 1.), (.45, 1.), (.95, -1.),
                           (1.15, -1.), (_SPEECH_GESTURE_S, 0.))
_HEAD_GESTURES = ("nod", "shake")


def _ease(progress: float) -> float:
    progress = max(0.0, min(1.0, progress))
    return progress ** 3 * (10 - 15 * progress + 6 * progress ** 2)


def _blend_factor(reference: float, elapsed_s: float) -> float:
    """Keep the same smoothing rate when target writes take variable time."""
    elapsed_s = max(0.0, min(0.1, elapsed_s))
    return 1 - (1 - reference) ** (elapsed_s / _MOTION_PERIOD_S)

COMMAND_SPECS = {
    "reachy.expression": {
        "description": (
            "Express an emotion with Reachy Mini. Built-in names: neutral, happy, "
            "sad, surprised, curious, nod, shake, listening, thinking, error. "
            "An exact recorded emotion name uses Pollen's official library. "
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


class ReachyHardwareError(RuntimeError):
    """The robot connection or media backend needs a fresh controller."""


@dataclass
class _Command:
    name: str
    params: dict
    deadline: float
    done: threading.Event = field(default_factory=threading.Event)
    cancelled: threading.Event = field(default_factory=threading.Event)
    result: dict | None = None


def _number(params: dict, key: str, default: float, low: float, high: float) -> float:
    value = params.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{key} must be a number")
    value = float(value)
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{key} must be between {low:g} and {high:g}")
    return value


class ReachyController:
    """Own the robot for a standalone Muse session.

    Inject factories only for tests. No SDK, numpy, or move datasets are loaded
    until start; the emotion library is loaded only for an explicit named move.
    Set sleep_on_close=False when a Reachy app host owns the sleep lifecycle.
    """

    def __init__(
        self, host: str = "reachy-mini.local", connection_mode: str = "auto",
        media_backend: str = "default", *, port: int = 8000,
        enable_motion: bool = True, sleep_on_close: bool = True,
        antenna_mode: str = "both",
        face_follow_model: Path | None = None,
        face_follower_factory: Callable | None = None,
        robot_factory: Callable | None = None,
        pose_factory: Callable | None = None, pose_interpolator: Callable | None = None,
        moves_factory: Callable | None = None,
    ) -> None:
        if antenna_mode not in ("both", "left", "right", "none"):
            raise ValueError("antenna_mode must be both, left, right, or none")
        self.antenna_mode = antenna_mode
        self._inactive_antennas = tuple(
            index for index, side in enumerate(("right", "left"))
            if antenna_mode not in ("both", side)
        )
        self._frozen_antennas = None
        self._robot_factory = robot_factory
        self._pose_factory = pose_factory
        self._interpolate_pose = pose_interpolator
        self._moves_factory = moves_factory
        self._face_follow_model = face_follow_model
        self._face_follower_factory = face_follower_factory
        self._face_follower = None
        self._face_failure_logged = False
        self._head_angles = (0.0, 0.0)
        self._connect_options = dict(host=host, port=port, connection_mode=connection_mode,
                                     media_backend=media_backend)
        self._sleep_on_close = sleep_on_close
        self._enable_motion = enable_motion
        self._initial_motor_mode = None
        self._robot = None
        self._speaker_eq = None
        self._np = None
        self._sample_rate = 0
        self._output_sample_rate = 0
        self.echo_cancelled_input = False
        self._motion_thread = None
        self._state_lock = threading.Lock()
        self._audio_lock = threading.Lock()
        self._state = "idle"
        self._state_started = time.monotonic()
        self._progress_started = None
        self._reply_expression = None
        self._stop = threading.Event()
        self._started = threading.Event()
        self._failure = None
        self._shutdown_failure = None
        self._commands: queue.Queue[_Command] = queue.Queue(maxsize=8)
        self._moves = None
        self._moves_load = None
        self._last_pose = None
        self._last_antennas = None
        self._last_body_yaw = 0.0
        self._gesture_epoch = None
        self._gesture_anchor = None
        self._gesture_was_active = False
        self.wobbling_available = False

    @property
    def motion_enabled(self) -> bool:
        return self._enable_motion

    @property
    def face_tracking_enabled(self) -> bool:
        return (self._face_follower is not None
                and getattr(self._face_follower, "failure", None) is None)

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def output_sample_rate(self) -> int:
        return self._output_sample_rate

    def start(self) -> None:
        if self._robot is not None:
            self._require_running()
            return
        import numpy as np
        self._np = np
        if self._robot_factory is None:
            from reachy_mini import ReachyMini
            self._robot_factory = ReachyMini
        if self._pose_factory is None:
            from reachy_mini.utils import create_head_pose
            self._pose_factory = create_head_pose
        if self._interpolate_pose is None:
            from reachy_mini.utils.interpolation import linear_pose_interpolation
            self._interpolate_pose = linear_pose_interpolation
        self._stop.clear()
        self._started.clear()
        self._failure = None
        self._shutdown_failure = None
        self._initial_motor_mode = None
        self._last_pose = None
        self._last_antennas = None
        self._frozen_antennas = None
        self._gesture_epoch = self._gesture_anchor = None
        self._gesture_was_active = False
        try:
            self._robot = self._robot_factory(**self._connect_options)
            from musegadget.reachy_speaker_eq import install_speaker_eq
            self._speaker_eq = install_speaker_eq(self._robot.media.audio)
            self._sample_rate = self._robot.media.get_input_audio_samplerate()
            self._output_sample_rate = self._robot.media.get_output_audio_samplerate()
            if min(self._sample_rate, self._output_sample_rate) <= 0:
                raise RuntimeError("Reachy Mini audio is unavailable; check the daemon media backend")
            self._robot.media.start_recording()
            # SDK 1.9's GStreamer queue defaults to 200 FIFO buffers. A live
            # conversation needs recent audio even while synthesis uses CPU.
            capture_sink = getattr(self._robot.media.audio, "_appsink_audio", None)
            if capture_sink is not None:
                capture_sink.set_property("max-buffers", 2)
                capture_sink.set_property("drop", True)
            self._robot.media.start_playing()
            from musegadget.reachy_duplex_audio import verified_echo_cancelled_input
            self.echo_cancelled_input = verified_echo_cancelled_input(self._robot.media.audio)
            log.info("Reachy microphone echo cancellation: %s",
                     "verified hardware duplex" if self.echo_cancelled_input else "playback gating")
            if self._speaker_eq is not None:
                speaker_rate = self._speaker_eq.verify_active()
                log.info("Reachy speaker equalization enabled at %d Hz", speaker_rate)
            if not self._enable_motion:
                return
            self._motion_thread = threading.Thread(target=self._motion_loop,
                                                   name="muse-reachy-motion", daemon=True)
            self._motion_thread.start()
            if not self._started.wait(10):
                raise RuntimeError("Reachy Mini motion startup timed out")
            self._require_running()
            self._start_face_follow()
        except BaseException:
            self.close()
            raise

    def _require_running(self) -> None:
        if self._failure is not None:
            raise ReachyHardwareError(f"Reachy Mini motion failed: {self._failure}") from self._failure
        if self._robot is None or self._stop.is_set():
            raise ReachyHardwareError("Reachy Mini is not running")

    def read_audio(self) -> np.ndarray | None:
        """Read mono float32 at sample_rate; SDK 1.9 waits up to 20 ms."""
        self._require_running()
        try:
            sample = self._robot.media.get_audio_sample()
            if sample is None:
                return None
            audio = self._np.asarray(sample, dtype=self._np.float32)
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            if audio.ndim != 1 or not self._np.isfinite(audio).all():
                raise ValueError("Reachy Mini returned invalid microphone audio")
            return self._np.ascontiguousarray(audio, dtype=self._np.float32)
        except Exception as exc:
            raise ReachyHardwareError(f"Reachy Mini microphone failed: {exc}") from exc

    def play_audio(self, samples: np.ndarray) -> None:
        """Queue mono float32 samples at output_sample_rate without waiting."""
        self._require_running()
        audio = self._np.asarray(samples, dtype=self._np.float32)
        if audio.ndim != 1 or not self._np.isfinite(audio).all():
            raise ValueError("Speaker audio must be a finite mono array")
        if audio.size:
            audio = self._np.ascontiguousarray(self._np.clip(audio, -1, 1))
            with self._audio_lock:
                try:
                    self._robot.media.push_audio_sample(audio)
                except Exception as exc:
                    raise ReachyHardwareError(f"Reachy Mini speaker failed: {exc}") from exc

    def clear_audio(self) -> None:
        """Flush queued speech without stopping the shared microphone pipeline."""
        self._require_running()
        with self._audio_lock:
            try:
                audio_backend = self._robot.media.audio
                clear_player = getattr(audio_backend, "clear_player", None)
                if clear_player is None:
                    raise RuntimeError("Reachy Mini audio backend cannot flush playback; use SDK 1.9 or newer")
                clear_player()
            except Exception as exc:
                raise ReachyHardwareError(f"Reachy Mini playback flush failed: {exc}") from exc

    def set_state(self, state: str, *, expression: str | None = None) -> None:
        if state not in STATES:
            raise ValueError(f"Unknown conversation state: {state}")
        if expression is not None and expression not in BUILTIN_EXPRESSIONS:
            raise ValueError(f"Unknown reply expression: {expression}")
        with self._state_lock:
            if (state != self._state or expression != self._reply_expression or state == "speaking"
                    or expression in _HEAD_GESTURES or state == expression == "listening"):
                self._state_started = time.monotonic()
            self._state = state
            self._reply_expression = expression
            if state != "thinking":
                self._progress_started = None

    def cue_progress(self) -> None:
        """Accent a spoken progress update without changing the thinking rhythm."""
        with self._state_lock:
            if self._state == "thinking":
                self._progress_started = time.monotonic()

    def run_command(self, name: str, params: dict, timeout_ms: int | None = None) -> dict:
        """Wait for a bounded motion command; all motor targets stay on one thread."""
        try:
            self._require_running()
            if not self._enable_motion:
                raise RuntimeError("Reachy Mini motion is disabled for this session")
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
                for index in self._inactive_antennas:
                    key = ("right_antenna", "left_antenna")[index]
                    if key in params:
                        raise ValueError(f"{key} is inactive for antenna_mode={self.antenna_mode}")
                for key, (low, high) in POSE_LIMITS.items():
                    if key in params:
                        checked[key] = _number(params, key, 0, low, high)
                checked["duration"] = _number(params, "duration", 0.8, 0.2, 5)
                checked["hold"] = _number(params, "hold", 0.5, 0, 5)
            timeout = _number({"timeout_ms": timeout_ms if timeout_ms is not None else spec["timeout_ms"]},
                              "timeout_ms", spec["timeout_ms"], 1, 30000) / 1000
            request = _Command(name, checked, time.monotonic() + timeout)
            self._commands.put_nowait(request)
            if not request.done.wait(timeout):
                request.cancelled.set()
                return {"ok": False, "error": "Reachy Mini command timed out"}
            return request.result
        except (ValueError, RuntimeError, queue.Full) as exc:
            return {"ok": False, "error": str(exc) or "Reachy Mini motion queue is full"}

    def close(self) -> None:
        self._stop.set()
        if self._face_follower is not None:
            self._face_follower.set_active(False)
        if self._motion_thread is not None:
            self._motion_thread.join(timeout=10)
            if self._motion_thread.is_alive():
                raise RuntimeError("Reachy Mini motion worker did not stop; media remains open")
            self._motion_thread = None
        if self._face_follower is not None:
            self._face_follower.close()
            self._face_follower = None
        if self._robot is not None:
            robot, self._robot = self._robot, None
            with self._audio_lock:
                robot.__exit__(None, None, None)
        self._sample_rate = 0
        self._output_sample_rate = 0
        self.echo_cancelled_input = False
        if self._shutdown_failure is not None:
            failure, self._shutdown_failure = self._shutdown_failure, None
            raise ReachyHardwareError("Reachy Mini could not stop speech motion during shutdown") from failure

    def _head(self, **kwargs):
        return self._pose_factory(degrees=True, mm=True, **kwargs)

    def _start_face_follow(self) -> None:
        if self._face_follow_model is None or not self._enable_motion:
            return
        factory = self._face_follower_factory
        if factory is None:
            from musegadget.local_face_tracking import FaceFollower
            factory = FaceFollower
        self._face_follower = factory(
            self._robot.media.get_frame, self._face_follow_model,
            read_head_angles=self._read_head_angles,
            camera_matrix=self._robot.media.camera.K,
            distortion=self._robot.media.camera.D,
        )
        self._face_follower.start()
        self._face_failure_logged = False
        log.info("Local face tracking ready; camera frames remain on Reachy")

    def _read_head_angles(self) -> tuple[float, float]:
        try:
            pose = self._robot.get_current_head_pose()
        except AssertionError:
            # The SDK caches daemon poses; before its first frame use our
            # primary writer's last command rather than waiting on the network.
            return self._head_angles
        coordinates = self._pose_parameters(pose)
        return (math.radians(coordinates["yaw"]), math.radians(coordinates["pitch"]))

    def _update_face_tracking(self, state: str) -> None:
        follower = self._face_follower
        if follower is None:
            return
        failure = getattr(follower, "failure", None)
        if failure is not None and not self._face_failure_logged:
            log.warning("Local face tracking stopped: %s; voice remains available", type(failure).__name__)
            self._face_failure_logged = True
        follower.set_active(failure is None and state in ("listening", "thinking", "speaking"))

    def _face_pose(self, pose):
        follower = self._face_follower
        target = follower.target() if follower is not None else None
        if target is None:
            return pose
        from musegadget.local_face_tracking import PITCH_LIMIT, YAW_LIMIT
        coordinates = self._pose_parameters(pose)
        coordinates["yaw"] = math.degrees(target.yaw)
        coordinates["pitch"] = math.degrees(target.pitch)
        for name, limit in (("yaw", YAW_LIMIT), ("pitch", PITCH_LIMIT)):
            low, high = POSE_LIMITS[name]
            low, high = max(low, -math.degrees(limit)), min(high, math.degrees(limit))
            coordinates[name] = max(low, min(high, coordinates[name]))
        return self._head(**coordinates)

    def _write(self, pose, antennas, body_yaw: float) -> None:
        antennas = self._np.asarray(antennas, dtype=float).copy()
        for index in self._inactive_antennas:
            antennas[index] = self._frozen_antennas[index]
        self._robot.set_target(head=pose, antennas=antennas, body_yaw=float(body_yaw))
        self._last_pose = pose.copy()
        coordinates = self._pose_parameters(pose)
        self._head_angles = (math.radians(coordinates["yaw"]),
                             math.radians(coordinates["pitch"]))
        self._last_antennas = self._np.asarray(antennas, dtype=float).copy()
        self._last_body_yaw = float(body_yaw)

    def _motion_loop(self) -> None:
        wobbling_active = False
        try:
            if hasattr(self._robot, "client"):
                status = self._robot.client.get_status()
                backend = getattr(status, "backend_status", None)
                mode = getattr(backend, "motor_control_mode", None)
                mode = getattr(mode, "value", mode)
                if mode not in ("enabled", "disabled", "gravity_compensation"):
                    raise ReachyHardwareError("Could not read Reachy Mini's initial motor control mode")
                self._initial_motor_mode = mode
            else:
                self._initial_motor_mode = "disabled"
            self._robot.enable_motors()
            self._last_pose = self._robot.get_current_head_pose().copy()
            joints, antennas = self._robot.get_current_joint_positions()
            self._last_antennas = self._np.asarray(antennas, dtype=float).copy()
            if self._last_antennas.shape != (2,) or not self._np.isfinite(self._last_antennas).all():
                raise ReachyHardwareError("Reachy Mini returned invalid antenna positions")
            self._frozen_antennas = self._last_antennas.copy()
            self._last_body_yaw = float(joints[0])
            self.wobbling_available = all(
                callable(getattr(self._robot, name, None))
                for name in ("enable_wobbling", "disable_wobbling")
            )
            if self.wobbling_available:
                try:
                    self._robot.disable_wobbling()
                except Exception as exc:
                    log.warning("Reachy speech motion is unavailable: %s", exc)
                    self.wobbling_available = False
            wake = _Command("wake", {}, time.monotonic() + 5)
            self._travel(wake, self._head(), self._np.deg2rad([-10, 10]), 0, 1.5)
            self._started.set()
            previous_update = time.monotonic() - _MOTION_PERIOD_S
            while not self._stop.is_set():
                with self._state_lock:
                    state = self._state
                    expression = self._reply_expression
                    now = time.monotonic()
                    state_started = self._state_started
                    elapsed = now - state_started
                    progress_age = None if self._progress_started is None else now - self._progress_started
                gesture = state in ("speaking", "listening", "thinking") and expression in _HEAD_GESTURES
                gesture_active = gesture and elapsed < _SPEECH_GESTURE_S
                wants_wobbling = state == "speaking" and not gesture_active
                if self.wobbling_available and wants_wobbling != wobbling_active:
                    try:
                        if wants_wobbling:
                            self._robot.enable_wobbling()
                        else:
                            self._robot.disable_wobbling()
                        wobbling_active = wants_wobbling
                    except Exception as exc:
                        if not wants_wobbling:
                            raise ReachyHardwareError("Reachy Mini could not stop speech motion") from exc
                        self._robot.disable_wobbling()
                        log.warning("Reachy speech motion is unavailable: %s", exc)
                        self.wobbling_available = False
                try:
                    request = self._commands.get_nowait()
                except queue.Empty:
                    request = None
                if request is not None:
                    if self._face_follower is not None:
                        self._face_follower.set_active(False)
                    try:
                        self._check_command(request)
                        payload = self._execute_motion(request)
                        request.result = {"ok": True, "payload": payload}
                    except Exception as exc:
                        request.result = {"ok": False, "error": str(exc)}
                    finally:
                        request.done.set()
                        self._gesture_epoch = self._gesture_anchor = None
                        self._gesture_was_active = False
                        previous_update = time.monotonic()
                        self._stop.wait(_MOTION_PERIOD_S)
                else:
                    if gesture and self._gesture_epoch != state_started:
                        if self._gesture_anchor is None or not self._gesture_was_active:
                            coordinates = self._pose_parameters(self._last_pose)
                            self._gesture_anchor = (coordinates["yaw"], coordinates["pitch"])
                        self._gesture_epoch = state_started
                    elif not gesture:
                        self._gesture_epoch = self._gesture_anchor = None
                    self._gesture_was_active = gesture_active
                    self._update_face_tracking("idle" if gesture_active else state)
                    pose, antennas = self._state_pose(
                        state, elapsed, expression, progress_age=progress_age,
                        gesture_anchor=self._gesture_anchor)
                    if not gesture_active:
                        pose = self._face_pose(pose)
                    quick_cue = ((state == "listening" and expression == "happy" and elapsed < 1.0)
                                 or (gesture_active and state != "speaking"))
                    if quick_cue:
                        head_blend, antenna_blend = 0.18, 0.3
                    elif state == "thinking":
                        head_blend, antenna_blend = 0.12, 0.3
                    elif state == "speaking":
                        head_blend, antenna_blend = 0.16, 0.22
                    else:
                        head_blend, antenna_blend = 0.08, 0.08
                    dt = now - previous_update
                    previous_update = now
                    head_blend = _blend_factor(head_blend, dt)
                    antenna_blend = _blend_factor(antenna_blend, dt)
                    self._write(self._interpolate_pose(self._last_pose, pose, head_blend),
                                self._last_antennas * (1 - antenna_blend) + antennas * antenna_blend,
                                self._last_body_yaw * (1 - _blend_factor(0.08, dt)))
                    self._stop.wait(_MOTION_PERIOD_S)
        except Exception as exc:
            self._failure = exc
            log.exception("Reachy Mini motion worker failed")
        finally:
            self._started.set()
            while True:
                try:
                    pending = self._commands.get_nowait()
                except queue.Empty:
                    break
                pending.result = {"ok": False, "error": "Reachy Mini stopped"}
                pending.done.set()
            try:
                if wobbling_active:
                    self._robot.disable_wobbling()
            except Exception as exc:
                self._shutdown_failure = exc
                log.exception("Could not stop Reachy Mini speech motion")
            finally:
                self.wobbling_available = False
            try:
                if (self._sleep_on_close and self._last_pose is not None
                        and self._initial_motor_mode == "disabled"):
                    if self.antenna_mode == "both":
                        self._robot.goto_sleep()
                    elif self._frozen_antennas is not None:
                        antennas = self._np.deg2rad([-10, 10])
                        for index in self._inactive_antennas:
                            antennas[index] = self._frozen_antennas[index]
                        self._robot.goto_target(head=self._head(pitch=10, z=-10),
                                                antennas=antennas, body_yaw=0.0, duration=1.0)
                    self._robot.disable_motors()
                elif self._initial_motor_mode == "gravity_compensation":
                    self._robot.enable_gravity_compensation()
            except Exception:
                log.exception("Could not park Reachy Mini")

    def _state_pose(self, state: str, now: float, expression: str | None = None,
                    *, progress_age: float | None = None,
                    gesture_anchor: tuple[float, float] | None = None):
        if expression in _HEAD_GESTURES and (state == "speaking" or state in ("listening", "thinking")
                                             and now < _SPEECH_GESTURE_S):
            yaw, pitch = gesture_anchor or (0.0, 0.0)
            wave = 0.0
            for (start, first), (end, last) in zip(
                    _SPEECH_GESTURE_KEYFRAMES, _SPEECH_GESTURE_KEYFRAMES[1:]):
                if now <= end:
                    wave = first + (last - first) * _ease((now - start) / (end - start))
                    break
            if expression == "nod":
                pitch += 8 * wave
            else:
                yaw += 10 * wave
            yaw = max(-20, min(20, yaw))
            pitch = max(-12, min(12, pitch))
            return self._head(yaw=yaw, pitch=pitch), self._np.deg2rad([-10, 10])
        if state == "listening":
            coordinates, antennas = EXPRESSION_POSES.get(expression, EXPRESSION_POSES["listening"])
            if expression == "listening":
                tilt = math.sin(math.pi * min(1.0, max(0.0, now) / _SPEECH_GESTURE_S)) ** 2
                coordinates = {**coordinates, "roll": 10 * tilt}
            if expression == "happy":
                greeting = _ease(now / .2) * (1 - _ease((now - .8) / .6))
                wiggle = math.sin(math.pi * min(1, max(0, now))) * math.sin(4 * math.pi * now)
                coordinates = {"pitch": -3, "z": 3 + 4 * greeting,
                               "roll": -8 * greeting, "yaw": 7 * wiggle}
                antennas = [-12 - 23 * greeting + 18 * wiggle,
                            12 + 23 * greeting - 18 * wiggle]
            return self._head(**coordinates), self._np.deg2rad(antennas)
        if state == "thinking":
            beat = max(0, now) % 3.6
            if beat < 0.45:
                energy = (1 - math.cos(math.pi * beat / 0.45)) / 2
            elif beat < 1.65:
                energy = 1.0
            elif beat < 2.25:
                energy = (1 + math.cos(math.pi * (beat - 1.65) / 0.6)) / 2
            else:
                energy = 0.0
            amplitude = 30 * energy
            flourish = 0.0
            if progress_age is not None and 0 <= progress_age < 1.4:
                accent = math.sin(math.pi * progress_age / 1.4) ** 2
                amplitude += (45 - amplitude) * accent
                flourish = 5 * accent * math.sin(progress_age * 2 * math.pi * 3.2)
            right = amplitude * math.sin(now * 2 * math.pi * 1.6) + flourish
            left = amplitude * math.sin((now - .14) * 2 * math.pi * 1.6) + flourish
            # A long wait gets a brief glance, not a continuous head loop.
            glance = 0.0
            direction = 1
            if now >= 9:
                glance_age = (now - 9) % 30
                direction = -1 if int((now - 9) / 30) % 2 else 1
                if glance_age < 1.8:
                    glance = math.sin(math.pi * glance_age / 1.8) ** 2
            lean = 10 if expression == "curious" else 0
            pose = self._head(pitch=-3 - lean / 2, z=4 + 2 * glance,
                              yaw=direction * 6 * glance, roll=lean + direction * 6 * glance)
            return pose, self._np.deg2rad([-10 + right, 10 - left])
        if state == "speaking":
            coordinates, antennas = EXPRESSION_POSES.get(expression, EXPRESSION_POSES["neutral"])
            strength = _ease(now / .35)
            strength *= 1 - .35 * _ease((now - .8) / .8)
            coordinates = {key: value * strength for key, value in coordinates.items()}
            neutral = self._np.asarray([-10, 10], dtype=float)
            antennas = neutral + (self._np.asarray(antennas) - neutral) * strength
            return self._head(**coordinates), self._np.deg2rad(antennas)
        if state == "error":
            return self._head(pitch=8, z=-3), self._np.deg2rad([-35, 35])
        z = 1.5 * math.sin(now * 2 * math.pi * 0.12) if state == "idle" else 0
        return self._head(z=z), self._np.deg2rad([-10, 10])

    def _check_command(self, request: _Command) -> None:
        if self._stop.is_set() or request.cancelled.is_set():
            raise RuntimeError("Reachy Mini command cancelled")
        if time.monotonic() >= request.deadline:
            raise RuntimeError("Reachy Mini command timed out")

    def _travel(self, request: _Command, pose, antennas, body_yaw: float, duration: float) -> None:
        start = time.monotonic()
        start_pose = self._last_pose.copy()
        start_antennas = self._last_antennas.copy()
        start_body = self._last_body_yaw
        while True:
            self._check_command(request)
            progress = min(1.0, (time.monotonic() - start) / duration)
            eased = _ease(progress)
            self._write(self._interpolate_pose(start_pose, pose, eased),
                        start_antennas * (1 - eased) + antennas * eased,
                        start_body * (1 - eased) + body_yaw * eased)
            if progress >= 1:
                return
            self._stop.wait(0.025)

    def _hold(self, request: _Command, duration: float) -> None:
        end = time.monotonic() + duration
        while time.monotonic() < end:
            self._check_command(request)
            self._stop.wait(min(0.025, max(0, end - time.monotonic())))

    def _pose_parameters(self, pose) -> dict:
        rotation = pose[:3, :3]
        pitch = math.asin(max(-1, min(1, -float(rotation[2, 0]))))
        return dict(x=float(pose[0, 3]) * 1000, y=float(pose[1, 3]) * 1000,
                    z=float(pose[2, 3]) * 1000,
                    roll=math.degrees(math.atan2(rotation[2, 1], rotation[2, 2])),
                    pitch=math.degrees(pitch),
                    yaw=math.degrees(math.atan2(rotation[1, 0], rotation[0, 0])))

    def _execute_motion(self, request: _Command) -> dict:
        params = request.params
        if request.name == "reachy.move":
            coordinates = self._pose_parameters(self._last_pose)
            coordinates.update({key: params[key] for key in coordinates if key in params})
            for key, value in coordinates.items():
                low, high = POSE_LIMITS[key]
                coordinates[key] = max(low, min(high, value))
            antennas = self._last_antennas.copy()
            for index, key in enumerate(("right_antenna", "left_antenna")):
                if key in params:
                    antennas[index] = math.radians(params[key])
            body_yaw = math.radians(params["body_yaw"]) if "body_yaw" in params else self._last_body_yaw
            self._travel(request, self._head(**coordinates), antennas, body_yaw, params["duration"])
            self._hold(request, params["hold"])
            return {"coordinates": coordinates, "duration": params["duration"]}
        expression = params["name"]
        if expression in BUILTIN_EXPRESSIONS:
            self._builtin_expression(request)
            return {"name": expression, "source": "builtin", "duration": params["duration"]}
        self._load_moves(request)
        self._check_command(request)
        if expression not in self._moves.list_moves():
            raise ValueError(f"Unknown recorded expression {expression!r}; available: {', '.join(self._moves.list_moves())}")
        move = self._moves.get(expression)
        duration = float(move.duration)
        if not math.isfinite(duration) or not 0 < duration <= 15:
            raise ValueError("Recorded expressions must last at most 15 seconds")
        first = self._bounded_recorded(move.evaluate(0), params["intensity"])
        self._travel(request, *first, duration=0.4)
        start = time.monotonic()
        while True:
            self._check_command(request)
            elapsed = min(time.monotonic() - start, max(0, duration - 0.001))
            self._write(*self._bounded_recorded(move.evaluate(elapsed), params["intensity"]))
            if time.monotonic() - start >= duration:
                break
            self._stop.wait(0.025)
        return {"name": expression, "source": "recorded", "dataset": EMOTIONS_DATASET,
                "duration": duration, "intensity": params["intensity"], "bounded": True}

    def _load_moves(self, request: _Command) -> None:
        if self._moves is not None:
            return
        if self._moves_load is None:
            loading = {"done": threading.Event(), "value": None, "error": None}
            self._moves_load = loading

            def load() -> None:
                try:
                    factory = self._moves_factory
                    if factory is None:
                        from reachy_mini.motion.recorded_move import RecordedMoves
                        factory = RecordedMoves
                    loading["value"] = factory(EMOTIONS_DATASET)
                except Exception as exc:
                    loading["error"] = exc
                finally:
                    loading["done"].set()

            threading.Thread(target=load, name="muse-reachy-emotions", daemon=True).start()
        loading = self._moves_load
        while not loading["done"].wait(0.025):
            self._check_command(request)
        self._check_command(request)
        if loading["error"] is not None:
            self._moves_load = None
            raise RuntimeError(f"Official Reachy emotion library is unavailable: {loading['error']}")
        self._moves = loading["value"]

    def _bounded_recorded(self, values, intensity: float):
        head, antennas, body_yaw = values
        if head is None:
            head = self._last_pose
        coordinates = self._pose_parameters(head)
        for key, value in coordinates.items():
            low, high = POSE_LIMITS[key]
            coordinates[key] = max(low, min(high, value * intensity))
        head = self._head(**coordinates)
        antennas = self._last_antennas if antennas is None else self._np.asarray(antennas)
        antennas = self._np.clip(antennas * intensity, -math.pi / 2, math.pi / 2)
        yaw = self._last_body_yaw if body_yaw is None else float(body_yaw)
        yaw = max(math.radians(-35), min(math.radians(35), yaw * intensity))
        return head, antennas, yaw

    def _builtin_expression(self, request: _Command) -> None:
        params = request.params
        intensity = params["intensity"]
        duration = params["duration"]
        if params["name"] in ("nod", "shake"):
            axis = "pitch" if params["name"] == "nod" else "yaw"
            for angle in (-9, 9, -6, 0):
                self._travel(request, self._head(**{axis: angle * intensity}),
                             self._np.deg2rad([-10, 10]), 0, duration / 4)
        else:
            coordinates, antennas = EXPRESSION_POSES[params["name"]]
            target = self._head(**{key: value * intensity for key, value in coordinates.items()})
            self._travel(request, target, self._np.deg2rad(antennas) * intensity, 0, duration / 2)
            self._hold(request, duration / 4)
            self._travel(request, self._head(), self._np.deg2rad([-10, 10]), 0, duration / 4)
