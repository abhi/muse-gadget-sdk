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

"""Read-only qualification of Reachy's local hardware echo cancellation."""

from __future__ import annotations

import importlib
import logging
import math


log = logging.getLogger(__name__)
_BACKEND_MODULE = "reachy_mini.media.audio_gstreamer"
_BACKEND_CLASS = "GStreamerAudio"
_CONTROL_MODULE = "reachy_mini.media.audio_control_utils"
_READ_ERRORS = (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError)


def _factory_name(element):
    factory = element.get_factory()
    return None if factory is None else factory.get_name()


def _elements(gst, pipeline):
    iterator = pipeline.iterate_recurse()
    for _ in range(256):
        result, element = iterator.next()
        if result == gst.IteratorResult.OK:
            yield element
        elif result == gst.IteratorResult.DONE:
            return
        else:
            raise RuntimeError("Audio topology changed during inspection")
    raise RuntimeError("Audio topology exceeds the inspection bound")


def _capture_source(appsink):
    element = appsink
    for _ in range(32):
        if _factory_name(element) == "alsasrc":
            return element
        pad = element.get_static_pad("sink")
        peer = None if pad is None else pad.get_peer()
        if peer is None:
            return None
        element = peer.get_parent_element()
        if element is None:
            return None
    return None


def _dsp_enabled():
    controls = importlib.import_module(_CONTROL_MODULE)
    board = controls.init_respeaker_usb()
    if board is None:
        return False
    try:
        if (board.dev.idVendor, board.dev.idProduct) != (0x38FB, 0x1001):
            return False
        board.TIMEOUT = 1000
        expected = {
            "SHF_BYPASS": (0,),
            "PP_ECHOONOFF": (1,),
            "AEC_NUM_MICS": (4,),
            "AUDIO_MGR_OP_L": (8, 0),
            "AUDIO_MGR_OP_R": (8, 0),
        }
        for name, values in expected.items():
            actual = board.read_values(name)
            if actual is None or len(actual) != len(values):
                return False
            if any(isinstance(value, bool) or not isinstance(value, (int, float))
                   for value in actual):
                return False
            if tuple(actual) != values:
                return False
        farends = board.read_values("AEC_NUM_FARENDS")
        reference_gain = board.read_values("AUDIO_MGR_REF_GAIN")
        if farends is None or len(farends) != 1 or farends[0] not in (1, 2):
            return False
        for value in (farends, reference_gain):
            if value is None or len(value) != 1 or isinstance(value[0], bool):
                return False
            if not isinstance(value[0], (int, float)) or not math.isfinite(value[0]) or value[0] <= 0:
                return False
        return True
    finally:
        board.close()


def verified_echo_cancelled_input(audio_backend) -> bool:
    """Verify known audio routing and enabled XVF3800 echo processing.

    This inspects the running local pipeline and reads USB controls. It neither
    captures audio nor changes DSP settings. True qualifies the hardware AEC
    path; it does not measure residual echo or double-talk recognition quality.
    Unknown backends and unavailable diagnostics retain playback-time gating.
    """
    backend_type = type(audio_backend)
    if backend_type.__module__ != _BACKEND_MODULE or backend_type.__name__ != _BACKEND_CLASS:
        return False
    read_errors = _READ_ERRORS
    try:
        backend_module = importlib.import_module(_BACKEND_MODULE)
        read_errors += (backend_module.GLib.Error,)
        gst = backend_module.Gst
        pipeline = audio_backend._pipeline
        result, current, _pending = pipeline.get_state(gst.SECOND)
        if result == gst.StateChangeReturn.FAILURE or current != gst.State.PLAYING:
            return False
        source = _capture_source(audio_backend._appsink_audio)
        if source is None or source.get_property("device") != "reachymini_audio_src":
            return False
        sinks = [element for element in _elements(gst, pipeline)
                 if _factory_name(element) == "alsasink"]
        if len(sinks) != 1 or sinks[0].get_property("device") != "reachymini_audio_sink":
            return False
        pad = sinks[0].get_static_pad("sink")
        if pad is None or pad.get_peer() is None:
            return False
        return _dsp_enabled()
    except read_errors as exc:
        log.info("Reachy microphone AEC qualification unavailable (%s)", type(exc).__name__)
        return False
