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

"""Reachy Mini 1.9 speaker correction from the official 1.10 audio path."""

from __future__ import annotations

from dataclasses import dataclass
import sys
from typing import Any, Optional


SPEAKER_EQ_GAINS = (0.0, -13.21, -5.55, -4.28, -4.32, 5.80, 4.65, 4.90, 3.41, 0.0)
SPEAKER_EQ_RATE = 48_000
EQ_BIN_NAME = "muse_reachy_speaker_eq"
_BACKEND_MODULE = "reachy_mini.media.audio_gstreamer"
_BACKEND_CLASS = "GStreamerAudio"
_SINK_DEVICE = "reachymini_audio_sink"


class SpeakerEqError(RuntimeError):
    """The known Reachy speaker pipeline could not be corrected safely."""


def _factory_name(element: Any) -> str | None:
    factory = element.get_factory()
    return None if factory is None else factory.get_name()


def _elements(gst: Any, pipeline: Any):
    iterator = pipeline.iterate_elements()
    while True:
        result, element = iterator.next()
        if result == gst.IteratorResult.OK:
            yield element
        elif result == gst.IteratorResult.DONE:
            return
        elif result == gst.IteratorResult.RESYNC:
            iterator.resync()
        else:
            raise SpeakerEqError("GStreamer could not inspect the speaker pipeline")


def _speaker_edges(gst: Any, pipeline: Any) -> list[tuple[Any, Any]]:
    edges = []
    for sink in _elements(gst, pipeline):
        if _factory_name(sink) != "alsasink":
            continue
        if sink.get_property("device") != _SINK_DEVICE:
            continue
        sink_pad = sink.get_static_pad("sink")
        peer = None if sink_pad is None else sink_pad.get_peer()
        upstream = None if peer is None else peer.get_parent_element()
        if upstream is not None and _factory_name(upstream) == "audioresample":
            edges.append((upstream, sink))
    return edges


def _make_element(gst: Any, factory: str, name: str) -> Any:
    element = gst.ElementFactory.make(factory, name)
    if element is None:
        raise SpeakerEqError(f"GStreamer element unavailable: {factory}")
    return element


def _build_eq_bin(gst: Any) -> Any:
    eq_bin = gst.Bin.new(EQ_BIN_NAME)
    if eq_bin is None:
        raise SpeakerEqError("GStreamer could not create the speaker EQ bin")
    in_caps = _make_element(gst, "capsfilter", EQ_BIN_NAME + "_float")
    equalizer = _make_element(gst, "equalizer-10bands", EQ_BIN_NAME + "_bands")
    limiter = _make_element(gst, "audiodynamic", EQ_BIN_NAME + "_limiter")
    out_convert = _make_element(gst, "audioconvert", EQ_BIN_NAME + "_convert")
    out_resample = _make_element(gst, "audioresample", EQ_BIN_NAME + "_resample")
    in_caps.set_property(
        "caps", gst.Caps.from_string(f"audio/x-raw,format=F32LE,rate={SPEAKER_EQ_RATE}")
    )
    for index, gain in enumerate(SPEAKER_EQ_GAINS):
        equalizer.set_property(f"band{index}", gain)
    limiter.set_property("threshold", 0.9)
    limiter.set_property("ratio", 0.0)
    for element in (in_caps, equalizer, limiter, out_convert, out_resample):
        if not eq_bin.add(element):
            raise SpeakerEqError("GStreamer could not assemble the speaker EQ bin")
    for first, second in (
        (in_caps, equalizer), (equalizer, limiter), (limiter, out_convert),
        (out_convert, out_resample),
    ):
        if not first.link(second):
            raise SpeakerEqError("GStreamer could not link the speaker EQ elements")
    sink_pad = in_caps.get_static_pad("sink")
    source_pad = out_resample.get_static_pad("src")
    if sink_pad is None or source_pad is None:
        raise SpeakerEqError("GStreamer speaker EQ pads are unavailable")
    if not eq_bin.add_pad(gst.GhostPad.new("sink", sink_pad)):
        raise SpeakerEqError("GStreamer could not expose the speaker EQ input")
    if not eq_bin.add_pad(gst.GhostPad.new("src", source_pad)):
        raise SpeakerEqError("GStreamer could not expose the speaker EQ output")
    return eq_bin


def _pipeline_is_null(gst: Any, pipeline: Any) -> bool:
    _result, current, pending = pipeline.get_state(0)
    return current == gst.State.NULL and pending in (gst.State.NULL, gst.State.VOID_PENDING)


@dataclass(frozen=True)
class SpeakerEqInstallation:
    """A structural installation that can be verified after playback starts."""

    pipeline: Any
    eq_bin: Any
    gst: Any

    def verify_active(self) -> int:
        """Return the negotiated native speaker rate or fail closed."""
        result, current, _pending = self.pipeline.get_state(2 * self.gst.SECOND)
        if result == self.gst.StateChangeReturn.FAILURE or current != self.gst.State.PLAYING:
            raise SpeakerEqError("Reachy speaker pipeline did not enter PLAYING")
        equalizer = self.eq_bin.get_by_name(EQ_BIN_NAME + "_bands")
        pad = None if equalizer is None else equalizer.get_static_pad("sink")
        caps = None if pad is None else pad.get_current_caps()
        if caps is None or caps.get_size() != 1:
            raise SpeakerEqError("Reachy speaker EQ did not negotiate audio caps")
        structure = caps.get_structure(0)
        ok, rate = structure.get_int("rate")
        if structure.get_string("format") != "F32LE" or not ok or rate != SPEAKER_EQ_RATE:
            raise SpeakerEqError("Reachy speaker EQ negotiated an unsafe format or rate")
        eq_state = self.eq_bin.get_state(0)[1]
        if eq_state not in (self.gst.State.PLAYING, self.gst.State.PAUSED):
            raise SpeakerEqError("Reachy speaker EQ is not active")
        return rate


def install_speaker_eq(audio_backend: Any, *, _gst: Any = None) -> Optional[SpeakerEqInstallation]:
    """Insert the official EQ topology into a Reachy SDK 1.9 local backend.

    Other media backends are deliberately ignored. SDK versions that already
    expose the official ``make_speaker_eq`` helper own their correction and are
    also left unchanged.
    """
    backend_type = type(audio_backend)
    if backend_type.__module__ != _BACKEND_MODULE or backend_type.__name__ != _BACKEND_CLASS:
        return None
    backend_module = sys.modules.get(_BACKEND_MODULE)
    if backend_module is not None and hasattr(backend_module, "make_speaker_eq"):
        return None
    pipeline = getattr(audio_backend, "_pipeline", None)
    appsrc = getattr(audio_backend, "_appsrc", None)
    if pipeline is None or appsrc is None:
        raise SpeakerEqError("Reachy GStreamer audio backend has no playback pipeline")
    if _gst is None:
        import gi
        gi.require_version("Gst", "1.0")
        from gi.repository import Gst as _gst
    existing = pipeline.get_by_name(EQ_BIN_NAME)
    if existing is not None:
        return SpeakerEqInstallation(pipeline, existing, _gst)
    if not _pipeline_is_null(_gst, pipeline):
        raise SpeakerEqError("Reachy speaker EQ must be installed before media starts")
    edges = _speaker_edges(_gst, pipeline)
    if len(edges) != 1:
        raise SpeakerEqError(f"Expected one Reachy speaker edge, found {len(edges)}")
    upstream, sink = edges[0]
    eq_bin = _build_eq_bin(_gst)
    if not pipeline.add(eq_bin):
        raise SpeakerEqError("GStreamer could not add the speaker EQ to the pipeline")
    upstream.unlink(sink)
    first_linked = upstream.link(eq_bin)
    second_linked = first_linked and eq_bin.link(sink)
    if not second_linked:
        if first_linked:
            upstream.unlink(eq_bin)
        eq_bin.unlink(sink)
        pipeline.remove(eq_bin)
        if not upstream.link(sink):
            raise SpeakerEqError("Speaker EQ insertion and pipeline rollback both failed")
        raise SpeakerEqError("GStreamer could not insert the speaker EQ; original link restored")
    return SpeakerEqInstallation(pipeline, eq_bin, _gst)
