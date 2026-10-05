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

from types import SimpleNamespace

import pytest

from musegadget import reachy_duplex_audio as duplex


class GstError(Exception):
    pass


class Element:
    def __init__(self, factory, device=None, upstream=None):
        self.factory = factory
        self.device = device
        self.upstream = upstream

    def get_factory(self):
        return SimpleNamespace(get_name=lambda: self.factory)

    def get_property(self, name):
        assert name == "device"
        return self.device

    def get_static_pad(self, name):
        assert name == "sink"
        peer = None if self.upstream is None else SimpleNamespace(get_parent_element=lambda: self.upstream)
        return SimpleNamespace(get_peer=lambda: peer)


class Iterator:
    def __init__(self, elements):
        self.elements = iter(elements)

    def next(self):
        element = next(self.elements, None)
        return ("done", None) if element is None else ("ok", element)


class Pipeline:
    def __init__(self, elements):
        self.elements = elements
        self.state = "playing"
        self.result = "success"

    def get_state(self, timeout):
        assert timeout == 1_000_000_000
        return self.result, self.state, "void"

    def iterate_recurse(self):
        return Iterator(self.elements)


class Board:
    def __init__(self):
        self.dev = SimpleNamespace(idVendor=0x38FB, idProduct=0x1001)
        self.values = {
            "SHF_BYPASS": (0,), "PP_ECHOONOFF": (1,), "AEC_NUM_MICS": (4,),
            "AUDIO_MGR_OP_L": (8, 0), "AUDIO_MGR_OP_R": (8, 0),
            "AEC_NUM_FARENDS": (2,), "AUDIO_MGR_REF_GAIN": (1.0,),
        }
        self.reads = []
        self.closed = False

    def read_values(self, name):
        self.reads.append(name)
        return self.values[name]

    def close(self):
        self.closed = True


@pytest.fixture
def known_audio(monkeypatch):
    gst = SimpleNamespace(
        SECOND=1_000_000_000,
        IteratorResult=SimpleNamespace(OK="ok", DONE="done"),
        State=SimpleNamespace(PLAYING="playing"),
        StateChangeReturn=SimpleNamespace(FAILURE="failure"),
    )
    source = Element("alsasrc", "reachymini_audio_src")
    queue = Element("queue", upstream=source)
    convert = Element("audioconvert", upstream=queue)
    resample = Element("audioresample", upstream=convert)
    appsink = Element("appsink", upstream=resample)
    speaker_resample = Element("audioresample")
    sink = Element("alsasink", "reachymini_audio_sink", upstream=speaker_resample)
    pipeline = Pipeline([source, queue, convert, resample, appsink, speaker_resample, sink])
    backend_class = type("GStreamerAudio", (), {"__module__": duplex._BACKEND_MODULE})
    backend = backend_class()
    backend._pipeline = pipeline
    backend._appsink_audio = appsink
    board = Board()
    modules = {
        duplex._BACKEND_MODULE: SimpleNamespace(Gst=gst, GLib=SimpleNamespace(Error=GstError)),
        duplex._CONTROL_MODULE: SimpleNamespace(init_respeaker_usb=lambda: board),
    }
    monkeypatch.setattr(duplex.importlib, "import_module", modules.__getitem__)
    return SimpleNamespace(backend=backend, board=board, pipeline=pipeline, source=source,
                           sink=sink, appsink=appsink, modules=modules)


def test_known_running_route_and_enabled_dsp_qualify(known_audio):
    assert duplex.verified_echo_cancelled_input(known_audio.backend) is True
    assert known_audio.board.closed is True
    assert set(known_audio.board.reads) == {
        "SHF_BYPASS", "PP_ECHOONOFF", "AEC_NUM_MICS", "AUDIO_MGR_OP_L",
        "AUDIO_MGR_OP_R", "AEC_NUM_FARENDS", "AUDIO_MGR_REF_GAIN",
    }


def test_unknown_backend_does_not_import_sdk(monkeypatch):
    monkeypatch.setattr(duplex.importlib, "import_module", lambda name: pytest.fail("SDK import"))
    assert duplex.verified_echo_cancelled_input(object()) is False


@pytest.mark.parametrize("endpoint", ["source", "sink"])
@pytest.mark.parametrize("device", [None, "default", "hw:0,0", "reachymini_audio_other"])
def test_unknown_audio_device_retains_gating(known_audio, endpoint, device):
    getattr(known_audio, endpoint).device = device
    assert duplex.verified_echo_cancelled_input(known_audio.backend) is False
    assert known_audio.board.reads == []


def test_named_source_elsewhere_does_not_qualify_wrong_capture_path(known_audio):
    known_audio.appsink.upstream = Element("pulsesrc", "default")
    assert duplex.verified_echo_cancelled_input(known_audio.backend) is False


@pytest.mark.parametrize("endpoint", ["appsink", "sink"])
def test_unlinked_audio_is_not_qualified(known_audio, endpoint):
    getattr(known_audio, endpoint).upstream = None
    assert duplex.verified_echo_cancelled_input(known_audio.backend) is False


def test_ambiguous_multiple_speaker_routes_are_not_qualified(known_audio):
    known_audio.pipeline.elements.append(Element("alsasink", "default", upstream=Element("audioresample")))
    assert duplex.verified_echo_cancelled_input(known_audio.backend) is False


@pytest.mark.parametrize("state", ["paused", "null", "ready"])
def test_pipeline_must_be_running(known_audio, state):
    known_audio.pipeline.state = state
    assert duplex.verified_echo_cancelled_input(known_audio.backend) is False


def test_failed_pipeline_is_not_qualified(known_audio):
    known_audio.pipeline.result = "failure"
    assert duplex.verified_echo_cancelled_input(known_audio.backend) is False


@pytest.mark.parametrize(("name", "value"), [
    ("SHF_BYPASS", (1,)), ("PP_ECHOONOFF", (0,)), ("AEC_NUM_MICS", (2,)),
    ("AUDIO_MGR_OP_L", (1, 0)), ("AUDIO_MGR_OP_R", (4, 0)),
    ("AUDIO_MGR_OP_L", (7, 3)), ("AEC_NUM_FARENDS", (0,)),
    ("AEC_NUM_FARENDS", (0.5,)), ("AEC_NUM_FARENDS", (3,)),
    ("AUDIO_MGR_REF_GAIN", (0.0,)), ("AUDIO_MGR_REF_GAIN", (float("nan"),)),
    ("AUDIO_MGR_REF_GAIN", (float("inf"),)), ("SHF_BYPASS", (False,)),
    ("PP_ECHOONOFF", ("1",)), ("AEC_NUM_FARENDS", (True,)),
    ("AUDIO_MGR_OP_L", (8,)), ("PP_ECHOONOFF", None),
])
def test_unverified_dsp_configuration_retains_gating(known_audio, name, value):
    known_audio.board.values[name] = value
    assert duplex.verified_echo_cancelled_input(known_audio.backend) is False
    assert known_audio.board.closed is True


def test_missing_usb_board_retains_gating(known_audio):
    known_audio.modules[duplex._CONTROL_MODULE].init_respeaker_usb = lambda: None
    assert duplex.verified_echo_cancelled_input(known_audio.backend) is False


def test_unrecognized_usb_identity_retains_gating(known_audio):
    known_audio.board.dev.idVendor = 0x2886
    known_audio.board.dev.idProduct = 0x001A
    assert duplex.verified_echo_cancelled_input(known_audio.backend) is False
    assert known_audio.board.reads == []
    assert known_audio.board.closed is True


def test_read_failure_closes_usb_and_logs_only_error_type(known_audio, caplog):
    def failed_read(name):
        raise OSError("private diagnostic details")
    known_audio.board.read_values = failed_read
    with caplog.at_level("INFO"):
        assert duplex.verified_echo_cancelled_input(known_audio.backend) is False
    assert known_audio.board.closed is True
    assert "OSError" in caplog.text
    assert "private diagnostic details" not in caplog.text


def test_missing_optional_sdk_dependency_retains_gating(known_audio, monkeypatch):
    def missing_sdk(name):
        raise ImportError("unavailable")
    monkeypatch.setattr(duplex.importlib, "import_module", missing_sdk)
    assert duplex.verified_echo_cancelled_input(known_audio.backend) is False


def test_topology_resync_retains_gating(known_audio):
    known_audio.pipeline.iterate_recurse = lambda: SimpleNamespace(next=lambda: ("resync", None))
    assert duplex.verified_echo_cancelled_input(known_audio.backend) is False


def test_native_gstreamer_error_retains_gating(known_audio, caplog):
    def failed_state(timeout):
        raise GstError("private native details")
    known_audio.pipeline.get_state = failed_state
    with caplog.at_level("INFO"):
        assert duplex.verified_echo_cancelled_input(known_audio.backend) is False
    assert "GstError" in caplog.text
    assert "private native details" not in caplog.text
