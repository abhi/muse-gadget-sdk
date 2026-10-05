from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from musegadget.reachy_speaker_eq import (
    EQ_BIN_NAME,
    SPEAKER_EQ_GAINS,
    SpeakerEqError,
    install_speaker_eq,
)


class FakeCapsStructure:
    def __init__(self, fmt="F32LE", rate=48000):
        self.fmt = fmt
        self.rate = rate

    def get_string(self, name):
        return self.fmt if name == "format" else None

    def get_int(self, name):
        return (True, self.rate) if name == "rate" else (False, 0)


class FakeCaps:
    def __init__(self, text="", fmt="F32LE", rate=48000):
        self.text = text
        self.structure = FakeCapsStructure(fmt, rate)

    def get_size(self):
        return 1

    def get_structure(self, index):
        assert index == 0
        return self.structure


class FakePad:
    def __init__(self, parent=None):
        self.parent = parent
        self.peer = None
        self.current_caps = None

    def get_peer(self):
        return self.peer

    def get_parent_element(self):
        return self.parent

    def get_current_caps(self):
        return self.current_caps


class FakeFactory:
    def __init__(self, name):
        self.name = name

    def get_name(self):
        return self.name


class FakeElement:
    def __init__(self, factory, name):
        self.factory = factory
        self.name = name
        self.properties = {}
        self.pads = {"sink": FakePad(self), "src": FakePad(self)}
        self.fail_link = False
        self.state = FakeGst.State.NULL

    def get_factory(self):
        return FakeFactory(self.factory) if self.factory else None

    def get_property(self, name):
        return self.properties.get(name)

    def set_property(self, name, value):
        self.properties[name] = value

    def get_static_pad(self, name):
        return self.pads.get(name)

    def link(self, other):
        if self.fail_link:
            return False
        first, second = self.get_static_pad("src"), other.get_static_pad("sink")
        if first is None or second is None or first.peer is not None or second.peer is not None:
            return False
        first.peer, second.peer = second, first
        return True

    def unlink(self, other):
        first, second = self.get_static_pad("src"), other.get_static_pad("sink")
        if first is not None and second is not None and first.peer is second:
            first.peer = second.peer = None

    def get_state(self, timeout):
        return FakeGst.StateChangeReturn.SUCCESS, self.state, FakeGst.State.VOID_PENDING


class FakeBin(FakeElement):
    def __init__(self, name):
        super().__init__(None, name)
        self.children = []
        self.pads = {}

    def add(self, element):
        self.children.append(element)
        return True

    def add_pad(self, pad):
        if pad.name in self.pads:
            return False
        pad.parent = self
        self.pads[pad.name] = pad
        return True

    def get_by_name(self, name):
        return next((item for item in self.children if item.name == name), None)


class FakeGhostPad(FakePad):
    def __init__(self, name, target):
        super().__init__()
        self.name = name
        self.target = target
        self.current_caps = target.current_caps


class FakePipeline(FakeBin):
    def __init__(self):
        super().__init__("pipeline")
        self.state = FakeGst.State.NULL
        self.pending = FakeGst.State.VOID_PENDING

    def iterate_elements(self):
        return FakeIterator(self.children)

    def get_by_name(self, name):
        return next((item for item in self.children if item.name == name), None)

    def remove(self, element):
        self.children.remove(element)
        return True

    def get_state(self, timeout):
        return FakeGst.StateChangeReturn.SUCCESS, self.state, self.pending


class FakeElementFactory:
    @staticmethod
    def make(factory, name):
        return FakeElement(factory, name)


class FakeBinFactory:
    @staticmethod
    def new(name):
        return FakeBin(name)


class FakeGhostPadFactory:
    @staticmethod
    def new(name, target):
        return FakeGhostPad(name, target)


class FakeCapsFactory:
    @staticmethod
    def from_string(text):
        return FakeCaps(text)


class FakeGst:
    SECOND = 1_000_000_000
    State = SimpleNamespace(NULL="null", VOID_PENDING="void", PLAYING="playing", PAUSED="paused")
    StateChangeReturn = SimpleNamespace(SUCCESS="success", FAILURE="failure")
    IteratorResult = SimpleNamespace(OK="ok", DONE="done", RESYNC="resync")
    ElementFactory = FakeElementFactory
    Bin = FakeBinFactory
    GhostPad = FakeGhostPadFactory
    Caps = FakeCapsFactory


class FakeIterator:
    def __init__(self, items):
        self.items = iter(items)

    def next(self):
        try:
            return FakeGst.IteratorResult.OK, next(self.items)
        except StopIteration:
            return FakeGst.IteratorResult.DONE, None

    def resync(self):
        raise AssertionError("static fake pipeline does not resync")


class GStreamerAudio:
    def __init__(self, pipeline):
        self._pipeline = pipeline
        self._appsrc = object()


GStreamerAudio.__module__ = "reachy_mini.media.audio_gstreamer"


def make_backend(*, device="reachymini_audio_sink"):
    pipeline = FakePipeline()
    source = FakeElement("audioresample", "speaker_resampler")
    sink = FakeElement("alsasink", "speaker")
    sink.set_property("device", device)
    assert source.link(sink)
    pipeline.children.extend((FakeElement("appsink", "capture"), source, sink))
    return GStreamerAudio(pipeline), source, sink


def test_other_backend_is_deliberately_ignored():
    assert install_speaker_eq(SimpleNamespace(), _gst=FakeGst) is None


def test_sdk_with_official_eq_is_not_double_filtered(monkeypatch):
    module = ModuleType("reachy_mini.media.audio_gstreamer")
    module.make_speaker_eq = object()
    monkeypatch.setitem(sys.modules, module.__name__, module)
    backend, source, sink = make_backend()
    assert install_speaker_eq(backend, _gst=FakeGst) is None
    assert source.get_static_pad("src").get_peer().get_parent_element() is sink


def test_installs_exact_official_eq_after_unique_speaker_resampler():
    backend, source, sink = make_backend()
    installation = install_speaker_eq(backend, _gst=FakeGst)
    assert installation is not None
    assert source.get_static_pad("src").get_peer().get_parent_element() is installation.eq_bin
    assert installation.eq_bin.get_static_pad("src").get_peer().get_parent_element() is sink
    by_factory = {child.factory: child for child in installation.eq_bin.children}
    assert by_factory["capsfilter"].properties["caps"].text == "audio/x-raw,format=F32LE,rate=48000"
    assert [by_factory["equalizer-10bands"].properties[f"band{i}"] for i in range(10)] == list(SPEAKER_EQ_GAINS)
    assert by_factory["audiodynamic"].properties == {"threshold": 0.9, "ratio": 0.0}
    assert [child.factory for child in installation.eq_bin.children] == [
        "capsfilter", "equalizer-10bands", "audiodynamic", "audioconvert", "audioresample"
    ]


def test_install_is_idempotent_and_preserves_capture_elements():
    backend, _source, _sink = make_backend()
    capture = backend._pipeline.children[0]
    first = install_speaker_eq(backend, _gst=FakeGst)
    second = install_speaker_eq(backend, _gst=FakeGst)
    assert second.eq_bin is first.eq_bin
    assert capture in backend._pipeline.children
    assert sum(item.name == EQ_BIN_NAME for item in backend._pipeline.children) == 1


@pytest.mark.parametrize("device", ["default", None])
def test_rejects_non_reachy_sink_without_mutating_pipeline(device):
    backend, source, sink = make_backend(device=device)
    with pytest.raises(SpeakerEqError, match="found 0"):
        install_speaker_eq(backend, _gst=FakeGst)
    assert source.get_static_pad("src").get_peer().get_parent_element() is sink


def test_rejects_ambiguous_speaker_edges():
    backend, _source, _sink = make_backend()
    other_source = FakeElement("audioresample", "other_resampler")
    other_sink = FakeElement("alsasink", "other_sink")
    other_sink.set_property("device", "reachymini_audio_sink")
    assert other_source.link(other_sink)
    backend._pipeline.children.extend((other_source, other_sink))
    with pytest.raises(SpeakerEqError, match="found 2"):
        install_speaker_eq(backend, _gst=FakeGst)


def test_rejects_started_pipeline_before_mutation():
    backend, source, sink = make_backend()
    backend._pipeline.state = FakeGst.State.PLAYING
    with pytest.raises(SpeakerEqError, match="before media starts"):
        install_speaker_eq(backend, _gst=FakeGst)
    assert source.get_static_pad("src").get_peer().get_parent_element() is sink


def test_failed_insert_restores_original_speaker_link():
    backend, source, sink = make_backend()
    sink.fail_link = True
    original_link = source.link
    attempts = 0

    def fail_eq_only(other):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return False
        return original_link(other)

    source.link = fail_eq_only
    with pytest.raises(SpeakerEqError, match="original link restored"):
        install_speaker_eq(backend, _gst=FakeGst)
    assert source.get_static_pad("src").get_peer().get_parent_element() is sink
    assert backend._pipeline.get_by_name(EQ_BIN_NAME) is None


@pytest.mark.parametrize(
    ("fmt", "rate", "message"),
    [("S16LE", 48000, "unsafe format"), ("F32LE", 16000, "unsafe format")],
)
def test_verify_requires_native_float_rate(fmt, rate, message):
    backend, _source, _sink = make_backend()
    installation = install_speaker_eq(backend, _gst=FakeGst)
    backend._pipeline.state = FakeGst.State.PLAYING
    installation.eq_bin.state = FakeGst.State.PLAYING
    installation.eq_bin.get_by_name(EQ_BIN_NAME + "_bands").get_static_pad("sink").current_caps = FakeCaps(fmt=fmt, rate=rate)
    with pytest.raises(SpeakerEqError, match=message):
        installation.verify_active()


def test_verify_returns_negotiated_rate_when_active():
    backend, _source, _sink = make_backend()
    installation = install_speaker_eq(backend, _gst=FakeGst)
    backend._pipeline.state = FakeGst.State.PLAYING
    installation.eq_bin.state = FakeGst.State.PLAYING
    installation.eq_bin.get_by_name(EQ_BIN_NAME + "_bands").get_static_pad("sink").current_caps = FakeCaps(rate=48000)
    assert installation.verify_active() == 48000
