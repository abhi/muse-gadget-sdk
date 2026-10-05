# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

import math
import threading
import time

import pytest

from musegadget.local_face_tracking import (
    FACE_LOSS_SECONDS, PITCH_LIMIT, YAW_LIMIT, FaceFollower, _Face, _FaceTracker,
    default_face_model,
)


def face(x=0.4, y=0.3, width=0.2, height=0.3, yaw=0, pitch=0):
    return _Face((x, y, width, height), math.radians(yaw), math.radians(pitch), 0.95)


def wait_for(condition, timeout=1):
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() >= deadline:
            raise AssertionError('Face detector did not reach the expected state')
        time.sleep(0.002)


def test_target_uses_current_head_pose_and_deadband():
    tracker = _FaceTracker(0.2)
    target = tracker.update([face(yaw=0.5, pitch=-0.5)],
                            (math.radians(8), math.radians(-4)), 1)
    assert target.yaw == pytest.approx(math.radians(8))
    assert target.pitch == pytest.approx(math.radians(-4))
    target = tracker.update([face(yaw=8, pitch=5)],
                            (math.radians(10), math.radians(-2)), 1.2)
    assert math.radians(8) < target.yaw < math.radians(18)
    assert math.radians(-4) < target.pitch < math.radians(3)
    assert target.observed_at == 1.2


def test_extreme_targets_stay_within_bounds_and_slew_rate():
    tracker = _FaceTracker(0.2)
    previous = (0.0, 0.0)
    for index in range(30):
        target = tracker.update([face(yaw=120, pitch=-90)], previous, 1 + index * 0.2)
        assert abs(target.yaw) <= YAW_LIMIT
        assert abs(target.pitch) <= PITCH_LIMIT
        assert abs(target.yaw - previous[0]) <= math.radians(7) + 1e-12
        assert abs(target.pitch - previous[1]) <= math.radians(5) + 1e-12
        previous = target.yaw, target.pitch
    assert target.yaw == pytest.approx(YAW_LIMIT, abs=1e-7)
    assert target.pitch == pytest.approx(-PITCH_LIMIT, abs=1e-7)


def test_hysteresis_keeps_existing_face_when_larger_face_arrives():
    tracker = _FaceTracker(0.2)
    first = face(x=0.1, width=0.15, yaw=8)
    other = face(x=0.65, width=0.3, yaw=-15)
    tracker.update([first], (0, 0), 1)
    target = tracker.update([other, face(x=0.11, width=0.15, yaw=8)], (0, 0), 1.2)
    assert target.yaw > 0


def test_largest_initial_face_is_selected_without_identity_recognition():
    target = _FaceTracker(0.2).update(
        [face(width=0.1, yaw=-8), face(x=0.65, width=0.3, yaw=8)], (0, 0), 1)
    assert target.yaw > 0


def test_unmatched_faces_do_not_extend_loss_timeout_and_reacquire_after_loss():
    tracker = _FaceTracker(0.2)
    original = tracker.update([face(x=0.1, width=0.1, yaw=8)], (0, 0), 1)
    different = face(x=0.75, width=0.2, yaw=-8)
    assert tracker.update([different], (0, 0), 1.2) == original
    assert tracker.update([], (0, 0), 1.6) == original
    assert tracker.update([], (0, 0), 1 + FACE_LOSS_SECONDS) is None
    assert tracker.update([different], (0, 0), 2).yaw < 0


@pytest.mark.parametrize('values', [(float('nan'), 0), (0, float('inf'))])
def test_invalid_head_snapshot_never_becomes_a_motion_target(values):
    assert _FaceTracker(0.2).update([face(yaw=8)], values, 1) is None


def test_default_packaged_model_is_small_and_hash_pinned():
    import hashlib
    import json

    model = default_face_model()
    manifest = json.loads((model.parent / 'SOURCE.json').read_text())
    assert model.stat().st_size == 232589
    assert hashlib.sha256(model.read_bytes()).hexdigest() == manifest['model_sha256']
    assert manifest['model_sha256'] == '8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4'
    assert (model.parent / 'LICENSE').read_text().startswith('MIT License')


class Detector:
    def __init__(self, *args):
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        return [face(yaw=10)]


def follower(read_frame, **kwargs):
    return FaceFollower(read_frame, read_head_angles=lambda: (0, 0),
                        camera_matrix=None, sample_hz=15, detector_factory=Detector, **kwargs)


def test_worker_is_inactive_until_enabled_and_clears_target_on_close():
    reads = []
    item = follower(lambda: reads.append('camera') or object())
    try:
        item.start()
        item.start()
        time.sleep(0.02)
        assert reads == []
        item.set_active(True)
        wait_for(lambda: item.target() is not None)
        assert item.target().yaw > 0
        item.set_active(False)
        assert item.target() is None
        count = len(reads)
        time.sleep(0.08)
        assert len(reads) == count
    finally:
        item.close()
    assert not item._thread.is_alive()
    assert item.target() is None


def test_deactivation_rejects_inflight_frame_result():
    entered, release = threading.Event(), threading.Event()

    def read():
        entered.set()
        assert release.wait(1)
        return object()

    item = follower(read)
    try:
        item.start()
        item.set_active(True)
        assert entered.wait(1)
        item.set_active(False)
        release.set()
        time.sleep(0.02)
        assert item.target() is None
    finally:
        release.set()
        item.close()


def test_old_session_target_is_not_reused_after_reactivation():
    reads = []
    item = follower(lambda: reads.append(1) or (object() if len(reads) == 1 else None))
    try:
        item.start()
        item.set_active(True)
        wait_for(lambda: item.target() is not None)
        item.set_active(False)
        item.set_active(True)
        wait_for(lambda: len(reads) >= 2)
        assert item.target() is None
    finally:
        item.close()


def test_no_new_camera_frame_expires_target_without_retaining_images():
    current = [1.0]
    reads = []
    item = follower(lambda: reads.append(1) or (object() if len(reads) == 1 else None),
                    clock=lambda: current[0])
    try:
        item.start()
        item.set_active(True)
        wait_for(lambda: item.target() is not None)
        current[0] += FACE_LOSS_SECONDS
        assert item.target() is None
        assert not any(name in vars(item) for name in ('frame', '_frame', 'image', '_image'))
    finally:
        item.close()


def test_camera_failure_is_reported_without_publishing_pose_or_retrying_forever():
    reads = []

    def failed():
        reads.append(1)
        raise RuntimeError('Camera unavailable')

    item = follower(failed)
    item.start()
    item.set_active(True)
    wait_for(lambda: item.failure is not None)
    assert isinstance(item.failure, RuntimeError)
    assert item.target() is None
    item.close()
    assert reads == [1]


def test_detector_failure_does_not_retain_camera_frame_in_traceback():
    import gc
    import weakref

    references = []

    class Frame:
        pass

    class FailedDetector:
        def __init__(self, *args):
            pass

        def detect(self, frame):
            raise RuntimeError('Detection unavailable')

    def read():
        frame = Frame()
        references.append(weakref.ref(frame))
        return frame

    item = FaceFollower(read, read_head_angles=lambda: (0, 0), camera_matrix=None,
                        detector_factory=FailedDetector)
    item.start()
    item.set_active(True)
    wait_for(lambda: item.failure is not None)
    item.close()
    gc.collect()
    assert item.failure.__traceback__ is None
    assert item.failure.__context__ is None
    assert references[0]() is None


def test_missing_camera_calibration_fails_before_worker_starts(tmp_path):
    np = pytest.importorskip('numpy')
    pytest.importorskip('cv2')
    item = FaceFollower(lambda: None, read_head_angles=lambda: (0, 0), camera_matrix=None)
    with pytest.raises(ValueError, match='calibrated camera matrix'):
        item.start()
    assert item._thread is None


@pytest.mark.parametrize('rate', [0, 0.5, 16, float('inf'), float('nan')])
def test_unbounded_detector_rate_is_rejected(rate):
    with pytest.raises(ValueError, match='sample rate'):
        FaceFollower(lambda: None, read_head_angles=lambda: (0, 0), camera_matrix=None,
                     sample_hz=rate)
