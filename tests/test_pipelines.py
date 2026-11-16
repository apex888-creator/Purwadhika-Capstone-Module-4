"""Image and video pipelines with a fake YOLO model, tracker and annotators."""
import io
import os
import tempfile
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np
from PIL import Image

from purwa_yolo import app_config as cfg
from purwa_yolo.vision import crops, detector, pipelines


class FakeDetections:
    def __init__(self, boxes, names):
        self.xyxy = np.array(boxes, dtype=float).reshape(-1, 4)
        self.data = {"class_name": np.array(names)} if names else {}
        self.tracker_id = None

    def __len__(self):
        return len(self.xyxy)

    def with_nms(self):
        return self


class FakeTracker:
    def __init__(self, frame_rate=30):
        pass

    def update_with_detections(self, detections):       # ids by position: 1, 2, 3...
        detections.tracker_id = np.arange(1, len(detections) + 1)
        return detections


class FakeLabelAnnotator:
    seen = []

    def annotate(self, scene, detections, labels=None):
        FakeLabelAnnotator.seen.append(None if labels is None else list(labels))
        return scene


class FakeBoxAnnotator:
    def annotate(self, scene, detections):
        return scene


def fake_model(boxes, names):
    """A callable like a YOLO model: always finds the same vehicles."""
    return lambda frame, **kw: [FakeDetections(boxes, names)]


@contextmanager
def fake_supervision(tracker_works=True):
    def broken(*a, **k):
        raise RuntimeError("no tracker in this supervision version")
    fake = SimpleNamespace(
        Detections=SimpleNamespace(from_ultralytics=lambda r: r),
        BoxAnnotator=FakeBoxAnnotator, LabelAnnotator=FakeLabelAnnotator,
        ByteTrack=FakeTracker if tracker_works else broken)
    detector.get_annotators.cache_clear()
    FakeLabelAnnotator.seen.clear()
    with mock.patch.object(detector, "sv", fake), mock.patch.object(pipelines, "sv", fake):
        yield
    detector.get_annotators.cache_clear()


def png_bytes(w=640, h=480):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (90, 90, 90)).save(buf, "PNG")
    return buf.getvalue()


def make_video(n_frames=30, w=320, h=240, fps=10):
    path = os.path.join(tempfile.mkdtemp(), "in.avi")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    for _ in range(n_frames):
        frame = np.full((h, w, 3), 100, np.uint8)
        frame[60:200, 10:150] = 220
        frame[60:200, 170:310] = 40
        writer.write(frame)
    writer.release()
    with open(path, "rb") as f:
        return f.read()


TWO_CARS = fake_model([[10, 60, 150, 200], [170, 60, 310, 200]], ["car", "car"])


# ---------------- image ----------------
def test_image_counts_labels_and_media():
    model = fake_model([[20, 100, 260, 300], [300, 100, 560, 300], [20, 320, 300, 470]],
                       ["car", "car", "van"])
    with fake_supervision():
        det = pipelines.process_image(png_bytes(), model, 0.1, 640)
    assert det.kind == "image" and det.counts == {"car": 2, "van": 1}
    assert det.density == 3 and det.media.shape == (480, 640, 3)
    assert FakeLabelAnnotator.seen[-1] == ["#1 car", "#2 car", "#3 van"]


def test_image_keeps_one_crop_per_vehicle_only_when_asked():
    model = fake_model([[20, 100, 260, 300], [300, 100, 560, 300]], ["car", "car"])
    with fake_supervision():
        with_crops = pipelines.process_image(png_bytes(), model, 0.1, 640, collect_crops=True)
        without = pipelines.process_image(png_bytes(), model, 0.1, 640)
    assert [v.vehicle_id for v in with_crops.vehicles] == [1, 2]
    assert all(len(v.crops) == 1 and isinstance(v.crops[0], bytes) for v in with_crops.vehicles)
    assert without.vehicles == []


def test_image_vehicle_that_is_too_small_is_kept_without_crops():
    model = fake_model([[10, 10, 50, 40]], ["car"])
    with fake_supervision():
        det = pipelines.process_image(png_bytes(), model, 0.1, 640, collect_crops=True)
    assert len(det.vehicles) == 1 and det.vehicles[0].crops == []    # still counted


def test_image_without_detections():
    with fake_supervision():
        det = pipelines.process_image(png_bytes(), fake_model([], []), 0.1, 640, collect_crops=True)
    assert det.counts == {} and det.vehicles == [] and det.density == 0


# ---------------- video ----------------
def test_video_counts_each_tracked_vehicle_once_and_keeps_best_crops():
    with fake_supervision():
        det = pipelines.process_video(make_video(), ".avi", TWO_CARS, 0.1, 640, 1, 5,
                                      lambda p: None, collect_crops=True)
    assert det.kind == "video" and det.counts == {"car": 2}
    assert det.tracked and det.frames == 30 and det.density == 2
    assert [v.vehicle_id for v in det.vehicles] == [1, 2]
    assert all(len(v.crops) == cfg.CROPS_PER_VEHICLE for v in det.vehicles)
    left = crops.decode_crop(det.vehicles[0].crops[0]).mean()
    right = crops.decode_crop(det.vehicles[1].crops[0]).mean()
    assert left > 150 > right                                      # crops belong to the right vehicle
    assert len(det.media) > 0


def test_video_without_crop_collection_keeps_no_crops():
    with fake_supervision():
        det = pipelines.process_video(make_video(), ".avi", TWO_CARS, 0.1, 640, 1, 5,
                                      lambda p: None)
    assert det.counts == {"car": 2} and all(v.crops == [] for v in det.vehicles)


def test_video_labels_show_tracker_ids():
    with fake_supervision():
        pipelines.process_video(make_video(), ".avi", TWO_CARS, 0.1, 640, 1, 5, lambda p: None)
    assert FakeLabelAnnotator.seen[-1] == ["#1 car", "#2 car"]


def test_video_without_tracker_falls_back_to_the_maximum_per_frame():
    with fake_supervision(tracker_works=False):
        det = pipelines.process_video(make_video(), ".avi", TWO_CARS, 0.1, 640, 1, 5,
                                      lambda p: None, collect_crops=True)
    assert det.tracked is False and det.counts == {"car": 2} and det.vehicles == []


def test_video_frame_skip_and_duration_limit():
    with fake_supervision():
        det = pipelines.process_video(make_video(n_frames=30, fps=10), ".avi", TWO_CARS,
                                      0.1, 640, 2, 1, lambda p: None)   # 1 s at 10 fps, every 2nd
    assert det.frames == 5


def test_corrupt_video_raises_a_friendly_error():
    with fake_supervision():
        try:
            pipelines.process_video(b"not a video", ".mp4", TWO_CARS, 0.1, 640, 1, 5, lambda p: None)
        except ValueError as e:
            assert "Could not open" in str(e)
        else:
            raise AssertionError("expected ValueError")


def test_temporary_files_are_removed():
    before = set(os.listdir(tempfile.gettempdir()))
    with fake_supervision():
        pipelines.process_video(make_video(), ".avi", TWO_CARS, 0.1, 640, 1, 5, lambda p: None)
    leftovers = {f for f in set(os.listdir(tempfile.gettempdir())) - before
                 if f.endswith((".avi", ".mp4"))}
    assert not leftovers
