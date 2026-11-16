"""Plate reading with a fake fast-alpr (no model download needed)."""
from types import SimpleNamespace
from unittest import mock

import numpy as np

from purwa_yolo import app_config as cfg
from purwa_yolo.core.data import VehicleCrops
from purwa_yolo.vision import crops, plate_ocr


def result(text, conf):
    return SimpleNamespace(ocr=SimpleNamespace(text=text, confidence=conf))


class FakeAlpr:
    """Answers by the brightness of the crop: {brightness: [results]}."""

    def __init__(self, by_brightness):
        self.by_brightness, self.calls = by_brightness, 0

    def predict(self, crop):
        self.calls += 1
        return self.by_brightness.get(round(float(crop.mean()) / 10) * 10, [])


def jpeg(brightness):
    return crops.encode_crop(np.full((200, 320, 3), brightness, np.uint8))


def test_text_is_cleaned_and_per_character_confidence_is_averaged():
    alpr = FakeAlpr({100: [result("b 1234-xyz", [0.9, 0.8, 0.85, 0.9])]})
    text, conf = plate_ocr.read_plate(alpr, np.full((200, 320, 3), 100, np.uint8))
    assert text == "B1234XYZ" and abs(conf - 0.8625) < 1e-9


def test_most_confident_plate_wins_when_several_are_found():
    alpr = FakeAlpr({100: [result("A1B", 0.4), result("B1234XYZ", 0.9)]})
    assert plate_ocr.read_plate(alpr, np.full((200, 320, 3), 100, np.uint8))[0] == "B1234XYZ"


def test_missing_ocr_empty_text_and_errors_give_none():
    img = np.full((200, 320, 3), 100, np.uint8)
    assert plate_ocr.read_plate(FakeAlpr({}), img) is None
    assert plate_ocr.read_plate(FakeAlpr({100: [SimpleNamespace(ocr=None)]}), img) is None
    assert plate_ocr.read_plate(FakeAlpr({100: [result("", 0.9)]}), img) is None
    assert plate_ocr.read_plate(FakeAlpr({}), None) is None

    class Broken:
        def predict(self, crop):
            raise RuntimeError("onnx failed")
    assert plate_ocr.read_plate(Broken(), img) is None


def test_readings_vote_across_the_crops_of_one_vehicle():
    alpr = FakeAlpr({
        100: [result("B1234XYZ", 0.9)],
        110: [result("B1234XYZ", 0.8)],
        120: [result("B1284XYZ", 0.95)],        # one misread
    })
    vehicle = VehicleCrops(5, "car", [jpeg(100), jpeg(110), jpeg(120)])
    (reading,) = plate_ocr.read_plates([vehicle], alpr)
    assert reading.vehicle_id == 5 and reading.cls == "car"
    assert reading.text == "B1234XYZ" and abs(reading.conf - 0.85) < 1e-9


def test_invalid_plate_text_never_wins():
    alpr = FakeAlpr({100: [result("HELLO", 0.99)]})
    (reading,) = plate_ocr.read_plates([VehicleCrops(1, "car", [jpeg(100)])], alpr)
    assert reading.text is None and reading.conf == 0.0


def test_vehicle_without_crops_is_returned_unreadable():
    alpr = FakeAlpr({})
    (reading,) = plate_ocr.read_plates([VehicleCrops(1, "van", [])], alpr)
    assert reading.text is None and alpr.calls == 0


def test_one_reading_per_vehicle_in_the_same_order():
    alpr = FakeAlpr({100: [result("B1234XYZ", 0.9)]})
    vehicles = [VehicleCrops(3, "car", [jpeg(100)]), VehicleCrops(9, "bus", [])]
    readings = plate_ocr.read_plates(vehicles, alpr)
    assert [r.vehicle_id for r in readings] == [3, 9]
    assert [r.text for r in readings] == ["B1234XYZ", None]


def test_vehicle_cap_limits_the_number_of_ocr_calls():
    alpr = FakeAlpr({100: [result("B1234XYZ", 0.9)]})
    vehicles = [VehicleCrops(i, "car", [jpeg(100)]) for i in range(1, 6)]
    with mock.patch.object(cfg, "MAX_OCR_VEHICLES", 2):
        readings = plate_ocr.read_plates(vehicles, alpr)
    assert len(readings) == 5 and alpr.calls == 2
    assert [r.text is not None for r in readings] == [True, True, False, False, False]


def test_progress_callback_reaches_one():
    seen = []
    alpr = FakeAlpr({})
    plate_ocr.read_plates([VehicleCrops(i, "car", []) for i in range(4)], alpr, seen.append)
    assert seen == [0.25, 0.5, 0.75, 1.0]
