"""Vehicle crops: cutting, resizing, JPEG round-trip and keeping the best ones."""
import numpy as np

from purwa_yolo.vision import crops


def solid(h, w, value):
    return np.full((h, w, 3), value, np.uint8)


def test_too_small_vehicle_gives_no_crop():
    frame = solid(480, 640, 100)
    assert crops.vehicle_crop(frame, [10, 10, 50, 40]) is None


def test_crop_is_clipped_to_the_frame():
    frame = solid(480, 640, 100)
    crop = crops.vehicle_crop(frame, [500, 300, 700, 520])    # sticks out of the frame
    assert crop is not None and crop.shape[0] <= 480 and crop.shape[1] <= 640


def test_fit_width_enlarges_small_and_shrinks_large_crops():
    small = crops.fit_width(solid(100, 100, 50))
    large = crops.fit_width(solid(900, 1200, 50))
    ok = crops.fit_width(solid(300, 400, 50))
    assert small.shape[1] == 320 and large.shape[1] == 480 and ok.shape[1] == 400
    assert small.shape[0] == 320                              # aspect ratio kept


def test_jpeg_round_trip_keeps_the_image():
    original = solid(200, 400, 120)
    restored = crops.decode_crop(crops.encode_crop(original))
    assert restored.shape == original.shape
    assert abs(float(restored.mean()) - 120) < 3


def test_make_crop_returns_jpeg_bytes_or_none():
    frame = solid(480, 640, 100)
    assert isinstance(crops.make_crop(frame, [20, 20, 300, 260]), bytes)
    assert crops.make_crop(frame, [0, 0, 30, 30]) is None


def test_keeper_keeps_only_the_largest_crops_of_a_vehicle():
    keeper = crops.CropKeeper(per_vehicle=3)
    # frame i is filled with brightness 20*i and the box grows with i
    for i in range(1, 9):
        frame = solid(480, 640, 20 * i)
        size = 80 + 20 * i
        keeper.add(7, frame, [10, 10, 10 + size, 10 + size])
    kept = keeper.crops(7)
    brightness = [round(float(crops.decode_crop(c).mean()) / 20) for c in kept]
    assert brightness == [8, 7, 6]                            # largest first


def test_keeper_separates_vehicles_and_handles_unknown_ids():
    keeper = crops.CropKeeper(per_vehicle=2)
    keeper.add(1, solid(480, 640, 50), [10, 10, 200, 200])
    keeper.add(2, solid(480, 640, 90), [10, 10, 200, 200])
    assert len(keeper.crops(1)) == 1 and len(keeper.crops(2)) == 1
    assert keeper.crops(99) == []


def test_keeper_skips_boxes_that_are_too_small():
    keeper = crops.CropKeeper()
    keeper.add(1, solid(480, 640, 50), [0, 0, 20, 20])
    assert keeper.crops(1) == []
