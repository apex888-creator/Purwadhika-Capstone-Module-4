"""Vehicle crops: cut out, compress and keep the best ones for plate reading.

Section 1 (YOLO) collects the crops, section 2 (OCR) reads them later, so the
video does not have to be processed twice.
"""
from collections import defaultdict

import cv2
import numpy as np

from purwa_yolo import app_config as cfg


def vehicle_crop(frame_bgr, box, pad=0.05):
    """Cut a vehicle out of a frame (with a small margin).
    Return None if it is too small for a plate to be readable."""
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in box)
    dx, dy = (x2 - x1) * pad, (y2 - y1) * pad
    x1, y1 = max(0, int(x1 - dx)), max(0, int(y1 - dy))
    x2, y2 = min(w, int(x2 + dx)), min(h, int(y2 + dy))
    crop = frame_bgr[y1:y2, x1:x2]
    if min(crop.shape[:2]) < cfg.MIN_CROP_PX:
        return None
    return crop


def fit_width(crop):
    """Enlarge small crops (plate detector needs detail), shrink huge ones (memory)."""
    width = crop.shape[1]
    if cfg.CROP_MIN_WIDTH <= width <= cfg.CROP_MAX_WIDTH:
        return crop
    target = cfg.CROP_MIN_WIDTH if width < cfg.CROP_MIN_WIDTH else cfg.CROP_MAX_WIDTH
    interpolation = cv2.INTER_CUBIC if width < target else cv2.INTER_AREA
    height = max(1, int(crop.shape[0] * target / width))
    return cv2.resize(crop, (target, height), interpolation=interpolation)


def encode_crop(crop):
    """Crop -> JPEG bytes (about 40 KB instead of ~500 KB raw)."""
    ok, buffer = cv2.imencode(".jpg", crop,
                              [cv2.IMWRITE_JPEG_QUALITY, cfg.CROP_JPEG_QUALITY])
    if not ok:
        raise ValueError("Could not encode the vehicle crop.")
    return buffer.tobytes()


def decode_crop(data):
    """JPEG bytes -> BGR image."""
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def make_crop(frame_bgr, box):
    """Crop a vehicle and return it as JPEG bytes (None if too small)."""
    # 1. Ambil koordinat box dari YOLO
    x1, y1, x2, y2 = map(int, box)

    # 2. Fokus ke bagian bawah mobil (60% ke bawah) tempat plat nomor berada
    h = y2 - y1
    y1_bawah = y1 + int(h * 0.5)

    # 3. Buat kotak baru yang hanya bagian bawah
    box_bawah = (x1, y1_bawah, x2, y2)

    # 4. Potong gambar menggunakan fungsi bawaan Anda (vehicle_crop)
    crop = vehicle_crop(frame_bgr, box_bawah)

    # 5. Simpan hasilnya
    return None if crop is None else encode_crop(fit_width(crop))


class CropKeeper:
    """Video: keeps the few largest crops of every tracked vehicle.

    A plate is easiest to read when the vehicle is close to the camera, i.e.
    when its box is largest.
    """

    def __init__(self, per_vehicle=cfg.CROPS_PER_VEHICLE):
        self.per_vehicle = per_vehicle
        # vehicle id -> [(area, jpeg)], largest first
        self._best = defaultdict(list)

    def add(self, vehicle_id, frame_bgr, box):
        x1, y1, x2, y2 = box
        area = (x2 - x1) * (y2 - y1)
        best = self._best[vehicle_id]
        if len(best) >= self.per_vehicle and area <= best[-1][0]:
            return                        # not better than what we already have
        jpeg = make_crop(frame_bgr, box)
        if jpeg is None:
            return
        best.append((area, jpeg))
        best.sort(key=lambda item: item[0], reverse=True)
        del best[self.per_vehicle:]

    def crops(self, vehicle_id):
        return [jpeg for _, jpeg in self._best.get(vehicle_id, [])]
