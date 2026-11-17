"""Section 2: plate reading with fast-alpr (plate detector + fast-plate-ocr).

Why fast-alpr and not fast-plate-ocr alone: fast-plate-ocr only READS a plate
that is already cropped. The vehicle model (bus / car / van) has no plate class,
so something must find the plate first. fast-alpr bundles a plate detector and
uses fast-plate-ocr for the OCR step.

Plate detector: if models/Detect_Plate_Model.pt exists it is used (see
CustomPlateDetector); otherwise fast-alpr's built-in detector is used.

Install (already in pyproject.toml):  fast-alpr[onnx]
The models are downloaded on first use, so the first run needs internet.
"""
import importlib.util
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from ultralytics import YOLO
import cv2
from purwa_yolo import app_config as cfg
from purwa_yolo.core.data import OcrStats, PlateReading
from purwa_yolo.core.plate_rules import (best_plate, clean_plate, plate_parity,
                                         plate_shape)
from purwa_yolo.device import INFER_KWARGS
from purwa_yolo.vision.crops import decode_crop


# ---------------- custom plate detector (plugs into fast-alpr) ----------------
@dataclass
class _Box:
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass
class _Detection:
    label: str
    confidence: float
    bounding_box: _Box


class CustomPlateDetector:
    """Our own YOLO plate model, shaped like the detectors fast-alpr expects:
    predict(frame) -> list of detections with .label, .confidence, .bounding_box
    (integer pixel coordinates x1, y1, x2, y2)."""

    def __init__(self, path, conf=cfg.PLATE_DETECTOR_CONF):
        self.model = YOLO(str(path))
        self.conf = conf

    def predict(self, frame):
        h, w = frame.shape[:2]
        result = self.model(frame, conf=self.conf,
                            verbose=False, **INFER_KWARGS)[0]
        if result.boxes is None:
            return []
        detections = []
        for (x1, y1, x2, y2), conf in zip(result.boxes.xyxy.tolist(),
                                          result.boxes.conf.tolist()):
            box = _Box(max(0, int(x1)), max(0, int(y1)),
                       min(w, int(x2)), min(h, int(y2)))
            if box.x2 > box.x1 and box.y2 > box.y1:
                detections.append(_Detection("plate", float(conf), box))
        return detections


def detector_name():
    """Which plate detector will be used (for display)."""
    if cfg.PLATE_MODEL_PATH.exists():
        return f"custom ({cfg.PLATE_MODEL_PATH.name})"
    return f"default ({cfg.PLATE_DETECTOR_MODEL})"


def is_available():
    """True if the fast-alpr package is installed."""
    return importlib.util.find_spec("fast_alpr") is not None


@lru_cache(maxsize=1)
def get_plate_reader():
    """Load the ALPR pipeline once (cached). Raises ValueError with a hint."""
    try:
        from fast_alpr import ALPR
        custom = (CustomPlateDetector(cfg.PLATE_MODEL_PATH)
                  if cfg.PLATE_MODEL_PATH.exists() else None)
        # `detector=None` makes fast-alpr build its own default detector
        alpr = ALPR(detector=custom, detector_model=cfg.PLATE_DETECTOR_MODEL,
                    ocr_model=cfg.PLATE_OCR_MODEL)
        alpr.predict(np.zeros((160, 320, 3), dtype=np.uint8))   # warm-up
        return alpr
    except Exception as e:
        raise ValueError(
            "The plate reader could not start. Check that fast-alpr[onnx] is "
            "installed (python -m poetry sync) and that you are online for "
            f"the first run (model download). Details: {type(e).__name__}") from e


def _mean_conf(c):
    """fast-alpr may return one float or one value per character."""
    try:
        arr = np.asarray(c, dtype=float).ravel()
        return float(arr.mean()) if arr.size else 0.0
    except Exception:
        return 0.0


def read_plate(alpr, crop_bgr, stats=None):
    """Return (plate_text, confidence) of the most confident plate found in a
    vehicle crop, or None. `stats` (OcrStats) records what happened."""
    if crop_bgr is None:
        return None
    if stats:
        stats.crops += 1
    try:
        # 1. Gunakan model YOLO Kaggle untuk cari lokasi plat di gambar mobil
        from purwa_yolo import app_config as cfg
        from ultralytics import YOLO

        # (Idealnya model ini di-load sekali saja, tapi untuk testing ini aman)
        plate_detector = YOLO(str(cfg.PLATE_MODEL_PATH))
        detections = plate_detector(crop_bgr, conf=0.15, verbose=False)

        best_plate_crop = None

        # Kalau YOLO menemukan plat, potong (crop) bagian platnya saja
        if len(detections[0].boxes) > 0:
            # Ambil kotak plat dengan confidence tertinggi
            box = detections[0].boxes[0].xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, box)
            # Potong gambar (tambahkan sedikit margin/padding)
            pad = 5
            h, w = crop_bgr.shape[:2]
            x1, y1 = max(0, x1-pad), max(0, y1-pad)
            x2, y2 = min(w, x2+pad), min(h, y2+pad)
            best_plate_crop = crop_bgr[y1:y2, x1:x2]

        # Kalau YOLO tidak nemu plat, terpaksa kirim gambar mobil utuh
        if best_plate_crop is None or best_plate_crop.size == 0:
            best_plate_crop = crop_bgr

        # 2. Perbesar gambar potongan plat (opsional, karena potongannya sudah fokus)
        h, w = best_plate_crop.shape[:2]
        if w < 200:  # Kalau masih kecil, perbesar 2x
            best_plate_crop = cv2.resize(
                best_plate_crop, (w*2, h*2), interpolation=cv2.INTER_CUBIC)

        # 3. Kirim potongan plat yang sudah bersih ke ALPR
        results = alpr.predict(best_plate_crop)

    except Exception as e:
        if stats:
            stats.errors += 1
            stats.first_error = stats.first_error or f"{type(e).__name__}: {e}"[
                :200]
        return None
    best = None
    for r in results:
        ocr = getattr(r, "ocr", None)
        if ocr is None or not getattr(ocr, "text", None):
            continue
        text, conf = clean_plate(ocr.text), _mean_conf(ocr.confidence)
        if text and (best is None or conf > best[1]):
            best = (text, conf)
    if best and stats:
        stats.plates_found += 1
    return best


def read_plates(vehicles, alpr, progress_cb=None, stats=None):
    """Read the plate of every vehicle: one PlateReading per vehicle.

    A vehicle can have several crops (video); the best-supported valid reading
    wins. Vehicles beyond MAX_OCR_VEHICLES, or without usable crops, are
    returned with text=None. Pass an OcrStats to learn why plates were missed.
    """
    stats = stats if stats is not None else OcrStats()
    readings = []
    for i, vehicle in enumerate(vehicles):
        votes = defaultdict(lambda: [0.0, 0])      # text -> [sum_conf, n]
        if i < cfg.MAX_OCR_VEHICLES:
            for jpeg in vehicle.crops:
                read = read_plate(alpr, decode_crop(jpeg), stats)
                # valid plates only
                if read:
                    votes[read[0]][0] += read[1]
                    votes[read[0]][1] += 1
                    stats.valid_reads += 1
        best = best_plate(dict(votes))
        readings.append(PlateReading(
            vehicle.vehicle_id, vehicle.cls,
            best[0] if best else None, best[1] if best else 0.0))
        if progress_cb:
            progress_cb((i + 1) / len(vehicles))
    return readings
