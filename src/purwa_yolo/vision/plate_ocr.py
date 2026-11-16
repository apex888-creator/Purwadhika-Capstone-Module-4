"""Section 2: plate reading with fast-alpr (plate detector + fast-plate-ocr).

Why fast-alpr and not fast-plate-ocr alone: fast-plate-ocr only READS a plate
that is already cropped. The vehicle model (bus / car / van) has no plate class,
so something must find the plate first. fast-alpr bundles a plate detector and
uses fast-plate-ocr for the OCR step.

Install (already in pyproject.toml):  fast-alpr[onnx]
The models are downloaded on first use, so the first run needs internet.
"""
import importlib.util
from collections import defaultdict
from functools import lru_cache

import numpy as np

from purwa_yolo import app_config as cfg
from purwa_yolo.core.data import PlateReading
from purwa_yolo.core.plate_rules import best_plate, clean_plate, plate_parity
from purwa_yolo.vision.crops import decode_crop


def is_available():
    """True if the fast-alpr package is installed."""
    return importlib.util.find_spec("fast_alpr") is not None


@lru_cache(maxsize=1)
def get_plate_reader():
    """Load the ALPR pipeline once (cached). Raises ValueError with a hint."""
    try:
        from fast_alpr import ALPR
        alpr = ALPR(detector_model=cfg.PLATE_DETECTOR_MODEL,
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


def read_plate(alpr, crop_bgr):
    """Return (plate_text, confidence) of the most confident plate found in a
    vehicle crop, or None."""
    if crop_bgr is None:
        return None
    try:
        results = alpr.predict(crop_bgr)
    except Exception:
        return None
    best = None
    for r in results:
        ocr = getattr(r, "ocr", None)
        if ocr is None or not getattr(ocr, "text", None):
            continue
        text, conf = clean_plate(ocr.text), _mean_conf(ocr.confidence)
        if text and (best is None or conf > best[1]):
            best = (text, conf)
    return best


def read_plates(vehicles, alpr, progress_cb=None):
    """Read the plate of every vehicle: one PlateReading per vehicle.

    A vehicle can have several crops (video); the best-supported valid reading
    wins. Vehicles beyond MAX_OCR_VEHICLES, or without usable crops, are
    returned with text=None.
    """
    readings = []
    for i, vehicle in enumerate(vehicles):
        votes = defaultdict(lambda: [0.0, 0])      # text -> [sum_conf, n]
        if i < cfg.MAX_OCR_VEHICLES:
            for jpeg in vehicle.crops:
                read = read_plate(alpr, decode_crop(jpeg))
                if read and plate_parity(read[0]) is not None:   # valid plates only
                    votes[read[0]][0] += read[1]
                    votes[read[0]][1] += 1
        best = best_plate(dict(votes))
        readings.append(PlateReading(
            vehicle.vehicle_id, vehicle.cls,
            best[0] if best else None, best[1] if best else 0.0))
        if progress_cb:
            progress_cb((i + 1) / len(vehicles))
    return readings
