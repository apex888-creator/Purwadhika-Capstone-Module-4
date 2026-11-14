"""Plate reading with fast-alpr (plate detector + fast-plate-ocr).

Why fast-alpr and not fast-plate-ocr alone: fast-plate-ocr only READS a plate
that is already cropped. The vehicle model (bus / car / van) has no plate class,
so something must find the plate first. fast-alpr bundles a plate detector and
uses fast-plate-ocr for the OCR step.

Install:  python -m pip install "fast-alpr[onnx]"
(The models are downloaded on first use, so the first run needs internet.)
"""
import importlib.util
from collections import Counter, defaultdict
from fast_alpr import ALPR
import cv2
import numpy as np
import streamlit as st

from . import app_config as cfg
from .plate_rules import best_plate, clean_plate, plate_parity

PLATE_OCR_AVAILABLE = importlib.util.find_spec("fast_alpr") is not None


@st.cache_resource(show_spinner="Loading plate reader (first run downloads the models)...")
def _load_alpr():
    alpr = ALPR(detector_model=cfg.PLATE_DETECTOR_MODEL,
                ocr_model=cfg.PLATE_OCR_MODEL)
    alpr.predict(np.zeros((160, 320, 3), dtype=np.uint8))   # warm-up
    return alpr


def get_plate_reader():
    """Return the cached ALPR pipeline, or raise a ValueError with a clear hint."""
    try:
        return _load_alpr()
    except Exception as e:
        raise ValueError(
            "The plate reader could not start. Install it with: "
            "python -m pip install \"fast-alpr[onnx]\" "
            "(the first run also needs internet to download the models). "
            f"Details: {type(e).__name__}") from e


def vehicle_crop(frame_bgr, xyxy, pad=0.05):
    """Crop a vehicle (with a small margin). Return None if it is too small
    for the plate to be readable."""
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    dx, dy = (x2 - x1) * pad, (y2 - y1) * pad
    x1, y1 = max(0, int(x1 - dx)), max(0, int(y1 - dy))
    x2, y2 = min(w, int(x2 + dx)), min(h, int(y2 + dy))
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.shape[0] < cfg.MIN_OCR_CROP_PX or crop.shape[1] < cfg.MIN_OCR_CROP_PX:
        return None
    if crop.shape[1] < 320:   # upscale small crops so the plate detector can see the plate
        scale = 320 / crop.shape[1]
        crop = cv2.resize(crop, (320, int(crop.shape[0] * scale)),
                          interpolation=cv2.INTER_CUBIC)
    return crop


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


def read_plates_for_boxes(frame_bgr, names, boxes):
    """Image pipeline: one reading per detected vehicle.
    Return a list of {"cls", "text", "conf"} aligned with `names`."""
    alpr = get_plate_reader()
    out = []
    for i, (name, box) in enumerate(zip(names, boxes)):
        read = None
        if i < cfg.MAX_PLATE_READS_IMAGE:
            read = read_plate(alpr, vehicle_crop(frame_bgr, box))
        text, conf = read if read else (None, 0.0)
        out.append({"cls": name, "text": text, "conf": conf})
    return out


class PlateVoter:
    """Video pipeline: collects plate readings per tracked vehicle.

    OCR is attempted only a few times per vehicle (MAX_OCR_TRIES), spread over
    different frames, and the best-supported reading wins. This keeps video
    processing fast instead of running OCR on every frame.
    """

    def __init__(self, alpr):
        self.alpr = alpr
        self.tries = Counter()              # track id -> OCR attempts so far
        self.last = {}                      # track id -> frame index of last attempt
        self.votes = defaultdict(dict)      # track id -> {text: [sum_conf, n]}

    def update(self, tid, frame_bgr, box, frame_index):
        if self.tries[tid] >= cfg.MAX_OCR_TRIES:
            return
        if frame_index - self.last.get(tid, -cfg.OCR_EVERY_N_FRAMES) < cfg.OCR_EVERY_N_FRAMES:
            return
        crop = vehicle_crop(frame_bgr, box)
        if crop is None:
            return
        self.tries[tid] += 1
        self.last[tid] = frame_index
        read = read_plate(self.alpr, crop)
        # only valid plates vote
        if read and plate_parity(read[0]) is not None:
            vote = self.votes[tid].setdefault(read[0], [0.0, 0])
            vote[0] += read[1]
            vote[1] += 1

    def best(self, tid):
        return best_plate(self.votes.get(tid))
