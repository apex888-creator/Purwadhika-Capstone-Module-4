"""YOLO model loading, detection and drawing (no Streamlit)."""
from functools import lru_cache

import numpy as np
import supervision as sv
from ultralytics import YOLO

from purwa_yolo import app_config as cfg
from purwa_yolo.device import INFER_KWARGS


def resolve_model_path(use_case):
    """Return the model path for a use case, or None if the file is missing."""
    path = cfg.MODELS_DIR / cfg.MODEL_FILES[use_case]
    if path.exists():
        return path
    fallback = cfg.MODELS_DIR / cfg.FALLBACK_VEHICLE_MODEL
    if use_case == "Vehicle" and fallback.exists():
        return fallback
    return None


@lru_cache(maxsize=4)
def load_model(model_path: str):
    """Load a YOLO model once per file (cached) and warm it up so the first
    real detection is not slow."""
    model = YOLO(model_path)
    model(np.zeros((cfg.DEFAULT_IMGSZ, cfg.DEFAULT_IMGSZ, 3), dtype=np.uint8),
          imgsz=cfg.DEFAULT_IMGSZ, verbose=False, **INFER_KWARGS)
    return model


@lru_cache(maxsize=1)
def get_annotators():
    return sv.BoxAnnotator(), sv.LabelAnnotator()


def run_detection(frame_bgr, model, conf, imgsz, tracker=None):
    """Run the model on one frame (BGR, OpenCV format).
    Return (list of class names, detections). With a tracker, every detection
    also gets a tracker_id."""
    results = model(frame_bgr, conf=conf, imgsz=imgsz, verbose=False,
                    **INFER_KWARGS)[0]
    detections = sv.Detections.from_ultralytics(results).with_nms()
    if tracker is not None:
        detections = tracker.update_with_detections(detections)
    names = [str(n) for n in detections.data.get("class_name", [])]
    return names, detections


def annotate_frame(frame_bgr, detections, labels=None):
    """Draw bounding boxes and labels. labels=None uses the class names."""
    annotated = frame_bgr.copy()
    if len(detections) > 0:
        box_annotator, label_annotator = get_annotators()
        annotated = box_annotator.annotate(scene=annotated, detections=detections)
        if labels is not None:
            annotated = label_annotator.annotate(
                scene=annotated, detections=detections, labels=labels)
        else:
            annotated = label_annotator.annotate(
                scene=annotated, detections=detections)
    return annotated
