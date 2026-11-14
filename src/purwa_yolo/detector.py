"""YOLO detection, tracking, annotation and the image / video pipelines."""
import os
import tempfile
from collections import Counter, defaultdict
from io import BytesIO

import cv2
import numpy as np
import streamlit as st
import supervision as sv
from PIL import Image
from ultralytics import YOLO

from . import app_config as cfg
from . import plate_ocr
from .device import DEVICE, USE_HALF
from .plate_rules import mask_plate, plate_parity


# ---------------- model files ----------------
def resolve_model_path(use_case):
    """Return the model path for a use case, or None if the file is missing."""
    path = cfg.MODELS_DIR / cfg.MODEL_FILES[use_case]
    if path.exists():
        return path
    fallback = cfg.MODELS_DIR / cfg.FALLBACK_VEHICLE_MODEL
    if use_case == "Vehicle" and fallback.exists():
        return fallback
    return None


# Cache model loading - models are loaded only once per file
@st.cache_resource
def load_model(model_path: str):
    """Load the YOLO model once (cached) and warm it up so the first real
    detection is not slow (lazy initialisation, layer fusing, etc.)."""
    model = YOLO(model_path)
    model(np.zeros((cfg.DEFAULT_IMGSZ, cfg.DEFAULT_IMGSZ, 3), dtype=np.uint8),
          imgsz=cfg.DEFAULT_IMGSZ, device=DEVICE, half=USE_HALF, verbose=False)
    return model


# Cache annotators - reuse same annotator objects
@st.cache_resource
def get_annotators():
    return sv.BoxAnnotator(), sv.LabelAnnotator()


# ---------------- detection ----------------
def run_detection(frame_bgr, model, conf, imgsz, tracker=None):
    """Run the YOLO model on one frame (BGR, OpenCV format).
    Return (list of class names, detections)."""
    results = model(frame_bgr, conf=conf, imgsz=imgsz, device=DEVICE,
                    half=USE_HALF, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results).with_nms()
    if tracker is not None:
        detections = tracker.update_with_detections(detections)
    names = [str(n) for n in detections.data.get("class_name", [])]
    return names, detections


def annotate_frame(frame_bgr, detections, labels=None):
    """Draw boxes and labels. labels=None uses the class names."""
    annotated = frame_bgr.copy()
    if len(detections) > 0:
        box_annotator, label_annotator = get_annotators()
        annotated = box_annotator.annotate(
            scene=annotated, detections=detections)
        if labels is not None:
            annotated = label_annotator.annotate(
                scene=annotated, detections=detections, labels=labels)
        else:
            annotated = label_annotator.annotate(
                scene=annotated, detections=detections)
    return annotated


# ---------------- image pipeline ----------------
def process_image(image_bytes, model, conf, imgsz, read_plates=False):
    """Return (annotated RGB image, dict of counts per class, list of
    per-vehicle plate readings)."""
    pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    pil_image.thumbnail((1920, 1920))   # saves memory, keeps aspect ratio
    frame_bgr = np.ascontiguousarray(np.array(pil_image)[:, :, ::-1])

    names, detections = run_detection(frame_bgr, model, conf, imgsz)

    labels, plates = list(names), []
    if read_plates and names:
        plates = plate_ocr.read_plates_for_boxes(
            frame_bgr, names, detections.xyxy)
        for i, p in enumerate(plates):
            if p["text"] and plate_parity(p["text"]):
                # always masked on the image
                labels[i] = f"{p['cls']} {mask_plate(p['text'])}"

    annotated_bgr = annotate_frame(frame_bgr, detections, labels)
    annotated_rgb = np.ascontiguousarray(annotated_bgr[:, :, ::-1])
    return annotated_rgb, dict(Counter(names)), plates


# ---------------- video pipeline ----------------
def open_video_writer(path, fps, size):
    """Open a video writer. Prefer H.264 (playable in the browser).
    Return (write(frame_bgr), close, is_h264)."""
    try:
        import imageio.v2 as imageio
        writer = imageio.get_writer(
            path, fps=fps, codec="libx264", pixelformat="yuv420p",
            macro_block_size=None, quality=7)

        def write(frame_bgr):
            writer.append_data(np.ascontiguousarray(frame_bgr[:, :, ::-1]))
        return write, writer.close, True
    except Exception:
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        return vw.write, vw.release, False


def process_video(video_bytes, suffix, model, conf, imgsz, frame_skip,
                  max_seconds, progress_cb, read_plates=False):
    """Detect + track a video frame by frame (streaming).
    Return (mp4 bytes, dict of UNIQUE vehicle counts per class, number of
    processed frames, is_h264, tracked, most vehicles seen in a single frame,
    list of per-vehicle plate readings)."""
    in_path = out_path = None
    cap = None
    close_writer = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(video_bytes)
            in_path = f.name

        cap = cv2.VideoCapture(in_path)
        if not cap.isOpened():
            raise ValueError(
                "Could not open the video. Try converting it to MP4 (H.264).")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        scale = min(1.0, cfg.MAX_VIDEO_SIDE / max(w, h))
        out_w, out_h = int(w * scale) // 2 * 2, int(h * scale) // 2 * 2  # even
        max_frames = int(fps * max_seconds)
        limit = min(total, max_frames) if total else max_frames

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
            out_path = f.name
        # Output fps is reduced so the duration stays the same when frames are skipped
        write, close_writer, is_h264 = open_video_writer(
            out_path, fps / frame_skip, (out_w, out_h))

        # Tracker so each vehicle is counted only once
        try:
            tracker = sv.ByteTrack(
                frame_rate=max(1, int(round(fps / frame_skip))))
        except Exception:
            tracker = None   # fallback: max count per frame

        # Plates need a per-vehicle identity, so they require the tracker
        voter = None
        if read_plates and tracker is not None:
            voter = plate_ocr.PlateVoter(plate_ocr.get_plate_reader())

        track_votes = defaultdict(Counter)   # track id -> class name votes
        max_counts = Counter()               # fallback when no tracker
        max_simultaneous = 0                 # most vehicles in a single frame

        idx = processed = 0
        while idx < max_frames:
            if idx % frame_skip == 0:
                ok, frame = cap.read()
            else:
                ok, frame = cap.grab(), None   # skip without full decode
            if not ok:
                break

            if frame is not None:
                if (frame.shape[1], frame.shape[0]) != (out_w, out_h):
                    frame = cv2.resize(frame, (out_w, out_h))
                names, detections = run_detection(
                    frame, model, conf, imgsz, tracker)

                labels = None
                if tracker is not None:
                    labels = []
                    for tid, name, box in zip(detections.tracker_id, names,
                                              detections.xyxy):
                        tid = int(tid)
                        label = f"#{tid} {name}"
                        if voter is not None:
                            voter.update(tid, frame, box, processed)
                            best = voter.best(tid)
                            if best:
                                label += f" {mask_plate(best[0])}"
                        labels.append(label)

                write(annotate_frame(frame, detections, labels))

                max_simultaneous = max(max_simultaneous, len(names))
                for name, n in Counter(names).items():
                    max_counts[name] = max(max_counts[name], n)
                if tracker is not None:
                    for tid, name in zip(detections.tracker_id, names):
                        track_votes[int(tid)][name] += 1
                processed += 1

            idx += 1
            if limit and idx % 5 == 0:
                progress_cb(min(idx / limit, 1.0))

        close_writer()
        close_writer = None
        with open(out_path, "rb") as f:
            result_bytes = f.read()

        plates = []
        if tracker is not None:
            unique = Counter()
            for tid, votes in track_votes.items():
                if sum(votes.values()) >= cfg.MIN_TRACK_FRAMES:
                    cls = votes.most_common(1)[0][0]
                    unique[cls] += 1
                    best = voter.best(tid) if voter is not None else None
                    plates.append({"cls": cls,
                                   "text": best[0] if best else None,
                                   "conf": best[1] if best else 0.0})
            counts = dict(unique)
        else:
            counts = dict(max_counts)

        return (result_bytes, counts, processed, is_h264,
                tracker is not None, max_simultaneous, plates)
    finally:
        if cap is not None:
            cap.release()
        if close_writer is not None:
            close_writer()
        for p in (in_path, out_path):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
