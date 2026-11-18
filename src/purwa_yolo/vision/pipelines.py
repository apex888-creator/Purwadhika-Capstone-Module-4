"""Section 1 pipelines: YOLO detection on an image or on a video.

Both return a DetectionResult. When `collect_crops` is True they also keep the
vehicle crops, so section 2 (plate OCR) does not need to run YOLO again.
"""
import os
import tempfile
from collections import Counter, defaultdict
from io import BytesIO

import cv2
import numpy as np
import supervision as sv
from PIL import Image

from purwa_yolo import app_config as cfg
from purwa_yolo.core.data import DetectionResult, VehicleCrops
from purwa_yolo.vision import crops
from purwa_yolo.vision.detector import annotate_frame, run_detection


# ---------------- image ----------------
def process_image(image_bytes, model, conf, imgsz, collect_crops=False):
    pil_full = Image.open(BytesIO(image_bytes)).convert("RGB")
    full_bgr = np.ascontiguousarray(
        np.array(pil_full)[:, :, ::-1])   # resolusi asli

    pil_image = pil_full.copy()
    pil_image.thumbnail((1920, 1920))
    frame_bgr = np.ascontiguousarray(np.array(pil_image)[:, :, ::-1])
    scale = full_bgr.shape[1] / frame_bgr.shape[1]                    # >= 1

    names, detections = run_detection(frame_bgr, model, conf, imgsz)
    labels = [f"#{i} {name}" for i, name in enumerate(names, start=1)]
    annotated_bgr = annotate_frame(frame_bgr, detections, labels)

    vehicles = []
    if collect_crops:
        for i, (name, box) in enumerate(zip(names, detections.xyxy), start=1):
            x1, y1, x2, y2 = (int(v * scale)
                              for v in box)            # skala ke gambar asli
            h, w = full_bgr.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop_img = full_bgr[y1:y2, x1:x2]

            jpeg = None
            if crop_img.size > 0:
                _, buffer = cv2.imencode(".jpg", crop_img,
                                         [cv2.IMWRITE_JPEG_QUALITY, 95])
                jpeg = buffer.tobytes()
            vehicles.append(VehicleCrops(i, name, [jpeg] if jpeg else []))

    return DetectionResult(
        kind="image",
        media=np.ascontiguousarray(
            annotated_bgr[:, :, ::-1]),   # RGB for display
        counts=dict(Counter(names)),
        density=len(names),
        vehicles=vehicles)


# ---------------- video ----------------
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


def _make_tracker(fps, frame_skip):
    """ByteTrack so each vehicle is counted once. None if unavailable."""
    try:
        return sv.ByteTrack(frame_rate=max(1, int(round(fps / frame_skip))))
    except Exception:
        return None


def process_video(video_bytes, suffix, model, conf, imgsz, frame_skip,
                  max_seconds, progress_cb, collect_crops=False):
    """Detect and track vehicles in a video, frame by frame (streaming)."""
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

        tracker = _make_tracker(fps, frame_skip)
        keeper = crops.CropKeeper() if (collect_crops and tracker) else None

        track_votes = defaultdict(Counter)   # track id -> class name votes
        max_counts = Counter()               # fallback when there is no tracker
        peak_vehicles = 0                    # most vehicles in a single frame

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
                if tracker:
                    ids = [int(t) for t in detections.tracker_id]
                    labels = [f"#{tid} {name}" for tid,
                              name in zip(ids, names)]
                    for tid, name, box in zip(ids, names, detections.xyxy):
                        track_votes[tid][name] += 1
                        if keeper:
                            keeper.add(tid, frame, box)

                write(annotate_frame(frame, detections, labels))
                peak_vehicles = max(peak_vehicles, len(names))
                for name, n in Counter(names).items():
                    max_counts[name] = max(max_counts[name], n)
                processed += 1

            idx += 1
            if limit and idx % 5 == 0:
                progress_cb(min(idx / limit, 1.0))

        close_writer()
        close_writer = None
        with open(out_path, "rb") as f:
            video_bytes_out = f.read()

        counts, vehicles = dict(max_counts), []
        if tracker:
            counts = Counter()
            display_id = 1
            for tid, votes in track_votes.items():
                if sum(votes.values()) >= cfg.MIN_TRACK_FRAMES:
                    cls = votes.most_common(1)[0][0]
                    counts[cls] += 1
                    vehicles.append(VehicleCrops(
                        display_id, cls, keeper.crops(tid) if keeper else []))
                    display_id += 1
            counts = dict(counts)

        return DetectionResult(
            kind="video", media=video_bytes_out, counts=counts,
            density=peak_vehicles, vehicles=vehicles, frames=processed,
            h264=is_h264, tracked=bool(tracker))
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
