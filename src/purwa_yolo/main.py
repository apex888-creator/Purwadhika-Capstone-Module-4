"""Page layout and flow (called from streamlit_app.py).

Upload  ->  1) YOLO object detection [button]  ->  2) plate OCR [button]

Section 1 shows the bounding boxes and the base ERP charge. Section 2 reads the
plates of the detected vehicles and shows the final charge with the odd-even
rule. Each section keeps its result in st.session_state.
"""
import time
import uuid
from io import BytesIO
from pathlib import Path

import streamlit as st
from PIL import Image

from purwa_yolo import app_config as cfg
from purwa_yolo.core.data import OcrResult
from purwa_yolo.core.pricing import compute_erp
from purwa_yolo.device import describe_device
from purwa_yolo.ui import results as view
from purwa_yolo.ui.gate import human_check
from purwa_yolo.ui.sidebar import erp_settings
from purwa_yolo.vision import detector, pipelines, plate_ocr

DETECTION_KEY, OCR_KEY = "detection", "ocr"     # st.session_state keys


# ---------------- inputs ----------------
def select_model():
    """Immediately load the YOLO Model for Vehicle Dataset."""
    use_case = "Vehicle"

    path = detector.resolve_model_path(use_case)
    if path is None:
        st.warning(f"Model file for **{use_case}** not found. "
                   f"Place `{cfg.MODEL_FILES[use_case]}` in the `models/` folder.")
        st.stop()

    with st.spinner(f"Loading ERP model..."):
        model = detector.load_model(str(path))
    st.success(f"✅ ERP System loaded! (running on {describe_device()})")
    return use_case, model


def pick_upload():
    """File uploader with a size check. Return (file, extension, is_video) or None."""
    file = st.file_uploader(
        "Upload Image / Video", accept_multiple_files=False,
        type=cfg.IMAGE_TYPES + cfg.VIDEO_TYPES,
        help=f"Images up to {cfg.MAX_IMAGE_MB} MB, "
        f"videos up to {cfg.MAX_VIDEO_MB} MB.")
    if file is None:
        return None

    ext = Path(file.name).suffix.lower().lstrip(".")
    is_video = ext in cfg.VIDEO_TYPES
    size_mb = file.size / (1024 * 1024)
    limit_mb = cfg.MAX_VIDEO_MB if is_video else cfg.MAX_IMAGE_MB
    if size_mb > limit_mb:
        st.error(f"File size {size_mb:.1f} MB exceeds the {limit_mb} MB limit "
                 f"for {'videos' if is_video else 'images'}.")
        st.stop()
    return file, ext, is_video


# ---------------- section 1: YOLO ----------------
def run_yolo(use_case, model, upload, conf, imgsz, frame_skip, max_seconds):
    """Run the detection pipeline. Return a DetectionResult."""
    file, ext, is_video = upload
    data = file.getvalue()
    collect = use_case == "Vehicle"       # crops are only needed for plate OCR
    t0 = time.perf_counter()

    if is_video:
        bar = st.progress(0.0, text="Processing video...")
        result = pipelines.process_video(
            data, f".{ext}", model, conf, imgsz, frame_skip, max_seconds,
            lambda p: bar.progress(p, text="Processing video..."),
            collect_crops=collect)
        bar.empty()
    else:
        with st.spinner("Detecting objects..."):
            result = pipelines.process_image(
                data, model, conf, imgsz, collect_crops=collect)

    result.elapsed = time.perf_counter() - t0
    return result


def show_detection(det, use_case, settings):
    """Annotated image/video with bounding boxes, plus counts or base charge."""
    st.subheader("Detection result")
    st.caption(f"Processing time: {det.elapsed:.1f} s")

    if det.kind == "image":
        st.image(det.media, caption="Detected objects (bounding boxes)",
                 use_container_width=True)
        buf = BytesIO()
        Image.fromarray(det.media).save(buf, format="PNG")
        st.download_button("⬇️ Download detection result", data=buf.getvalue(),
                           file_name="detection_result.png", mime="image/png")
        note = "Each vehicle detected in the image is counted once."
    else:
        if not det.h264:
            st.warning(
                "H.264 encoder is not available, so the video may not play in "
                "the browser. Use the download button, or reinstall "
                "imageio-ffmpeg (python -m poetry sync).")
        st.video(det.media)
        st.download_button("⬇️ Download detection result", data=det.media,
                           file_name="detection_result.mp4", mime="video/mp4")
        st.caption(f"{det.frames} frames processed.")
        note = ("Counted as unique vehicles via tracking (must be tracked for "
                f"at least {cfg.MIN_TRACK_FRAMES} frames). This is an estimate, "
                "not an exact count." if det.tracked else
                "⚠️ Tracking is unavailable, so the figures shown are the "
                "maximum count per class in a single frame.")

    if use_case == "Vehicle":
        b = compute_erp(det.counts, det.density, settings)
        view.show_erp(b, settings, det.density, det.density_basis,
                      "🚦 ERP charge (before the odd-even rule)")
    else:
        view.show_counts(det.counts, "📊 Object Counts", "Total objects")
    st.caption(note)


def detection_section(use_case, model, upload, settings):
    """Section 1. Return the current DetectionResult, or None if not run yet."""
    file, _, is_video = upload
    st.header("1️⃣ Object detection (YOLO)")

    conf = st.slider("Confidence threshold", min_value=0.01,
                     max_value=0.95, value=0.10, step=0.01)
    imgsz = st.select_slider(
        "Inference size (px)", options=cfg.INFER_SIZES, value=cfg.DEFAULT_IMGSZ,
        help="Smaller is faster; larger can detect small objects better but "
             "is slower. Try 480 or 640 if processing feels slow.")

    frame_skip = max_seconds = None
    if is_video:
        st.video(file)
        c1, c2 = st.columns(2)
        frame_skip = c1.slider(
            "Process every N frames", 1, 5, 2,
            help="Higher is faster, but vehicle tracking is less accurate. "
                 "Use 1 or 2 when counting vehicles.")
        max_seconds = c2.slider("Maximum duration to process (seconds)", 5, 60,
                                20, help="Limits processing time and memory.")
    else:
        st.image(file, caption="Uploaded image", use_container_width=True)

    # A stored result is shown only while these inputs stay the same.
    # (Rate settings are not here: changing them re-calculates the charge
    # from the stored result without running YOLO again.)
    params = (file.name, file.size, use_case, conf, imgsz,
              frame_skip, max_seconds)

    if st.button("🔍 Detect Objects", type="primary"):
        try:
            det = run_yolo(use_case, model, upload, conf, imgsz,
                           frame_skip, max_seconds)
        except Exception as e:
            st.session_state.pop(DETECTION_KEY, None)
            st.session_state.pop(OCR_KEY, None)
            st.error(f"Failed to process file: {e}")
            return None
        det.params, det.run_id = params, uuid.uuid4().hex
        st.session_state[DETECTION_KEY] = det
        st.session_state.pop(OCR_KEY, None)   # a new detection voids old OCR

    det = st.session_state.get(DETECTION_KEY)
    if det is None or det.params != params:
        return None
    show_detection(det, use_case, settings)
    return det


# ---------------- section 2: plate OCR ----------------
def run_ocr(det):
    """Read the plates of all detected vehicles. Return an OcrResult."""
    t0 = time.perf_counter()
    with st.spinner("Loading plate reader (the first run downloads the models)..."):
        alpr = plate_ocr.get_plate_reader()
    bar = st.progress(0.0, text="Reading plates...")
    plates = plate_ocr.read_plates(
        det.vehicles, alpr,
        lambda p: bar.progress(p, text="Reading plates..."))
    bar.empty()
    return OcrResult(det.run_id, plates, time.perf_counter() - t0)


def ocr_section(det, settings):
    """Section 2: plate OCR, the odd-even rule and the final charge."""
    st.header("2️⃣ License plate OCR (fast-alpr)")

    if not plate_ocr.is_available():
        st.info("Plate reading needs fast-alpr. Run `python -m poetry sync` "
                "to install the project dependencies.")
        return
    if det is None:
        st.info("Run the object detection above first.")
        return
    if not det.tracked:
        st.warning("Tracking is unavailable, so plates cannot be matched to "
                   "individual vehicles.")
        return
    if not det.vehicles:
        st.info("No vehicles were detected, so there is nothing to read.")
        return

    n_crops = sum(len(v.crops) for v in det.vehicles)
    st.caption(
        f"{len(det.vehicles)} vehicle(s) ready, {n_crops} crop(s) to read.")

    if st.button("🔎 Read Plates", type="primary"):
        try:
            st.session_state[OCR_KEY] = run_ocr(det)
        except Exception as e:
            st.session_state.pop(OCR_KEY, None)
            st.error(f"Plate reading failed: {e}")
            return

    ocr = st.session_state.get(OCR_KEY)
    if ocr is None or ocr.run_id != det.run_id:
        return

    st.subheader("OCR result")
    st.caption(f"Processing time: {ocr.elapsed:.1f} s")
    b = compute_erp(det.counts, det.density, settings, ocr.plates)
    view.show_plate_table(b, settings)
    view.show_erp(b, settings, det.density, det.density_basis,
                  "🚦 ERP charge (with the odd-even rule)")


# ---------------- page ----------------
def main():
    st.set_page_config(page_title="Object Detection", page_icon="🎯")
    human_check()   # front gate: the app below only runs after this is passed

    st.title("🎯 Electronic Road Pricing")
    use_case, model = select_model()
    settings = erp_settings() if use_case == "Vehicle" else None

    upload = pick_upload()
    if upload is None:
        return

    det = detection_section(use_case, model, upload, settings)
    if use_case == "Vehicle":
        ocr_section(det, settings)
