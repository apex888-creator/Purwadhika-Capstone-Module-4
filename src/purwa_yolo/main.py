"""Electronic Road Pricing app: Streamlit entry point.

Run from the project root:  python -m streamlit run src/purwa_yolo/main.py

Modules (all in this folder):
    app_config.py    constants
    device.py        GPU/CPU selection + diagnostics
    plate_rules.py   plate parsing, masking and the odd-even rule (pure)
    plate_ocr.py     fast-alpr plate reading
    detector.py      YOLO detection, tracking, image / video pipelines
    pricing.py       ERP price calculation (pure)
    ui_components.py sidebar settings and result rendering
    gate.py          human check in front of the app
"""
import time
from io import BytesIO
from pathlib import Path

import streamlit as st
from PIL import Image

from purwa_yolo import app_config as cfg
from purwa_yolo import detector
from purwa_yolo import describe_device
from purwa_yolo import human_check
from purwa_yolo import erp_settings, render_erp, show_counts

st.set_page_config(page_title="Object Detection", page_icon="🎯")
human_check()   # front gate: the app below only runs after this is passed

st.title("🎯 Electronic Road Pricing")

# ---------------- model selection ----------------
use_cases = list(cfg.MODEL_FILES.keys())
default_index = next(
    (i for i, name in enumerate(use_cases) if detector.resolve_model_path(name)), 0)

selected_model = st.selectbox(
    "Select Usecase",
    use_cases,
    index=default_index,
    format_func=lambda n: n if detector.resolve_model_path(
        n) else f"{n} (model not available)",
)

model_path = detector.resolve_model_path(selected_model)
if model_path is None:
    st.warning(
        f"Model file for **{selected_model}** not found. "
        f"Place `{cfg.MODEL_FILES[selected_model]}` in the `models/` folder."
    )
    st.stop()

with st.spinner(f"Loading {selected_model} model..."):
    model = detector.load_model(str(model_path))

st.success(
    f"✅ {selected_model} model loaded! (running on {describe_device()})")

# ---------------- settings ----------------
# Rate settings only apply to the Vehicle use case
settings = erp_settings() if selected_model == "Vehicle" else None
plate_on = bool(settings and settings.plate_on)

conf = st.slider("Confidence threshold", min_value=0.01,
                 max_value=0.95, value=0.10, step=0.01)

imgsz = st.select_slider(
    "Inference size (px)", options=cfg.INFER_SIZES, value=cfg.DEFAULT_IMGSZ,
    help="Smaller is faster; larger can detect small objects better but "
         "is slower. Try 480 or 640 if processing feels slow.")

# ---------------- upload ----------------
uploaded_file = st.file_uploader(
    "Upload Image / Video",
    accept_multiple_files=False,
    type=cfg.IMAGE_TYPES + cfg.VIDEO_TYPES,
    help=f"Images up to {cfg.MAX_IMAGE_MB} MB, "
    f"videos up to {cfg.MAX_VIDEO_MB} MB.",
)

if uploaded_file is not None:
    ext = Path(uploaded_file.name).suffix.lower().lstrip(".")
    is_video = ext in cfg.VIDEO_TYPES
    size_mb = uploaded_file.size / (1024 * 1024)
    limit_mb = cfg.MAX_VIDEO_MB if is_video else cfg.MAX_IMAGE_MB

    if size_mb > limit_mb:
        st.error(
            f"File size {size_mb:.1f} MB exceeds the {limit_mb} MB limit "
            f"for {'videos' if is_video else 'images'}.")
        st.stop()

    frame_skip = max_seconds = None
    if is_video:
        st.video(uploaded_file)
        c1, c2 = st.columns(2)
        frame_skip = c1.slider(
            "Process every N frames", 1, 5, 2,
            help="Higher is faster, but vehicle tracking is less accurate. "
                 "Use 1 or 2 when counting vehicles.")
        max_seconds = c2.slider("Maximum duration to process (seconds)", 5, 60,
                                20, help="Limits processing time and memory.")
    else:
        st.image(uploaded_file, caption="Uploaded Image",
                 use_container_width=True)

    # Rate settings are NOT part of params: changing them (currency, tariffs,
    # date, surcharge...) does not need a new detection run, the charge is
    # recalculated from the stored result. plate_on IS part of it, because it
    # decides whether OCR runs during detection.
    params = (uploaded_file.name, uploaded_file.size,
              selected_model, conf, imgsz, plate_on, frame_skip, max_seconds)

    # ---------------- run detection ----------------
    if st.button("🔍 Detect Objects", type="primary"):
        data = uploaded_file.getvalue()
        t0 = time.perf_counter()
        try:
            if is_video:
                bar = st.progress(0.0, text="Processing video...")
                (video_bytes, counts, n_frames, is_h264, tracked,
                 max_simul, plates) = detector.process_video(
                    data, f".{ext}", model, conf, imgsz, frame_skip,
                    max_seconds,
                    lambda p: bar.progress(p, text="Processing video..."),
                    read_plates=plate_on)
                bar.empty()
                st.session_state["result"] = {
                    "params": params, "kind": "video", "data": video_bytes,
                    "counts": counts, "frames": n_frames, "h264": is_h264,
                    "tracked": tracked, "density": max_simul,
                    "plates": plates,
                    "elapsed": time.perf_counter() - t0}
            else:
                with st.spinner("Detecting objects and reading plates..."
                                if plate_on else "Detecting objects..."):
                    img, counts, plates = detector.process_image(
                        data, model, conf, imgsz, read_plates=plate_on)
                st.session_state["result"] = {
                    "params": params, "kind": "image", "data": img,
                    "counts": counts, "density": sum(counts.values()),
                    "plates": plates,
                    "elapsed": time.perf_counter() - t0}
        except Exception as e:
            st.session_state.pop("result", None)
            st.error(f"Failed to process file: {e}")

    # ---------------- show results ----------------
    # Read from session_state so the result does not disappear when a widget
    # changes or a download button is pressed (Streamlit reruns the script).
    res = st.session_state.get("result")
    if res and res["params"] == params:
        st.subheader("Detection Results")
        is_vehicle = selected_model == "Vehicle"
        st.caption(f"Processing time: {res['elapsed']:.1f} s")

        if res["kind"] == "image":
            st.image(res["data"], caption="Detected Objects",
                     use_container_width=True)
            buf = BytesIO()
            Image.fromarray(res["data"]).save(buf, format="PNG")
            st.download_button("⬇️ Download detection result",
                               data=buf.getvalue(),
                               file_name="detection_result.png",
                               mime="image/png")
            if is_vehicle:
                render_erp(res["counts"], res["density"], "in the image",
                           settings, plates=res.get("plates"))
                st.caption("Each vehicle detected in the image "
                           "is counted once.")
            else:
                show_counts(res["counts"], "📊 Object Counts", "Total objects")
        else:
            if not res["h264"]:
                st.warning(
                    "H.264 encoder is not available, so the video may not "
                    "play in the browser. Use the download button, or "
                    "install: `python -m pip install imageio imageio-ffmpeg`.")
            st.video(res["data"])
            st.download_button("⬇️ Download detection result",
                               data=res["data"],
                               file_name="detection_result.mp4",
                               mime="video/mp4")
            st.caption(f"{res['frames']} frames processed.")

            if res["tracked"]:
                note = ("Counted as unique vehicles via tracking "
                        f"(must be tracked for at least {cfg.MIN_TRACK_FRAMES} "
                        "frames). This is an estimate, not an exact count.")
            else:
                note = ("⚠️ Tracking is unavailable, so the figures shown are "
                        "the maximum count per class in a single frame.")

            if is_vehicle:
                render_erp(res["counts"], res["density"],
                           "at most in a single frame", settings,
                           plates=res.get("plates"))
                st.caption(note)
            else:
                show_counts(res["counts"], "📊 Object Counts", "Total objects")
                st.caption(note)
