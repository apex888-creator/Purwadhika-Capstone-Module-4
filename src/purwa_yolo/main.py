import os
import tempfile
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import supervision as sv
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Object Detection", page_icon="🎯")

# Get project root directory (2 levels up from this file)
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# Map selection to model filename
MODEL_FILES = {
    "Construction Equipment": "best_construction.pt",
    "Vehicle": "best_vehicle.pt",
    "Fruit": "best_fruit.pt",
}

IMAGE_TYPES = ["jpg", "jpeg", "png", "webp"]
VIDEO_TYPES = ["mp4", "mov", "avi", "mkv"]
MAX_IMAGE_MB = 10      # image size limit (checked in code)
MAX_VIDEO_MB = 100     # video size limit; server.maxUploadSize must be >= this
MAX_VIDEO_SIDE = 960   # longest side of the output video (px), for speed

# Base ERP rates (USD per vehicle). Key = model class name (lowercase).
DEFAULT_RATES_USD = {"car": 2.0, "van": 3.0, "bus": 5.0}
ERP_LABELS = {"car": "Car", "van": "Van", "bus": "Bus"}

# Rate multipliers by time period (editable in the sidebar)
DEFAULT_PERIODS = {"Normal": 1.0, "Peak hours": 1.5, "Off-peak": 0.5}

# Supported currencies. Rates are stored in USD and converted for display.
# The exchange rate defaults are placeholders: enter the current rate manually.
CURRENCIES = {
    "USD": {"symbol": "$", "default_rate": 1.0, "decimals": 2, "step": 0.01},
    "IDR": {"symbol": "Rp ", "default_rate": 16000.0, "decimals": 0, "step": 100.0},
    "CNY": {"symbol": "CN¥ ", "default_rate": 7.2, "decimals": 2, "step": 0.05},
}

# Video only: a vehicle is counted only if it is tracked for at least this
# many processed frames (filters out momentary detections / short-lived IDs)
MIN_TRACK_FRAMES = 3


def resolve_model_path(use_case: str):
    """Return model path if the file exists, otherwise None."""
    path = MODELS_DIR / MODEL_FILES[use_case]
    if path.exists():
        return path
    # Fallback: the vehicle model may still be named best.pt
    if use_case == "Vehicle" and (MODELS_DIR / "best.pt").exists():
        return MODELS_DIR / "best.pt"
    return None


# Cache model loading - models are loaded only once per file
@st.cache_resource
def load_model(model_path: str):
    """Load YOLO model with caching for performance"""
    return YOLO(model_path)


# Cache annotators - reuse same annotator objects
@st.cache_resource
def get_annotators():
    """Get cached annotator objects"""
    return sv.BoxAnnotator(), sv.LabelAnnotator()


def detect_frame(frame_bgr, model, conf, tracker=None):
    """Run detection on one frame (BGR, OpenCV format).
    Return (annotated BGR frame, list of class names, detections)."""
    results = model(frame_bgr, conf=conf, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results).with_nms()
    if tracker is not None:
        detections = tracker.update_with_detections(detections)

    names = [str(n) for n in detections.data.get("class_name", [])]

    box_annotator, label_annotator = get_annotators()
    annotated = frame_bgr.copy()
    if len(detections) > 0:
        annotated = box_annotator.annotate(
            scene=annotated, detections=detections)
        if tracker is not None:
            labels = [f"#{int(t)} {n}"
                      for t, n in zip(detections.tracker_id, names)]
            annotated = label_annotator.annotate(
                scene=annotated, detections=detections, labels=labels)
        else:
            annotated = label_annotator.annotate(
                scene=annotated, detections=detections)
    return annotated, names, detections


def process_image(image_bytes, model, conf):
    """Image pipeline. Return (annotated RGB image, dict of counts per class)."""
    pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    pil_image.thumbnail((1920, 1920))   # saves memory, keeps aspect ratio
    frame_bgr = np.ascontiguousarray(np.array(pil_image)[:, :, ::-1])

    annotated_bgr, names, _ = detect_frame(frame_bgr, model, conf)
    annotated_rgb = np.ascontiguousarray(annotated_bgr[:, :, ::-1])
    return annotated_rgb, dict(Counter(names))


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


def process_video(video_bytes, suffix, model, conf, frame_skip, max_seconds,
                  progress_cb):
    """Detect + track a video frame by frame (streaming).
    Return (mp4 bytes, dict of UNIQUE vehicle counts per class, number of
    processed frames, is_h264, tracked, most vehicles seen in a single frame)."""
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

        scale = min(1.0, MAX_VIDEO_SIDE / max(w, h))
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
                annotated, names, detections = detect_frame(
                    frame, model, conf, tracker)
                write(annotated)

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

        if tracker is not None:
            unique = Counter()
            for votes in track_votes.values():
                if sum(votes.values()) >= MIN_TRACK_FRAMES:
                    unique[votes.most_common(1)[0][0]] += 1
            counts = dict(unique)
        else:
            counts = dict(max_counts)

        return (result_bytes, counts, processed, is_h264,
                tracker is not None, max_simultaneous)
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


def show_counts(counts, title, total_label):
    if counts:
        st.subheader(title)
        st.metric(label=total_label, value=sum(counts.values()))
        cols = st.columns(len(counts))
        for col, (name, n) in zip(cols, counts.items()):
            col.metric(label=name, value=n)
    else:
        st.info("No objects detected. Try lowering the confidence threshold.")


# ---------------- Dynamic ERP pricing ----------------

def erp_settings():
    """ERP rate settings in the sidebar. Return a configuration dict."""
    with st.sidebar:
        st.header("⚙️ ERP Rate Settings")

        currency = st.radio("Currency", list(CURRENCIES.keys()),
                            horizontal=True)
        cur = CURRENCIES[currency]
        fx_rate = cur["default_rate"]
        if currency != "USD":
            fx_rate = st.number_input(
                f"Exchange rate ({currency} per 1 USD), enter manually",
                min_value=0.0001, value=cur["default_rate"],
                step=cur["step"], format="%.4f")

        st.subheader("Base rates (USD)")
        base_rates = {}
        for key, default in DEFAULT_RATES_USD.items():
            base_rates[key] = st.number_input(
                f"{ERP_LABELS[key]} (USD)", min_value=0.0,
                value=default, step=0.5)

        st.subheader("Time period")
        with st.expander("Adjust period multipliers"):
            period_mults = {}
            for name, default in DEFAULT_PERIODS.items():
                period_mults[name] = st.number_input(
                    f"{name} multiplier", min_value=0.0,
                    value=default, step=0.1)
        period = st.selectbox("Current period", list(period_mults.keys()))

        st.subheader("Traffic density")
        density_on = st.checkbox("Raise rates when busy", value=True)
        busy_min, congested_min = 5, 10
        mult_busy, mult_congested = 1.25, 1.5
        if density_on:
            busy_min = st.number_input("Busy starts at (vehicles)",
                                       min_value=1, value=5, step=1)
            congested_min = st.number_input("Congested starts at (vehicles)",
                                            min_value=1, value=10, step=1)
            mult_busy = st.number_input("Busy multiplier", min_value=1.0,
                                        value=1.25, step=0.05)
            mult_congested = st.number_input("Congested multiplier",
                                             min_value=1.0, value=1.5,
                                             step=0.05)

        max_mult = st.number_input(
            "Maximum total multiplier", min_value=1.0,
            value=3.0, step=0.5,
            help="Period multiplier × density multiplier will not "
                 "exceed this value.")

    return {
        "currency": currency, "fx_rate": fx_rate,
        "base_rates": base_rates,
        "period_mults": period_mults, "period": period,
        "density_on": density_on,
        "busy_min": int(busy_min),
        "congested_min": max(int(congested_min), int(busy_min)),
        "mult_busy": mult_busy, "mult_congested": mult_congested,
        "max_mult": max_mult,
    }


def fmt_money(usd, cfg):
    """Format a USD amount in the selected currency."""
    cur = CURRENCIES[cfg["currency"]]
    value = usd * cfg["fx_rate"]
    return f"{cur['symbol']}{value:,.{cur['decimals']}f}"


def density_level(n_vehicles, cfg):
    """Return (density level name, multiplier)."""
    if not cfg["density_on"]:
        return "Disabled", 1.0
    if n_vehicles >= cfg["congested_min"]:
        return "Congested", cfg["mult_congested"]
    if n_vehicles >= cfg["busy_min"]:
        return "Busy", cfg["mult_busy"]
    return "Free-flowing", 1.0


def show_erp(counts, density, density_basis, cfg):
    """Show total vehicles and the ERP charge breakdown with dynamic pricing.
    final rate = base rate x period multiplier x density multiplier
    (capped at max_mult)."""
    norm = Counter()
    for name, n in counts.items():
        norm[name.lower()] += n

    period_mult = cfg["period_mults"][cfg["period"]]
    level, dens_mult = density_level(density, cfg)
    raw_mult = period_mult * dens_mult
    total_mult = min(raw_mult, cfg["max_mult"])

    rows = []
    total_vehicles, total_fee, total_static = 0, 0.0, 0.0
    for key, base in cfg["base_rates"].items():
        n = norm.get(key, 0)
        final_rate = base * total_mult
        rows.append({
            "Type": ERP_LABELS[key],
            "Count": n,
            "Base rate": fmt_money(base, cfg),
            "Final rate": fmt_money(final_rate, cfg),
            "Subtotal": fmt_money(n * final_rate, cfg),
        })
        total_vehicles += n
        total_fee += n * final_rate
        total_static += n * base
    rows.append({
        "Type": "TOTAL", "Count": total_vehicles,
        "Base rate": "", "Final rate": "",
        "Subtotal": fmt_money(total_fee, cfg),
    })

    st.subheader("🚦 ERP Calculation (dynamic pricing)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total vehicles", total_vehicles)
    diff = total_fee - total_static
    if abs(diff) > 1e-9:
        sign = "+" if diff > 0 else "-"
        c2.metric("Total charge", fmt_money(total_fee, cfg),
                  delta=f"{sign}{fmt_money(abs(diff), cfg)} vs base rate",
                  delta_color="inverse")
    else:
        c2.metric("Total charge", fmt_money(total_fee, cfg))
    c3.metric("Rate multiplier", f"×{total_mult:.2f}")

    st.table(pd.DataFrame(rows).set_index("Type"))

    st.caption(
        f"Period: **{cfg['period']}** (×{period_mult:.2f}) · "
        f"Density: **{level}** (×{dens_mult:.2f}; {density} vehicles "
        f"{density_basis}) · Total multiplier ×{total_mult:.2f}")
    if raw_mult > cfg["max_mult"]:
        st.caption(
            f"Multiplier capped from ×{raw_mult:.2f} to "
            f"×{cfg['max_mult']:.2f} (maximum limit).")

    other = {k: v for k, v in norm.items() if k not in cfg["base_rates"]}
    if other:
        st.caption("Classes without a rate (not counted): "
                   + ", ".join(f"{k} ({v})" for k, v in other.items()))
    if total_vehicles == 0:
        st.info("No vehicles detected. "
                "Try lowering the confidence threshold.")


# --- Main Streamlit section ---

st.title("🎯 Object Detection")

# Model selection (default: first model whose file exists)
use_cases = list(MODEL_FILES.keys())
default_index = next(
    (i for i, name in enumerate(use_cases) if resolve_model_path(name)), 0)

selected_model = st.selectbox(
    "Select Usecase",
    use_cases,
    index=default_index,
    format_func=lambda n: n if resolve_model_path(
        n) else f"{n} (model not available)",
)

model_path = resolve_model_path(selected_model)
if model_path is None:
    st.warning(
        f"Model file for **{selected_model}** not found. "
        f"Place `{MODEL_FILES[selected_model]}` in the `models/` folder."
    )
    st.stop()

with st.spinner(f"Loading {selected_model} model..."):
    model = load_model(str(model_path))

st.success(f"✅ {selected_model} model loaded!")

# Rate settings only apply to the Vehicle use case
erp_cfg = erp_settings() if selected_model == "Vehicle" else None

conf = st.slider("Confidence threshold", min_value=0.05,
                 max_value=0.95, value=0.25, step=0.05)

uploaded_file = st.file_uploader(
    "Upload Image / Video",
    accept_multiple_files=False,
    type=IMAGE_TYPES + VIDEO_TYPES,
    help=f"Images up to {MAX_IMAGE_MB} MB, videos up to {MAX_VIDEO_MB} MB.",
)

if uploaded_file is not None:
    ext = Path(uploaded_file.name).suffix.lower().lstrip(".")
    is_video = ext in VIDEO_TYPES
    size_mb = uploaded_file.size / (1024 * 1024)
    limit_mb = MAX_VIDEO_MB if is_video else MAX_IMAGE_MB

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

    # Rate settings are NOT part of params: changing them does not require
    # re-running detection, the charge is recalculated from the stored result.
    params = (uploaded_file.name, uploaded_file.size,
              selected_model, conf, frame_skip, max_seconds)

    if st.button("🔍 Detect Objects", type="primary"):
        data = uploaded_file.getvalue()
        try:
            if is_video:
                bar = st.progress(0.0, text="Processing video...")
                (video_bytes, counts, n_frames, is_h264, tracked,
                 max_simul) = process_video(
                    data, f".{ext}", model, conf, frame_skip, max_seconds,
                    lambda p: bar.progress(p, text="Processing video..."))
                bar.empty()
                st.session_state["result"] = {
                    "params": params, "kind": "video", "data": video_bytes,
                    "counts": counts, "frames": n_frames, "h264": is_h264,
                    "tracked": tracked, "density": max_simul}
            else:
                with st.spinner("Detecting objects..."):
                    img, counts = process_image(data, model, conf)
                st.session_state["result"] = {
                    "params": params, "kind": "image", "data": img,
                    "counts": counts, "density": sum(counts.values())}
        except Exception as e:
            st.session_state.pop("result", None)
            st.error(f"Failed to process file: {e}")

    # Show the result from session_state so it does not disappear when the
    # download button is pressed (Streamlit reruns the script).
    res = st.session_state.get("result")
    if res and res["params"] == params:
        st.subheader("Detection Results")
        is_vehicle = selected_model == "Vehicle"

        if res["kind"] == "image":
            st.image(res["data"], caption="Detected Objects",
                     use_container_width=True)
            buf = BytesIO()
            Image.fromarray(res["data"]).save(buf, format="PNG")
            st.download_button("⬇️ Download result", data=buf.getvalue(),
                               file_name="detection_result.png",
                               mime="image/png")
            if is_vehicle:
                show_erp(res["counts"], res["density"],
                         "in the image", erp_cfg)
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
            st.download_button("⬇️ Download result video", data=res["data"],
                               file_name="detection_result.mp4",
                               mime="video/mp4")
            st.caption(f"{res['frames']} frames processed.")

            if res["tracked"]:
                note = ("Counted as unique vehicles via tracking "
                        f"(must be tracked for at least {MIN_TRACK_FRAMES} "
                        "frames). This is an estimate, not an exact count.")
            else:
                note = ("⚠️ Tracking is unavailable, so the figures shown are "
                        "the maximum count per class in a single frame.")

            if is_vehicle:
                show_erp(res["counts"], res["density"],
                         "at most in a single frame", erp_cfg)
                st.caption(note)
            else:
                show_counts(res["counts"], "📊 Object Counts", "Total objects")
                st.caption(note)
