"""Page layout and flow (called from streamlit_app.py).

Upload  ->  1) YOLO object detection [button]  ->  2) plate OCR [button]

Section 1 shows the bounding boxes and the base ERP charge. Section 2 reads the
plates of the detected vehicles and shows the final charge with the odd-even
rule. Each section keeps its result in st.session_state.

All visible text goes through ui/i18n.py (English / Simplified Chinese).
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
from purwa_yolo.ui import theme
from purwa_yolo.ui.gate import human_check
from purwa_yolo.ui.i18n import t, tr
from purwa_yolo.ui.sidebar import erp_settings
from purwa_yolo.vision import detector, pipelines, plate_ocr

DETECTION_KEY, OCR_KEY = "detection", "ocr"     # st.session_state keys
OCR_CROPS_KEY = "ocr_crops"

# ---------------- inputs ----------------


def select_model():
    """Immediately load the YOLO Model for Vehicle Dataset."""
    use_case = "Vehicle"

    path = detector.resolve_model_path(use_case)
    if path is None:
        st.warning(t("model_missing", uc=use_case,
                     f=cfg.MODEL_FILES[use_case]))
        st.stop()

    with st.spinner(t("loading_model")):
        model = detector.load_model(str(path))
    theme.status(t("model_ready", dev=describe_device()))
    return use_case, model


def pick_upload():
    """File uploader with a size check. Return (file, extension, is_video) or None."""
    file = st.file_uploader(
        t("upload_label"), accept_multiple_files=False,
        type=cfg.IMAGE_TYPES + cfg.VIDEO_TYPES, key="upload",
        help=t("upload_help", img=cfg.MAX_IMAGE_MB, vid=cfg.MAX_VIDEO_MB))
    if file is None:
        return None

    ext = Path(file.name).suffix.lower().lstrip(".")
    is_video = ext in cfg.VIDEO_TYPES
    size_mb = file.size / (1024 * 1024)
    limit_mb = cfg.MAX_VIDEO_MB if is_video else cfg.MAX_IMAGE_MB
    if size_mb > limit_mb:
        st.error(t("size_err", size=size_mb, limit=limit_mb,
                   kind=t("kind_videos" if is_video else "kind_images")))
        st.stop()
    return file, ext, is_video


# ---------------- section 1: YOLO ----------------
def run_yolo(use_case, model, upload, conf, imgsz, frame_skip, max_seconds):
    """Run the detection pipeline. Return a DetectionResult."""
    file, ext, is_video = upload
    data = file.getvalue()
    collect = use_case == "Vehicle"
    t0 = time.perf_counter()

    if is_video:
        label = t("processing_video")
        bar = st.progress(0.0, text=label)
        result = pipelines.process_video(
            data, f".{ext}", model, conf, imgsz, frame_skip, max_seconds,
            lambda p: bar.progress(p, text=label),
            collect_crops=collect)
        bar.empty()
    else:
        with st.spinner(t("detecting")):
            result = pipelines.process_image(
                data, model, conf, imgsz, collect_crops=collect)

    result.elapsed = time.perf_counter() - t0
    return result


def show_detection(det, use_case, settings):
    """Annotated image/video with bounding boxes, plus counts or base charge."""
    st.subheader(t("det_result"))
    st.caption(t("proc_time", s=det.elapsed))

    if det.kind == "image":
        st.image(det.media, caption=t("det_caption"), width="stretch")
        buf = BytesIO()
        Image.fromarray(det.media).save(buf, format="PNG")
        st.download_button(t("download"), data=buf.getvalue(),
                           file_name="detection_result.png", mime="image/png")
        note = t("note_image")
    else:
        if not det.h264:
            st.warning(t("h264_warn"))
        st.video(det.media)
        st.download_button(t("download"), data=det.media,
                           file_name="detection_result.mp4", mime="video/mp4")
        st.caption(t("frames_done", n=det.frames))
        note = (t("note_tracked", n=cfg.MIN_TRACK_FRAMES) if det.tracked
                else t("note_untracked"))

    if use_case == "Vehicle":
        b = compute_erp(det.counts, det.density, settings)
        view.show_erp(b, settings, det.density, det.density_basis,
                      t("erp_before"))
    else:
        view.show_counts(det.counts, t("counts_title"), t("counts_total"))
    st.caption(note)


def detection_section(use_case, model, upload, settings):
    """Section 1. Return the current DetectionResult, or None if not run yet."""
    file, _, is_video = upload
    theme.step_header(1, t("step1_title"), t("step1_desc"))

    with st.container(border=True):
        a, b = st.columns(2)
        conf = a.slider(t("conf"), min_value=0.01, max_value=0.95,
                        value=0.10, step=0.01, key="conf")
        imgsz = b.select_slider(
            t("imgsz"), options=cfg.INFER_SIZES, value=cfg.DEFAULT_IMGSZ,
            help=t("imgsz_help"), key="imgsz")

    frame_skip = max_seconds = None
    if is_video:
        st.video(file)
        c1, c2 = st.columns(2)
        frame_skip = c1.slider(t("frame_skip"), 1, 5, 2,
                               help=t("frame_skip_help"), key="frame_skip")
        max_seconds = c2.slider(t("max_sec"), 5, 60, 20,
                                help=t("max_sec_help"), key="max_sec")
    else:
        st.image(file, caption=t("uploaded_img"), width="stretch")

    params = (file.name, file.size, use_case, conf, imgsz,
              frame_skip, max_seconds)

    if st.button(t("btn_detect"), type="primary"):
        try:
            det = run_yolo(use_case, model, upload, conf, imgsz,
                           frame_skip, max_seconds)
        except Exception as e:
            st.session_state.pop(DETECTION_KEY, None)
            st.session_state.pop(OCR_KEY, None)
            st.error(t("fail_process", e=e))
            return None
        det.params, det.run_id = params, uuid.uuid4().hex
        st.session_state[DETECTION_KEY] = det
        st.session_state.pop(OCR_KEY, None)

    det = st.session_state.get(DETECTION_KEY)
    if det is None or det.params != params:
        return None
    show_detection(det, use_case, settings)
    return det


# ---------------- section 2: plate OCR ----------------
def run_ocr(det):
    """Read the plates of all detected vehicles. Return an OcrResult."""
    t0 = time.perf_counter()
    with st.spinner(t("loading_reader")):
        alpr = plate_ocr.get_plate_reader()
    label = t("reading")
    bar = st.progress(0.0, text=label)
    plates, plate_crops = plate_ocr.read_plates(
        det.vehicles, alpr,
        lambda p: bar.progress(p, text=label))
    bar.empty()
    st.session_state[OCR_CROPS_KEY] = {
        "run_id": det.run_id, "crops": plate_crops}
    return OcrResult(det.run_id, plates, time.perf_counter() - t0)


def ocr_section(det, settings):
    """Section 2: plate OCR, the odd-even rule and the final charge."""
    theme.step_header(2, t("step2_title"), t("step2_desc"))

    if not plate_ocr.is_available():
        st.info(t("need_alpr"))
        return
    if det is None:
        st.info(t("need_detect"))
        return
    if not det.tracked:
        st.warning(t("no_tracking"))
        return
    if not det.vehicles:
        st.info(t("no_vehicles"))
        return

    n_crops = sum(len(v.crops) for v in det.vehicles)
    st.caption(t("ready_crops", v=len(det.vehicles), c=n_crops))

    if st.button(t("btn_read"), type="primary"):
        try:
            st.session_state[OCR_KEY] = run_ocr(det)
        except Exception as e:
            st.session_state.pop(OCR_KEY, None)
            st.error(t("ocr_fail", e=e))
            return

    ocr = st.session_state.get(OCR_KEY)
    if ocr is None or ocr.run_id != det.run_id:
        return

    st.subheader(t("ocr_result"))
    st.caption(t("proc_time", s=ocr.elapsed))
    show_plate_debug(det, ocr)
    b = compute_erp(det.counts, det.density, settings, ocr.plates)
    view.show_plate_table(b, settings)
    view.show_erp(b, settings, det.density, det.density_basis,
                  t("erp_after"))


def show_plate_debug(det, ocr):
    """Per vehicle: the crop sent to the OCR and the plate text that was read."""
    saved = st.session_state.get(OCR_CROPS_KEY)
    if not saved or saved["run_id"] != det.run_id:
        return
    st.subheader(t("crops_title"))
    cols_per_row = 3
    items = list(zip(ocr.plates, saved["crops"]))
    for start in range(0, len(items), cols_per_row):
        cols = st.columns(cols_per_row)
        for col, (reading, crop) in zip(cols, items[start:start + cols_per_row]):
            with col:
                if crop is not None and crop.size > 0:
                    st.image(crop[:, :, ::-1], width="stretch")
                    st.caption(
                        t("crop_size", w=crop.shape[1], h=crop.shape[0]))
                else:
                    st.caption(t("no_crop"))
                text = reading.text or t("not_readable")
                st.markdown(f"**#{reading.vehicle_id} {tr(reading.cls)}**")
                st.code(text, language=None)
                conf = getattr(reading, "confidence",
                               getattr(reading, "conf", None))
                if conf:
                    st.caption(t("confidence", c=conf))

# ---------------- page ----------------


def main():
    st.set_page_config(page_title="Electronic Road Pricing", page_icon="🚦",
                       layout="wide")
    theme.inject_css()
    human_check()

    theme.hero(t("page_title"), t("hero_sub"),
               flow=(t("flow1"), t("flow2"), t("flow3")))
    use_case, model = select_model()
    settings = erp_settings() if use_case == "Vehicle" else None

    upload = pick_upload()
    if upload is None:
        return

    det = detection_section(use_case, model, upload, settings)
    if use_case == "Vehicle":
        ocr_section(det, settings)
