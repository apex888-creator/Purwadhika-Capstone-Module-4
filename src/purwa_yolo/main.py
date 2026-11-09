import numpy as np
import streamlit as st
import supervision as sv
from ultralytics import YOLO
from PIL import Image
from io import BytesIO
from pathlib import Path
from collections import Counter

# Get project root directory (2 levels up from this file)
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# Cache model loading - models are loaded only once per type


@st.cache_resource
def load_model(model_name: str):
    """Load YOLO model with caching for performance"""
    model_path = MODELS_DIR / f"best_{model_name}.pt"

    # Fallback: kalau best_vehicle.pt tidak ada, coba best.pt
    if not model_path.exists() and model_name == "vehicle":
        model_path = MODELS_DIR / "best.pt"

    if not model_path.exists():
        st.error(f"File model tidak ditemukan: {model_path}")
        st.stop()

    return YOLO(str(model_path))

# Cache annotators - reuse same annotator objects


@st.cache_resource
def get_annotators():
    """Get cached annotator objects"""
    return sv.BoxAnnotator(), sv.LabelAnnotator()


def detector_pipeline_pillow(image_bytes, model):
    """Optimized detection pipeline"""
    # Load and convert image
    pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image_np_rgb = np.array(pil_image)

    # Run inference
    results = model(image_np_rgb, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results).with_nms()

    # Get cached annotators
    box_annotator, label_annotator = get_annotators()

    # Annotate image
    annotated_image = pil_image.copy()
    annotated_image = box_annotator.annotate(
        scene=annotated_image, detections=detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections)
    annotated_image_np = np.asarray(annotated_image)

    # Optimized class counting using Counter
    class_names = detections.data.get("class_name", [])
    classcounts = dict(Counter(class_names))

    return annotated_image_np, classcounts


# --- Bagian Streamlit Utama ---

st.title("🎯 Object Detection")

# Model selection
selected_model = st.selectbox(
    "Select Usecase", ("Construction Equipment", "Vehicle", "Fruit"))

# Map selection to model filename
model_map = {
    "Construction Equipment": "construction",
    "Vehicle": "vehicle",
    "Fruit": "fruit"
}

# Load model with caching (only loads once per model type)
with st.spinner(f"Loading {selected_model} model..."):
    model = load_model(model_map[selected_model])

st.success(f"✅ {selected_model} model loaded!")

# File upload
uploaded_file = st.file_uploader(
    "Upload Image", accept_multiple_files=False, type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # Detect button
    if st.button("🔍 Detect Objects", type="primary"):
        bytes_data = uploaded_file.getvalue()

        # Run detection with spinner
        with st.spinner("Detecting objects..."):
            annotated_image_rgb, classcounts = detector_pipeline_pillow(
                bytes_data, model)

        # Display results
        st.subheader("Detection Results")
        st.image(annotated_image_rgb, caption="Detected Objects",
                 use_container_width=True)

        # Display class counts in a nice format
        if classcounts:
            st.subheader("📊 Object Counts")
            col1, col2 = st.columns([1, 2])
            with col1:
                for class_name, count in classcounts.items():
                    st.metric(label=class_name, value=count)
        else:
            st.info("No objects detected in the image.")
