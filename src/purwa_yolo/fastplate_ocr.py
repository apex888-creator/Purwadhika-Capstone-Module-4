import cv2
import numpy as np
import streamlit as st
import torch

# Tentukan device otomatis
DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"


@st.cache_resource
def load_plate_recognizer():
    """Mencoba mengimpor dan memuat fungsi prediksi dari fast-plate-ocr secara dinamis."""
    try:
        # Coba import berbagai kemungkinan nama fungsi utama dari fast-plate-ocr
        import fast_plate_ocr
        if hasattr(fast_plate_ocr, "fast_plate_ocr"):
            return fast_plate_ocr.fast_plate_ocr
        elif hasattr(fast_plate_ocr, "predict"):
            return fast_plate_ocr.predict
        else:
            # Fallback jika struktur package berbeda
            return fast_plate_ocr
    except Exception as e:
        st.error(f"Gagal memuat modul fast_plate_ocr: {e}")
        return None


def preprocess_plate_crop(crop_bgr):
    """Preprocessing ringan untuk mempertajam potongan gambar plat nomor."""
    if crop_bgr is None or crop_bgr.size == 0:
        return None

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    resized = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    return resized


def recognize_plate(crop_bgr):
    """Fungsi utama untuk membaca teks dari potongan gambar (crop) plat nomor."""
    ocr_func = load_plate_recognizer()
    if ocr_func is None:
        return ""

    processed_img = preprocess_plate_crop(crop_bgr)
    if processed_img is None:
        return ""

    try:
        # Jalankan fungsi prediksi yang tersedia
        if callable(ocr_func):
            plate_text = ocr_func(processed_img)
        elif hasattr(ocr_func, "fast_plate_ocr"):
            plate_text = ocr_func.fast_plate_ocr(processed_img)
        else:
            return ""

        return str(plate_text).strip().upper()
    except Exception:
        return ""
