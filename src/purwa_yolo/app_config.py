"""Central configuration: constants only (no logic)."""
from pathlib import Path

# ---------------- paths ----------------
# this file lives in src/purwa_yolo/
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# Map selection to model filename
MODEL_FILES = {
    "Construction Equipment": "best_construction.pt",
    "Vehicle": "best_vehicle.pt",
    "Fruit": "best_fruit.pt",
}
FALLBACK_VEHICLE_MODEL = "best.pt"   # used when best_vehicle.pt does not exist

# ---------------- uploads ----------------
IMAGE_TYPES = ["jpg", "jpeg", "png", "webp"]
VIDEO_TYPES = ["mp4", "mov", "avi", "mkv"]
MAX_IMAGE_MB = 10      # image size limit (checked in code)
MAX_VIDEO_MB = 100     # video size limit; server.maxUploadSize must be >= this
MAX_VIDEO_SIDE = 960   # longest side of the output video (px), for speed

# ---------------- inference ----------------
# smaller = faster, larger = better for small objects
INFER_SIZES = [480, 640, 800]
DEFAULT_IMGSZ = 640
# Video only: a vehicle is counted only if it is tracked for at least this
# many processed frames (filters out momentary detections / short-lived IDs)
MIN_TRACK_FRAMES = 3

# ---------------- ERP pricing ----------------
# Base rates (USD per vehicle). Key = model class name (lowercase).
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

# ---------------- odd-even plate rule ----------------
# fast-alpr = plate detector + fast-plate-ocr (the OCR step)
PLATE_DETECTOR_MODEL = "yolo-v9-t-384-license-plate-end2end"
PLATE_OCR_MODEL = "cct-xs-v2-global-model"
# applied when plate parity differs from the date parity
DEFAULT_PLATE_SURCHARGE = 1.5
# below this OCR confidence a plate counts as unreadable
DEFAULT_PLATE_MIN_CONF = 0.5
# skip vehicle crops smaller than this (plate unreadable)
MIN_OCR_CROP_PX = 64
MAX_PLATE_READS_IMAGE = 40      # safety cap on OCR calls for one image
MAX_OCR_TRIES = 4               # video: OCR attempts per tracked vehicle
OCR_EVERY_N_FRAMES = 3          # video: spread the attempts over different frames
# Indonesia Western Time (WIB), no daylight saving
UTC_OFFSET_HOURS = 7
