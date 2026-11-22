"""Entry point:  python -m poetry run streamlit run streamlit_app.py"""
import os
import sys
from pathlib import Path

# FIX: Konflik library MKL (Conda) dan PyTorch di Windows menyebabkan error "OMP: Error #15".
# Kita set environment variable ini agar aplikasi tidak crash saat startup.
if sys.platform == "win32":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# FIX: Agar folder 'src' bisa diakses sebagai package 'purwa_yolo'
# meskipun aplikasi di-deploy tanpa di-install (misal di Streamlit Cloud).
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from purwa_yolo.main import main  # noqa: E402

main()
