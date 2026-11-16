"""Entry point:  python -m poetry run streamlit run streamlit_app.py"""
import os
import sys
from pathlib import Path

# Windows + conda: conda's MKL and the PyTorch wheel each ship libiomp5md.dll,
# and the second one to start aborts the process ("OMP: Error #15"). This lets
# it continue. Intel calls it unsafe/unsupported; the clean fix is to leave
# only one copy of that DLL in the environment. setdefault lets you override it
# from the terminal (set KMP_DUPLICATE_LIB_OK=FALSE to turn it off).
if sys.platform == "win32":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Make `import purwa_yolo` work even when the package is not installed
# (for example on Streamlit Cloud, which only installs requirements.txt).
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from purwa_yolo.main import main  # noqa: E402

main()
