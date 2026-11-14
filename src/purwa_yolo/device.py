"""Device selection and GPU diagnostics (replaces gpu_availability.py).

Run `python device.py` to print the diagnostics.
"""
import torch


def get_device():
    """Ultralytics `device` argument: 0 (first GPU) if CUDA is available, else 'cpu'."""
    return 0 if torch.cuda.is_available() else "cpu"


DEVICE = get_device()
USE_HALF = DEVICE != "cpu"   # FP16 only makes sense on a GPU


def describe_device():
    return "GPU" if DEVICE != "cpu" else "CPU"


def print_diagnostics():
    print("--- Status CUDA & PyTorch ---")
    print(f"Versi PyTorch        : {torch.__version__}")
    print(f"CUDA Tersedia        : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"Versi CUDA PyTorch   : {torch.version.cuda}")
        print(f"Nama GPU             : {torch.cuda.get_device_name(0)}")
        print(f"Compute Capability   : {torch.cuda.get_device_capability(0)}")
        print(f"Daftar Arsitektur    : {torch.cuda.get_arch_list()}")
    else:
        print("Peringatan: PyTorch berjalan di CPU.")


if __name__ == "__main__":
    print_diagnostics()
