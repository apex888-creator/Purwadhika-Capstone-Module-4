import torch

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
