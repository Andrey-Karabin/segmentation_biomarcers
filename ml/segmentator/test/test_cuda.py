import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

x = torch.randn(1000, 1000, device=DEVICE)
y = torch.matmul(x, x)
print(y.device)  # должно показать cuda:0

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version (PyTorch build): {torch.version.cuda}")
print(f"GPU Name: {torch.cuda.get_device_name(0)}")
print(f"GPU Compute Capability (sm_): {torch.cuda.get_device_capability(0)}")
