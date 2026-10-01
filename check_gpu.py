import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version (PyTorch built with): {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Supported architectures: {torch.cuda.get_arch_list()}")

    # Actually run something on the GPU to confirm it truly works,
    # not just that CUDA initializes.
    try:
        a = torch.rand(1024, 1024, device="cuda")
        b = torch.rand(1024, 1024, device="cuda")
        c = a @ b
        torch.cuda.synchronize()
        print(f"[OK] Matrix multiply on GPU succeeded, result shape: {c.shape}")
    except Exception as e:
        print(f"[ERROR] GPU compute test failed: {e}")
else:
    print("[WARNING] No GPU detected -- training will run on CPU and be much slower.")
