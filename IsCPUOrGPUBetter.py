import torch
import time
import pytorch_ocl

# Prüfen, ob OpenCL verfügbar ist

def benchmark(device, size=2048):
    print(f"\nRunning on: {device}")

    # Zufalls-Matrizen erzeugen
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)

    # Warmup (wichtig für GPU/CL)
    for _ in range(3):
        _ = a @ b

    # Benchmark
    start = time.time()
    for _ in range(10):
        _ = a @ b
    end = time.time()

    print(f"Time: {end - start:.4f} seconds")
    return end - start


if __name__ == "__main__":
    print("PyTorch version:", torch.__version__)

    # CPU Benchmark
    cpu_time = benchmark("cpu")

    # OpenCL Benchmark
    try:
        ocl_time = benchmark("ocl:0")
    except Exception as e:
        print("Error using OpenCL:", e)
        ocl_time = None

    print("\n===== RESULTS =====")
    print(f"CPU time: {cpu_time:.4f} s")
    if ocl_time is not None:
        print(f"OCL time: {ocl_time:.4f} s")
        faster = "OpenCL" if ocl_time < cpu_time else "CPU"
        print(f"Faster device: {faster}")
    else:
        print("OpenCL not available or failed.")
