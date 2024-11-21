import random
from collections import defaultdict
import minitorch
import time
import numpy as np
import matplotlib.pyplot as plt

FastTensorBackend = minitorch.TensorBackend(minitorch.FastOps)
GPUBackend = minitorch.TensorBackend(minitorch.CudaOps)


def run_matmul(backend, size=16) -> None:
    batch_size = 2
    x = minitorch.rand((batch_size, size, size), backend=backend)
    y = minitorch.rand((batch_size, size, size), backend=backend)
    z = x @ y


if __name__ == "__main__":
    # Warmup
    run_matmul(FastTensorBackend)
    run_matmul(GPUBackend)

    ntrials = 3
    times = {}
    sizes = [64, 128, 256, 512, 1024]  # Define sizes for matrices

    for size in sizes:
        print(f"Running size {size}")
        times[size] = {}
        fast_times = []
        gpu_times = []

        for _ in range(ntrials):
            # Timing FastOps (CPU)
            start_fast = time.time()
            run_matmul(FastTensorBackend, size)
            end_fast = time.time()

            # Timing GPUOps (CUDA)
            start_gpu = time.time()
            run_matmul(GPUBackend, size)
            end_gpu = time.time()

            fast_time = end_fast - start_fast
            gpu_time = end_gpu - start_gpu

            fast_times.append(fast_time)
            gpu_times.append(gpu_time)

        times[size]["fast"] = np.mean(fast_times)
        times[size]["gpu"] = np.mean(gpu_times)
        print(times[size])

    print()
    print("Timing summary")
    for size, stimes in times.items():
        print(f"Size: {size}")
        for b, t in stimes.items():
            print(f"    {b}: {t:.5f}")

    # Plot results
    plt.figure()
    plt.plot(sizes, [times[size]["fast"] for size in sizes], label="FastOps (CPU)", marker="o")
    plt.plot(sizes, [times[size]["gpu"] for size in sizes], label="GPUOps (CUDA)", marker="o")
    plt.xlabel("Matrix Size")
    plt.ylabel("Time (s)")
    plt.title("Matrix Multiplication Timing: FastOps vs GPUOps")
    plt.legend()
    plt.grid(True)
    plt.show()