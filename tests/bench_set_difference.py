import torch
import torch_npu
import ascend_kernel
import time


def get_set_diff_mask(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """PyTorch reference implementation"""
    assert a.shape == b.shape
    assert a.ndim == 2
    comparison_mask = a.unsqueeze(-1) == b.unsqueeze(1)  # [bs, topk, topk]
    intersect_mask = comparison_mask.any(-1)              # [bs, topk]
    return ~intersect_mask                                # [bs, topk]


def benchmark_torch(a: torch.Tensor, b: torch.Tensor, warmup: int = 5, iters: int = 20):
    """Benchmark PyTorch reference implementation on CPU (since it's O(n^3) memory on NPU)"""
    a_cpu = a.cpu()
    b_cpu = b.cpu()

    # Warmup
    for _ in range(warmup):
        _ = get_set_diff_mask(a_cpu, b_cpu)

    # Synchronize before timing
    torch.cuda.synchronize() if torch.cuda.is_available() else None

    start = time.perf_counter()
    for _ in range(iters):
        result = get_set_diff_mask(a_cpu, b_cpu)
    torch_cpu_sync = time.perf_counter() - start

    # Also test NPU fallback (memory heavy)
    try:
        for _ in range(warmup):
            _ = get_set_diff_mask(a, b)

        torch.npu.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            result = get_set_diff_mask(a, b)
        torch.npu.synchronize()
        torch_npu_sync = time.perf_counter() - start
    except RuntimeError as e:
        torch_npu_sync = None
        print(f"  PyTorch NPU OOM/failed: {e}")

    return torch_cpu_sync, torch_npu_sync


def benchmark_npu(a: torch.Tensor, b: torch.Tensor, warmup: int = 5, iters: int = 20):
    """Benchmark ascendC NPU kernel"""
    # Warmup
    for _ in range(warmup):
        _ = torch.ops.npu.set_difference(a, b)

    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        result = torch.ops.npu.set_difference(a, b)
    torch.npu.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed


def run_benchmark(batch_size, topk, iters=20):
    """Run a single benchmark configuration"""
    print(f"\n{'='*60}")
    print(f"Config: batch_size={batch_size}, topk={topk}, iters={iters}")
    print(f"{'='*60}")

    torch.manual_seed(42)
    a = torch.randint(0, 10000, (batch_size, topk), dtype=torch.int32).npu()
    b = torch.randint(0, 10000, (batch_size, topk), dtype=torch.int32).npu()

    # Verify correctness first
    npu_result = torch.ops.npu.set_difference(a, b)
    cpu_ref = get_set_diff_mask(a.cpu(), b.cpu())
    match = (npu_result.cpu() == cpu_ref).all()
    print(f"Correctness: {'PASS' if match else 'FAIL'}")
    if not match:
        print(f"  Mismatched: {(npu_result.cpu() != cpu_ref).sum()}")
        return None

    # Benchmark PyTorch
    print(f"\n--- PyTorch Reference ---")
    torch_cpu_time, torch_npu_time = benchmark_torch(a, b, warmup=5, iters=iters)
    print(f"  CPU time:  {torch_cpu_time*1000/iters:.3f} ms/iter")
    if torch_npu_time:
        print(f"  NPU time:  {torch_npu_time*1000/iters:.3f} ms/iter")

    # Benchmark NPU kernel
    print(f"\n--- AscendC Kernel ---")
    npu_time = benchmark_npu(a, b, warmup=5, iters=iters)
    print(f"  NPU time:  {npu_time*1000/iters:.3f} ms/iter")

    # Speedup
    print(f"\n--- Speedup vs PyTorch CPU ---")
    speedup = torch_cpu_time / npu_time
    print(f"  {speedup:.2f}x faster")

    if torch_npu_time:
        print(f"\n--- Speedup vs PyTorch NPU ---")
        speedup_npu = torch_npu_time / npu_time
        print(f"  {speedup_npu:.2f}x faster")

    return {
        "batch_size": batch_size,
        "topk": topk,
        "torch_cpu_ms": torch_cpu_time * 1000 / iters,
        "torch_npu_ms": torch_npu_time * 1000 / iters if torch_npu_time else None,
        "ascendc_ms": npu_time * 1000 / iters,
        "speedup_vs_cpu": torch_cpu_time / npu_time,
        "speedup_vs_npu": torch_npu_time / npu_time if torch_npu_time else None,
    }


def main():
    print("="*60)
    print("Performance Benchmark: AscendC set_difference vs PyTorch")
    print("="*60)

    configs = [
        (4, 128),
        (4, 256),
        (4, 512),
        (4, 1024),
        (4, 2048),
        (8, 128),
        (8, 512),
        (8, 2048),
        (16, 128),
        (16, 512),
        (16, 2048),
    ]

    results = []
    for bs, tk in configs:
        try:
            result = run_benchmark(bs, tk, iters=20)
            if result:
                results.append(result)
        except Exception as e:
            print(f"ERROR: {e}")

    # Summary table
    print(f"\n\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"{'BS':>4} {'TopK':>6} {'TorchCPU(ms)':>12} {'AscendC(ms)':>12} {'Speedup(vs CPU)':>15}")
    print("-" * 60)
    for r in results:
        print(f"{r['batch_size']:>4} {r['topk']:>6} {r['torch_cpu_ms']:>12.3f} {r['ascendc_ms']:>12.3f} {r['speedup_vs_cpu']:>15.2f}x")


if __name__ == "__main__":
    main()
