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


def benchmark_torch_npu(a: torch.Tensor, b: torch.Tensor, warmup: int = 10, iters: int = 50):
    """Benchmark PyTorch reference implementation on NPU"""
    # Warmup
    for _ in range(warmup):
        _ = get_set_diff_mask(a, b)
    torch.npu.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        result = get_set_diff_mask(a, b)
    torch.npu.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed


def benchmark_ascendc(a: torch.Tensor, b: torch.Tensor, warmup: int = 10, iters: int = 50):
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


def run_single_config(batch_size, topk, iters=50):
    """Run a single benchmark configuration on NPU only"""
    print(f"\n{'='*60}")
    print(f"Config: batch_size={batch_size}, topk={topk}, iters={iters}")
    print(f"{'='*60}")

    torch.manual_seed(42)
    a = torch.randint(0, 10000, (batch_size, topk), dtype=torch.int32).npu()
    b = torch.randint(0, 10000, (batch_size, topk), dtype=torch.int32).npu()

    # Verify correctness
    npu_result = torch.ops.npu.set_difference(a, b)
    torch_ref = get_set_diff_mask(a, b)
    match = (npu_result == torch_ref).all()
    print(f"Correctness: {'PASS' if match else 'FAIL'}")
    if not match:
       print(f"  Mismatched: {(npu_result != torch_ref).sum()}")
       return None

    # Benchmark PyTorch NPU
    print(f"\n--- PyTorch NPU ---")
    torch_time = benchmark_torch_npu(a, b, warmup=10, iters=iters)
    torch_ms = torch_time * 1000 / iters
    print(f"  NPU time:  {torch_ms:.3f} ms/iter")

    # Benchmark AscendC NPU
    print(f"\n--- AscendC Kernel ---")
    ascendc_time = benchmark_ascendc(a, b, warmup=0, iters=iters)
    ascendc_ms = ascendc_time * 1000 / iters
    print(f"  NPU time:  {ascendc_ms:.3f} ms/iter")

    # Speedup
    if ascendc_time > 0:
        speedup = torch_time / ascendc_time
    else:
        speedup = float('inf')
    print(f"\n--- Speedup (PyTorch NPU / AscendC) ---")
    if speedup > 1:
        print(f"  AscendC is {speedup:.2f}x FASTER than PyTorch NPU")
    else:
        print(f"  AscendC is {1/speedup:.2f}x SLOWER than PyTorch NPU")

    return {
        "batch_size": batch_size,
        "topk": topk,
        "torch_npu_ms": torch_ms,
        "ascendc_ms": ascendc_ms,
        "speedup": speedup,
    }


def main():
    print("="*60)
    print("NPU Performance Benchmark: AscendC set_difference vs PyTorch NPU")
    print("Fixed topk=2048, varying batch_size")
    print("="*60)

    TOPK = 2048
    batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    results = []

    for bs in batch_sizes:
        try:
            result = run_single_config(bs, TOPK, iters=1)
            if result:
                results.append(result)
        except Exception as e:
            print(f"ERROR: {e}")

    # Summary table
    print(f"\n\n{'='*70}")
    print("Summary (topk=2048)")
    print(f"{'='*70}")
    print(f"{'BS':>6} {'PyTorch NPU(ms)':>16} {'AscendC(ms)':>14} {'Speedup':>12}")
    print("-" * 70)
    for r in results:
        tag = "FASTER" if r['speedup'] > 1 else "SLOWER"
        print(f"{r['batch_size']:>6} {r['torch_npu_ms']:>16.3f} {r['ascendc_ms']:>14.3f} {r['speedup']:>11.2f}x {tag}")


if __name__ == "__main__":
    main()
