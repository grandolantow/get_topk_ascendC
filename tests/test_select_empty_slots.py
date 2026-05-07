import torch
import torch_npu
import ascend_kernel
import time


def select_empty_slots_torch(cache_miss_token_mask, available_slot_mask, topk_indices_old):
    """PyTorch reference implementation"""
    num_tokens_to_load = cache_miss_token_mask.sum(dim=1)
    num_available_slot = available_slot_mask.sum(dim=1)
    num_shortage_slot = num_tokens_to_load - num_available_slot
    num_shortage_slot = num_shortage_slot.unsqueeze(1)

    empty_slot_mask = (topk_indices_old == -1)
    empty_slot_cumsum = torch.cumsum(empty_slot_mask.int(), dim=1)
    selected_empty_slot_mask = (empty_slot_cumsum <= num_shortage_slot) & empty_slot_mask
    result = available_slot_mask | selected_empty_slot_mask
    return result


def test_select_empty_slots():
    batch_size = 4
    seq_len = 2048

    torch.manual_seed(42)
    cache_miss = torch.randint(0, 2, (batch_size, seq_len), dtype=torch.bool).npu()
    avail_slot = torch.randint(0, 2, (batch_size, seq_len), dtype=torch.bool).npu()
    topk_idx = torch.randint(-1, 10000, (batch_size, seq_len), dtype=torch.int32).npu()

    npu_result = torch.ops.npu.select_empty_slots(cache_miss, avail_slot, topk_idx)
    cpu_ref = select_empty_slots_torch(cache_miss.cpu(), avail_slot.cpu(), topk_idx.cpu())

    match = (npu_result.cpu() == cpu_ref).all()
    print(f"Test basic: {'PASS' if match else 'FAIL'}")
    if not match:
        print(f"  Mismatched elements: {(npu_result.cpu() != cpu_ref).sum()}")
        diff = (npu_result.cpu() != cpu_ref).nonzero()
        if len(diff) > 0:
            idx = diff[0].item()
            batch = idx // seq_len
            pos = idx % seq_len
            print(f"  First mismatch at batch={batch}, pos={pos}")
            print(f"  NPU: {npu_result.cpu()[batch, pos].item()}")
            print(f"  Ref: {cpu_ref[batch, pos].item()}")

    # Test case 2: all shortage = 0
    cache_miss2 = torch.zeros((batch_size, seq_len), dtype=torch.bool).npu()
    avail_slot2 = torch.zeros((batch_size, seq_len), dtype=torch.bool).npu()
    topk_idx2 = torch.full((batch_size, seq_len), -1, dtype=torch.int32).npu()
    npu_result2 = torch.ops.npu.select_empty_slots(cache_miss2, avail_slot2, topk_idx2)
    cpu_ref2 = select_empty_slots_torch(cache_miss2.cpu(), avail_slot2.cpu(), topk_idx2.cpu())
    match2 = (npu_result2.cpu() == cpu_ref2).all()
    print(f"Test all_empty_no_shortage: {'PASS' if match2 else 'FAIL'}")

    # Test case 3: all shortage
    cache_miss3 = torch.ones((batch_size, seq_len), dtype=torch.bool).npu()
    avail_slot3 = torch.zeros((batch_size, seq_len), dtype=torch.bool).npu()
    topk_idx3 = torch.full((batch_size, seq_len), -1, dtype=torch.int32).npu()
    npu_result3 = torch.ops.npu.select_empty_slots(cache_miss3, avail_slot3, topk_idx3)
    cpu_ref3 = select_empty_slots_torch(cache_miss3.cpu(), avail_slot3.cpu(), topk_idx3.cpu())
    match3 = (npu_result3.cpu() == cpu_ref3).all()
    print(f"Test all_shortage: {'PASS' if match3 else 'FAIL'}")

    print("\nAll tests completed!")


def benchmark_select_empty_slots(batch_size, seq_len, warmup=5, iters=20):
    print(f"\n{'='*60}")
    print(f"Config: batch_size={batch_size}, seq_len={seq_len}, iters={iters}")
    print(f"{'='*60}")

    torch.manual_seed(42)
    cache_miss = torch.randint(0, 2, (batch_size, seq_len), dtype=torch.bool).npu()
    avail_slot = torch.randint(0, 2, (batch_size, seq_len), dtype=torch.bool).npu()
    topk_idx = torch.randint(-1, 10000, (batch_size, seq_len), dtype=torch.int32).npu()

    # Verify correctness
    npu_result = torch.ops.npu.select_empty_slots(cache_miss, avail_slot, topk_idx)
    cpu_ref = select_empty_slots_torch(cache_miss.cpu(), avail_slot.cpu(), topk_idx.cpu())
    match = (npu_result.cpu() == cpu_ref).all()
    print(f"Correctness: {'PASS' if match else 'FAIL'}")
    if not match:
        print(f"  Mismatched: {(npu_result.cpu() != cpu_ref).sum()}")
        return None

    # Benchmark PyTorch NPU (run on NPU tensors directly)
    for _ in range(warmup):
        _ = select_empty_slots_torch(cache_miss, avail_slot, topk_idx)

    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        result = select_empty_slots_torch(cache_miss, avail_slot, topk_idx)
    torch.npu.synchronize()
    torch_npu_time = time.perf_counter() - start
    print(f"\n--- PyTorch NPU ---")
    print(f"  NPU time:  {torch_npu_time*1000/iters:.3f} ms/iter")

    # Benchmark AscendC
    for _ in range(warmup):
        _ = torch.ops.npu.select_empty_slots(cache_miss, avail_slot, topk_idx)

    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        result = torch.ops.npu.select_empty_slots(cache_miss, avail_slot, topk_idx)
    torch.npu.synchronize()
    npu_time = time.perf_counter() - start
    print(f"\n--- AscendC Kernel ---")
    print(f"  NPU time:  {npu_time*1000/iters:.3f} ms/iter")

    # Speedup
    if npu_time > 0:
        speedup = torch_npu_time / npu_time
        print(f"\n--- Speedup vs PyTorch NPU ---")
        print(f"  {speedup:.2f}x faster")

    return {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "torch_npu_ms": torch_npu_time * 1000 / iters,
        "ascendc_ms": npu_time * 1000 / iters,
    }


def main():
    print("="*60)
    print("Performance Benchmark: AscendC select_empty_slots")
    print("="*60)

    # Run correctness tests
    test_select_empty_slots()

    # Run benchmarks
    configs = [
        (1, 2048),
        (4, 2048),
        (8, 2048),
        (16, 2048),
        (32, 2048),
        (64, 2048),
    ]

    results = []
    for bs, sl in configs:
        try:
            result = benchmark_select_empty_slots(bs, sl, warmup=5, iters=20)
            if result:
                results.append(result)
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Summary table
    print(f"\n\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"{'BS':>4} {'SeqLen':>6} {'TorchNPU(ms)':>12} {'AscendC(ms)':>12} {'Speedup':>10}")
    print("-" * 60)
    for r in results:
        speedup = r['torch_npu_ms'] / r['ascendc_ms'] if r['ascendc_ms'] > 0 else float('inf')
        print(f"{r['batch_size']:>4} {r['seq_len']:>6} {r['torch_npu_ms']:>12.3f} {r['ascendc_ms']:>12.3f} {speedup:>9.2f}x")


if __name__ == "__main__":
    main()
