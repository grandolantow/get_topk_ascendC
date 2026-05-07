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


def cache_slot_update_torch(cache_miss, avail, topk_old, topk_new, last_step_topk):
    """PyTorch reference for fused cache_slot_update"""
    # Step 1: compute updated available mask
    new_avail = select_empty_slots_torch(cache_miss, avail, topk_old)

    # Step 2: masked_select topk_new using cache_miss
    topk_to_load = torch.masked_select(topk_new, cache_miss)

    # Step 3: scatter into last_step_topk (needs int64)
    last_step_topk.masked_scatter_(new_avail, topk_to_load.to(torch.int64))

    # Step 4: fill topk_new with -1
    topk_new.fill_(-1)

    # Step 5: scatter back into topk_new
    topk_new.masked_scatter_(new_avail, topk_to_load)

    return new_avail


def generate_test_data(batch_size, seq_len, seed=42):
    """Generate test data where cache_miss count == new_avail count"""
    torch.manual_seed(seed)

    # Generate cache_miss mask
    cache_miss = torch.randint(0, 2, (batch_size, seq_len), dtype=torch.bool)

    # Generate topk_old with some -1 values (empty slots)
    topk_old = torch.randint(0, 10000, (batch_size, seq_len), dtype=torch.int32)
    empty_mask = torch.randint(0, 2, (batch_size, seq_len), dtype=torch.bool)
    topk_old[empty_mask] = -1

    # Generate avail mask ensuring avail count <= cache_miss count
    # so that select_empty_slots doesn't add more slots than needed
    avail = torch.zeros((batch_size, seq_len), dtype=torch.bool)
    for b in range(batch_size):
        miss_count = cache_miss[b].sum().item()
        # Make avail count <= miss_count to avoid shortage
        avail_count = min(miss_count, torch.randint(0, miss_count + 1, (1,)).item()) if miss_count > 0 else 0
        if avail_count > 0:
            indices = torch.randperm(seq_len)[:avail_count]
            avail[b, indices] = True

    topk_new = torch.randint(0, 10000, (batch_size, seq_len), dtype=torch.int32)
    last_step_topk = torch.randint(0, 10000, (batch_size, seq_len), dtype=torch.int64)

    return cache_miss.npu(), avail.npu(), topk_old.npu(), topk_new.npu(), last_step_topk.npu()


def test_cache_slot_update():
    batch_size = 4
    seq_len = 2048

    cache_miss, avail, topk_old, topk_new, last_step_topk = generate_test_data(batch_size, seq_len, seed=42)

    # Run NPU fused op (in-place)
    topk_new_npu = topk_new.clone()
    last_step_topk_npu = last_step_topk.clone()
    torch.ops.npu.cache_slot_update(cache_miss, avail, topk_old, topk_new_npu, last_step_topk_npu)

    # Run CPU reference
    topk_new_ref = topk_new.cpu().clone()
    last_step_topk_ref = last_step_topk.cpu().clone()
    new_avail_ref = cache_slot_update_torch(
        cache_miss.cpu(), avail.cpu(), topk_old.cpu(), topk_new_ref, last_step_topk_ref
    )

    # Compare
    match_new = (topk_new_npu.cpu() == topk_new_ref).all()
    match_last = (last_step_topk_npu.cpu() == last_step_topk_ref).all()

    print(f"Test cache_slot_update topk_new: {'PASS' if match_new else 'FAIL'}")
    print(f"Test cache_slot_update last_step_topk: {'PASS' if match_last else 'FAIL'}")

    if not match_new:
        print(f"  topk_new mismatched: {(topk_new_npu.cpu() != topk_new_ref).sum()}")
    if not match_last:
        print(f"  last_step_topk mismatched: {(last_step_topk_npu.cpu() != last_step_topk_ref).sum()}")

    return match_new and match_last


def benchmark_cache_slot_update(batch_size, seq_len, warmup=5, iters=20):
    print(f"\n{'='*60}")
    print(f"Config: batch_size={batch_size}, seq_len={seq_len}, iters={iters}")
    print(f"{'='*60}")

    cache_miss, avail, topk_old, topk_new, last_step_topk = generate_test_data(batch_size, seq_len, seed=42)

    # Verify correctness
    topk_new_npu = topk_new.clone()
    last_step_topk_npu = last_step_topk.clone()
    torch.ops.npu.cache_slot_update(cache_miss, avail, topk_old, topk_new_npu, last_step_topk_npu)

    topk_new_ref = topk_new.cpu().clone()
    last_step_topk_ref = last_step_topk.cpu().clone()
    cache_slot_update_torch(cache_miss.cpu(), avail.cpu(), topk_old.cpu(), topk_new_ref, last_step_topk_ref)

    match_new = (topk_new_npu.cpu() == topk_new_ref).all()
    match_last = (last_step_topk_npu.cpu() == last_step_topk_ref).all()
    print(f"Correctness: {'PASS' if (match_new and match_last) else 'FAIL'}")
    if not (match_new and match_last):
        return None

    # Benchmark PyTorch NPU (separate ops)
    for _ in range(warmup):
        _ = cache_slot_update_torch(cache_miss, avail, topk_old, topk_new.clone(), last_step_topk.clone())

    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        _ = cache_slot_update_torch(cache_miss, avail, topk_old, topk_new.clone(), last_step_topk.clone())
    torch.npu.synchronize()
    torch_npu_time = time.perf_counter() - start
    print(f"\n--- PyTorch NPU (separate ops) ---")
    print(f"  Time:  {torch_npu_time*1000/iters:.3f} ms/iter")

    # Benchmark fused AscendC + ACLNN
    topk_new_bench = topk_new.clone()
    last_step_topk_bench = last_step_topk.clone()
    for _ in range(warmup):
        torch.ops.npu.cache_slot_update(cache_miss, avail, topk_old, topk_new_bench.clone(), last_step_topk_bench.clone())

    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        torch.ops.npu.cache_slot_update(cache_miss, avail, topk_old, topk_new_bench.clone(), last_step_topk_bench.clone())
    torch.npu.synchronize()
    fused_time = time.perf_counter() - start
    print(f"\n--- Fused AscendC + ACLNN ---")
    print(f"  Time:  {fused_time*1000/iters:.3f} ms/iter")

    if fused_time > 0:
        speedup = torch_npu_time / fused_time
        print(f"\n--- Speedup ---")
        print(f"  {speedup:.2f}x faster")

    return {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "torch_npu_ms": torch_npu_time * 1000 / iters,
        "fused_ms": fused_time * 1000 / iters,
    }


def main():
    print("="*60)
    print("Test: cache_slot_update")
    print("="*60)

    test_cache_slot_update()

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
            result = benchmark_cache_slot_update(bs, sl, warmup=5, iters=20)
            if result:
                results.append(result)
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"{'BS':>4} {'SeqLen':>6} {'TorchNPU(ms)':>12} {'Fused(ms)':>12} {'Speedup':>10}")
    print("-" * 60)
    for r in results:
        speedup = r['torch_npu_ms'] / r['fused_ms'] if r['fused_ms'] > 0 else float('inf')
        print(f"{r['batch_size']:>4} {r['seq_len']:>6} {r['torch_npu_ms']:>12.3f} {r['fused_ms']:>12.3f} {speedup:>9.2f}x")


if __name__ == "__main__":
    main()
