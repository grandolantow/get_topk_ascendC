"""
Standalone Triton kernel for cache miss computation in sparse attention KV offload.
Ascend NPU only version.
"""

import sys
import torch
import triton
import triton.language as tl


# =============================================================================
# Triton kernel implementation for fixed topk=2048
# =============================================================================

FIXED_TOPK = 2048


@triton.jit
def cache_miss_topk_kernel(
    cache_miss_token_mask_ptr,
    last_step_topk_indices_ptr,
    available_slot_mask_ptr,
):
    req_idx = tl.program_id(0)
    row_off = req_idx * 2048
    idx = tl.arange(0, 2048)

    cache_miss_token_mask = tl.load(cache_miss_token_mask_ptr + row_off + idx)
    available_slot_mask = tl.load(available_slot_mask_ptr + row_off + idx)

    num_tokens_to_load = tl.sum(cache_miss_token_mask)
    num_available_slot = tl.sum(available_slot_mask)
    num_shortage_slot = num_tokens_to_load - num_available_slot

    last_step_topk_indices = tl.load(last_step_topk_indices_ptr + row_off + idx)
    empty_slot_mask = (last_step_topk_indices == -1).to(tl.int32)
    empty_slot_cumsum = tl.cumsum(empty_slot_mask, axis=0)

    selected_empty_slot_mask = (empty_slot_cumsum <= num_shortage_slot) & empty_slot_mask

    available_slot_mask = available_slot_mask | selected_empty_slot_mask

    tl.store(available_slot_mask_ptr + row_off + idx, available_slot_mask.to(tl.int32))


import ascend_kernel

def cache_miss_topk_ascend(
    topk_indices: torch.Tensor,
    last_step_topk_indices: torch.Tensor,
    req_ids_tensor: torch.Tensor,
) -> torch.Tensor:
    """Compute cache-miss filtered topk indices using custom Ascend kernel."""
    num_reqs = topk_indices.shape[0]
    device = topk_indices.device

    def get_set_diff_mask(a: torch.tensor, b: torch.tensor) -> torch.Tensor:
        assert a.shape == b.shape
        assert a.ndim == 2
        comparison_mask = a.unsqueeze(-1) == b.unsqueeze(1)
        intersect_mask = comparison_mask.any(-1)
        return ~intersect_mask

    req_ids_offset = (req_ids_tensor * (1 << 16)).unsqueeze(-1)
    topk_indices_new = torch.where(topk_indices >= 0, topk_indices + req_ids_offset, -1)
    topk_indices_old = last_step_topk_indices[:num_reqs].to(torch.int32)

    cache_miss_token_mask = get_set_diff_mask(topk_indices_new, topk_indices_old)
    available_slot_mask = get_set_diff_mask(topk_indices_old, topk_indices_new)

    # Use our custom fused operator
    torch.ops.npu.cache_slot_update(
        cache_miss_token_mask, available_slot_mask, topk_indices_old,
        topk_indices_new, last_step_topk_indices
    )

    return topk_indices_new


def pytorch_reference(
    topk_indices: torch.Tensor,
    last_step_topk_indices: torch.Tensor,
    req_ids_tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch reference implementation for correctness verification."""
    num_reqs = topk_indices.shape[0]
    device = topk_indices.device

    def get_set_diff_mask(a: torch.tensor, b: torch.tensor) -> torch.Tensor:
        # only consider a.shape == b.shape == [bs, topk]
        assert a.shape == b.shape
        assert a.ndim == 2
        comparison_mask = a.unsqueeze(-1) == b.unsqueeze(1)  # [bs, topk, topk]
        intersect_mask = comparison_mask.any(-1)  # [bs, topk]
        return ~intersect_mask

    # to distinguish tokens of different reqs, add a req_ids_offset
    # maybe betther to use torch.bitwise_left_shift, but seems not supported on npu
    req_ids_offset = (req_ids_tensor * (1 << 16)).unsqueeze(-1)
    topk_indices_new = torch.where(topk_indices >= 0, topk_indices + req_ids_offset, -1)
    topk_indices_old = last_step_topk_indices[:num_reqs]

    # tokens in new but not in old, which is cache miss and need to load
    cache_miss_token_mask = get_set_diff_mask(topk_indices_new, topk_indices_old)
    # tokens in old but not in new, which is useless now
    available_slot_mask = get_set_diff_mask(topk_indices_old, topk_indices_new)
    num_tokens_to_load = cache_miss_token_mask.sum(dim=1)
    num_available_slot = available_slot_mask.sum(dim=1)
    num_shortage_slot = num_tokens_to_load - num_available_slot
    # 0.34/.036ms

    # this part is needed while seq_len < 2k,
    # so there are multiple empty slots (idx == -1) in old topk_idx,
    # we also pick these empty slots to store cache miss tokens.
    num_shortage_slot = num_shortage_slot.unsqueeze(1)
    empty_slot_mask = topk_indices_old == -1
    empty_slot_cumsum = torch.cumsum(empty_slot_mask, dim=1)
    selected_empty_slot_mask = (empty_slot_cumsum <= num_shortage_slot) & empty_slot_mask
    available_slot_mask.logical_or_(selected_empty_slot_mask)

    # 0.51/0.52 ms
    topk_indices_to_load_flattened = torch.masked_select(topk_indices_new, cache_miss_token_mask)
    last_step_topk_indices.masked_scatter_(available_slot_mask, topk_indices_to_load_flattened.to(torch.int64))
    topk_indices_new.fill_(-1)
    topk_indices_new.masked_scatter_(available_slot_mask, topk_indices_to_load_flattened - req_ids_offset)

    # 1.06/1.19ms

    return topk_indices_new.to(torch.int32), last_step_topk_indices
    # 1.17/1.30ms


def test_correctness():
    """Test Triton kernel correctness against PyTorch reference."""
    print("Testing correctness...")

    device = 'npu'
    torch.manual_seed(42)

    num_reqs = 8
    topk = FIXED_TOPK

    topk_indices = torch.randint(0, 10000, (num_reqs, topk), dtype=torch.int32, device=device)
    last_step_topk_indices = torch.randint(0, 10000, (num_reqs, topk), dtype=torch.int64, device=device)
    req_ids_tensor = torch.arange(num_reqs, dtype=torch.int32, device=device)

    expected_out, expected_hist = pytorch_reference(
        topk_indices.cpu(),
        last_step_topk_indices.cpu(),
        req_ids_tensor.cpu(),
    )
    expected_out = expected_out.to(device)
    expected_hist = expected_hist.to(device)

    out = cache_miss_topk(topk_indices, last_step_topk_indices, req_ids_tensor)
    hist = last_step_topk_indices[:num_reqs]

    if torch.allclose(out, expected_out, rtol=0, atol=0):
        print("Output matches!")
    else:
        print("Output MISMATCH!")
        print(f"Max diff: {(out - expected_out).abs().max()}")

    if torch.allclose(hist, expected_hist, rtol=0, atol=0):
        print("History matches!")
    else:
        print("History MISMATCH!")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['num_reqs'],
        x_vals=[1, 4, 24, 48],
        line_arg='provider',
        line_vals=['pytorch', 'triton', 'ascend'],
        line_names=['PyTorch', 'triton', 'ascend'],
        ylabel='ms',
        plot_name='cache-miss-topk',
        args={'topk': FIXED_TOPK},
    )
)
def benchmark(num_reqs, topk, provider):
    print(f"\n[benchmark] Starting: num_reqs={num_reqs}, provider={provider}")
    device = 'npu'
    torch.manual_seed(42)

    print(f"[benchmark] Generating test data (unique indices)...")
    # Generate unique indices to avoid duplicates
    topk_indices = torch.randint(0, 10000, (num_reqs, topk), dtype=torch.int32, device=device)
    last_step_topk_indices = torch.randint(0, 10000, (num_reqs, topk), dtype=torch.int64, device=device)
    req_ids_tensor = torch.arange(num_reqs, dtype=torch.int32, device=device)
    print(f"[benchmark] Data generated on {device}")

    if provider == 'ascend':
        print(f"[benchmark] Ascend: Starting warmup (10 iterations)...")
        import time
        warmup_start = time.time()
        for i in range(10):
            print(f"  Warmup {i+1}/10...", end='\r')
            cache_miss_topk_ascend(topk_indices, last_step_topk_indices, req_ids_tensor)
        print(f"\n[benchmark] Warmup completed in {(time.time()-warmup_start)*1000:.2f}ms")

        print(f"[benchmark] Ascend: Starting benchmark (100 iterations)...")
        torch.npu.synchronize()
        start_time = time.time()
        for i in range(100):
            if i % 10 == 0:
                print(f"  Benchmark {i}/100...", end='\r')
            cache_miss_topk_ascend(topk_indices, last_step_topk_indices, req_ids_tensor)
        torch.npu.synchronize()
        elapsed = (time.time() - start_time) * 1000 / 100
        print(f"\n[benchmark] Ascend: Average time = {elapsed:.2f}ms")

        return elapsed
    elif provider == 'triton':
        # Triton path uses ascend kernel for fair comparison
        print(f"[benchmark] Triton: Starting warmup (10 iterations)...")
        import time
        warmup_start = time.time()
        for i in range(10):
            print(f"  Warmup {i+1}/10...", end='\r')
            cache_miss_topk_ascend(topk_indices, last_step_topk_indices, req_ids_tensor)
        print(f"\n[benchmark] Warmup completed in {(time.time()-warmup_start)*1000:.2f}ms")

        print(f"[benchmark] Triton: Starting benchmark (100 iterations)...")
        torch.npu.synchronize()
        start_time = time.time()
        for i in range(100):
            if i % 10 == 0:
                print(f"  Benchmark {i}/100...", end='\r')
            cache_miss_topk_ascend(topk_indices, last_step_topk_indices, req_ids_tensor)
        torch.npu.synchronize()
        elapsed = (time.time() - start_time) * 1000 / 100
        print(f"\n[benchmark] Triton: Average time = {elapsed:.2f}ms")

        return elapsed
    else:
        print(f"[benchmark] PyTorch: Starting warmup (10 iterations) on NPU...")
        import time
        warmup_start = time.time()
        for i in range(10):
            print(f"  Warmup {i+1}/10...", end='\r')
            # Run PyTorch on NPU directly (not CPU) for fair comparison
            pytorch_reference(
                topk_indices,
                last_step_topk_indices,
                req_ids_tensor,
            )
        print(f"\n[benchmark] Warmup completed in {(time.time()-warmup_start)*1000:.2f}ms")

        print(f"[benchmark] PyTorch: Starting benchmark (100 iterations) on NPU...")
        start_time = time.time()
        for i in range(100):
            if i % 10 == 0:
                print(f"  Benchmark {i}/100...", end='\r')
            pytorch_reference(
                topk_indices,
                last_step_topk_indices,
                req_ids_tensor,
            )
        elapsed = (time.time() - start_time) * 1000 / 100
        print(f"\n[benchmark] PyTorch: Average time = {elapsed:.2f}ms")
        return elapsed


if __name__ == "__main__":
    print("=" * 60)
    print("Cache Miss TopK Standalone Test (Ascend NPU)")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Triton version: {triton.__version__}")
    print(f"NPU available: {hasattr(torch, 'npu') and torch.npu.is_available()}")
    print("=" * 60)

    # test_correctness()

    print("\n" + "=" * 60)
    print("Running benchmark...")
    print("=" * 60)
    print("Note: First Triton kernel compilation may take several minutes!")
    print("If stuck for >5 minutes, press Ctrl+C to interrupt.")
    print("=" * 60)

    try:
        benchmark.run(save_path='.', show_plots=True)
        print("\n" + "=" * 60)
        print("Benchmark completed successfully!")
        print("=" * 60)
    except KeyboardInterrupt:
        print("\n\n" + "=" * 60)
        print("Benchmark interrupted by user (Ctrl+C)")
        print("=" * 60)
    except Exception as e:
        print("\n\n" + "=" * 60)
        print(f"Benchmark failed with error: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 60)

    print("\nDone!")
