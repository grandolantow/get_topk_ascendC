"""
Cache miss topk performance benchmark.
Compares PyTorch vs AscendC implementation.
"""

import argparse
import sys
import time
import torch

import ascend_kernel


FIXED_TOPK = 2048
MAX_NUM_REQS = 64  # Maximum number of requests (buffer size for last_step_topk_indices)


def pytorch_reference(
    topk_indices: torch.Tensor,
    last_step_topk_indices: torch.Tensor,
    req_ids_tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch reference implementation."""
    def get_set_diff_mask(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute set difference mask: elements in a but not in b."""
        assert a.shape == b.shape
        assert a.ndim == 2
        comparison_mask = a.unsqueeze(-1) == b.unsqueeze(1)
        intersect_mask = comparison_mask.any(-1)
        return ~intersect_mask

    num_reqs = topk_indices.shape[0]

    req_ids_offset = (req_ids_tensor[:num_reqs] * (1 << 16)).unsqueeze(-1)
    topk_indices_new = torch.where(topk_indices >= 0, topk_indices + req_ids_offset, -1)
    topk_indices_old = last_step_topk_indices[:num_reqs]

    cache_miss_token_mask = get_set_diff_mask(topk_indices_new, topk_indices_old)
    available_slot_mask = get_set_diff_mask(topk_indices_old, topk_indices_new)
    num_tokens_to_load = cache_miss_token_mask.sum(dim=1)
    num_available_slot = available_slot_mask.sum(dim=1)
    num_shortage_slot = num_tokens_to_load - num_available_slot

    num_shortage_slot = num_shortage_slot.unsqueeze(1)
    empty_slot_mask = topk_indices_old == -1
    empty_slot_cumsum = torch.cumsum(empty_slot_mask, dim=1)
    selected_empty_slot_mask = (empty_slot_cumsum <= num_shortage_slot) & empty_slot_mask
    available_slot_mask = torch.where(selected_empty_slot_mask, True, available_slot_mask)

    topk_indices_to_load_flattened = topk_indices_new[cache_miss_token_mask]
    topk_indices_new.fill_(-1)
    topk_indices_new[available_slot_mask] = topk_indices_to_load_flattened

    last_step_topk_indices[:num_reqs] = torch.where(available_slot_mask, topk_indices_new,
                                                         last_step_topk_indices[:num_reqs])

    topk_indices_new = torch.where(topk_indices_new >= 0, topk_indices_new - req_ids_offset, -1)

    return topk_indices_new.to(torch.int32)


def ascend_impl(
    topk_indices: torch.Tensor,
    last_step_topk_indices: torch.Tensor,
    req_ids_tensor: torch.Tensor,
) -> torch.Tensor:
    """AscendC implementation - all logic moved to op_host."""
    torch.ops.npu.get_cache_miss_topk_indices(
        topk_indices, last_step_topk_indices, req_ids_tensor
    )
    return topk_indices


def generate_topk_indices(batch_size: int, topk: int, max_value: int = 10000, device='npu', seed: int = None, num_neg_ones: int = 256):
    """Generate topk indices with no duplicates, padded with -1 if needed.
    """
    if seed is not None:
        torch.manual_seed(seed)
    result = torch.full((batch_size, topk), -1, dtype=torch.int32, device=device)

    actual_topk = topk - num_neg_ones
    
    for b in range(batch_size):
        unique_indices = torch.randperm(max_value)[:actual_topk]
        result[b, :actual_topk] = unique_indices.to(torch.int32)

    return result


def benchmark(num_reqs: int, topk: int, provider: str, warmup_iters: int = 10, bench_iters: int = 100) -> float:
    """Benchmark a single configuration."""
    device = 'npu'

    last_step_topk_indices = torch.full((MAX_NUM_REQS, topk), -1, dtype=torch.int64, device=device)
    req_ids_tensor = torch.arange(MAX_NUM_REQS, dtype=torch.int64, device=device)
    topk_indices_list = []
    for i in range(warmup_iters + bench_iters):
        num_neg_ones = max(0, 256 - i)  # Decrease by 1 each iteration, min 0
        topk_indices_list.append(generate_topk_indices(num_reqs, topk, device=device, seed=1000 + i, num_neg_ones=num_neg_ones))

    if provider == 'ascend':
        impl = ascend_impl
    elif provider == 'pytorch':
        impl = pytorch_reference
    else:
        raise ValueError(f"Unknown provider: {provider}")

    # Warmup
    for i in range(warmup_iters):
        impl(topk_indices_list[i], last_step_topk_indices, req_ids_tensor)
    torch.npu.synchronize()

    # Benchmark (pure execution time, no data generation overhead)
    start_time = time.time()
    for i in range(bench_iters):
        impl(topk_indices_list[warmup_iters + i], last_step_topk_indices, req_ids_tensor)
    torch.npu.synchronize()

    elapsed = (time.time() - start_time) * 1000 / bench_iters
    return elapsed


def test_correctness(num_reqs: int, topk: int) -> bool:
    """Test correctness by comparing PyTorch and AscendC implementations."""
    device = 'npu'
    torch.manual_seed(42)

    # Generate test data
    topk_indices = generate_topk_indices(num_reqs, topk, device=device)
    last_step_topk_indices = torch.full((MAX_NUM_REQS, topk), -1, dtype=torch.int64, device=device)
    req_ids_tensor = torch.arange(MAX_NUM_REQS, dtype=torch.int64, device=device)

    # Clone inputs for both implementations
    topk_indices_pt = topk_indices.clone()
    last_step_topk_indices_pt = last_step_topk_indices.clone()
    
    topk_indices_asc = topk_indices.clone()
    last_step_topk_indices_asc = last_step_topk_indices.clone()

    # Run both implementations
    pytorch_result = pytorch_reference(topk_indices_pt, last_step_topk_indices_pt, req_ids_tensor)
    ascend_result = ascend_impl(topk_indices_asc, last_step_topk_indices_asc, req_ids_tensor)

    # Compare results for both output tensors
    result_match = torch.allclose(pytorch_result.cpu(), ascend_result.cpu(), rtol=0, atol=0)
    last_step_match = torch.allclose(last_step_topk_indices_pt.cpu(), last_step_topk_indices_asc.cpu(), rtol=0, atol=0)
    match = result_match and last_step_match

    if not match:
        print(f"\n  Mismatch detected!")
        if not result_match:
            print(f"  [topk_indices_new mismatch]")
            print(f"    PyTorch result shape: {pytorch_result.shape}")
            print(f"    AscendC result shape: {ascend_result.shape}")
            diff_mask = (pytorch_result.cpu() != ascend_result.cpu())
            diff_count = diff_mask.sum().item()
            print(f"    Different elements: {diff_count}")
            if diff_count > 0:
                print(f"    Max diff: {(pytorch_result.cpu() - ascend_result.cpu()).abs().max()}")
        if not last_step_match:
            print(f"  [last_step_topk_indices mismatch]")
            print(f"    PyTorch shape: {last_step_topk_indices_pt.shape}")
            print(f"    AscendC shape: {last_step_topk_indices_asc.shape}")
            diff_mask = (last_step_topk_indices_pt.cpu() != last_step_topk_indices_asc.cpu())
            diff_count = diff_mask.sum().item()
            print(f"    Different elements: {diff_count}")
            if diff_count > 0:
                print(f"    Max diff: {(last_step_topk_indices_pt.cpu() - last_step_topk_indices_asc.cpu()).abs().max()}")

    return match


def run_correctness_tests():
    """Run correctness tests and return True if all passed."""
    print("\n" + "=" * 60)
    print("Running Correctness Tests...")
    print("=" * 60)
    
    test_configs = [(1, 2048), (4, 2048), (8, 2048)]
    all_passed = True
    
    for num_reqs, topk in test_configs:
        result = test_correctness(num_reqs, topk)
        status = "PASS" if result else "FAIL"
        print(f"  num_reqs={num_reqs:3d}, topk={topk:4d}: {status}")
        all_passed = all_passed and result
    
    print("-" * 60)
    return all_passed


def run_benchmark():
    """Run performance benchmark."""
    num_reqs_list = [1, 4, 8, 16]
    topk = FIXED_TOPK

    print(f"\nBenchmarking with topk={topk}")
    print("-" * 60)
    print(f"{'Num Requests':<15} {'PyTorch (ms)':<15} {'AscendC (ms)':<15} {'Speedup':<10}")
    print("-" * 60)

    for num_reqs in num_reqs_list:
        pytorch_time = benchmark(num_reqs, topk, 'pytorch')
        ascend_time = benchmark(num_reqs, topk, 'ascend')
        speedup = pytorch_time / ascend_time if ascend_time > 0 else float('inf')

        print(f"{num_reqs:<15} {pytorch_time:<15.3f} {ascend_time:<15.3f} {speedup:<10.2f}x")

    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description='Cache Miss TopK Performance Benchmark')
    parser.add_argument('-c', '--check-correctness', action='store_true',
                        help='Run correctness tests before benchmark')
    args = parser.parse_args()

    print("=" * 60)
    print("Cache Miss TopK Performance Benchmark")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"NPU available: {hasattr(torch, 'npu') and torch.npu.is_available()}")
    print("=" * 60)

    # Run correctness tests if requested
    if args.check_correctness:
        if not run_correctness_tests():
            print("CORRECTNESS TEST FAILED! Stopping benchmark.")
            return
        print("All correctness tests passed!")

    # Run performance benchmark
    run_benchmark()
    print("\nDone!")


if __name__ == "__main__":
    main()
