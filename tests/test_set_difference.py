import torch
import torch_npu
import ascend_kernel

def get_set_diff_mask(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """PyTorch reference implementation"""
    assert a.shape == b.shape
    assert a.ndim == 2
    comparison_mask = a.unsqueeze(-1) == b.unsqueeze(1)  # [bs, topk, topk]
    intersect_mask = comparison_mask.any(-1)              # [bs, topk]
    return ~intersect_mask                                # [bs, topk]

def test_set_difference():
    # Test case 1: basic functionality
    batch_size = 4
    topk = 2048

    # Create test data
    torch.manual_seed(42)
    a = torch.randint(0, 10000, (batch_size, topk), dtype=torch.int32).npu()
    b = torch.randint(0, 10000, (batch_size, topk), dtype=torch.int32).npu()

    # Run NPU operator
    npu_result = torch.ops.npu.set_difference(a, b)

    # Run reference implementation
    cpu_result = get_set_diff_mask(a.cpu(), b.cpu())

    # Compare results
    npu_cpu = npu_result.cpu()
    match = (npu_cpu == cpu_result).all()

    print(f"Test basic: {'PASS' if match else 'FAIL'}")
    if not match:
        print(f"  Mismatched elements: {(npu_cpu != cpu_result).sum()}")

    # Test case 2: all same (should be all False)
    a_same = torch.arange(topk, dtype=torch.int32).unsqueeze(0).repeat(batch_size, 1).npu()
    b_same = a_same.clone()
    npu_result_same = torch.ops.npu.set_difference(a_same, b_same)
    expected_same = torch.zeros((batch_size, topk), dtype=torch.bool)
    match_same = (npu_result_same.cpu() == expected_same).all()
    print(f"Test all_same: {'PASS' if match_same else 'FAIL'}")

    # Test case 3: all different (should be all True)
    a_diff = torch.arange(0, topk, dtype=torch.int32).unsqueeze(0).repeat(batch_size, 1).npu()
    b_diff = torch.arange(topk, 2*topk, dtype=torch.int32).unsqueeze(0).repeat(batch_size, 1).npu()
    npu_result_diff = torch.ops.npu.set_difference(a_diff, b_diff)
    expected_diff = torch.ones((batch_size, topk), dtype=torch.bool)
    match_diff = (npu_result_diff.cpu() == expected_diff).all()
    print(f"Test all_diff: {'PASS' if match_diff else 'FAIL'}")

    # Test case 4: small batch
    a_small = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32).npu()
    b_small = torch.tensor([[1, 3, 5], [4, 6, 8]], dtype=torch.int32).npu()
    npu_result_small = torch.ops.npu.set_difference(a_small, b_small)
    expected_small = torch.tensor([[False, True, False], [False, True, True]], dtype=torch.bool)
    match_small = (npu_result_small.cpu() == expected_small).all()
    print(f"Test small: {'PASS' if match_small else 'FAIL'}")
    if not match_small:
        print(f"  NPU: {npu_result_small.cpu()}")
        print(f"  Expected: {expected_small}")

    print("\nAll tests completed!")

if __name__ == "__main__":
    test_set_difference()
