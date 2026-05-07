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

def test_debug():
    # Small test case for debugging
    a_small = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32).npu()
    b_small = torch.tensor([[1, 3, 5], [4, 6, 8]], dtype=torch.int32).npu()

    # Run NPU operator
    npu_result = torch.ops.npu.set_difference(a_small, b_small)

    # Run reference implementation
    cpu_result = get_set_diff_mask(a_small.cpu(), b_small.cpu())

    print("a:", a_small.cpu())
    print("b:", b_small.cpu())
    print("NPU result:", npu_result.cpu())
    print("CPU reference:", cpu_result)

    # Check intermediate comparison
    for i in range(a_small.size(0)):
        for j in range(a_small.size(1)):
            val = a_small[i, j].item()
            found = (b_small[i] == val).any().item()
            print(f"  a[{i},{j}]={val}, found_in_b={found}, mask_should_be={not found}")

if __name__ == "__main__":
    test_debug()
