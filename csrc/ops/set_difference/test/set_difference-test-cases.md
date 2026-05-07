# set_difference 用例设计文档

## 1. 算子标杆

PyTorch参考实现：
```python
import torch

def get_set_diff_mask(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    计算 set_difference mask。
    a, b: [BatchSize, topK] int32
    返回: [BatchSize, topK] bool
    """
    # 仅考虑 a.shape == b.shape == [bs, topk]
    assert a.shape == b.shape
    assert a.ndim == 2
    comparison_mask = a.unsqueeze(-1) == b.unsqueeze(1)  # [bs, topk, topk]
    intersect_mask = comparison_mask.any(-1)              # [bs, topk]
    return ~intersect_mask                                # [bs, topk]
```

**功能说明**: 对 `a` 中每个元素，判断其是否存在于**同 batch** 的 `b` 中。若不存在（差集），输出 `True`；若存在，输出 `False`。

---

## 2. 用例说明

### 2.1 测试配置

```python
# 支持的数据类型
SUPPORTED_DTYPES = ["int32"]

# 典型用例（生产环境常见 shape）
TEST_SHAPES = [
    ("Small",   "1 batch,  topK=2048",   (1,   2048)),
    ("Small",   "2 batch,  topK=2048",   (2,   2048)),
    ("Small",   "4 batch,  topK=2048",   (4,   2048)),
    ("Medium",  "8 batch,  topK=2048",   (8,   2048)),
    ("Medium",  "16 batch, topK=2048",   (16,  2048)),
    ("Medium",  "32 batch, topK=2048",   (32,  2048)),
    ("Large",   "48 batch, topK=2048",   (48,  2048)),
    ("Large",   "64 batch, topK=2048",   (64,  2048)),
]

# 泛化用例（边界场景 + 压力测试）
GENERAL_SHAPES = [
    # 小 Shape 边界测试
    ("Small",   "1 batch,  topK=64",     (1,   64)),
    ("Small",   "1 batch,  topK=128",    (1,   128)),
    ("Small",   "1 batch,  topK=256",    (1,   256)),
    ("Small",   "2 batch,  topK=512",    (2,   512)),
    ("Small",   "4 batch,  topK=1024",   (4,   1024)),
    ("Small",   "unaligned topK=1000",   (2,   1000)),
    ("Small",   "unaligned topK=1500",   (2,   1500)),

    # 大 Shape 压力测试（生产环境）
    ("Large",   "96 batch,  topK=2048",  (96,  2048)),
    ("Large",   "128 batch, topK=2048",  (128, 2048)),
    ("Large",   "256 batch, topK=2048",  (256, 2048)),
    ("Large",   "512 batch, topK=2048",  (512, 2048)),
]

# 边界值测试
BOUNDARY_VALUES = [
    # 全部相同（a == b，输出全 False）
    {"desc": "all same",      "a": "same_as_b",        "b": "arbitrary",        "expected": "all False"},
    # 全部不同（a 和 b 无交集，输出全 True）
    {"desc": "all different", "a": "disjoint_with_b",  "b": "arbitrary",        "expected": "all True"},
    # 部分交集（a 部分元素在 b 中）
    {"desc": "partial intersect", "a": "half_in_b",    "b": "arbitrary",        "expected": "half True"},
    # 空 batch
    {"desc": "empty batch",   "a": "empty",            "b": "empty",            "expected": "empty"},
    # 单元素
    {"desc": "single element", "a": "[[1]]",           "b": "[[1]]",            "expected": "[[False]]"},
    # 负数测试
    {"desc": "negative values", "a": "neg_values",     "b": "neg_values",       "expected": "mixed"},
    # 大数值测试
    {"desc": "large values",  "a": "large_int32",      "b": "large_int32",      "expected": "mixed"},
    # 重复元素
    {"desc": "duplicates",    "a": "duplicates",       "b": "duplicates",       "expected": "mixed"},
]
```

### 2.2 用例覆盖统计

| 类别 | Shape数量 | 边界值数量 | dtype数量 | 总用例数 |
|------|----------|-----------|----------|---------|
| 常规形状 | 8 | - | 1 | 8 |
| 泛化形状 | 11 | - | 1 | 11 |
| 边界值 | - | 8 | 1 | 8 |
| **总计** | **19** | **8** | **1** | **27** |

> 注：当前算子仅支持 int32，如需扩展至其他 dtype，总用例数将按比例增加。

---

## 3. 使用说明

### 生成测试数据示例

```python
import torch

# 常规测试数据生成
def generate_test_data(batch_size, topk, dtype=torch.int32, seed=42):
    torch.manual_seed(seed)
    a = torch.randint(0, 10000, (batch_size, topk), dtype=dtype)
    b = torch.randint(0, 10000, (batch_size, topk), dtype=dtype)
    return a, b

# 边界值测试数据生成
def generate_boundary_data(case_type, batch_size=4, topk=2048, dtype=torch.int32):
    if case_type == "all same":
        a = torch.arange(topk, dtype=dtype).unsqueeze(0).repeat(batch_size, 1)
        return a, a.clone()
    elif case_type == "all different":
        a = torch.arange(0, topk, dtype=dtype).unsqueeze(0).repeat(batch_size, 1)
        b = torch.arange(topk, 2*topk, dtype=dtype).unsqueeze(0).repeat(batch_size, 1)
        return a, b
    elif case_type == "partial intersect":
        a = torch.arange(topk, dtype=dtype).unsqueeze(0).repeat(batch_size, 1)
        b = torch.arange(topk//2, topk + topk//2, dtype=dtype).unsqueeze(0).repeat(batch_size, 1)
        return a, b
    elif case_type == "duplicates":
        a = torch.randint(0, 100, (batch_size, topk), dtype=dtype)
        b = torch.randint(0, 100, (batch_size, topk), dtype=dtype)
        return a, b
    else:
        raise ValueError(f"Unknown case_type: {case_type}")
```

### 注意事项

1. **数据类型**: 当前算子仅支持 `int32`，所有测试用例均使用 `int32` 生成。
2. **Shape 约束**: `a` 和 `b` 的 shape 必须相同，且第二维（topK）建议为 2048（生产环境典型值），但算子应支持任意 topK。
3. **Batch 切分**: Block 级切分会对齐到 batch 边界，避免跨 batch 加载不同的 `b_batch`。
4. **topK 约束**: 当 topK 不是 2048 时，CompareScalar 的 repeat/count 需要动态计算（`repeat = topK / 64`, `count = 64`）。
5. **精度**: int32 为精确比较，不存在浮点精度问题，预期精度误差为 0。
