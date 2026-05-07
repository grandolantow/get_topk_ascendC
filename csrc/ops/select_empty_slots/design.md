# select_empty_slots 设计文档

## 1. 算子接口

### 1.1 函数签名
```cpp
at::Tensor select_empty_slots(
    const at::Tensor &cache_miss_token_mask,
    const at::Tensor &available_slot_mask,
    const at::Tensor &topk_indices_old
);
```

### 1.2 参数说明
| 参数名 | 类型 | 输入/输出 | 支持的数据类型 | 描述 | 约束条件 |
|--------|------|-------|-------|--------|------|
| cache_miss_token_mask | at::Tensor | 输入 | bool | 输入mask1 | shape = (batchSize, 2048) |
| available_slot_mask | at::Tensor | 输入 | bool | 输入mask2 | shape = (batchSize, 2048) |
| topk_indices_old | at::Tensor | 输入 | int32 | 输入indices | shape = (batchSize, 2048) |
| output | at::Tensor | 输出 | bool | 输出mask | shape = (batchSize, 2048) |

### 1.3 支持的数据类型
- [x] bool (mask输入输出)
- [x] int32 (topk_indices_old)

---

## 2. 计算逻辑

### 2.1 算法描述

`select_empty_slots` 算子实现以下逻辑：

```python
num_tokens_to_load = cache_miss_token_mask.sum(dim=1)  # [batchSize]
num_available_slot = available_slot_mask.sum(dim=1)     # [batchSize]
num_shortage_slot = num_tokens_to_load - num_available_slot  # [batchSize]
num_shortage_slot = num_shortage_slot.unsqueeze(1)  # [batchSize, 1]

empty_slot_mask = (topk_indices_old == -1)  # [batchSize, 2048]
empty_slot_cumsum = torch.cumsum(empty_slot_mask, dim=1)  # [batchSize, 2048]
selected_empty_slot_mask = (empty_slot_cumsum <= num_shortage_slot) & empty_slot_mask
available_slot_mask.logical_or_(selected_empty_slot_mask)
```

### 2.2 核心计算流程

按batch处理，对每个batch：
1. 计算 `num_tokens_to_load` = sum(cache_miss_token_mask[batch])
2. 计算 `num_available_slot` = sum(available_slot_mask[batch])
3. 计算 `num_shortage_slot` = num_tokens_to_load - num_available_slot
4. 扫描 `topk_indices_old[batch]`，找到 == -1 的位置（empty slots）
5. 对 empty slots 做 cumsum
6. 选择 cumsum <= num_shortage_slot 的 empty slots
7. 将这些 slots 标记到 available_slot_mask

### 2.3 AscendC API 调用序列

**核心 API 序列**：
```cpp
// Step 1: 加载 cache_miss_token_mask, available_slot_mask, topk_indices_old 到 UB
DataCopy(maskLocal, maskGlobal + offset, 2048);
DataCopy(idxLocal, idxGlobal + offset, 2048);

// Step 2: 计算 sum (使用 ReduceSum)
ReduceSum(sumLocal, maskLocal, 2048);

// Step 3: 计算 shortage = tokens_to_load - available
Sub(shortageLocal, tokensToLoadLocal, availableLocal);

// Step 4: 扫描 topk_indices_old == -1，cumsum，选择
// 使用标量循环（2048个元素，每个batch内串行）
// empty_slot_mask = (topk_indices_old == -1)
// cumsum[i] = cumsum[i-1] + empty_slot_mask[i]
// selected = (cumsum <= shortage) & empty_slot_mask
// result = available_slot_mask | selected

// Step 5: 写回结果
DataCopy(outGlobal + offset, resultLocal, 2048);
```

### 2.4 实现路径选择
- [x] AscendC Kernel（纯vector实现）
- [ ] CATLASS模板库（矩阵乘法类）
- [ ] ACLNN封装（CANN内置算子）

**选择理由**: 算子核心是比较、累加、逻辑操作，不涉及矩阵乘法，适合 AscendC vector 路径实现。

---

## 3. Tiling策略

### 核心原则

- **Block级Tiling (核间切分)**: 按 batch 切分，每个 Core 处理若干个 batch
- **UB级Tiling (核内切分)**: 每个 Core 内部，2048个元素可完整放入 UB

**算子类型**: 逐元素 + 归约 + 逻辑操作

### 3.1 Tiling参数结构体定义

```cpp
struct SelectEmptySlotsTilingData {
    int64_t batchSize;          // 总batch数
    int64_t seqLen;             // 每个batch的元素数，固定2048
    int64_t totalElements;      // 总元素数 = batchSize * seqLen

    int64_t formerNum;          // 整核数量
    int64_t formerLength;       // 整核处理的batch数
    int64_t tailNum;            // 尾核数量
    int64_t tailLength;         // 尾核处理的batch数
};
```

### 3.2 Block级Tiling（核间切分）

**策略要点**:
1. **切分维度**: 按 batch 切分
2. **Cache Line对齐**: 512 字节对齐

**切分逻辑**:
```cpp
// 每个batch = 2048个元素
totalBatches = batchSize;
batchesPerCore = (totalBatches + CORE_NUM - 1) / CORE_NUM;

// 对齐到batch边界
usedCoreNum = (totalBatches + batchesPerCore - 1) / batchesPerCore;
formerNum = usedCoreNum - 1;
tailNum = 1;
formerLength = batchesPerCore;
tailLength = totalBatches - formerNum * formerLength;
```

### 3.3 UB级Tiling（核内切分）

**策略要点**:
1. 每个 batch 的 2048 个元素完整处理
2. 需要 buffer 存储中间结果

#### UB 分配表

| Buffer名称 | 大小(字节) | 用途 | 数量 | 总大小 |
|-----------|-----------|------|------|--------|
| missMask | 2048 * 1 = 2KB | cache_miss_token_mask | 1 | 2KB |
| availMask | 2048 * 1 = 2KB | available_slot_mask | 1 | 2KB |
| idxData | 2048 * 4 = 8KB | topk_indices_old | 1 | 8KB |
| resultMask | 2048 * 1 = 2KB | 输出结果 | 1 | 2KB |
| cumsumBuf | 2048 * 4 = 8KB | cumsum中间结果 | 1 | 8KB |
| **总计** | - | - | - | **~22KB** |

**UB 约束验证**:
- **UB使用**: ~22KB
- **UB限制**: ~192KB
- **是否满足约束**: 是
- **对齐要求**: 32字节对齐

---

## 4. Workspace需求

### 4.1 Workspace 大小计算

| 算子类别 | workspace size | 说明 |
|----------|---------------|------|
| 逐元素+归约类 | SYSTEM_WORKSPACE_SIZE | 通常为 16MB |

---

## 5. 性能优化

### 5.1 关键优化点
1. **按batch并行**: 不同batch之间完全独立，可在不同Core上并行
2. **完整batch加载**: 每个batch的2048个元素完整加载到UB处理
3. **标量循环优化**: cumsum逻辑在UB内用标量循环完成，避免频繁的GM访问

### 5.2 算子特性
- **计算模式**: memory-bound
- **访存模式**: 顺序访问
- **并行性**: 高（batch间完全并行）

---

## 6. Kernel端实现要点

### 6.1 执行流程（核内）

```cpp
__aicore__ inline void Process() {
    int64_t blockIdx = AscendC::GetBlockIdx();
    int64_t blockBatches = (blockIdx == usedCoreNum - 1) ? tailLength : formerLength;
    int64_t startBatch = blockIdx * formerLength;
    
    for (int64_t b = 0; b < blockBatches; ++b) {
        int64_t batchIdx = startBatch + b;
        if (batchIdx >= batchSize) break;
        
        // 加载当前batch的数据
        int64_t offset = batchIdx * seqLen;
        DataCopy(missLocal, missGlobal + offset, seqLen);
        DataCopy(availLocal, availGlobal + offset, seqLen);
        DataCopy(idxLocal, idxGlobal + offset, seqLen);
        
        // 计算 sum (使用 ReduceSum 或标量循环)
        int32_t numTokensToLoad = 0;
        int32_t numAvailableSlot = 0;
        for (int64_t i = 0; i < seqLen; ++i) {
            numTokensToLoad += missLocal.GetValue(i);
            numAvailableSlot += availLocal.GetValue(i);
        }
        
        int32_t numShortage = numTokensToLoad - numAvailableSlot;
        
        // 扫描 topk_indices_old == -1，cumsum，选择
        int32_t cumsum = 0;
        for (int64_t i = 0; i < seqLen; ++i) {
            int32_t idxVal = idxLocal.GetValue(i);
            bool isEmpty = (idxVal == -1);
            if (isEmpty) {
                cumsum++;
            }
            bool selected = (cumsum <= numShortage) && isEmpty;
            bool availVal = availLocal.GetValue(i);
            bool result = availVal || selected;
            resultLocal.SetValue(i, result);
        }
        
        // 写回结果
        DataCopy(outGlobal + offset, resultLocal, seqLen);
    }
}
```

---

## 7. 实现检查清单

### 7.1 文件结构
- [ ] `csrc/ops/select_empty_slots/CMakeLists.txt`
- [ ] `csrc/ops/select_empty_slots/op_host/select_empty_slots.cpp`
- [ ] `csrc/ops/select_empty_slots/op_kernel/select_empty_slots.cpp`
- [ ] `csrc/ops.h` (添加声明)
- [ ] `csrc/register.cpp` (添加注册)

### 7.2 Host端实现
- [ ] 定义SelectEmptySlotsTilingData结构体
- [ ] 实现Block级Tiling参数计算（按batch切分）
- [ ] 分配workspace
- [ ] 调用kernel入口函数

### 7.3 Kernel端实现
- [ ] 实现Init函数（初始化queue和buffer）
- [ ] 实现Process函数（按batch处理）
- [ ] 实现sum计算
- [ ] 实现cumsum和选择逻辑
- [ ] 实现CopyOut函数（UB -> GM）

### 7.4 测试验证
- [ ] 准备测试数据（小规模）
- [ ] 正确性验证（与PyTorch对比）
- [ ] 性能测试

---

## 8. 参考实现

- **相似算子**: `set_difference`（基础 AscendC 算子模板）
- **PyTorch参考**:
  ```python
  num_tokens_to_load = cache_miss_token_mask.sum(dim=1)
  num_available_slot = available_slot_mask.sum(dim=1)
  num_shortage_slot = num_tokens_to_load - num_available_slot
  num_shortage_slot = num_shortage_slot.unsqueeze(1)
  empty_slot_mask = topk_indices_old == -1
  empty_slot_cumsum = torch.cumsum(empty_slot_mask, dim=1)
  selected_empty_slot_mask = (empty_slot_cumsum <= num_shortage_slot) & empty_slot_mask
  available_slot_mask.logical_or_(selected_empty_slot_mask)
  ```
