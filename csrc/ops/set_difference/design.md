# set_difference 设计文档

## 1. 算子接口

### 1.1 函数签名
```cpp
at::Tensor set_difference(
    const at::Tensor &a,
    const at::Tensor &b
);
```

### 1.2 参数说明
| 参数名 | 类型 | 输入/输出 | 支持的数据类型 | 描述 | 约束条件 |
|--------|------|-------|-------|--------|------|
| a | at::Tensor | 输入 | int32 | 输入tensor A | shape = (BatchSize, topK), topK = 2048 |
| b | at::Tensor | 输入 | int32 | 输入tensor B | shape = (BatchSize, topK), topK = 2048 |
| output | at::Tensor | 输出 | bool | 输出mask | shape = (BatchSize, topK) |

### 1.3 支持的数据类型
- [ ] float16
- [ ] float32
- [x] int32

---

## 2. 计算逻辑

### 2.1 算法描述

`set_difference` 算子实现批量集合差集判断功能：

对于输入 `a` (shape `[BatchSize, topK]`) 和 `b` (shape `[BatchSize, topK]`)，输出 `mask` (shape `[BatchSize, topK]`)，其中：

```
mask[batch, i] = True  如果 a[batch, i] 不在 b[batch, :] 中
mask[batch, i] = False 如果 a[batch, i] 在 b[batch, :] 中
```

即：对于 `a` 中每个元素，判断其是否存在于**同 batch** 的 `b` 中。不存在则为 `True`（差集元素），存在则为 `False`。

### 2.2 核心计算流程

```cpp
// 总元素数 = BatchSize * topK
// 每个 block 处理一部分元素
// 对每个处理的元素 a[idx]：
//   1. 计算其所属 batch: batch_id = idx / topK
//   2. 加载 b[batch_id, :] (2048个int32) 到 UB
//   3. 用 CompareScalar 将 a[idx] 与 b[batch_id, :] 逐元素比较
//      - int32 = 4字节, 每次处理 256/4 = 64 个元素
//      - repeat = 32, count = 64, 一条指令覆盖 2048 个元素
//      - 结果: 256个 uint8, 共 256*8 = 2048 bit, 对应 2048 个元素的 mask
//   4. 对 256 个 uint8 结果调用 ReduceMax（需先 Cast 到 half）
//      - 若 max > 0, 说明 a[idx] 在 b 中存在
//   5. 结果取反 (!found) 写入临时 buffer
// 最后统一将结果从 UB 写回 GM
```

### 2.3 AscendC API 调用序列

**核心 API 序列**（int32 路径）：
```cpp
// Step 1: 加载 b_batch (2048个int32) 到 UB
DataCopy(bLocal, bGlobal + batchOffset, 2048);

// Step 2: 用 CompareScalar 比较 a_element 与 b_batch 所有元素
// 结果生成 256 个 uint8 (2048 bit mask)
CompareScalar(cmpResult, bLocal, aElement, CMP_EQ, 64, 32, 0);
// count=64, repeat=32, stride=0

// Step 3: Cast uint8 结果到 half，调用 ReduceMax
Cast(cmpHalf, cmpResult, CAST_NONE, 256);
half maxVal = ReduceMax(cmpHalf, 256);

// Step 4: 判断 maxVal > 0，得到 found 标志
// mask[i] = (maxVal == 0)  // not found -> True
```

### 2.4 实现路径选择
- [x] AscendC Kernel（纯vector实现）
- [ ] CATLASS模板库（矩阵乘法类）
- [ ] ACLNN封装（CANN内置算子）

**选择理由**: 算子核心是比较 + 归约操作，不涉及矩阵乘法，适合 AscendC vector 路径实现。利用 CompareScalar 和 ReduceMax 指令高效完成。

---

## 3. Tiling策略

AscendC 算子采用**两级 Tiling 策略**来充分利用硬件并行能力。

### 两级 Tiling 架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    全局内存 (GM)                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              totalElements = BS * topK               │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
    ┌──────────┐     ┌──────────┐     ┌──────────┐
    │  Core 0  │     │  Core 1  │ ... │ Core 47  │   ← Block级Tiling (核间切分)
    │ elements │     │ elements │     │ elements │
    └──────────┘     └──────────┘     └──────────┘
          │                │                │
          ▼                ▼                ▼
    ┌──────────┐     ┌──────────┐     ┌──────────┐
    │   UB     │     │   UB     │     │   UB     │   ← UB级Tiling (核内切分)
    │ b_buffer │     │ b_buffer │     │ b_buffer │
    │a_elements│     │a_elements│     │a_elements│
    │cmp_result│     │cmp_result│     │cmp_result│
    └──────────┘     └──────────┘     └──────────┘
```

### 核心原则

- **Block级Tiling (核间切分)**: 按总元素数切分（BatchSize * topK），每个 Core 处理若干个元素（可能跨batch）
- **UB级Tiling (核内切分)**: 每个 Core 内部，将需要处理的 A 元素分批加载，但 B 的一个 batch (2048元素) 可完整放入 UB

**算子类型**: 比较类 + 标量广播 + 归约

**参考文档**:
- 硬件说明: `references/hardware-architecture.md`
- 通用原则: `references/general-tiling-principles.md`

### 3.1 Tiling参数结构体定义

```cpp
struct SetDifferenceTilingData {
    int64_t batchSize;          // 总batch数
    int64_t topK;               // 每个batch的元素数，固定2048
    int64_t totalElements;      // 总元素数 = batchSize * topK

    int64_t formerNum;          // 整核数量
    int64_t formerLength;       // 整核处理的元素数
    int64_t tailNum;            // 尾核数量
    int64_t tailLength;         // 尾核处理的元素数

    int64_t tileLength;         // UB单次处理的A元素数
};
```

### 3.2 Block级Tiling（核间切分）

**策略要点**:
1. **切分维度**: 按**总元素数**（BatchSize * topK）切分，而非仅按 batch 切分
2. **避免跨batch加载B**: 每个 block 分配的元素尽量属于同一个 batch
3. **Cache Line对齐**: 512 字节对齐

**切分逻辑**:
```cpp
// 理想情况：每个 block 处理整数个 batch 的元素
// 但如果 batchSize 较小（如 < 48），则每个 batch 由多个 block 处理
// 或一个 block 处理多个 batch

// 策略：优先保证每个 block 处理的元素属于连续的 batch
// 即 block 处理 [startBatch, endBatch] 内的元素，且尽量完整 batch

// 计算：
totalElements = batchSize * topK;
elementsPerCore = (totalElements + CORE_NUM - 1) / CORE_NUM;

// 对齐到 batch 边界（避免跨batch加载不同的B）
batchesPerCore = (elementsPerCore + topK - 1) / topK;  // 向上取整到 batch
alignedElementsPerCore = batchesPerCore * topK;

usedCoreNum = (totalElements + alignedElementsPerCore - 1) / alignedElementsPerCore;
formerNum = usedCoreNum - 1;
tailNum = 1;
formerLength = alignedElementsPerCore;
tailLength = totalElements - formerNum * formerLength;
```

| 参数 | 计算公式 | 示例值 (BS=4, topK=2048, 48核) |
|------|----------|--------------------------------|
| totalElements | batchSize * topK | 8192 |
| elementsPerCore | 8192 / 48 = 171 | 171 |
| batchesPerCore | (171 + 2048 - 1) / 2048 = 1 | 1 batch |
| alignedElementsPerCore | 1 * 2048 = 2048 | 2048 |
| usedCoreNum | (8192 + 2048 - 1) / 2048 = 4 | 4 |
| formerNum | 3 | 3 |
| tailNum | 1 | 1 |
| formerLength | 2048 | 2048 (1 batch) |
| tailLength | 8192 - 3*2048 = 2048 | 2048 (1 batch) |

**结果**: 4 个 core 各处理 1 个 batch，其余 44 个 core 闲置。

| 参数 | 计算公式 | 示例值 (BS=64, topK=2048, 48核) |
|------|----------|--------------------------------|
| totalElements | 64 * 2048 = 131072 | 131072 |
| elementsPerCore | 131072 / 48 = 2731 | 2731 |
| batchesPerCore | (2731 + 2048 - 1) / 2048 = 2 | 2 batch |
| alignedElementsPerCore | 2 * 2048 = 4096 | 4096 |
| usedCoreNum | (131072 + 4096 - 1) / 4096 = 32 | 32 |
| formerNum | 31 | 31 |
| tailNum | 1 | 1 |
| formerLength | 4096 | 4096 (2 batch) |
| tailLength | 131072 - 31*4096 = 4096 | 4096 (2 batch) |

**结果**: 32 个 core 各处理 2 个 batch，其余 16 个 core 闲置。所有 core 都不跨 batch。

**负载均衡验证**:
- 整核/尾核数据差异: formerLength >= tailLength (4096 >= 4096)
**核间切分验证**:
- 计算数据量是否正确: formerNum * formerLength + tailNum * tailLength == totalElements (31*4096 + 4096 = 131072)

### 3.3 UB级Tiling（核内切分）

**策略要点**:
1. 每个 block 处理 N 个 batch 的元素
2. 将 B 的一个 batch (2048 * 4B = 8KB) 加载到 UB 后复用
3. 从 GM 分批加载 A 的元素到 UB，每个元素依次与 B_batch 比较
4. 32 字节对齐

#### UB 分配表

**int32 路径**:

| Buffer名称 | 大小(字节) | 用途 | 数量 | 总大小 |
|-----------|-----------|------|------|--------|
| inQueueB | 2048 * 4 = 8KB | b_batch 数据缓冲 | 1 | 8KB |
| inQueueA | tileLength * 4 | a_elements 数据缓冲 | BUFFER_NUM | tileLength * 4 * 2 |
| compareResult | 256 | CompareScalar 结果 (256 uint8) | 1 | 256B |
| castBuffer | 256 * 2 = 512 | Cast uint8 -> half | 1 | 512B |
| outQueueMask | tileLength * 1 | 输出 mask (bool) | BUFFER_NUM | tileLength * 2 |
| **总计** | - | - | - | **8KB + tileLength * 8 + 768B** |

**简化 UB 分配（假设 tileLength = 256）**:

| Buffer名称 | 大小(字节) | 用途 | 数量 | 总大小 |
|-----------|-----------|------|------|--------|
| bBuffer | 8KB | b_batch 数据 | 1 | 8KB |
| aBuffer | 256 * 4 = 1KB | a_elements | BUFFER_NUM | 2KB |
| cmpResult | 256 | CompareScalar 结果 | 1 | 256B |
| castBuf | 512 | Cast 缓冲 | 1 | 512B |
| maskBuffer | 256 | 输出 mask | BUFFER_NUM | 512B |
| **总计** | - | - | - | **~11.3KB** |

**UB 约束验证**:
- **UB使用**: ~11.3KB (tileLength=256)
- **UB限制**: ~192KB
- **是否满足约束**: 是（剩余大量空间可增大 tileLength）
- **对齐要求**: 32字节对齐（已满足）

#### tileLength 计算

| 参数 | 计算公式 | 值 (int32) |
|------|----------|-------------|
| bufferCoefficient (不含B) | 8 (A双缓冲) + 2 (mask双缓冲) = 10 | 10 |
| maxTileElements | (UB_SIZE_LIMIT - 8KB - 768B) / 4 / 10 | ~4500 |
| alignElements | 32 / 4 = 8 | 8 |
| tileLength | (maxTileElements / alignElements) * alignElements | 4504 |

> 实际上 tileLength 不需要这么大，因为 block 处理的元素数 = batchesPerCore * topK，而 topK=2048。tileLength 设为 2048 或更小即可。

---

## 4. Workspace需求

### 4.1 Workspace 大小计算

本算子输出为 bool mask，不需要额外的 workspace 存储中间结果（中间计算在 UB 完成）。

| 算子类别 | workspace size | 说明 |
|----------|---------------|------|
| 比较类算子 | SYSTEM_WORKSPACE_SIZE | 通常为 16MB |

- **tiling data size**: sizeof(SetDifferenceTilingData)

### 4.2 Workspace 分配示例
```cpp
// Host 端分配 workspace
constexpr int64_t SYSTEM_WORKSPACE_SIZE = 16 * 1024 * 1024;  // 16MB
size_t workspaceSize = SYSTEM_WORKSPACE_SIZE;
auto workspace = at::empty({static_cast<int64_t>(workspaceSize)},
                           at::TensorOptions().dtype(at::kByte).device(a.device()));
```

---

## 5. 性能优化

### 5.1 关键优化点
1. **不跨 batch**: Block 级切分对齐到 batch 边界，避免加载多个 b_batch
2. **B 数据复用**: b_batch (2048元素) 加载到 UB 后，被所有 A 元素复用
3. **CompareScalar 向量化**: 一条指令完成 2048 个元素的比较（repeat=32, count=64）
4. **ReduceMax 快速归约**: 256 个 uint8 -> half -> ReduceMax，快速判断是否存在
5. **批量写回**: 结果先累积在 UB buffer，最后统一写回 GM

### 5.2 算子特性
- **计算模式**: compute-bound（CompareScalar + ReduceMax）
- **访存模式**: 顺序访问（batch 内按 topK 顺序）
- **并行性**: 高（元素间完全独立，batch 内也可并行）

---

## 6. Kernel端实现要点

### 6.1 执行流程（核内循环）

```cpp
__aicore__ inline void Process() {
    // 获取当前核处理的元素范围
    int64_t blockIdx = AscendC::GetBlockIdx();
    int64_t blockLength = (blockIdx == tiling->usedCoreNum - 1) ? tiling->tailLength : tiling->formerLength;
    int64_t startElement = blockIdx * tiling->formerLength;
    int64_t endElement = startElement + blockLength;

    // 计算当前 block 处理的 batch 范围
    int64_t startBatch = startElement / topK;
    int64_t endBatch = (endElement - 1) / topK;

    for (int64_t batch = startBatch; batch <= endBatch; ++batch) {
        // 加载 b[batch, :] 到 UB
        int64_t bOffset = batch * topK;
        DataCopy(bLocal, bGlobal + bOffset, topK);

        // 计算该 batch 内需要处理的 A 元素范围
        int64_t aStart = (batch == startBatch) ? (startElement % topK) : 0;
        int64_t aEnd = (batch == endBatch) ? ((endElement - 1) % topK + 1) : topK;

        // 分批加载 A 元素
        for (int64_t tile = aStart; tile < aEnd; tile += tileLength) {
            int64_t currentTileLength = min(tileLength, aEnd - tile);
            // 加载 a[batch, tile:tile+currentTileLength]
            DataCopy(aLocal, aGlobal + bOffset + tile, currentTileLength);

            // 对每个 A 元素，与 b_batch 比较
            for (int64_t i = 0; i < currentTileLength; ++i) {
                int32_t aVal = aLocal.GetValue(i);

                // CompareScalar: aVal vs b_batch[0:2048]
                // count=64, repeat=32, 覆盖 2048 个元素
                CompareScalar(cmpResult, bLocal, aVal, CMP_EQ, 64, 32, 0);

                // Cast uint8 -> half, 然后 ReduceMax
                Cast(castBuf, cmpResult, CAST_NONE, 256);
                half maxVal = ReduceMax(castBuf, 256);

                // mask = (maxVal == 0)  // not found -> True
                maskLocal.SetValue(i, (maxVal == 0));
            }

            // 写回 GM
            DataCopy(outGlobal + bOffset + tile, maskLocal, currentTileLength);
        }
    }
}
```

### 6.2 关键指令说明

**CompareScalar**:
```cpp
// 比较 bLocal (int32) 与标量 aVal，结果写入 cmpResult (uint8)
// count = 64 (每次处理64个int32 = 256字节)
// repeat = 32 (重复32次 = 32*64 = 2048个元素)
// stride = 0 (连续访问)
CompareScalar(cmpResult, bLocal, aVal, CMP_EQ, 64, 32, 0);
```

**Cast + ReduceMax**:
```cpp
// 将 256 个 uint8 Cast 到 256 个 half
Cast(castBuf, cmpResult, CAST_NONE, 256);

// ReduceMax 归约，如果任意 bit 为 1，maxVal > 0
half maxVal = ReduceMax(castBuf, 256);
```

**批量 Not 操作**:
```cpp
// 在 UB 中完成取反逻辑
// mask = (maxVal == 0) 等价于 !found
```

---

## 7. 实现检查清单

### 7.1 文件结构
- [x] `csrc/ops/set_difference/CMakeLists.txt`
- [x] `csrc/ops/set_difference/op_host/set_difference.cpp`
- [x] `csrc/ops/set_difference/op_kernel/set_difference.cpp`
- [ ] `csrc/ops.h` (添加声明)
- [ ] `csrc/register.cpp` (添加注册)

### 7.2 Host端实现
- [ ] 定义SetDifferenceTilingData结构体（包含batch、topK、totalElements参数）
- [ ] 实现Block级Tiling参数计算（按总元素数切分，对齐batch边界）
- [ ] 根据UB分配表确定buffer系数
- [ ] 实现UB级Tiling参数计算（32字节对齐）
- [ ] 分配workspace（SYSTEM_WORKSPACE_SIZE）
- [ ] 调用kernel入口函数

### 7.3 Kernel端实现
- [ ] 实现Init函数（初始化queue和buffer）
- [ ] 实现CopyInB函数（GM -> UB，加载b_batch）
- [ ] 实现CopyInA函数（GM -> UB，加载a_tile）
- [ ] 实现Compare函数（CompareScalar + Cast + ReduceMax）
- [ ] 实现CopyOut函数（UB -> GM，输出bool mask）
- [ ] 实现Process主循环（按batch遍历，处理所有A元素）

### 7.4 测试验证
- [ ] 准备测试数据（小规模：BS=2, topK=8）
- [ ] 准备测试数据（标准规模：BS=32, topK=2048）
- [ ] 正确性验证（与PyTorch `get_set_diff_mask` 对比）
- [ ] 边界case测试（全部相同、全部不同、空交集）
- [ ] 性能测试（对比PyTorch实现）

---

## 8. 参考实现

- **相似算子**: `helloworld`（基础 AscendC 算子模板）
- **PyTorch参考**: `get_set_diff_mask` 函数（用户提供）
- **功能等价代码**:
  ```python
  comparison_mask = a.unsqueeze(-1) == b.unsqueeze(1)  # [bs, topk, topk]
  intersect_mask = comparison_mask.any(-1)             # [bs, topk]
  return ~intersect_mask                               # [bs, topk]
  ```

---

## 使用说明

1. 本算子假设 `a` 和 `b` 的 shape 相同，均为 `[BatchSize, topK]`，且 `topK = 2048`
2. 比较是**逐 batch** 的，不会跨 batch 比较
3. 输出为 `bool` 类型 mask，`True` 表示元素在 `a` 中但不在 `b` 中
4. 相等判断使用严格的元素相等（`==`），适用于 int32 精确比较
5. Block 级切分会对齐到 batch 边界，避免加载多个 b_batch 导致的性能下降

**设计完成后**，使用 `ascendc-operator-code-gen` skill 生成具体代码实现。
