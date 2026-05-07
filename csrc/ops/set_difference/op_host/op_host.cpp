// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "torch_kernel_helper.h"
#include "tiling/platform/platform_ascendc.h"

#include "aclrtlaunch_set_diff_exchange.h"

namespace ascend_kernel {

    struct SetDiffExchangeTilingData {
        int64_t batchSize;
        int64_t topK;
        int64_t totalElements;

        int64_t formerNum;
        int64_t formerLength;
        int64_t tailNum;
        int64_t tailLength;

        int64_t tileLength;
    };

    at::Tensor set_difference(const at::Tensor &a, const at::Tensor &b)
    {
        // ����У��
        TORCH_CHECK(a.dim() == 2, "set_difference: a must be 2D tensor");
        TORCH_CHECK(b.dim() == 2, "set_difference: b must be 2D tensor");
        TORCH_CHECK(a.sizes() == b.sizes(), "set_difference: a and b must have the same shape");
        TORCH_CHECK(a.dtype() == at::kInt, "set_difference: a must be int32");
        TORCH_CHECK(b.dtype() == at::kInt, "set_difference: b must be int32");

        int64_t batchSize = a.size(0);
        int64_t topK = a.size(1);
        int64_t totalElements = batchSize * topK;

        // ������� tensor (int32)
        at::Tensor output = at::empty({batchSize, topK},
                                      at::TensorOptions().dtype(at::kInt).device(a.device()));

        // ��� totalElements == 0��ֱ�ӷ��ؿ�tensor
        if (totalElements == 0) {
            return output;
        }

        // ��ȡƽ̨����
        auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
        uint32_t coreNum = platform->GetCoreNumAiv();
        uint64_t ubSize = 0;
        platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

        constexpr int64_t CACHE_LINE_BYTE_LENGTH = 512;
        constexpr int64_t UB_ALIGN_BYTES = 32;

        // Block��Tiling���˼��з֣�
        // ���ԣ�����Ԫ�����з֣�չ����һά����������batch�߽�
        // �������Գ����������core����ʹbatch size��С
        // ÿ��AԪ�ض����Լ������Ӧ��batch idx��Ȼ����ض�Ӧ��B����
        int64_t dtypeSize = sizeof(int32_t);
        int64_t alignElements = CACHE_LINE_BYTE_LENGTH / dtypeSize;

        // ����ÿ��core�����Ԫ���������뵽cache line
        int64_t elementsPerCore = (totalElements + coreNum - 1) / coreNum;
        elementsPerCore = ((elementsPerCore + alignElements - 1) / alignElements) * alignElements;

        // ����ʵ��ʹ�õ�core��
        int64_t usedCoreNum = (totalElements + elementsPerCore - 1) / elementsPerCore;
        int64_t formerNum = usedCoreNum - 1;
        int64_t tailNum = 1;
        int64_t formerLength = elementsPerCore;
        int64_t tailLength = totalElements - formerNum * formerLength;

        // UB��Tiling�������з֣�
        // UB������ԣ�
        // 1. B buffer: topK * sizeof(int32) = 2048 * 4 = 8192 bytes - �洢��ǰbatch��B���ݣ��ɸ��ã�
        // 2. A buffer: tileLength * sizeof(int32) - �洢��ǰtile��A����
        // 3. CompareResult buffer: 256 bytes - CompareScalar�����2048 bits��
        // 4. Cast buffer: 512 bytes - uint8_tתhalf (256 * 2)
        // 5. ReduceDst buffer: 64 bytes - ReduceMax��� (32 * 2)
        // 6. ReduceTmp buffer: 1024 bytes - ReduceMax��ʱ�ռ�
        //
        // B����(2048��int32=8192bytes)����һ���Լ��ص�UB��
        // A���ݰ�tile���أ�ѭ������

        int64_t ubOverhead = topK * dtypeSize           // B buffer: 8192 bytes
                             + 256                        // CompareScalar result (256 uint8)
                             + 512                        // Cast buffer (256 half = 512 bytes)
                             + 64                         // ReduceDst buffer (32 half = 64 bytes)
                             + 2048                       // ReduceTmp buffer (1024 half = 2048 bytes)
                             + 32;                        // ���⿪��

        // Aʹ�õ����壨��Ϊѭ����˳��ִ�У�����Ҫ���뵽32�ֽ�
        int64_t maxTileElements = (ubSize - ubOverhead) / dtypeSize;
        int64_t ubAlignElements = UB_ALIGN_BYTES / dtypeSize;
        int64_t tileLength = (maxTileElements / ubAlignElements) * ubAlignElements;

        if (tileLength < ubAlignElements) {
            tileLength = ubAlignElements;
        }

        // ��� TilingData
        SetDiffExchangeTilingData tilingData;
        tilingData.batchSize = batchSize;
        tilingData.topK = topK;
        tilingData.totalElements = totalElements;
        tilingData.formerNum = formerNum;
        tilingData.formerLength = formerLength;
        tilingData.tailNum = tailNum;
        tilingData.tailLength = tailLength;
        tilingData.tileLength = tileLength;

        // ���� workspace
        constexpr int64_t SYSTEM_WORKSPACE_SIZE = 16 * 1024 * 1024;
        auto workspace = at::empty({static_cast<int64_t>(SYSTEM_WORKSPACE_SIZE)},
                                   at::TensorOptions().dtype(at::kByte).device(a.device()));

        // ���� block dim
        uint32_t blockDim = static_cast<uint32_t>(usedCoreNum);

        // ���� kernel
        EXEC_KERNEL_CMD(set_diff_exchange, blockDim, a, b, output, workspace,
                        tilingData.batchSize, tilingData.topK, tilingData.totalElements,
                        tilingData.formerNum, tilingData.formerLength,
                        tilingData.tailNum, tilingData.tailLength,
                        tilingData.tileLength);

        return output;
    }

}  // namespace ascend_kernel

