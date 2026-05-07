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
#include "torch_aclnn_helper.h"

#include "aclrtlaunch_select_empty_slots.h"

namespace ascend_kernel {

    struct SelectEmptySlotsTilingData {
        int64_t batchSize;

        int64_t formerNum;
        int64_t formerLength;
        int64_t tailNum;
        int64_t tailLength;
    };

    at::Tensor select_empty_slots(const at::Tensor &cache_miss_token_mask,
                                  const at::Tensor &available_slot_mask,
                                  const at::Tensor &topk_indices_old)
    {
        // Input validation
        TORCH_CHECK(cache_miss_token_mask.dim() == 2, "select_empty_slots: cache_miss_token_mask must be 2D tensor");
        TORCH_CHECK(available_slot_mask.dim() == 2, "select_empty_slots: available_slot_mask must be 2D tensor");
        TORCH_CHECK(topk_indices_old.dim() == 2, "select_empty_slots: topk_indices_old must be 2D tensor");
        TORCH_CHECK(cache_miss_token_mask.sizes() == available_slot_mask.sizes(),
                    "select_empty_slots: cache_miss_token_mask and available_slot_mask must have the same shape");
        TORCH_CHECK(cache_miss_token_mask.sizes() == topk_indices_old.sizes(),
                    "select_empty_slots: cache_miss_token_mask and topk_indices_old must have the same shape");
        TORCH_CHECK(cache_miss_token_mask.dtype() == at::kBool, "select_empty_slots: cache_miss_token_mask must be bool");
        TORCH_CHECK(available_slot_mask.dtype() == at::kBool, "select_empty_slots: available_slot_mask must be bool");
        TORCH_CHECK(topk_indices_old.dtype() == at::kInt, "select_empty_slots: topk_indices_old must be int32");

        int64_t batchSize = cache_miss_token_mask.size(0);
        int64_t seqLen = cache_miss_token_mask.size(1);

        // Create output tensor (bool)
        at::Tensor output = at::empty({batchSize, seqLen},
                                      at::TensorOptions().dtype(at::kBool).device(cache_miss_token_mask.device()));

        if (batchSize == 0) {
            return output;
        }

        // Get platform info
        auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
        uint32_t coreNum = platform->GetCoreNumAiv();

        // Block-level Tiling: distribute batches across cores
        int64_t batchesPerCore = (batchSize + coreNum - 1) / coreNum;
        int64_t usedCoreNum = (batchSize + batchesPerCore - 1) / batchesPerCore;
        int64_t formerNum = usedCoreNum - 1;
        int64_t tailNum = 1;
        int64_t formerLength = batchesPerCore;
        int64_t tailLength = batchSize - formerNum * formerLength;

        if (tailLength <= 0 && formerNum > 0) {
            tailLength = formerLength;
        }

        // Workspace
        constexpr int64_t SYSTEM_WORKSPACE_SIZE = 16 * 1024 * 1024;
        auto workspace = at::empty({static_cast<int64_t>(SYSTEM_WORKSPACE_SIZE)},
                                   at::TensorOptions().dtype(at::kByte).device(cache_miss_token_mask.device()));

        uint32_t blockDim = static_cast<uint32_t>(usedCoreNum);

        EXEC_KERNEL_CMD(select_empty_slots, blockDim,
                        cache_miss_token_mask, available_slot_mask, topk_indices_old, output,
                        batchSize, formerNum, formerLength, tailNum, tailLength);

        return output;
    }

    void cache_slot_update(const at::Tensor &cache_miss_token_mask,
                           const at::Tensor &available_slot_mask,
                           const at::Tensor &topk_indices_old,
                           at::Tensor &topk_indices_new,
                           at::Tensor &last_step_topk_indices)
    {
        // Input validation
        TORCH_CHECK(cache_miss_token_mask.dim() == 2, "cache_slot_update: cache_miss_token_mask must be 2D tensor");
        TORCH_CHECK(available_slot_mask.dim() == 2, "cache_slot_update: available_slot_mask must be 2D tensor");
        TORCH_CHECK(topk_indices_old.dim() == 2, "cache_slot_update: topk_indices_old must be 2D tensor");
        TORCH_CHECK(topk_indices_new.dim() == 2, "cache_slot_update: topk_indices_new must be 2D tensor");
        TORCH_CHECK(last_step_topk_indices.dim() == 2, "cache_slot_update: last_step_topk_indices must be 2D tensor");
        TORCH_CHECK(cache_miss_token_mask.sizes() == available_slot_mask.sizes(),
                    "cache_slot_update: cache_miss_token_mask and available_slot_mask must have the same shape");
        TORCH_CHECK(cache_miss_token_mask.sizes() == topk_indices_old.sizes(),
                    "cache_slot_update: cache_miss_token_mask and topk_indices_old must have the same shape");
        TORCH_CHECK(cache_miss_token_mask.sizes() == topk_indices_new.sizes(),
                    "cache_slot_update: cache_miss_token_mask and topk_indices_new must have the same shape");
        TORCH_CHECK(cache_miss_token_mask.sizes() == last_step_topk_indices.sizes(),
                    "cache_slot_update: cache_miss_token_mask and last_step_topk_indices must have the same shape");
        TORCH_CHECK(cache_miss_token_mask.dtype() == at::kBool, "cache_slot_update: cache_miss_token_mask must be bool");
        TORCH_CHECK(available_slot_mask.dtype() == at::kBool, "cache_slot_update: available_slot_mask must be bool");
        TORCH_CHECK(topk_indices_old.dtype() == at::kInt, "cache_slot_update: topk_indices_old must be int32");
        TORCH_CHECK(topk_indices_new.dtype() == at::kInt, "cache_slot_update: topk_indices_new must be int32");
        TORCH_CHECK(last_step_topk_indices.dtype() == at::kLong, "cache_slot_update: last_step_topk_indices must be int64");

        // Step 1: Compute updated available mask using existing AscendC kernel
        at::Tensor new_available_mask = select_empty_slots(
            cache_miss_token_mask, available_slot_mask, topk_indices_old);

        // Step 2: masked_select topk_indices_new using cache_miss_token_mask
        at::Tensor topk_to_load = at::masked_select(topk_indices_new, cache_miss_token_mask);

        // Step 3: Cast selected indices to int64 for scattering into last_step_topk_indices
        at::Tensor topk_to_load_int64 = topk_to_load.to(at::kLong);

        // Step 4: Inplace masked_scatter into last_step_topk_indices
        EXEC_NPU_CMD(aclnnInplaceMaskedScatter, last_step_topk_indices, new_available_mask, topk_to_load_int64);

        // Step 5: Inplace fill topk_indices_new with -1
        topk_indices_new.fill_(-1);

        // Step 6: Inplace masked_scatter into topk_indices_new
        EXEC_NPU_CMD(aclnnInplaceMaskedScatter, topk_indices_new, new_available_mask, topk_to_load);
    }

}  // namespace ascend_kernel
