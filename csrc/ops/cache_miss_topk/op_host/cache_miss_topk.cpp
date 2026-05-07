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

#include "aclrtlaunch_cache_miss_topk.h"

namespace ascend_kernel {

    void cache_miss_topk(const at::Tensor &cache_miss_token_mask,
                         at::Tensor &available_slot_mask,
                         const at::Tensor &topk_indices_old)
    {
        // Input validation
        TORCH_CHECK(cache_miss_token_mask.dim() == 2, "cache_miss_topk: cache_miss_token_mask must be 2D tensor");
        TORCH_CHECK(available_slot_mask.dim() == 2, "cache_miss_topk: available_slot_mask must be 2D tensor");
        TORCH_CHECK(topk_indices_old.dim() == 2, "cache_miss_topk: topk_indices_old must be 2D tensor");
        TORCH_CHECK(cache_miss_token_mask.sizes() == available_slot_mask.sizes(),
                    "cache_miss_topk: cache_miss_token_mask and available_slot_mask must have the same shape");
        TORCH_CHECK(cache_miss_token_mask.sizes() == topk_indices_old.sizes(),
                    "cache_miss_topk: cache_miss_token_mask and topk_indices_old must have the same shape");
        TORCH_CHECK(cache_miss_token_mask.dtype() == at::kBool, "cache_miss_topk: cache_miss_token_mask must be bool");
        TORCH_CHECK(available_slot_mask.dtype() == at::kBool, "cache_miss_topk: available_slot_mask must be bool");
        TORCH_CHECK(topk_indices_old.dtype() == at::kLong, "cache_miss_topk: topk_indices_old must be int64");

        uint32_t batchSize = static_cast<uint32_t>(cache_miss_token_mask.size(0));
        uint32_t seqLen = static_cast<uint32_t>(cache_miss_token_mask.size(1));

        if (batchSize == 0) {
            return;
        }

        // Get platform info
        auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
        uint32_t coreNum = platform->GetCoreNumAiv();

        // Block-level Tiling: distribute batches across cores
        uint32_t batchesPerCore = (batchSize + coreNum - 1) / coreNum;
        uint32_t usedCoreNum = (batchSize + batchesPerCore - 1) / batchesPerCore;
        uint32_t formerNum = usedCoreNum - 1;
        uint32_t tailNum = 1;
        uint32_t formerLength = batchesPerCore;
        uint32_t tailLength = batchSize - formerNum * formerLength;

        if (tailLength <= 0 && formerNum > 0) {
            tailLength = formerLength;
        }

        uint32_t blockDim = usedCoreNum;

        EXEC_KERNEL_CMD(cache_miss_topk, blockDim,
                        cache_miss_token_mask, available_slot_mask, topk_indices_old,
                        batchSize, formerNum, formerLength, tailNum, tailLength);
    }

    // Helper function: Compute set difference mask (elements in a but not in b)
    static at::Tensor get_set_diff_mask(const at::Tensor &a, const at::Tensor &b)
    {
        at::Tensor a_expanded = a.unsqueeze(-1);  // [batch, topk, 1]
        at::Tensor b_expanded = b.unsqueeze(1);   // [batch, 1, topk]
        at::Tensor comparison_mask = a_expanded.eq(b_expanded);  // [batch, topk, topk]
        at::Tensor intersect_mask = comparison_mask.any(-1);     // [batch, topk]
        return intersect_mask.logical_not();                     // [batch, topk]
    }

    void get_cache_miss_topk_indices(at::Tensor &topk_indices,
                                     at::Tensor &last_step_topk_indices,
                                     const at::Tensor &req_ids_tensor)
    {
        /* remove the cache hit (already in topk_buffer) tokens from topk_idx,
           only keep the cache miss part for following npu/cpu loading. */

        // Input validation
        TORCH_CHECK(topk_indices.dim() == 2, "get_cache_miss_topk_indices: topk_indices must be 2D tensor");
        TORCH_CHECK(last_step_topk_indices.dim() == 2, "get_cache_miss_topk_indices: last_step_topk_indices must be 2D tensor");
        TORCH_CHECK(req_ids_tensor.dim() == 1, "get_cache_miss_topk_indices: req_ids_tensor must be 1D tensor");
        TORCH_CHECK(topk_indices.dtype() == at::kInt, "get_cache_miss_topk_indices: topk_indices must be int32");
        TORCH_CHECK(last_step_topk_indices.dtype() == at::kLong, "get_cache_miss_topk_indices: last_step_topk_indices must be int64");
        TORCH_CHECK(req_ids_tensor.dtype() == at::kLong, "get_cache_miss_topk_indices: req_ids_tensor must be int64");

        int64_t num_reqs = topk_indices.size(0);
        int64_t topk = topk_indices.size(1);

        TORCH_CHECK(last_step_topk_indices.size(0) >= num_reqs,
                    "get_cache_miss_topk_indices: last_step_topk_indices batch size must be >= num_reqs");
        TORCH_CHECK(last_step_topk_indices.size(1) == topk,
                    "get_cache_miss_topk_indices: last_step_topk_indices seq_len must match topk_indices");
        TORCH_CHECK(req_ids_tensor.size(0) >= num_reqs,
                    "get_cache_miss_topk_indices: req_ids_tensor size must be >= num_reqs");

        // to distinguish tokens of different reqs, add a req_ids_offset
        at::Tensor req_ids_offset = req_ids_tensor.slice(0, 0, num_reqs).mul(1 << 16).unsqueeze(-1);
        at::Tensor topk_indices_new = at::where(topk_indices >= 0, topk_indices + req_ids_offset, -1);
        at::Tensor topk_indices_old = last_step_topk_indices.slice(0, 0, num_reqs);

        // tokens in new but not in old, which is cache miss and need to load
        at::Tensor cache_miss_token_mask = get_set_diff_mask(topk_indices_new, topk_indices_old);
        // tokens in old but not in new, which is useless now
        at::Tensor available_slot_mask = get_set_diff_mask(topk_indices_old, topk_indices_new);

        /* Compute updated available mask, this part is needed while seq_len < 2k,
           so there are multiple empty slots (idx == -1) in old topk_idx,
           we also pick these empty slots to store cache miss tokens. */
        cache_miss_topk(cache_miss_token_mask, available_slot_mask, topk_indices_old);

        // Inplace gather and scatter using cache_miss_token_mask and available_slot_mask
        at::Tensor topk_to_load = at::masked_select(topk_indices_new, cache_miss_token_mask);
        topk_indices_new.fill_(-1);
        EXEC_NPU_CMD(aclnnInplaceMaskedScatter, topk_indices_new, available_slot_mask, topk_to_load);

        // update history topk_indices for next step usage
        at::Tensor last_step_slice = last_step_topk_indices.slice(0, 0, num_reqs);
        at::Tensor topk_to_load_int64 = topk_to_load.to(at::kLong);
        EXEC_NPU_CMD(aclnnInplaceMaskedScatter, last_step_slice, available_slot_mask, topk_to_load_int64);

        // recover topk_indices (remove req offset)
        at::Tensor result = at::where(topk_indices_new >= 0, topk_indices_new - req_ids_offset, -1);
        topk_indices.copy_(result);
    }

}  // namespace ascend_kernel
